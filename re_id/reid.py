import torch
import cv2
import numpy as np
import faiss
import albumentations as A
from torch.nn.functional import cosine_similarity
from torchvision.transforms import functional as F
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from collections import Counter
from sklearn.cluster import KMeans
from .transform import get_transform

class Player:
    def __init__(self, id):
        self.id = id
        self.stm=0
        self.smm=0

    def shot_try_plus(self):
        self.stm+=1
    
    def shot_made_plus(self):
        self.smm+=1

class ReId:
    def __init__(self, model, checkpoint, person_thr=0.6, cosine_thr=0.5) -> None:
        self.model = model
        self.model.load_state_dict(torch.load(checkpoint), strict=True)         
        self.model = self.model.to("cuda") if torch.cuda.is_available() else self.model.to("cpu")
        
        self.tf = get_transform()
        
        random_tensor = torch.randn(1, 3, 224, 224).to("cuda")
        embedding_dim = self.model(random_tensor).shape[-1]
        
        self.player_dict = dict()
        self.person_thr = person_thr
        self.cosine_thr = cosine_thr
        
        res = faiss.StandardGpuResources()
        self.faiss_index = faiss.GpuIndexFlatIP(res, embedding_dim)
        self.faiss_index = faiss.IndexIDMap(self.faiss_index)
        
    def shot_re_id_inference(self, frame, results): 
        person_img_lst = self.shot_person_query_lst(frame, results)
        detected_query = person_img_lst
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        detected_query_stack = torch.stack(detected_query, dim=0).to(device)
    
        with torch.no_grad():
            detected_query_vector = self.model(detected_query_stack).detach().cpu().numpy()                       
        
        faiss.normalize_L2(detected_query_vector)
        C, I = self.faiss_index.search(detected_query_vector, 1)
        return I[0,0]
    
    def re_id_process(self, frame, results, frame_num, first_frame):
        person_idx_lst, person_img_lst = self.person_query_lst(frame, results, thr=self.person_thr)
        detected_query = person_img_lst
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        detected_query_stack = torch.stack(detected_query, dim=0).to(device)

        with torch.no_grad():
            detected_query_vector = self.model(detected_query_stack).detach().cpu().numpy()                       
        
        faiss.normalize_L2(detected_query_vector)

        if first_frame:
            ids = np.arange(len(detected_query))
            self.faiss_index.add_with_ids(detected_query_vector, ids)
            matched_list = ids.tolist()
            first_frame = False
            
            id_dict = dict()
            
            for i in ids:
                id_dict[i] = results[person_idx_lst[i]] 
                self.player_dict[i] = Player(i)
            
            return id_dict

        else:
            total_id = len(faiss.vector_to_array(self.faiss_index.id_map))
            if total_id < 5:
                C, I = self.faiss_index.search(detected_query_vector, 1)
            else:
                C, I = self.faiss_index.search(detected_query_vector, 5)
            
            I = np.where(C > self.cosine_thr, I, -1)
            matched_list = self.hard_voting(I)

            exist_id_idx = C[:,0] >= self.cosine_thr
            if frame_num%12==0:
                exist_ids = np.array(matched_list)[exist_id_idx]
                self.faiss_index.add_with_ids(detected_query_vector[exist_id_idx], exist_ids)
            
            id_dict = dict()
            for idx, id in enumerate(matched_list):
                if id_dict.get(id) != None:
                    if id_dict[id][4] < results[person_idx_lst[idx]][4]:
                        id_dict[id] = results[person_idx_lst[idx]]
                else:
                    if id == -1:
                        highest_id = max(faiss.vector_to_array(self.faiss_index.id_map))
                        self.faiss_index.add_with_ids(detected_query_vector[idx].reshape(1,-1), np.array([highest_id+1]))
                        self.player_dict[highest_id+1] = Player(highest_id+1)
                        id_dict[highest_id+1] = results[person_idx_lst[idx]]
                    else:
                        id_dict[id] = results[person_idx_lst[idx]]
            
            return id_dict
     
    def person_query_lst(self, frame, results,thr):
        img = frame

        person_img_lst = []
        person_idx_lst = []
        for idx, bb in enumerate(results):
            x1, y1, x2, y2, c, l = bb
        
            if (l == 3) and (c >= thr):
                x1, y1, x2, y2, l = int(x1), int(y1), int(x2), int(y2), int(l)
                person_img = img[y1:y2, x1:x2, :]
                if (person_img.shape[0] ==0) or (person_img.shape[1] ==0):
                    continue
                person_idx_lst.append(idx)
                transformed = self.tf(image=person_img)
                tf_person_img = transformed['image']
                tf_person_img = tf_person_img.astype(np.float32)
                tf_person_img = torch.from_numpy(tf_person_img)
                tf_person_img = torch.permute(tf_person_img, (2,0,1))
                person_img_lst.append(tf_person_img)
    
        return person_idx_lst, person_img_lst

    def hard_voting(self, matched_list):
        final = []
        for list in matched_list:
            tmp = []
            for j in list:
                if j >=0:
                    tmp.append(j)
            if not tmp:
                final.append(-1)
                continue
            
            count = Counter(tmp)
            final.append(count.most_common(1)[0][0])
        return final
    
    def shot_person_query_lst(self, frame, results):
        img = frame
        person_img_lst = []
        x1, y1, x2, y2, c, l = results
        x1, y1, x2, y2, l = int(x1), int(y1), int(x2), int(y2), int(l)
        person_img = img[y1:y2, x1:x2, :]
        transformed = self.tf(image=person_img)
        tf_person_img = transformed['image']
        tf_person_img = tf_person_img.astype(np.float32)
        tf_person_img = torch.from_numpy(tf_person_img)
        tf_person_img = torch.permute(tf_person_img, (2,0,1))
        person_img_lst.append(tf_person_img)
        return person_img_lst