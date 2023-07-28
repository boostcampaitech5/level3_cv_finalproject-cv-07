import torch
import numpy as np
import faiss
import os

from torch.nn.functional import cosine_similarity
from torchvision.transforms import functional as F
from collections import Counter
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from .transform import get_transform

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

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
    def __init__(self, model, checkpoint, gallery_path=None, person_thr=0.6, cosine_thr=0.5) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = model
        self.model.load_state_dict(torch.load(checkpoint), strict=True)         
        self.model = self.model.to(self.device)
        
        self.tf = get_transform()
        
        self.origin_embedding = None
        self.embedding = None
        self.cluster = None
        
        random_tensor = torch.randn(1, 3, 224, 224).to(self.device)
        embedding_dim = self.model(random_tensor).shape[-1]
        
        self.player_dict = dict()
        self.player_dict_init(10)
        
        self.person_thr = person_thr
        self.cosine_thr = cosine_thr
        
        res = faiss.StandardGpuResources()
        self.faiss_index = faiss.GpuIndexFlatIP(res, embedding_dim)
        self.faiss_index = faiss.IndexIDMap(self.faiss_index)

        self.gallery_path = gallery_path

        
        if self.gallery_path:
            self.gallery_dataset = GalleryDataset(self.gallery_path, self.tf)
            self.faiss_index_init()
            
            
    def shot_re_id_inference(self, frame, results): 
        person_img_lst = self.shot_person_query_lst(frame, results)
        detected_query = person_img_lst
        detected_query_stack = torch.stack(detected_query, dim=0).to(self.device)
    
        with torch.no_grad():
            detected_query_vector = self.model(detected_query_stack).detach().cpu().numpy()                       
        
        faiss.normalize_L2(detected_query_vector)
        C, I = self.faiss_index.search(detected_query_vector, 1)
        return I[0,0]
    
    def re_id_process(self, frame, results):
        id_dict = dict()
        
        person_idx_lst, person_img_lst = self.person_query_lst(frame, results, thr=self.person_thr)

        if len(person_img_lst) == 0:
            return id_dict
        
        detected_query = person_img_lst
        detected_query_stack = torch.stack(detected_query, dim=0).to(self.device)
        with torch.no_grad():
            detected_query_vector = self.model(detected_query_stack).detach().cpu().numpy()                       
            
        faiss.normalize_L2(detected_query_vector)

        C, I = self.faiss_index.search(detected_query_vector, 5)
        I = np.where(C > self.cosine_thr, I, -1)
        C = np.where(C > self.cosine_thr, C, -1)
        matched_list = self.hard_voting(I, C)

        id_index_dict = dict()
        fail_person = []
        for idx, id in enumerate(matched_list):
            if id_dict.get(id) != None:
                if id_dict[id][4] < results[person_idx_lst[idx]][4]:
                    fail_person.append((id_index_dict[id], id_dict[id]))
                    id_dict[id] = results[person_idx_lst[idx]]
                    id_index_dict[id] = idx
                else:
                    fail_person.append((idx, results[person_idx_lst[idx]]))
            else:
                id_dict[id] = results[person_idx_lst[idx]]
                id_index_dict[id] = idx
                
        remain_id = list(set(range(10)) - set(id_dict))

        if fail_person:
            for i, p_data in fail_person:
                for p_id in I[i]:
                    if p_id in remain_id:
                        id_dict[p_id] = results[person_idx_lst[i]]
                        break
            
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
                person_img_lst.append(tf_person_img)
    
        return person_idx_lst, person_img_lst

    def hard_voting(self, matched_list, cosine_list):
        final = []
        for list, c_list in zip(matched_list, cosine_list):
            tmp = []
            for j, k in zip(list, c_list):
                # print(c_list)
                if k >=0.9:
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
    
    
    
    def init_gallery(self, frames):
        """_summary_

        Args:
            frames (list):   (3, W, H) * (person) * (frames num)
        """
        
        accumulated_query_vector = np.empty((0, 1000), dtype=np.float32)
        
        for frame in frames:
            person_img_lst = []
            for person in frame:
                person_img_lst.append(person)

            if person_img_lst == []:
                continue
            detected_query_stack = torch.stack(person_img_lst, dim=0).to(self.device)
            
            with torch.no_grad():
                detected_query_vector = self.model(detected_query_stack).detach().cpu().numpy()
            
            accumulated_query_vector = np.concatenate((accumulated_query_vector, detected_query_vector), axis=0)


        faiss.normalize_L2(accumulated_query_vector) 

        cluster = DBSCAN(eps=0.007, min_samples=25).fit_predict(accumulated_query_vector)
        cluster = np.asarray(cluster.astype('int64'))
        
        self.player_dict = dict()
        self.player_dict_init(len(np.unique(cluster, return_counts = True)[0]))
        
        self.origin_embedding = accumulated_query_vector
        
        accumulated_query_vector = accumulated_query_vector[cluster != -1]
        cluster = cluster[cluster != -1]
        
        self.embedding = accumulated_query_vector
        self.cluster = cluster
        
        # cluster = cluster.reshape(-1, 1)
        self.faiss_index.add_with_ids(accumulated_query_vector, cluster)
            
        print('Gallery initialized')
        
        
    def player_dict_init(self, num):
        for i in range(0,num):
            self.player_dict[i] = Player(i)
        print(f'Setting {num} Players')

    def faiss_index_init(self):
        print('gallery imgs are added to faiss_index')
        g_dataset = self.gallery_dataset
        g_loader = DataLoader(g_dataset, batch_size=16, shuffle=True, pin_memory=False, drop_last=False, num_workers=4)
        
        for img, label in tqdm(g_loader):
            inputs = img.to(self.device)
            vector = self.model(inputs)
            vector = vector.detach().cpu().numpy()

            faiss.normalize_L2(vector)
            self.faiss_index.add_with_ids(vector, np.array(label))
        
        print('Done')


class GalleryDataset(Dataset):
    def __init__(self, gallery_path, transform):
        super().__init__()
        self.gallery_path = gallery_path
        self.transform = transform
        self.img_lst = os.listdir(self.gallery_path)
        
    def __len__(self):
        return len(self.img_lst)

    def __getitem__(self, item):
        img_name = self.img_lst[item]
        label = int(img_name.split('_')[0][2:])
        image = np.array(Image.open(os.path.join(self.gallery_path,img_name)))

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = image.astype(np.float32)
            image /= 255.
            image = torch.from_numpy(image)
            image = torch.permute(image, (2,0,1))

        return image, label
