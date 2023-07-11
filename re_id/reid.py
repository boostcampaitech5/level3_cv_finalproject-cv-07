from torch.nn.functional import cosine_similarity
import torch
from torchvision.transforms import functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2
import numpy as np
import faiss
from collections import Counter
# from sklearn.cluster import KMeans

class RectResize(ImageOnlyTransform):
    def __init__(self,
                 size, 
                 padding_value=0,
                 interpolate=cv2.INTER_LINEAR_EXACT,
                 always_apply=False,
                 p=1.0):
        
        super(RectResize, self).__init__(always_apply, p)
        self.size = size
        self.padding_value = padding_value
        self.interpolate = interpolate

    def apply(self, image, **params):

        h, w, c = image.shape
        
        if w == h:
            img_resize = cv2.resize(image, dsize=(self.size[1], self.size[0]), interpolation = self.interpolate)
            return img_resize
        
        else:
            img_pad = np.full((self.size[0], self.size[1], c), self.padding_value, dtype=np.uint8)
            if h > w:
                resize_value = self.size[0] / h
                w_new = round(w * resize_value)
                img_resize = cv2.resize(image, dsize=(w_new, self.size[0]), interpolation = self.interpolate)
                padding = (self.size[1] - w_new) // 2
                img_pad[:, padding:padding+w_new,:] = img_resize
            else:
                resize_value = self.size[1] / w
                h_new = round(h * resize_value)
                img_resize = cv2.resize(image, dsize=(self.size[1], h_new), interpolation = self.interpolate)
                padding = (self.size[0] - h_new) // 2
                img_pad[padding:padding+h_new,:,:] = img_resize
            return img_pad
            
    def get_transform_init_args_names(self):
        return ("size", "padding_value", "interpolate")


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
    def __init__(self, model, checkpoint, person_thr=0.6, cosine_thr=0.5, embedding=960) -> None:

        self.model = model
        
        # load pretrained Checkpoint     
        model_state_dict = torch.load(checkpoint)
        self.model.load_state_dict(model_state_dict, strict=True)    
      
        # Model to device 'cuda' if it is avaliable  
        self.model = self.model.to("cuda") if torch.cuda.is_available() else self.model.to("cpu")
        
        # self.gallery = torch.Tensor()       ## empty gallery
        # # self.new_ID = 0
        # self.label = []
        
        self.player_dict = dict()
        self.person_thr = person_thr
        self.cosine_thr = cosine_thr
        
        res = faiss.StandardGpuResources()
        self.faiss_index = faiss.GpuIndexFlatIP(res, embedding)
        self.faiss_index = faiss.IndexIDMap(self.faiss_index)
        
    def _get_transform(self):
        return A.Compose([RectResize(size=(224, 224) , padding_value=0, interpolate=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                            (0.26862954, 0.26130258, 0.27577711)),      ## normalize for openclip model
                                ToTensorV2(),
                                ])

    def re_id_process(self, frame, results, frame_num, first_frame):

        # 사람 객체 detection에서 찾은 후 query로 만들기
        person_idx_lst, person_img_lst = self.person_query_lst(frame, results, thr=self.person_thr)
        detected_query = person_img_lst
        device = 'cuda'

        detected_query_stack = torch.stack(detected_query, dim=0).to(device)

        #이미지 vectorize
        with torch.no_grad():
            detected_query_vector = self.model(detected_query_stack).detach().cpu().numpy()                       
        
        # 단위백터로 정규화
        faiss.normalize_L2(detected_query_vector)

        #첫 번째 프레임일 경우
        if first_frame:
            ids = np.arange(len(detected_query))
        
            # faiss_index에 query와 id를 매칭하며 add
            self.faiss_index.add_with_ids(detected_query_vector, ids)
            matched_list = ids.tolist()
            first_frame = False
            
            # id만큼 dict에 player객체 생성
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
            
            #cosine_thr보다 낮으면 -1 ID 부여
            I = np.where(C > self.cosine_thr, I, -1)
            matched_list = self.hard_voting(I)

            # 60프레임 중 5번만 faiss_index에 저장하기 
            exist_id_idx = C[:,0] >= self.cosine_thr
            if frame_num%12==0:
                exist_ids = np.array(matched_list)[exist_id_idx]
                self.faiss_index.add_with_ids(detected_query_vector[exist_id_idx], exist_ids)

            # id에 맞는 bbox를 dict에 매칭하기
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tf = A.Compose([A.Resize(224,224),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    
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
                transformed = tf(image=person_img)
                tf_person_img = transformed['image']
                tf_person_img = tf_person_img.astype(np.float32)
                tf_person_img = torch.from_numpy(tf_person_img)
                tf_person_img = torch.permute(tf_person_img, (2,0,1))
                person_img_lst.append(tf_person_img)
    
        return person_idx_lst, person_img_lst

    
    def hard_voting(matched_list):
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
    
    