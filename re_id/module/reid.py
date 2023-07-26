import torch
import numpy as np
import faiss
from torch.nn.functional import cosine_similarity
from torchvision.transforms import functional as F
from collections import Counter
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from .transform import get_transform

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os
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
        
        self.origin_embedding = None
        self.embedding = None
        self.cluster = None
        
        random_tensor = torch.randn(1, 3, 224, 224).to("cuda")
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
            self.gallery_dataset = GalleryDataset(self.gallery_path, self._get_transform())
            self.faiss_index_init()
            
        
    def _get_transform(self):
    
        return A.Compose([A.Resize(224,224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
                ])
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
        matched_list = self.hard_voting(I)

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
        tf = self._get_transform()
    
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
        tf = A.Compose([A.Resize(224,224),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        person_img_lst = []
        x1, y1, x2, y2, c, l = results
        x1, y1, x2, y2, l = int(x1), int(y1), int(x2), int(y2), int(l)
        person_img = img[y1:y2, x1:x2, :]
        transformed = tf(image=person_img)
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
                # print(person.shape)
                # person = np.array(person)
                # transformed = self.tf(image=person)
                # tf_person_img = transformed['image']
                # tf_person_img = tf_person_img.astype(np.float32)
                # tf_person_img = torch.from_numpy(tf_person_img)
                # tf_person_img = torch.permute(tf_person_img, (2,0,1))
                # person_img_lst.append(tf_person_img)
                person_img_lst.append(person)

            if person_img_lst == []:
                continue
            detected_query_stack = torch.stack(person_img_lst, dim=0).to(self.device)
            
            with torch.no_grad():
                detected_query_vector = self.model(detected_query_stack).detach().cpu().numpy()
            
            accumulated_query_vector = np.concatenate((accumulated_query_vector, detected_query_vector), axis=0)


        faiss.normalize_L2(accumulated_query_vector) 

        print(accumulated_query_vector.shape)  ## (person, dim)
        # print(cluster.shape)
        cluster = DBSCAN(eps=0.007, min_samples=25).fit_predict(accumulated_query_vector)
        # cluster = DBSCAN(eps=0.07, min_samples=10).fit_predict(accumulated_query_vector)
        # cluster = GaussianMixture(n_components=10, random_state=0).fit_predict(accumulated_query_vector)
        # cluster = HDBSCAN(min_cluster_size=15, min_samples=15,cluster_selection_epsilon=1).fit_predict(accumulated_query_vector)
        # cluster = KMeans(n_clusters=10, random_state=0, n_init="auto").fit_predict(accumulated_query_vector)
        print(cluster.shape)
        cluster = np.asarray(cluster.astype('int64'))

        # reshape to (batch, 1)
        # cluster = cluster.reshape(-1, 1)
        
        print(cluster)
        
        self.player_dict = dict()
        self.player_dict_init(len(cluster))
        
        self.origin_embedding = accumulated_query_vector
        
        accumulated_query_vector = accumulated_query_vector[cluster != -1]
        cluster = cluster[cluster != -1]
        
        print(np.unique(cluster, return_counts = True))
        print(accumulated_query_vector.shape)
        print(cluster.shape)
        
        
        
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
        
        assert 10 == max(faiss.vector_to_array(self.faiss_index.id_map))
        
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
