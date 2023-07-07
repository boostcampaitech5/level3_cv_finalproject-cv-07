import torch
import torchvision.models as models
import warnings
import os
import random
import cv2
import faiss
import faiss.contrib.torch_utils
import torch.nn as nn
import matplotlib.pyplot as plt
import albumentations as A
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from torchvision.models import ResNet50_Weights, MobileNet_V3_Large_Weights
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import Counter
from PIL import Image

class ETHZ_GALLERY(Dataset):
    def __init__(self, gallery_path, transform):
        super().__init__()
        self.gallery_path = gallery_path
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.gallery_path))

    def __getitem__(self, item):
        gallery_image_name = os.listdir(self.gallery_path)[item]
        gallery_label = gallery_image_name
        gallery_image = cv2.imread(os.path.join(self.gallery_path, gallery_image_name), cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=gallery_image)
            gallery_image = transformed['image']

        gallery_image = gallery_image.astype(np.float32)
        gallery_image = torch.from_numpy(gallery_image)
        gallery_image = torch.permute(gallery_image, (2,0,1))

        return gallery_image, gallery_label
        

class ETHZ_QUERY(Dataset):
    def __init__(self, query_path, transform):
        super().__init__()
        self.query_path = query_path
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.query_path))

    def __getitem__(self, item):
        query_image_name = os.listdir(self.query_path)[item]
        query_label = query_image_name
        query_image = cv2.imread(os.path.join(self.query_path, query_image_name), cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=query_image)
            query_image = transformed['image']

        query_image = query_image.astype(np.float32)
        query_image = torch.from_numpy(query_image)
        query_image = torch.permute(query_image, (2,0,1))

        return query_image, query_label


class ETHZ(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.people_list = sorted(os.listdir(path))
        
    def __len__(self):
        return len(self.people_list)

    def __getitem__(self, item):
        # anchor
        anchor_name = self.people_list[item]
        anchor_id = int(anchor_name[:5])
        anchor = cv2.imread(os.path.join(self.path, anchor_name), cv2.COLOR_BGR2RGB)
        
        # positive
        positive_name = self.people_list[item-1]
        positive_id = int(positive_name[:5])
        try:
            while positive_id == 0:
                positive_name = self.people_list[item-5]
        except:
                positive_name = anchor_name
        positive = cv2.imread(os.path.join(self.path, positive_name), cv2.COLOR_BGR2RGB)

        # negative
        negative_name = random.choice(self.people_list) 
        negative_id = int(negative_name[:5])
        while negative_id == anchor_id:
            negative_name = random.choice(self.people_list) 
            negative_id = int(negative_name[:5])
        negative = cv2.imread(os.path.join(self.path, negative_name), cv2.COLOR_BGR2RGB)

        set_images = [anchor, positive, negative]

        if (anchor.shape[:2] != positive.shape[:2]) or (anchor.shape[:2] != negative.shape[:2]) or (positive.shape[:2] != negative.shape[:2]):
            tf = A.Compose([A.Resize(224,224),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

            for idx, i in enumerate(set_images):
                transformed = tf(image=i)
                set_images[idx] = transformed['image']
                set_images[idx] = set_images[idx].astype(np.float32)
                set_images[idx] = torch.from_numpy(set_images[idx])
                set_images[idx] = torch.permute(set_images[idx], (2,0,1))

        return set_images[0], set_images[1], set_images[2], anchor_id
     

class ResNet_Triplet(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet50.fc = nn.Linear(in_features=2048, out_features=embedding_dim, bias=True)
     
    def forward(self, x):
        triplet = self.resnet50(x)
        return triplet
        
        
class MobileNetV3_Triplet(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.mobilenetv3 = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights)
        self.mobilenetv3.classifier[3] = nn.Linear(in_features=1280, out_features=embedding_dim, bias=True)
     
    def forward(self, x):
        embedding_feature = self.mobilenetv3(x)
        return embedding_feature
    
    
class MobileNetV3_TripletV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenetv3 = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.mobilenetv3.classifier = nn.Sequential()
     
    def forward(self, x):
        embedding_feature = self.mobilenetv3(x)
        return embedding_feature


def re_id_inference(model, detected_query, gallery_dataset, emd_dim): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    res = faiss.StandardGpuResources()
    faiss_index = faiss.GpuIndexFlatIP(res, emd_dim)
    model = model.to(device)
    model.eval()

    gallery_list = []
    matched_list = []
    with torch.no_grad():
        for gallery, labels in gallery_dataset:
            gallery = gallery.to(device)

            outputs = model(gallery).cpu().numpy() 
            faiss.normalize_L2(outputs)
            faiss_index.add(outputs)
            
            for label in labels:
                gallery_list.append(label)
                
    with torch.no_grad():
        for query in detected_query:
            query = query.to(device)
            outputs = model(query)

            _, I = faiss_index.search(outputs, 3)
            for x in I:
                tmp = [gallery_list[x[i]] for i in range(3)]
                matched_list.append(tmp)
                    
    return matched_list
    

def person_query_lst(frame, results):
    img = frame
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tf = A.Compose([A.Resize(224,224),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    person_img_lst = []
    person_idx_lst = []
    for idx, bb in enumerate(results):
        x1, y1, x2, y2, c, l = bb
    
        if l == 3:
            x1, y1, x2, y2, l = int(x1), int(y1), int(x2), int(y2), int(l)
            person_img = img[y1:y2, x1:x2, :]
            person_idx_lst.append(idx)
            transformed = tf(image=person_img)
            tf_person_img = transformed['image']
            tf_person_img = tf_person_img.astype(np.float32)
            tf_person_img = torch.from_numpy(tf_person_img)
            tf_person_img = torch.permute(tf_person_img, (2,0,1))
            tf_person_img = tf_person_img.unsqueeze(dim=0)
            person_img_lst.append(tf_person_img)

    return person_idx_lst, person_img_lst

def shot_person_query_lst(frame, results):
    img = frame
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    tf_person_img = tf_person_img.unsqueeze(dim=0)
    person_img_lst.append(tf_person_img)

    return person_img_lst
    

def hard_voting(matched_list):
    final = []
    tmp = []
    for list in matched_list:
        for j in list:
            tmp.append(int(j[:3]))
        count = Counter(tmp)
        final.append(count.most_common(1)[0][0])
        tmp = []

    return final