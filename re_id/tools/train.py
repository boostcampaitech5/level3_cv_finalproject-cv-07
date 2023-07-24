import os
import cv2
import timm
import torch
import faiss
import random
import warnings
import argparse
import numpy as np
import tqdm.auto as tqdm
import albumentations as A
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import faiss.contrib.torch_utils

from ..models.model import *
from ..module.loss import quadruplet_loss
from ..data.download_data import config
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from timeit import default_timer as timer

#----------------------------------------------------------------------------------------------------------------------#  
# Initialization & Data Augmentations                                                                                  #
#----------------------------------------------------------------------------------------------------------------------# 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
warnings.filterwarnings("ignore")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark_enabled = True
tf = A.Compose([A.Resize(224,224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.HorizontalFlip(p=0.5)])

#----------------------------------------------------------------------------------------------------------------------#  
# Argument Parser                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------# 
parser = argparse.ArgumentParser()
parser.add_argument('--demo', type=bool, default=False) 
parser.add_argument('--seed', type=int, default=1) 
parser.add_argument('--epoch', type=int, default=100) 
parser.add_argument('--train_batch', type=int, default=64) 
parser.add_argument('--valid_batch', type=int, default=256) 
parser.add_argument('--lr', type=float, default=0.001)  
parser.add_argument('--fp16', type=bool, default=False) 
parser.add_argument('--model', type=str, default='mobilenetv3') 
parser.add_argument('--scheduler', type=bool, default=False) 
parser.add_argument('--num_workers', type=int, default=8) 
parser.add_argument('--quadruplet', type=bool, default=False) 
args = parser.parse_args()

#----------------------------------------------------------------------------------------------------------------------#  
# Seed                                                                                                                 #
#----------------------------------------------------------------------------------------------------------------------# 
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

#----------------------------------------------------------------------------------------------------------------------#  
# Picture Statistics                                                                                                   #
#----------------------------------------------------------------------------------------------------------------------# 
def get_picture_statistic(image_path):
    widths = []
    heights = []
    
    for img in os.listdir(image_path):
        im = cv2.imread(os.path.join(image_path, img), cv2.COLOR_BGR2RGB)
        widths.append(im.shape[1])
        heights.append(im.shape[0])

    avg_width = round(sum(widths)/len(widths),2)
    avg_height = round(sum(heights)/len(heights),2)
    max_width = max(widths)
    max_height = max(heights)

    return avg_width, avg_height, max_width, max_height

#----------------------------------------------------------------------------------------------------------------------#  
# Datasets                                                                                                             #
#----------------------------------------------------------------------------------------------------------------------# 
class GALLERY(Dataset):
    def __init__(self, gallery_path, transform):
        super().__init__()
        self.gallery_path = gallery_path
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.gallery_path))

    def __getitem__(self, item):
        gallery_image_name = os.listdir(self.gallery_path)[item]
        gallery_label = gallery_image_name
        gallery_image = cv2.imread(os.path.join(self.gallery_path, gallery_image_name))
        gallery_image = cv2.cvtColor(gallery_image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=gallery_image)
            gallery_image = transformed['image']
            gallery_image = gallery_image.astype(np.float32)
            gallery_image = torch.from_numpy(gallery_image)
            gallery_image = torch.permute(gallery_image, (2,0,1))
        else:
            gallery_image = gallery_image.astype(np.float32)
            gallery_image /= 255.
            gallery_image = torch.from_numpy(gallery_image)
            gallery_image = torch.permute(gallery_image, (2,0,1))

        return gallery_image, gallery_label
    
class QUERY(Dataset):
    def __init__(self, query_path, transform):
        super().__init__()
        self.query_path = query_path
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.query_path))

    def __getitem__(self, item):
        query_image_name = os.listdir(self.query_path)[item]
        query_label = query_image_name
        query_image = cv2.imread(os.path.join(self.query_path, query_image_name))
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=query_image)
            query_image = transformed['image']
            query_image = query_image.astype(np.float32)
            query_image = torch.from_numpy(query_image)
            query_image = torch.permute(query_image, (2,0,1))
        else:
            query_image = query_image.astype(np.float32)
            query_image /= 255.
            query_image = torch.from_numpy(query_image)
            query_image = torch.permute(query_image, (2,0,1))

        return query_image, query_label

class TRAIN(Dataset):
    def __init__(self, path, transform):
        super().__init__()
        self.path = path
        self.transfrom = transform
        self.people_list = sorted(os.listdir(path))
        
    def __len__(self):
        return len(self.people_list)

    def __getitem__(self, item):
        anchor_name = self.people_list[item]
        anchor_id = int(anchor_name[:5])
        anchor = cv2.imread(os.path.join(self.path, anchor_name))
        anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB)
        
        positive_list = [filename for filename in self.people_list if filename.startswith(anchor_name[:5])]
        positive_name = random.choice(positive_list)
        while positive_name == anchor_name:
            positive_name = random.choice(positive_list)
        positive = cv2.imread(os.path.join(self.path, positive_name))
        positive = cv2.cvtColor(positive, cv2.COLOR_BGR2RGB)

        negative_name = random.choice(self.people_list) 
        negative_id = int(negative_name[:5])
        while negative_id == anchor_id:
            negative_name = random.choice(self.people_list) 
            negative_id = int(negative_name[:5])
        negative = cv2.imread(os.path.join(self.path, negative_name))
        negative = cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)

        negative_name2 = random.choice(self.people_list) 
        negative_id2 = int(negative_name2[:5])
        while negative_id2 == anchor_id or negative_id2 == negative_id:
            negative_name2 = random.choice(self.people_list) 
            negative_id2 = int(negative_name2[:5])
        negative2 = cv2.imread(os.path.join(self.path, negative_name2))
        negative2 = cv2.cvtColor(negative2, cv2.COLOR_BGR2RGB)

        set_images = [anchor, positive, negative, negative2]

        if self.transfrom:
            for idx, i in enumerate(set_images):
                transformed = self.transfrom(image=i)
                set_images[idx] = transformed['image']
                set_images[idx] = set_images[idx].astype(np.float32)
                set_images[idx] = torch.from_numpy(set_images[idx])
                set_images[idx] = torch.permute(set_images[idx], (2,0,1))
                
        else:
            tf = A.Compose([A.Resize(224,224)])
            for idx, i in enumerate(set_images):
                transformed = tf(image=i)
                set_images[idx] = transformed['image']
                set_images[idx] = set_images[idx].astype(np.float32)
                set_images[idx] /= 255.
                set_images[idx] = torch.from_numpy(set_images[idx])
                set_images[idx] = torch.permute(set_images[idx], (2,0,1))

               
        return set_images[0], set_images[1], set_images[2], set_images[3], anchor_id

#----------------------------------------------------------------------------------------------------------------------#  
# Model Configurations                                                                                                 #
#----------------------------------------------------------------------------------------------------------------------# 
model_dict = {"convnextv2_a": ConvNextV2_A(),
              "convnextv2_f": ConvNextV2_F(),
              "convnextv2_p": ConvNextV2_P(),
              "convnextv2_n": ConvNextV2_N(),
              "convnextv2_t": ConvNextV2_T(),
              "convnextv2_b": ConvNextV2_B(),
              "convnextv2_l": ConvNextV2_L(),
              "mobilenetv3": MobileNetV3(),
              "squeezenet": SqueezeNet(),
              "squeezenet_cbam": SqueezeNetMod(),
              "mobilevitv2": MobileVitV2()}

model = model_dict.get(args.model)
assert model != None
embedding_dim = model(torch.randn(1, 3, 224, 224)).shape[-1]
epochs = args.epoch
learning_rate = args.lr
criterion = quadruplet_loss if args.quadruplet else nn.TripletMarginLoss(margin=1.0)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
weight_path = "./model_weights"
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95**epoch, verbose=False)

if os.path.isdir(weight_path) == False:
    os.mkdir(weight_path)

#----------------------------------------------------------------------------------------------------------------------#  
# Train & Validation Methods                                                                                           #
#----------------------------------------------------------------------------------------------------------------------# 
def train(model, epochs, criterion, optimizer, lr_scheduler, train_loader, query_loader, gallery_loader, gallery_path, embedding_dim, topk, scheduler, fp16, quadruplet):
    print(f"Start Training...")
    if fp16:
        print(f"Training with Mixed Precision...")
    print()
    best_mAP = 0
    changes = 0
    scaler = torch.cuda.amp.GradScaler()
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        for step, (anchor, positive, negative, negative2, _) in enumerate(train_loader):
            anchor, positive, negative, negative2 = anchor.to(device), positive.to(device), negative.to(device), negative2.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            if fp16:
                with torch.cuda.amp.autocast():
                    anchor_features = model(anchor) 
                    positive_features = model(positive)
                    negative_features = model(negative)
                    if quadruplet:
                        negative2_features = model(negative2)
                        loss = criterion(anchor_features, positive_features, negative_features, negative2_features)
                    else:
                        loss = criterion(anchor_features, positive_features, negative_features)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                anchor_features = model(anchor) 
                positive_features = model(positive)
                negative_features = model(negative) 
                if quadruplet:
                    negative2_features = model(negative2)
                    loss = criterion(anchor_features, positive_features, negative_features, negative2_features)
                else:
                    loss = criterion(anchor_features, positive_features, negative_features)
                
                loss.backward()
                optimizer.step()

            if (step+1) % 5 == 0:
                print(f"Epoch:[{epoch}/{epochs}] | Step:[{step+1}/{len(train_loader)}] | Loss:{loss.item():.4f}")
        if scheduler:
            lr_scheduler.step()

        mAP = validation(model=model, query_loader=query_loader, gallery_loader=gallery_loader, gallery_path=gallery_path, embedding_dim=embedding_dim, topk=topk)
        if mAP >= best_mAP:
            print(f"Best mAP is achieved!!")
            print("Saving Best and Latest Model...")
            torch.save(model.state_dict(), os.path.join(weight_path, f"{args.model}6_best.pth"))
            changes = mAP - best_mAP
            best_mAP = mAP

        torch.save(model.state_dict(), os.path.join(weight_path, f"{args.model}6_latest.pth"))
        print("All Model Checkpoints Saved!")
        print("----------------------------")
        print(f"Best mAP: {best_mAP:.4f}")
        if mAP >= best_mAP:
            print(f"Current mAP: {mAP:.4f} (+{(changes):.4f})")
        elif mAP < best_mAP:
            print(f"Current mAP: {mAP:.4f} (-{(best_mAP-mAP):.4f})")
        print()
    print("Training is finished!")

def validation(model, query_loader, gallery_loader, gallery_path, embedding_dim, topk): 
    res = faiss.StandardGpuResources()
    faiss_index = faiss.GpuIndexFlatIP(res, embedding_dim)
    model = model.to(device)
    model.eval()

    gallery_list = []
    query_list = []
    matched_list = []
    with torch.no_grad():
        for gallery, labels in gallery_loader:
            gallery = gallery.to(device)

            outputs = model(gallery).cpu().numpy()
            faiss.normalize_L2(outputs)
            faiss_index.add(outputs)
            for label in labels:
                gallery_list.append(label)
                
    with torch.no_grad():
        for query, label in query_loader:
            for i in label:
                query_list.append(i)
            query = query.to(device)

            outputs = model(query)
            _, I = faiss_index.search(outputs, topk)
            for x in I:
                tmp = [gallery_list[x[i]] for i in range(topk)]
                matched_list.append(tmp)

    def calculate_map(query_list, matched_list, gallery_path):   
        total_query_gt = 0
        precision = 0
        count = 0
        AP = 0
        mAP = []
        for query_name, matched_name in zip(query_list, matched_list):
            for x in os.listdir(gallery_path):
                if query_name[:5] == x[:5]:
                    total_query_gt += 1

            tmp_total_query_gt = total_query_gt
            for _, i in enumerate(matched_name,start=1):
                count += 1
                if tmp_total_query_gt == 0:
                    break
                elif query_name[:5] == i[:5]:
                    precision += 1
                    tmp_total_query_gt -= 1
                else:
                    continue
                AP += (precision/count)
            mAP.append(AP/total_query_gt)

            AP = 0
            total_query_gt = 0
            precision = 0
            count = 0
            
        return sum(mAP)/len(mAP)
       
    return calculate_map(query_list, matched_list, gallery_path)

#----------------------------------------------------------------------------------------------------------------------#  
# Main                                                                                                                 #
#----------------------------------------------------------------------------------------------------------------------# 
if __name__ == "__main__":
    print("**** Training Configurations ****")
    print(f"1. Model: {args.model}")
    print(f"2. Device: {device}")
    print(f"3. Demo: {args.demo}")
    print(f"4. Seed: {args.seed}")
    print(f"5. Epoch: {args.epoch}")
    print(f"6. Train Batch Size: {args.train_batch}")
    print(f"7. Valid Batch Size: {args.valid_batch}")
    print(f"8. Learning Rate: {args.lr}")
    print(f"9. FP16: {args.fp16}")
    print(f"10. Scheduler: {args.scheduler}")
    print(f"11. Num Workers: {args.num_workers}")
    print(f"12. Quadruplet Loss: {args.quadruplet}\n")

    if args.demo:
        path = "./data/data_reid/reid_training" 
        gallery_path = "./data/data_reid/reid_test/gallery"
        query_path = "./data/data_reid/reid_test/query"
    else:
        path = "./data/custom_dataset/training"
        gallery_path = "./data/custom_dataset/gallery"
        query_path = "./data/custom_dataset/query"
        
    avg_width, avg_height, max_width, max_height= get_picture_statistic(image_path=path)
    print("**** Image Statistics ****")
    print(f"Average Width: {avg_width}")
    print(f"Average Height: {avg_height}")
    print(f"Max Width: {max_width}")
    print(f"Max Height: {max_height}\n")

    train_dataset = TRAIN(path, tf)
    query_dataset = QUERY(query_path, tf)
    gallery_dataset = GALLERY(gallery_path, tf)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True,  num_workers=args.num_workers)
    query_loader = DataLoader(query_dataset, batch_size=args.valid_batch, shuffle=False, num_workers=args.num_workers)
    gallery_loader = DataLoader(gallery_dataset, batch_size=args.valid_batch, shuffle=False, num_workers=args.num_workers)

    total_gallery_images = len(os.listdir(gallery_path))
    
    train(model=model,
          epochs=args.epoch,
          criterion=criterion,
          optimizer=optimizer,
          lr_scheduler=lr_scheduler,
          train_loader=train_loader,
          query_loader=query_loader,
          gallery_loader=gallery_loader,
          gallery_path=gallery_path,
          embedding_dim=embedding_dim,
          topk=total_gallery_images,
          scheduler=args.scheduler,
          fp16=args.fp16,
          quadruplet=args.quadruplet)