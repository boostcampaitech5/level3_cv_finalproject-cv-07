import sys
import os
import cv2
import torch
import faiss
import random
import argparse
import albumentations as A
import faiss.contrib.torch_utils
import matplotlib.pyplot as plt
import warnings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from models.model import *
from timeit import default_timer as timer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#----------------------------------------------------------------------------------------------------------------------#  
# Initialization & Data Augmentations                                                                                  #
#----------------------------------------------------------------------------------------------------------------------# 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
warnings.filterwarnings("ignore")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark_enabled = True
tf = A.Compose([A.Resize(224,224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

#----------------------------------------------------------------------------------------------------------------------#  
# Argumment Parser                                                                                                     #
#----------------------------------------------------------------------------------------------------------------------# 
parser = argparse.ArgumentParser()
parser.add_argument('--demo', type=bool, default=False)
parser.add_argument('--model', type=str, default='mobilenetv3') 
parser.add_argument('--batch_size', type=int, default=256)  
parser.add_argument('--num_workers', type=int, default=8) 
parser.add_argument('--model_weight', type=str, default=None)
parser.add_argument('--query_index', type=int, default=0)
args = parser.parse_args()

#----------------------------------------------------------------------------------------------------------------------#  
# Load Model                                                                                                           #
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

assert args.model_weight != None
model = model_dict.get(args.model)
model.load_state_dict(torch.load(os.path.join("../model_weights", args.model_weight)))
embedding_dim = model(torch.randn(1, 3, 224, 224)).shape[-1]

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

#----------------------------------------------------------------------------------------------------------------------#  
# Helper Methods                                                                                                       #
#----------------------------------------------------------------------------------------------------------------------# 
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model Size: {:.2f}MB'.format(size_all_mb))

def calculate_map(query_list, matched_list, gallery_path):
    print(f'Number of Queries: {len(query_list)}')
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
        for _, i in enumerate(matched_name, start=1):
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
        
    print(f"mAP: {(sum(mAP)/len(mAP)):4f}")

def calculate_cmc(query_list, matched_list, topk):
    count = 0
    total = 0
    rank = []
    plt.figure()

    for x in range(1, topk+1):
        for (query_name, matched_name) in zip(query_list, matched_list):
            for gallery in matched_name[:x]:
                if query_name[:5] == gallery[:5]:
                    count += 1
                    break
            total += 1
        rank.append((count/total)*100)
        count, total = 0, 0
    
    x_label = [i for i in range(1, 20+1)]
    y_label = rank

    plt.plot(x_label, y_label, label=f"{args.model}", linestyle="--", marker='o')
    plt.title("CMC Rank")
    plt.xlabel("Rank (m)")
    plt.ylabel("Rank-m Identification Rate (%)")
    plt.xticks(range(1,21))
    plt.legend()
    plt.show()
    plt.savefig(f'../results/cmc_result_{args.model}')

#----------------------------------------------------------------------------------------------------------------------#  
# Inference                                                                                                            #
#----------------------------------------------------------------------------------------------------------------------# 
def inference(model, query_loader, gallery_loader, embedding_dim, topk):
    res = faiss.StandardGpuResources()
    faiss_index = faiss.GpuIndexFlatIP(res, embedding_dim)
    model = model.to(device)
    model.eval()

    gallery_list = []
    query_list = []
    matched_list = []
    inference_time = []
    with torch.no_grad():
        for (gallery, labels) in gallery_loader:
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

            inference1 = timer()
            outputs = model(query)
            inference2 = timer()
            torch.cuda.synchronize()
            inference_time.append(inference2-inference1)
            
            start1 = timer()
            _, I = faiss_index.search(outputs, topk)
            end1 = timer()
            print(f"Search Elapsed Time: {end1-start1:.5f} seconds")

            for x in I:
                tmp = [gallery_list[x[i]] for i in range(topk)]
                matched_list.append(tmp)

    print(f"Average Inference Time Elapsed: {sum(inference_time)/len(inference_time):.4f} seconds")
                    
    return query_list, matched_list

#----------------------------------------------------------------------------------------------------------------------#  
# Show Inference                                                                                                       #
#----------------------------------------------------------------------------------------------------------------------# 
def show_inference(topk, query_list, matched_list, stop=0):
    _, ax = plt.subplots(1,1+topk, figsize=(6,3))
    for step, (query_name, matched_name) in enumerate(zip(query_list, matched_list)):
        query = cv2.imread(os.path.join(query_path, query_name))
        query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)
        ax[0].imshow(query)
        ax[0].set_title(f"Q: {query_name[:5]}")
        ax[0].axis('off')

        for i in range(topk):
            matched = cv2.cvtColor(cv2.imread(os.path.join(gallery_path, matched_name[i])), cv2.COLOR_BGR2RGB)
            ax[i+1].imshow(matched)
            ax[i+1].axis('off')
            if int(query_name[:5]) == int(matched_name[i][:5]):
                ax[i+1].set_title(f"M: {matched_name[i][:5]}", color='green')
            else:
                ax[i+1].set_title(f"M: {matched_name[i][:5]}", color='red')
        
        if step == stop:
            if not os.path.isdir("../results"):
                os.mkdir("../results")
            print("Saving infereced image...")
            plt.savefig(f'../results/{args.model}_result_{args.query_index}.jpg')
            print("Completed!\n")
            break

#----------------------------------------------------------------------------------------------------------------------#  
# Main                                                                                                                 #
#----------------------------------------------------------------------------------------------------------------------# 
if __name__== "__main__":
    if args.demo:
        gallery_path = "../data/data_reid/reid_test/gallery"
        query_path = "../data/data_reid/reid_test/query"
    else:   
        gallery_path = "../data/custom_dataset/gallery"
        query_path = "../data/custom_dataset/query"

    query_dataset = QUERY(query_path, tf)
    gallery_dataset = GALLERY(gallery_path, tf)

    query_loader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    gallery_loader = DataLoader(gallery_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    total_gallery_images = len(os.listdir(gallery_path))

    print("Inferencing...")
    print(f"Model Name: {args.model}")
    get_model_size(model)   
    query_list, matched_list = inference(model=model, query_loader=query_loader, gallery_loader=gallery_loader, embedding_dim=embedding_dim, topk=total_gallery_images)
    show_inference(topk=5, query_list=query_list, matched_list=matched_list, stop=args.query_index)
    
    calculate_map(query_list, matched_list, gallery_path)
    calculate_cmc(query_list, matched_list, topk=20)
    print("Inference Finished!\n")