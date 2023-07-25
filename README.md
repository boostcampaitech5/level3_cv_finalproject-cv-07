# Basketball Players' Field Goal Percentage Tracker✨

## 🏀 Background
Have you ever wondered the accuracy of each player's shooting during a game? Attempting to manually record it would be arduous, as you would need to discern which player took each shot, making the task highly laborious. Besides not everyone has a spare time to watch and analyze a full game match. For these reasons, we have created player's FG tracker using deep learning! :) 

## 🧠 Models
In this project, we utilized two models to achieve our aim. The first model is used to detect players, basketball, ring, shooting attempts, and successful shots made, whereas the second model focused on matching a person's identity across different locations in a video. In essence, we employed Object Detection and Person Re-Identification models to accomplish this project. For our object detection model, we used YOLO-NAS-L, which was trained using super-gradients library made by Deci-AI. As for Person Re-Identification, we opted for MobileNetV3 to ensure faster inference speed and a more compact model size.    

## Ⓜ️ Faiss
Faiss is a library created by Meta that enables rapid searching of similarity betweeen multiple vector representations. This library was indispensable for our task as it allowed us to accurately determine and match a person's identity. We initially tested with L2 (Euclidean) distance to measure similarity and obtained good results. Nevertheless, upon further experiments, we discovered that utilizing cosine similarity yielded better outputs. Therefore, we have chosen to adopt cosine similarity as our definitive searching method. 

![](pictures/faiss.jpg) 

## 🖼️ Object Detection + Person Re-Identification Inference Diagram
This picture below presents an overview of our project's flow using a diagram. When an input frame is received, it undergoes object detection model, which identifies various entities such as players, basketball, ring, shot attempts and successful shots. Among these, we specifially extract instances of the 'player' class and feed them into the Re-ID model. The Re-ID model then produces embedded vectors representing each person's image. These vectors are added to Faiss, allowing us to obtain top 5 IDs corresponding to each embedded vector. Consequently, we leverage hard voting on these results to obtain the final ID with the highest confidence level.

![](pictures/inference_diagram.jpg) 

## 📝 Training Configurations & Results
### Person Re-Identification Model
| Models | Dataset[^1] | Embedded Dimensions | Epochs | Batch Size | Optimizer | LR | Loss | Augmentations | mAP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3 | V1 | 1000 | 100 | 64 | AdamW | 0.001 | TripletLoss | Resize <br>Normalize <br>HorizontalFlip | 0.9829 |
| MobileVitV2 | V1 | 1000 | 100 | 64 | AdamW | 0.001 | TripletLoss | Resize <br>Normalize <br>HorizontalFlip | 0.9748 |
| ConvNextV2-A | V1 | 1000 | 100 | 64 | AdamW | 0.001 | TripletLoss | Resize <br>Normalize <br>HorizontalFlip | 0.9721 |
| SqueezeNet | V1 | 1000 | 100 | 64 | AdamW | 0.001 | TripletLoss | Resize <br>Normalize <br>HorizontalFlip | 0.9758 |
| MobileNetV3 | V2 | 1000 | 100 | 64 | AdamW | 0.001 | TripletLoss | Resize <br>Normalize <br>HorizontalFlip | 0.8743 |
| MobileNetV3 | V2 | 1000 | 100 | 64 | AdamW | 0.001 | QuadrupletLoss | Resize <br>Normalize <br>HorizontalFlip | 0.9782 |
| SqueezeNetMod | V2 | 1000 | 500 | 64 | AdamW | 0.001 | QuadrupletLoss | Resize <br>Normalize <br>HorizontalFlip | 0.9857 |
| MobileNetV3 | V2 | 1000 | 500 | 64 | AdamW | 0.001 | QuadrupletLoss | Resize <br>Normalize <br>HorizontalFlip | 0.9908 |
[^1]: Dataset V1 and V2 are our own custom datasets. V2 contains more data and identities
## 🛠️ Installation
```py
git clone https://github.com/boostcampaitech5/level3_cv_finalproject-cv-07.git
cd level3_cv_finalproject-cv-07
conda env create --name <env_name> -f env.yaml
```

## 👨🏻‍💻 Train & Inference with Just 1 Command Line
### Train Detection Model
---
```
cd detection/tools
python3 train.py --exp_name exp1 --input_dim (1920,1088) --epochs 100 --lr 0.0001 --batch_size 8 --optimizer AdamW --num_workers 4 --warmup_initial_lr 0.00001 --lr_warmup_epochs 5 --score_thr 0.8 --nms_thr 0.8 -- metric [F1@0.50/Map@0.50] --fp16 [True/False]
```
* --exp_name: experiement directory name
* --input_dim: input dimensions
* --epochs: epoch
* --lr: learning rate
* --batch_size: batch_size
* --optimizer: optimizer
* --num_workers: dataloader num workers
* --warmup_initial_lr: warmup initial learning rate
* --lr_warmup_epochs: learning rate warmup epochs
* --score_thr: score threshold
* --nms_thr: non-max suppression threshold
* --metric: evaluation metric
* --fp16: mixed precision training

### Inference Detection Model
---
```
cd detection/tools
python3 inference.py --image [True/False] --video [True/False] --file_name image1.png --conf 0.5 --iou 0.5 --model_weight yolo_nas_l_best.pth
```
Both `image` and `video` cannot be set into `True` at the same time!
* --image: inference on image
* --video: inference on video
* --file_name: image/video file to be inferenced
* --conf: confidence threshold
* --iou: iou threshold
* --model_weight: model weight file

### Train Person Re-Identification Model
---
```
cd re_id/tools
python3 train.py --demo [True/False] --seed 1 --model mobilenetv3 --epoch 100 --train_batch 64 --valid_batch 256 --lr 0.001  --num_workers 4 --quadruplet True --scheduler [True/False] --fp [True/False]
```
* --demo: `True` uses DeepSportsRadar dataset | `False` uses custom dataset
* --seed: seed number
* --model: model
* --epoch: epoch
* --train_batch: train batch size
* --valid_batch: valid batch size
* --lr: learning rate
* --num_workers: dataloader num workers
* --quadruplet:  `True` uses quadruplet loss | `False` uses triple loss
* --fp: mixed precision training

### Inference Person Re-Identification Model
---
```
cd re_id/tools
python3 inference.py --demo [True/False] --model mobilenetv3 --model_weight mobilenetv3_best.pth --batch_size 256 --num_workers 8 --query_index 0
```
* --demo: `True` uses DeepSportsRadar dataset | `False` uses custom dataset
* --model: model
* --model_weight: model weight file
* --batch size: test batch size
* --num worker: dataloader num worker
* --query_index: query index