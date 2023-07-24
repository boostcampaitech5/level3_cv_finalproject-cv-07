import os
import cv2
import torch
import shutil
import argparse
import matplotlib.pyplot as plt

from classes import class_names
from tqdm import tqdm
from super_gradients.training import models
from super_gradients.common.object_names import Models

#----------------------------------------------------------------------------------------------------------------------#  
# Initialization                                                                                                       #
#----------------------------------------------------------------------------------------------------------------------#
class_names = class_names
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#----------------------------------------------------------------------------------------------------------------------#  
# Argument Parser                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------# 
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=bool, default=False) 
parser.add_argument('--video', type=bool, default=False)
parser.add_argument('--path', type=str, default=None)
parser.add_argument('--conf', type=float, default=0.25)
parser.add_argument('--iou', type=float, default=0.35)
parser.add_argument('--model_weight', type=str, default=None)
args = parser.parse_args()

if __name__ == "__main__":
    assert args.image != None or args.video != None
    assert args.path != None or args.model_weight != None

    if args.image:
        fig, ax = plt.subplots(1, figsize=(16, 9))
        fig.tight_layout()

        image = cv2.imread(os.path.join("../data/images", args.path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        net = models.get(Models.YOLO_NAS_L, num_classes=len(class_names), checkpoint_path=os.path.join("../model_weights", args.model_weight)).half().to(device)
        net.set_dataset_processing_params(conf=args.conf)
        net.set_dataset_processing_params(iou=args.iou)
        prediction = net.predict(image, fuse_model=False)
        frame_prediction = prediction[0]

        print(f"Inferencing image using {device}...")
        labels = frame_prediction.prediction.labels
        confidence = frame_prediction.prediction.confidence
        bboxes = frame_prediction.prediction.bboxes_xyxy

        for label, conf, bbox in zip(labels, confidence, bboxes):
            # Draw detected bounding boxes
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1)
            ax.add_patch(rect)

            # Convert label index to class name
            class_index = int(label)
            class_name = class_names[class_index]

            # Adjust label position
            text_x, text_y = x1, y1 - 10
            if text_y < 10:
                text_y = y2 + 10

            ax.text(text_x, text_y, f'{class_name}: {conf:.2f}', fontsize=10, color='white')
            ax.axis('off')
            ax.imshow(image, aspect='auto')
        fig.savefig(f"../results/{(args.path).split('.')[0]}_result.jpg", bbox_inches='tight', pad_inches=0)
        print("Image Inference Completed!")
        print()

    elif args.video:
        net = models.get(Models.YOLO_NAS_L, num_classes=len(class_names), checkpoint_path=os.path.join("../model_weights", args.model_weight)).half().to(device)
        net.set_dataset_processing_params(conf=args.conf)
        net.set_dataset_processing_params(iou=args.iou)
        print(f"Inferencing video using {device}...")
        prediction1 = net.predict(os.path.join("../data/video",args.path), fuse_model=True)

        cap = cv2.VideoCapture(os.path.join("../data/video",args.path))
        if not cap.isOpened():
            print("Video is failed to open!")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        filename = os.path.join("../data/video",args.path).split('/')[-1].split('.')[0] + '_result.mp4'
        video_out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        shutil.move(filename, os.path.join("../results", filename))

        with tqdm(total = len(prediction1)) as pbar:
            for frame_index, frame_prediction in enumerate(prediction1):
                labels = frame_prediction.prediction.labels
                confidence = frame_prediction.prediction.confidence
                bboxes = frame_prediction.prediction.bboxes_xyxy
                frame = frame_prediction.draw(box_thickness=1)
            
                video_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                pbar.update(1)

        video_out.release()
        print("Video Inference Completed!")
        print()