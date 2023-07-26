import torch
import argparse
import datetime

from re_id import *
from detection import *
from utils.visualize import *
from detection.tools.classes import class_names
from re_id.models.model import *

#----------------------------------------------------------------------------------------------------------------------#  
# Initialization                                                                                                       #
#----------------------------------------------------------------------------------------------------------------------# 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

#----------------------------------------------------------------------------------------------------------------------#  
# Argument Parser                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------# 
parser = argparse.ArgumentParser()
parser.add_argument('--detection_weight', type=str, default=None) 
parser.add_argument('--reid_weight', type=str, default=None)
parser.add_argument('--reid_model', type=str, default='mobilenetv3')
parser.add_argument('--person_thr', type=float, default=0.5)
parser.add_argument('--cosine_thr', type=float, default=0.5)
parser.add_argument('--video_file', type=str, default=None) 
args = parser.parse_args()

#----------------------------------------------------------------------------------------------------------------------#  
# Load Object Detection & Person Re-Identification Model                                                               #
#----------------------------------------------------------------------------------------------------------------------# 
re_id_model = model_dict.get(args.reid_model).to(device)
re_id_checkpoint = f'./re_id/model_weights/{args.reid_weight}'
re_id = ReId(model=re_id_model, checkpoint=re_id_checkpoint, person_thr=args.person_thr, cosine_thr=args.cosine_thr)

detection_checkpoint = f'./detection/model_weights/{args.detection_weight}'
detection_model = Yolo_Nas_L(num_classes=len(class_names), checkpoint_path=detection_checkpoint).to(device)

#----------------------------------------------------------------------------------------------------------------------#  
# Video & Inference Path                                                                                               #
#----------------------------------------------------------------------------------------------------------------------# 
if not os.path.isdir("./final_results/"):
    os.mkdir("./final_results")

now = datetime.datetime.now()
now = f"{now.year}-{now.month}-{now.day}({now.hour}:{now.minute}:{now.second})"

video_path = f'./datasets/{args.video_file}'
os.mkdir(f"./final_results/{now}")
save_path = f'./final_results/{now}/result_{args.video_file}'

if __name__ == "__main__":
    make_predicted_video(detection_model, re_id, video_path, save_path)
    print('Inference is finished!')