import torch
import argparse
import wandb

from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.utils.detection_utils import DetectionCollateFN
from super_gradients.training.datasets.datasets_utils import worker_init_reset_seed
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.datasets.detection_datasets.coco_format_detection import COCOFormatDetectionDataset
from super_gradients.training.transforms.transforms import DetectionMosaic, DetectionRandomAffine, DetectionHSV, DetectionHorizontalFlip, \
                                                           DetectionPaddedRescale, DetectionStandardize, DetectionTargetsFormatTransform
from classes import class_names

#----------------------------------------------------------------------------------------------------------------------#  
# Initialization                                                                                                       #
#----------------------------------------------------------------------------------------------------------------------#
class_names = class_names
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#----------------------------------------------------------------------------------------------------------------------#  
# Argument Parser                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------# 
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--lr', type=int, default=0.0001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--fp16', type=bool, default=True)
parser.add_argument('--exp_name', type=str, default="exp1")
parser.add_argument('--warmup_initial_lr', type=int, default=0.00001)
parser.add_argument('--lr_warmup_epochs', type=int, default=5)
parser.add_argument('--optimizer', type=str, default="AdamW")
parser.add_argument('--score_thr', type=int, default=0.8)
parser.add_argument('--nms_thr', type=int, default=0.8)
parser.add_argument('--metric', type=str, default="F1@0.50")
parser.add_argument('--input_dim', type=tuple, default=(1920,1088))
args = parser.parse_args()

#----------------------------------------------------------------------------------------------------------------------#  
# Train & Validation Dataset                                                                                           #
#----------------------------------------------------------------------------------------------------------------------#
class_names = class_names
trainset = COCOFormatDetectionDataset(data_dir="../data/dataset/",
                                      images_dir="train",
                                      json_annotation_file="train.json",
                                      input_dim=args.input_dim,
                                      ignore_empty_annotations=False,
                                      transforms=[
                                          DetectionMosaic(prob=0.5, input_dim=args.input_dim),
                                          DetectionRandomAffine(degrees=10., scales=(0.75, 1.0), shear=0., target_size=args.input_dim),
                                          DetectionHSV(prob=0.5, hgain=2, vgain=15, sgain=15),
                                          DetectionHorizontalFlip(prob=0.5),
                                          DetectionPaddedRescale(input_dim=args.input_dim),
                                          DetectionStandardize(max_value=255),
                                          DetectionTargetsFormatTransform(input_dim=args.input_dim, output_format="LABEL_CXCYWH")])

valset = COCOFormatDetectionDataset(data_dir="../data/dataset/",
                                    images_dir="valid",
                                    json_annotation_file="valid.json",
                                    input_dim=args.input_dim,
                                    ignore_empty_annotations=False,
                                    transforms=[
                                        DetectionPaddedRescale(input_dim=args.input_dim),
                                        DetectionStandardize(max_value=255),
                                        DetectionTargetsFormatTransform(input_dim=args.input_dim, output_format="LABEL_CXCYWH") ])

if __name__ == "__main__":
    model = models.get(Models.YOLO_NAS_L, num_classes=len(class_names), pretrained_weights="coco").to(device)

    train_loader = dataloaders.get(dataset=trainset, dataloader_params={
    "shuffle": True,
    "batch_size": args.batch_size,
    "num_workers": args.num_workers,
    "drop_last": False,
    "pin_memory": True,
    "collate_fn": DetectionCollateFN(),
    "worker_init_fn": worker_init_reset_seed})

    valid_loader = dataloaders.get(dataset=valset, dataloader_params={
    "shuffle": False,
    "batch_size": args.batch_size,
    "num_workers": args.num_workers,
    "drop_last": False,
    "pin_memory": True,
    "collate_fn": DetectionCollateFN(),
    "worker_init_fn": worker_init_reset_seed})
    
    train_params = {
        "warmup_initial_lr": args.warmup_initial_lr,
        "initial_lr": args.lr,
        "batch_accumulate": 64,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": args.optimizer,
        "zero_weight_decay_on_bias_and_bn": True,
        "lr_warmup_epochs": args.lr_warmup_epochs,
        "warmup_mode": "linear_epoch_step",
        "optimizer_params": {"weight_decay": 0.00001},
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": args.epochs,
        "mixed_precision": args.fp16,
        "loss": PPYoloELoss(use_static_assigner=False, num_classes=len(class_names), reg_max=16),
        "valid_metrics_list": [
            DetectionMetrics_050(score_thres=args.score_thr, top_k_predictions=300, num_cls=len(class_names), normalize_targets=True,
                                post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.15,
                                                                                        nms_top_k=5000, max_predictions=300,
                                                                                        nms_threshold=args.nms_thr))],
        "metric_to_watch": args.metric,
        "sg_logger": "wandb_sg_logger",
        "sg_logger_params": {
                        "project_name": "basketball", # wandb project name
                        "save_checkpoints_remote": True,
                        "save_tensorboard_remote": True,
                        "save_logs_remote": True}}

    trainer = Trainer(experiment_name=args.exp_name, ckpt_root_dir="../model_weights")
    trainer.train(model=model, training_params=train_params, train_loader=train_loader, valid_loader=valid_loader)