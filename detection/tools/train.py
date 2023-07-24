from super_gradients.training import models
from super_gradients.common.object_names import Models
from super_gradients.training.datasets.detection_datasets.coco_format_detection import COCOFormatDetectionDataset
from super_gradients.training.transforms.transforms import DetectionMosaic, DetectionRandomAffine, DetectionHSV, \
    DetectionHorizontalFlip, DetectionPaddedRescale, DetectionStandardize, DetectionTargetsFormatTransform
from super_gradients.training.utils.detection_utils import DetectionCollateFN
from super_gradients.training import dataloaders
from super_gradients.training.datasets.datasets_utils import worker_init_reset_seed
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050, DetectionMetrics_050_095
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training import Trainer
from super_gradients.common.object_names import Models
from super_gradients.training import models


model = models.get(Models.YOLO_NAS_L, num_classes=6, pretrained_weights="coco").cuda()


INPUT_DIM = (1920,1088)
trainset = COCOFormatDetectionDataset(data_dir="/opt/ml/final_project/final_dataset/",
                                      images_dir="train",
                                      json_annotation_file="train/_annotations.coco.json",
                                      input_dim=INPUT_DIM,
                                      ignore_empty_annotations=False,
                                      transforms=[
                                          DetectionMosaic(prob=0.5, input_dim=INPUT_DIM),
                                          DetectionRandomAffine(degrees=10., scales=(0.75, 1.0), shear=0.,
                                                                target_size=INPUT_DIM),
                                          DetectionHSV(prob=0.5, hgain=2, vgain=15, sgain=15),
                                          DetectionHorizontalFlip(prob=0.5),
                                          DetectionPaddedRescale(input_dim=INPUT_DIM),
                                          DetectionStandardize(max_value=255),
                                          DetectionTargetsFormatTransform(input_dim=INPUT_DIM, output_format="LABEL_CXCYWH")
                                      ])


valset = COCOFormatDetectionDataset(data_dir="/opt/ml/final_project/final_dataset/",
                                    images_dir="valid",
                                    json_annotation_file="valid/_annotations.coco.json",
                                    input_dim=INPUT_DIM,
                                    ignore_empty_annotations=False,
                                    transforms=[
                                        DetectionPaddedRescale(input_dim=INPUT_DIM),
                                        DetectionStandardize(max_value=255),
                                        DetectionTargetsFormatTransform(input_dim=INPUT_DIM, output_format="LABEL_CXCYWH")
                                    ])


train_loader = dataloaders.get(dataset=trainset, dataloader_params={
    "shuffle": True,
    "batch_size": 8,
    "num_workers": 4,
    "drop_last": False,
    "pin_memory": True,
    "collate_fn": DetectionCollateFN(),
    "worker_init_fn": worker_init_reset_seed
})

valid_loader = dataloaders.get(dataset=valset, dataloader_params={
    "shuffle": False,
    "batch_size": 32,
    "num_workers": 2,
    "drop_last": False,
    "pin_memory": True,
    "collate_fn": DetectionCollateFN(),
    "worker_init_fn": worker_init_reset_seed
})


train_params = {
    "warmup_initial_lr": 1e-5,
    "initial_lr": 1e-4,
    "batch_accumulate": 64,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "AdamW",
    "zero_weight_decay_on_bias_and_bn": True,
    "lr_warmup_epochs": 5,
    "warmup_mode": "linear_epoch_step",
    "optimizer_params": {"weight_decay": 0.00001},
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    "max_epochs": 500,
    "mixed_precision": True,
    "loss": PPYoloELoss(use_static_assigner=False, num_classes=6, reg_max=16),
    "valid_metrics_list": [
        DetectionMetrics_050(score_thres=0.8, top_k_predictions=300, num_cls=6, normalize_targets=True,
                             post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.15,
                                                                                    nms_top_k=5000, max_predictions=300,
                                                                                    nms_threshold=0.8)),
        DetectionMetrics_050_095(score_thres=0.8, top_k_predictions=300, num_cls=6, normalize_targets=True,
                             post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.15,
                                                                                    nms_top_k=5000, max_predictions=300,
                                                                                    nms_threshold=0.8))],
    "metric_to_watch": 'F1@0.50',
    "sg_logger": "wandb_sg_logger",
    "sg_logger_params": {
                    "project_name": "basketball", # W&B project name
                    "save_checkpoints_remote": True,
                    "save_tensorboard_remote": True,
                    "save_logs_remote": True
                    } 
    }


trainer = Trainer(experiment_name="final", ckpt_root_dir="/opt/ml/final_project/super_gradients/workspace")
trainer.train(model=model, training_params=train_params, train_loader=train_loader, valid_loader=valid_loader)