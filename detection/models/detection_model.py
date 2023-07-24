import os
import torch
import numpy as np
import torch.nn as nn

#super_gradients
from super_gradients.training import models
from super_gradients.common.object_names import Models


class Yolo_Nas_L(nn.Module):
    def __init__(self, num_classes=6, checkpoint_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        if self.checkpoint_path:
            model =  models.get(Models.YOLO_NAS_L, num_classes=self.num_classes,
                    checkpoint_path=self.checkpoint_path).cuda()
            print('weight loaded!!')

        else:
            model =  models.get(Models.YOLO_NAS_L, num_classes=self.num_classes).cuda()

        self.model = model

    def forward(self,x):
        return self.model(x)

    def predict(self, img):
        results = self.model.predict(img, fuse_model=False)
        
        labels = np.array(results[0].prediction.labels)
        confidence = np.array(results[0].prediction.confidence)
        bboxes = results[0].prediction.bboxes_xyxy
        outs = (bboxes, confidence, labels)
        outs = list(zip(*outs))

        lst = []
        for bb in outs:
            (x1, y1, x2, y2), c, l = bb
            lst.append([x1,y1,x2,y2,c,l])
        
        return torch.tensor(lst)
        
        
    