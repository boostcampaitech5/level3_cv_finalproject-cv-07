import os
import sys
import torch
import timm
import open_clip
import numpy as np
import torch.nn as nn
import torchvision.models as models

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from module.cbam import SAM, CAM, CBAM
from torchvision.models.squeezenet import SqueezeNet1_1_Weights

class SqueezeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.squeezenet = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
    
    def forward(self, x):
        embedding_feature = self.squeezenet(x)
        return embedding_feature
    
class SqueezeNetMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.squeezenet = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
        self.squeezenet.features[5] = nn.Sequential(CBAM(128, r=8), nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True))
        self.squeezenet.features[8] = nn.Sequential(CBAM(256, r=8), nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True))
        self.squeezenet.features[12] = nn.Sequential(self.squeezenet.features[12], CBAM(512, r=8))
        
    def forward(self, x):
        return self.squeezenet(x)

class MobileVitV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilevitv2 = timm.create_model("mobilevitv2_100.cvnets_in1k")
    
    def forward(self, x):
        embedding_feature = self.mobilevitv2(x)
        return embedding_feature
      
class MobileNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenetv3 = timm.create_model('mobilenetv3_large_100.miil_in21k_ft_in1k')
    
    def forward(self, x):
        embedding_feature = self.mobilenetv3(x)
        return embedding_feature

class ConvNextV2_A(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnextv2_t = timm.create_model('convnextv2_atto.fcmae_ft_in1k')
     
    def forward(self, x):
        embedding_feature = self.convnextv2_t(x)
        return embedding_feature
    
class ConvNextV2_F(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnextv2_f = timm.create_model('convnextv2_femto.fcmae_ft_in1k')
     
    def forward(self, x):
        embedding_feature = self.convnextv2_f(x)
        return embedding_feature
    
class ConvNextV2_P(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnextv2_p = timm.create_model('convnextv2_pico.fcmae_ft_in1k')
     
    def forward(self, x):
        embedding_feature = self.convnextv2_p(x)
        return embedding_feature
    
class ConvNextV2_N(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnextv2_n = timm.create_model('convnextv2_nano.fcmae_ft_in22k_in1k')
     
    def forward(self, x):
        embedding_feature = self.convnextv2_n(x)
        return embedding_feature

class ConvNextV2_T(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnextv2_t = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k')
     
    def forward(self, x):
        embedding_feature = self.convnextv2_t(x)
        return embedding_feature

class ConvNextV2_B(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnextv2_b = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k')
     
    def forward(self, x):
        embedding_feature = self.convnextv2_b(x)
        return embedding_feature
    
class ConvNextV2_L(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnextv2_l = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k')
     
    def forward(self, x):
        embedding_feature = self.convnextv2_l(x)
        return embedding_feature
    
class TimmModel(nn.Module):
    def __init__(self, 
                 model_name='convnextv2_huge.fcmae_ft_in1k',
                 pretrained=True):
        super(TimmModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0) 
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            
    def forward(self, img1, img2=None):
        if img2 is not None:
            images = torch.cat([img1, img2], dim=0)
            image_features = self.model(images)
            
            image_features1 = image_features[:len(img1), :]
            image_features2 = image_features[len(img1):, :]
            
            return image_features1, image_features2
            
        else:
            image_features = self.model(img1)
            
        return image_features
    
class OpenClipModel(nn.Module):
    def __init__(self,
                 model_name='ViT-L-14',
                 pretrained=True,
                 remove_proj=True):
        super(OpenClipModel, self).__init__()
        self.model_name = model_name
        self.model = open_clip.create_model(model_name, pretrained=pretrained)  
         
        # delete text parts of clip model
        del(self.model.transformer)
        del(self.model.token_embedding)
        del(self.model.ln_final)
        del(self.model.positional_embedding)
        del(self.model.text_projection)
        
        if remove_proj and "ViT" in model_name:
            width, output_dim = self.model.visual.proj.shape
            print("Remove Projection Layer - old output size: {} - new output size: {}".format(output_dim, width))
            self.model.visual.proj = None

    def set_grad_checkpoint(self, enable=True): 
        if "ViT" in self.model_name:
            self.model.visual.set_grad_checkpointing(enable)
            print("Use Gradient Checkpointing for {}".format(self.model_name))
        else:
            print("Gradient Checkpointing not available for {}".format(self.model_name))
        
    def get_image_size(self):
            return self.model.visual.image_size
    
    def forward(self, img1, img2=None):
        if img2 is not None:
            images = torch.cat([img1, img2], dim=0)
            image_features = self.model.encode_image(images)
            
            image_features1 = image_features[:len(img1), :]
            image_features2 = image_features[len(img1):, :]
            
            return image_features1, image_features2
            
        else:
            image_features = self.model.encode_image(img1)
            
        return image_features