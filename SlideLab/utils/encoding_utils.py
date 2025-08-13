import os
import torch
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
from torchvision import transforms as T
import torch.nn as nn
import timm
from huggingface_hub import hf_hub_download
import sys

torch.backends.cudnn.benchmark = True
####### HELPER

## dummy module
#
class empty(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = "empty"
    def forward(self,x):
        return torch.zeros((1, 2048), dtype=torch.float32)


def get_truncated_resnet50():
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    truncated = nn.Sequential(
        resnet50.conv1,
        resnet50.bn1,
        resnet50.relu,
        resnet50.maxpool,
        resnet50.layer1,
        resnet50.layer2,
        resnet50.layer3,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1)
    )
    for param in truncated.parameters():
        param.requires_grad = False
    return truncated
def download_and_initiate_UNI(token, device):
    model_path = os.path.join(sys.prefix,"model_weights", "pytorch_model.bin")
    dir_path =  os.path.join(sys.prefix,"model_weights")
    os.makedirs(dir_path, exist_ok = True)
    if not os.path.exists(model_path):
        if token is None:
            raise Exception("Please request access to UNI model from https://huggingface.co/MahmoodLab/UNI2-h and obtain a valid token from huggingface/profile/settings/Access Tokens")
        try:
            file_path = hf_hub_download(repo_id = "MahmoodLab/UNI2-h",
                                        filename = "pytorch_model.bin",local_dir = dir_path,  token = token,
                                        force_download = True)
        except Exception as e:
            raise Exception(f"Something went wrong when installing UNI2-h: {e}")
    timm_kwargs = {
        'model_name': 'vit_giant_patch14_224',
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    }
    model = timm.create_model(
        pretrained=False, **timm_kwargs
    )
    model.load_state_dict(torch.load(model_path, map_location = "cpu"), strict = True)
    model.eval().to(device)
    return model


#### MAIN ######################
class Encoder:
    def __init__(self, device, type, token= None):
        self.type = type
        self.device = device
        self.token = token
        self.model, self.transforms = self.get_model_and_transform()


    def get_attributes(self):
        return self.model, self.transforms
        # TODO: add more options
    def get_model_and_transform(self):
        transforms  =  T.Compose(
            [
                T.Resize(224, antialias=True),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        if self.type == "resnet50":
            encoder_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            encoder_model = torch.nn.Sequential(*list(encoder_model.children())[:-1])
            encoder_model.eval().to(self.device)

        elif self.type == "resnet50_truncated":

            encoder_model = get_truncated_resnet50()
            encoder_model.eval().to(self.device)
        elif self.type == "mahmood-uni":
            encoder_model = download_and_initiate_UNI(self.token, self.device)

        elif self.type == "empty":
            encoder_model = empty()
            # self.transforms = None

        return encoder_model, transforms



