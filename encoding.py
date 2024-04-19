import os.path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional
import torchvision
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import pickle
import torchvision.transforms as transforms
# for pathology encoder
from PIL import Image
from encoders.CONCH.conch.open_clip_custom import create_model_from_pretrained


def encoder(encoder_type = 0):
    if encoder_type == 0:
       encoder_model = models.resnet50(pretrained = True)
       encoder_model = torch.nn.Sequential(*list(encoder_model.children())[:-1])
       # encoder_model.add_module("GlobalAvgPool", torch.nn.AdaptiveAvgPool2d((1, 1)))
       # encoder_model.add_module("Flatten", torch.nn.Flatten())
       encoder_model.eval()
       return encoder_model

    # elif encoder_type == 1:
    # else:

def encode_tiles(patient_id,tile_path, result_path):
    encoder_model = encoder(encoder_type = 0)
    patient_tiles = {}
    read = pd.read_csv(tile_path)
    preprocess = transforms.Compose([transforms.Resize((512, 512)),  # Resize the image to fit ResNet-50 input size
                                     transforms.ToTensor()])
    for i, row in read.iterrows():
        # get individual patients
        path_to_tile = row["path_to_slide"]
        try:
            tile = Image.open(path_to_tile)
            if tile is not None:
                tile = preprocess(tile)
                tile = tile.unsqueeze(0)
                with torch.no_grad():
                    encoded_features = encoder_model(tile)
                patient_tiles[path_to_tile] = encoded_features
        except:
            print("tile is none")
    patient_results = os.path.join(result_path, patient_id)
    with open(patient_results, "wb") as f:
        pickle.dump(patient_tiles, f)


