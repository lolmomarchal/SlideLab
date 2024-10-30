import os
import h5py
import pandas as pd
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.io import imread  
import time
from torchvision.models import ResNet50_Weights

def encoder(encoder_type=0, device='cpu'):
    if encoder_type == 0:
        encoder_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove the last fully connected layer
        encoder_model = torch.nn.Sequential(*list(encoder_model.children())[:-1])
        encoder_model.eval()
        encoder_model.to(device)
    return encoder_model

def preprocess_image(path_to_tile, device='cpu'):
    preprocess = transforms.Compose([transforms.ToTensor()])
    try:
        tile = imread(path_to_tile) =
        if tile is not None:
            tile = preprocess(tile).unsqueeze(0).to(device)
            return tile
        else:
            return None
    except Exception as e:
        print(f"Error processing tile {path_to_tile}: {e}")
        return None

def encode_tiles(patient_id, tile_path, result_path, device='cpu'):
    print(tile_path)
    start = time.time()
    encoder_model = encoder(encoder_type=0, device=device)

    read = pd.read_csv(tile_path).dropna()
    read["tile_path"] = np.array(read["tile_path"])
    total_data = []

    for path in read["tile_path"]:
        tile = preprocess_image(path, device)
        if tile is not None:
            with torch.no_grad():
                encoded_feature = encoder_model(tile).squeeze().cpu()
            total_data.append(encoded_feature)

    # print("finished encoding tiles")
    finish_encoding = time.time()
    print(f"Encoding time: {finish_encoding - start}")

    #total_data = [data for data in total_data if data.numel() < 2048]

    with h5py.File(os.path.join(result_path, f"{patient_id}.h5"), "w") as hdf:
        hdf.create_dataset('tile_paths', data=read["tile_path"], dtype=h5py.string_dtype(encoding='utf-8'))
        hdf.create_dataset('x', data=read["x"])
        hdf.create_dataset('y', data=read["y"])
        hdf.create_dataset('scale', data=read["scale"])
        hdf.create_dataset('mag', data=read["desired_magnification"])
        hdf.create_dataset('size', data=read["desired_size"])
        hdf.create_dataset('features', data=torch.stack(total_data))
    # print(f"Finished writing to HDF5 in {time.time() - finish_encoding} seconds")
