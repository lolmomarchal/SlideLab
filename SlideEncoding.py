import os

import h5py
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import ResNet50_Weights


def encoder(encoder_type=0, device='cpu'):
    if encoder_type == 0:
        encoder_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove the last fully connected layer
        encoder_model = torch.nn.Sequential(*list(encoder_model.children())[:-1])
        encoder_model.eval()
        encoder_model.to(device)
    # option 1: histopathology trained
    return encoder_model


def encode_tiles(patient_id, tile_path, result_path, device='cpu'):
    encoder_model = encoder(encoder_type=0, device=device)
    patient_tiles = {}
    read = pd.read_csv(tile_path)
    total_data = []
    patient_id = read["patient_id"].iloc[0]

    preprocess = transforms.Compose([transforms.ToTensor()])
    for i, row in read.iterrows():
        path_to_tile = row["path_to_slide"]
        try:
            tile = Image.open(path_to_tile)
            if tile is not None:
                tile = preprocess(tile).unsqueeze(0).to(device)
                with torch.no_grad():
                    encoded_features = encoder_model(tile)
                total_data.append(encoded_features.squeeze())
        except Exception as e:
            print(f"Error processing tile {path_to_tile}: {e}")
    with h5py.File(os.path.join(result_path, str(patient_id) + ".h5"), "w") as hdf:
        hdf.create_dataset('tile_paths', data=read["path_to_slide"])
        hdf.create_dataset('x', data=read["x"])
        hdf.create_dataset('y', data=read["y"])
        hdf.create_dataset('scale', data=read["scale"])
        hdf.create_dataset('mag', data=read["magnification"])
        hdf.create_dataset('size', data=read["size"])
        features_dataset = hdf.create_dataset('features', data=torch.stack(total_data))

    # # create hdf5 file
    # for i, row in read.iterrows():
    #     # get individual tiles
    #     path_to_tile = row["path_to_slide"]
    #     try:
    #         tile = Image.open(path_to_tile)
    #         if tile is not None:
    #             tile = preprocess(tile).unsqueeze(0).to(device)
    #             with torch.no_grad():
    #                 encoded_features = encoder_model(tile)
    #             patient_tiles[path_to_tile] = encoded_features.cpu()  # Move tensor to CPU before saving
    #     except Exception as e:
    #         print(f"Error processing tile {path_to_tile}: {e}")
    # patient_results = os.path.join(result_path, f"{patient_id}.pkl")
    # with open(patient_results, "wb") as f:
    #     pickle.dump(patient_tiles, f)
