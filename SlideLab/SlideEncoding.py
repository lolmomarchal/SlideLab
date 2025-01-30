import os
import tqdm
import torch
import time
import numpy as np
import pandas as pd
import h5py
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch.multiprocessing as mp
import gc

def encoder(encoder_type="resnet50", device="cpu"):
    """Initialize encoder model and move to the specified device (CPU or GPU)."""
    if encoder_type == "resnet50":
        encoder_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        encoder_model = torch.nn.Sequential(*list(encoder_model.children())[:-1])
        encoder_model.eval().to(device)  # Move model to the device

        if device == "cpu":
            encoder_model = torch.quantization.quantize_dynamic(encoder_model, {torch.nn.Linear}, dtype=torch.qint8)

    return encoder_model

class TilePreprocessing(Dataset):
    """Dataset class to preprocess the tiles from the CSV file."""
    def __init__(self, df_file, device="cpu"):
        df = pd.read_csv(df_file)
        self.data = df[["x", "y", "tile_path"]].values  
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y, tile_path = self.data[idx]
        image = read_image(tile_path).float() / 255.0  
        return x, y, image, tile_path

def encode_tiles(patient_id, tile_path, result_path, device="cpu", batch_size=256, encoder_model="resnet50", high_qual=False):
    """Function to encode tiles and save the results in an HDF5 file."""
    print(f"Encoding: {patient_id} on {device}")
    encoder_ = encoder(encoder_type=encoder_model, device=device)
    tile_dataset = TilePreprocessing(tile_path, device=device)

    all_features, all_x, all_y, all_tile_paths = [], [], [], []
    batch_counter = 0


    for x, y, images, tile_paths in tqdm.tqdm(DataLoader(tile_dataset, batch_size=batch_size, shuffle=False), desc="Encoding Tiles"):
        images = images.to(device)

        start_time = time.time()
        features = encoder_(images) 
        batch_time = time.time() - start_time

        print(f"Processed batch in {batch_time:.4f} sec | GPU Memory: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

        all_features.append(features.squeeze(-1).squeeze(-1).detach().cpu())
        all_x.extend(x)
        all_y.extend(y)
        all_tile_paths.extend(tile_paths)

        del features, x, y, images, tile_paths
        torch.cuda.empty_cache()
        gc.collect()
        
    df = pd.read_csv(tile_path)
    all_features = torch.cat(all_features, dim=0).numpy().astype(np.float32)
    all_x = np.array(all_x, dtype=np.float32)
    all_y = np.array(all_y, dtype=np.float32)
    mag = np.array(df["desired_magnification"], dtype=np.float32)
    size = np.array(df["desired_size"], dtype=np.float32)

    result_path = os.path.join(result_path, f"{patient_id}.h5")
    
    with h5py.File(result_path, "w") as hdf:
        hdf.create_dataset("tile_path", data=np.array(all_tile_paths, dtype="S"))
        hdf.create_dataset("x", data=all_x)
        hdf.create_dataset("y", data=all_y)
        hdf.create_dataset("mag", data=mag)
        hdf.create_dataset("size", data=size)
        hdf.create_dataset("features", data=all_features)

    print(f"Encoding for patient {patient_id} completed. Results saved to {result_path}.")
