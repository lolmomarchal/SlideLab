import os
import tqdm
import torch
import numpy as np
import pandas as pd
import h5py
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import gc
import random

torch.backends.cudnn.benchmark = True 

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

def encoder(encoder_type="resnet50", device="cpu"):
    if encoder_type == "resnet50":
        encoder_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        encoder_model = torch.nn.Sequential(*list(encoder_model.children())[:-1])
        encoder_model.eval().to(device)  
        if device == "cpu":
            encoder_model = torch.quantization.quantize_dynamic(encoder_model, {torch.nn.Linear}, dtype=torch.qint8)
    return encoder_model

class TilePreprocessing(Dataset):
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

def encode_tiles(patient_id, tile_path, result_path, device="cpu", batch_size=64, encoder_model="resnet50", seed=42):
    if seed is not None:
        set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    encoder_ = encoder(encoder_type=encoder_model, device=device)
    tile_dataset = TilePreprocessing(tile_path, device=device)
    data_loader = DataLoader(tile_dataset, batch_size=batch_size, shuffle=False)
    result_path = os.path.join(result_path, f"{patient_id}.h5")
    
    with h5py.File(result_path, "a") as hdf:
        if "x" not in hdf:
            mag = pd.read_csv(tile_path)["desired_magnification"].to_numpy(dtype=np.float32)
            size = pd.read_csv(tile_path)["desired_size"].to_numpy(dtype=np.float32)
            hdf.create_dataset("mag", data=mag)
            hdf.create_dataset("size", data=size)
            hdf.create_dataset("x", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=True)
            hdf.create_dataset("y", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=True)
            hdf.create_dataset("tile_path", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(), chunks=True)
            hdf.create_dataset("features", shape=(0, 2048), maxshape=(None, 2048), dtype=np.float32, chunks=True)
        
        all_x, all_y, all_tile_paths, all_features = [], [], [], []
        batch_ = 0
        
        with torch.no_grad():
            for x, y, images, tile_paths in tqdm.tqdm(data_loader, desc=f"Encoding Tiles: {patient_id} on {device}"):
                all_x.extend(x.numpy())
                all_y.extend(y.numpy())
                all_tile_paths.extend(tile_paths)
                images = images.to(device)
                features = encoder_(images).squeeze(-1).squeeze(-1).cpu().numpy()
                all_features.append(features)
                
                batch_ += 1
                if batch_ % 50 == 0:
                    hdf["x"].resize((hdf["x"].shape[0] + len(all_x),))
                    hdf["x"][-len(all_x):] = all_x
                    
                    hdf["y"].resize((hdf["y"].shape[0] + len(all_y),))
                    hdf["y"][-len(all_y):] = all_y
                    
                    hdf["tile_path"].resize((hdf["tile_path"].shape[0] + len(all_tile_paths),))
                    hdf["tile_path"][-len(all_tile_paths):] = np.array(all_tile_paths, dtype=h5py.string_dtype())
                    
                    all_features = np.concatenate(all_features, axis=0)
                    hdf["features"].resize((hdf["features"].shape[0] + all_features.shape[0], 2048))
                    hdf["features"][-all_features.shape[0]:] = all_features
                    
                    all_x, all_y, all_tile_paths, all_features = [], [], [], []
                    gc.collect()
                    torch.cuda.empty_cache()
    
    del encoder_
    gc.collect()
    torch.cuda.empty_cache()
