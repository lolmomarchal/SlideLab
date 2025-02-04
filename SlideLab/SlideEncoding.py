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
import torch.multiprocessing as mp
import queue
import threading
import gc
torch.backends.cudnn.benchmark = True 

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

def encode_tiles(patient_id, tile_path, result_path, device="cpu", batch_size=64,  encoder_model="resnet50", high_qual = False ):
    encoder_ = encoder(encoder_type=encoder_model, device=device)
    tile_dataset = TilePreprocessing(tile_path, device=device)
    all_features, all_x, all_y, all_tile_paths, high_qual_all= [], [], [], [], []
    data_loader = DataLoader(tile_dataset, batch_size=batch_size, shuffle=False)
    batch_ = 0

    with torch.no_grad():
            for x, y, images, tile_paths in tqdm.tqdm(data_loader, desc=f"Encoding Tiles: {patient_id} on {device}"):
                all_x.extend(x.numpy())
                all_y.extend(y.numpy())
                all_tile_paths.extend(tile_paths)
                images = images.to(device) 
                features = encoder_(images).squeeze(-1).squeeze(-1)
                all_features.append(features.cpu().numpy())
                
                del features, images,x,y,tile_paths
                batch_+=1
                if batch_%50 ==0:
                    gc.collect()
                    torch.cuda.empty_cache()
    
    del encoder_
    gc.collect()
    torch.cuda.empty_cache()
    df = pd.read_csv(tile_path)
    mag = df["desired_magnification"].to_numpy(dtype=np.float32)
    size = df["desired_size"].to_numpy(dtype=np.float32)
    all_features = [torch.tensor(f) if isinstance(f, np.ndarray) else f for f in all_features]
    all_features = torch.cat(all_features, dim=0).numpy().astype(np.float32)
    all_x = np.array(all_x, dtype=np.float32)
    all_y = np.array(all_y, dtype=np.float32)
    result_path = os.path.join(result_path, f"{patient_id}.h5")
    
    with h5py.File(result_path, "w") as hdf:
        hdf.create_dataset("tile_path", data=np.array(all_tile_paths, dtype="S"))
        hdf.create_dataset("x", data=all_x)
        hdf.create_dataset("y", data=all_y)
        hdf.create_dataset("features", data=all_features)
        hdf.create_dataset("mag", data=mag)
        hdf.create_dataset("size", data=size)
