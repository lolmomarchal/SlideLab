import os
import tqdm
import torch
import numpy as np
import pandas as pd
import h5py
# import psutil
import time
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import gc
import threading

torch.backends.cudnn.benchmark = True  # Optimize for varying input sizes

def encoder(encoder_type="resnet50", device="cpu"):
    if encoder_type == "resnet50":
        encoder_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        encoder_model = torch.nn.Sequential(*list(encoder_model.children())[:-1])
        encoder_model.eval().to(device)
    return encoder_model

class TilePreprocessing(Dataset):
    def __init__(self, df_file):
        df = pd.read_csv(df_file)
        self.data = df[["x", "y", "tile_path"]].values  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y, tile_path = self.data[idx]
        image = read_image(tile_path).float() / 255.0  
        return x, y, image, tile_path

# def monitor_system():
#     pid = os.getpid()
#     process = psutil.Process(pid)
#     while True:
#         cpu_usage = process.cpu_percent()
#         mem_usage = process.memory_info().rss / (1024 ** 3)  
#         gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
#         print(f"CPU: {cpu_usage:.2f}% | Memory: {mem_usage:.2f} GB | GPU Memory: {gpu_mem:.2f} GB")
#         time.sleep(5)

def encode_tiles(patient_id, tile_path, result_path, device="cpu", batch_size=16, encoder_model="resnet50"):
    print(f"Encoding: {patient_id} on {device}")
    
    if device == "cuda":
        torch.cuda.empty_cache()  # Ensure clean GPU memory

    # monitor_thread = threading.Thread(target=monitor_system, daemon=True)
    # monitor_thread.start()
    
    encoder_ = encoder(encoder_type=encoder_model, device=device)
    tile_dataset = TilePreprocessing(tile_path)
    data_loader = DataLoader(tile_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device == "cuda"))

    all_features, all_x, all_y, all_tile_paths = [], [], [], []
    
    with torch.no_grad():
        for x, y, images, tile_paths in tqdm.tqdm(data_loader, desc="Encoding Tiles"):
            all_x.extend(x.numpy())
            all_y.extend(y.numpy())
            all_tile_paths.extend(tile_paths)

            images = images.to(device, dtype= torch.float32)  
            features = encoder_(images).squeeze(-1).squeeze(-1)
            
            all_features.append(features.cpu().numpy()) 
            del features, images
            torch.cuda.empty_cache()

    del encoder_
    gc.collect()
    torch.cuda.empty_cache()

    df = pd.read_csv(tile_path)
    all_features = np.concatenate(all_features, axis=0).astype(np.float32)
    all_x = np.array(all_x, dtype=np.float32)
    all_y = np.array(all_y, dtype=np.float32)
    mag = df["desired_magnification"].to_numpy(dtype=np.float32)
    size = df["desired_size"].to_numpy(dtype=np.float32)

    result_path = os.path.join(result_path, f"{patient_id}.h5")
    
    with h5py.File(result_path, "w") as hdf:
        hdf.create_dataset("tile_path", data=np.array(all_tile_paths, dtype="S"))
        hdf.create_dataset("x", data=all_x)
        hdf.create_dataset("y", data=all_y)
        hdf.create_dataset("mag", data=mag)
        hdf.create_dataset("size", data=size)
        hdf.create_dataset("features", data=all_features)

    # monitor_thread.join()
