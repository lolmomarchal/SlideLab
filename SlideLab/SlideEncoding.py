import os
import tqdm
import torch
import numpy as np
import pandas as pd
import h5py
import psutil
import time
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch.multiprocessing as mp
import queue
import gc
import threading
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

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

def monitor_system():
    """Monitors and logs CPU, memory, and GPU usage."""
    pid = os.getpid()
    process = psutil.Process(pid)
    while True:
        cpu_usage = process.cpu_percent()
        mem_usage = process.memory_info().rss / (1024 ** 3)  
        gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
        print(f"CPU: {cpu_usage:.2f}% | Memory: {mem_usage:.2f} GB | GPU Memory: {gpu_mem:.2f} GB")
        time.sleep(5)

def encode_tiles(patient_id, tile_path, result_path, device="cpu", batch_size=1000, max_queue=4, encoder_model="resnet50", high_qual=False):
    print(f"Encoding: {patient_id} on {device}")
    print(torch.cuda.is_available())
    monitor_thread = threading.Thread(target=monitor_system, daemon=True)
    monitor_thread.start()
    
    encoder_ = encoder(encoder_type=encoder_model, device=device)
    tile_dataset = TilePreprocessing(tile_path, device=device)

    # batch_queue = queue.Queue(maxsize=max_queue) 
    all_features, all_x, all_y, all_tile_paths = [], [], [], []
    # stop_signal = object()
    batch_counter = 0
    for x, y, images, tile_paths in tqdm.tqdm(DataLoader(tile_dataset, batch_size=batch_size, shuffle=False), desc="Encoding Tiles"):
        with torch.no_grad():
            all_x.extend(x)
            all_y.extend(y)
            all_tile_paths.extend(tile_paths)
            images = images.to(device)
            features = encoder_(images)
            all_features.append(features.squeeze(-1).squeeze(-1).cpu())
            del features, x, y, images, tile_paths
            batch_counter+=1
            if batch_counter%50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    del encoder_
    gc.collect()
    torch.cuda.empty_cache()
    monitor_thread.join()

            

            
    # monitor_thread = threading.Thread(target=monitor_system, daemon=True)
    # monitor_thread.start()

    # def encode_worker():
    #     nonlocal batch_counter
    #     with torch.no_grad():
    #         while True:
    #             batch = batch_queue.get()
    #             if batch is stop_signal:
    #                 break
    #             x, y, images, tile_paths = batch
    #             images = images.to(device)
                
    #             start_time = time.time()
    #             # print(f"Images are on device: {images.device}")
    #             # print(f"encoder is on device: {encoder_.device}")
    #             features = encoder_(images)
                
    #             end_time = time.time()
                
    #             batch_time = end_time - start_time
    #             # print(f"Batch {batch_counter}: Processing Time = {batch_time:.4f} sec")
                
    #             all_features.append(features.squeeze(-1).squeeze(-1).cpu())
    #             all_x.extend(x)
    #             all_y.extend(y)
    #             all_tile_paths.extend(tile_paths)
    #             batch_queue.task_done()
    #             del features, x, y, images, tile_paths
                
    #             batch_counter += 1
    #             if batch_counter % 50 == 0:
    #                 # print("Clearing CUDA cache and collecting garbage")
    #                 torch.cuda.empty_cache()
    #                 gc.collect()

    # worker_thread = threading.Thread(target=encode_worker)
    # worker_thread.start()
    
    # for x, y, images, tile_paths in tqdm.tqdm(DataLoader(tile_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cpu")), desc="Encoding Tiles"):
    #     batch_queue.put((x, y, images, tile_paths))

    # batch_queue.put(stop_signal)  
    # worker_thread.join() 
    
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
