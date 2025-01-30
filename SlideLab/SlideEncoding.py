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
    def __init__(self, df_file):
        df = pd.read_csv(df_file)
        self.data = df[["x", "y", "tile_path"]].values
        self.mag = df["desired_magnification"].values
        self.size = df["desired_size"].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y, tile_path = self.data[idx]
        image = read_image(tile_path).float() / 255.0
        return x, y, image, tile_path

def encode_tiles(patient_id, tile_path, result_path, device="cpu", batch_size=16, save_every=100, encoder_model="resnet50"):
    print(f"Encoding: {patient_id} on {device}")
    encoder_ = encoder(encoder_type=encoder_model, device=device)
    dataset = TilePreprocessing(tile_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cpu"))

    result_path = os.path.join(result_path, f"{patient_id}.h5")
    with h5py.File(result_path, "w") as hdf:
        hdf.create_dataset("tile_path", (len(dataset),), dtype="S256")
        hdf.create_dataset("x", (len(dataset),), dtype=np.float32)
        hdf.create_dataset("y", (len(dataset),), dtype=np.float32)
        hdf.create_dataset("mag", data=dataset.mag, dtype=np.float32)
        hdf.create_dataset("size", data=dataset.size, dtype=np.float32)
        hdf.create_dataset("features", (len(dataset), 2048), dtype=np.float32)  # Assuming ResNet50 output

        idx = 0
        with torch.no_grad():
            for x, y, images, tile_paths in tqdm.tqdm(dataloader, desc="Encoding Tiles"):
                features = encoder_(images.to(device)).squeeze(-1).squeeze(-1).cpu().numpy()

                batch_size = len(tile_paths)
                hdf["x"][idx:idx + batch_size] = x.numpy()
                hdf["y"][idx:idx + batch_size] = y.numpy()
                hdf["tile_path"][idx:idx + batch_size] = np.array(tile_paths, dtype="S")
                hdf["features"][idx:idx + batch_size] = features

                idx += batch_size
                if idx % save_every == 0:
                    hdf.flush()  # Save progress to disk

                del features
                torch.cuda.empty_cache()
