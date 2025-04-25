import os
import tqdm
import torch
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import gc
import math
torch.backends.cudnn.benchmark = True
from tiling.EncodingDatasets import TilePreprocessing_nosaving
# get encoder and  datasets
from utils.encoding_utils import Encoder
def collate_fn(batch):
        batch = [item for item in batch if not (item[0][0] == -1 and item[0][1] == -1)]
        if not batch:
            return None  # Skip entire batch

        coords = torch.stack([item[0] for item in batch])
        images = torch.stack([item[1] for item in batch])
        vars = torch.stack([item[2] for item in batch])
        x = coords[:, 0]
        y = coords[:, 1]
        return x, y, images, vars
# TODO: Instead of passing path, give dataloader already :"D
# Warning: Reduce batch_size when using augmentations!!!!!!
def encode_tiles(patient_id, tile_dataset, result_path, device="cpu", batch_size=512, encoder_model="resnet50",
                 high_qual=False, number_of_augmentation=0, token = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_, transforms = Encoder(device, encoder_model, token).get_model_and_transform()
    if number_of_augmentation != 0:
        batch_size = math.ceil(batch_size/number_of_augmentation)
    all_features, all_x, all_y, all_tile_paths = [], [], [], []
    high_qual_all = []
    pin_memory = device != "cpu"
    if isinstance(tile_dataset, TilePreprocessing_nosaving):


        data_loader = DataLoader(tile_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(),
                             pin_memory=pin_memory, collate_fn = collate_fn)
    else:
        data_loader = DataLoader(tile_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(),
                                 pin_memory=pin_memory)

    with torch.inference_mode():
        for batch in tqdm.tqdm(data_loader, desc=f"Encoding Tiles: {patient_id} on {device}"):
            # print(batch.shape)
            if batch is None:
                continue
            x, y, images, tile_paths = batch
            all_x.extend(x.numpy())
            all_y.extend(y.numpy())
            all_tile_paths.extend(tile_paths)
            images = images.to(device, non_blocking=True)

            if images.ndimension() == 4: 
                batch_size, C, H, W = images.shape
                features = encoder_(images).squeeze(-1).squeeze(-1)
                features = features.cpu().numpy()
    
            else:  
                batch_size, num_versions, C, H, W = images.shape
                stacked_images = images.view(-1, C, H, W)
                features = encoder_(stacked_images).squeeze(-1).squeeze(-1)
                features = features.view(batch_size, num_versions, -1).cpu().numpy()
    
            all_features.append(features)

    if high_qual:
        encoder_hq =  torch.nn.Sequential(*list(encoder_.children())[:-1])
        tile_dataset_hq = TilePreprocessing(tile_path, device=device, num_augmentations=0)
        data_loader_hq = DataLoader(tile_dataset_hq, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(),
                                    pin_memory=True)
        with torch.inference_mode():
            for _, _, images, _ in tqdm.tqdm(data_loader_hq, desc=f"High Quality Tiles: {patient_id} on {device}"):
                images = images.to(device, non_blocking=True)
                features = encoder_hq(images).squeeze(-1).squeeze(-1).cpu()
                high_qual_all.append(features.numpy())

    del encoder_, tile_dataset, data_loader
    gc.collect()
    torch.cuda.empty_cache()

    all_features = np.concatenate(all_features, axis=0).astype(np.float32)
    all_x = np.array(all_x, dtype=np.float32)
    all_y = np.array(all_y, dtype=np.float32)

    result_path = os.path.join(result_path, f"{patient_id}.h5")
    with h5py.File(result_path, "w") as hdf:
        hdf.create_dataset("tile_path", data=np.array(all_tile_paths, dtype="S"))
        hdf.create_dataset("x", data=all_x)
        hdf.create_dataset("y", data=all_y)

        hdf.create_dataset("features", data=all_features)
        if high_qual:
            hdf.create_dataset("high_quality", data=np.concatenate(high_qual_all, axis=0))
