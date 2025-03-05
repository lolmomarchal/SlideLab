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
import torchvision.transforms as transforms
import gc
import math

torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True


def encoder(encoder_type="resnet50", device="cpu"):
    if encoder_type == "resnet50":
        encoder_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        encoder_model = torch.nn.Sequential(*list(encoder_model.children())[:-1])
        encoder_model.eval().to(device)
        if device == "cpu":
            encoder_model = torch.quantization.quantize_dynamic(encoder_model, {torch.nn.Linear}, dtype=torch.qint8)
    return encoder_model


class TilePreprocessing(Dataset):
    def __init__(self, df_file, device="cpu", num_augmentations=0):
        df = pd.read_csv(df_file)
        self.data = df[["x", "y", "tile_path"]].values
        self.device = device
        self.num_augmentations = num_augmentations
        self.normalize = transforms.Normalize(mean=[0.485,0.406,0.406],std=[0.229,0.224,0.225])
        self.augmentations = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.5, contrast=[0.2, 1.8], saturation=0, hue=0),
            # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))
            self.normalize
        ])
        self.no_augmentations = transforms.Compose([ self.normalize])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y, tile_path = self.data[idx]
        image = read_image(tile_path).float() / 255.0

        if self.num_augmentations == 0:
            return x, y, image, tile_path

        augmented_images = [self.augmentations(image.clone()) for _ in range(self.num_augmentations)]
        augmented_images.insert(0, image)
        stacked_images = torch.stack(augmented_images, dim=0)

        return x, y, stacked_images, tile_path


# Warning: Reduce batch_size when using augmentations
def encode_tiles(patient_id, tile_path, result_path, device="cpu", batch_size=512, encoder_model="resnet50",
                 high_qual=False, number_of_augmentation=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_ = encoder(encoder_type=encoder_model, device=device)
    if number_of_augmentation != 0:
        batch_size = math.ceil(batch_size/number_of_augmentation)
    tile_dataset = TilePreprocessing(tile_path, device=device, num_augmentations=number_of_augmentation)
    all_features, all_x, all_y, all_tile_paths = [], [], [], []
    high_qual_all = []

    data_loader = DataLoader(tile_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(),
                             pin_memory=True)

    with torch.inference_mode():
        for x, y, images, tile_paths in tqdm.tqdm(data_loader, desc=f"Encoding Tiles: {patient_id} on {device}"):
            all_x.extend(x.numpy())
            all_y.extend(y.numpy())
            all_tile_paths.extend(tile_paths)
            images = images.to(device, non_blocking=True)
    
            # Check if there are augmentations
            if images.ndimension() == 4:  # Non-augmented case: [batch_size, C, H, W]
                batch_size, C, H, W = images.shape
                features = encoder_(images).squeeze(-1).squeeze(-1)
                features = features.cpu().numpy()
    
            else:  # Augmented case: [batch_size, num_versions, C, H, W]
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

    df = pd.read_csv(tile_path)
    mag = df["desired_magnification"].to_numpy(dtype=np.float32)
    size = df["desired_size"].to_numpy(dtype=np.float32)

    all_features = np.concatenate(all_features, axis=0).astype(np.float32)
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
        if high_qual:
            hdf.create_dataset("high_quality", data=np.concatenate(high_qual_all, axis=0))
