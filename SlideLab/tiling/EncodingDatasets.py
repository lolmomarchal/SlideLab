import os
import tqdm
import torch
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import numpy as np
import openslide
import cv2
# internal
from TissueMask import is_tissue, get_region_mask, TissueMask
from normalization.TileNormalization import normalizeStaining
from TileQualityFilters import LaplaceFilter


#=== TODO: ADD MPP SUPPORT FOR NO SAVING
# ====== HELPER

def best_size(desired_mag, natural_mag, desired_size) -> int:
    new_size = natural_mag / desired_mag
    return int(desired_size * new_size)
def get_valid_coordinates(width, height, overlap, mask, size, scale, threshold):
    # example overlap of 2 and size of 256 = 128 stride
    if overlap > 1:
        stride = size // overlap
    else:
        stride = size
    x_coords = np.arange(0, width - size + stride, stride)
    y_coords = np.arange(0, height - size + stride, stride)
    coordinates = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)
    total_cords = coordinates.shape[0]
    valid_coords = [coord for coord in coordinates if
                    is_tissue(get_region_mask(mask, scale, coord, (size, size)), threshold=threshold)]
    return total_cords, len(valid_coords), valid_coords, coordinates
# ======= version 1: Using the h5 file
import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TileEncoding_h5(Dataset):
    def __init__(self, h5_file, device="cpu", num_augmentations=0,
                 model_transforms=transforms.Normalize(mean=[0.485,0.406,0.406],
                                                       std=[0.229,0.224,0.225])):
        self.h5_file = h5_file  # Store file path instead of loading data
        self.device = device
        self.num_augmentations = num_augmentations
        self.normalize = model_transforms

        # Initialize transforms
        self.augmentations = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.5, contrast=[0.2, 1.8], saturation=0, hue=0),
            self.normalize
        ])

        self.no_augmentations = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

        # Open file just to get length (then close immediately)
        with h5py.File(self.h5_file, "r") as f:
            self._length = len(f["coords"])

    def __len__(self):
        return self._length

    def __getitem__(self, item):
        with h5py.File(self.h5_file, "r") as f:
            try:
                x, y = f["coords"][item]
                image = f["tiles"][item]

                if self.num_augmentations == 0:
                    return x, y, self.no_augmentations(image), item

                augmented_images = [self.augmentations(image.copy()) for _ in range(self.num_augmentations)]
                augmented_images.insert(0, self.no_augmentations(image))
                stacked_images = torch.stack(augmented_images, dim=0)
                return torch.tensor([x,y]), stacked_images, item

            except Exception as e:
                print(f"Error loading item {item}: {str(e)}")
                raise
# ======= version 2: Using the csv/png files

class TilePreprocessing_png(Dataset):
    def __init__(self, df_file, device="cpu", num_augmentations=0, model_transforms =transforms.Normalize(mean=[0.485,0.406,0.406],std=[0.229,0.224,0.225])):
        # reading csv
        df = pd.read_csv(df_file)
        self.data = df[["x", "y", "tile_path"]].values
        self.device = device
        self.num_augmentations = num_augmentations

        # setting up normalizer
        self.normalize = model_transforms # will default to the resnet50 unless otherwise said

        self.augmentations = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.5, contrast=[0.2, 1.8], saturation=0, hue=0),
            self.normalize
        ])
        self.no_augmentations = transforms.Compose([transforms.ToTensor(), self.normalize])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            x, y, tile_path = self.data[idx]
            image = Image.open(tile_path)
            if self.num_augmentations == 0:
                return torch.tensor([x,y]), self.no_augmentations(image), tile_path

            augmented_images = [self.augmentations(image.copy()) for _ in range(self.num_augmentations)]
            augmented_images.insert(0, self.no_augmentations(image))
            stacked_images = torch.stack(augmented_images, dim=0)

            return torch.tensor([x,y]), stacked_images, tile_path
        except:
            del self.data[idx]
            return self.__getitem__(idx)
# ======= version 3: No previously saving, directly to just encoding

class TilePreprocessing_nosaving(Dataset):
    def __init__(self, slide, coordinates=None, mask=None, tissue_threshold=0.8, size=256,
                 magnification=20, overlap=1, adjusted_size=None, normalizer=None,
                 remove_blurry=False,blur_threshold = None,  num_augmentations=0, model_transforms=None):

        self.tissue_threshold = tissue_threshold
        self.size = size
        self.magnification = magnification
        self.normalizer = normalizer
        self.remove_blurry = remove_blurry
        self.num_augmentations = num_augmentations
        self.blur_threshold = blur_threshold

        self.model_transforms = model_transforms or transforms.Normalize(
            mean=[0.485, 0.406, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.augmentations = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.5, contrast=[0.2, 1.8], saturation=0, hue=0),
            self.model_transforms
        ])

        self.no_augmentations = transforms.Compose([
            transforms.ToTensor(),
            self.model_transforms
        ])

        self.slide = openslide.OpenSlide(slide) if isinstance(slide, str) else slide
        self.mask = mask if mask is not None else TissueMask(self.slide, threshold=tissue_threshold)
        self.adjusted_size = adjusted_size or best_size(magnification, self.mask.magnification, size)
        self.overlap = overlap

        if coordinates is not None:
            self.coordinates = coordinates
        else:
            w, h = self.slide.dimensions
            _, _, self.coordinates, _ = get_valid_coordinates(
                w, h, overlap, self.mask.mask, self.adjusted_size, self.mask.SCALE, threshold=tissue_threshold
            )

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, item):
        coord = self.coordinates[item]
        x, y = coord[0], coord[1]

        try:
            tile = np.array(self.slide.read_region((x, y), 0, (self.adjusted_size, self.adjusted_size)).convert('RGB'))
            if self.adjusted_size != self.size:
                tile = cv2.resize(tile, (self.size, self.size), interpolation=cv2.INTER_AREA)

            if self.normalizer is not None:
                tile = normalizeStaining(tile)
                if tile is None:
                    print("Error during normalization!")
                    # Return zero tensors instead
                    dummy_tile = torch.zeros((3, self.size, self.size), dtype=torch.float32)
                    return torch.tensor([-1, -1], dtype=torch.float32), dummy_tile, torch.tensor(0.0)

            if self.remove_blurry:
                blur, var = LaplaceFilter(tile)
                if blur:
                    print("Tile was blurry!")
                    # Return zero tensors with flag values
                    dummy_tile = torch.zeros((3, self.size, self.size), dtype=torch.float32)
                    return torch.tensor([-1, -1], dtype=torch.float32), dummy_tile, torch.tensor(0.0)
            else:
                var = torch.tensor(0.0)  # Convert to tensor

            if self.num_augmentations == 0:
                tile = self.no_augmentations(tile)
                return torch.tensor([x, y], dtype=torch.float32), tile, torch.tensor(var)
            else:
                augmented_tiles = [self.augmentations(tile.copy()) for _ in range(self.num_augmentations)]
                augmented_tiles.insert(0, self.no_augmentations(tile))
                stacked_tiles = torch.stack(augmented_tiles, dim=0)
                return torch.tensor([x, y], dtype=torch.float32), stacked_tiles, torch.tensor(var)
        except:
            dummy_tile = torch.zeros((3, self.size, self.size), dtype=torch.float32)

            return torch.tensor([-1, -1], dtype=torch.float32), dummy_tile, torch.tensor(0.0)



