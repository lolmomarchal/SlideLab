# Tile Dataset
import torch
from torch.utils.data import Dataset
import numpy as np
import openslide
import cv2

from TissueMask import is_tissue, get_region_mask, TissueMask
from TileNormalization import normalizeStaining
from TileQualityFilters import LaplaceFilter

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
    # valid coordinates according to mask
    valid_coords = [coord for coord in coordinates if
                    is_tissue(get_region_mask(mask, scale, coord, (size, size)), threshold=threshold)]
    return total_cords, len(valid_coords), valid_coords, coordinates
 # ================= MAIN + COLLATE

def collate_fn(batch):
    return [b for b in batch if b[0] is not None]

class TileDataset(Dataset):
    def __init__(self, slide, coordinates=None, mask=None, tissue_threshold=0.8, size=256,
                 magnification=20, overlap=1, adjusted_size=None, normalizer = None, remove_blurry = True):
        self.tissue_threshold = tissue_threshold
        self.size = size
        self.magnification = magnification
        self.normalizer = normalizer
        self.remove_blurry = remove_blurry

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

    def __getitem__(self, i):
        coord = self.coordinates[i]
        tile = np.array(self.slide.read_region((coord[0], coord[1]), 0,
                                               (self.adjusted_size, self.adjusted_size)).convert('RGB'))
        if self.adjusted_size != self.size:
            tile = cv2.resize(tile, (self.size, self.size), interpolation=cv2.INTER_AREA)

        if self.normalizer is not None:
            tile = normalizeStaining(tile)
            if tile is None:
                return None, None, None
        if self.remove_blurry:
            blur, var = LaplaceFilter(tile)
            if blur:
                return None, None, torch.tensor(var)
            return torch.from_numpy(tile), torch.tensor(coord), torch.tensor(var)

            
        return torch.from_numpy(tile), torch.tensor(coord), torch.tensor(0)
