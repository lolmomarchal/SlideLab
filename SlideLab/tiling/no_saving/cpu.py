import os
import numpy as np
import torch
from torch.utils.data import Dataset
from functools import partial
import multiprocessing
from typing import Dict, Tuple, List, Optional
import cv2
import openslide
import time
import pandas as pd
from PIL import Image

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

class CPUTileDataset(Dataset):
    def __init__(self, slide, coordinates, adjusted_size, desired_size,*,
                 pipeline_steps=None, transforms=None):
        self.slide = slide
        self.coordinates = coordinates
        self.pipeline_steps = pipeline_steps or []
        self.transforms = transforms
        self.adjusted_size = adjusted_size
        self.desired_size = desired_size
        self.vars_dict = {}

    def __len__(self):
        return len(self.coordinates)

    def _read_region_np(self, coord):
        region = np.array(self.slide.read_region(coord, 0, self.adjusted_size).convert('RGB'))
        return cv2.resize(region, (self.desired_size, self.desired_size),
                          interpolation=cv2.INTER_LINEAR)

    def _preprocess_tile(self, img_np, coord):
        for step in self.pipeline_steps:
            img_np = step(img_np, self.vars_dict, coord)
            if img_np is None:
                return None
        return img_np

    def __getitem__(self, idx):
        coord = self.coordinates[idx]
        img_np = self._read_region_np(coord)
        processed_np = self._preprocess_tile(img_np, coord)
        if processed_np is None:
            return None, torch.tensor(coord)

        if self.transforms:
            img_tensor = torch.from_numpy(processed_np).permute(2, 0, 1)  # HWC → CHW
            img_tensor = img_tensor.to(torch.uint8)
            img_tensor = self.transforms(img_tensor)
        else:
            img_tensor = torch.from_numpy(processed_np)


        return img_tensor, torch.tensor(coord)

class GPUTileDataset(Dataset):
    """Dataset for GPU processing without saving tiles"""
    def __init__(self,slide,coordinates,mask,adjusted_size,desired_size,pipeline_steps,transforms = None, device= "cpu"
    ):
        self.slide = slide
        self.coordinates = coordinates
        self.mask = mask
        self.adjusted_size = adjusted_size
        self.desired_size = desired_size
        self.adjusted_size = adjusted_size
        self.pipeline_steps = pipeline_steps or []
        self.device = device
        self.batch_size = batch_size
        self.vars_dict = {}

    def __len__(self) -> int:
        return len(self.coordinates)

    def _read_region(self,coord):
        return self.slide.read_region(
            location=coord,
            level=0,
            size=self.adjusted_size
        ).convert('RGB').resize((self.desired_size, self.desired_size))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        coord = self.coordinates[idx]
        img = self._read_region(coord)

        # Convert to tensor and permute dimensions to CxHxW
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
        return img_tensor, coord

    def collate_fn(self, batch: List[Tuple[torch.Tensor, Tuple[int, int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom collate function to process batches on GPU"""
        # Filter out None values (from blurry tiles)
        batch = [x for x in batch if x[0] is not None]
        if not batch:
            return torch.empty(0), torch.empty(0)

        imgs, coords = zip(*batch)
        imgs = torch.stack(imgs).to(self.device)
        coords = torch.tensor(coords).to(self.device)

        # Apply processing pipeline
        for step in self.pipeline_steps:
            imgs, coords = step(imgs, self.vars_dict, coords)
            if imgs.numel() == 0:
                return torch.empty(0), torch.empty(0)

        return imgs, coords
