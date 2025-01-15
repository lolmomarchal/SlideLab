from TissueMask import TissueMask
import numpy as np
import openslide
from TileNormalization import normalizeStaining
from PIL import Image
from TissueMask import is_tissue, get_region_mask, TissueMask

def best_size(desired_mag, natural_mag, desired_size) -> int:
    new_size = natural_mag / desired_mag
    return int(desired_size * new_size)


def get_valid_coordinates(width, height, overlap, mask, size, scale, threshold):
    # example overlap of 2 and size of 256 = 128 stride
    if overlap != 1:
        stride = size / overlap
    else:
        stride = size
    x_coords = np.arange(0, width - size + stride, stride)
    y_coords = np.arange(0, height - size + stride, stride)
    coordinates = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)
    total_cords = coordinates.shape[0]
    # valid coordinates according to mask
    valid_coords = [coord for coord in coordinates if
                    is_tissue(get_region_mask(mask, scale, coord, (size, size)), threshold=threshold)]
    return total_cords, len(valid_coords), valid_coords


class TileIterator:
    def __init__(self, slide, coordinates=None, mask=None, tissue_threshold=0.8, normalizer=None, size=256,
                 magnification=20, overlap=1, adjusted_size = None):
        self.normalizer = normalizer
        self.tissue_threshold = tissue_threshold
        self.size = size
        self.magnification = magnification
        self.index = 0
        try:
            if type(slide) == str:
                self.slide = openslide.OpenSlide(slide)
            else:
                self.slide = slide
        except exception as e:
            print(f"Error opening Slide: {e}")
        # mask
        if mask is None:
            self.mask = TissueMask(self.slide, threshold=tissue_threshold)
        else:
            self.mask = mask
        if adjusted_size is None:
            self.adjusted_size = best_size(magnification, self.mask.magnification, size)
        else:
            self.adjusted_size = adjusted_size
        self.overlap = overlap
        if coordinates is not None:
            self.coordinates = coordinates
        else:
            # valid coords
            w, h = self.slide.dimensions
            all_coords, valid_coordinates, self.coordinates = get_valid_coordinates(w, h, overlap,
                                                                                    self.mask.mask,
                                                                                    self.adjusted_size,
                                                                                    self.mask.SCALE,
                                                                                    threshold=tissue_threshold)

    def __len__(self):
        return len(self.coordinates)

    def __iter__(self):
        self.index = 0
        return self

    def __getitem__(self, i):
        coord = self.coordinates[i]
        tile = self.slide.read_region((coord[0], coord[1]), 0, (self.adjusted_size, self.adjusted_size)).convert(
            'RGB').resize(
            (self.size, self.size), Image.BILINEAR)
        if self.normalizer is not None:
            tile = Image.fromarray(normalizeStaining(np.array(tile)))
        return tile, coord


    def __next__(self):
        if self.index >= len(self.coordinates):
            raise StopIteration
        coord = self.coordinates[self.index]
        tile = self.slide.read_region((coord[0], coord[1]), 0, (self.adjusted_size, self.adjusted_size)).convert(
            'RGB').resize(
            (self.size, self.size), Image.BILINEAR)
        self.index += 1

        # if normalize
        if self.normalizer is not None:
            tile = Image.fromarray(normalizeStaining(np.array(tile)))
        return tile, coord

    def reset(self):
        self.index = 0
