import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def reconstruct_slide(image,included_coords,all_coords, scale, adjusted_size, line_thickness = 1, save_path = None):
    step_size = adjusted_size//scale
    mask_ = image.copy()
    for item in all_coords:
        x, y = item
        x_scaled = x // scale
        y_scaled = y // scale
        if any((included_coords == item).all(1)):
            # draw y borders
            mask_[y_scaled-line_thickness:y_scaled, x_scaled:x_scaled+step_size] = 255
            mask_[y_scaled+step_size-line_thickness:y_scaled+step_size, x_scaled:x_scaled+step_size] = 255

            # draw x borders
            mask_[y_scaled:y_scaled+step_size, x_scaled-line_thickness:x_scaled] = 255
            mask_[y_scaled:y_scaled+step_size, x_scaled+step_size-line_thickness:x_scaled+step_size] = 255
        else:
            mask_[y_scaled:y_scaled+step_size, x_scaled:x_scaled+step_size] = 0
    if save_path is not None:
        Image.fromarray(mask_).save(save_path)

    return mask_


