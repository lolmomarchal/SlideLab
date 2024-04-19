import numpy as np
import skimage.morphology as sk_morphology
import gc
import os
import cv2
import math
import pandas as pd
from PIL import Image
import tifffile as tiff
import matplotlib.pyplot as plt
SCALE = 32


class TissueMask():
    def __init__(self, slide, SCALE = 32):
        self.thumbnail = slide.get_thumbnail((slide.dimensions[0]/SCALE, slide.dimensions[1]/SCALE))
        self.SCALE = 32

#returns array of thumbnail
def get_thumbnail(slide):
    return np.array(slide.get_thumbnail((slide.dimensions[0]/SCALE, slide.dimensions[1]/SCALE)))

def get_region_mask(mask, original_size, size):
    mask_region_location = (original_size[0] // SCALE, original_size[1] // SCALE)
    mask_region_size = (size[0]//SCALE, size[1]//SCALE)
    return mask[mask_region_location[1]:mask_region_location[1] + mask_region_size[1],
           mask_region_location[0]:mask_region_location[0] + mask_region_size[0]]
def is_tissue(masked_region, threshold = 0.8):
    tissue = np.count_nonzero(masked_region)
    total_elements = masked_region.size
    if tissue/total_elements >threshold:
        return True
    return False

def threshold(img, method="otsu", kernel_size=None):
    """
    Perform thresholding on an image.

    Params:
        img: Input image
        method: Thresholding method ('otsu' or 'triangle')
        kernel_size: Size of the kernel for morphological operations
    Returns:
        thres: Threshold value
        thres_img: Thresholded image
        img_c: Grayscale image
    """
    # first convert to gray scale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_c = 255 - grayscale_img

    thres, thres_img = 0, img_c.copy()

    if method == 'otsu':
        thres, thres_img = cv2.threshold(img_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'triangle':
        thres, thres_img = cv2.threshold(img_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)


    plt.imshow(thres_img)
    # Apply morphological operations with the provided kernel size
    if kernel_size is None:
        kernel_size = 2
    else:
        kernel_size = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    thres_img = cv2.morphologyEx(thres_img, cv2.MORPH_CLOSE, kernel)

    return thres, thres_img, img_c

def remove_small_objects(binary_mask, min_size=None, default=True, avoid_overmask=True, overmask_thresh=95, kernel_size= 1):
    """
    Recursive method to remove small fragments
    Params:
    binary mask [np.array] -> mask obtained through otsu method
    min_size [int] -> min size of pixels for fragment to be considered a component
            - Default is  > than 0.5% of whole image
    avoid_overmask [bool] -> Flag to avoid overmasking

    overmask_thresh [float] -> Threshold percentage to consider overmasking
    kernel_size [int] -> Size of the kernel for morphological dilation
    """
    if default:
        min_size = binary_mask.shape[0] * binary_mask.shape[1] * 0.0001
    else:
        min_size = min_size

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=4)

    # Create a structuring element for morphological dilation if kernel_size is provided
    if kernel_size is not None:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for label in range(1, num_labels):
        # Check the area of the connected component
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_size:
            binary_mask[labels == label] = 0

    # Avoid overmasking if enabled
    if avoid_overmask:
        mask_percentage = (np.sum(binary_mask) / binary_mask.size) * 100
        if mask_percentage >= overmask_thresh and min_size >= 1:
            new_min_size = min_size // 2
            binary_mask = remove_small_objects(binary_mask, new_min_size, False, avoid_overmask, overmask_thresh)

    # Morphological dilation to preserve pixels near components
    if kernel_size is not None:
        binary_mask = cv2.dilate(binary_mask.astype(np.uint8), kernel)
    return binary_mask
def get_whole_slide_mask(wsi_thumb, results):
    _, mask, image_c = threshold(wsi_thumb)
    # save original image
    plt.imshow(wsi_thumb)
    plt.axis("off")
    plt.savefig(os.path.join(results, "original_slide"), bbox_inches='tight', pad_inches=0)
    plt.close()

    # save with
    plt.imshow(mask)
    plt.axis("off")
    plt.savefig(os.path.join(results, "masked_slide"), bbox_inches='tight', pad_inches=0)
    plt.close()


    # save small fragment removal

    small_objs = remove_small_objects(mask, 0, default= True)
    # r = remove_small_objects(~small_objs, 0, default= True)
    # r = 255-r

    binary_mask = np.array((small_objs> 1).astype(np.uint8))
    wsi_thumb_masked = np.copy(wsi_thumb)
    wsi_thumb_masked[binary_mask == 0] = 255
    plt.imshow(binary_mask)
    plt.axis("off")
    plt.savefig(os.path.join(results, "masked_slide_nofragments"), bbox_inches='tight', pad_inches=0)
    plt.close()


    # Apply actual mask to image
    overlay = wsi_thumb.copy()
    overlay[small_objs == 0] = 0
    plt.imshow(overlay)
    plt.axis("off")
    plt.savefig(os.path.join(results, "masked_overlay"), bbox_inches='tight', pad_inches=0)
    plt.close()

    return small_objs
    del wsi_thumb
    del binary_mask
    del wsi_thumb_masked
    gc.collect()



# OLD IGNORE SECTION
def is_background(mask, original_image, gray_range=(100, 150), white_range=(200, 255), threshold=0.2):
    # white_condition = np.any(original_image> 200, axis = -1)
    # white_condition = np.sum(white_condition)/ np.prod(original_image.shape[:-1])

    background_count = np.sum(mask == 0)
    total_elements = mask.size
    if background_count/total_elements >= threshold:
        return True
    return False
def tile_mask(tile):
    _, mask, image_c = threshold(tile)
    small_objs = remove_small_objects(mask)
    #removing small holes
    r = remove_small_objects(~mask,512*512*0.05, default = False)
    r = 255-r
    return r

def clean_up_tile(tile):
    _, mask, image_c = threshold(tile)
    small_objects = remove_small_objects(mask, 512*512*0.005, default= False)
    r = remove_small_objects(~small_objects, 512*512*0.0005, default= False, kernel_size=2)
    r = 255-r
    tile_masked = np.copy(tile)
    tile_masked[r == 0] = 255
    return tile_masked
