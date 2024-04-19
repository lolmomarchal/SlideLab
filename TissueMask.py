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

class TissueMask:
    def __init__(self, slide, result_path):
        self.slide = slide
        self.SCALE = self.slide.SCALE
        self.thumbnail = self.slide.thumbnail
        self.id = self.slide.id
        self.result_path = result_path
    def is_tissue(self, masked_region, threshold = 0.8):
        tissue = np.count_nonzero(masked_region)
        total_elements = masked_region.size
        if tissue/total_elements >=threshold:
            return True
        return False

    #MASKS

    # Otsu threshold
    def outsu_mask_threshold(self,img, kernel_size = None):
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_c = 255 - grayscale_img
        thres, thres_img = 0, img_c.copy()
        thres, thres_img = cv2.threshold(img_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Apply morphological operations with the provided kernel size
        if kernel_size is None:
            kernel_size = 2
        else:
            kernel_size = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        thres_img = cv2.morphologyEx(thres_img, cv2.MORPH_CLOSE, kernel)

        return thres, thres_img, img_c
    # removing small objects
    def remove_small_objects(self, binary_mask,  min_size=None, default=True, avoid_overmask=True, overmask_thresh=95, kernel_size= 1):
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
                binary_mask = self.remove_small_objects(binary_mask, new_min_size, False, avoid_overmask, overmask_thresh)

        # Morphological dilation to preserve pixels near components
        if kernel_size is not None:
            binary_mask = cv2.dilate(binary_mask.astype(np.uint8), kernel)
        return binary_mask


    # RED PEN
    def red_pen_filter(self):
    # BLUE PEN
    def blue_pen_filer(self):
    # GREEN PEN
    def green_pen_filer(self):
    # BLACK PEN
    def black_pen_filer(self):

    # UNIFY MASKS
    def get_whole_slide_mask(self):
        _, mask, image_c = self.outsu_mask_threshold(self.thumbnail)
        # save original image
        plt.imshow(self.thumbnail)
        plt.axis("off")
        plt.savefig(os.path.join(self.result_path, "original_slide"), bbox_inches='tight', pad_inches=0)
        plt.close()

        # save with
        plt.imshow(mask)
        plt.axis("off")
        plt.savefig(os.path.join(self.result_path, "masked_slide"), bbox_inches='tight', pad_inches=0)
        plt.close()


        # save small fragment removal

        small_objs = self.remove_small_objects(mask, 0, default= True)
        # r = remove_small_objects(~small_objs, 0, default= True)
        # r = 255-r

        binary_mask = np.array((small_objs> 1).astype(np.uint8))
        wsi_thumb_masked = np.copy(self.thumbnail)
        wsi_thumb_masked[binary_mask == 0] = 255
        plt.imshow(binary_mask)
        plt.axis("off")
        plt.savefig(os.path.join(self.result_path, "masked_slide_nofragments"), bbox_inches='tight', pad_inches=0)
        plt.close()


        # Apply actual mask to image
        overlay = self.thumbnail.copy()
        overlay[small_objs == 0] = 0
        plt.imshow(overlay)
        plt.axis("off")
        plt.savefig(os.path.join(self.result_path, "masked_overlay"), bbox_inches='tight', pad_inches=0)
        plt.close()
        return small_objs







