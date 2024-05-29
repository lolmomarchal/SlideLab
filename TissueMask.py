import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class TissueMask:
    def __init__(self, slide, masks=['whole_slide', 'red_pen', 'green_pen', 'blue_pen', 'black_pen'], result_path=None,
                 threshold=0.7):
        self.slide = slide
        self.SCALE = self.slide.SCALE
        self.thumbnail = self.slide.thumbnail
        self.id = self.slide.id
        self.result_path = result_path
        self.detected_tissue = np.count_nonzero(np.array(self.otsu_mask_threshold()[1]))
        if masks == "default" or masks == "all":
            self.masks_list = ['whole_slide', 'red_pen', 'green_pen', 'blue_pen', 'black_pen']
        else:
            self.masks_list = masks

        self.mask, self.applied = self.save_original_with_mask()
        self.applied = Image.fromarray(self.applied)

    def metadata(self):
        """Retrieves the different metadata for the Tissue Mask
        Returns:
            Metadata Dictionary
        """
        return {"slide": self.slide, "masks_list used (self.mask_list)": self.masks_list, "thumbnail": self.thumbnail,
                "save path": self.result_path, "id": self.id, "scale": self.SCALE, "mask": self.mask,
                "mask applied to original slide": self.applied}

    def is_tissue(self, masked_region, threshold=0.7):
        tissue = np.count_nonzero(masked_region)
        total_elements = masked_region.size
        if tissue / total_elements >= threshold:
            return True
        return False

    # MASK METHODS

    def otsu_mask_threshold(self, kernel_size=None):
        """Apply Otsu thresholding to create a binary mask."""
        img_array = np.array(self.thumbnail)
        grayscale_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img_c = 255 - grayscale_img
        thres, thres_img = cv2.threshold(img_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if kernel_size is None:
            kernel_size = 2
        else:
            kernel_size = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        thres_img = cv2.morphologyEx(thres_img, cv2.MORPH_CLOSE, kernel)

        return thres, thres_img, img_c

    def remove_small_objects(self, binary_mask: np.array, min_size=None, default=True, avoid_overmask=True,
                             overmask_thresh=95,
                             kernel_size=1):
        """
        Remove small objects from a binary mask.
        """
        if default:
            min_size = binary_mask.shape[0] * binary_mask.shape[1] * 0.0001
        else:
            min_size = min_size

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=4)

        if kernel_size is not None:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_size:
                binary_mask[labels == label] = 0

        if avoid_overmask:
            mask_percentage = (np.sum(binary_mask) / binary_mask.size) * 100
            if mask_percentage >= overmask_thresh and min_size >= 1:
                new_min_size = min_size // 2
                binary_mask = self.remove_small_objects(binary_mask, new_min_size, False, avoid_overmask,
                                                        overmask_thresh)

        if kernel_size is not None:
            binary_mask = cv2.dilate(binary_mask.astype(np.uint8), kernel)
        return binary_mask

    def combined_mask(self):
        mask_methods = self.masks_list
        method_dict = {
            'red_pen': self.red_pen_filter,
            'blue_pen': self.blue_pen_filter,
            'green_pen': self.green_pen_filter,
            'black_pen': self.black_pen_filter,
            'whole_slide': self.get_whole_slide_mask
        }

        # Initialize combined mask with whole_slide if present, else with the first method
        first_method = method_dict[mask_methods[0]]
        combined_mask = first_method().astype(np.uint8)
        if len(mask_methods) == 1:
            return combined_mask
        for method in mask_methods:
            mask = method_dict[method]().astype(bool)
            combined_mask = np.logical_and(combined_mask, mask)
        return combined_mask

    # Pen Filters

    # Blue Pen Filter
    def blue_filter(self, img_array, red_thresh, green_thresh, blue_thresh):
        """Filter blue pen marks from the image based on given thresholds."""

        r = img_array[:, :, 0] < red_thresh
        g = img_array[:, :, 1] < green_thresh
        b = img_array[:, :, 2] > blue_thresh
        mask = r & g & b
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        return mask

    def blue_pen_filter(self):
        """Filter out blue pen marks and return a binary mask."""
        parameters = [
            {"red_thresh": 60, "green_thresh": 120, "blue_thresh": 190},
            {"red_thresh": 120, "green_thresh": 170, "blue_thresh": 200},
            {"red_thresh": 175, "green_thresh": 210, "blue_thresh": 230},
            {"red_thresh": 145, "green_thresh": 180, "blue_thresh": 210},
            {"red_thresh": 37, "green_thresh": 95, "blue_thresh": 160},
            {"red_thresh": 30, "green_thresh": 65, "blue_thresh": 130},
            {"red_thresh": 130, "green_thresh": 155, "blue_thresh": 180},
            {"red_thresh": 40, "green_thresh": 35, "blue_thresh": 85},
            {"red_thresh": 30, "green_thresh": 20, "blue_thresh": 65},
            {"red_thresh": 90, "green_thresh": 90, "blue_thresh": 140},
            {"red_thresh": 60, "green_thresh": 60, "blue_thresh": 120},
            {"red_thresh": 110, "green_thresh": 110, "blue_thresh": 175},
        ]

        pen_masks = [self.blue_filter(np.array(self.thumbnail), **param) for param in parameters]
        combined_mask = np.any(pen_masks, axis=0)
        return ~combined_mask.astype(bool)

    # red pen filter

    # for sanity purposes: if more than > 20% of tissue identified by primary otsu thresh, ignore
    def red_filter(self, img_array, red_thresh, green_thresh, blue_thresh) -> np.ndarray:
        """Filter red pen marks from the image based on given thresholds."""

        r = img_array[:, :, 0] > red_thresh
        g = img_array[:, :, 1] < green_thresh
        b = img_array[:, :, 2] < blue_thresh
        mask = r & g & b
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        return mask

    def red_pen_filter(self):
        """Filter out red pen marks and return a binary mask."""
        parameters = [
            {"red_thresh": 150, "green_thresh": 80, "blue_thresh": 90},
            {"red_thresh": 110, "green_thresh": 20, "blue_thresh": 30},
            {"red_thresh": 185, "green_thresh": 65, "blue_thresh": 105},
            {"red_thresh": 195, "green_thresh": 85, "blue_thresh": 125},
            {"red_thresh": 220, "green_thresh": 115, "blue_thresh": 145},
            {"red_thresh": 125, "green_thresh": 40, "blue_thresh": 70},
            {"red_thresh": 100, "green_thresh": 50, "blue_thresh": 65},
            {"red_thresh": 85, "green_thresh": 25, "blue_thresh": 45},
        ]
        pen_masks = [self.red_filter(np.array(self.thumbnail), **param) for param in parameters]
        combined_mask = np.any(pen_masks, axis=0)
        true_percentage = np.count_nonzero(combined_mask.astype(np.uint8)) / self.detected_tissue
        # Check if the percentage exceeds 20%
        if true_percentage > 0.2:
            return ~np.zeros_like(combined_mask, dtype=np.uint8)

        return ~combined_mask.astype(bool)

    # green pen filter
    def filter_green(self, img_array, red_upper_thresh, green_lower_thresh, blue_lower_thresh):

        r = img_array[:, :, 0] < red_upper_thresh
        g = img_array[:, :, 1] > green_lower_thresh
        b = img_array[:, :, 2] > blue_lower_thresh
        mask = r & g & b

        # Dilate the mask using a kernel of size 3x3 -> for pen going through tissue
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        return mask.astype(bool)

    def green_pen_filter(self):
        """Filter out green pen marks and return a binary mask."""
        thresholds = [
            {"red_upper_thresh": 150, "green_lower_thresh": 160, "blue_lower_thresh": 140},
            {"red_upper_thresh": 70, "green_lower_thresh": 110, "blue_lower_thresh": 110},
            {"red_upper_thresh": 45, "green_lower_thresh": 115, "blue_lower_thresh": 100},
            {"red_upper_thresh": 30, "green_lower_thresh": 75, "blue_lower_thresh": 60},
            {"red_upper_thresh": 195, "green_lower_thresh": 220, "blue_lower_thresh": 210},
            {"red_upper_thresh": 225, "green_lower_thresh": 230, "blue_lower_thresh": 225},
            {"red_upper_thresh": 170, "green_lower_thresh": 210, "blue_lower_thresh": 200},
            {"red_upper_thresh": 20, "green_lower_thresh": 30, "blue_lower_thresh": 20},
            {"red_upper_thresh": 50, "green_lower_thresh": 60, "blue_lower_thresh": 40},
            {"red_upper_thresh": 30, "green_lower_thresh": 50, "blue_lower_thresh": 35},
            {"red_upper_thresh": 65, "green_lower_thresh": 70, "blue_lower_thresh": 60},
            {"red_upper_thresh": 100, "green_lower_thresh": 110, "blue_lower_thresh": 105},
            {"red_upper_thresh": 165, "green_lower_thresh": 180, "blue_lower_thresh": 180},
            {"red_upper_thresh": 140, "green_lower_thresh": 140, "blue_lower_thresh": 150},
            {"red_upper_thresh": 185, "green_lower_thresh": 195, "blue_lower_thresh": 195},
        ]

        masks = [self.filter_green(np.array(self.thumbnail), **params) for params in thresholds]
        combined_mask = np.any(masks, axis=0)
        return ~combined_mask.astype(bool)

    # Black Pen Filter

    def black_filter(self, img_array, red_thresh, green_thresh, blue_thresh) -> np.ndarray:
        """Filter black pen marks from the image based on given thresholds."""
        r = img_array[:, :, 0] < red_thresh
        g = img_array[:, :, 1] < green_thresh
        b = img_array[:, :, 2] < blue_thresh
        mask = r & g & b
        # Dilate the mask using a kernel of size 3x3 -> for pen going through tissue
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        return mask

    def black_pen_filter(self):
        """Filter out black pen marks and return a binary mask."""
        parameters = [
            {"red_thresh": 50, "green_thresh": 50, "blue_thresh": 50},
            {"red_thresh": 30, "green_thresh": 30, "blue_thresh": 30},
            {"red_thresh": 20, "green_thresh": 20, "blue_thresh": 20},
            {"red_thresh": 10, "green_thresh": 10, "blue_thresh": 10},
        ]
        pen_masks = [self.black_filter(np.array(self.thumbnail), **param) for param in parameters]
        combined_mask = np.any(pen_masks, axis=0)
        return ~combined_mask.astype(bool)

    def get_whole_slide_mask(self):
        _, mask, image_c = self.otsu_mask_threshold()

        # Save original image
        plt.imshow(self.thumbnail)
        plt.axis("off")
        if self.result_path is not None:
            plt.savefig(os.path.join(self.result_path, "original_slide"), bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save masked slide
        plt.imshow(mask)
        plt.axis("off")
        if self.result_path is not None:
            plt.savefig(os.path.join(self.result_path, "masked_slide"), bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save masked slide without small fragments
        small_objs = self.remove_small_objects(mask, default=True)
        binary_mask = np.array((small_objs > 1).astype(np.uint8))
        wsi_thumb_masked = np.copy(self.thumbnail)
        wsi_thumb_masked[binary_mask == 0] = 255
        plt.imshow(binary_mask)
        plt.axis("off")
        if self.result_path is not None:
            plt.savefig(os.path.join(self.result_path, "masked_slide_nofragments"), bbox_inches='tight', pad_inches=0)
        plt.close()
        return small_objs

    def save_original_with_mask(self):
        """Save the original image with the combined mask applied to it."""
        original_image = np.array(self.thumbnail)
        combined_mask = self.combined_mask().astype(np.uint8)
        applied_mask = np.copy(original_image)
        applied_mask[combined_mask == 0] = 0  # Apply the combined mask to the original image

        # Save the masked image
        plt.imshow(applied_mask)
        plt.axis("off")
        plt.axis("off")
        if self.result_path is not None:
            plt.savefig(os.path.join(self.result_path, "original_with_mask"), bbox_inches='tight', pad_inches=0)
        plt.close()
        return combined_mask, applied_mask

    def get_region_mask(self, original_size, size):
        mask_region_location = (original_size[0] // self.SCALE, original_size[1] // self.SCALE)
        mask_region_size = (size[0] // self.SCALE, size[1] // self.SCALE)
        return self.mask[mask_region_location[1]:mask_region_location[1] + mask_region_size[1],
               mask_region_location[0]:mask_region_location[0] + mask_region_size[0]]
