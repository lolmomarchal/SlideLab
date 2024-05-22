import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


class TissueMask:
    def __init__(self, slide, result_path):
        self.slide = slide
        self.SCALE = self.slide.SCALE
        self.thumbnail = self.slide.thumbnail
        self.id = self.slide.id
        self.result_path = result_path
        self.mask = self.save_original_with_mask()

    def is_tissue(self, masked_region, threshold=0.7):
        tissue = np.count_nonzero(masked_region)
        total_elements = masked_region.size
        if tissue / total_elements >= threshold:
            return True
        return False

    # MASK METHODS

    def otsu_mask_threshold(self, img, kernel_size=None):
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

        pen_masks = [self.blue_filter(**param) for param in parameters]
        combined_mask = np.any(pen_masks, axis=0)
        return combined_mask

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
        pen_masks = [self.red_filter(self.thumbnail, **param) for param in parameters]
        combined_mask = np.any(pen_masks, axis=0)
        return combined_mask

    def green_pen_filter(self):
        """Filter out green pen marks and return a binary mask."""
        parameters = [
            {"red_thresh": 150, "green_thresh": 160, "blue_thresh": 140},
            {"red_thresh": 70, "green_thresh": 110, "blue_thresh": 110},
            {"red_thresh": 45, "green_thresh": 115, "blue_thresh": 100},
            {"red_thresh": 30, "green_thresh": 75, "blue_thresh": 60},
            {"red_thresh": 195, "green_thresh": 220, "blue_thresh": 210},
            {"red_thresh": 225, "green_thresh": 230, "blue_thresh": 225},
            {"red_thresh": 170, "green_thresh": 210, "blue_thresh": 200},
            {"red_thresh": 20, "green_thresh": 30, "blue_thresh": 20},
            {"red_thresh": 50, "green_thresh": 60, "blue_thresh": 40},
            {"red_thresh": 30, "green_thresh": 50, "blue_thresh": 35},
            {"red_thresh": 65, "green_thresh": 70, "blue_thresh": 60},
            {"red_thresh": 100, "green_thresh": 110, "blue_thresh": 105},
            {"red_thresh": 165, "green_thresh": 180, "blue_thresh": 180},
            {"red_thresh": 140, "green_thresh": 140, "blue_thresh": 150},
            {"red_thresh": 185, "green_thresh": 195, "blue_thresh": 195},
        ]

        pen_masks = [self.green_filter(**param) for param in parameters]
        combined_mask = np.any(pen_masks, axis=0)
        return combined_mask

    def combined_mask(self):
        red_mask = self.red_pen_filter().astype(np.uint8)
        blue_mask = self.blue_pen_filter().astype(np.uint8)
        green_mask = self.green_pen_filter().astype(np.uint8)
        black_mask = self.black_pen_filter().astype(np.uint8)
        whole_mask = self.get_whole_slide_mask()

        combined_mask = np.bitwise_or(whole_mask, np.bitwise_or(np.bitwise_or(red_mask, blue_mask),
                                                                np.bitwise_or(green_mask, black_mask)))
        return combined_mask

    # Helper Methods

    def blue_filter(self, red_thresh, green_thresh, blue_thresh):
        """Filter blue pen marks from the image based on given thresholds."""
        img_array = np.array(self.thumbnail)
        mask = (img_array[:, :, 0] < red_thresh) & (img_array[:, :, 1] < green_thresh) & (
                img_array[:, :, 2] > blue_thresh)
        return mask

    def red_filter(self, img, red_thresh, green_thresh, blue_thresh) -> np.ndarray:
        """Filter red pen marks from the image based on given thresholds."""
        img_array = np.array(self.thumbnail)
        mask = (img_array[:, :, 0] > red_thresh) & (img_array[:, :, 1] < green_thresh) & (
                img_array[:, :, 2] < blue_thresh)
        return mask

    def green_filter(self, red_thresh, green_thresh, blue_thresh) -> np.ndarray:
        """Filter green pen marks from the image based on given thresholds."""
        img_array = np.array(self.thumbnail)
        mask = (img_array[:, :, 0] < red_thresh) & (img_array[:, :, 1] > green_thresh) & (
                img_array[:, :, 2] < blue_thresh)
        return mask

    def black_pen_filter(self):
        """Filter out black pen marks and return a binary mask."""
        parameters = [
            {"red_thresh": 50, "green_thresh": 50, "blue_thresh": 50},
            {"red_thresh": 30, "green_thresh": 30, "blue_thresh": 30},
            {"red_thresh": 20, "green_thresh": 20, "blue_thresh": 20},
            {"red_thresh": 10, "green_thresh": 10, "blue_thresh": 10},
        ]

        pen_masks = [self.black_filter(**param) for param in parameters]
        combined_mask = np.any(pen_masks, axis=0)
        return combined_mask

    def black_filter(self, red_thresh, green_thresh, blue_thresh) -> np.ndarray:
        """Filter black pen marks from the image based on given thresholds."""
        img_array = np.array(self.thumbnail)
        mask = (img_array[:, :, 0] < red_thresh) & (img_array[:, :, 1] < green_thresh) & (
                img_array[:, :, 2] < blue_thresh)
        return mask

    def get_whole_slide_mask(self):
        _, mask, image_c = self.otsu_mask_threshold(self.thumbnail)

        # Save original image
        plt.imshow(self.thumbnail)
        plt.axis("off")
        plt.savefig(os.path.join(self.result_path, "original_slide"), bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save masked slide
        plt.imshow(mask)
        plt.axis("off")
        plt.savefig(os.path.join(self.result_path, "masked_slide"), bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save masked slide without small fragments
        small_objs = self.remove_small_objects(mask, 0, default=True)
        binary_mask = np.array((small_objs > 1).astype(np.uint8))
        wsi_thumb_masked = np.copy(self.thumbnail)
        wsi_thumb_masked[binary_mask == 0] = 255
        plt.imshow(binary_mask)
        plt.axis("off")
        plt.savefig(os.path.join(self.result_path, "masked_slide_nofragments"), bbox_inches='tight', pad_inches=0)
        plt.close()
        return small_objs

    def save_original_with_mask(self):
        """Save the original image with the combined mask applied to it."""
        original_image = np.array(self.thumbnail)

        combined_mask = self.combined_mask()
        applied_mask = np.copy(original_image)
        applied_mask[combined_mask == 0] = 0  # Apply the combined mask to the original image

        # Save the masked image
        plt.imshow(applied_mask)
        plt.axis("off")
        plt.savefig(os.path.join(self.result_path, "original_with_mask"), bbox_inches='tight', pad_inches=0)
        plt.close()
        return combined_mask

    def get_region_mask(self, original_size, size):
        mask_region_location = (original_size[0] // self.SCALE, original_size[1] // self.SCALE)
        mask_region_size = (size[0] // self.SCALE, size[1] // self.SCALE)
        return self.mask[mask_region_location[1]:mask_region_location[1] + mask_region_size[1],
               mask_region_location[0]:mask_region_location[0] + mask_region_size[0]]
