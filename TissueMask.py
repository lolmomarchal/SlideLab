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

    def is_tissue(self, masked_region, threshold=0.65):
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

    def remove_small_objects(self, binary_mask, min_size=None, default=True, avoid_overmask=True, overmask_thresh=95,
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
            # Add more threshold combinations as needed
        ]

        pen_masks = [self.blue_filter(**param) for param in parameters]
        combined_mask = np.any(pen_masks, axis=0)
        return combined_mask

    def red_pen_filter(self):
        """Filter out red pen marks and return a binary mask."""
        parameters = [
            {"red_thresh": 150, "green_thresh": 80, "blue_thresh": 90},
            {"red_thresh": 110, "green_thresh": 20, "blue_thresh": 30},
            # Add more threshold combinations as needed
        ]
        pen_masks = [self.red_filter(self.thumbnail, **param) for param in parameters]
        combined_mask = np.any(pen_masks, axis=0)
        return combined_mask

    def green_pen_filter(self):
        """Filter out green pen marks and return a binary mask."""
        parameters = [
            {"red_thresh": 150, "green_thresh": 160, "blue_thresh": 140},
            {"red_thresh": 70, "green_thresh": 110, "blue_thresh": 110},
            # Add more threshold combinations as needed
        ]

        pen_masks = [self.green_filter(**param) for param in parameters]
        combined_mask = np.any(pen_masks, axis=0)
        return combined_mask

    def combined_mask(self):
        """Generate a combined mask."""
        red_mask = self.red_pen_filter().astype(np.uint8)
        blue_mask = self.blue_pen_filter().astype(np.uint8)
        green_mask = self.green_pen_filter().astype(np.uint8)
        whole_mask = self.get_whole_slide_mask()

        combined_mask = np.bitwise_or(whole_mask, np.bitwise_or(red_mask, np.bitwise_or(blue_mask, green_mask)))
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

        # Apply actual mask to image
        overlay = np.copy(self.thumbnail)
        overlay[binary_mask == 0] = 0  # Ensure binary_mask is a boolean mask
        plt.imshow(overlay)
        plt.axis("off")
        plt.savefig(os.path.join(self.result_path, "masked_overlay"), bbox_inches='tight', pad_inches=0)
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
        mask_region_size = (size[0]//self.SCALE, size[1]//self.SCALE)
        return self.mask[mask_region_location[1]:mask_region_location[1] + mask_region_size[1],
               mask_region_location[0]:mask_region_location[0] + mask_region_size[0]]
