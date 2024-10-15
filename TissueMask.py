import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
import openslide


class TissueMask:
    def __init__(self, slide, masks=None, result_path=None,
                 threshold=0.8, SCALE = None):

        if type(slide) == openslide.OpenSlide:
            self.slide = slide
            self.magnification = int(self.slide.properties.get("openslide.objective-power"))
            self.SCALE = int(self.slide.level_downsamples[-1])
            self.thumbnail = np.array(self.slide.get_thumbnail(
                (self.slide.dimensions[0] // self.SCALE, self.slide.dimensions[1] // self.SCALE)))
        else:
            self.slide = slide
            self.SCALE = slide.SCALE
            self.thumbnail = np.array(self.slide.thumbnail)
            self.id = self.slide.id

        if masks is None:
            self.masks_list = ['whole_slide', 'green_pen', 'red_pen', 'blue_pen', 'black_pen']
        else:
            self.masks_list = list(masks)
        self.threshold = threshold
        self.otsu = self.otsu_mask_threshold()[1]
        self.result_path = result_path
        if SCALE is not None:
            self.SCALE = SCALE

        self.mask, self.applied = self.save_original_with_mask()

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
        return tissue / total_elements >= threshold
    # MASK METHODS


    def otsu_mask_threshold(self, kernel_size=2, clip_limit=2.0, tile_grid_size=(8, 8)):
        #start = time.time()
        img_array = np.array(self.thumbnail)

        # Convert to grayscale
        grayscale_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_img = clahe.apply(grayscale_img)

        # Invert the image so that tissue appears as white and background as black
        img_c = 255 - enhanced_img

        # Apply Otsu's thresholding method
        _, thres_img = cv2.threshold(img_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply morphological closing with kernel size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        thres_img = cv2.morphologyEx(thres_img, cv2.MORPH_CLOSE, kernel)
        #print(f"otsu: {time.time()-start}")

        return _, thres_img, img_c


    def remove_small_holes(self, binary_mask: np.array, min_size=None, default=True, avoid_overmask=True,
                             overmask_thresh=95, kernel_size=2):
        #start = time.time()

        if default:
            min_size = binary_mask.size * 0.0001
        binary_mask = binary_mask.astype(bool)
        cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_size)
        mask_percentage = (np.sum(cleaned_mask) / cleaned_mask.size) * 100

        while avoid_overmask and mask_percentage >= overmask_thresh and min_size >= 1:
            min_size //= 2
            cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_size)
            mask_percentage = (np.sum(cleaned_mask) / cleaned_mask.size) * 100
            if mask_percentage < overmask_thresh:
                break

        if kernel_size > 1:
            selem = morphology.square(kernel_size)
            cleaned_mask = morphology.dilation(cleaned_mask, selem)


        #print(f"small objects : {time.time() - start}")

        # Return the mask in uint8 format
        return (cleaned_mask.astype(np.uint8)) * 255

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
        for i in range(1, len(mask_methods)):
            mask = method_dict[mask_methods[i]]().astype(bool)
            combined_mask = np.logical_and(combined_mask, mask)
        return combined_mask

    # Pen Filters

    # Blue Pen Filter
    def blue_filter(self, img_array, thresholds):
        """Filter blue pen marks from the image based on given thresholds."""

        combined_mask = np.zeros(img_array.shape[:2], dtype=bool)

        r_mask = np.zeros_like(combined_mask)
        g_mask = np.zeros_like(combined_mask)
        b_mask = np.zeros_like(combined_mask)
        for red_thresh, green_thresh, blue_thresh in thresholds:
            r_mask[:] = img_array[:, :, 0] < red_thresh
            g_mask[:] = img_array[:, :, 1] < green_thresh
            b_mask[:] = img_array[:, :, 2] > blue_thresh

        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(combined_mask.astype(np.uint8), kernel, iterations=1)

        return dilated_mask

    def blue_pen_filter(self):
        #start = time.time()
        """Filter out blue pen marks and return a binary mask."""
        parameters = [
            (60, 120, 190),
            (120, 170, 200),
            (175, 210, 230),
            (145, 180, 210),
            (37, 95, 160),
            (30, 65, 130),
            (130, 155, 180),
            (40, 35, 85),
            (30, 20, 65),
            (90, 90, 140),
            (60, 60, 120),
            (110, 110, 175),
        ]

        pen_mask = self.blue_filter(self.thumbnail, parameters)

        #print(f"blue pen : {time.time() - start}")
        return ~pen_mask.astype(bool)

    # red pen filter

    # for sanity purposes: if more than > 20% of tissue identified by primary otsu thresh, ignore
    def red_filter(self, img_array, thresholds) -> np.ndarray:
        """Filter red pen marks from the image based on given thresholds."""

        combined_mask = np.zeros(img_array.shape[:2], dtype=bool)


        r_mask = np.zeros_like(combined_mask)
        g_mask = np.zeros_like(combined_mask)
        b_mask = np.zeros_like(combined_mask)

        for red_thresh, green_thresh, blue_thresh in thresholds:
            r_mask[:] = img_array[:, :, 0] > red_thresh
            g_mask[:] = img_array[:, :, 1] < green_thresh
            b_mask[:] = img_array[:, :, 2] < blue_thresh

            combined_mask |= r_mask & g_mask & b_mask


            if np.count_nonzero(combined_mask) > 0.8 * np.prod(combined_mask.shape):
                break


        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(combined_mask.astype(np.uint8), kernel, iterations=1)

        return dilated_mask

    def red_pen_filter(self):
        """Filter out red pen marks and return a binary mask."""
        #start = time.time()
        parameters = [
            (150, 80, 90),
            (110, 20, 30),
            (185, 65, 105),
            (195, 85, 125),
            (220, 115, 145),
            (125, 40, 70),
            (100, 50, 65),
            (85, 25, 45),
        ]

        pen_mask = self.red_filter(self.thumbnail, parameters)


        true_percentage = np.count_nonzero(pen_mask.astype(np.uint8)) / np.count_nonzero(self.otsu)

        # Check if the percentage exceeds 40%
        if true_percentage > 0.4:
            return ~np.zeros_like(pen_mask, dtype=np.uint8)

        #print(f"red pen : {time.time() - start}")
        return ~pen_mask.astype(bool)

    def filter_green(self, img_array, thresholds):
        """Filter green based on multiple thresholds."""
        combined_mask = np.zeros(img_array.shape[:2], dtype=bool)

        for red_upper_thresh, green_lower_thresh, blue_lower_thresh in thresholds:
            r_mask = img_array[:, :, 0] < red_upper_thresh
            g_mask = img_array[:, :, 1] > green_lower_thresh
            b_mask = img_array[:, :, 2] > blue_lower_thresh
            combined_mask |= r_mask & g_mask & b_mask

        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(combined_mask.astype(np.uint8), kernel, iterations=1)

        return dilated_mask.astype(bool)

    def green_pen_filter(self):
        #start = time.time()
        """Filter out green pen marks and return a binary mask."""
        thresholds = [
            [150, 160, 140],
            [70, 110, 110],
            [45, 115, 100],
            [30, 75, 60],
            [195, 220, 210],
            [225, 230, 225],
            [170, 210, 200],
            [20, 30, 20],
            [50, 60, 40],
            [30, 50, 35],
            [65, 70, 60],
            [100, 110, 105],
            [165, 180, 180],
            [140, 140, 150],
            [185, 195, 195],
        ]

        masks = self.filter_green(self.thumbnail, thresholds)
        combined_mask = ~masks  # Invert the mask
        #print(f"green pen : {time.time() - start}")
        return combined_mask

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
        pen_masks = [self.black_filter(self.thumbnail, **param) for param in parameters]
        combined_mask = np.any(pen_masks, axis=0)
        return ~combined_mask.astype(bool)

    def get_whole_slide_mask(self):
        mask = self.otsu
        if self.result_path is not None:
            plt.imsave(os.path.join(self.result_path, "original_slide.png"), self.thumbnail)
        small_objs = self.remove_small_holes(mask, default=True)

        return small_objs

    def save_original_with_mask(self):
        """Save the original image with the combined mask applied to it."""
        original_image = self.thumbnail
        combined_mask = self.combined_mask().astype(np.uint8)
        applied_mask = np.copy(original_image)
        applied_mask[combined_mask == 0] = 0
        if self.result_path is not None:
            plt.imsave(os.path.join(self.result_path, "original_with_mask.png"), applied_mask)
        return combined_mask, applied_mask

    def get_region_mask(self, original_size, size):
        mask_region_location = (original_size[0] // self.SCALE, original_size[1] // self.SCALE)
        mask_region_size = (size[0] // self.SCALE, size[1] // self.SCALE)
        return self.mask[mask_region_location[1]:mask_region_location[1] + mask_region_size[1],
               mask_region_location[0]:mask_region_location[0] + mask_region_size[0]]
