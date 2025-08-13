import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
import openslide
import numba
import time
from skimage.filters import threshold_otsu # pylint: disable=no-name-in-module

@numba.njit(parallel = True)
def compute_color_masks(img: np.ndarray):
    """
    Helper method for he_otsu. Calculates the red/blue vs green intensities at each pixel of the image
    Also calculates the tissue_heatmap. Unlike original method described in Schreiber et. al., uses minimum differences of r vs g and b vs g
    to calculate the tissue heatmap to circumvent exclusion of lighter tissues.
    """
    h, w, _ = img.shape
    red_to_green = np.zeros((h, w), dtype=np.float32)
    blue_to_green = np.zeros((h, w), dtype=np.float32)
    for i in numba.prange(h):
        for j in range(w):
            r = float(img[i, j, 0])
            g = float(img[i, j, 1])
            b = float(img[i, j, 2])
            red_to_green[i, j] = max(r - g, 0.0)
            blue_to_green[i, j] = max(b - g, 0.0)
    tissue_heatmap = np.minimum(red_to_green, blue_to_green)
    # tissue_heatmap = red_to_green *blue_to_green
    return tissue_heatmap


def he_otsu(image: np.ndarray, blur_kernel_width: int = 3,
            nbins: int = 256, hist=None):
    """
   Otsu thresholding method adapted for H & E images modified from Schreiber et. al.
   Args:
       :param image (np.array): An RGB image (e.g., WSI thumbnail)
       :param blur_kernel_width (int): Blurs the mask with the given blur kernel
       :param nbins (int, optional): Number of bins used to calculate histogram.
                           This value is ignored for integer arrays.
                           Defaults to 256.
       :param hist (Union[np.ndarray, Tuple[np.ndarray,np.ndarray]], optional):
                           Histogram from which to determine the threshold, and optionally a
                           corresponding array of bin center intensities. If no hist provided,
                           this function will compute it from the image. Default to None.
   Returns:
   :return: Tuple[np.ndarray, float]: The Tissue mask and upper threshold value. All pixels with an intensity higher than
       this value are assumed to be tissue
   Citations:
   B. A. Schreiber, J. Denholm, F. Jaeckle, M. J. Arends, K. M. Branson, C.-B.Schönlieb, and E. J. Soilleux. Bang and the artefacts are gone! Rapid artefact
   removal and tissue segmentation in haematoxylin and eosin stained biopsies, 2023.
   URL http://arxiv.org/abs/2308.13304.
   Original code: https://gitlab.developers.cam.ac.uk/bas43/h_and_e_otsu_thresholding#otsu
   """
    tissue_heatmap = compute_color_masks(image)
    threshold = threshold_otsu(tissue_heatmap, nbins=nbins, hist=hist)
    mask = tissue_heatmap > threshold

    # Original blur processing
    if blur_kernel_width != 0:
        blur_kernel = np.ones((blur_kernel_width, blur_kernel_width))
        mask = cv2.filter2D(mask.astype(np.uint8), -1, blur_kernel, borderType=cv2.BORDER_CONSTANT)
        mask = mask > 0

    return mask, threshold



@numba.njit
def is_tissue(masked_region, threshold=0.7):
    tissue = np.count_nonzero(masked_region)
    total_elements = masked_region.size
    if total_elements == 0:
        return False
    return tissue / total_elements >= threshold


@numba.njit
def get_region_mask(mask, scale, original_size, size):
    mask_region_location = (original_size[0] // scale, original_size[1] // scale)
    mask_region_size = (size[0] // scale, size[1] // scale)
    return mask[mask_region_location[1]:mask_region_location[1] + mask_region_size[1],
           mask_region_location[0]:mask_region_location[0] + mask_region_size[0]]


# to remove tissue folds


class TissueMask:
    """"
    TissueMask: creates tissue mask object to obtain tissue regions from a WSI
    args:
    - :param slide (str, openslide.Openslide, or TissueSlide): object with reference to WSI object
    - :param masks (list): list of filters to be used to create mask (options: 'whole_slide', 'green_pen', 'red_pen', 'blue_pen', 'black_pen', "filter_grays", "tissue_folds")
           - default: Use all
    - :param result_path (str): if provided path, will save image representation of mask overlaid with thumbnail of slide
    - :param threshold (float in range 0-1): threshold needed for a tile to be considered "tissue"
    - :param SCALE (int): scale to down sample to for masking. If given None will choose the closest available down sample to 64
    """
    def __init__(self, slide, masks=None, result_path=None,
                 threshold=0.8, SCALE=64, red_pen_thresh = 0.4, blue_pen_thresh = 0.4, remove_folds = False):

        self.red_pen_thresh =red_pen_thresh
        self.blue_pen_thresh = blue_pen_thresh
        self.slide = openslide.OpenSlide(slide) if isinstance(slide, str) else slide
        self.SCALE = SCALE or min(64,int(self.slide.level_downsamples[-1]))

        self.thumbnail = np.array(
            self.slide.get_thumbnail((self.slide.dimensions[0] // self.SCALE, self.slide.dimensions[1] // self.SCALE))
        )
        self.magnification = int(self.slide.properties.get("openslide.objective-power", 40))
        self.masks_list = masks or ['whole_slide', 'green_pen', 'red_pen', 'blue_pen', 'black_pen', "filter_grays"]
        if remove_folds:
            self.masks_list.append("tissue_folds")
        self.threshold = threshold
        self.result_path = result_path
        self.otsu = self.otsu_mask_threshold()[1]
        self.mask, self.applied = self.save_original_with_mask()

    def metadata(self):
        """Retrieves the different metadata for the Tissue Mask
        Returns:
            Metadata Dictionary
        """
        return {"slide": self.slide, "masks_list": self.masks_list, "thumbnail": self.thumbnail,
                "save path": self.result_path, "scale": self.SCALE, "mask": self.mask,
                "mask applied to original slide": self.applied}

    def get_mask_attributes(self):
        """Retrieves copy of mask array and scale
        Returns:
            - np.ndarray: mask
            - int: scale of down sample
        """
        return np.copy(self.mask), self.SCALE



    # MASK METHODS

    def otsu_mask_threshold(self, kernel_size=3, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Otsu threshold of to get tissue from glass slide
        :param kernel_size (odd int): size of kernel for iteration
        :param clip_limit (even float): size of clip limit for adaptive histogram equalization
        :param tile_grid_size: size of grid for CLAHE
        :return: otsu threshold mask (bool) of self.thumbnail
        """
        # start = time.process_time()
        # grayscale_img = cv2.cvtColor(self.thumbnail, cv2.COLOR_BGR2GRAY)
        # clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        # equalized_img = clahe.apply(grayscale_img)
        # img_inverted = 255 - equalized_img
        # _, threshold_img = cv2.threshold(img_inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold_img, _ = he_otsu(self.thumbnail)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        threshold_img = cv2.morphologyEx(threshold_img.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        ##print(f"otsu {time.process_time()-start}")
        return _, threshold_img.astype(bool)

    def remove_small_holes(self, binary_mask: np.array, min_size=None, avoid_overmask=True,
                           overmask_thresh=99, kernel_size=2):
        """
        Removes small objects (artifacts) left by masking
        :param binary_mask: np.ndarray
        :param min_size (float): min_size for something to be considered an object and not "artifact"
        :param avoid_overmask (bool):
        :param overmask_thresh (int):
        :param kernel_size (int):
        :return:
        """
        # start = time.process_time()
        if min_size is None:
            min_size = binary_mask.size * 0.0001

        binary_mask = binary_mask.astype(np.uint8)
        _, binary_mask = cv2.threshold(binary_mask, 0, 255, cv2.THRESH_BINARY)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        sizes = stats[1:, cv2.CC_STAT_AREA]
        cleaned_mask = np.zeros_like(binary_mask)

        for i in range(1, num_labels):
            if sizes[i - 1] >= min_size:
                cleaned_mask[labels == i] = 255

        mask_percentage = (np.sum(cleaned_mask > 0) / cleaned_mask.size) * 100
        if avoid_overmask:
            while mask_percentage >= overmask_thresh and min_size >= 1:
                min_size //= 2
                cleaned_mask[:] = 0
                for i in range(1, num_labels):
                    if sizes[i - 1] >= min_size:
                        cleaned_mask[labels == i] = 255
                mask_percentage = (np.sum(cleaned_mask > 0) / cleaned_mask.size) * 100
        if kernel_size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            cleaned_mask = cv2.dilate(cleaned_mask, kernel)

        return cleaned_mask

    def get_saturation_intensity(self, image_rgb):
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        s = hsv[:, :, 1] / 255.0
        v = hsv[:, :, 2] / 255.0
        return s, v

    def compute_difference_image(self, s, i):
        return s - i

    def get_connected_object_counts(self, d_img, thresholds):
        counts = []
        for t in thresholds:
            binary = (d_img > t).astype(np.uint8)
            num_labels, _ = cv2.connectedComponents(binary, connectivity=8)
            counts.append(num_labels - 1)
        return np.array(counts)

    def find_thresholds(self,counts, thresholds, alpha, beta):
        max_c = np.max(counts)
        T = lambda c: thresholds[np.where(counts >= c)[0][-1]]
        t_hard = T(alpha * max_c)
        t_soft = T(beta * max_c)
        return t_hard, t_soft

    def apply_thresholds(self,d_img, t_soft, t_hard):
        soft_mask = (d_img > t_soft).astype(np.uint8)
        hard_mask = (d_img > t_hard).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        dilated_hard = cv2.dilate(hard_mask, kernel)
        combined = cv2.bitwise_and(soft_mask, dilated_hard)
        return combined

    def detect_tissue_folds(self, alpha=0.65, beta=0.34, min_size=64):
        s, i = self.get_saturation_intensity(self.thumbnail)
        d_img = self.compute_difference_image(s, i)

        thresholds = np.arange(-1.0, 1.01, 0.05)
        counts = self.get_connected_object_counts(d_img, thresholds)

        t_hard, t_soft = self.find_thresholds(counts, thresholds, alpha, beta)

        fold_mask = self.apply_thresholds(d_img, t_soft, t_hard)
        fold_mask = self.remove_small_holes(fold_mask, min_size = min_size, kernel_size = 2)
        true_percentage = np.count_nonzero(fold_mask.astype(np.uint8)/ np.count_nonzero(self.otsu))

        if true_percentage > 0.7:
            return ~np.zeros_like(fold_mask, dtype=np.uint8)

        return ~fold_mask.astype(bool)



    def combined_mask(self):
        mask_methods = self.masks_list
        method_dict = {
            'red_pen': self.red_pen_filter,
            'blue_pen': self.blue_pen_filter,
            'green_pen': self.green_pen_filter,
            'black_pen': self.black_pen_filter,
            'whole_slide': self.get_whole_slide_mask,
            "filter_grays": self.filter_grays,
            "tissue_folds": self.detect_tissue_folds
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
            combined_mask |= r_mask & g_mask & b_mask

        kernel = np.ones((6, 6), np.uint8)
        dilated_mask = cv2.dilate(combined_mask.astype(np.uint8), kernel, iterations=1)

        return dilated_mask

    def blue_pen_filter(self):
        start = time.process_time()
        # start = time.time()
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
        true_percentage = np.count_nonzero(pen_mask.astype(np.uint8)) / np.count_nonzero(self.otsu)

        # Check if the percentage exceeds 20%
        if true_percentage > self.blue_pen_thresh:
            return ~np.zeros_like(pen_mask, dtype=np.uint8)

        # print(f"blue pen  {time.process_time()-start}")
        # #print(f"blue pen : {time.time() - start}")
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

        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(combined_mask.astype(np.uint8), kernel, iterations=1)

        return dilated_mask

    def filter_grays(self, tolerance=15, output_type="bool"):
        start = time.process_time()
        rgb = self.thumbnail

        (h, w, c) = rgb.shape

        rgb = rgb.astype(int)
        rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
        rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
        gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
        result = ~(rg_diff & rb_diff & gb_diff)

        result = result.astype("uint8") * 255
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(result, kernel, iterations=1)
        true_percentage = np.count_nonzero(dilated_mask.astype(np.uint8)) / np.count_nonzero(self.otsu)

        # Check if the percentage exceeds 20%
        if true_percentage > 0.6:
            return ~np.zeros_like(dilated_mask, dtype=np.uint8)
        return dilated_mask.astype(bool)

    def red_pen_filter(self):
        """Filter out red pen marks and return a binary mask."""
        start = time.process_time()
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

        # Check if the percentage exceeds 20%
        if true_percentage > self.red_pen_thresh:
            return ~np.zeros_like(pen_mask, dtype=np.uint8)

        # print(f"red pen  {time.process_time()-start}")
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
        start = time.process_time()
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
        combined_mask = ~masks
        # print(f"green pen  {time.process_time()-start}")
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
            {"red_thresh": 40, "green_thresh": 40, "blue_thresh": 40},
            {"red_thresh": 20, "green_thresh": 20, "blue_thresh": 20},
            {"red_thresh": 10, "green_thresh": 10, "blue_thresh": 10},
            {"red_thresh": 0, "green_thresh": 0, "blue_thresh": 0},
        ]
        pen_masks = [self.black_filter(self.thumbnail, **param) for param in parameters]
        combined_mask = np.any(pen_masks, axis=0)
        return ~combined_mask.astype(bool)

    def get_whole_slide_mask(self):
        return self.otsu

    def remove_small_holes_positive_region(self, binary_mask: np.array, min_size=None, default=True, kernel_size=2):
        start = time.process_time()

        # Set default minimum size
        if default:
            min_size = binary_mask.size * 0.0001

        # Focus only on positive regions
        binary_mask = binary_mask.astype(bool)

        # Remove small holes within positive regions
        cleaned_mask = morphology.remove_small_holes(binary_mask, area_threshold=min_size)

        return (cleaned_mask.astype(np.uint8)) * 255

    def save_original_with_mask(self):
        """Save the original image with the combined mask applied to it."""
        combined_mask = self.combined_mask().astype(np.uint8)
        combined_mask = self.remove_small_holes(combined_mask)
        combined_mask = self.remove_small_holes_positive_region(combined_mask, default=True)

        # kernel = np.ones((5, 5), np.uint8)
        # combined_mask = cv2.dilate(combined_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        applied_mask = np.copy(self.thumbnail)
        applied_mask[combined_mask == 0] = 0
        if self.result_path is not None:
            plt.imsave(os.path.join(self.result_path, "original_with_mask.png"), applied_mask)
        return combined_mask, applied_mask

    def get_region_mask(self, original_size, size):
        mask_region_location = (original_size[0] // self.SCALE, original_size[1] // self.SCALE)
        mask_region_size = (size[0] // self.SCALE, size[1] // self.SCALE)
        return self.mask[mask_region_location[1]:mask_region_location[1] + mask_region_size[1],
               mask_region_location[0]:mask_region_location[0] + mask_region_size[0]]
