import cv2
import numpy as np
import numba
from skimage.filters import threshold_otsu

# ================================ HELPER METHODS ===============================
#  Tissue Segmentation
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
    tissue_heatmap = tissue_heatmap / np.max(tissue_heatmap)
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


# Category 3: Artefact cleanup

# ======================= PENS & MARKS =============================
def get_pen_parameters():
    parameters = {"blue_pen": [
        (60, 120, 190),
        (120, 170, 200),
        (175, 210, 230),
        (145, 180, 210),
        (37, 95, 160),
        (30, 65, 130),
        (130, 155, 180),
        (40, 35, 85),
        (30, 20, 65),
        # (90, 90, 140), excluded for being too harsh
        (60, 60, 120),
        (110, 110, 175),
    ],
        "red_pen": [
            (150, 80, 90),
            (110, 20, 30),
            (185, 65, 105),
            (195, 85, 125),
            (220, 115, 145),
            (125, 40, 70),
            (100, 50, 65),
            (85, 25, 45),
        ],
        "green_pen": [
            (150, 160, 140),
            (70, 110, 110),
            (45, 115, 100),
            (30, 75, 60),
            (195, 220, 210),
            (225, 230, 225),
            (170, 210, 200),
            (20, 30, 20),
            (50, 60, 40),
            (30, 50, 35),
            (65, 70, 60),
            (100, 110, 105),
            (165, 180, 180),
            (140, 140, 150),
            (185, 195, 195),
        ],
        "black_pen":[
            (50, 50, 50),
            # (40, 40, 40),
            (30, 30, 30),
            (20, 20, 20),
            (10, 10, 10)]

    }
    return parameters
# ---- blue pen
@numba.njit(parallel=True)
def blue_filter_numba(img_array, thresholds):
    h, w, _ = img_array.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in numba.prange(h):
        for j in range(w):
            r, g, b = img_array[i, j]
            for t in thresholds:
                if r < t[0] and g < t[1] and b > t[2]:
                    mask[i, j] = 1
                    break
    return mask


@numba.njit(parallel=True)
def blue_filter_fast(img_array, mask_area, thresholds):
    h, w, _ = img_array.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in numba.prange(h):
        for j in range(w):
            if not mask_area[i, j]:
                continue  # skip areas that are already excluded in otsu

            r, g, b = img_array[i, j]
            for t in thresholds:
                if r < t[0] and g < t[1] and b > t[2]:
                    mask[i, j] = 1
                    break
    return mask


# -- red pen
@numba.njit(parallel=True)
def red_filter_numba(img_array, thresholds):
    h, w, _ = img_array.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in numba.prange(h):
        for j in range(w):
            r, g, b = img_array[i, j]
            for t in thresholds:
                if r > t[0] and g < t[1] and b < t[2]:
                    mask[i, j] = 1
                    break
    return mask


@numba.njit(parallel=True)
def red_filter_fast(img_array, mask_area, thresholds):
    h, w, _ = img_array.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in numba.prange(h):
        for j in range(w):
            if not mask_area[i, j]:
                continue  # skip areas that are already excluded in otsu

            r, g, b = img_array[i, j]
            for t in thresholds:
                if r > t[0] and g < t[1] and b < t[2]:
                    mask[i, j] = 1
                    break
    return mask


# -- green pen
@numba.njit(parallel=True)
def green_filter_numba(img_array, thresholds):
    h, w, _ = img_array.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in numba.prange(h):
        for j in range(w):
            r, g, b = img_array[i, j]
            for t in thresholds:
                if r < t[0] and g > t[1] and b > t[2]:
                    mask[i, j] = 1
                    break
    return mask


@numba.njit(parallel=True)
def green_filter_fast(img_array, mask_area, thresholds):
    h, w, _ = img_array.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in numba.prange(h):
        for j in range(w):
            if not mask_area[i, j]:
                continue  # skip areas that are already excluded in otsu

            r, g, b = img_array[i, j]
            for t in thresholds:
                if r < t[0] and g > t[1] and b > t[2]:
                    mask[i, j] = 1
                    break
    return mask

@numba.njit(parallel= True)
def black_filter_numba(img_array, thresholds):
    h, w, _ = img_array.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in numba.prange(h):
        for j in range(w):
            r, g, b = img_array[i, j]
            for t in thresholds:
                if r < t[0] and g < t[1] and b < t[2]:
                    mask[i, j] = 1
                    break
    return mask
@numba.njit(parallel=True)
def black_filter_fast(img_array, mask_area, thresholds):
    h, w, _ = img_array.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in numba.prange(h):
        for j in range(w):
            if not mask_area[i, j]:
                continue  # skip areas that are already excluded in otsu

            r, g, b = img_array[i, j]
            for t in thresholds:
                if r < t[0] and g < t[1] and b < t[2]:
                    mask[i, j] = 1
                    break
    return mask
@numba.njit(parallel = True)
def gray_filter_fast(rgb, mask_area, tolerance= 15):
    h, w, _ = rgb.shape
    mask = np.ones((h, w), dtype=np.uint8)

    for i in numba.prange(h):
        for j in range(w):
            if not mask_area[i,j]:
                continue
            r = rgb[i, j, 0]
            g = rgb[i, j, 1]
            b = rgb[i, j, 2]

            rg_diff = abs(r - g) <= tolerance
            rb_diff = abs(r - b) <= tolerance
            gb_diff = abs(g - b) <= tolerance

            if not (rg_diff & rb_diff & gb_diff):
                mask[i, j] = 0
    return mask

@numba.njit(parallel=True)
def gray_filter_numba(rgb, tolerance=15):
    h, w, _ = rgb.shape
    mask = np.ones((h, w), dtype=np.uint8)

    for i in numba.prange(h):
        for j in range(w):
            r = rgb[i, j, 0]
            g = rgb[i, j, 1]
            b = rgb[i, j, 2]

            rg_diff = abs(r - g) <= tolerance
            rb_diff = abs(r - b) <= tolerance
            gb_diff = abs(g - b) <= tolerance

            if not (rg_diff & rb_diff & gb_diff):
                mask[i, j] = 0
    return mask



# ========= TISSUE FOLDS ======