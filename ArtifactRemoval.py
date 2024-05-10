import cv2
import numpy as np
from PIL import Image

from skimage import data
from skimage.color import rgb2gray
from skimage.filters import laplace


# laplace filter to discard blurry images
def LaplaceFilter(img,var_threshold=0.015 ):
    # turn into gray
    grayscale = rgb2gray(img)
    laplace_img = cv2.Laplacian(grayscale, cv2.CV_64F, ksize=3)
    if laplace_img.var() >= var_threshold:
        return True
    return False

#
