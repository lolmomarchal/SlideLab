import cv2
import numpy as np

def LaplaceFilter(img, var_threshold=0.015):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    laplace_img = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    variance = laplace_img.var()
    return (variance <= var_threshold), variance

