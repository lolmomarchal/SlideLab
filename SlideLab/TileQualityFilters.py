import cv2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# laplace filter to discard blurry images
def LaplaceFilter(img, var_threshold=0.015):
    # turn into gray
    grayscale = rgb2gray(img)
    laplace_img = cv2.Laplacian(grayscale, cv2.CV_64F, ksize=3)
    variance = laplace_img.var()
    if variance <= var_threshold:
        return True, variance
    return False, variance


def plot_distribution(values, save_path, var_threshold = 0.015):
    plt.figure(figsize = (6,6))
    distribution = pd.Series(values, name = "Laplacian Variance")
    sns.histplot(data = distribution, kde = True)
    plt.axvline(x= var_threshold, color = "red", linestyle = "--", label = f"threshold = {var_threshold}")
    plt.title("Laplacian variance distribution")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(save_path)


