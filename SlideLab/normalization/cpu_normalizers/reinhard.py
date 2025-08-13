import numba
import numpy as np
import cv2
@numba.jit(nopython=True)
def lab_adjust(I):
    I1 = I[:,:,0].astype(np.float32)/2.55
    I2 = I[:,:,1].astype(np.float32)-128.0
    I3 = I[:,:,2].astype(np.float32)-128.0
    return I1, I2, I3

@numba.jit(nopython=True, parallel = True)
def mean_std_channel_masked(I, mask):
    means = np.empty(3, dtype=np.float32)
    stds = np.empty(3, dtype=np.float32)
    for c in numba.prange(3):
        channel = I[:, :, c].ravel()
        masked_channel = channel[mask.ravel()]
        mean = np.mean(masked_channel) if masked_channel.size > 0 else 0.0
        std = np.std(masked_channel) if masked_channel.size > 0 else 1.0  #
        means[c] = mean
        stds[c] = std
    return means, stds

def reinhardNormalizer(img, threshold = None):
    target_means = np.array([65.22132,  28.934267, -14.142519])
    target_stds =  np.array([15.800227,  9.263783,  6.0213304])
    I_lab  = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    mask = np.all((I_lab != [255, 255, 255]) & (I_lab != [0,0,0]), axis=-1)
    I1, I2, I3 = lab_adjust(I_lab)
    means, stds = mean_std_channel_masked(np.stack([I1, I2, I3], axis=-1).astype(np.float32), mask)
    q = (target_stds[0] - stds[0]) / target_stds[0]
    q = 0.05 if q <= 0 else q
    norm1 = means[0] + (I1 - means[0]) * (1 + q)
    norm2 = target_means[1] + (I2 - means[1])
    norm3 = target_means[2] + (I3 - means[2])
    norm1 *= 2.55
    norm2 += 128.0
    norm3 += 128.0
    I_norm_lab = np.clip(cv2.merge((norm1, norm2, norm3)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(I_norm_lab, cv2.COLOR_LAB2RGB)