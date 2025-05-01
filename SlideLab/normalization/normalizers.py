import os
import sys
import subprocess
import numba
import cupy as cp
import torch.nn as nn
import torch
import numpy as np


@numba.njit
def process_OD_standalone(OD, vMin, vMax, w, h, Io, HERef, maxCRef):
    if vMin[0] > vMax[0]:
        HE = np.stack((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.stack((vMax[:, 0], vMin[:, 0])).T

    Y = np.reshape(OD, (-1, 3)).T
    HE_pinv = np.linalg.pinv(HE)
    C = HE_pinv @ Y

    maxC0 = percentile_numba(C[0, :], 99)
    maxC1 = percentile_numba(C[1, :], 99)
    maxC = np.array([maxC0, maxC1])

    tmp = maxC / maxCRef
    C2 = C / tmp[:, np.newaxis]

    stain_matrix = HERef @ C2
    Inorm = Io * np.exp(-stain_matrix)
    return Inorm

@numba.njit
def percentile_numba(arr, q):
    arr_sorted = np.sort(arr)
    n = len(arr_sorted)
    rank = q / 100 * (n - 1)
    low = int(np.floor(rank))
    high = int(np.ceil(rank))
    weight = rank - low
    return arr_sorted[low] * (1 - weight) + arr_sorted[high] * weight
@numba.njit
def cov_numba_standalone(x):
    n_vars, n_obs = x.shape
    means = np.zeros((n_vars, 1))
    for i in range(n_vars):
        s = 0.0
        for j in range(n_obs):
            s += x[i, j]
        means[i, 0] = s / n_obs
    deviations = x - means
    cov_matrix = (deviations @ deviations.T) / (n_obs - 1)
    return cov_matrix
@numba.njit
def get_eig_standalone(x):
    eigval, eigvec = np.linalg.eigh(x)
    return eigval, eigvec

@numba.njit
def calculate_that_standalone(v1, v2):
    return v1.dot(v2[:, 1:3])

@numba.njit
def calculate_percentiles_standalone(phi, alpha):
    minPhi = percentile_numba(phi, alpha)
    maxPhi = percentile_numba(phi, 100 - alpha)
    return minPhi, maxPhi
@numba.njit
def vmin_vmax_standalone(V, minPhi, maxPhi):
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]).T)
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]).T)
    return v1.reshape(-1,1), v2.reshape(-1,1)


class MacenkoNormalizer(nn.Module):
    def __init__(self, device="cpu", alpha=1, beta=0.15, Io=240):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.Io = Io
        if self.device == "cpu":
            self.HERef = np.array([[0.5626, 0.2159],
                                   [0.7201, 0.8012],
                                   [0.4062, 0.5581]])
            self.maxCRef = np.array([1.9705, 1.0308])
        else:
            self.HERef = torch.tensor([[0.5626, 0.2159],
                                       [0.7201, 0.8012],
                                       [0.4062, 0.5581]], dtype=torch.float32, device=self.device)
            self.maxCRef = torch.tensor([1.9705, 1.0308], dtype=torch.float32, device=self.device)

    def normalizeStaining(self, img):
        norm = []
        if len(img.shape) == 4:
            b, h, w, c = img.shape
        else:
            img = img[np.newaxis, :]
            b, h, w, c = img.shape
        for tile in img:
            try:

                tile_np = tile.cpu().numpy() if torch.is_tensor(tile) else tile
                tile_np = tile_np.reshape((-1, 3)).astype(float)
                tile_np[tile_np == 0] = 1
                OD = -np.log(tile_np / self.Io)

                ODhat = OD[~np.any(OD < self.beta, axis=1)]

                cov_ODhat = cov_numba_standalone(ODhat.T)

                eigvals, eigvecs = get_eig_standalone(cov_ODhat)

                That = calculate_that_standalone(ODhat, eigvecs)

                phi = np.arctan2(That[:, 1], That[:, 0])

                minPhi, maxPhi = calculate_percentiles_standalone(phi, self.alpha)
                vMin, vMax = vmin_vmax_standalone(eigvecs[:, 1:3], minPhi, maxPhi)

                Inorm= process_OD_standalone(OD, vMin, vMax, w, h, self.Io, self.HERef, self.maxCRef)
                Inorm[Inorm > 255] = 254

                Inorm = np.reshape(Inorm.T, (w, h, 3)).astype(np.uint8)

                norm.append(Inorm)
            except:
                continue
        return np.array(norm)
    def forward(self, x):
        if self.device == "cpu":
            norm = self.normalizeStaining(x)
            return norm

