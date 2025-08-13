import numba
import numpy as np
import warnings
warnings.filterwarnings("ignore")

@numba.jit(nopython=True)
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
@numba.jit(nopython=True)
def percentile_numba(arr, q):
    arr_sorted = np.sort(arr)
    n = len(arr_sorted)
    rank = q / 100 * (n - 1)
    low = int(np.floor(rank))
    high = int(np.ceil(rank))
    weight = rank - low
    return arr_sorted[low] * (1 - weight) + arr_sorted[high] * weight
@numba.jit(nopython=True, parallel = True)
def cov_numba_standalone(x):
    n_vars, n_obs = x.shape
    means = np.zeros((n_vars, 1))
    for i in numba.prange(n_vars):
        s = 0.0
        for j in range(n_obs):
            s += x[i, j]
        means[i, 0] = s / n_obs
    deviations = x - means
    cov_matrix = (deviations @ deviations.T) / (n_obs - 1)
    return cov_matrix
@numba.jit(nopython=True)
def get_eig_standalone(x):
    eigval, eigvec = np.linalg.eigh(x)
    return eigval, eigvec

@numba.jit(nopython=True)
def calculate_that_standalone(v1, v2):
    v1 = np.ascontiguousarray(v1)
    return v1.dot(np.ascontiguousarray(v2[:, 1:3]))

@numba.jit(nopython=True)
def calculate_percentiles_standalone(phi, alpha):
    minPhi = percentile_numba(phi, alpha)
    maxPhi = percentile_numba(phi, 100 - alpha)
    return minPhi, maxPhi
@numba.jit(nopython=True)
def vmin_vmax_standalone(V, minPhi, maxPhi):
    V = np.ascontiguousarray(V)
    v1 = np.dot(V, np.ascontiguousarray(np.array([np.cos(minPhi), np.sin(minPhi)]).T))
    v2 = np.dot(V, np.ascontiguousarray(np.array([np.cos(maxPhi), np.sin(maxPhi)]).T))
    return v1.reshape(-1,1), v2.reshape(-1,1)

def macenkoNormalizer_cpu(img, alpha = 1, beta =0.15, Io = 240):
    # constants
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    HERef = np.ascontiguousarray(HERef)
    maxCRef = np.array([1.9705, 1.0308])
    maxCRef = np.ascontiguousarray(maxCRef)

    h,w,c = img.shape
    try:
        tile_np = np.ascontiguousarray(img)
        tile_np = tile_np.reshape((-1, 3)).astype(float)
        tile_np[tile_np == 0] = 1
        OD = -np.log(tile_np / Io)
        ODhat = OD[~np.any(OD < beta, axis=1)]
        cov_ODhat = np.cov(ODhat.T)
        eigvals, eigvecs = get_eig_standalone(cov_ODhat)
        That = calculate_that_standalone(ODhat, eigvecs)
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi, maxPhi = calculate_percentiles_standalone(phi, alpha)
        vMin, vMax = vmin_vmax_standalone(eigvecs[:, 1:3], minPhi, maxPhi)
        Inorm= process_OD_standalone(OD, vMin, vMax, w, h, Io, HERef, maxCRef)
        Inorm[Inorm > 255] = 254
        Inorm = np.reshape(Inorm.T, (w, h, 3)).astype(np.uint8)
    except:
        return None
    return Inorm