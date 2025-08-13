import numpy as np
import warnings
import torch
import numba
warnings.filterwarnings("ignore", category=UserWarning)

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
@numba.jit(nopython=True)
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
@numba.jit(nopython=True)
def get_eig_standalone(x):
    eigval, eigvec = np.linalg.eigh(x)
    return eigval, eigvec

@numba.jit(nopython=True)
def calculate_that_standalone(v1, v2):
    return np.ascontiguousarray(v1).dot(np.ascontiguousarray(v2[:, 1:3]))

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



########## MACKENCO ####################
# In a separate Numba-focused utility file or similar

# Original normalizeStaining function (in TileNormalization.py or similar)
@numba.jit(nopython=True)
def _normalize_staining_numba_core(tile_np_reshaped, h, w, Io, alpha, beta, HERef, maxCRef):
    # tile_np_reshaped is the (num_pixels, 3) version
    # h and w are the original image dimensions (e.g., 256, 256)

    OD = -np.log(tile_np_reshaped / Io)
    ODhat = OD[~((OD < beta).sum(axis=1) > 0)]

    cov_ODhat = np.cov(ODhat.T)
    eigvals, eigvecs = get_eig_standalone(cov_ODhat)
    That = calculate_that_standalone(ODhat, eigvecs)

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi, maxPhi = calculate_percentiles_standalone(phi, alpha)
    vMin, vMax = vmin_vmax_standalone(eigvecs[:, 1:3], minPhi, maxPhi)

    # Pass the correct original w and h to process_OD_standalone
    Inorm = process_OD_standalone(OD, vMin, vMax, w, h, Io, HERef, maxCRef)

    # Clamp values
    Inorm = np.where(Inorm > 255, 254, Inorm)

    # Reshape back to original dimensions (h, w, 3) using the passed h and w
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    return Inorm

def normalizeStaining(tile, Io=240, alpha=1, beta=0.15):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)

            HERef = np.array([[0.5626, 0.2159],
                              [0.7201, 0.8012],
                              [0.4062, 0.5581]], dtype=np.float64)
            maxCRef = np.array([1.9705, 1.0308], dtype=np.float64)

            # Ensure HERef and maxCRef are contiguous
            HERef = np.ascontiguousarray(HERef)
            maxCRef = np.ascontiguousarray(maxCRef)

            # Convert tile to numpy array and get original dimensions
            tile_np_original = tile.cpu().numpy() if torch.is_tensor(tile) else tile
            h_orig, w_orig, c_orig = tile_np_original.shape # Get original height and width

            # Reshape for processing within the numba core function
            tile_np_reshaped = tile_np_original.reshape((-1, 3)).astype(np.float64)
            tile_np_reshaped[tile_np_reshaped == 0] = 1 # Handle zero values

            # Call the numba-optimized core function, passing original h and w
            Inorm = _normalize_staining_numba_core(tile_np_reshaped, h_orig, w_orig, Io, alpha, beta, HERef, maxCRef)
            return Inorm
    except (FloatingPointError, ValueError, np.linalg.LinAlgError, RuntimeWarning, Exception) as e:
        print(f"Normalization error: {e}")
        return None
# def normalizeStaining(tile, Io=240, alpha=1, beta=0.15):
#     try:
#         with warnings.catch_warnings():
#             warnings.simplefilter("error", category=RuntimeWarning)
#
#             HERef = np.array([[0.5626, 0.2159],
#                               [0.7201, 0.8012],
#                               [0.4062, 0.5581]])
#             maxCRef = np.array([1.9705, 1.0308])
#             maxCRef = np.ascontiguousarray(maxCRef)
#
#             HERef = np.ascontiguousarray(HERef)
#
#             h, w, c = tile.shape
#             tile_np = tile.cpu().numpy() if torch.is_tensor(tile) else tile
#             tile_np = np.ascontiguousarray(tile_np)
#             tile_np = tile_np.reshape((-1, 3)).astype(float)
#
#             tile_np[tile_np == 0] = 1
#             OD = -np.log(tile_np / Io)
#
#             ODhat = OD[~np.any(OD < beta, axis=1)]
#
#             cov_ODhat = np.cov(ODhat.T)
#             eigvals, eigvecs = get_eig_standalone(cov_ODhat)
#             That = calculate_that_standalone(ODhat, eigvecs)
#
#             phi = np.arctan2(That[:, 1], That[:, 0])
#
#             minPhi, maxPhi = calculate_percentiles_standalone(phi, alpha)
#             vMin, vMax = vmin_vmax_standalone(eigvecs[:, 1:3], minPhi, maxPhi)
#             Inorm= process_OD_standalone(OD, vMin, vMax, w, h, Io, HERef, maxCRef)
#             Inorm[Inorm > 255] = 254
#             Inorm = np.reshape(Inorm.T, (w, h, 3)).astype(np.uint8)
#             return Inorm
#     except (FloatingPointError, ValueError, np.linalg.LinAlgError, RuntimeWarning):
#         return None


# if gpu available, use torch
def normalizeStaining_torch(tile, Io=240, alpha=1, beta=0.15, device=None):
    try:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        tile = tile.to(device)
        HERef = torch.tensor([[0.5626, 0.2159],
                              [0.7201, 0.8012],
                              [0.4062, 0.5581]], dtype=torch.float32, device=device)

        maxCRef = torch.tensor([1.9705, 1.0308], dtype=torch.float32, device=device)
        h, w, c = tile.shape
        tile = tile.reshape(-1, 3)

        tile = tile.float()
        tile = torch.where(tile == 0, torch.tensor(1.0, device=device), tile)  # Avoid division by zero
        OD = -torch.log(tile / Io)

        ODhat = OD[(OD >= beta).all(dim=1)]
        n_obs = ODhat.shape[1]
        if n_obs <=1:
            return None

        cov_matrix = torch.cov(ODhat.T)
        eigvals, eigvecs = torch.linalg.eigh(cov_matrix)

        That = torch.matmul(ODhat, eigvecs[:, 1:3])

        phi = torch.atan2(That[:, 1], That[:, 0])
        minPhi = torch.quantile(phi, alpha / 100.0)
        maxPhi = torch.quantile(phi, 1 - alpha / 100.0)

        vMin = torch.matmul(eigvecs[:, 1:3], torch.tensor([[torch.cos(minPhi)], [torch.sin(minPhi)]], device=device))
        vMax = torch.matmul(eigvecs[:, 1:3], torch.tensor([[torch.cos(maxPhi)], [torch.sin(maxPhi)]], device=device))

        if vMin[0] > vMax[0]:
            HE = torch.stack([vMin[:, 0], vMax[:, 0]], dim=1)
        else:
            HE = torch.stack([vMax[:, 0], vMin[:, 0]], dim=1)

        Y = OD.T

        C = torch.linalg.lstsq(HE, Y).solution

        maxC = torch.tensor([torch.quantile(C[0, :], 0.99), torch.quantile(C[1, :], 0.99)], device=device)
        tmp = maxC / maxCRef
        C2 = C / tmp.unsqueeze(1)

        Inorm = Io * torch.exp(-torch.matmul(HERef, C2))
        Inorm = torch.clamp(Inorm, 0, 255)
        Inorm = Inorm.T.reshape(h, w, 3)

        return Inorm.byte().to('cpu')

    except (RuntimeError, ValueError, FloatingPointError):
        return None
#### REINHARD

### VALDEHANE
