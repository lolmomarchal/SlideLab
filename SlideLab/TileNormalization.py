import numpy as np
import warnings
import torch
warnings.filterwarnings("ignore", category=UserWarning)

# if gpu not available, use numpy vers
def normalizeStaining(tile, Io=240, alpha=1, beta=0.15):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)

            HERef = np.array([[0.5626, 0.2159],
                              [0.7201, 0.8012],
                              [0.4062, 0.5581]])
            maxCRef = np.array([1.9705, 1.0308])

            h, w, c = tile.shape

            tile = tile.reshape((-1, 3))

            tile = tile.astype(float)
            tile[tile == 0] = 1
            OD = -np.log(tile / Io)

            ODhat = OD[~np.any(OD < beta, axis=1)]

            eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

            # project on the plane spanned by the eigenvectors corresponding to the two
            # largest eigenvalues
            That = ODhat.dot(eigvecs[:, 1:3])

            phi = np.arctan2(That[:, 1], That[:, 0])

            minPhi = np.percentile(phi, alpha)
            maxPhi = np.percentile(phi, 100 - alpha)

            vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
            vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

            # heuristic to make hematoxylin first and eosin second
            if vMin[0] > vMax[0]:
                HE = np.array((vMin[:, 0], vMax[:, 0])).T
            else:
                HE = np.array((vMax[:, 0], vMin[:, 0])).T

            Y = np.reshape(OD, (-1, 3)).T

            C = np.linalg.lstsq(HE, Y, rcond=None)[0]

            maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
            tmp = np.divide(maxC, maxCRef).astype(np.float64)
            C2 = np.divide(C, tmp[:, np.newaxis]).astype(np.float64)

            with np.errstate(over='raise', divide='raise', invalid='raise'):
                Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
                Inorm[Inorm > 255] = 254
                Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

            return Inorm
    except (FloatingPointError, ValueError, np.linalg.LinAlgError, RuntimeWarning):
        return None


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
