import numpy as np
from PIL import Image


def normalizeStaining(tile, Io=240, alpha=1, beta=0.15):
    try:
        HERef = np.array([[0.5626, 0.2159],
                          [0.7201, 0.8012],
                          [0.4062, 0.5581]])

        maxCRef = np.array([1.9705, 1.0308])
        h, w, c = tile.shape

        # reshape image
        tile = tile.reshape((-1, 3))

        # calculate optical density
        tile = tile.astype(float)
        tile[tile == 0] = 1
        OD = -np.log(tile / Io)
        ODhat = OD[~np.any(OD < beta, axis=1)]
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
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

        # rows correspond to channels (RGB), columns to OD values
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
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return None
