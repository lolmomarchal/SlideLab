import torch
def macenkoNormalizer(batch, alpha = 1, beta =0.15, Io = 240, device = "cuda"):
    batch = batch.to(device).float()
    B, H, W, C = map(int, batch.shape)
    batch = batch.reshape(B, -1, 3)
    batch = torch.where(batch == 0, 1.0, batch)

    HERef = torch.tensor([[0.5626, 0.2159],
                          [0.7201, 0.8012],
                          [0.4062, 0.5581]], dtype=torch.float32, device=device)
    maxCRef = torch.tensor([1.9705, 1.0308], dtype=torch.float32, device=device)

    def _normalize_single(tile):
        OD = -torch.log(tile / Io)
        ODhat = OD[(OD >= beta).all(dim=1)]
        if ODhat.shape[0] <= 1:
            return torch.zeros(H * W, 3, device=device)

        cov_matrix = torch.cov(ODhat.T)
        eigvals, eigvecs = torch.linalg.eigh(cov_matrix)
        if torch.isnan(eigvals).any() or torch.isinf(eigvals).any() or torch.all(eigvals.abs() < 1e-6):
            return torch.zeros(H * W, 3, device=device)

        That = torch.matmul(ODhat, eigvecs[:, 1:3])
        phi = torch.atan2(That[:, 1], That[:, 0])
        minPhi = torch.quantile(phi, alpha / 100.0)
        maxPhi = torch.quantile(phi, 1 - alpha / 100.0)

        vMin = eigvecs[:, 1:3] @ torch.tensor([[torch.cos(minPhi)], [torch.sin(minPhi)]], device=device)
        vMax = eigvecs[:, 1:3] @ torch.tensor([[torch.cos(maxPhi)], [torch.sin(maxPhi)]], device=device)
        HE = torch.stack([vMin[:, 0], vMax[:, 0]], dim=1) if vMin[0] > vMax[0] else torch.stack([vMax[:, 0], vMin[:, 0]], dim=1)
        Y = OD.T
        if torch.isnan(HE).any() or torch.linalg.matrix_rank(HE) < 2:
            return torch.zeros(H * W, 3, device=device)

        C = torch.linalg.lstsq(HE, Y).solution
        maxC = torch.tensor([torch.quantile(C[0], 0.99), torch.quantile(C[1], 0.99)], device=device)
        tmp = maxC / maxCRef
        C2 = C / tmp[:, None]
        Inorm = Io * torch.exp(-HERef @ C2)
        Inorm = torch.clamp(Inorm.T, 0, 255)
        return Inorm

    result = torch.stack([_normalize_single(batch[i]) for i in range(B)], dim=0)
    return result.view(B, H, W, C).byte().cpu()