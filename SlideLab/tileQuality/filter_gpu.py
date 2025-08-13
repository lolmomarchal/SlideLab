import torch
import torch.nn.functional as F

def LaplaceFilter(batch, threshold=0.015):

    # convert batch from B,H,W,C to B,C,H,W and normalize
    batch = batch.permute(0, 3, 1, 2).float()
    r, g, b = batch[:, 0:1], batch[:, 1:2], batch[:, 2:3]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray = gray / 255.0
    sobel_kernel_dx2 = torch.tensor([
        [1., 0., -1.],
        [2., 0., -2.],
        [1., 0., -1.]
    ], dtype=torch.float32, device=gray.device).view(1, 1, 3, 3)

    sobel_kernel_dy2 = torch.tensor([
        [1., 2., 1.],
        [0., 0., 0.],
        [-1., -2., -1.]
    ], dtype=torch.float32, device=gray.device).view(1, 1, 3, 3)
    ddx = F.conv2d(gray, sobel_kernel_dx2, padding=1)
    ddy = F.conv2d(gray, sobel_kernel_dy2, padding=1)

    laplace_img = ddx + ddy
    laplace_img /= 8.0
    B = laplace_img.shape[0]
    laplace_flat = laplace_img.view(B, -1)
    variance = laplace_flat.var(dim=1, unbiased=False)*10

    return (variance <= threshold), variance