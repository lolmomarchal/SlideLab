import torch

# constants
device = "cuda" if torch.cuda.is_available() else "cpu"
_rgb2xyz = torch.tensor([[0.412453, 0.357580, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]], dtype=torch.float32, device=device)

_white = torch.tensor([0.95047, 1., 1.08883], dtype=torch.float32, device=device)
_xyz2rgb = torch.linalg.inv(_rgb2xyz)
target_means = torch.tensor([65.22132, 28.934267, -14.142519], dtype=torch.float32, device=device)
target_stds = torch.tensor([15.800227, 9.263783, 6.0213304], dtype=torch.float32, device=device)

# =========================== Helper Functions ===============================
def csplit(I):

    return [I[i] for i in range(I.shape[0])]

def cmerge(I1, I2, I3):
    return torch.stack([I1, I2, I3], dim=0)

def lab_split(I):
    I = I.type(torch.float32)
    I1, I2, I3 = csplit(I)
    return I1 / 2.55, I2 - 128, I3 - 128

def lab_merge(I1, I2, I3):
    return cmerge(I1 * 2.55, I2 + 128, I3 + 128)
def rgb2lab(rgb):

    arr = rgb.type(torch.float32)

    mask = arr > 0.04045
    not_mask = torch.logical_not(mask)
    arr.masked_scatter_(mask, torch.pow((torch.masked_select(arr, mask) + 0.055) / 1.055, 2.4))
    arr.masked_scatter_(not_mask, torch.masked_select(arr, not_mask) / 12.92)
    xyz = torch.tensordot(torch.t(_rgb2xyz), arr, dims=([0], [0]))
    _white_reshaped = _white.type(xyz.dtype).unsqueeze(dim=-1).unsqueeze(dim=-1)
    arr = torch.mul(xyz, 1 / _white_reshaped)
    mask = arr > 0.008856
    not_mask = torch.logical_not(mask)
    arr.masked_scatter_(mask, torch.pow(torch.masked_select(arr, mask), 1 / 3))
    arr.masked_scatter_(not_mask, 7.787 * torch.masked_select(arr, not_mask) + 16 / 116)

    x, y, z = csplit(arr)
    L = (116. * y) - 16.
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    L *= 2.55
    a += 128
    b += 128

    return torch.stack([L, a, b], dim=0)
def lab2rgb(lab):
    lab = lab.type(torch.float32)
    L, a, b = lab[0] / 2.55, lab[1] - 128, lab[2] - 128
    y = (L + 16.) / 116.
    x = (a / 500.) + y
    z = y - (b / 200.)
    out = torch.stack([x, y, z], dim=0)
    mask = out > 0.2068966
    not_mask = torch.logical_not(mask)
    out.masked_scatter_(mask, torch.pow(torch.masked_select(out, mask), 3))
    out.masked_scatter_(not_mask, (torch.masked_select(out, not_mask) - 16 / 116) / 7.787)
    _white_reshaped = _white.type(out.dtype).unsqueeze(dim=-1).unsqueeze(dim=-1)
    out = torch.mul(out, _white_reshaped)
    rgb_arr = torch.tensordot(out, torch.t(_xyz2rgb).type(out.dtype), dims=([0], [0]))
    rgb_arr = rgb_arr.permute(2, 0, 1)
    mask = rgb_arr > 0.0031308
    not_mask = torch.logical_not(mask)
    rgb_arr.masked_scatter_(mask, 1.055 * torch.pow(torch.masked_select(rgb_arr, mask), 1 / 2.4) - 0.055)
    rgb_arr.masked_scatter_(not_mask, torch.masked_select(rgb_arr, not_mask) * 12.92)
    return torch.clamp(rgb_arr, 0, 1)

# ========================== main method ===========================
def reinhardNormalizer(batch, device = "cuda"):
    batch = batch.permute(0,3,1,2)
    B,C,H,W = map(int, batch.shape)
    def _normalize_single_tile(tile):
        I = tile.type(torch.float32) / 255.0
        lab = rgb2lab(I)
        labs = lab_split(lab)
        mus = torch.stack([torch.mean(x) for x in labs], dim=0)
        stds = torch.stack([torch.std(x) for x in labs], dim=0)
        q = (target_stds[0] - stds[0]) / target_stds[0]
        q = torch.where(q <= 0, torch.tensor(0.05, device=q.device), q)
        l_norm = mus[0] + (labs[0] - mus[0]) * (1 + q)
        a_norm = target_means[1] + (labs[1] - mus[1])
        b_norm = target_means[2] + (labs[2] - mus[2])
        result = [l_norm, a_norm, b_norm]
        lab_normalized = lab_merge(*result)
        rgb_normalized = lab2rgb(lab_normalized)
        return (rgb_normalized * 255).type(torch.uint8)
    result = torch.stack([_normalize_single_tile(batch[i]) for i in range(B)], dim=0)
    result = result.permute(0,2,3,1)
    return result

