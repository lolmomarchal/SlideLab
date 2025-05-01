import os
import sys
import subprocess
import torch
import torch.nn as nn
import numpy as np

class StainNet(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        saved_path = self.install_requirements()
        sys.path.append(saved_path)

        from models import StainNet as StainNetModel
        self.model = StainNetModel().to(self.device)

        weights_path = os.path.join(
            sys.prefix,
            "models/StainNet/checkpoints/aligned_histopathology_dataset/StainNet-Public_layer3_ch32.pth"
        )
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def install_requirements(self):
        env_directory = sys.prefix
        model_dir = os.path.join(env_directory, "models/StainNet")
        os.makedirs(model_dir, exist_ok=True)

        if len(os.listdir(model_dir)) > 1:
            return model_dir

        print("Cloning StainNet repository...")
        subprocess.run(
            f"git clone https://github.com/khtao/StainNet.git {model_dir}",
            shell=True,
            check=True
        )
        return model_dir
    # can deal with batches
    def forward(self, x):
        x = x.to(self.device)
        if x.dtype != torch.float32:
            x = x.float()

        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)

        x = ((x / 255.0) - 0.5) / 0.5

        with torch.inference_mode():
            out = self.model(x)
            out = ((out * 0.5 + 0.5) * 255).clamp(0, 255).byte()
        return out.permute(0, 2, 3, 1).cpu().numpy()
