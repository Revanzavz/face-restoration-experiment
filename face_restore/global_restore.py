from __future__ import annotations

import numpy as np
from PIL import Image


def pad_to_multiple(image: Image.Image, multiple: int = 16):
    w, h = image.size
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_h = ((h + multiple - 1) // multiple) * multiple
    padded = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    padded.paste(image, (0, 0))
    return padded, w, h


class GlobalRestorer:
    def __init__(self, model_path: str, unet_factory=None, device: str = "cuda"):
        import torch

        self.device = device
        if unet_factory is None:
            from model import UNet
            unet_factory = UNet

        self.model = unet_factory().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def restore_pil(self, image: Image.Image) -> Image.Image:
        import torch
        import torchvision.transforms as transforms

        padded, ow, oh = pad_to_multiple(image)
        tensor = transforms.ToTensor()(padded).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(tensor)
        pred = pred.squeeze(0).cpu().permute(1, 2, 0).numpy()
        pred = np.clip(pred, 0, 1)[:oh, :ow]
        return Image.fromarray((pred * 255).astype(np.uint8))

    def restore_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        import cv2

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        out = self.restore_pil(Image.fromarray(rgb))
        return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
