from __future__ import annotations

import cv2
import numpy as np

DEFAULT_FACE_LABELS = (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)


class FaceParser:
    def __init__(self, device: str = "cuda", face_labels=DEFAULT_FACE_LABELS):
        from facexlib.parsing import init_parsing_model

        self.device = device
        self.face_labels = set(face_labels)
        self.net = init_parsing_model(model_name="bisenet", device=device)
        self.net.eval()

    def get_mask(self, aligned_face_bgr: np.ndarray, feather_px: int = 11) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        h, w = aligned_face_bgr.shape[:2]
        rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (rgb - mean) / std
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            inp = F.interpolate(tensor, size=(512, 512), mode="bilinear", align_corners=False)
            out = self.net(inp)[0]
            labels = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        if labels.shape != (h, w):
            labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)

        mask = np.isin(labels, list(self.face_labels)).astype(np.float32)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )
        if feather_px > 0:
            if feather_px % 2 == 0:
                feather_px += 1
            mask = cv2.GaussianBlur(mask, (feather_px, feather_px), 0)
        return np.clip(mask, 0.0, 1.0)
