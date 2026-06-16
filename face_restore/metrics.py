from __future__ import annotations

import cv2
import numpy as np


def psnr(pred_bgr: np.ndarray, gt_bgr: np.ndarray) -> float:
    pred = pred_bgr.astype(np.float64)
    gt = gt_bgr.astype(np.float64)
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0) - 10 * np.log10(mse))


def ssim(pred_bgr: np.ndarray, gt_bgr: np.ndarray) -> float:
    pg = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2GRAY)
    gg = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY)
    try:
        from skimage.metrics import structural_similarity as _ssim
        return float(_ssim(pg, gg, data_range=255))
    except Exception:
        return _ssim_manual(pg.astype(np.float64), gg.astype(np.float64))


def _ssim_manual(x: np.ndarray, y: np.ndarray) -> float:
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mux, muy = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mux) * (y - muy)).mean()
    return float(
        ((2 * mux * muy + C1) * (2 * cov + C2))
        / ((mux**2 + muy**2 + C1) * (vx + vy + C2))
    )


class LPIPS:
    def __init__(self, net: str = "alex", device: str = "cuda"):
        import lpips as _lpips
        import torch

        self.torch = torch
        self.device = device
        self.model = _lpips.LPIPS(net=net).to(device).eval()

    def __call__(self, pred_bgr: np.ndarray, gt_bgr: np.ndarray) -> float:
        torch = self.torch

        def to_t(img):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
            return (t * 2 - 1).to(self.device)

        with torch.no_grad():
            d = self.model(to_t(pred_bgr), to_t(gt_bgr))
        return float(d.item())


def sharpness(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def full_reference_report(pred_bgr, gt_bgr, lpips_fn: LPIPS | None = None) -> dict:
    rep = {"psnr": psnr(pred_bgr, gt_bgr), "ssim": ssim(pred_bgr, gt_bgr)}
    if lpips_fn is not None:
        rep["lpips"] = lpips_fn(pred_bgr, gt_bgr)
    return rep
