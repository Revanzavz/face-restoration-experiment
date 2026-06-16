from __future__ import annotations

import cv2
import numpy as np


def _soft_mask(mask: np.ndarray, erode_px: int = 6, blur_px: int = 15) -> np.ndarray:
    m = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
    if erode_px > 0:
        k = np.ones((erode_px, erode_px), np.uint8)
        m = cv2.erode(m, k, iterations=1)
    if blur_px > 0:
        if blur_px % 2 == 0:
            blur_px += 1
        m = cv2.GaussianBlur(m, (blur_px, blur_px), 0)
    return (m.astype(np.float32) / 255.0)


def feather_blend(
    base_bgr: np.ndarray,
    warped_face_bgr: np.ndarray,
    warped_mask: np.ndarray,
    erode_px: int = 6,
    blur_px: int = 15,
) -> np.ndarray:
    soft = _soft_mask(warped_mask, erode_px, blur_px)[..., None]
    base = base_bgr.astype(np.float32)
    face = warped_face_bgr.astype(np.float32)
    out = soft * face + (1.0 - soft) * base
    return np.clip(out, 0, 255).astype(np.uint8)


def poisson_blend(
    base_bgr: np.ndarray,
    warped_face_bgr: np.ndarray,
    warped_mask: np.ndarray,
    erode_px: int = 3,
) -> np.ndarray:
    m = (np.clip(warped_mask, 0.0, 1.0) * 255).astype(np.uint8)
    if erode_px > 0:
        m = cv2.erode(m, np.ones((erode_px, erode_px), np.uint8), iterations=1)

    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return base_bgr.copy()

    cx = int((xs.min() + xs.max()) / 2)
    cy = int((ys.min() + ys.max()) / 2)
    center = (cx, cy)

    try:
        out = cv2.seamlessClone(
            warped_face_bgr, base_bgr, m, center, cv2.NORMAL_CLONE
        )
    except cv2.error:
        out = feather_blend(base_bgr, warped_face_bgr, warped_mask)
    return out


def blend(
    base_bgr: np.ndarray,
    warped_face_bgr: np.ndarray,
    warped_mask: np.ndarray,
    method: str = "feather",
    **kwargs,
) -> np.ndarray:
    if method == "feather":
        return feather_blend(base_bgr, warped_face_bgr, warped_mask, **kwargs)
    if method == "poisson":
        return poisson_blend(base_bgr, warped_face_bgr, warped_mask, **kwargs)
    raise ValueError(f"Metode blending tidak dikenal: {method}")
