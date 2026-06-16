from __future__ import annotations

import cv2
import numpy as np


def synthesize_degradation(
    img_bgr: np.ndarray,
    blur_sigma: float = 3.0,
    downscale: float = 4.0,
    noise_sigma: float = 12.0,
    jpeg_quality: int = 40,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    h, w = img_bgr.shape[:2]
    out = img_bgr.astype(np.float32)

    ksize = int(2 * round(blur_sigma) + 1)
    out = cv2.GaussianBlur(out, (ksize, ksize), blur_sigma)

    small = cv2.resize(
        out, (max(1, int(w / downscale)), max(1, int(h / downscale))),
        interpolation=cv2.INTER_AREA,
    )
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    out = out + rng.normal(0, noise_sigma, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)

    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    if ok:
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def make_comparison_grid(images: list[np.ndarray], titles: list[str], cols: int = 3):
    import math

    import matplotlib.pyplot as plt

    n = len(images)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            ax.set_title(titles[i], fontsize=12)
        ax.axis("off")
    fig.tight_layout()
    return fig


def imread_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def imwrite_bgr(path: str, img_bgr: np.ndarray) -> None:
    cv2.imwrite(path, img_bgr)
