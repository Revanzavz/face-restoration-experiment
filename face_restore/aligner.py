from __future__ import annotations

import cv2
import numpy as np

FFHQ_512_TEMPLATE = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19360],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)

FACE_SIZE = 512


class AlignedFace:
    def __init__(self, crop: np.ndarray, affine: np.ndarray, inverse_affine: np.ndarray):
        self.crop = crop
        self.affine = affine
        self.inverse_affine = inverse_affine


class FaceAligner:
    def __init__(self, face_size: int = FACE_SIZE, template: np.ndarray | None = None):
        self.face_size = face_size
        base = FFHQ_512_TEMPLATE if template is None else template
        self.template = base * (face_size / 512.0)

    def align(self, image_bgr: np.ndarray, landmarks: np.ndarray) -> AlignedFace:
        src = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
        affine, _ = cv2.estimateAffinePartial2D(
            src, self.template, method=cv2.LMEDS
        )
        if affine is None:
            raise ValueError("Gagal mengestimasi transform alignment dari landmark.")

        crop = cv2.warpAffine(
            image_bgr,
            affine,
            (self.face_size, self.face_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(135, 133, 132),
        )
        inverse_affine = cv2.invertAffineTransform(affine)
        return AlignedFace(crop=crop, affine=affine, inverse_affine=inverse_affine)

    def paste_back(
        self,
        base_image_bgr: np.ndarray,
        restored_face: np.ndarray,
        inverse_affine: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = base_image_bgr.shape[:2]
        warped_face = cv2.warpAffine(
            restored_face, inverse_affine, (w, h), flags=cv2.INTER_LINEAR
        )
        mask = np.ones((self.face_size, self.face_size), dtype=np.float32)
        warped_mask = cv2.warpAffine(mask, inverse_affine, (w, h), flags=cv2.INTER_LINEAR)
        warped_mask = np.clip(warped_mask, 0.0, 1.0)
        return warped_face, warped_mask
