from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .aligner import FaceAligner
from .blending import blend
from .detector import DetectedFace, RetinaFaceDetector


@dataclass
class FaceArtifact:
    bbox: np.ndarray
    score: float
    aligned_input: np.ndarray
    aligned_restored: np.ndarray
    parsing_mask: np.ndarray | None = None


@dataclass
class RestorationResult:
    original: np.ndarray
    global_only: np.ndarray
    region_aware: np.ndarray
    faces: list[FaceArtifact] = field(default_factory=list)

    @property
    def num_faces(self) -> int:
        return len(self.faces)


class RegionAwareRestorer:
    def __init__(
        self,
        global_restorer,
        face_restorer,
        detector: RetinaFaceDetector | None = None,
        parser=None,
        face_source: str = "global",
        blend_method: str = "feather",
        min_face_size: int = 32,
        device: str = "cuda",
    ):
        self.global_restorer = global_restorer
        self.face_restorer = face_restorer
        self.detector = detector or RetinaFaceDetector(device=device)
        self.parser = parser
        self.aligner = FaceAligner(face_size=512)
        self.face_source = face_source
        self.blend_method = blend_method
        self.min_face_size = min_face_size

    def restore(self, image_bgr: np.ndarray) -> RestorationResult:
        original = image_bgr.copy()

        if self.global_restorer is not None:
            global_only = self.global_restorer.restore_bgr(original)
        else:
            global_only = original.copy()

        source = global_only if self.face_source == "global" else original

        faces = self.detector.detect(source)

        result = RestorationResult(
            original=original,
            global_only=global_only,
            region_aware=global_only.copy(),
        )

        for face in faces:
            x1, y1, x2, y2 = face.bbox
            if min(x2 - x1, y2 - y1) < self.min_face_size:
                continue

            aligned = self.aligner.align(source, face.landmarks)
            restored_crop = self.face_restorer.restore(aligned.crop)

            parsing_mask = None
            if self.parser is not None:
                parsing_mask = self.parser.get_mask(restored_crop)
                restored_for_warp = restored_crop
                h, w = result.region_aware.shape[:2]
                warped_face = cv2.warpAffine(
                    restored_for_warp, aligned.inverse_affine, (w, h)
                )
                warped_mask = cv2.warpAffine(
                    parsing_mask, aligned.inverse_affine, (w, h)
                )
                warped_mask = np.clip(warped_mask, 0.0, 1.0)
            else:
                warped_face, warped_mask = self.aligner.paste_back(
                    result.region_aware, restored_crop, aligned.inverse_affine
                )

            result.region_aware = blend(
                result.region_aware, warped_face, warped_mask, method=self.blend_method
            )

            result.faces.append(
                FaceArtifact(
                    bbox=face.bbox,
                    score=face.score,
                    aligned_input=aligned.crop,
                    aligned_restored=restored_crop,
                    parsing_mask=parsing_mask,
                )
            )

        return result
