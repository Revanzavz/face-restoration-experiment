from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectedFace:
    bbox: np.ndarray
    score: float
    landmarks: np.ndarray

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


class RetinaFaceDetector:
    def __init__(
        self,
        model_name: str = "retinaface_resnet50",
        conf_threshold: float = 0.8,
        device: str = "cuda",
    ):
        from facexlib.detection import init_detection_model

        self.conf_threshold = conf_threshold
        self.device = device
        self.model_name = model_name
        self.det_net = init_detection_model(
            model_name, half=False, device=device
        )

    @staticmethod
    def _to_bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        import torch

        img = self._to_bgr(image_bgr)
        with torch.no_grad():
            bboxes = self.det_net.detect_faces(
                img, conf_threshold=self.conf_threshold
            )

        faces: list[DetectedFace] = []
        if bboxes is None or len(bboxes) == 0:
            return faces

        for row in bboxes:
            score = float(row[4])
            if score < self.conf_threshold:
                continue
            landmarks = np.array(row[5:15], dtype=np.float32).reshape(5, 2)
            faces.append(
                DetectedFace(
                    bbox=np.array(row[0:4], dtype=np.float32),
                    score=score,
                    landmarks=landmarks,
                )
            )

        faces.sort(key=lambda f: f.area, reverse=True)
        return faces


class MTCNNDetector:
    def __init__(self, conf_threshold: float = 0.9, device: str = "cuda"):
        from facenet_pytorch import MTCNN

        self.device = device
        self.conf_threshold = conf_threshold
        self.mtcnn = MTCNN(keep_all=True, device=device)

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        import cv2

        rgb = cv2.cvtColor(
            RetinaFaceDetector._to_bgr(image_bgr), cv2.COLOR_BGR2RGB
        )
        boxes, probs, landmarks = self.mtcnn.detect(rgb, landmarks=True)

        faces: list[DetectedFace] = []
        if boxes is None:
            return faces
        for box, prob, lm in zip(boxes, probs, landmarks):
            if prob is None or prob < self.conf_threshold:
                continue
            faces.append(
                DetectedFace(
                    bbox=np.array(box, dtype=np.float32),
                    score=float(prob),
                    landmarks=np.array(lm, dtype=np.float32).reshape(5, 2),
                )
            )
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces
