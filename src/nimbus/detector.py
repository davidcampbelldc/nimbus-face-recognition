"""RetinaFace face detector (via DeepFace)."""

from __future__ import annotations

import numpy as np

from .types import Detection


class FaceDetector:
    """Detects faces using DeepFace's RetinaFace backend.

    DeepFace.extract_faces with enforce_detection=False returns a synthetic
    full-image entry (confidence 0) when no face is found. We filter those
    so callers only see real detections.
    """

    def __init__(self, min_confidence: float = 0.5) -> None:
        self.min_confidence = min_confidence
        self._deepface = None  # lazy-loaded to keep import cheap for tests

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._deepface is None:
            from deepface import DeepFace
            self._deepface = DeepFace

        raw = self._deepface.extract_faces(
            img_path=frame,
            detector_backend="retinaface",
            enforce_detection=False,
            align=False,
        )

        results: list[Detection] = []
        for entry in raw:
            conf = float(entry.get("confidence", 0.0))
            if conf < self.min_confidence:
                continue

            fa = entry.get("facial_area", {})
            x = int(fa.get("x", 0))
            y = int(fa.get("y", 0))
            w = int(fa.get("w", 0))
            h = int(fa.get("h", 0))
            if w <= 0 or h <= 0:
                continue

            results.append(Detection(bbox=(x, y, w, h), confidence=conf))

        return results
