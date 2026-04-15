"""Facenet512 face-embedding wrapper (via DeepFace).

Produces L2-normalised 512-d float32 vectors. Exposes two modes:

  - `embed_image_path(path)` — reads a file, detects + aligns + embeds.
    Used by offline ref-building (scripts/build_references.py).
  - `embed_aligned_face(bgr)` — skips detection, embeds an already-aligned
    crop. Used at inference via the aligned face attached to each Detection
    by the detector.

Cosine distance helpers are here too so the recogniser has one import.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"


def l2_normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Inputs must be L2-normalised. Returns distance in [0, 2]."""
    return float(1.0 - np.dot(a, b))


class Embedder:
    """Thin wrapper around DeepFace.represent with lazy import."""

    def __init__(self) -> None:
        self._deepface = None

    def _load(self):
        if self._deepface is None:
            from deepface import DeepFace
            self._deepface = DeepFace
        return self._deepface

    def embed_image_path(self, jpg_path: Path) -> np.ndarray:
        """Detect + align + embed a single reference image from disk."""
        deepface = self._load()
        img = cv2.imread(str(jpg_path))
        if img is None:
            raise RuntimeError(f"cv2 failed to read {jpg_path}")
        result = deepface.represent(
            img_path=img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
        )
        if not result:
            raise RuntimeError(f"no embedding for {jpg_path}")
        vec = np.asarray(result[0]["embedding"], dtype=np.float32)
        return l2_normalise(vec)

    def embed_aligned_face(self, face_bgr: np.ndarray) -> np.ndarray:
        """Embed a pre-aligned face crop (skips detection).

        Use `detector_backend="skip"` so DeepFace doesn't re-detect on a crop
        that's already a tight face. `enforce_detection=False` guards against
        the "skip" backend still raising on unusual shapes.
        """
        deepface = self._load()
        result = deepface.represent(
            img_path=face_bgr,
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False,
            align=False,
        )
        if not result:
            raise RuntimeError("embedder returned no result for aligned face")
        vec = np.asarray(result[0]["embedding"], dtype=np.float32)
        return l2_normalise(vec)
