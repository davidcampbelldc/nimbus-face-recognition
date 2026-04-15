"""Shared dataclasses and type aliases used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np

# (x, y, w, h) in pixels, top-left origin.
Bbox: TypeAlias = tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    """One face found by the detector in one frame."""

    bbox: Bbox
    confidence: float  # detector confidence [0, 1]
    label: str = "Face"                     # overwritten by recogniser in later phases
    label_confidence: float | None = None   # recogniser confidence; None in detection-only mode
    # BGR aligned face crop; set when the detector runs with align=True.
    aligned_face: np.ndarray | None = None


@dataclass
class FrameResult:
    """Everything the pipeline produces for a single frame."""

    frame_idx: int
    detections: list[Detection] = field(default_factory=list)
    scene_cut: bool = False
