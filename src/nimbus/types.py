"""Shared dataclasses and type aliases used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

# (x, y, w, h) in pixels, top-left origin.
Bbox: TypeAlias = tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    """One face found by the detector in one frame."""

    bbox: Bbox
    confidence: float  # detector confidence [0, 1]
    label: str = "Face"                     # overwritten by recogniser in later phases
    label_confidence: float | None = None   # recogniser confidence; None in detection-only mode


@dataclass
class FrameResult:
    """Everything the pipeline produces for a single frame."""

    frame_idx: int
    detections: list[Detection] = field(default_factory=list)
    scene_cut: bool = False
