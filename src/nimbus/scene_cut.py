"""Scene-cut detector via mean absolute pixel difference (MABS).

A hard cut produces a step change in the pixel statistics between consecutive
frames. Downsampling to 1/4 size greyscale + MABS is cheap (sub-millisecond
per frame at 1080p) and sufficient for shot boundaries in a live-action clip
with standard editing.

Used by the tracker (Phase 3) to flush track IDs on cut — without this,
label-history hysteresis bleeds across scenes and contaminates the output.
"""

from __future__ import annotations

import cv2
import numpy as np


class SceneCutDetector:
    """Flags scene cuts by thresholding MABS between consecutive frames.

    Default threshold of 25 (on 0-255 pixel scale) is tuned empirically on
    standard film editing and was confirmed reasonable on the nimbus clip.
    NOTES.md documents the calibration; `scripts/plot_distributions.py` can
    re-tune if needed.
    """

    def __init__(self, threshold: float = 25.0, downsample: int = 4) -> None:
        self.threshold = threshold
        self.downsample = downsample
        self._prev_gray: np.ndarray | None = None

    def is_cut(self, frame_bgr: np.ndarray) -> bool:
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(
            frame_bgr,
            (w // self.downsample, h // self.downsample),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return False

        mabs = float(np.mean(np.abs(gray.astype(np.int16) - self._prev_gray.astype(np.int16))))
        self._prev_gray = gray
        return mabs > self.threshold

    def reset(self) -> None:
        """Clear history (e.g., when starting a fresh video)."""
        self._prev_gray = None
