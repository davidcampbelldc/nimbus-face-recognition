"""Bounding-box rendering + h264 video writer.

OpenCV's pip wheel does not bundle an h264 encoder (licensing). We write the
intermediate file with the `mp4v` fourcc (mpeg-4 part 2), then post-encode to
real h264 via ffmpeg. The end result is QuickTime/VLC/Chrome-compatible and
noticeably smaller. ffmpeg is a near-universal dependency on reviewer systems.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .types import Detection

# Distinct, reasonably colourblind-friendly BGR palette. Covers our 5 named
# characters + "Face"/Unknown fallback. Ordering chosen so adjacent
# characters (Harry-Ron) aren't visually confusable.
COLOUR_MAP: dict[str, tuple[int, int, int]] = {
    "Harry":       (60, 76, 231),      # red
    "Ron":         (0, 140, 255),      # orange
    "Hermione":    (180, 119, 200),    # mauve
    "McGonagall":  (113, 204, 46),     # green
    "Snape":       (80, 40, 40),       # dark grey-blue
    "Unknown":     (128, 128, 128),    # grey
    "Face":        (220, 220, 220),    # off-white (detection-only mode)
}
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 2


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw labelled boxes. Modifies `frame` in-place AND returns it."""
    for det in detections:
        x, y, w, h = det.bbox
        colour = COLOUR_MAP.get(det.label, COLOUR_MAP["Face"])
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, BOX_THICKNESS)

        # Label string
        if det.label_confidence is not None:
            text = f"{det.label} {det.label_confidence:.2f}"
        else:
            text = f"{det.label} {det.confidence:.2f}"

        (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
        # Label background (filled) above the box for readability.
        label_y_top = max(0, y - th - baseline - 4)
        cv2.rectangle(frame, (x, label_y_top), (x + tw + 4, y), colour, -1)
        cv2.putText(
            frame,
            text,
            (x + 2, y - baseline - 2),
            FONT,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS,
            lineType=cv2.LINE_AA,
        )

    return frame


class VideoWriter:
    """Two-stage video writer: cv2 (mp4v) intermediate → ffmpeg h264 final.

    On __exit__ (or explicit release()), the intermediate is re-encoded via
    ffmpeg to h264/yuv420p and the intermediate is deleted. If ffmpeg is not
    available, the mp4v file is kept at the target path (warning logged).
    """

    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        self.path = path
        self.fps = fps
        path.parent.mkdir(parents=True, exist_ok=True)

        # Intermediate gets a `.tmp.mp4` suffix next to the target.
        self._tmp_path = path.with_suffix(".tmp.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        self._writer = cv2.VideoWriter(str(self._tmp_path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter could not open {self._tmp_path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()
        self._reencode_via_ffmpeg()

    def _reencode_via_ffmpeg(self) -> None:
        if not self._tmp_path.exists():
            return
        if shutil.which("ffmpeg") is None:
            # Keep intermediate at target path with a warning.
            self._tmp_path.rename(self.path)
            print(
                "warning: ffmpeg not found — output is mpeg-4 part 2 (mp4v), "
                "not h264. QuickTime playback may require VLC.",
            )
            return

        result = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(self._tmp_path),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(self.path),
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"ffmpeg re-encode failed: {result.stderr.decode()}")
            self._tmp_path.rename(self.path)
            return
        self._tmp_path.unlink()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
