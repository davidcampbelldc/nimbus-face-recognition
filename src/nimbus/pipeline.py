"""Main pipeline: read → detect (+ scene-cut) → render → write."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

from .detector import FaceDetector
from .renderer import VideoWriter, draw_detections
from .scene_cut import SceneCutDetector
from .types import FrameResult


@dataclass
class PipelineStats:
    frames_processed: int
    total_detections: int
    scene_cuts: int
    runtime_seconds: float

    @property
    def fps(self) -> float:
        return self.frames_processed / self.runtime_seconds if self.runtime_seconds > 0 else 0.0


def run(
    video_in: Path,
    video_out: Path,
    frame_limit: int | None = None,
    show_progress: bool = True,
) -> PipelineStats:
    """Process one video: detect faces per frame, render boxes, write output.

    Args:
        video_in: path to input mp4.
        video_out: path to output mp4 (parent dir auto-created).
        frame_limit: if set, process only the first N frames (smoke mode).
        show_progress: render a tqdm progress bar.

    Returns PipelineStats (frames, detections, cuts, runtime).
    """
    import time

    if not video_in.exists():
        raise FileNotFoundError(f"input video not found: {video_in}")

    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_limit is not None:
        total_frames = min(total_frames, frame_limit)

    detector = FaceDetector()
    scene_cut = SceneCutDetector()

    total_detections = 0
    scene_cuts = 0
    t_start = time.monotonic()

    iterator = range(total_frames)
    if show_progress:
        iterator = tqdm(iterator, desc="Processing", unit="frame")

    with VideoWriter(video_out, fps, width, height) as writer:
        for frame_idx in iterator:
            ok, frame = cap.read()
            if not ok:
                break

            cut = scene_cut.is_cut(frame)
            if cut:
                scene_cuts += 1

            detections = detector.detect(frame)
            total_detections += len(detections)

            _ = FrameResult(frame_idx=frame_idx, detections=detections, scene_cut=cut)
            annotated = draw_detections(frame, detections)
            writer.write(annotated)

    cap.release()
    runtime = time.monotonic() - t_start
    return PipelineStats(
        frames_processed=total_frames,
        total_detections=total_detections,
        scene_cuts=scene_cuts,
        runtime_seconds=runtime,
    )
