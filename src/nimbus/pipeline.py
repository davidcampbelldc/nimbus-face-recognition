"""Main pipeline: read → detect (+ scene-cut) → embed → recognise → render → write."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
from tqdm import tqdm

from .detector import FaceDetector
from .embedder import Embedder
from .recogniser import Recogniser
from .renderer import VideoWriter, draw_detections
from .scene_cut import SceneCutDetector
from .tracker import Tracker
from .types import Detection, FrameResult

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDINGS = REPO_ROOT / "refs" / "embeddings.npz"
DEFAULT_CALIBRATION = REPO_ROOT / "refs" / "calibration.json"


@dataclass
class PipelineStats:
    frames_processed: int
    total_detections: int
    scene_cuts: int
    runtime_seconds: float

    @property
    def fps(self) -> float:
        return self.frames_processed / self.runtime_seconds if self.runtime_seconds > 0 else 0.0


def _recognise_detections(
    detections: list[Detection],
    embedder: Embedder,
    recogniser: Recogniser,
) -> list[Detection]:
    """Embed each detection's aligned face and assign a character label."""
    labelled: list[Detection] = []
    for det in detections:
        if det.aligned_face is None:
            labelled.append(det)
            continue
        try:
            emb = embedder.embed_aligned_face(det.aligned_face)
            result = recogniser.recognise(emb)
            labelled.append(replace(
                det,
                label=result.label,
                label_confidence=result.confidence,
            ))
        except Exception as e:
            # Embedding failure on a pathological crop — keep the detection,
            # mark as Unknown. Pipeline must not die on one bad face.
            print(f"  warning: recognition skipped for a detection: {e}")
            labelled.append(replace(det, label="Unknown", label_confidence=0.0))
    return labelled


def run(
    video_in: Path,
    video_out: Path,
    frame_limit: int | None = None,
    show_progress: bool = True,
    recognise: bool = True,
    track: bool = True,
    embeddings_path: Path | None = None,
    calibration_path: Path | None = None,
) -> PipelineStats:
    """Process one video: detect → (recognise) → (track) → render → write.

    Args:
        video_in: path to input mp4.
        video_out: path to output mp4 (parent dir auto-created).
        frame_limit: if set, process only the first N frames (smoke mode).
        show_progress: render a tqdm progress bar.
        recognise: when True, embed each detected face and label it with the
            most likely character (or "Unknown"). Falls back to detection-only
            output if the refs/calibration artefacts are missing.
        track: when True, smooth labels via the IoU tracker. Scene cuts flush
            tracks to avoid label bleed across shots.
        embeddings_path: override for refs/embeddings.npz.
        calibration_path: override for refs/calibration.json.
    """
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

    embedder: Embedder | None = None
    recogniser: Recogniser | None = None
    if recognise:
        try:
            embedder = Embedder()
            recogniser = Recogniser(
                embeddings_path or DEFAULT_EMBEDDINGS,
                calibration_path or DEFAULT_CALIBRATION,
            )
        except FileNotFoundError as e:
            print(f"warning: recognition disabled — {e}")
            embedder = None
            recogniser = None

    tracker: Tracker | None = Tracker() if track else None

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

            if embedder is not None and recogniser is not None:
                detections = _recognise_detections(detections, embedder, recogniser)

            if tracker is not None:
                detections = tracker.update(detections, scene_cut=cut)

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
