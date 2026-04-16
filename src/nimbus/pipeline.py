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
from .types import LABEL_UNKNOWN, Detection

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


def _rescale_bbox(det: Detection, scale: float) -> Detection:
    """Scale a detection's bbox from a downsampled frame back to original
    coords. `scale` is the ratio applied to the frame before detection
    (e.g. 0.5 = half-size); we multiply bboxes by 1/scale."""
    x, y, w, h = det.bbox
    inv = 1.0 / scale
    return replace(
        det,
        bbox=(int(round(x * inv)), int(round(y * inv)),
              int(round(w * inv)), int(round(h * inv))),
    )


def _detect(frame, detector: FaceDetector, scale: float | None) -> list[Detection]:
    """Run the detector; optionally downsample first and scale bboxes back.

    Output video stays at original resolution regardless of `scale` — the
    frame passed to the renderer is the untouched original.
    """
    if scale is None:
        return detector.detect(frame)
    h, w = frame.shape[:2]
    small = cv2.resize(
        frame,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return [_rescale_bbox(d, scale) for d in detector.detect(small)]


def _init_recogniser(
    recognise: bool,
    embeddings_path: Path,
    calibration_path: Path,
) -> tuple[Embedder | None, Recogniser | None]:
    """Build the embedder + recogniser, or return (None, None) if disabled
    or refs are missing. Missing-refs is a soft failure: we log and fall back
    to detection-only output so smoke runs work on a fresh clone."""
    if not recognise:
        return None, None
    try:
        return Embedder(), Recogniser(embeddings_path, calibration_path)
    except FileNotFoundError as e:
        print(f"warning: recognition disabled — {e}")
        return None, None


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
            # DeepFace raises a heterogeneous set (ValueError, RuntimeError,
            # cv2.error, model-specific asserts) on pathological crops. A 3044-
            # frame render must not die on one bad face — log and continue.
            print(f"  warning: recognition skipped for a detection: {e}")
            labelled.append(replace(det, label=LABEL_UNKNOWN, label_confidence=0.0))
    return labelled


def run(
    video_in: Path,
    video_out: Path,
    frame_limit: int | None = None,
    show_progress: bool = True,
    recognise: bool = True,
    track: bool = True,
    downsample: int | None = None,
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
        downsample: if set, resize the frame to this short-side (in px) before
            running detection + alignment, then scale bboxes back to the
            original frame for rendering. Speeds up detection at the cost of
            small-face recall + a small embedding-quality delta (alignment
            happens at the lower res). Output video stays at original res.
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

    embedder, recogniser = _init_recogniser(
        recognise,
        embeddings_path or DEFAULT_EMBEDDINGS,
        calibration_path or DEFAULT_CALIBRATION,
    )
    tracker: Tracker | None = Tracker() if track else None

    total_detections = 0
    scene_cuts = 0
    t_start = time.monotonic()

    iterator = range(total_frames)
    if show_progress:
        iterator = tqdm(iterator, desc="Processing", unit="frame")

    # Resize so short side = `downsample` px (aspect preserved). None = native.
    scale: float | None = None
    if downsample is not None and downsample > 0 and downsample < min(width, height):
        scale = downsample / min(width, height)

    with VideoWriter(video_out, fps, width, height) as writer:
        for frame_idx in iterator:
            ok, frame = cap.read()
            if not ok:
                break

            cut = scene_cut.is_cut(frame)
            if cut:
                scene_cuts += 1

            detections = _detect(frame, detector, scale)
            total_detections += len(detections)

            if embedder is not None and recogniser is not None:
                detections = _recognise_detections(detections, embedder, recogniser)

            if tracker is not None:
                detections = tracker.update(detections, scene_cut=cut)

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
