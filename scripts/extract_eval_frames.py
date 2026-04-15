#!/usr/bin/env python3
"""Phase 4 step 1: stratified 40-frame eval sampler.

Evenly samples N frames across the video, runs the detector on each, and
writes three artefacts under eval/:

  - frames_raw/fXXXXX.png        raw frame, no overlay (archival)
  - frames_labelling/fXXXXX.png  same frame with NUMBERED bbox overlays
                                 (B0, B1, ...). No character labels shown —
                                 keeps labellers blind to the recogniser
                                 to avoid anchoring the ground truth on the
                                 thing under test.
  - detections.json              scaffold: {frame_idx: [{box_id, bbox,
                                 det_conf}]} — ground-truth labels get
                                 slotted in by a separate labelling pass.

Usage:
    .venv/bin/python scripts/extract_eval_frames.py
    .venv/bin/python scripts/extract_eval_frames.py --n 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from nimbus.detector import FaceDetector  # noqa: E402

VIDEO = REPO_ROOT / "data" / "input" / "nimbus.mp4"
EVAL = REPO_ROOT / "eval"
RAW_DIR = EVAL / "frames_raw"
LBL_DIR = EVAL / "frames_labelling"
OUT_JSON = EVAL / "detections.json"

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _annotate_for_labelling(frame: np.ndarray, dets: list[dict]) -> np.ndarray:
    """Draw numbered box IDs (B0, B1, ...) with yellow boxes so a human
    labeller can refer to each face by a stable identifier. No character
    labels rendered — labelling must be anchoring-free."""
    annotated = frame.copy()
    for det in dets:
        x, y, w, h = det["bbox"]
        colour = (0, 255, 255)  # yellow BGR
        cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 2)
        text = det["box_id"]
        (tw, th), baseline = cv2.getTextSize(text, FONT, 0.7, 2)
        label_y_top = max(0, y - th - baseline - 4)
        cv2.rectangle(
            annotated, (x, label_y_top), (x + tw + 6, y), colour, -1,
        )
        cv2.putText(
            annotated, text, (x + 2, y - baseline - 2),
            FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA,
        )
    return annotated


def _sample_frame_indices(total_frames: int, n: int) -> list[int]:
    """n evenly-spaced frame indices over [0, total_frames - 1]."""
    return [int(round(x)) for x in np.linspace(0, total_frames - 1, n)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=40, help="number of frames to sample")
    args = p.parse_args()

    if not VIDEO.exists():
        print(f"error: {VIDEO} not found", file=sys.stderr)
        return 2

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"error: could not open {VIDEO}", file=sys.stderr)
        return 2

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _sample_frame_indices(total_frames, args.n)

    detector = FaceDetector()
    scaffold: dict[str, list[dict]] = {}

    print(f"sampling {args.n} frames evenly across {total_frames} total frames")
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"  warning: could not read frame {frame_idx}, skipping")
            continue

        dets = detector.detect(frame)
        det_records = [
            {
                "box_id": f"B{i}",
                "bbox": list(d.bbox),
                "det_conf": round(float(d.confidence), 3),
            }
            for i, d in enumerate(dets)
        ]
        scaffold[str(frame_idx)] = det_records

        raw_path = RAW_DIR / f"f{frame_idx:05d}.png"
        lbl_path = LBL_DIR / f"f{frame_idx:05d}.png"
        cv2.imwrite(str(raw_path), frame)
        annotated = _annotate_for_labelling(frame, det_records)
        cv2.imwrite(str(lbl_path), annotated)

        print(f"  f{frame_idx:05d}: {len(det_records)} detections")

    cap.release()
    OUT_JSON.write_text(json.dumps(scaffold, indent=2))

    n_faces = sum(len(v) for v in scaffold.values())
    print()
    print(f"=== extracted {len(scaffold)} frames, {n_faces} face detections total ===")
    print(f"  raw frames:       {RAW_DIR}")
    print(f"  labelling frames: {LBL_DIR}")
    print(f"  scaffold JSON:    {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
