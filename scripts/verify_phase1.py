#!/usr/bin/env python3
"""Phase 1 verification: confirm detector fires on face-bearing frames.

Samples frames across the first N frames of nimbus.mp4, runs the detector,
prints per-frame detection counts, and saves one annotated frame to
data/output/verify_phase1_frame.png for visual inspection.

Run from repo root:
    .venv/bin/python scripts/verify_phase1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nimbus.detector import FaceDetector
from nimbus.renderer import draw_detections


def main() -> int:
    video = Path("data/input/nimbus.mp4")
    out_frame = Path("data/output/verify_phase1_frame.png")
    sample_indices = [0, 150, 300, 450, 600, 750, 900, 1200, 1500, 2000]

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"error: cannot open {video}", file=sys.stderr)
        return 2

    detector = FaceDetector()
    best_frame_idx = -1
    best_count = 0
    best_annotated = None

    print(f"{'frame':>6}  {'faces':>5}  note")
    print("-" * 40)
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"{idx:>6}  {'--':>5}  read failed")
            continue
        detections = detector.detect(frame)
        n = len(detections)
        note = ", ".join(f"{d.confidence:.2f}" for d in detections) if n else "—"
        print(f"{idx:>6}  {n:>5}  {note}")
        if n > best_count:
            best_count = n
            best_frame_idx = idx
            best_annotated = draw_detections(frame.copy(), detections)

    cap.release()

    if best_annotated is not None:
        out_frame.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_frame), best_annotated)
        print(f"\nBest: frame {best_frame_idx} with {best_count} faces "
              f"→ {out_frame}")
        return 0
    print("\nNo detections across sampled frames — check detector wiring.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
