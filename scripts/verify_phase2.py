#!/usr/bin/env python3
"""Phase 2 verification: run the full detect → embed → recognise stack on
a handful of sampled frames from nimbus.mp4 and save the best annotated
frame for visual inspection.

Run from repo root:
    .venv/bin/python scripts/verify_phase2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dataclasses import replace  # noqa: E402

from nimbus.detector import FaceDetector  # noqa: E402
from nimbus.embedder import Embedder  # noqa: E402
from nimbus.recogniser import Recogniser  # noqa: E402
from nimbus.renderer import draw_detections  # noqa: E402
from nimbus.types import Detection  # noqa: E402

EMBEDDINGS_PATH = REPO_ROOT / "refs" / "embeddings.npz"
CALIBRATION_PATH = REPO_ROOT / "refs" / "calibration.json"


def main() -> int:
    video = REPO_ROOT / "data" / "input" / "nimbus.mp4"
    out_frame = REPO_ROOT / "data" / "output" / "verify_phase2_frame.png"
    sample_indices = [150, 300, 450, 600, 750, 900, 1200, 1500]

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"error: cannot open {video}", file=sys.stderr)
        return 2

    detector = FaceDetector()
    embedder = Embedder()
    recogniser = Recogniser(EMBEDDINGS_PATH, CALIBRATION_PATH)

    best_frame_idx = -1
    best_score = -1.0          # prefer frames with more named (non-Unknown) labels
    best_annotated = None

    print(f"{'frame':>6}  {'faces':>5}  decisions")
    print("-" * 90)
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"{idx:>6}  read failed")
            continue
        detections = detector.detect(frame)
        labelled: list[Detection] = []
        decisions: list[str] = []
        for det in detections:
            if det.aligned_face is None:
                labelled.append(det)
                decisions.append("(no crop)")
                continue
            emb = embedder.embed_aligned_face(det.aligned_face)
            r = recogniser.recognise(emb)
            labelled.append(replace(det, label=r.label, label_confidence=r.confidence))
            decisions.append(
                f"{r.label}({r.confidence:.2f}|top1={r.top1_name}:{r.top1_distance:.2f},"
                f"top2={r.top2_name}:{r.top2_distance:.2f})"
            )
        n_named = sum(1 for d in labelled if d.label not in ("Unknown", "Face"))
        score = n_named + len(detections) * 0.01  # prefer named; tiebreak by total
        print(f"{idx:>6}  {len(detections):>5}  {' | '.join(decisions) or '—'}")
        if score > best_score:
            best_score = score
            best_frame_idx = idx
            best_annotated = draw_detections(frame.copy(), labelled)

    cap.release()

    if best_annotated is not None:
        out_frame.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_frame), best_annotated)
        print(f"\nBest: frame {best_frame_idx} → {out_frame.relative_to(REPO_ROOT)}")
        return 0
    print("\nNo usable detections across sampled frames.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
