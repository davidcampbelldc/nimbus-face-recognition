#!/usr/bin/env python3
"""Phase 3 verification: does the IoU tracker + mode-smoothing kill the
per-frame label strobe?

What this does:
  - Iterates a window of consecutive frames from nimbus.mp4.
  - For each frame, runs detect -> embed -> recognise exactly once.
  - Records the raw per-frame labels (recogniser output).
  - Feeds the same detections through Tracker.update() to get smoothed labels.
  - Saves side-by-side annotated PNGs (raw on the left, tracker-smoothed on
    the right) for a sampled subset of frames, so a human reviewer (or
    Claude vision via Read) can eyeball the strobe collapse directly.
  - Dumps a compact JSON report (report.json) with per-frame label tuples
    so the strobe-vs-stable pattern is inspectable as data, not just video.

Run from repo root:
    .venv/bin/python scripts/verify_phase3.py
    .venv/bin/python scripts/verify_phase3.py --start 150 --window 50 --save-every 5

The annotated frames land in data/output/verify_phase3/ .
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
from nimbus.embedder import Embedder  # noqa: E402
from nimbus.pipeline import _recognise_detections  # noqa: E402
from nimbus.recogniser import Recogniser  # noqa: E402
from nimbus.renderer import draw_detections  # noqa: E402
from nimbus.scene_cut import SceneCutDetector  # noqa: E402
from nimbus.tracker import Tracker  # noqa: E402
from nimbus.types import Detection  # noqa: E402

VIDEO = REPO_ROOT / "data" / "input" / "nimbus.mp4"
OUT_DIR = REPO_ROOT / "data" / "output" / "verify_phase3"
EMBEDDINGS = REPO_ROOT / "refs" / "embeddings.npz"
CALIBRATION = REPO_ROOT / "refs" / "calibration.json"


def _stack_side_by_side(
    frame: np.ndarray,
    raw_dets: list[Detection],
    tracked_dets: list[Detection],
) -> np.ndarray:
    """Render raw-labels | tracker-smoothed-labels side-by-side on the same frame."""
    left = draw_detections(frame.copy(), raw_dets)
    right = draw_detections(frame.copy(), tracked_dets)
    # thin divider
    divider = np.full((left.shape[0], 4, 3), 255, dtype=np.uint8)
    composite = np.hstack([left, divider, right])
    h = 30
    banner = np.zeros((h, composite.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        banner, "RAW (no tracker)", (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        banner, "TRACKER-SMOOTHED",
        (left.shape[1] + 20, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return np.vstack([banner, composite])


def _summarise_detection(d: Detection) -> dict:
    return {
        "bbox": list(d.bbox),
        "det_conf": round(float(d.confidence), 3),
        "label": d.label,
        "label_conf": (
            round(float(d.label_confidence), 3)
            if d.label_confidence is not None else None
        ),
    }


def _count_label_flips(label_sequences: list[list[str]]) -> int:
    """Sum label transitions across all frames. We pair detections across
    consecutive frames by first-index only — this is a coarse proxy for
    strobe count and assumes detection order is roughly stable, which is
    what we expect in short windows of a continuous scene."""
    flips = 0
    for f in range(1, len(label_sequences)):
        prev = label_sequences[f - 1]
        cur = label_sequences[f]
        for i in range(min(len(prev), len(cur))):
            if prev[i] != cur[i]:
                flips += 1
    return flips


def _process_window(
    cap: cv2.VideoCapture,
    start: int,
    window: int,
    save_every: int,
) -> dict:
    """Run the stack on a consecutive window. Returns the report dict and
    writes side-by-side PNGs to OUT_DIR as a side effect."""
    detector = FaceDetector()
    embedder = Embedder()
    recogniser = Recogniser(EMBEDDINGS, CALIBRATION)
    scene_cut = SceneCutDetector()
    tracker = Tracker()

    records: list[dict] = []
    raw_seqs: list[list[str]] = []
    tracked_seqs: list[list[str]] = []

    for k in range(window):
        frame_idx = start + k
        ok, frame = cap.read()
        if not ok:
            print(f"  EOF at frame {frame_idx}")
            break

        cut = scene_cut.is_cut(frame)
        dets = detector.detect(frame)
        dets = _recognise_detections(dets, embedder, recogniser)
        tracked = tracker.update(list(dets), scene_cut=cut)

        raw_labels = [d.label for d in dets]
        tracked_labels = [d.label for d in tracked]
        raw_seqs.append(raw_labels)
        tracked_seqs.append(tracked_labels)
        records.append({
            "frame": frame_idx, "scene_cut": cut, "n_detections": len(dets),
            "raw": [_summarise_detection(d) for d in dets],
            "tracked": [_summarise_detection(d) for d in tracked],
        })

        tags = (" [CUT]" if cut else "") + (
            " [SMOOTHED]" if raw_labels != tracked_labels else ""
        )
        print(f"  f{frame_idx:04d} n={len(dets)}"
              f" raw={raw_labels} tracked={tracked_labels}{tags}")

        if k % save_every == 0:
            composite = _stack_side_by_side(frame, dets, tracked)
            cv2.imwrite(str(OUT_DIR / f"frame_{frame_idx:05d}.png"), composite)

    raw_flips = _count_label_flips(raw_seqs)
    tracked_flips = _count_label_flips(tracked_seqs)
    return {
        "start_frame": start,
        "window": window,
        "frames_processed": len(records),
        "raw_label_flips": raw_flips,
        "tracked_label_flips": tracked_flips,
        "flip_reduction_pct": (
            round(100 * (1 - tracked_flips / raw_flips), 1) if raw_flips > 0 else None
        ),
        "frames": records,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=150,
                   help="first frame index to process (default: 150)")
    p.add_argument("--window", type=int, default=50,
                   help="number of consecutive frames (default: 50)")
    p.add_argument("--save-every", type=int, default=5,
                   help="save annotated PNG every N frames (default: 5)")
    args = p.parse_args()

    if not VIDEO.exists():
        print(f"error: {VIDEO} not found", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in OUT_DIR.glob("*.png"):
        f.unlink()

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"error: could not open {VIDEO}", file=sys.stderr)
        return 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    print(f"processing frames [{args.start}, {args.start + args.window})...")
    report = _process_window(cap, args.start, args.window, args.save_every)
    cap.release()

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print()
    print("=== summary ===")
    print(f"  frames processed:     {report['frames_processed']}")
    print(f"  raw label flips:      {report['raw_label_flips']}")
    print(f"  tracker flips:        {report['tracked_label_flips']}")
    if report["flip_reduction_pct"] is not None:
        print(f"  flip reduction:       {report['flip_reduction_pct']}%")
    print(f"  PNGs + report saved to: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
