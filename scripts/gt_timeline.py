#!/usr/bin/env python3
"""Timeline view of eval ground truth (+ predictions if present).

Prints a table mapping each sampled frame to a MM:SS.mmm timecode with
the labelled characters listed, plus prediction comparison if predictions
have been generated. Also writes eval/timeline.json so the mapping is
inspectable as data.

Use when you need to jump to a specific moment in the video to spot-check
a label, or to see at a glance which frames the model got right/wrong.

Usage:
    .venv/bin/python scripts/gt_timeline.py
    .venv/bin/python scripts/gt_timeline.py --mismatches-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from nimbus.tracker import iou  # noqa: E402

EVAL = REPO_ROOT / "eval"
GT_PATH = EVAL / "ground_truth.json"
PRED_PATH = EVAL / "predictions.json"
VIDEO_PATH = REPO_ROOT / "data" / "input" / "nimbus.mp4"
OUT_JSON = EVAL / "timeline.json"

# Video is 30000/1001 per ffprobe. Keep this local rather than importing
# opencv so the script runs fast with no heavy deps.
FPS = 30000.0 / 1001.0


def timecode(seconds: float) -> str:
    """MM:SS.mmm — matches the format most video players show."""
    mm, ss = divmod(seconds, 60)
    return f"{int(mm):02d}:{ss:06.3f}"


def _match_by_bbox(
    gt_boxes: list[dict],
    pred_boxes: list[dict],
    iou_threshold: float = 0.5,
) -> list[dict | None]:
    """For each GT box, return the best-matching prediction (or None)."""
    # Greedy: for each GT in order, find best unclaimed pred over threshold.
    claimed: set[int] = set()
    out: list[dict | None] = []
    for g in gt_boxes:
        best_i = -1
        best_iou = iou_threshold
        for i, p in enumerate(pred_boxes):
            if i in claimed:
                continue
            v = iou(g["bbox"], p["bbox"])
            if v >= best_iou:
                best_iou = v
                best_i = i
        if best_i >= 0:
            claimed.add(best_i)
            out.append(pred_boxes[best_i])
        else:
            out.append(None)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mismatches-only", action="store_true",
                   help="show only frames where prediction disagrees with GT")
    args = p.parse_args()

    if not GT_PATH.exists():
        print(f"error: {GT_PATH} missing", file=sys.stderr)
        return 2

    ground_truth = json.loads(GT_PATH.read_text())
    predictions = json.loads(PRED_PATH.read_text()) if PRED_PATH.exists() else {}

    # Build the flat timeline
    timeline: list[dict] = []
    for frame_str in sorted(ground_truth, key=int):
        idx = int(frame_str)
        t = idx / FPS
        gt_boxes = ground_truth[frame_str]
        pred_boxes = predictions.get(frame_str, [])
        matched = _match_by_bbox(gt_boxes, pred_boxes) if pred_boxes else [None] * len(gt_boxes)

        entry = {
            "frame": idx,
            "timestamp_sec": round(t, 3),
            "timecode": timecode(t),
            "n_detections": len(gt_boxes),
            "boxes": [],
        }
        for g, pred in zip(gt_boxes, matched, strict=True):
            box_entry = {
                "box_id": g.get("box_id"),
                "bbox": g["bbox"],
                "gt_label": g.get("label", ""),
                "labeller_conf": g.get("labeller_conf", ""),
            }
            if pred is not None:
                box_entry["pred_label"] = pred["label"]
                box_entry["pred_rec_conf"] = pred.get("rec_conf")
                box_entry["match"] = pred["label"] == g.get("label")
            else:
                box_entry["pred_label"] = "__no_prediction__"
                box_entry["match"] = False
            entry["boxes"].append(box_entry)
        timeline.append(entry)

    OUT_JSON.write_text(json.dumps(timeline, indent=2))

    # Print human-readable table
    print(f"{'Frame':>6}  {'Timecode':>10}  N  box  GT           pred         result")
    print("-" * 80)
    shown = 0
    for entry in timeline:
        if entry["n_detections"] == 0:
            if args.mismatches_only:
                continue
            print(f"{entry['frame']:>6}  {entry['timecode']:>10}  0  —    (no detections)")
            shown += 1
            continue
        for b in entry["boxes"]:
            if args.mismatches_only and b.get("match", False):
                continue
            mark = "✓" if b.get("match") else ("✗" if b.get("pred_label") else "—")
            pred = b.get("pred_label", "?")
            if pred == "__no_prediction__":
                pred = "(no pred)"
            conf = f"[{b.get('labeller_conf', '?')}]"
            print(f"{entry['frame']:>6}  {entry['timecode']:>10}  "
                  f"{entry['n_detections']}  {b['box_id']:>3}"
                  f"  {b.get('gt_label', '?'):12s} {pred:12s} {mark} {conf}")
            shown += 1

    print(f"\n{shown} rows shown; wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
