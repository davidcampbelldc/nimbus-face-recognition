#!/usr/bin/env python3
"""Phase 4 evaluation harness.

Compares the recogniser's per-frame labels to hand-verified ground truth on
the 40-frame eval set. Reports detection recall, per-character
precision/recall/F1 with bootstrap 95% CIs, confusion matrix (including
"Unknown" as a first-class class), and runtime.

Inputs:
  - eval/ground_truth.json:
      {
        "<frame_idx>": [
          {"bbox": [x,y,w,h], "label": "Harry" | ... | "Unknown"},
          ...
        ], ...
      }
  - eval/frames_raw/f<idx>.png — the raw frames (already extracted)
  - refs/embeddings.npz, refs/calibration.json — recogniser artefacts

Outputs:
  - eval/predictions.json  — per-frame model predictions (for debugging)
  - eval/metrics.json      — headline numbers (README-ready)

IoU threshold for matching predictions to GT is 0.5 (plan §7).

Notes on tracker ablation:
  This harness evaluates per-frame labels with the tracker OFF (isolated
  frame eval doesn't have temporal context). The tracker ablation is a
  separate angle: Phase 3's verify_phase3.py already quantifies the
  flip-reduction from IoU-matched tracker smoothing. README references
  both.

Usage:
    .venv/bin/python scripts/evaluate.py
    .venv/bin/python scripts/evaluate.py --bootstrap 1000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from nimbus.detector import FaceDetector  # noqa: E402
from nimbus.embedder import Embedder  # noqa: E402
from nimbus.pipeline import _recognise_detections  # noqa: E402
from nimbus.recogniser import Recogniser  # noqa: E402

EVAL = REPO_ROOT / "eval"
GT_PATH = EVAL / "ground_truth.json"
RAW_DIR = EVAL / "frames_raw"
PRED_PATH = EVAL / "predictions.json"
METRICS_PATH = EVAL / "metrics.json"
EMBEDDINGS = REPO_ROOT / "refs" / "embeddings.npz"
CALIBRATION = REPO_ROOT / "refs" / "calibration.json"

CLASSES = ["Harry", "Ron", "Hermione", "McGonagall", "Snape", "Unknown"]
IOU_MATCH = 0.5


def iou(a: list[int], b: list[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _match_predictions(
    gt: list[dict],
    preds: list[dict],
) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """Greedy IoU match predictions to GT. Returns:
       - matched pairs (gt_entry, pred_entry) with IoU >= IOU_MATCH
       - unmatched GT entries (missed detections)
       - unmatched predictions (false detections)"""
    candidates: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(preds):
            score = iou(g["bbox"], p["bbox"])
            if score >= IOU_MATCH:
                candidates.append((score, gi, pi))
    candidates.sort(reverse=True)

    claimed_gt: set[int] = set()
    claimed_pred: set[int] = set()
    matched: list[tuple[dict, dict]] = []
    for _score, gi, pi in candidates:
        if gi in claimed_gt or pi in claimed_pred:
            continue
        claimed_gt.add(gi)
        claimed_pred.add(pi)
        matched.append((gt[gi], preds[pi]))

    unmatched_gt = [g for i, g in enumerate(gt) if i not in claimed_gt]
    unmatched_pred = [p for i, p in enumerate(preds) if i not in claimed_pred]
    return matched, unmatched_gt, unmatched_pred


def run_predictions(frame_indices: list[int]) -> dict[str, list[dict]]:
    """Run detect + recognise on each frame, return per-frame predictions."""
    from nimbus.types import Detection
    _ = Detection  # keep import (avoid tree-shake by type checkers)

    detector = FaceDetector()
    embedder = Embedder()
    recogniser = Recogniser(EMBEDDINGS, CALIBRATION)

    preds: dict[str, list[dict]] = {}
    for idx in frame_indices:
        path = RAW_DIR / f"f{idx:05d}.png"
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  warning: {path} missing, skipping")
            preds[str(idx)] = []
            continue
        t0 = time.monotonic()
        dets = detector.detect(frame)
        dets = _recognise_detections(dets, embedder, recogniser)
        elapsed = time.monotonic() - t0
        preds[str(idx)] = [
            {
                "bbox": list(d.bbox),
                "label": d.label,
                "rec_conf": (round(float(d.label_confidence), 3)
                             if d.label_confidence is not None else None),
                "det_conf": round(float(d.confidence), 3),
            }
            for d in dets
        ]
        print(f"  f{idx:05d}: {len(dets)} dets, {elapsed:.2f}s")
    return preds


# --------------------------------------------------------------- metrics


def per_frame_outcomes(
    ground_truth: dict[str, list[dict]],
    predictions: dict[str, list[dict]],
) -> list[dict]:
    """Flatten all (GT face, prediction) outcomes across the eval set.
    Each outcome is one of: matched (with both labels), missed_gt, false_pred."""
    outcomes: list[dict] = []
    for frame_idx, gt_list in ground_truth.items():
        preds = predictions.get(frame_idx, [])
        matched, unmatched_gt, unmatched_pred = _match_predictions(gt_list, preds)
        for gt_entry, pred_entry in matched:
            outcomes.append({
                "frame": frame_idx,
                "kind": "matched",
                "gt_label": gt_entry["label"],
                "pred_label": pred_entry["label"],
            })
        for gt_entry in unmatched_gt:
            outcomes.append({
                "frame": frame_idx,
                "kind": "missed",  # detection miss — counts as FN for gt_label
                "gt_label": gt_entry["label"],
                "pred_label": None,
            })
        for pred_entry in unmatched_pred:
            outcomes.append({
                "frame": frame_idx,
                "kind": "extra",   # detection spawned on non-face — counts as FP for pred_label
                "gt_label": None,
                "pred_label": pred_entry["label"],
            })
    return outcomes


def _precision_recall_f1(
    outcomes: list[dict],
    cls: str,
) -> tuple[float, float, float]:
    tp = sum(1 for o in outcomes
             if o["kind"] == "matched"
             and o["gt_label"] == cls and o["pred_label"] == cls)
    fp = sum(1 for o in outcomes
             if (o["kind"] == "matched" and o["gt_label"] != cls
                 and o["pred_label"] == cls)
             or (o["kind"] == "extra" and o["pred_label"] == cls))
    fn = sum(1 for o in outcomes
             if (o["kind"] == "matched" and o["gt_label"] == cls
                 and o["pred_label"] != cls)
             or (o["kind"] == "missed" and o["gt_label"] == cls))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def confusion_matrix(outcomes: list[dict]) -> dict[str, dict[str, int]]:
    """Rows = gt label (incl. 'missed' pseudo-row); cols = predicted label
    (incl. 'extra' pseudo-col). 'missed' means detector didn't find the GT
    face; 'extra' means model produced a prediction where no GT face
    existed at that location."""
    grid: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for o in outcomes:
        if o["kind"] == "matched":
            grid[o["gt_label"]][o["pred_label"]] += 1
        elif o["kind"] == "missed":
            grid[o["gt_label"]]["__missed__"] += 1
        elif o["kind"] == "extra":
            grid["__extra__"][o["pred_label"]] += 1
    return {k: dict(v) for k, v in grid.items()}


def detection_recall(
    ground_truth: dict[str, list[dict]],
    predictions: dict[str, list[dict]],
) -> tuple[float, int, int]:
    """Over all GT faces, fraction whose bbox was matched by some prediction.
    Returns (recall, n_matched, n_total)."""
    total = 0
    matched = 0
    for frame_idx, gt_list in ground_truth.items():
        preds = predictions.get(frame_idx, [])
        pairs, _unmatched_gt, _unmatched_pred = _match_predictions(gt_list, preds)
        total += len(gt_list)
        matched += len(pairs)
    return (matched / total if total > 0 else 0.0), matched, total


def bootstrap_ci(
    outcomes: list[dict],
    cls: str,
    metric: str,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap 95% CI for precision, recall, or F1 on one class.
    Resamples at the frame level (group outcomes by frame) so dependence
    within a frame is respected."""
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for o in outcomes:
        by_frame[o["frame"]].append(o)
    frames = list(by_frame.keys())
    rng = random.Random(seed)

    samples: list[float] = []
    for _ in range(n_resamples):
        resample_frames = [rng.choice(frames) for _ in range(len(frames))]
        resample_outcomes = [
            o for f in resample_frames for o in by_frame[f]
        ]
        p, r, f1 = _precision_recall_f1(resample_outcomes, cls)
        value = {"precision": p, "recall": r, "f1": f1}[metric]
        samples.append(value)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def build_metrics(
    ground_truth: dict[str, list[dict]],
    predictions: dict[str, list[dict]],
    n_resamples: int,
) -> dict:
    outcomes = per_frame_outcomes(ground_truth, predictions)

    per_class: dict[str, dict] = {}
    for cls in CLASSES:
        p, r, f1 = _precision_recall_f1(outcomes, cls)
        p_lo, p_hi = bootstrap_ci(outcomes, cls, "precision", n_resamples, seed=1)
        r_lo, r_hi = bootstrap_ci(outcomes, cls, "recall", n_resamples, seed=2)
        f1_lo, f1_hi = bootstrap_ci(outcomes, cls, "f1", n_resamples, seed=3)
        per_class[cls] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "precision_ci95": [round(p_lo, 4), round(p_hi, 4)],
            "recall_ci95": [round(r_lo, 4), round(r_hi, 4)],
            "f1_ci95": [round(f1_lo, 4), round(f1_hi, 4)],
        }

    macro_f1 = float(np.mean([per_class[c]["f1"] for c in CLASSES]))
    det_recall, det_matched, det_total = detection_recall(ground_truth, predictions)

    return {
        "n_frames": len(ground_truth),
        "n_gt_faces": sum(len(v) for v in ground_truth.values()),
        "detection_recall": round(det_recall, 4),
        "detection_matched": det_matched,
        "detection_total": det_total,
        "iou_threshold": IOU_MATCH,
        "classes": CLASSES,
        "per_class": per_class,
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": confusion_matrix(outcomes),
        "bootstrap_resamples": n_resamples,
    }


# --------------------------------------------------------------- entry


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="bootstrap resamples for CI estimation (default 1000)")
    p.add_argument("--skip-predict", action="store_true",
                   help="reuse existing eval/predictions.json instead of re-running")
    args = p.parse_args()

    if not GT_PATH.exists():
        print(f"error: {GT_PATH} missing — run labelling first", file=sys.stderr)
        return 2

    ground_truth = json.loads(GT_PATH.read_text())
    frame_indices = sorted(int(k) for k in ground_truth)

    if args.skip_predict and PRED_PATH.exists():
        print(f"reusing {PRED_PATH}")
        predictions = json.loads(PRED_PATH.read_text())
    else:
        print(f"running predictions on {len(frame_indices)} frames...")
        predictions = run_predictions(frame_indices)
        PRED_PATH.write_text(json.dumps(predictions, indent=2))

    print(f"computing metrics with {args.bootstrap} bootstrap resamples...")
    metrics = build_metrics(ground_truth, predictions, args.bootstrap)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print()
    print("=== HEADLINE NUMBERS ===")
    print(f"  frames:             {metrics['n_frames']}")
    print(f"  GT faces:           {metrics['n_gt_faces']}")
    print(f"  detection recall:   {metrics['detection_recall']:.3f}"
          f"  ({metrics['detection_matched']}/{metrics['detection_total']})")
    print(f"  macro-F1:           {metrics['macro_f1']:.3f}")
    print()
    print("  per-class (F1 with 95% bootstrap CI):")
    for cls in CLASSES:
        pc = metrics["per_class"][cls]
        print(f"    {cls:12s}"
              f" P={pc['precision']:.3f}"
              f" R={pc['recall']:.3f}"
              f" F1={pc['f1']:.3f}"
              f" CI=[{pc['f1_ci95'][0]:.3f}, {pc['f1_ci95'][1]:.3f}]")
    print()
    print(f"  written to: {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
