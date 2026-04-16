#!/usr/bin/env python3
"""Phase 4 diagnostic plots for the README.

Generates:
  - eval/plots/distances_<character>.png
      Per-character KDE curves of:
        (a) intra-class ref-to-ref distances (how tight is this cluster)
        (b) inter-class ref-to-ref distances (how far to neighbours)
      with the calibrated threshold overlaid as a vertical line.
  - eval/plots/threshold_sweep.png
      For each character: on-eval F1 as a function of threshold. Shows
      whether the calibrated threshold is near-optimal on the eval set
      (independent of how it was selected via LOO on refs).

Requires: eval/ground_truth.json, eval/predictions.json, refs/embeddings.npz,
refs/calibration.json.

Usage:
    .venv/bin/python scripts/plot_distributions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no DISPLAY needed
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from nimbus.embedder import cosine_distance  # noqa: E402

EVAL = REPO_ROOT / "eval"
PLOTS = EVAL / "plots"
EMBEDDINGS = REPO_ROOT / "refs" / "embeddings.npz"
CALIBRATION = REPO_ROOT / "refs" / "calibration.json"

CHARACTERS = ["harry", "ron", "hermione", "mcgonagall", "snape"]


def _pairwise_distances(vectors: np.ndarray) -> list[float]:
    out = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            out.append(cosine_distance(vectors[i], vectors[j]))
    return out


def _cross_distances(a: np.ndarray, b: np.ndarray) -> list[float]:
    return [cosine_distance(x, y) for x in a for y in b]


def plot_per_character_distances(
    refs: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> None:
    for name in CHARACTERS:
        my_refs = refs[name]
        intra = _pairwise_distances(my_refs)
        inter: list[float] = []
        for other in CHARACTERS:
            if other == name:
                continue
            inter.extend(_cross_distances(my_refs, refs[other]))

        fig, ax = plt.subplots(figsize=(7, 4))
        # simple histogram — robust even with small N, no scipy dep needed.
        bins = np.linspace(0, 1.2, 40)
        if intra:
            ax.hist(intra, bins=bins, alpha=0.6, label=f"intra-class (n={len(intra)})",
                    color="#2ca02c", density=True)
        if inter:
            ax.hist(inter, bins=bins, alpha=0.5, label=f"inter-class (n={len(inter)})",
                    color="#d62728", density=True)

        thr = thresholds[name]
        ax.axvline(thr, color="black", linestyle="--", linewidth=1.5,
                   label=f"threshold = {thr:.3f}")

        ax.set_xlabel("cosine distance")
        ax.set_ylabel("density")
        ax.set_title(f"{name.capitalize()} — reference embedding geometry")
        ax.legend(loc="upper center")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = PLOTS / f"distances_{name}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  wrote {out}")


def _eval_threshold(
    ground_truth: dict[str, list[dict]],
    predictions: dict[str, list[dict]],
    cls: str,
    threshold: float,
) -> float:
    """Compute F1 for class `cls` on the eval set, using `threshold` in place
    of the calibrated one. Predictions were recorded BEFORE thresholding, so
    we re-threshold using `top1_distance` embedded in predictions if present.

    Simplification: predictions.json only has the final labels + rec_conf,
    not top1_distance. So for the sweep we instead reinterpret rec_conf:
    a prediction with rec_conf >= (1 - threshold) passes the threshold test
    (since rec_conf = 1 - top1_distance per the recogniser). Labels flip to
    Unknown when rec_conf < (1 - threshold).
    """
    tp = fp = fn = 0
    from scripts.evaluate import _match_predictions  # noqa: PLC0415

    for frame_idx, gt_list in ground_truth.items():
        preds = predictions.get(frame_idx, [])
        # Build thresholded predictions: label -> Unknown if confidence too low.
        thresholded = []
        for p in preds:
            if p.get("rec_conf") is not None and p["label"] != "Unknown":
                if p["rec_conf"] < (1.0 - threshold):
                    thresholded.append({**p, "label": "Unknown"})
                else:
                    thresholded.append(p)
            else:
                thresholded.append(p)
        matched, unmatched_gt, unmatched_pred = _match_predictions(gt_list, thresholded)
        for g, pred in matched:
            if g["label"] == cls and pred["label"] == cls:
                tp += 1
            elif pred["label"] == cls:
                fp += 1
            elif g["label"] == cls:
                fn += 1
        for g in unmatched_gt:
            if g["label"] == cls:
                fn += 1
        for p in unmatched_pred:
            if p["label"] == cls:
                fp += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0)


def plot_threshold_sweep(
    ground_truth: dict[str, list[dict]],
    predictions: dict[str, list[dict]],
    thresholds: dict[str, float],
) -> None:
    sweep = np.linspace(0.1, 0.7, 25)
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = {"harry": "#d62728", "ron": "#ff7f0e", "hermione": "#9467bd",
               "mcgonagall": "#2ca02c", "snape": "#17becf"}
    for name in CHARACTERS:
        label = name.capitalize()
        f1s = [_eval_threshold(ground_truth, predictions, label, t) for t in sweep]
        ax.plot(sweep, f1s, marker="o", markersize=4, label=name.capitalize(),
                color=colours[name])
        ax.axvline(thresholds[name], linestyle=":", alpha=0.4, color=colours[name])

    ax.set_xlabel("cosine-distance threshold")
    ax.set_ylabel("F1 on eval set")
    ax.set_title("Threshold sweep per character (dotted = calibrated threshold)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", ncol=5)
    fig.tight_layout()
    out = PLOTS / "threshold_sweep.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> int:
    if not EMBEDDINGS.exists():
        print(f"error: {EMBEDDINGS} missing", file=sys.stderr)
        return 2
    if not CALIBRATION.exists():
        print(f"error: {CALIBRATION} missing", file=sys.stderr)
        return 2

    PLOTS.mkdir(parents=True, exist_ok=True)

    refs_data = np.load(EMBEDDINGS)
    refs = {name: refs_data[name] for name in refs_data.files}
    calib = json.loads(CALIBRATION.read_text())
    thresholds = {name: float(entry["threshold"])
                  for name, entry in calib["characters"].items()}

    print("plotting per-character distance distributions...")
    plot_per_character_distances(refs, thresholds)

    gt_path = EVAL / "ground_truth.json"
    preds_path = EVAL / "predictions.json"
    if gt_path.exists() and preds_path.exists():
        ground_truth = json.loads(gt_path.read_text())
        predictions = json.loads(preds_path.read_text())
        print("plotting threshold sweep on eval set...")
        plot_threshold_sweep(ground_truth, predictions, thresholds)
    else:
        print("(skipping threshold sweep — run scripts/evaluate.py first)")

    print(f"\nplots saved to: {PLOTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
