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

import matplotlib  # noqa: I001

matplotlib.use("Agg")  # no DISPLAY needed — must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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


def _iou(a: list[int], b: list[int]) -> float:
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
    iou_match: float = 0.5,
) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    candidates: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(preds):
            score = _iou(g["bbox"], p["bbox"])
            if score >= iou_match:
                candidates.append((score, gi, pi))
    candidates.sort(reverse=True)
    claimed_g, claimed_p = set(), set()
    matched: list[tuple[dict, dict]] = []
    for _score, gi, pi in candidates:
        if gi in claimed_g or pi in claimed_p:
            continue
        claimed_g.add(gi)
        claimed_p.add(pi)
        matched.append((gt[gi], preds[pi]))
    unmatched_g = [g for i, g in enumerate(gt) if i not in claimed_g]
    unmatched_p = [p for i, p in enumerate(preds) if i not in claimed_p]
    return matched, unmatched_g, unmatched_p


def _eval_threshold(
    ground_truth: dict[str, list[dict]],
    predictions: dict[str, list[dict]],
    cls: str,
    threshold: float,
) -> float:
    """F1 for `cls` on eval set, simulated at a different threshold.

    Predictions were recorded under the calibrated threshold, so rec_conf =
    1 - top1_distance only for predictions whose label is named (not
    Unknown). We use this to approximate a tightening sweep: raise the
    threshold and prior named-class predictions with rec_conf < (1 -
    threshold) flip to Unknown.

    Limitation: we cannot lower the effective threshold below what was used
    originally — a prediction that was Unknown due to the margin test
    stays Unknown under any threshold change in this approximation. So the
    sweep is honest about "what if we were stricter" but silent about
    "what if we were laxer". The calibration doc notes this.
    """
    tp = fp = fn = 0
    for frame_idx, gt_list in ground_truth.items():
        preds = predictions.get(frame_idx, [])
        thresholded = [_apply_threshold(p, threshold) for p in preds]
        ftp, ffp, ffn = _count_one_class(gt_list, thresholded, cls)
        tp += ftp
        fp += ffp
        fn += ffn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0)


def _apply_threshold(p: dict, threshold: float) -> dict:
    rc = p.get("rec_conf")
    if rc is not None and p["label"] != "Unknown" and rc < (1.0 - threshold):
        return {**p, "label": "Unknown"}
    return p


def _count_one_class(
    gt_list: list[dict],
    thresholded: list[dict],
    cls: str,
) -> tuple[int, int, int]:
    matched, unmatched_gt, unmatched_pred = _match_predictions(gt_list, thresholded)
    tp = fp = fn = 0
    for g, pred in matched:
        if g["label"] == cls and pred["label"] == cls:
            tp += 1
        elif pred["label"] == cls:
            fp += 1
        elif g["label"] == cls:
            fn += 1
    fn += sum(1 for g in unmatched_gt if g["label"] == cls)
    fp += sum(1 for p in unmatched_pred if p["label"] == cls)
    return tp, fp, fn


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
