"""Validate the reference set and calibrate per-class thresholds.

Reads `refs/embeddings.npz` (from build_references.py) and:

1. Sanity-gates the ref set (plan §6):
     - intra-class: every ref-to-centroid cosine distance < 0.30
     - inter-class: each centroid to nearest other centroid > 0.40

2. Computes per-class thresholds via leave-one-out on the refs (plan §5.5):
     - For each ref_i in char C, temporarily remove it from C's set.
     - Compute the k-NN-mean distance (k=3, or n-1 if tiny) from ref_i to
       the remaining refs of C.
     - The distribution of LOO distances across C's refs is the "self-
       similarity" benchmark — a genuine inference-time match should land
       within this distribution.
     - threshold[C] = max(LOO_distances) + safety buffer.

3. Uses a global margin (plan §5.6): `margin = 0.05`. The margin test
   requires `d(top2) - d(top1) > margin`. 0.05 is a conservative v1; we
   can tune per-class after Phase 4 eval if any character misbehaves.

Writes `refs/calibration.json` (committed — single source of truth for
thresholds/margins used by the recogniser). Prints a full report.

Run from repo root:
    .venv/bin/python scripts/validate_references.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from nimbus.embedder import cosine_distance, l2_normalise  # noqa: E402

EMBEDDINGS_PATH = REPO_ROOT / "refs" / "embeddings.npz"
CALIBRATION_PATH = REPO_ROOT / "refs" / "calibration.json"

K = 3                     # k-NN k (plan §5.4)
GLOBAL_MARGIN = 0.05      # plan §5.6 v1
INTRA_GATE = 0.30         # plan §6 validation gate
INTER_GATE = 0.40         # plan §6 validation gate
THRESHOLD_BUFFER = 0.05   # added on top of LOO max for safety
THRESHOLD_FLOOR = 0.25    # prevents over-tight threshold from small ref sets
THRESHOLD_CEILING = 0.50  # prevents ref-set outliers from blowing threshold wide


@dataclass
class CharStats:
    name: str
    n_refs: int
    embeddings: np.ndarray
    centroid: np.ndarray
    intra_mean: float
    intra_max: float
    loo_distances: list[float]
    threshold: float


def knn_mean_distance(query: np.ndarray, refs: np.ndarray, k: int) -> float:
    """Mean cosine distance from `query` to its k nearest neighbours in `refs`."""
    if refs.shape[0] == 0:
        return float("inf")
    effective_k = min(k, refs.shape[0])
    dists = np.array([cosine_distance(query, r) for r in refs])
    dists.sort()
    return float(dists[:effective_k].mean())


def compute_char_stats(name: str, embeddings: np.ndarray) -> CharStats:
    n = embeddings.shape[0]
    centroid = l2_normalise(embeddings.mean(axis=0))

    intra = np.array([cosine_distance(v, centroid) for v in embeddings])

    # Leave-one-out: for each ref, compute its k-NN-mean distance to the rest.
    loo: list[float] = []
    for i in range(n):
        held_out = embeddings[i]
        remaining = np.delete(embeddings, i, axis=0)
        loo.append(knn_mean_distance(held_out, remaining, K))

    loo_max = max(loo) if loo else 0.0
    threshold = min(max(loo_max + THRESHOLD_BUFFER, THRESHOLD_FLOOR), THRESHOLD_CEILING)

    return CharStats(
        name=name,
        n_refs=n,
        embeddings=embeddings,
        centroid=centroid,
        intra_mean=float(intra.mean()),
        intra_max=float(intra.max()),
        loo_distances=loo,
        threshold=float(threshold),
    )


def main() -> int:
    if not EMBEDDINGS_PATH.exists():
        print(f"error: {EMBEDDINGS_PATH} missing — run build_references.py first",
              file=sys.stderr)
        return 1

    data = np.load(EMBEDDINGS_PATH)
    char_names = sorted(data.files)
    stats = [compute_char_stats(n, data[n]) for n in char_names]

    # --- Intra-class gate -------------------------------------------------
    print("=== Intra-class spread (target < 0.30) ===")
    intra_fail: list[str] = []
    for s in stats:
        ok = s.intra_max < INTRA_GATE
        marker = "✓" if ok else "⚠ FAIL"
        print(f"  {s.name:12s} n={s.n_refs}  mean={s.intra_mean:.3f}  "
              f"max={s.intra_max:.3f}  {marker}")
        if not ok:
            intra_fail.append(s.name)

    # --- Inter-class gate -------------------------------------------------
    names = [s.name for s in stats]
    matrix = np.array([
        [cosine_distance(a.centroid, b.centroid) for b in stats]
        for a in stats
    ])
    print("\n=== Pairwise centroid distances (target each row's min > 0.40) ===")
    header = "              " + "  ".join(f"{n[:9]:>9s}" for n in names)
    print(header)
    inter_fail: list[str] = []
    for i, s in enumerate(stats):
        row = "  ".join(f"{matrix[i, j]:9.3f}" for j in range(len(stats)))
        others_min = float(min(matrix[i, j] for j in range(len(stats)) if j != i))
        marker = "✓" if others_min > INTER_GATE else "⚠ FAIL"
        print(f"  {s.name:12s}{row}   nearest-other={others_min:.3f} {marker}")
        if others_min <= INTER_GATE:
            inter_fail.append(s.name)

    # --- LOO + threshold calibration --------------------------------------
    print("\n=== Leave-one-out calibration (threshold = max(LOO) + buffer) ===")
    print(f"  (buffer={THRESHOLD_BUFFER}, floor={THRESHOLD_FLOOR}, "
          f"ceiling={THRESHOLD_CEILING})")
    for s in stats:
        loo_max = max(s.loo_distances) if s.loo_distances else 0.0
        loo_mean = float(np.mean(s.loo_distances)) if s.loo_distances else 0.0
        print(f"  {s.name:12s} LOO mean={loo_mean:.3f}  max={loo_max:.3f}  "
              f"→ threshold={s.threshold:.3f}")

    # --- Persist ----------------------------------------------------------
    payload = {
        "model": "Facenet512",
        "detector": "retinaface",
        "k": K,
        "global_margin": GLOBAL_MARGIN,
        "threshold_buffer": THRESHOLD_BUFFER,
        "threshold_floor": THRESHOLD_FLOOR,
        "threshold_ceiling": THRESHOLD_CEILING,
        "characters": {
            s.name: {
                "n_refs": s.n_refs,
                "threshold": s.threshold,
                "margin": GLOBAL_MARGIN,
                "intra_mean": s.intra_mean,
                "intra_max": s.intra_max,
                "loo_max": float(max(s.loo_distances)) if s.loo_distances else 0.0,
                "loo_mean": float(np.mean(s.loo_distances)) if s.loo_distances else 0.0,
            }
            for s in stats
        },
    }
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {CALIBRATION_PATH.relative_to(REPO_ROOT)}")

    # --- Final gate -------------------------------------------------------
    if intra_fail or inter_fail:
        print(f"\n⚠ Gate failures — intra: {intra_fail or 'none'}  "
              f"inter: {inter_fail or 'none'}")
        print("  Review the failing refs before relying on this calibration.")
        return 2
    print("\n✓ All gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
