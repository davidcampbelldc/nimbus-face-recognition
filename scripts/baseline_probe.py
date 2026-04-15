"""Baseline probe — measure the feature-space geometry of our reference set.

Runs BEFORE threshold design (plan §7, §9 Phase 0.5). Answers:
  1. Per-character intra-class spread (how tightly do our refs cluster?)
  2. Pairwise inter-character distances (which characters are nearest — the
     hardest pair to distinguish?)

If the hardest pair (expected: Harry vs Ron) has centroid distance < 0.4, the
recogniser must lean harder on margin + per-class thresholds, not on a global
threshold alone.

Outputs:
  - refs/baseline_probe.json (committed) — raw numbers
  - stdout — human-readable report
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCES_DIR = REPO_ROOT / "references"
OUT_PATH = REPO_ROOT / "refs" / "baseline_probe.json"

CHARACTERS = ["harry", "ron", "hermione", "mcgonagall", "snape"]


def l2_normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Inputs are L2-normalised → 1 - dot == cosine distance in [0, 2].
    return float(1.0 - np.dot(a, b))


@dataclass
class CharacterStats:
    name: str
    n_refs: int
    embeddings: np.ndarray   # (n_refs, 512), L2-normalised
    centroid: np.ndarray     # (512,), L2-normalised
    intra_mean: float        # mean cosine distance ref → centroid
    intra_max: float         # max  cosine distance ref → centroid


def embed_image(jpg_path: Path, model_name: str = "Facenet512") -> np.ndarray:
    """Detect + embed one face. Returns L2-normalised 512-d vector."""
    from deepface import DeepFace
    img = cv2.imread(str(jpg_path))
    if img is None:
        raise RuntimeError(f"cv2 failed to read {jpg_path}")
    result = DeepFace.represent(
        img_path=img,
        model_name=model_name,
        detector_backend="retinaface",
        enforce_detection=True,
        align=True,
    )
    if not result:
        raise RuntimeError(f"no embedding for {jpg_path}")
    vec = np.asarray(result[0]["embedding"], dtype=np.float32)
    return l2_normalise(vec)


def measure_character(character: str) -> CharacterStats:
    char_dir = REFERENCES_DIR / character
    jpgs = sorted(char_dir.glob("*.jpg"))
    if not jpgs:
        raise RuntimeError(f"no references for {character}")
    vecs = np.stack([embed_image(p) for p in jpgs])
    centroid = l2_normalise(vecs.mean(axis=0))
    dists = np.array([cosine_distance(v, centroid) for v in vecs])
    return CharacterStats(
        name=character,
        n_refs=len(vecs),
        embeddings=vecs,
        centroid=centroid,
        intra_mean=float(dists.mean()),
        intra_max=float(dists.max()),
    )


def main() -> None:
    print("=== Baseline probe — feature-space geometry ===\n")
    print(f"Model: Facenet512 (via DeepFace), detector: retinaface\n")

    stats: list[CharacterStats] = []
    for character in CHARACTERS:
        print(f"Embedding {character}... ", end="", flush=True)
        s = measure_character(character)
        stats.append(s)
        print(f"{s.n_refs} refs, intra mean={s.intra_mean:.3f}, max={s.intra_max:.3f}")

    print("\n=== Intra-class spread (lower = tighter; plan target < 0.30) ===")
    for s in stats:
        marker = "✓" if s.intra_max < 0.30 else "⚠ LOOSE"
        print(f"  {s.name:12s} n={s.n_refs}  mean={s.intra_mean:.3f}  max={s.intra_max:.3f}  {marker}")

    print("\n=== Pairwise centroid distances (higher = easier to distinguish) ===")
    names = [s.name for s in stats]
    matrix = np.zeros((len(stats), len(stats)))
    for i, a in enumerate(stats):
        for j, b in enumerate(stats):
            matrix[i, j] = cosine_distance(a.centroid, b.centroid)

    # Pretty-print matrix
    header = "              " + "  ".join(f"{n[:9]:>9s}" for n in names)
    print(header)
    for i, n in enumerate(names):
        row = "  ".join(f"{matrix[i, j]:9.3f}" for j in range(len(names)))
        print(f"  {n:12s}{row}")

    # Find hardest pair (excluding diagonal)
    hardest_pair = None
    hardest_dist = 2.0
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            if matrix[i, j] < hardest_dist:
                hardest_dist = matrix[i, j]
                hardest_pair = (names[i], names[j])

    print(f"\n=== Hardest pair ===")
    print(f"  {hardest_pair[0]} vs {hardest_pair[1]}: centroid distance = {hardest_dist:.3f}")

    if hardest_dist < 0.40:
        print(f"\n  ⚠ Hardest pair below 0.40 — recogniser should lean on margin + per-class")
        print(f"    thresholds, not a single global threshold. Confirms plan §5 design.")
    elif hardest_dist < 0.60:
        print(f"\n  ✓ Hardest pair ≥ 0.40. Margin test remains valuable but is not load-bearing.")
    else:
        print(f"\n  ✓ Wide separation. Any reasonable threshold will work.")

    # Persist
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "Facenet512",
        "detector": "retinaface",
        "characters": [
            {
                "name": s.name,
                "n_refs": s.n_refs,
                "intra_mean": s.intra_mean,
                "intra_max": s.intra_max,
            }
            for s in stats
        ],
        "pairwise_centroid_distance": {
            names[i]: {names[j]: float(matrix[i, j]) for j in range(len(names))}
            for i in range(len(names))
        },
        "hardest_pair": {
            "a": hardest_pair[0],
            "b": hardest_pair[1],
            "distance": float(hardest_dist),
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\n  Wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
