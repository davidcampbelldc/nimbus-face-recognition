"""Character recogniser: k-NN(k=3) with per-class threshold + margin.

Loads the pre-computed reference embeddings and calibration (from
build_references.py + validate_references.py), then for each query face
embedding decides which named character (if any) it matches.

Decision logic per plan §5:
  1. For each known character, compute the mean cosine distance from the
     query to the k=3 nearest same-character reference embeddings.
  2. Let top1 = closest character, top2 = second closest.
  3. Accept top1's label iff:
       - top1_mean_distance < threshold[top1_name], AND
       - top2_mean_distance - top1_mean_distance > margin.
     Otherwise return "Unknown".

The margin test is the quant-honesty gate: a query that's almost as close
to Hermione as to Ron gets "Unknown" rather than a coin-flip guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embedder import cosine_distance


@dataclass(frozen=True)
class RecognitionResult:
    label: str           # character name or "Unknown"
    confidence: float    # in [0, 1]; 1 - top1_mean_distance (how close we got)
    top1_name: str       # the closest named character (even if label is Unknown)
    top1_distance: float
    top2_name: str
    top2_distance: float


class Recogniser:
    """k-NN recogniser with per-class thresholds loaded from calibration.json."""

    def __init__(
        self,
        embeddings_path: Path,
        calibration_path: Path,
    ) -> None:
        if not embeddings_path.exists():
            raise FileNotFoundError(f"{embeddings_path} missing — run build_references.py")
        if not calibration_path.exists():
            raise FileNotFoundError(f"{calibration_path} missing — run validate_references.py")

        data = np.load(embeddings_path)
        self.refs: dict[str, np.ndarray] = {n: data[n] for n in data.files}

        calib = json.loads(calibration_path.read_text())
        self.k = int(calib["k"])
        self.global_margin = float(calib["global_margin"])
        self.thresholds: dict[str, float] = {
            name: float(entry["threshold"])
            for name, entry in calib["characters"].items()
        }
        self.margins: dict[str, float] = {
            name: float(entry.get("margin", self.global_margin))
            for name, entry in calib["characters"].items()
        }

    def _knn_mean(self, query: np.ndarray, refs: np.ndarray) -> float:
        n = refs.shape[0]
        if n == 0:
            return float("inf")
        effective_k = min(self.k, n)
        dists = np.array([cosine_distance(query, r) for r in refs])
        dists.sort()
        return float(dists[:effective_k].mean())

    def recognise(self, query: np.ndarray) -> RecognitionResult:
        """Classify a query embedding. Query must be L2-normalised."""
        # Per-character k-NN mean distance.
        scored = [
            (name, self._knn_mean(query, self.refs[name]))
            for name in self.refs
        ]
        scored.sort(key=lambda pair: pair[1])
        top1_name, top1_dist = scored[0]
        top2_name, top2_dist = scored[1]

        threshold = self.thresholds[top1_name]
        margin = self.margins[top1_name]

        passes_threshold = top1_dist < threshold
        passes_margin = (top2_dist - top1_dist) > margin

        label = top1_name.capitalize() if (passes_threshold and passes_margin) else "Unknown"

        confidence = max(0.0, min(1.0, 1.0 - top1_dist))

        return RecognitionResult(
            label=label,
            confidence=confidence,
            top1_name=top1_name,
            top1_distance=top1_dist,
            top2_name=top2_name,
            top2_distance=top2_dist,
        )
