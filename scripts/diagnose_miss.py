#!/usr/bin/env python3
"""Diagnose a single frame: run detect + embed, then for each detection
report the recogniser's full decision trace (top1, top2, threshold,
margin), so we can explain *why* a prediction came out the way it did.

Useful for understanding individual eval misses — e.g. the McGonagall
miss at f02965: is her embedding close but just over the threshold, or
genuinely far from her references?

Usage:
    .venv/bin/python scripts/diagnose_miss.py --frame 2965
    .venv/bin/python scripts/diagnose_miss.py --frame 2965 --verbose
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
from nimbus.embedder import Embedder, cosine_distance  # noqa: E402

EMBEDDINGS = REPO_ROOT / "refs" / "embeddings.npz"
CALIBRATION = REPO_ROOT / "refs" / "calibration.json"
FRAMES_DIR = REPO_ROOT / "eval" / "frames_raw"


def _diagnose_detection(
    det,
    idx: int,
    embedder,
    refs: dict,
    thresholds: dict,
    margins: dict,
    k: int,
    verbose: bool,
) -> None:
    print(f"  detection B{idx}: bbox={det.bbox} det_conf={det.confidence:.3f}")
    if det.aligned_face is None:
        print("    (no aligned face — skip)")
        return
    emb = embedder.embed_aligned_face(det.aligned_face)

    scored = []
    per_ref_dists: dict[str, list[float]] = {}
    for name, vecs in refs.items():
        dists = np.array([cosine_distance(emb, r) for r in vecs])
        per_ref_dists[name] = sorted(dists.tolist())
        dists.sort()
        eff_k = min(k, len(vecs))
        scored.append((name, float(dists[:eff_k].mean())))
    scored.sort(key=lambda pair: pair[1])

    top1_name, top1_dist = scored[0]
    top2_name, top2_dist = scored[1]
    thr = thresholds[top1_name]
    mg = margins[top1_name]
    passes_thr = top1_dist < thr
    passes_mg = (top2_dist - top1_dist) > mg
    label = top1_name.capitalize() if (passes_thr and passes_mg) else "Unknown"

    print(f"    top1 = {top1_name:10s} dist={top1_dist:.4f}  "
          f"threshold={thr:.4f}  {'PASS' if passes_thr else 'FAIL (too far)'}")
    print(f"    top2 = {top2_name:10s} dist={top2_dist:.4f}  "
          f"gap={top2_dist - top1_dist:.4f}  margin={mg:.4f}  "
          f"{'PASS' if passes_mg else 'FAIL (too similar to top2)'}")
    print(f"    → final label: {label}")
    if verbose:
        print("    per-character knn-mean distances:")
        for name, d in scored:
            print(f"      {name:12s} knn-mean={d:.4f}  "
                  f"(indiv: {', '.join(f'{x:.3f}' for x in per_ref_dists[name])})")
    print()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--verbose", action="store_true",
                   help="also print distance to every individual reference")
    args = p.parse_args()

    frame_path = FRAMES_DIR / f"f{args.frame:05d}.png"
    if not frame_path.exists():
        print(f"error: {frame_path} missing", file=sys.stderr)
        return 2

    refs_data = np.load(EMBEDDINGS)
    refs = {name: refs_data[name] for name in refs_data.files}
    calib = json.loads(CALIBRATION.read_text())
    thresholds = {n: float(e["threshold"]) for n, e in calib["characters"].items()}
    margins = {n: float(e.get("margin", calib["global_margin"]))
               for n, e in calib["characters"].items()}
    k = int(calib["k"])

    print(f"=== diagnosing f{args.frame:05d} ===")
    print(f"  refs loaded: {sorted(refs)}")
    print(f"  k={k}, global_margin={calib['global_margin']}\n")

    detector = FaceDetector()
    embedder = Embedder()

    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"error: cv2 could not read {frame_path}", file=sys.stderr)
        return 2

    dets = detector.detect(frame)
    print(f"detector found {len(dets)} face(s)\n")
    for i, det in enumerate(dets):
        _diagnose_detection(det, i, embedder, refs, thresholds, margins, k, args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
