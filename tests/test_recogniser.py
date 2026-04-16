"""Unit tests for the k-NN recogniser decision logic.

Uses synthetic embeddings (no DeepFace), so these run in milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nimbus.embedder import l2_normalise
from nimbus.recogniser import Recogniser
from nimbus.types import LABEL_UNKNOWN

DIM = 512
RNG = np.random.default_rng(seed=42)


def _basis(n: int, dim: int = DIM, rng: np.random.Generator = RNG) -> np.ndarray:
    """Return n random L2-normalised unit vectors of given dimension."""
    v = rng.normal(size=(n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _near(v: np.ndarray, jitter: float, rng: np.random.Generator = RNG) -> np.ndarray:
    """Return v perturbed by small-norm noise, re-normalised."""
    noise = rng.normal(size=v.shape).astype(np.float32) * jitter
    out = v + noise
    return out / np.linalg.norm(out)


def _write_refs(
    tmp_path: Path,
    per_char_refs: dict[str, np.ndarray],
    thresholds: dict[str, float],
    margin: float = 0.05,
    k: int = 3,
) -> tuple[Path, Path]:
    """Write a recogniser-compatible embeddings.npz + calibration.json pair."""
    emb_path = tmp_path / "embeddings.npz"
    calib_path = tmp_path / "calibration.json"
    np.savez(emb_path, **per_char_refs)
    calib = {
        "model": "Facenet512",
        "k": k,
        "global_margin": margin,
        "characters": {
            name: {"threshold": thresholds[name], "margin": margin, "n_refs": refs.shape[0]}
            for name, refs in per_char_refs.items()
        },
    }
    calib_path.write_text(json.dumps(calib))
    return emb_path, calib_path


# ------------------------------------------------------------------- init

def test_recogniser_raises_on_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build_references"):
        Recogniser(tmp_path / "nope.npz", tmp_path / "nope.json")

    (tmp_path / "embeddings.npz").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="validate_references"):
        Recogniser(tmp_path / "embeddings.npz", tmp_path / "nope.json")


def test_recogniser_loads_refs_and_calibration(tmp_path: Path) -> None:
    refs = {"harry": _basis(4), "ron": _basis(4)}
    thresholds = {"harry": 0.45, "ron": 0.45}
    emb, calib = _write_refs(tmp_path, refs, thresholds)

    rec = Recogniser(emb, calib)
    assert rec.k == 3
    assert set(rec.refs.keys()) == {"harry", "ron"}
    assert rec.thresholds == {"harry": 0.45, "ron": 0.45}
    # Refs should be float32 contiguous post-load (matmul-friendly).
    for name in rec.refs:
        assert rec.refs[name].dtype == np.float32
        assert rec.refs[name].flags["C_CONTIGUOUS"]


# ----------------------------------------------------------- decision logic

def test_recognise_confident_match_returns_character_label(tmp_path: Path) -> None:
    # Build a scenario where the query is near one character and far from the other.
    harry_seed = _basis(1)[0]
    ron_seed = _basis(1)[0]
    harry_refs = np.stack([_near(harry_seed, 0.01) for _ in range(5)])
    ron_refs = np.stack([_near(ron_seed, 0.01) for _ in range(5)])

    refs = {"harry": harry_refs, "ron": ron_refs}
    thresholds = {"harry": 0.3, "ron": 0.3}
    emb, calib = _write_refs(tmp_path, refs, thresholds)
    rec = Recogniser(emb, calib)

    # Query near harry, should land as "Harry".
    query = _near(harry_seed, 0.01)
    result = rec.recognise(query)

    assert result.label == "Harry"
    assert result.top1_name == "harry"
    assert result.top2_name == "ron"
    assert result.top1_distance < result.top2_distance
    assert 0.0 <= result.confidence <= 1.0


def test_recognise_falls_back_to_unknown_when_threshold_breached(tmp_path: Path) -> None:
    # Query distance exceeds threshold → Unknown even if it's closest.
    harry_seed = _basis(1)[0]
    ron_seed = _basis(1)[0]
    refs = {
        "harry": np.stack([_near(harry_seed, 0.01) for _ in range(5)]),
        "ron": np.stack([_near(ron_seed, 0.01) for _ in range(5)]),
    }
    # Very tight threshold — no realistic query passes.
    thresholds = {"harry": 0.001, "ron": 0.001}
    emb, calib = _write_refs(tmp_path, refs, thresholds)
    rec = Recogniser(emb, calib)

    query = _near(harry_seed, 0.05)
    result = rec.recognise(query)
    assert result.label == LABEL_UNKNOWN
    assert result.top1_name == "harry"  # still reports closest


def test_recognise_falls_back_to_unknown_when_margin_breached(tmp_path: Path) -> None:
    # Query is equidistant from harry and ron → margin test fails → Unknown.
    shared = _basis(1)[0]
    refs = {
        "harry": np.stack([_near(shared, 0.01) for _ in range(5)]),
        "ron": np.stack([_near(shared, 0.01) for _ in range(5)]),
    }
    thresholds = {"harry": 0.5, "ron": 0.5}
    emb, calib = _write_refs(tmp_path, refs, thresholds, margin=0.1)
    rec = Recogniser(emb, calib)

    query = _near(shared, 0.02)
    result = rec.recognise(query)
    assert result.label == LABEL_UNKNOWN
    # top1/top2 distances should be very close (< margin apart).
    assert (result.top2_distance - result.top1_distance) < 0.1


# ---------------------------------------------------- vectorisation parity

def test_knn_mean_matches_scalar_loop(tmp_path: Path) -> None:
    """The vectorised _knn_mean must equal the scalar-loop equivalent."""
    refs = {"harry": _basis(8), "ron": _basis(8)}
    thresholds = {"harry": 0.5, "ron": 0.5}
    emb, calib = _write_refs(tmp_path, refs, thresholds, k=3)
    rec = Recogniser(emb, calib)

    query = l2_normalise(RNG.normal(size=DIM).astype(np.float32))

    for name, ref_matrix in rec.refs.items():
        vectorised = rec._knn_mean(query, ref_matrix)
        # Scalar reference implementation.
        scalar_dists = np.array([1.0 - float(np.dot(query, r)) for r in ref_matrix])
        scalar_dists.sort()
        scalar = float(scalar_dists[:rec.k].mean())
        assert abs(vectorised - scalar) < 1e-5, name


def test_knn_mean_handles_fewer_refs_than_k(tmp_path: Path) -> None:
    # n_refs < k: should fall back to mean over all refs.
    refs = {"harry": _basis(2), "ron": _basis(5)}
    thresholds = {"harry": 0.5, "ron": 0.5}
    emb, calib = _write_refs(tmp_path, refs, thresholds, k=3)
    rec = Recogniser(emb, calib)

    query = l2_normalise(RNG.normal(size=DIM).astype(np.float32))
    out = rec._knn_mean(query, rec.refs["harry"])  # only 2 refs, k=3
    expected = float(np.mean([1.0 - float(np.dot(query, r)) for r in rec.refs["harry"]]))
    assert abs(out - expected) < 1e-5
