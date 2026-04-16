"""Unit tests for the MABS scene-cut detector."""

from __future__ import annotations

import numpy as np

from nimbus.scene_cut import SceneCutDetector


def _frame(value: int, h: int = 120, w: int = 160) -> np.ndarray:
    """Uniform BGR frame filled with `value` in all channels (uint8)."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_first_frame_is_never_a_cut() -> None:
    sc = SceneCutDetector()
    assert sc.is_cut(_frame(128)) is False


def test_identical_consecutive_frames_not_a_cut() -> None:
    sc = SceneCutDetector(threshold=25.0)
    sc.is_cut(_frame(128))
    assert sc.is_cut(_frame(128)) is False


def test_large_pixel_change_is_a_cut() -> None:
    sc = SceneCutDetector(threshold=25.0)
    sc.is_cut(_frame(0))      # prime with all-black
    # All-white after all-black: MABS = 255 >> threshold
    assert sc.is_cut(_frame(255)) is True


def test_small_pixel_change_below_threshold_not_a_cut() -> None:
    sc = SceneCutDetector(threshold=25.0)
    sc.is_cut(_frame(100))
    # A 10-unit shift across every pixel → MABS = 10, well below 25
    assert sc.is_cut(_frame(110)) is False


def test_threshold_is_honoured_at_the_boundary() -> None:
    # Threshold = 30; shift by 25 should stay below, shift by 35 should cut.
    sc_below = SceneCutDetector(threshold=30.0)
    sc_below.is_cut(_frame(0))
    assert sc_below.is_cut(_frame(25)) is False

    sc_above = SceneCutDetector(threshold=30.0)
    sc_above.is_cut(_frame(0))
    assert sc_above.is_cut(_frame(35)) is True


def test_reset_clears_previous_frame() -> None:
    sc = SceneCutDetector(threshold=25.0)
    sc.is_cut(_frame(0))
    sc.reset()
    # After reset, next frame is again the "first" — never a cut.
    assert sc.is_cut(_frame(255)) is False


def test_downsample_does_not_break_cut_detection() -> None:
    # A non-uniform frame with a real change should still register a cut
    # when heavily downsampled (the module's default is /4).
    rng = np.random.default_rng(seed=0)
    a = rng.integers(0, 60, size=(120, 160, 3), dtype=np.uint8)
    b = rng.integers(200, 256, size=(120, 160, 3), dtype=np.uint8)

    sc = SceneCutDetector(threshold=25.0, downsample=4)
    sc.is_cut(a)
    assert sc.is_cut(b) is True
