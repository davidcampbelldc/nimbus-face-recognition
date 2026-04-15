"""Unit tests for the IoU tracker + label hysteresis."""

from __future__ import annotations

from nimbus.tracker import Tracker, iou
from nimbus.types import Detection


def _det(bbox: tuple[int, int, int, int], label: str = "Face", conf: float = 0.9) -> Detection:
    return Detection(bbox=bbox, confidence=0.99, label=label, label_confidence=conf)


# ---------------------------------------------------------------------- IoU

def test_iou_identical_boxes_is_one() -> None:
    assert iou((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0


def test_iou_disjoint_boxes_is_zero() -> None:
    assert iou((0, 0, 100, 100), (200, 200, 50, 50)) == 0.0


def test_iou_half_overlap() -> None:
    # Two 100x100 boxes overlapping in a 50x100 strip.
    # intersection = 50*100 = 5000
    # union = 10000 + 10000 - 5000 = 15000
    # IoU = 1/3
    assert abs(iou((0, 0, 100, 100), (50, 0, 100, 100)) - 1 / 3) < 1e-6


def test_iou_contained_box() -> None:
    # 50x50 inside 100x100. intersection=2500, union=10000. IoU=0.25.
    assert abs(iou((0, 0, 100, 100), (25, 25, 50, 50)) - 0.25) < 1e-6


# ------------------------------------------------------------- Track lifecycle

def test_single_detection_creates_one_track() -> None:
    tr = Tracker(history_len=5)
    out = tr.update([_det((100, 100, 50, 50), label="Harry")], scene_cut=False)
    assert len(out) == 1
    assert out[0].label == "Harry"


def test_matching_detection_across_frames_keeps_same_track() -> None:
    tr = Tracker(history_len=5, max_missed_frames=3)
    # Frame 1
    tr.update([_det((100, 100, 50, 50), label="Harry")], scene_cut=False)
    ids1 = tr.active_ids()
    # Frame 2: small drift → IoU still high
    tr.update([_det((102, 101, 50, 50), label="Harry")], scene_cut=False)
    ids2 = tr.active_ids()
    assert ids1 == ids2
    assert len(ids1) == 1


def test_disjoint_detection_creates_new_track() -> None:
    tr = Tracker(history_len=5)
    tr.update([_det((100, 100, 50, 50), label="Harry")], scene_cut=False)
    # Second face far away — no IoU overlap with first
    tr.update(
        [
            _det((100, 100, 50, 50), label="Harry"),
            _det((500, 500, 50, 50), label="Ron"),
        ],
        scene_cut=False,
    )
    assert len(tr.active_ids()) == 2


def test_track_dropped_after_missed_frames() -> None:
    tr = Tracker(history_len=5, max_missed_frames=2)
    tr.update([_det((100, 100, 50, 50), label="Harry")], scene_cut=False)
    # Three frames with no detections — track should age out (>2 missed).
    tr.update([], scene_cut=False)
    tr.update([], scene_cut=False)
    tr.update([], scene_cut=False)
    assert len(tr.active_ids()) == 0


# ---------------------------------------------------------- Label hysteresis

def test_majority_label_wins_over_single_spike() -> None:
    tr = Tracker(history_len=5)
    bbox = (100, 100, 50, 50)
    # Three Harry votes first.
    for _ in range(3):
        tr.update([_det(bbox, label="Harry", conf=0.9)], scene_cut=False)
    # One noisy "Unknown" spike.
    out = tr.update([_det(bbox, label="Unknown", conf=0.3)], scene_cut=False)
    # Majority in last 4 frames is Harry → smoothed output should say Harry.
    assert out[0].label == "Harry"


def test_recent_majority_flips_label_after_enough_evidence() -> None:
    tr = Tracker(history_len=5)
    bbox = (100, 100, 50, 50)
    # Two initial Unknown reads.
    tr.update([_det(bbox, label="Unknown", conf=0.3)], scene_cut=False)
    tr.update([_det(bbox, label="Unknown", conf=0.3)], scene_cut=False)
    # Three Harry reads — now majority is Harry.
    tr.update([_det(bbox, label="Harry", conf=0.8)], scene_cut=False)
    tr.update([_det(bbox, label="Harry", conf=0.85)], scene_cut=False)
    out = tr.update([_det(bbox, label="Harry", conf=0.9)], scene_cut=False)
    assert out[0].label == "Harry"


def test_smoothed_confidence_matches_smoothed_label_history() -> None:
    """When smoothing flips the label away from the current frame's raw
    label, the rendered confidence should be the history-mean for the
    smoothed label, not this frame's raw confidence."""
    tr = Tracker(history_len=5)
    bbox = (100, 100, 50, 50)
    for _ in range(3):
        tr.update([_det(bbox, label="Harry", conf=0.9)], scene_cut=False)
    out = tr.update([_det(bbox, label="Unknown", conf=0.3)], scene_cut=False)
    assert out[0].label == "Harry"
    # Confidence should reflect Harry's recent history, not the Unknown 0.3.
    assert out[0].label_confidence is not None
    assert out[0].label_confidence > 0.5


# --------------------------------------------------------- Scene-cut flush

def test_scene_cut_flushes_all_tracks() -> None:
    tr = Tracker(history_len=5)
    tr.update([_det((100, 100, 50, 50), label="Harry")], scene_cut=False)
    assert len(tr.active_ids()) == 1
    tr.update([], scene_cut=True)
    assert len(tr.active_ids()) == 0


def test_new_detections_after_cut_get_fresh_ids() -> None:
    tr = Tracker(history_len=5)
    tr.update([_det((100, 100, 50, 50), label="Harry")], scene_cut=False)
    ids_before = set(tr.active_ids())
    # Scene cut + new detection in the same bbox position
    tr.update([_det((100, 100, 50, 50), label="Harry")], scene_cut=True)
    ids_after = set(tr.active_ids())
    # New track must have a fresh ID (no bleed)
    assert not (ids_before & ids_after)
