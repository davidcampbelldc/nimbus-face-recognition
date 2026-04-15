"""IoU-based face tracker with label hysteresis + scene-cut flush.

Purpose (plan §5 tracking strategy):
  - Associate detections across consecutive frames so we can smooth the
    recogniser's per-frame labels into a stable per-track label. Without
    this, the output video strobes — one frame "Harry", next "Unknown",
    next "Harry" — because the recogniser is noisy near thresholds.
  - Flush tracks on scene cuts. Label hysteresis across cuts would
    contaminate new scenes with the previous scene's labels.

Design:
  - Greedy IoU matching (acknowledged tradeoff vs Hungarian in NOTES.md —
    greedy fails only under overlapping bounding boxes which RetinaFace
    rarely emits for human faces at our scales).
  - Mode smoothing over a fixed-length history. Majority class wins ties
    in dict insertion order — sufficient for v1.
  - Confidence is smoothed *for the chosen label*: when the smoothed label
    disagrees with the current frame's raw label, the confidence reported
    is the mean confidence across frames whose raw label matched the
    smoothed choice. Keeps the displayed confidence coherent with the
    displayed label.

State is all private. `update(...)` returns a list of Detection objects
with smoothed label/confidence so pipeline.py + renderer.py don't need
to know about Track dataclasses.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field, replace

from .types import Bbox, Detection


def iou(a: Bbox, b: Bbox) -> float:
    """Intersection-over-Union of two (x, y, w, h) boxes. Returns 0 for
    degenerate or disjoint boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    last_bbox: Bbox
    last_seen_frame: int
    label_history: deque[str] = field(default_factory=deque)
    conf_history: deque[float] = field(default_factory=deque)


class Tracker:
    """IoU tracker with label hysteresis. Stateless between videos; construct
    a fresh Tracker per run. `reset()` is provided for explicit clears."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        history_len: int = 7,
        max_missed_frames: int = 5,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.history_len = history_len
        self.max_missed_frames = max_missed_frames
        self._tracks: list[_Track] = []
        self._next_id = 0
        self._frame_idx = 0

    def reset(self) -> None:
        self._tracks = []
        # Don't reset _next_id — keeps track_ids monotonic across resets for
        # easier debugging across a single run.

    def active_ids(self) -> list[int]:
        return [t.track_id for t in self._tracks]

    # --------------------------------------------------------------- update

    def update(self, detections: list[Detection], scene_cut: bool) -> list[Detection]:
        self._frame_idx += 1

        if scene_cut:
            self._tracks = []

        assignments = self._greedy_match(detections)
        assigned_track_ids: set[int] = set()
        assigned_det_indices: set[int] = set()
        out: list[Detection] = [None] * len(detections)  # type: ignore[list-item]

        for det_idx, track_idx in assignments:
            det = detections[det_idx]
            track = self._tracks[track_idx]
            self._extend_history(track, det)
            track.last_bbox = det.bbox
            track.last_seen_frame = self._frame_idx
            assigned_track_ids.add(track.track_id)
            assigned_det_indices.add(det_idx)
            out[det_idx] = self._smoothed_detection(det, track)

        # Unmatched detections → new tracks.
        for det_idx, det in enumerate(detections):
            if det_idx in assigned_det_indices:
                continue
            track = self._spawn_track(det)
            out[det_idx] = self._smoothed_detection(det, track)

        self._prune_stale_tracks()
        return out

    # ------------------------------------------------------------- internals

    def _greedy_match(self, detections: list[Detection]) -> list[tuple[int, int]]:
        """Return list of (det_idx, track_idx) assignments in descending IoU
        order. Greedy — each det and each track is claimed at most once."""
        if not self._tracks or not detections:
            return []

        candidates: list[tuple[float, int, int]] = []
        for di, det in enumerate(detections):
            for ti, track in enumerate(self._tracks):
                score = iou(det.bbox, track.last_bbox)
                if score >= self.iou_threshold:
                    candidates.append((score, di, ti))

        candidates.sort(reverse=True)  # highest IoU first

        claimed_det: set[int] = set()
        claimed_track: set[int] = set()
        out: list[tuple[int, int]] = []
        for _score, di, ti in candidates:
            if di in claimed_det or ti in claimed_track:
                continue
            claimed_det.add(di)
            claimed_track.add(ti)
            out.append((di, ti))
        return out

    def _spawn_track(self, det: Detection) -> _Track:
        track = _Track(
            track_id=self._next_id,
            last_bbox=det.bbox,
            last_seen_frame=self._frame_idx,
            label_history=deque(maxlen=self.history_len),
            conf_history=deque(maxlen=self.history_len),
        )
        self._next_id += 1
        self._extend_history(track, det)
        self._tracks.append(track)
        return track

    def _extend_history(self, track: _Track, det: Detection) -> None:
        track.label_history.append(det.label)
        track.conf_history.append(
            det.label_confidence if det.label_confidence is not None else 0.0
        )

    def _smoothed_detection(self, current: Detection, track: _Track) -> Detection:
        smoothed_label = Counter(track.label_history).most_common(1)[0][0]
        # Confidence consistent with the chosen label: mean confidence across
        # history entries whose raw label equalled the smoothed label.
        matching = [
            c for lbl, c in zip(track.label_history, track.conf_history, strict=True)
            if lbl == smoothed_label
        ]
        smoothed_conf = sum(matching) / len(matching) if matching else current.label_confidence
        return replace(
            current,
            label=smoothed_label,
            label_confidence=smoothed_conf,
        )

    def _prune_stale_tracks(self) -> None:
        self._tracks = [
            t for t in self._tracks
            if (self._frame_idx - t.last_seen_frame) <= self.max_missed_frames
        ]
