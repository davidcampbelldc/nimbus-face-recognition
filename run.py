#!/usr/bin/env python3
"""One-shot CLI: detect faces in a video, write annotated output.

Usage:
    python run.py input.mp4 output.mp4
    python run.py --frames 30 input.mp4 smoke.mp4    # smoke-test 30 frames
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nimbus.pipeline import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Face detection + character recognition pipeline.")
    parser.add_argument("video_in", type=Path, help="input video (mp4)")
    parser.add_argument("video_out", type=Path, help="output video path (mp4)")
    parser.add_argument("--frames", type=int, default=None,
                        help="process only the first N frames (smoke mode)")
    parser.add_argument("--no-progress", action="store_true",
                        help="hide tqdm progress bar")
    parser.add_argument("--no-recognise", action="store_true",
                        help="detection-only mode (skip embedder + recogniser)")
    parser.add_argument("--no-track", action="store_true",
                        help="disable tracker/label-smoothing (emit raw per-frame labels)")
    parser.add_argument("--downsample", type=int, default=None,
                        help="resize frame so short side = N px before detection "
                             "(e.g. 540 → ~4× speedup at small accuracy cost)")
    args = parser.parse_args()

    try:
        stats = run(
            video_in=args.video_in,
            video_out=args.video_out,
            frame_limit=args.frames,
            show_progress=not args.no_progress,
            recognise=not args.no_recognise,
            track=not args.no_track,
            downsample=args.downsample,
        )
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print()
    print(f"Frames processed: {stats.frames_processed}")
    print(f"Total detections: {stats.total_detections}")
    print(f"Scene cuts:       {stats.scene_cuts}")
    print(f"Runtime:          {stats.runtime_seconds:.1f}s ({stats.fps:.2f} fps)")
    print(f"Output:           {args.video_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
