"""Build the reference embedding store.

Reads every `references/<character>/*.jpg`, embeds it via Facenet512, and
writes all embeddings to `refs/embeddings.npz` (one key per character,
(n_refs, 512) L2-normalised float32).

This is the authoritative input for the recogniser. Re-run whenever the
reference set changes. Idempotent — overwrites the npz.

Run from repo root:
    .venv/bin/python scripts/build_references.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from nimbus.embedder import Embedder  # noqa: E402

REFERENCES_DIR = REPO_ROOT / "references"
OUT_PATH = REPO_ROOT / "refs" / "embeddings.npz"
CHARACTERS = ["harry", "ron", "hermione", "mcgonagall", "snape"]


def main() -> int:
    embedder = Embedder()
    payload: dict[str, np.ndarray] = {}

    for character in CHARACTERS:
        char_dir = REFERENCES_DIR / character
        jpgs = sorted(char_dir.glob("*.jpg"))
        if not jpgs:
            print(f"error: no references for {character}", file=sys.stderr)
            return 1

        vecs: list[np.ndarray] = []
        print(f"Embedding {character}... ", end="", flush=True)
        for jpg in jpgs:
            try:
                vecs.append(embedder.embed_image_path(jpg))
            except Exception as e:
                print(f"\n  error on {jpg.name}: {e}", file=sys.stderr)
                return 2
        stacked = np.stack(vecs).astype(np.float32)
        payload[character] = stacked
        print(f"{stacked.shape[0]} refs")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH, **payload)
    total = sum(v.shape[0] for v in payload.values())
    print(f"\nWrote {OUT_PATH.relative_to(REPO_ROOT)} — {total} embeddings across "
          f"{len(payload)} characters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
