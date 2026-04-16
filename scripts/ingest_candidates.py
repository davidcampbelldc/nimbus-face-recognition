"""Ingest candidate reference images.

Takes raw candidates (mixed formats: jpg/jpeg/webp/avif), normalises each to
JPG via PIL (falling back to ffmpeg for AVIF if PIL can't decode), runs
RetinaFace detection, and stages survivors into `references/<character>/NN.jpg`.

Rejection rules (per plan §6):
  - can't decode
  - RetinaFace detects 0 or >1 faces

One-shot utility. Re-runs are idempotent (overwrites references/<char>/*.jpg).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCES_DIR = REPO_ROOT / "references"

# Plan §6: below this, the character is under-represented and recognition will suffer.
MIN_REFS_PER_CHARACTER = 3

# Mapping: character → list of candidate filenames (relative to source dir).
# harry3.webp (PoA-era, older) and ron1.webp (Chamber-era?) dropped per review.
CANDIDATES: dict[str, list[str]] = {
    "harry":      ["harry.webp", "harry2.jpeg", "harry4.webp", "harry5.avif"],
    "hermione":   ["hermione.jpg", "her2.jpeg", "her3.jpg", "her4.avif", "her5.jpg"],
    "ron":        ["ron2.jpg", "ron3.jpg", "ron3.webp",
                   "ron4.jpg", "ron5.webp", "ron6.png", "ron7.jpg", "ron8.jpg"],
    "mcgonagall": ["mc1.avif", "mc2.webp", "mc3.avif", "mc4.jpeg", "mc5.webp"],
    "snape":      ["snape.avif", "snape2.jpg", "snape3.jpg", "snape4.jpeg", "snape5.webp"],
}


@dataclass
class IngestResult:
    source: str
    character: str
    status: str  # "accepted" | "decode_fail" | "no_face" | "multi_face"
    detail: str = ""
    dest: str = ""


def decode_to_jpg(src: Path, dest: Path) -> bool:
    """Decode any supported format to JPG. Returns True on success."""
    try:
        img = Image.open(src).convert("RGB")
        img.save(dest, "JPEG", quality=95)
        return True
    except Exception:
        # PIL can't handle AVIF without libavif plugin; fall back to ffmpeg.
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                 "-frames:v", "1", str(dest)],
                check=True,
                capture_output=True,
            )
            return dest.exists() and dest.stat().st_size > 0
        except Exception:
            return False


def detect_faces(jpg_path: Path) -> int:
    """Return face count via DeepFace RetinaFace. -1 on error."""
    # Lazy import — DeepFace init is slow; do it once, not at module import.
    from deepface import DeepFace
    try:
        img = cv2.imread(str(jpg_path))
        if img is None:
            return -1
        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend="retinaface",
            enforce_detection=False,
            align=False,
        )
        # DeepFace returns at least one entry even with enforce_detection=False;
        # filter for real detections (confidence > 0).
        real = [f for f in faces if f.get("confidence", 0) > 0]
        return len(real)
    except Exception as e:
        print(f"  [detect error] {e}", file=sys.stderr)
        return -1


def format_result_line(fname: str, result: IngestResult) -> str:
    """Human-readable per-candidate line for the ingest log."""
    if result.status == "accepted":
        return f"✓ {fname} → {result.dest}"
    if result.status == "decode_fail" and "source not found" in result.detail:
        return f"✗ {fname} — source missing"
    if result.status == "decode_fail":
        return f"✗ {fname} — decode failed (PIL + ffmpeg both)"
    if result.status == "no_face":
        return f"✗ {fname} — RetinaFace detected 0 faces"
    if result.status == "multi_face":
        return f"✗ {fname} — multi-face ({result.detail}); reject"
    return f"? {fname} — {result.status} {result.detail}"


def process_candidate(
    fname: str, character: str, source_dir: Path, char_dir: Path, accepted_so_far: int
) -> IngestResult:
    """Decode + detect one candidate. Returns result; caller tracks accepted count."""
    src = source_dir / fname
    if not src.exists():
        return IngestResult(fname, character, "decode_fail", f"source not found: {src}")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        if not decode_to_jpg(src, tmp_path):
            return IngestResult(fname, character, "decode_fail")

        n_faces = detect_faces(tmp_path)
        if n_faces == 0:
            return IngestResult(fname, character, "no_face")
        if n_faces > 1:
            return IngestResult(fname, character, "multi_face", f"{n_faces} faces")

        dest = char_dir / f"{accepted_so_far + 1:02d}.jpg"
        shutil.move(str(tmp_path), dest)
        return IngestResult(
            fname, character, "accepted", dest=str(dest.relative_to(REPO_ROOT))
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main(source_dir: Path) -> None:
    print(f"Ingesting candidates from: {source_dir}")
    print(f"Destination root:          {REFERENCES_DIR}\n")

    results: list[IngestResult] = []

    for character, filenames in CANDIDATES.items():
        char_dir = REFERENCES_DIR / character
        # Clear prior runs' staged jpgs (but keep _clip_frames_used.json etc.).
        for existing in char_dir.glob("*.jpg"):
            existing.unlink()
        char_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== {character} ===")
        accepted = 0
        for fname in filenames:
            result = process_candidate(fname, character, source_dir, char_dir, accepted)
            results.append(result)
            if result.status == "accepted":
                accepted += 1
            print(f"  {format_result_line(fname, result)}")

    # Summary
    print("\n=== Summary ===")
    by_char: dict[str, int] = {}
    for r in results:
        if r.status == "accepted":
            by_char[r.character] = by_char.get(r.character, 0) + 1
    for character in CANDIDATES:
        n = by_char.get(character, 0)
        marker = "✓" if n >= MIN_REFS_PER_CHARACTER else "⚠ LOW"
        print(f"  {character:12s} {n} accepted   {marker}")

    rejects = [r for r in results if r.status != "accepted"]
    if rejects:
        print(f"\n  {len(rejects)} rejected:")
        for r in rejects:
            print(f"    - {r.character}/{r.source}: {r.status} {r.detail}")


DEFAULT_SOURCE = (
    Path.home() / "Documents" / "loadmagic" / "magic-mandy"
    / "docs" / "myCV" / "recruitment-assessments" / "images"
)

if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    if not src.is_dir():
        print(f"Source not found: {src}", file=sys.stderr)
        sys.exit(1)
    main(src)
