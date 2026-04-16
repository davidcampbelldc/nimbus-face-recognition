# Nimbus — Face Detection + Character Recognition

Identifies named *Harry Potter* characters (Harry, Ron, Hermione, McGonagall,
Snape) in a 101-second clip from *Philosopher's Stone*, using
[DeepFace](https://github.com/serengil/deepface) with RetinaFace detection
and Facenet512 embeddings.

Built as a take-home assessment for White Swan Data.

---

## Results

Evaluated on a 40-frame stratified sample (76 GT faces) from the full clip.
All numbers include 95% bootstrap CIs (1000 resamples).

| Character      | Precision | Recall | F1    | F1 95% CI          | GT faces |
|----------------|-----------|--------|-------|--------------------|----------|
| Harry          | 1.000     | 0.783  | **0.878** | [0.743, 0.974] | 23       |
| Snape          | 1.000     | 0.667  | **0.800** | [0.333, 1.000] | 6        |
| Ron            | 1.000     | 0.500  | 0.667 | [0.000, 1.000]     | 6        |
| Hermione       | 1.000     | 0.364  | 0.533 | [0.167, 0.800]     | 11       |
| McGonagall     | 0.000     | 0.000  | 0.000 | [0.000, 0.000]     | 1        |
| Unknown        | 0.617     | 1.000  | 0.763 | [0.462, 0.907]     | 29       |
| **Macro avg**  |           |        | **0.607** |                |          |

**Detection recall: 100%** (76/76) — every face got a bounding box,
including back-of-head and profile shots that the classifier can't label.

**Precision = 1.000 on every named class** — the system never confidently
mislabels a face. When uncertain, it falls back to `Unknown`.

Raw numbers: [`eval/metrics.json`](eval/metrics.json) —
methodology: [§Evaluation](#evaluation) —
known limits: [§Known limitations](#known-limitations) —
design decisions: [NOTES.md](NOTES.md).

## Output video

_(Drive link to `nimbus_final.mp4` goes here before submission.)_

---

## Quickstart

```bash
git clone <repo-url>
cd whiteswan-deepface
pip install -e .
python run.py data/input/nimbus.mp4 data/output/nimbus_annotated.mp4
```

**First run** downloads ~250MB of model weights (Facenet512 + RetinaFace)
from DeepFace's hosted repo. One-time, cached to `~/.deepface/` thereafter.

**Prerequisites:**
- Python 3.11 (pinned in `pyproject.toml`)
- `ffmpeg` on PATH (used for h264 re-encode; near-universal install)

Tested on Linux x86_64 and macOS arm64.

---

## Smoke mode

Quick ~1-minute sanity check before a full render:

```bash
python run.py --frames 60 data/input/nimbus.mp4 data/output/nimbus_smoke.mp4
```

Full render is ~45 min – 2 hrs on CPU depending on hardware and face density.

---

## What's in the repo

```
src/nimbus/        # pipeline: detector, embedder, recogniser, tracker, renderer, scene_cut
scripts/           # ref building, baseline probe, evaluation, plots, diagnostics
references/        # curated reference stills per character (26 total)
refs/              # generated embeddings + LOO-calibrated thresholds (committed)
tests/             # 27 pytest unit tests (tracker, recogniser, scene_cut)
eval/              # ground truth + computed metrics + distribution plots
NOTES.md           # design rationale — read this for decisions + trade-offs
```

---

## Architecture

```
Reader → Detector → Embedder → Recogniser → Render → h264 writer
          (RetinaFace) (Facenet512) (k-NN + threshold)    ↑
                                          ↓               │
                                        Tracker ──────────┘
                                        (IoU + hysteresis,
                                         flushed on scene cut)
```

- **Detector** — DeepFace's RetinaFace. Returns bounding box + aligned crop per face.
- **Embedder** — DeepFace's Facenet512. 512-d L2-normalised vector per aligned face.
- **Recogniser** — k-NN(k=3) cosine distance against curated references,
  per-class thresholds calibrated leave-one-out, plus a top-1 vs top-2
  margin gate that falls back to `Unknown` when uncertain.
- **Tracker** — IoU-matched across frames, label smoothed by history mode.
  Flushed on scene cuts so labels don't bleed across shots.
- **Renderer** — coloured bounding boxes + confidence-suffixed label;
  h264 mp4 output via ffmpeg re-encode.

Full design rationale in [NOTES.md](NOTES.md).

---

## Evaluation

- 40 frames sampled stratified across the 3044-frame clip
- Each frame's faces hand-labelled via an anchoring-free VLM
  pre-labelling workflow (VLM proposes, human verifies flagged items)
- IoU-matching (≥ 0.5) to align predictions with GT
- Per-class precision/recall/F1 + confusion matrix + 95% bootstrap CIs
- `Unknown` is a first-class class — not excluded from the macro average

Reproduce:

```bash
python scripts/evaluate.py           # regenerates eval/metrics.json
python scripts/plot_distributions.py # regenerates eval/plots/
```

Deeper methodology (stratified sampling, near-duplicate leakage guard,
no-tuning-on-test discipline): [NOTES.md §6](NOTES.md#6-evaluation-methodology).

---

## Known limitations

- **Profile faces** (>45° yaw) and **back-of-head shots** are detected but
  intentionally labelled `Unknown` — Facenet512 can't embed them reliably.
- **McGonagall** in this clip appears in exactly one sampled frame. The
  model correctly identifies her but misses by **0.005 cosine-distance
  units** against a tight per-class threshold.
  See [NOTES.md §8](NOTES.md#8-the-mcgonagall-near-miss) for the full
  decision trace and why threshold tuning was deliberately avoided.
- **Overhead / extreme-angle shots** degrade both detection and recognition
  — known limit of the models used as recommended.
- **First run is slow** — ~250MB of model weights are downloaded on first
  invocation. Subsequent runs use the cached weights in `~/.deepface/`.

---

## Development

Tests:
```bash
pip install -e .[dev]
pytest           # 27 tests, < 1s, no DeepFace dependency
ruff check .     # lint
```

Build references from scratch:
```bash
python scripts/ingest_candidates.py    # normalise candidate stills
python scripts/build_references.py     # embed via Facenet512
python scripts/validate_references.py  # LOO calibration → refs/calibration.json
python scripts/baseline_probe.py       # feature-space geometry sanity check
```

Diagnose a specific frame:
```bash
python scripts/diagnose_miss.py --frame 2965 --verbose
```

---

## Design notes

See [`NOTES.md`](NOTES.md) for design decisions, trade-offs, epistemic
boundaries, and what I'd do with another week.

## Citations

- [DeepFace](https://github.com/serengil/deepface) — Serengil & Ozpinar (2020).
- [RetinaFace](https://arxiv.org/abs/1905.00641) — Deng et al. (2019).
- [Facenet](https://arxiv.org/abs/1503.03832) — Schroff, Kalenichenko & Philbin (2015).

## License

MIT — see [`LICENSE`](LICENSE).
