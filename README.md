# Nimbus — Face Detection + Character Recognition

Identifies named *Harry Potter* characters (Harry, Ron, Hermione, McGonagall,
Snape) in a 101-second clip from *Philosopher's Stone*, using
[DeepFace](https://github.com/serengil/deepface) with RetinaFace detection
and Facenet512 embeddings.

Built as a take-home assessment for White Swan Data.

---

## In plain English

**The problem.** Given a video, find every face and — where possible —
name the specific character.

**The approach.** Use off-the-shelf AI models to do two jobs:
1. **Find faces** — a face detector draws a box around every visible face,
   whether the system can name them or not.
2. **Name faces** — a face recogniser turns each face into a list of
   numbers (a "fingerprint"), compares it against a small set of reference
   photos we collected for each target character, and assigns a name if
   the match is close enough. Otherwise labels the face as `Unknown`.

**What makes it cautious.** The system is deliberately tuned to prefer
`Unknown` over a guess. If the closest match isn't *clearly* better than
the runner-up, or if the distance is too large, it doesn't commit to a
name. That's why it never mislabels anyone in the evaluation —
**precision is 100% on every named character** (see the [Results](#results)
table below).

**Headline numbers** (on a 76-face evaluation sample, hand-checked):
- **Detection:** every face found — 100%.
- **Named correctly when it committed:** every time (zero false alarms).
- **Characters recognised most reliably:** Harry (~78% caught), Snape (~67%).
- **Weakest:** McGonagall — only one frame of her in the sample, and the
  system identified her correctly but just barely missed its own
  confidence cutoff. We deliberately *didn't* nudge the threshold to fix
  it, because that would be "gaming the exam" — see
  [NOTES.md §8](NOTES.md#8-the-mcgonagall-near-miss).

**Why this matters for a real deployment.** Face-recognition systems that
confidently mislabel people are dangerous. A system that honestly reports
"I don't know" when it's unsure is much safer to put in front of real
decisions — in sports broadcast tagging, betting markets, or anywhere
else where a wrong name causes downstream harm.

If terminology like "precision", "F1", "embedding", "k-NN" is new,
there's a short [Glossary](#glossary) at the bottom of this file.

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

> **Reading the table in plain English.**
> - **Precision** = "when the system said it was Harry, how often was it right?"
>   1.000 means 100% — zero false alarms on any named character.
> - **Recall** = "of all the actual Harry faces in the sample, how many did
>   we catch?" 0.783 means we caught about 78%.
> - **F1** = a single score balancing precision and recall, 0 to 1, higher
>   is better.
> - **95% CI** = the uncertainty range on that F1 score. Our sample is
>   small (76 faces), so the intervals are wide — a deliberately honest
>   picture of what we know vs. guess.
> - The **Unknown** row captures over- and under-confidence: 0.617 precision
>   means ~38% of our `Unknown` labels were actually named characters we
>   were too cautious to name. Recall of 1.000 means every genuine
>   `Unknown` was labelled as such.

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
- `ffmpeg` on PATH (used for h264 video re-encoding; near-universal install)

**Platform support:**
- **Developed and tested on Linux x86_64** (Ubuntu-family, Python 3.11).
- **macOS arm64 (Apple Silicon)** should also work via the
  platform-conditional `tensorflow-macos` pin in `pyproject.toml` — but
  not verified on that platform. If you hit issues, please flag them.

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

> **In plain English.** Frame by frame: find the faces (detector), turn
> each one into a fingerprint of numbers (embedder), compare each
> fingerprint against our reference photos to pick the closest match and
> decide if we're confident enough to name it (recogniser). A tracker
> watches the same face across several frames and only changes its mind
> when the evidence genuinely shifts — that's what stops the output video
> flickering between "Harry" and "Unknown" on the same face. A
> scene-change detector resets the tracker when the shot cuts, so labels
> don't bleed across unrelated shots.

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

---

## Glossary

A short guide to the terminology used here. No prior ML background needed.

- **Bounding box** — the rectangle drawn around a face in the video.
- **Face detection** — the job of *finding* faces in an image (and drawing
  bounding boxes around them).
- **Face recognition** — the next step: deciding *whose* face it is.
- **Embedding** — a "fingerprint" of a face, represented as a list of 512
  numbers. Faces of the same person produce similar fingerprints; different
  people produce different ones. The magic is that the fingerprint
  captures the person's identity while ignoring pose, lighting, expression.
- **Cosine distance** — a way to measure how different two fingerprints
  are. Ranges from 0 (identical) to 2 (opposite). Lower = more similar.
- **Reference set** — a small collection of known photos per character
  (3–8 stills each in our system) that the recogniser compares new faces
  against.
- **Precision** — of the faces we *said* were Harry, what fraction *actually*
  were? 1.000 = 100%, i.e. no false alarms.
- **Recall** — of all the *actual* Harry faces, how many did we *catch*?
  0.78 = we caught 78%.
- **F1 score** — a single number combining precision and recall. Ranges
  0–1, higher is better. Used when both kinds of error matter.
- **Macro-F1** — the average F1 across all classes, weighted equally. A
  system that's great at Harry but awful at McGonagall gets pulled down.
- **95% Confidence Interval (CI)** — the statistical uncertainty around a
  score. "F1 = 0.878 CI [0.743, 0.974]" means: if we re-sampled the
  evaluation set many times, we'd expect 95% of the F1 scores to land in
  that range. Wide intervals = small sample = more uncertainty.
- **Bootstrap resampling** — a statistical technique for computing those
  confidence intervals from a single evaluation set by repeatedly
  re-sampling with replacement (1000 times in our case).
- **k-NN (k-Nearest Neighbours)** — a classic classification approach. To
  name a new face, look at the `k` closest reference fingerprints; if
  they agree, assign that name. We use k=3.
- **Threshold** — the cutoff distance beyond which the recogniser decides
  "too different from any reference, I don't know who this is".
- **Margin gate** — a second safety check: the gap between the best and
  second-best match must be big enough. If top-1 is only barely closer
  than top-2, the system is uncertain and falls back to `Unknown`.
- **Calibration** — the process of setting thresholds based on measured
  data rather than guesswork. We use *leave-one-out*: hold out one
  reference at a time and see how far it is from its own class centroid;
  use the worst case (plus buffer) as the threshold.
- **Tracker** — software that follows the same face across consecutive
  frames so we can smooth out single-frame errors.
- **IoU (Intersection over Union)** — how much two bounding boxes overlap,
  0 to 1. Used by the tracker to say "this box in this frame is probably
  the same face as that box in the previous frame".
- **Hysteresis** — fancy word for "don't flip your mind at the first sign
  of disagreement". Why the label on a face doesn't strobe — it takes
  several frames of consistent contradictory evidence to change.
- **Scene cut** — a hard change of camera angle. The tracker flushes its
  state when one happens, so the label for a new character doesn't get
  polluted by the previous character's history.
- **Ground truth** — the human-verified correct answer. Ours: 76 faces
  hand-labelled (and double-checked) across 40 sample frames.
- **Stratified sampling** — taking samples evenly across the clip rather
  than clumping them at the start or end. Prevents systematic bias.
- **`Unknown` as a first-class class** — we treat "I don't know" as a
  full-fledged label, not a silent fallback. It appears in the precision/
  recall table because *over-using* `Unknown` (being too cautious) is
  also wrong, just in a different direction.
- **DeepFace / RetinaFace / Facenet512** — the pre-built open-source AI
  models we use. **DeepFace** is the Python library; **RetinaFace** is
  the specific face-detection model; **Facenet512** is the specific
  face-embedding model. All three are recommended in the assessment brief.

---

## Citations

- [DeepFace](https://github.com/serengil/deepface) — Serengil & Ozpinar (2020).
- [RetinaFace](https://arxiv.org/abs/1905.00641) — Deng et al. (2019).
- [Facenet](https://arxiv.org/abs/1503.03832) — Schroff, Kalenichenko & Philbin (2015).

## License

MIT — see [`LICENSE`](LICENSE).
