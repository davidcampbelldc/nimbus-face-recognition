# Nimbus — Face Detection + Character Recognition

Identifies named Harry Potter characters (Harry, Ron, Hermione, McGonagall, Snape) in a 101-second clip from *Philosopher's Stone*, using [DeepFace](https://github.com/serengil/deepface) with RetinaFace detection and Facenet512 embeddings.

Built as a take-home assessment for White Swan Data.

> **Status:** in development. Results table, screenshot, and output-video link will land here before submission.

## Results

_(placeholder — populated in Phase 5)_

| Character    | Precision | Recall | F1 (95% CI) | Detections (N) |
|--------------|-----------|--------|-------------|----------------|
| Harry        | —         | —      | —           | —              |
| Ron          | —         | —      | —           | —              |
| Hermione     | —         | —      | —           | —              |
| McGonagall   | —         | —      | —           | —              |
| Snape        | —         | —      | —           | —              |
| **Macro avg**| —         | —      | —           | —              |

**Detection recall:** — &nbsp;&nbsp;•&nbsp;&nbsp; **Runtime:** — fps CPU &nbsp;&nbsp;•&nbsp;&nbsp; **Peak memory:** — MB

Full evaluation methodology in [§ Evaluation](#evaluation) below. Raw numbers in [`eval/metrics.json`](eval/metrics.json).

## Output video

_(placeholder — Drive link goes here before submission)_

## Quickstart

```bash
git clone <repo-url>
cd whiteswan-deepface
bash scripts/fetch_weights.sh   # downloads + verifies Facenet512 + RetinaFace weights
pip install -e .
python run.py data/input/nimbus.mp4 data/output/nimbus_annotated.mp4
```

**Prerequisites:** Python 3.11. Tested on Linux (x86_64) and macOS (arm64). See [Install](#install) for platform notes.

## Smoke mode

Quick 1-second sanity check before a full 45-minute run:

```bash
python run.py --frames 30 data/input/nimbus.mp4 data/output/nimbus_smoke.mp4
```

## What's in the repo

```
src/nimbus/        # pipeline: detector, embedder, recogniser, tracker, renderer
scripts/           # ref-building, baseline probe, evaluation, plots
references/        # curated reference stills, 10-15 per character
refs/              # generated embeddings, centroids, calibration (committed)
weights/           # vendored model weights (Git LFS, sha256-verified)
tests/             # unit + integration tests
eval/              # ground truth + computed metrics
docker/            # optional install-fallback image
```

## Install

### Native (primary path)

```bash
pip install -e .
```

**Platform notes:**
- Linux + Windows + Intel Mac: `tensorflow-cpu==2.16.1`
- Apple Silicon: `tensorflow-macos==2.16.1` (auto-selected via platform marker)
- `tf-keras==2.16.0` is pinned explicitly — DeepFace 0.0.93 requires the legacy Keras API.

### Docker (fallback if native install fails)

```bash
docker build -f docker/Dockerfile.optional -t nimbus .
docker run --rm -v "$PWD/data:/app/data" nimbus
```

## How it works

_(pipeline diagram + 3-paragraph design explanation — populated Phase 5)_

## Evaluation

_(eval methodology + leakage discipline — populated Phase 5)_

## Known limitations

_(epistemic boundaries — populated Phase 5)_

## Design notes

See [`NOTES.md`](NOTES.md) for the full design rationale — decisions made, tradeoffs, and what I'd do with another week.

## Citations

- [DeepFace](https://github.com/serengil/deepface) — Serengil, S. I., & Ozpinar, A. (2020).
- [RetinaFace](https://arxiv.org/abs/1905.00641) — Deng, J., Guo, J., et al. (2019).
- [Facenet](https://arxiv.org/abs/1503.03832) — Schroff, F., Kalenichenko, D., & Philbin, J. (2015).

## License

MIT — see [`LICENSE`](LICENSE).
