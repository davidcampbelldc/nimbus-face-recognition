.PHONY: install install-dev test lint typecheck run eval fetch-weights clean submission-check

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

fetch-weights:
	bash scripts/fetch_weights.sh

test:
	pytest -v

lint:
	ruff check src/ scripts/ tests/

lint-fix:
	ruff check --fix src/ scripts/ tests/

typecheck:
	mypy src/nimbus

run:
	python run.py data/input/nimbus.mp4 data/output/nimbus_annotated.mp4

smoke:
	python run.py --frames 30 data/input/nimbus.mp4 data/output/nimbus_smoke.mp4

eval:
	python scripts/evaluate.py

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

submission-check: lint typecheck test smoke
	@echo ""
	@echo "Submission checklist (manual verification):"
	@echo "  [ ] eval/metrics.json matches README numbers"
	@echo "  [ ] refs/calibration.json committed"
	@echo "  [ ] references/_clip_frames_used.json committed"
	@echo "  [ ] output video plays in QuickTime + VLC + Chrome"
	@echo "  [ ] docker/Dockerfile.optional builds + runs"
	@echo "  [ ] no secrets in repo (security-review skill)"
	@echo "  [ ] Alex Holmes invited as collaborator"
	@echo "  [ ] output video uploaded to Drive + link tested"
