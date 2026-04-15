#!/usr/bin/env bash
# Fetch + verify model weights for Facenet512 and RetinaFace.
# DeepFace normally downloads these on first use from a GitHub URL that has
# 404'd during incidents. Vendoring + sha256 verification gives the reviewer
# a deterministic first-run experience.
set -euo pipefail

cd "$(dirname "$0")/.."
WEIGHTS_DIR="weights"
mkdir -p "$WEIGHTS_DIR"

# URLs and checksums populated during Phase 0 — placeholder for now.
# TODO(Phase 0): fill in URLs from DeepFace's weight_utils + verify sha256.

if [ -f "$WEIGHTS_DIR/CHECKSUMS.sha256" ]; then
  echo "Verifying existing weights against CHECKSUMS.sha256..."
  cd "$WEIGHTS_DIR" && sha256sum -c CHECKSUMS.sha256 && cd ..
  echo "Weights OK."
else
  echo "No CHECKSUMS.sha256 yet — weights will be populated in Phase 0 completion."
  echo "DeepFace will auto-download on first use in the meantime."
fi
