#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p artifacts/proposed_factorial_review
LOG="artifacts/proposed_factorial_review/$RUN"
mkdir "$LOG"
.venv/bin/python -u scripts/review_proposed_factorial.py \
  --pilot-root runs/proposed_factorial_v1/20260927T075837Z \
  --output "$LOG/results" 2>&1 | tee "$LOG/review.log"
echo "ALL EIGHT CHECKPOINTS REVIEWED: $LOG"
