#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN=$(date -u +%Y%m%dT%H%M%SZ)
LOG="artifacts/candidate_r_validation_review/$RUN"
mkdir -p artifacts/candidate_r_validation_review
mkdir "$LOG"
.venv/bin/python -u scripts/review_candidate_r.py \
  --pilot-root runs/candidate_r_v1/20261002T150633Z \
  --output "$LOG/results" 2>&1 | tee "$LOG/review.log"
echo "ALL FOUR CANDIDATE R CHECKPOINTS REVIEWED: $LOG"
