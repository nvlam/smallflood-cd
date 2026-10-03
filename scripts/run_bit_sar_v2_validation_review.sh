#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN=$(date -u +%Y%m%dT%H%M%SZ)
LOG_ROOT="artifacts/bit_sar_v2_validation_review/$RUN"
mkdir -p artifacts/bit_sar_v2_validation_review
mkdir "$LOG_ROOT"
.venv/bin/python -u scripts/bit_sar_v2_entry.py review \
  --pilot-root runs/bit_sar_v2_pilot15/20260925T015023Z/pilot \
  --models bit_sar_v2 \
  --device cuda --batch-size 8 --object-boundary \
  --output-root "$LOG_ROOT/results" \
  2>&1 | tee "$LOG_ROOT/review.log"
echo "BIT-SAR v2 VALIDATION REVIEW COMPLETE: $LOG_ROOT"
