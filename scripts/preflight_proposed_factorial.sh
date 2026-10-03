#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p artifacts/proposed_factorial_preflight
LOG="artifacts/proposed_factorial_preflight/$RUN"
mkdir "$LOG"
.venv/bin/python scripts/proposed_factorial.py test 2>&1 | tee "$LOG/tests.log"
.venv/bin/python -u scripts/proposed_factorial.py preflight \
  --output "runs/proposed_factorial_preflight/$RUN" 2>&1 | tee "$LOG/preflight.log"
echo "PREFLIGHT ONLY COMPLETE. No 15-epoch run launched."
