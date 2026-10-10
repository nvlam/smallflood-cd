#!/usr/bin/env bash
# efficiency_v1 (docs/efficiency_protocol_v1_20261010.md). Run from the server project root,
# inside tmux. Validation split only for fidelity; never trains; never reads test.
# Any failure stops: no automatic retry.
set -euo pipefail
cd "$(dirname "$0")/.."
test -f pyproject.toml
test -x .venv/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venv/bin/python
RUN=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="artifacts/efficiency_v1/$RUN"
LOG="artifacts/efficiency_v1_logs/$RUN"
mkdir -p artifacts/efficiency_v1 "$LOG"
echo "Outputs: $ROOT"
echo "Logs: $LOG"
sha256sum scripts/efficiency.py scripts/review_pilot_validation.py tests/unit/test_efficiency.py \
  src/smallflood_cd/deployment/onnx_export.py > "$LOG/script_sha256.txt"
step() { echo "=== $(date -u +%FT%TZ) $1" | tee -a "$LOG/steps.log"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> "$LOG/steps.log"; uptime >> "$LOG/steps.log"; }
step "synthetic tests"
PYTHONHASHSEED=42 $PY -u scripts/efficiency.py test 2>&1 | tee "$LOG/tests.log"
step "smoke (no timing kept)"
PYTHONHASHSEED=42 $PY -u scripts/efficiency.py smoke --output "$LOG/smoke" 2>&1 | tee "$LOG/smoke.log"
step "measure"
PYTHONHASHSEED=42 $PY -u scripts/efficiency.py measure --output "$ROOT" 2>&1 | tee "$LOG/measure.log"
step "fidelity (validation only)"
PYTHONHASHSEED=201 $PY -u scripts/efficiency.py fidelity --output "$ROOT" 2>&1 | tee "$LOG/fidelity.log"
step "complete"
PYTHONHASHSEED=42 $PY -u scripts/efficiency.py complete --output "$ROOT" 2>&1 | tee "$LOG/complete.log"
step "done"
echo "EFFICIENCY COMPLETE: $ROOT (test_used=false)"
