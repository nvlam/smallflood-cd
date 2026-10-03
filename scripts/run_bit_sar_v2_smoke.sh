#!/usr/bin/env bash
set -euo pipefail
test -f configs/experiment/bit_sar_v2.yaml
test -x .venv/bin/python
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="runs/bit_sar_v2_smoke/$RUN"
LOG="artifacts/bit_sar_v2_smoke/$RUN"
mkdir -p artifacts/bit_sar_v2_smoke
mkdir "$LOG"
echo "Outputs: $ROOT"
echo "Logs: $LOG"
nvidia-smi > "$LOG/nvidia_smi.txt"
.venv/bin/python -m pip freeze > "$LOG/pip_freeze.txt"
for PASS in a b
do
  .venv/bin/python -u scripts/bit_sar_v2_entry.py smoke run \
    --output-dir "$ROOT/$PASS" 2>&1 | tee "$LOG/$PASS.log"
done
.venv/bin/python -u scripts/bit_sar_v2_entry.py smoke compare \
  "$ROOT/a" "$ROOT/b" 2>&1 | tee "$LOG/compare.log"
echo "ALL BIT-SAR v2 GPU SMOKE CHECKS PASSED: $ROOT"
