#!/usr/bin/env bash
set -euo pipefail
test -f configs/experiment/bit_sar_v2.yaml
test -x .venv/bin/python
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="runs/bit_sar_v2_pilot15/$RUN"
LOG="artifacts/bit_sar_v2_pilot15/$RUN"
mkdir -p artifacts/bit_sar_v2_pilot15
mkdir "$LOG"
echo "Outputs: $ROOT"
echo "Logs: $LOG"
nvidia-smi > "$LOG/nvidia_smi.txt"
.venv/bin/python -m pip freeze > "$LOG/pip_freeze.txt"
tar --exclude='__pycache__' -czf "$LOG/source_snapshot.tar.gz" src scripts configs pyproject.toml
.venv/bin/python -u scripts/bit_sar_v2_entry.py pilot batch8 \
  --output-dir "$ROOT/batch8_preflight" 2>&1 | tee "$LOG/batch8.log"
# New Python process: no preflight weights, optimizer or RNG are reused.
.venv/bin/python -u scripts/bit_sar_v2_entry.py pilot pilot \
  --preflight "$ROOT/batch8_preflight" --output-root "$ROOT/pilot" \
  2>&1 | tee "$LOG/pilot.log"
echo "ALL BIT-SAR v2 PILOT15 CHECKS COMPLETE: $ROOT"
