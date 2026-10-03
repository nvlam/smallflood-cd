#!/usr/bin/env bash
# Explicit bounded GPU check only. This script never launches the full pilot.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
AUDIT=${1:-artifacts/candidate_r_eligibility/20260928T134626505807Z}
RUN=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT="runs/candidate_r_preflight/$RUN"
LOGS="artifacts/candidate_r_preflight/$RUN"
mkdir -p "$LOGS"
echo "Outputs: $OUTPUT"
echo "Logs: $LOGS"
.venv/bin/python -u scripts/candidate_r_runner.py preflight \
  --audit "$AUDIT" --output "$OUTPUT" 2>&1 | tee "$LOGS/preflight.log"
echo "PREFLIGHT COMPLETE. Stop here; no full training was launched."
