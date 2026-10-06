#!/usr/bin/env bash
# recipe_stability_v1 (docs/recipe_stability_protocol_v1_20261006.md).
# Run from the server project root inside tmux. Training events only: the current
# validation events and the test set are never loaded. Any failure stops: no retry.
set -euo pipefail
cd "$(dirname "$0")/.."
test -f pyproject.toml
test -x .venv/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venv/bin/python
RUN=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="runs/recipe_stability_v1/$RUN"
LOG="artifacts/recipe_stability_v1/$RUN"
mkdir -p artifacts/recipe_stability_v1 runs/recipe_stability_v1
mkdir "$LOG"
echo "Outputs: $ROOT"
echo "Logs: $LOG"
nvidia-smi > "$LOG/nvidia_smi_start.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >> "$LOG/nvidia_smi_start.txt"
uptime >> "$LOG/nvidia_smi_start.txt"
$PY -m pip freeze > "$LOG/pip_freeze.txt"
tar --exclude='__pycache__' -czf "$LOG/source_snapshot.tar.gz" src scripts configs tests pyproject.toml
sha256sum scripts/recipe_stability.py scripts/direction_b_factorial.py scripts/proposed_factorial.py \
  tests/unit/test_recipe_stability.py > "$LOG/script_sha256.txt"

step() { echo "=== $(date -u +%FT%TZ) $1" | tee -a "$LOG/steps.log"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> "$LOG/steps.log"; uptime >> "$LOG/steps.log"; }

step "synthetic tests"
PYTHONHASHSEED=42 $PY -u scripts/recipe_stability.py test 2>&1 | tee "$LOG/tests.log"
step "seed-42 regression and workers 0 vs 4"
PYTHONHASHSEED=42 $PY -u scripts/recipe_stability.py regress --output "$ROOT/regress_s42" 2>&1 | tee "$LOG/regress.log"
step "preflight R0-R3"
PYTHONHASHSEED=101 $PY -u scripts/recipe_stability.py preflight --output "$ROOT/preflight" 2>&1 | tee "$LOG/preflight.log"
for SEED in 101 102 103
do
  for RECIPE in R0 R1 R2 R3
  do
    step "run $RECIPE s$SEED"
    PYTHONHASHSEED=$SEED $PY -u scripts/recipe_stability.py run --recipe $RECIPE --seed $SEED \
      --approve-training --preflight "$ROOT/preflight" --output "$ROOT/${RECIPE}_s${SEED}" \
      2>&1 | tee "$LOG/${RECIPE}_s${SEED}.log"
  done
done
step "decision"
PYTHONHASHSEED=42 $PY -u scripts/recipe_stability.py decide --output "$ROOT" 2>&1 | tee "$LOG/decision.log"
step "done"
echo "RECIPE STABILITY COMPLETE: $ROOT (validation_events_loaded=false, test_used=false)"
