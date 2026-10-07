#!/usr/bin/env bash
# recipe_confirmation_v1 (docs/recipe_confirmation_protocol_v1_20261008.md).
# Run manually from the server project root, inside tmux. Validation only; test sealed.
# Any failure stops everything: no retry, no extra seeds.
set -euo pipefail
cd "$(dirname "$0")/.."
test -f pyproject.toml
test -x .venv/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venv/bin/python
RUN=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="runs/recipe_confirmation_v1/$RUN"
LOG="artifacts/recipe_confirmation_v1/$RUN"
REVIEW="artifacts/recipe_confirmation_review/$RUN"
mkdir -p artifacts/recipe_confirmation_v1 artifacts/recipe_confirmation_review runs/recipe_confirmation_v1
mkdir "$LOG"
echo "Outputs: $ROOT"
echo "Logs: $LOG"
nvidia-smi > "$LOG/nvidia_smi_start.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >> "$LOG/nvidia_smi_start.txt"
uptime >> "$LOG/nvidia_smi_start.txt"
$PY -m pip freeze > "$LOG/pip_freeze.txt"
tar --exclude='__pycache__' -czf "$LOG/source_snapshot.tar.gz" src scripts configs tests pyproject.toml
sha256sum scripts/recipe_confirmation_factorial.py scripts/recipe_confirmation_pilot.py \
  scripts/review_recipe_confirmation.py scripts/direction_b_factorial.py scripts/direction_b_pilot.py \
  scripts/review_direction_b.py scripts/recipe_stability.py scripts/proposed_factorial.py \
  scripts/pilot_bit_sar_v2.py scripts/next_steps.py tests/unit/test_recipe_confirmation.py \
  configs/experiment/recipe_confirmation_v1/fc_siam_diff.yaml configs/experiment/fc_siam_diff.yaml \
  configs/experiment/bit_sar_v2.yaml > "$LOG/script_sha256.txt"

step() { echo "=== $(date -u +%FT%TZ) $1" | tee -a "$LOG/steps.log"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> "$LOG/steps.log"; uptime >> "$LOG/steps.log"; }

step "synthetic tests"
PYTHONHASHSEED=42 $PY -u scripts/recipe_confirmation_factorial.py test 2>&1 | tee "$LOG/tests_factorial.log"
PYTHONHASHSEED=42 $PY -u scripts/recipe_confirmation_pilot.py test 2>&1 | tee "$LOG/tests_pilot.log"
PYTHONHASHSEED=42 $PY -u scripts/review_recipe_confirmation.py test 2>&1 | tee "$LOG/tests_review.log"

step "seed-42 regression: anchors, R1 tie to phase 1, workers 0 vs 4 for A and C"
PYTHONHASHSEED=42 $PY -u scripts/recipe_confirmation_factorial.py regress --seed 42 \
  --output "$ROOT/regress_s42" 2>&1 | tee "$LOG/regress_s42.log"

for SEED in 201 202 203
do
  step "factorial preflight s$SEED"
  PYTHONHASHSEED=$SEED $PY -u scripts/recipe_confirmation_factorial.py preflight --seed $SEED \
    --regression "$ROOT/regress_s42" --output "$ROOT/factorial_s${SEED}_preflight" \
    2>&1 | tee "$LOG/factorial_s${SEED}_preflight.log"
  step "factorial run s$SEED"
  PYTHONHASHSEED=$SEED $PY -u scripts/recipe_confirmation_factorial.py run --seed $SEED --approve-training \
    --regression "$ROOT/regress_s42" --preflight "$ROOT/factorial_s${SEED}_preflight" \
    --output "$ROOT/factorial_s${SEED}" 2>&1 | tee "$LOG/factorial_s${SEED}.log"
  step "BIT-SAR v2 batch-8 gate s$SEED"
  PYTHONHASHSEED=42 $PY -u scripts/bit_sar_v2_entry.py pilot batch8 \
    --output-dir "$ROOT/bit_s${SEED}_batch8" 2>&1 | tee "$LOG/bit_s${SEED}_batch8.log"
  step "BIT-SAR v2 run s$SEED"
  PYTHONHASHSEED=$SEED $PY -u scripts/recipe_confirmation_pilot.py bit_sar_v2 --seed $SEED \
    --preflight "$ROOT/bit_s${SEED}_batch8" \
    --output-root "$ROOT/pilots_s${SEED}" 2>&1 | tee "$LOG/bit_s${SEED}.log"
  step "FC-Siam-Diff preflight s$SEED"
  PYTHONHASHSEED=$SEED $PY -u scripts/recipe_confirmation_pilot.py fc_preflight --seed $SEED \
    --output-root "$ROOT/fc_s${SEED}_preflight" 2>&1 | tee "$LOG/fc_s${SEED}_preflight.log"
  step "FC-Siam-Diff run s$SEED"
  PYTHONHASHSEED=$SEED $PY -u scripts/recipe_confirmation_pilot.py fc_siam_diff --seed $SEED \
    --preflight "$ROOT/fc_s${SEED}_preflight" \
    --output-root "$ROOT/pilots_s${SEED}" 2>&1 | tee "$LOG/fc_s${SEED}.log"
done

for SEED in 201 202 203
do
  step "validation review s$SEED"
  PYTHONHASHSEED=$SEED $PY -u scripts/review_recipe_confirmation.py review --seed $SEED \
    --factorial-root "$ROOT/factorial_s${SEED}" --pilot-root "$ROOT/pilots_s${SEED}" \
    --output "$REVIEW/s${SEED}" 2>&1 | tee "$LOG/review_s${SEED}.log"
done
step "decision"
PYTHONHASHSEED=42 $PY -u scripts/review_recipe_confirmation.py decide --output "$REVIEW" \
  2>&1 | tee "$LOG/decision.log"
step "done"
nvidia-smi > "$LOG/nvidia_smi_end.txt"
echo "RECIPE CONFIRMATION COMPLETE: runs=$ROOT reviews=$REVIEW (test_used=false)"
