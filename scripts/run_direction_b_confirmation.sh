#!/usr/bin/env bash
# direction_b_confirmation_v1 (docs/direction_b_confirmation_protocol_v1_20261003.md).
# Run manually from the server project root, inside tmux. Validation only; test sealed.
# Any failure stops everything: no retry, no extra seeds.
set -euo pipefail
cd "$(dirname "$0")/.."
test -f pyproject.toml
test -x .venv/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venv/bin/python
RUN=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="runs/direction_b_confirmation_v1/$RUN"
LOG="artifacts/direction_b_confirmation/$RUN"
mkdir -p artifacts/direction_b_confirmation runs/direction_b_confirmation_v1
mkdir "$LOG"
echo "Outputs: $ROOT"
echo "Logs: $LOG"
nvidia-smi > "$LOG/nvidia_smi_start.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >> "$LOG/nvidia_smi_start.txt"
$PY -m pip freeze > "$LOG/pip_freeze.txt"
tar --exclude='__pycache__' -czf "$LOG/source_snapshot.tar.gz" src scripts configs tests pyproject.toml
sha256sum scripts/direction_b_factorial.py scripts/direction_b_pilot.py scripts/review_direction_b.py \
  scripts/proposed_factorial.py scripts/pilot_bit_sar_v2.py scripts/next_steps.py > "$LOG/script_sha256.txt"

step() { echo "=== $(date -u +%FT%TZ) $1" | tee -a "$LOG/steps.log"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> "$LOG/steps.log"; }

step "synthetic tests"
PYTHONHASHSEED=42 $PY -u scripts/direction_b_factorial.py test 2>&1 | tee "$LOG/tests_factorial.log"
PYTHONHASHSEED=42 $PY -u scripts/direction_b_pilot.py test 2>&1 | tee "$LOG/tests_pilot.log"
PYTHONHASHSEED=42 $PY -m pytest -q tests/unit/test_pilot15_preflight.py tests/unit/test_validation_protocol_v2.py 2>&1 | tee "$LOG/tests_fc_preflight.log"

step "seed-42 regression (no training)"
PYTHONHASHSEED=42 $PY -u scripts/direction_b_factorial.py regress --seed 42 \
  --output "$ROOT/regress_s42" 2>&1 | tee "$LOG/regress_s42.log"

for SEED in 1337 2026
do
  step "factorial preflight s$SEED"
  PYTHONHASHSEED=$SEED $PY -u scripts/direction_b_factorial.py preflight --seed $SEED \
    --output "$ROOT/factorial_s${SEED}_preflight" 2>&1 | tee "$LOG/factorial_s${SEED}_preflight.log"
  step "factorial run s$SEED"
  PYTHONHASHSEED=$SEED $PY -u scripts/direction_b_factorial.py run --seed $SEED --approve-training \
    --preflight "$ROOT/factorial_s${SEED}_preflight" \
    --output "$ROOT/factorial_s${SEED}" 2>&1 | tee "$LOG/factorial_s${SEED}.log"
  step "BIT-SAR v2 batch-8 gate s$SEED"
  PYTHONHASHSEED=42 $PY -u scripts/bit_sar_v2_entry.py pilot batch8 \
    --output-dir "$ROOT/bit_s${SEED}_batch8" 2>&1 | tee "$LOG/bit_s${SEED}_batch8.log"
  step "BIT-SAR v2 run s$SEED"
  PYTHONHASHSEED=$SEED $PY -u scripts/direction_b_pilot.py bit_sar_v2 --seed $SEED \
    --preflight "$ROOT/bit_s${SEED}_batch8" \
    --output-root "$ROOT/pilots_s${SEED}" 2>&1 | tee "$LOG/bit_s${SEED}.log"
  step "FC-Siam-Diff run s$SEED"
  PYTHONHASHSEED=$SEED $PY -u scripts/direction_b_pilot.py fc_siam_diff --seed $SEED \
    --output-root "$ROOT/pilots_s${SEED}" 2>&1 | tee "$LOG/fc_s${SEED}.log"
done

REVIEW="artifacts/direction_b_confirmation_review/$RUN"
mkdir -p "$REVIEW"
for SEED in 1337 2026
do
  step "validation review s$SEED"
  PYTHONHASHSEED=$SEED $PY -u scripts/review_direction_b.py --seed $SEED \
    --factorial-root "$ROOT/factorial_s${SEED}" --pilot-root "$ROOT/pilots_s${SEED}" \
    --output "$REVIEW/s${SEED}" 2>&1 | tee "$LOG/review_s${SEED}.log"
done
step "done"
nvidia-smi > "$LOG/nvidia_smi_end.txt"
echo "DIRECTION B CONFIRMATION COMPLETE: runs=$ROOT reviews=$REVIEW (test_used=false)"
