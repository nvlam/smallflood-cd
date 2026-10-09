#!/usr/bin/env bash
# heldout_test_v1 (docs/heldout_test_protocol_v1_20261009.md): THE single test evaluation.
# Run only after the user approves test access (section 9), from the server project root,
# inside tmux. Never trains. On interruption, rerun the remaining seeds with --resume only
# (section 6.3); completed evaluations are never re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
test -f pyproject.toml
test -x .venv/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venv/bin/python
if [ -d artifacts/heldout_test_v1 ] && [ -n "$(ls -A artifacts/heldout_test_v1)" ]; then
  echo "heldout_test_v1 outputs already exist; single use only. Use --resume manually." >&2
  exit 1
fi
RUN=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="artifacts/heldout_test_v1/$RUN"
LOG="artifacts/heldout_test_v1_logs/$RUN"
mkdir -p "$LOG"
echo "Outputs: $ROOT"
echo "Logs: $LOG"
nvidia-smi > "$LOG/nvidia_smi_start.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >> "$LOG/nvidia_smi_start.txt"
uptime >> "$LOG/nvidia_smi_start.txt"
tar --exclude='__pycache__' -czf "$LOG/source_snapshot.tar.gz" src scripts configs tests pyproject.toml
sha256sum scripts/heldout_test.py scripts/review_pilot_validation.py tests/unit/test_heldout_test.py \
  data/splits/split_v1.yaml > "$LOG/script_sha256.txt"
step() { echo "=== $(date -u +%FT%TZ) $1" | tee -a "$LOG/steps.log"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> "$LOG/steps.log"; }
step "synthetic tests"
PYTHONHASHSEED=42 $PY -u scripts/heldout_test.py test 2>&1 | tee "$LOG/tests.log"
for SEED in 42 1337 2026 201 202 203
do
  step "test evaluation s$SEED"
  PYTHONHASHSEED=$SEED $PY -u scripts/heldout_test.py evaluate --seed $SEED --output "$ROOT" --approve-test \
    2>&1 | tee "$LOG/evaluate_s$SEED.log"
done
step "analysis"
PYTHONHASHSEED=42 $PY -u scripts/heldout_test.py analyze --output "$ROOT" 2>&1 | tee "$LOG/analyze.log"
step "done"
echo "HELDOUT TEST COMPLETE: $ROOT (evaluations=72, training_performed=false)"
