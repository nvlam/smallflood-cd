#!/usr/bin/env bash
# Run manually from the server project root, inside tmux.
set -euo pipefail
test -f pyproject.toml
test -x .venv/bin/python
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN=$(date -u +%Y%m%dT%H%M%SZ)
LOG_ROOT="artifacts/pilot15/$RUN"
OUTPUT_ROOT="runs/pilot15/$RUN"
mkdir -p "$LOG_ROOT"
echo "Logs: $LOG_ROOT"
echo "Results: $OUTPUT_ROOT"
.venv/bin/python -m pytest -q tests/unit/test_pilot15_preflight.py tests/unit/test_validation_protocol_v2.py 2>&1 | tee "$LOG_ROOT/preflight_tests.log"
.venv/bin/python -c "import sys; sys.path.insert(0, 'scripts'); from next_steps import band_provenance; from smoke_train import CONFIGS; from smallflood_cd.utils.config import load_experiment_config; from smallflood_cd.engine.validator import SELECTION_PROTOCOL; assert SELECTION_PROTOCOL['version'] == 'event_pixel_v2'; [print(name, band_provenance(load_experiment_config(path))) for name, path in CONFIGS.items()]" 2>&1 | tee "$LOG_ROOT/data_metadata_check.log"
.venv/bin/python -m pip freeze > "$LOG_ROOT/pip_freeze.txt"
nvidia-smi > "$LOG_ROOT/nvidia_smi.txt"
tar -czf "$LOG_ROOT/source_snapshot.tar.gz" src scripts configs pyproject.toml
for MODEL in proposed fc_siam_diff bit
do
  if .venv/bin/python -u scripts/next_steps.py pilot \
      --model "$MODEL" --epochs 15 --batch-size 8 --seed 42 \
      --engineering-only --output-root "$OUTPUT_ROOT" \
      2>&1 | tee "$LOG_ROOT/${MODEL}.log"
  then
    echo "COMPLETED: $MODEL"
  else
    echo "STOP: $MODEL failed. No retry and no next model."
    exit 1
  fi
done
echo "ALL THREE PILOTS COMPLETE: $OUTPUT_ROOT"
echo "No test evaluation performed."
