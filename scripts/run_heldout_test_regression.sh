#!/usr/bin/env bash
# heldout_test_v1 engineering checks (docs/heldout_test_protocol_v1_20261009.md section 6).
# Validation split only: synthetic tests, then validation-mode regression against existing
# validation reports. Never reads the test split. Any failure stops; no retry.
set -euo pipefail
cd "$(dirname "$0")/.."
test -f pyproject.toml
test -x .venv/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venv/bin/python
RUN=$(date -u +%Y%m%dT%H%M%SZ)
OUT="artifacts/heldout_test_regression/$RUN"
mkdir -p "$OUT"
echo "Outputs: $OUT"
sha256sum scripts/heldout_test.py scripts/review_pilot_validation.py tests/unit/test_heldout_test.py > "$OUT/script_sha256.txt"
PYTHONHASHSEED=42 $PY -u scripts/heldout_test.py test 2>&1 | tee "$OUT/tests.log"
for SEED in 42 201
do
  PYTHONHASHSEED=$SEED $PY -u scripts/heldout_test.py regress --seed $SEED --output "$OUT/s$SEED" \
    2>&1 | tee "$OUT/regress_s$SEED.log"
done
echo "HELDOUT TEST REGRESSION COMPLETE: $OUT (test_used=false)"
