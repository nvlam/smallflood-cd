#!/usr/bin/env bash
set -euo pipefail

python -m smallflood_cd.cli.run_experiments \
  --matrix configs/experiment/experiment_matrix.yaml
