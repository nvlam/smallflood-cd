# Factorial runner handoff

## Material Passport

User approved runner implementation and packaging, NOT training launch.
Protocol: proposed_factorial_v1. No production model/loss/trainer code modified.
Local synthetic CPU integration + config tests: 8 passed. CUDA/data preflight pending.

## Install and test

Extract proposed_factorial_runner_20260927.tar.gz from project root; package adds
scripts, four configs, tests and docs only. Does not replace existing training modules.
Activate existing .venv, then:

```bash
python scripts/proposed_factorial.py test
```

Expected 8 passed. These tests do synthetic CPU optimizer steps; no research data.

## Separately invoked CUDA preflight

Inside tmux from project root:

```bash
bash scripts/preflight_proposed_factorial.sh
```

This command performs training on a bounded balanced subset (16 train, 8 validation,
batch 8, 1 epoch = 2 optimizer steps per cell, all four cells). It is not a full pilot.
It may create/reuse train/validation component caches for weighted cells, unlike the
earlier read-only mask audit. It never constructs test datasets or reads test arrays.
Output runs/proposed_factorial_preflight/<UTC>, logs artifacts/proposed_factorial_preflight/<UTC>.
Gate requires common initial tensors and all batch-order hashes identical across A–D;
A/B and C/D full initial state hashes identical. Checkpoints reload with exact model
state; optimizer-state resume and prediction-level replay are not claimed by this gate.
Fixed data metadata hashes, band provenance, patch counts and event separation enforced.
Source imports pinned to src to avoid stale root package. Stop on failure, no retry.
Uses a single GPU sequentially. Model RNG reset after initialization; data order uses
an independent seed42 generator. Full-run order hashes are checked separately.

## Full-run capability — do NOT execute until separate user approval

```bash
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u scripts/proposed_factorial.py run \
  --preflight runs/proposed_factorial_preflight/REPLACE_WITH_COMPLETED_UTC \
  --output runs/proposed_factorial_v1/REPLACE_WITH_NEW_UTC \
  --approve-training
```

Gate fingerprint must match current source/config/data metadata and environment.
Changing any Python source/config requires a new preflight. This is intentional.
Existing output roots are refused. No resume, no automatic retry; no reused preflight
weights. Four fresh cells A–D, each 15 epochs, batch8, no early stopping or scheduler.
Preflight success does not automatically launch this command.

Each cell writes config, environment, epoch metrics (component loss terms, scaled
terms, clipping fraction, preclip norm, weights, residual scale), model initialization
hashes, epoch batch hashes, best/last checkpoints, timing/memory and summary. Root
provenance and COMPLETE.json cover all cells. Only whole-group COMPLETE certifies
pairing; partial cell output after failure is not a completed factorial experiment.
Per-epoch component term logging is batch-mean, not per-object gradient attribution.
Raw unweighted counterparts of weighted losses are not computed in this version;
logged weighted_bce/weighted_tversky are before the .4 coefficients.

Component/boundary review after training still needs a factorial-aware collection
wrapper: existing reviewer searches one *_proposed directory, whereas group output
contains four. Do not feed the four-cell root to the old default reviewer. No test
evaluation or 135-matrix integration is included. No Orin deployment benchmark claim.
