# BIT-SAR v2 validation review

## Material Passport

Scope: user-approved review integration, validation only, no training or test access.
Input: completed pilot runs/bit_sar_v2_pilot15/20260925T015023Z/pilot.
Change: accept bit_sar_v2 in existing reviewer CLI and route through source-pinned entry.
Metric implementations, threshold 0.5, default legacy model list, and training code unchanged.
Local integration tests: 6 passed, including real BIT-SAR v2 CPU checkpoint loading,
both best/last reviews, object/boundary outputs, unchanged checkpoint hashes, and
inaccessible train/test files proving validation-only loading on synthetic fixtures.
This is not a real-data GPU evaluation or full training reproducibility claim.

## Server procedure

Back up scripts/review_pilot_validation.py, scripts/bit_sar_v2_entry.py and
tests/unit/test_review_pilot_validation.py before extracting the update archive.
From /data/fit.lamnv/smallflood_cd with .venv activated:

```bash
python scripts/bit_sar_v2_entry.py test-review
# Continue only after 6 passed, inside tmux:
bash scripts/run_bit_sar_v2_validation_review.sh
```

Launcher uses strict seed 42, FP32, batch 8, CUDA and source import guard.
Evaluates trusted local best_composite.ckpt and last.ckpt, not a new training run.
Do not pass untrusted checkpoints (torch.load uses weights_only=False).
Outputs are under artifacts/bit_sar_v2_validation_review/<UTC>/results/<UTC>/:
summary.csv, COMPLETE.json, and per-checkpoint report.json, events.csv, patches.csv,
object_boundary_report.json, object_boundary_events.csv, object_boundary_patch_counts.csv,
images and image index. review.log is one level above results.
COMPLETE.json must contain evaluations=2 and test_used=false.

## Interpretation boundaries

Pixel report and separate object/boundary report retain the existing protocols.
Components remain patch-scoped, not merged into full scenes. Small-region threshold
comes from train-fitted statistics (expected 35 pixels); interior and censored counts
must be reported. Preview selection is deterministic, not representative sampling.
No threshold tuning, test evaluation, deployment timing or automatic retraining.
Best epoch from training log is epoch 15; verify returned checkpoint epochs and compare
pixel counts with training before interpreting object/boundary results.
