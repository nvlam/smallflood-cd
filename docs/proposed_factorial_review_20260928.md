# Factorial validation review: eight checkpoints

## Material Passport

Origin: academic-research-suite / experiment-agent, approved validation tooling.
Status: local tests 9 passed, no remote inference launched by assistant.
Inputs: runs/proposed_factorial_v1/20260927T075837Z, A/B/C/D best and last.
No training, test arrays, threshold tuning or model changes.

## Guards

Require complete 15-epoch factorial provenance and common-initial/batch-order pairing.
Check source files under src against training hashes and manifest/statistics/normalization
hashes. New review scripts do not invalidate historical source provenance.
Checkpoint config, epoch and global_step must match resolved config and expected row.
Best uses earliest maximum event F1: expected A12, B13, C12, D14; all last epoch15.
The wrapper derives these from logs, not hardcoded values. Trusted local checkpoint
loading uses weights_only=False; do not use arbitrary downloaded checkpoint files.
Reuses existing evaluate() unchanged: validation-only data, threshold .5, batch8,
FP32, strict deterministic seed42, no postprocessing, object_boundary_patch_v2.
Compare global TP/FP/FN/TN exactly with training log for every reviewed checkpoint;
check checkpoint file hashes before and after evaluation. Any discrepancy stops run.
Reports already written before a failure are partial, not a complete analysis.

## Installation and execution

Back up tests/unit/test_review_pilot_validation.py before archive extraction: that is
the only pre-existing file updated, adding Proposed boundary on/off integration tests.
No model/loss/training/metric modules are changed. The package reuses already-installed
scripts/review_pilot_validation.py and scripts/bit_sar_v2_entry.py.

```bash
python scripts/review_proposed_factorial.py --self-test
# Expected 9 passed; then inside tmux:
bash scripts/run_proposed_factorial_review.sh
```

Only CPU synthetic tests run in --self-test. Real review uses CUDA. Integration
fixtures make train/test array paths unreadable and verify checkpoints unchanged;
new wrapper helper tests cover tie policy, incomplete history, config/epoch and count
mismatches. Tests are not a real-data reproduction or deployment benchmark.

Output: artifacts/proposed_factorial_review/<UTC>/results/summary.csv and
<cell>_proposed/<best_composite|last>/{report.json,events.csv,patches.csv,
object_boundary_report.json,object_boundary_events.csv,object_boundary_patch_counts.csv,images/}.
Review log lives at artifacts/proposed_factorial_review/<UTC>/review.log.
Success: ALL EIGHT CHECKPOINTS REVIEWED, results/COMPLETE.json evaluations=8,
test_used=false, training_performed=false, pixel_counts_match_training=true.
Do not edit source/data/checkpoints while reviewing. No automatic retry.

Interpretation remains patch-scoped; small-component threshold=35 from train, interior
and censored denominators retained. Boundary F1 and inner-band IoU are distinct.
Review all cells/best/last before considering contrasts; no conclusions from selected
preview images alone. No new training or test evaluation is implied by completion.
