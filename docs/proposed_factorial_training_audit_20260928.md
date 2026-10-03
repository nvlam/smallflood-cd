# Proposed factorial training audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: validate; Date: 2026-09-28
- Verification Status: ANALYZED (logs and metadata, no checkpoint/inference rerun)
- Source: proposed_factorial_review_20260928.tar.gz, run 20260927T075837Z
- Confidence: arithmetic/pairing internally consistent; scientific interpretation CAUTION.

## Checks completed

All four cells have epochs 0–14, 2396 steps per epoch, 35940 per cell (143760 total).
COMPLETE.json and provenance.json fingerprints agree. Four summaries match the group
completion records. Common initialization hash matches across all cells, full state
hashes match A/B and C/D, and all 15 batch-order hashes match across four cells and logs.
85 source/config files available locally match recorded hashes. Resolved loss/model/
training settings match provenance. No test usage is reported. Checkpoint files were
excluded from archive; their contents and reload claims cannot be independently
reverified here.

For all 60 epochs, recomputed global F1/IoU and event-macro F1 match logs. Valid pixel
count is 295762355 and positive count 3182289 in every validation. Scaled component
losses sum to train loss. Train foreground count 23436067 every epoch; positive weight
mass matches unit weights for A/C and approximately 28626572.57 for B/D. Best scores
match summary maxima, selecting earlier epochs on ties.

## Results (percent, best selected by event-macro F1)

| Cell | Size | Boundary | Best epoch | Best event F1 | Pixel F1 at best | Pixel IoU at best | Last event F1 |
|---|---|---|---:|---:|---:|---:|---:|
| A | off | off | 12 | 82.84 | 84.64 | 73.37 | 82.57 |
| B | on | off | 13 | 82.82 | 83.56 | 71.76 | 80.56 |
| C | off | on | 12 | 82.95 | 83.97 | 72.37 | 76.21 |
| D | on | on | 14 | 82.64 | 83.17 | 71.19 | 80.20 |

Best-checkpoint event F1 contrasts (percentage points): B-A=-0.0203; D-C=-0.3141;
C-A=+0.1087; D-B=-0.1852; interaction D-C-B+A=-0.2939. These are descriptive contrasts
of a validation-selected pipeline, not same-epoch causal effects or statistical tests.
Last-epoch contrasts: B-A=-2.0080; D-C=+3.9894; C-A=-6.3582; D-B=-0.3607;
interaction=+5.9974. Ranking/interaction depends on checkpoint choice, so do not omit
last results. Neither measure establishes effects on small-component/boundary metrics.

## Optimization and timing

Mean clipping fractions across 15 equally sized epochs: A 34.86%, B 31.21%, C 46.38%,
D 45.40%. Boundary package increases clipping frequency in these runs; this is not
proof of exploding gradients or sole explanation of quality. C/D logged scales stay
inside (0,1), ending .815858 and .752798. No logged epoch-end saturation.

C best-to-last event F1 drops 6.74 points; Lumberton F1 goes from 77.49% (best) to
60.80% (last). This is observed validation deterioration, not proof of overfitting
without additional diagnostics/repeats. Loss values between different objectives
must not be interpreted as comparative quality scores.

Reported run hours: A 1.52, B 1.62, C 2.59, D 1.69 (about 7.42h total). C epochs 3/4/5
take about 1432/1454/1137 seconds, versus many cycles around 370–450 seconds. A first
epoch is 721 seconds. No system telemetry establishes causes. Timing includes data,
training, validation/checkpoint overhead and is NOT a deployment/architecture benchmark.
Parameters: A/B 1079865, C/D 1080805; boundary package adds 940 parameters.

## Fallacy scan — 11/11

1. Aggregation: event and pooled pixel rankings differ; report both and events.
2. Ecological: pixel/event quality does not establish building/road damage detection.
3. Selection: validation-selected maxima and interior-component eligibility limit scope.
4. Collider: no covariate-adjusted causal analysis; not applicable.
5. Base rate: validation positive prevalence ~1.076%; accuracy is not primary evidence.
6. Regression to mean: one-seed maxima are not stable replicated estimates.
7. Survivorship: all 60 scheduled epoch records are present; archive preserves all cells.
8. Look-elsewhere: report both best/last and all predefined metrics, not selected wins.
9. Forking paths: no threshold/weight changes based on these results; this is development.
10. Causation: contrasts concern package/optimizer interaction under one training budget;
    no general mechanism claims, and no independent head-vs-supervision separation.
11. Reverse causality: no observational directional inference; not applicable.

No p-values/CIs/repeated-seed uncertainty available. Do not claim equivalence from
small differences, or absence of small-region benefit from global pixel metrics.
New D is not an exact replication of historical D: frozen factorial ordering and RNG
reset differ by design. Historical results must remain labeled separately.

## Next gate

Prepare a factorial-aware validation-only review wrapper for all eight best/last
checkpoints, preserving threshold .5 and object_boundary_patch_v2. Verify checkpoint
epochs against logs, collect small-region denominators and event-level diagnostics,
then compute the predefined contrasts. No retraining, test access, automatic model
revision or 135-experiment launch is justified by this log-only audit.
