# Factorial validation audit

## Material Passport

- Origin: academic-research-suite / experiment-agent, validate
- Date: 2026-09-28; Verification Status: ANALYZED
- Source: proposed_factorial_validation_results_20260928.tar.gz
- Scope: eight imported reports; no checkpoint inference or training rerun.

## Integrity

All eight checkpoint epochs and global pixel TP/FP/FN/TN match training logs.
Recomputed object scores, aggregate counts and event macro values from event counts
match report fields. Same protocol, module hash, component-statistics hash and manifest
hash throughout. Each evaluation has 4767 patches, 6745 GT components, 370 eligible
interior-small GT components, threshold 35. COMPLETE records eight evaluations,
test_used=false and training_performed=false. This verifies internal report consistency,
not an independent raw-data prediction rerun. Reports use object_boundary_patch_v2.

## Best checkpoint results

All percentages except epoch and counts. Selection remains event-macro pixel F1;
object/boundary metrics below use global count pooling.

| Cell | Best epoch | Event F1 | Pixel F1 | Component precision | Component F1 | Small recall | Boundary F1 | Boundary inner-band IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A: neither | 12 | 82.84 | 84.64 | 57.93 | 52.85 | 34.05 (126/370) | 76.92 | 30.46 |
| B: size | 13 | 82.82 | 83.56 | 63.95 | 53.82 | 20.81 (77/370) | 75.70 | 31.92 |
| C: boundary package | 12 | 82.95 | 83.97 | 62.71 | 53.87 | 24.86 (92/370) | 77.07 | 33.24 |
| D: both | 14 | 82.64 | 83.17 | 61.43 | 54.15 | 28.38 (105/370) | 76.64 | 33.17 |

Overall component precision is NOT precision restricted to small predictions. No
small-component F1 is claimed. D has slightly higher overall component F1 than A,
but lower small recall and pixel F1. No single model dominates all diagnostic axes.

## Predefined best-checkpoint contrasts (percentage points)

| Contrast | Small recall | Component F1 | Boundary F1 | Boundary IoU |
|---|---:|---:|---:|---:|
| B-A (size without boundary) | -13.24 | +0.97 | -1.21 | +1.46 |
| D-C (size with boundary) | +3.51 | +0.27 | -0.43 | -0.07 |
| C-A (boundary without size) | -9.19 | +1.02 | +0.15 | +2.78 |
| D-B (boundary with size) | +7.57 | +0.33 | +0.94 | +1.25 |
| D-C-B+A (interaction) | +16.76 | -0.70 | +0.78 | -1.53 |

A positive interaction does not mean D beats A: D-A small recall=-5.68 points,
21 fewer detections. These contrasts involve independently validation-selected epochs,
and describe the selected pipelines, not a fixed-time mechanistic effect.

## Last checkpoint sensitivity (epoch 15 for all)

| Cell | Small recall | Component F1 | Boundary F1 | Boundary IoU |
|---|---:|---:|---:|---:|
| A | 22.97 (85/370) | 53.20 | 76.22 | 32.79 |
| B | 20.54 (76/370) | 51.26 | 73.88 | 32.05 |
| C | 19.46 (72/370) | 45.32 | 73.07 | 32.49 |
| D | 22.97 (85/370) | 53.33 | 74.48 | 30.71 |

Small-recall contrasts B-A=-2.43, D-C=+3.51, C-A=-3.51, D-B=+2.43; interaction=+5.95.
Boundary-IoU contrasts B-A=-0.75, D-C=-1.77, C-A=-0.31, D-B=-1.33. Boundary-on IoU
advantages at best do not hold at last. Equal A/D total small matches at last do NOT
establish identical detected objects (object identity matching across models not done).

## Event evidence

Best interior-small matches: Coraki A24/B22/C23/D22 out of42; Hebei A102/B55/C69/D83
out of328. Lumberton has zero eligible interior-small GT: recall undefined, not zero.
A is highest in both defined events at best. Size helps with boundary in the pooled
result because D-C gains 14 Hebei objects while losing one Coraki object. Hence even
the positive conditional size effect is not uniform across events. At last, Coraki
A25/B20/C21/D24; Hebei A60/B56/C51/D61. The set is dominated by Hebei (328/370).
Interior eligibility excludes 1181/1551 small patch-clipped GT components; conclusions
are not about all physical small floods or building/road damage.

## Interpretation

Current evidence does not support the original strong claim that the full size-aware
plus boundary method improves small-region detection while preserving baseline
quality. Weighting is active, but increased per-pixel weight is not evidence of better
recall. No implementation defect or causal explanation is established by these scores.
There is a limited best-checkpoint boundary-IoU signal, accompanied by small-recall
loss and checkpoint sensitivity. Do not rename that a robust boundary improvement.

All runs are seed42, 15-epoch development pilots, with three validation events and two
defined small-region events. No significance/equivalence claim or multi-seed uncertainty.
No post-hoc change to checkpoint selection, threshold or metric definition. Historical
Proposed pilot is not substituted for D; new factorial RNG/order protocol differs.

## Fallacy scan — 11/11

1. Aggregation: per-event conditional size effects differ; pooled scores not universal.
2. Ecological: no physical asset-damage inference from patch components.
3. Selection: interior and checkpoint-selected populations explicitly restricted.
4. Collider: no covariate-conditioned causal model, not applicable.
5. Base rates: pixel accuracy not used as primary evidence with ~1.076% positives.
6. Regression to mean: selected epoch maxima and one seed may be unstable.
7. Survivorship: all eight reports and both best/last retained.
8. Look-elsewhere: report all four predefined contrasts and both checkpoint policies.
9. Forking paths: any future redesign is new development; preserve these negatives.
10. Causation: package ablation does not isolate architecture vs auxiliary gradients;
    cannot explain the mechanism from metrics alone.
11. Reverse causality: no observational direction claim, not applicable.

## Recommended decision gate — no action executed

Do not launch the 135-experiment matrix or claim the initial novelty has succeeded.
Retain A as a diagnostic reference, not a final selected method. Before further GPU
spending, decide whether to permit one bounded redesign of the small-change objective
with explicit rationale, fixed controls and a stop criterion; or preserve this as a
negative pilot and reconsider the research question. Simply increasing weights,
training longer or selecting by small recall now would not validate the frozen claim.
