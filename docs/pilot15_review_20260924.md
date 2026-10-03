# Pilot 15 epoch — validation review

## Material Passport

- Status: ANALYZED; no GPU rerun, training, test evaluation or code change.
- Source: user-supplied pilot15_review_20260924.tar.gz; extracted under artifacts/imported/pilot15_20260924/.
- Run: runs/pilot15/20260924T032911Z; review: 20260924T082109094745Z.
- Evidence: six pixel JSON reports, six object/boundary JSON reports, three training console logs, COMPLETE.json.
- Scope: one seed, three validation events, prepared-patch evaluation; not scene-level or edge-device benchmarking.

## Checks

All three console logs contain 15 epoch validation records and COMPLETE summaries.
Review reports six evaluations and test_used=false. All six pixel reports have 4,767
patches, 295,762,355 valid pixels and 3,182,289 positive pixels (1.076%).
Recomputed pixel F1 from confusion counts; object/boundary event counts sum to global
counts (floating IoU sums checked within 1e-6). Pixel and object reports reference the
same checkpoint SHA per evaluation. Best checkpoint epochs and macro F1 exactly agree
with the maximum event-macro F1 in the 15-epoch logs. No checkpoint binaries rerun here.

## Best checkpoints

All scores below in percent; pixel and object/boundary metrics globally pooled except
the explicitly named event-macro F1. Boundary IoU means two-pixel inner-band IoU.

| Metric | Proposed | FC-Siam-Diff | BIT implementation |
|---|---:|---:|---:|
| Selected epoch (1-based) | 13 | 12 | 7 |
| Event-macro F1 | 83.65 | 85.52 | 65.95 |
| Pixel precision | 86.99 | 92.14 | 67.46 |
| Pixel recall | 81.93 | 79.70 | 72.76 |
| Pixel F1 | 84.39 | 85.47 | 70.01 |
| Pixel IoU | 72.99 | 74.62 | 53.86 |
| Component F1 | 55.89 | 47.63 | 19.61 |
| Small interior component recall | 27.84 | 30.54 | 0.00 |
| Small interior matched / target | 103/370 | 113/370 | 0/370 |
| Boundary F1 | 77.42 | 78.49 | 22.54 |
| Boundary inner-band IoU | 32.45 | 47.80 | 5.08 |
| Training + validation minutes | 102.00 | 76.03 | 58.85 |
| Peak allocated training GPU GiB | 0.486 | 1.295 | 0.699 |

Proposed minus FC: event-macro F1 -1.88 percentage points, pixel F1 -1.08,
component F1 +8.25, small recall -2.70, boundary F1 -1.07, boundary IoU -15.36.
Component precision/recall: Proposed 65.45/48.76%; FC 43.53/52.59%.
Thus Proposed has a more precision-oriented component tradeoff, not uniformly better detection.
BIT's high component precision (77.03%) accompanies only 11.24% component recall.

Event F1 (Lumberton / Coraki / Hebei): Proposed 78.92 / 91.76 / 80.26;
FC 82.34 / 94.59 / 79.65; BIT 52.65 / 79.80 / 65.40.
Proposed exceeds FC on Hebei only; pooled scores do not imply every-event superiority.

## Training trajectory and missing evidence

Last-epoch event F1: Proposed 80.57, FC 85.17, BIT 57.60%, versus selected
83.65, 85.52, 65.95%. Validation trajectories fluctuate; this alone does not diagnose
overfitting. Console logs do not contain per-epoch training loss. Request metrics.jsonl,
config_resolved.yaml, environment.json, timing.jsonl and summary.json from each run
before examining train-loss/validation divergence or confirming resolved configuration fairness.
Training time and allocated memory are not inference latency or TensorRT memory evidence.

## Small-object and boundary caveats

Threshold is train-fitted 35 pixels, matching IoU 0.1, 8-connectivity. Only 370 of
1,551 small patch-clipped targets qualify as interior; 76.14% are excluded by the
rim/invalid-neighbor rule. Interior targets: Lumberton 0, Coraki 42, Hebei 328.
Small recall is therefore supported by two events and dominated by Hebei counts.
Annotations remove some tiny objects; recall is against released labels, not all real floods.
Boundary F1 tolerates two pixels; it is distinct from exact overlap of inner bands.
No visual inspection of preview PNGs performed in this review.

## Statistical/methodological screen (11/11)

1. Aggregation reversal: inspected global vs event F1; mixed event ranking, no universal superiority.
2. Ecological inference: no extrapolation from pooled pixels to every event or physical object.
3. Selection bias: limited validation events and interior-only subset; generalization unestablished.
4. Collider bias: no causal adjustment analysis; not applicable here.
5. Base rates: positives 1.076%; accuracy alone is misleading.
6. Regression to mean: best-epoch selection creates optimism; this is not an independent evaluation.
7. Survivorship: all three planned pilots completed; no inference about unreported historical runs.
8. Look-elsewhere: report all planned metrics, not only favorable component F1; no significance tests.
9. Forking paths: retain fixed 0.5 threshold and event-F1 selection; further tuning stays exploratory.
10. Causality: architecture/loss contribution cannot be isolated without ablations.
11. Reverse causality: not a causal observational analysis; not applicable.

No confidence intervals or significance claims: one seed and three events are insufficient
for claims of stable model superiority. Prior exact-match smoke evidence does not prove
15-epoch reproducibility. Scope of this review is arithmetic/protocol consistency, not replication.

## Recommended next gate (not executed)

1. Collect small per-run metadata and metrics.jsonl files; inspect actual losses and configurations.
2. Audit BIT implementation/output resolution and baseline fairness without modifying it yet.
3. Predefine controlled Proposed ablations (size-aware loss / boundary head) before new runs.
Do not launch the 135-run matrix, adjust probability threshold post hoc or open the test set yet.
