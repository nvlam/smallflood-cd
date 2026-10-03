# BIT-SAR v2 validation audit — 2026-09-25

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Verification Status: ANALYZED (imported reports, no local GPU inference rerun)
- Version Label: validation_v1
- Overall Confidence: CAUTION for generalization; report arithmetic consistent.
- Source: bit_sar_v2_validation_results_20260925.tar.gz, imported under artifacts/imported/bit_sar_v2_validation_20260925.
- Comparator source: artifacts/imported/pilot15_20260924, best checkpoints.

## Integrity checks

COMPLETE.json reports two evaluations, test_used=false. Both checkpoint reports identify
epoch 15. Their object counts, scores, event metrics and protocols match exactly;
checkpoint SHA256 differs, so identical serialized checkpoint files are not claimed.
Recomputed object scores and event macro values from report event counts match both
reports. Pixel counts match the previously audited epoch-15 training log.
No original checkpoint or raw NPY files were supplied; no independent inference rerun.

All comparator reports share manifest SHA256, object module SHA256, component-statistics
SHA256, full object_boundary_patch_v2 protocol and small threshold 35. GT denominators
match: 4,767 patches, 6,745 components, 1,551 small patch-clipped components, 370 interior
small components. Components are patch-scoped, not scene-reconstructed objects.

## Best-checkpoint comparison

All entries are percentages except epoch and small matched/target counts. Only event
F1 is event-macro; other columns pool counts globally. Checkpoints selected by event
macro pixel F1 at threshold 0.5, not by object/boundary diagnostics.

| Model | Epoch | Event F1 | Pixel F1 | Pixel IoU | Component F1 | Small recall | Boundary F1 | Inner-band IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Proposed | 13 | 83.65 | 84.39 | 72.99 | 55.89 | 27.84 (103/370) | 77.42 | 32.45 |
| FC-Siam-Diff | 12 | 85.52 | 85.47 | 74.62 | 47.63 | 30.54 (113/370) | 78.49 | 47.80 |
| BIT-SAR v2 | 15 | 85.00 | 86.30 | 75.89 | 57.88 | 29.19 (108/370) | 78.95 | 34.02 |

These are descriptive single-seed, 15-epoch validation pilots, not significance tests.
BIT-SAR v2 leads pooled pixel F1/IoU, component F1 and tolerance-based boundary F1.
FC-Siam-Diff leads event-macro F1, interior-small recall and inner-band IoU.
Proposed does not lead these quality metrics. Its small recall is 5 detections below
BIT-SAR v2 and 10 below FC-Siam-Diff; this is not evidence of statistical superiority
or inferiority. Boundary F1 (2-pixel tolerance) and inner-band IoU measure different
properties, so their ranking need not agree.

## Small-region scope

BIT-SAR v2: Coraki 24/42 (57.14%), Hebei 84/328 (25.61%), Lumberton 0/0 (undefined).
Event-macro small recall is 41.38% across only two defined events; pooled recall is
29.19%. Hebei supplies 328/370 eligible objects. The interior criterion excludes
1,181/1,551 small patch-clipped objects (76.14%); do not generalize to all small floods.
FC-Siam-Diff has 29/42 and 84/328; Proposed has 23/42 and 80/328 respectively.

## Visual spot checks

Viewed BIT best images 000 (Lumberton), 006 (Coraki), 010 (Hebei), one positive
example per event. 000 shows missed labeled foreground and isolated false alarms;
006 shows broad region overlap with smoother contours and false positives around
parts of the boundary; 010 shows missed small labeled regions. These are deterministic
preview examples, not a random sample and not a quantitative small-area measurement.
No building/road damage identity can be inferred from these previews alone.

## Fallacy scan — 11/11 covered

1. Simpson/aggregation: model ranking differs between pooled and event-macro pixel
   F1; report both, not a claim of formal Simpson's paradox.
2. Ecological inference: patch/event results do not establish building-level damage.
3. Selection/Berkson: interior eligibility restricts the evaluated small-object set;
   causal selection mechanism not established.
4. Collider: no causal covariate-adjusted model here; not applicable.
5. Base rates: positive pixels are 1.076%; accuracy is not primary evidence.
6. Regression to mean: one selected checkpoint/seed; independent repeats needed.
7. Survivorship: all 4,767 validation patches accounted for; keep documented failed
   engineering runs and censored-component counts, not only successful outcomes.
8. Look-elsewhere: multiple metrics, no inferential testing; do not cherry-pick wins.
9. Forking paths: preserve BIT replacement provenance and freeze protocol before
   confirmation; any Proposed revision is further development, not held-out evidence.
10. Causation: loss/head effects need controlled within-Proposed ablations.
11. Reverse causality: no observational causal-direction claim; not applicable.

No p-values, confidence intervals or repeated-seed estimates are available. Pixels,
patches and components within events must not be treated as independent replicates
for significance claims. Three validation events do not demonstrate broad cross-season
or cross-region robustness. A6000 training measurements do not validate Orin deployment.

## Recommended next gate (not executed)

Audit the active Proposed loss/head wiring and gradient paths before changing settings.
If consistent, predefine a bounded within-backbone 2x2 ablation: size-aware loss off/on
and boundary head plus supervision off/on; keep data, seed, epochs, optimizer and
selection fixed and explicitly document loss normalization. Obtain user approval of
the concrete protocol before launching. Do not tune on test or launch the full matrix.
