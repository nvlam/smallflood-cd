# Step 2: patch-scoped object and boundary diagnostics

Status: engineering validation only. No training, no test data, no change to checkpoint selection.
User scope: all three pilot models, both existing best and last checkpoints.

## Components

Threshold probability at 0.5, apply valid mask, label with 8-connectivity in each patch.
Use sparse overlap intersections to compute IoU. Candidates have IoU >= 0.1.
Sort by decreasing IoU, then ascending predicted ID and GT ID; accept each ID once.
This is deterministic greedy matching, NOT maximum-cardinality or optimal assignment.
Splits/merges therefore receive at most one match per component. IoU 0.1 is a permissive
detection threshold, not evidence of high segmentation fidelity; also report matched mean IoU.

With M matches, P predicted and T target components:
precision=M/P; recall=M/T; F1=2M/(P+T).
Threshold for small GT components comes from train-fitted component_stats_v1.json (currently 35 pixels).
All component metrics are PATCH-CLIPPED; a scene object crossing tiles can count more than once.

Report both small recalls:
- clipped: matched small GT / all small GT;
- interior: matched small GT not touching a patch/invalid rim / all such interior small GT.

The rim is valid pixels outside a 1-pixel 8-neighbor erosion of valid, with outside-image invalid.
Report the number of predicted/target components touching it. Matching is done on all components
before stratifying small GT; one large prediction cannot independently match several small GTs.
Interior recall is a completeness-aware diagnostic, not a replacement for scene-level evaluation.
Do not invent small-object precision/F1 from GT-size-conditioned recall: this needs a separate,
agreed definition of eligible predictions. Current module reports overall component F1 plus small recall.

## Boundaries

For mask X, inner contour C(X)=X\erosion_1(X), inner band B(X)=X\erosion_2(X).
Use square 3x3 morphology (Chebyshev distances). Compute from valid-masked binary regions,
then restrict contours and bands to V_safe=erosion_3(valid), with outside-image invalid.
This excludes artificial edges at tile borders and NoData transitions, at the cost of not evaluating
boundary pixels in that rim. Report boundary_valid_pixels as coverage.

Precision: predicted contour pixels within 2 pixels of a target contour / predicted contour count.
Recall: target contour pixels within 2 pixels of a predicted contour / target contour count.
F1: harmonic mean; one contour empty -> 0, both empty -> null.
Boundary inner-band IoU: |B(pred) intersect B(GT) intersect V_safe| /
|[B(pred) union B(GT)] intersect V_safe|. This is NOT the IoU of dilated contours.

## Aggregation and integrity

Sum counts within event and across all patches BEFORE computing ratios. Report event macro separately.
Zero denominators yield null, omit undefined event values from macro and disclose defined counts.
Keep raw per-patch counts and per-event denominators in CSV. Never average per-patch F1s.
All models use identical rules, unchanged saved predictions at threshold 0.5 and the same validation set.
No uncertain masks are fabricated. This script rejects manifests with uncertainty/coherence inputs.
Existing checkpoints and pilot reports are not modified; version and hashes saved with new reports.

## Remaining gate

This is deliberately a separate diagnostic module. Legacy full-matrix evaluator still needs integration
after definitions are checked. Do NOT run all 135 experiments based on this patch alone.
Scene reconstruction is needed before claiming scene-level small-object/boundary results.
Do not refit the 35-pixel threshold on validation/test. Data bands and CUDA reproducibility remain pending.
