# Bounded redesign proposal — component-normalized auxiliary supervision

## Material Passport

- Origin: academic-research-suite / experiment-agent, plan
- Date: 2026-09-28; Version: redesign_candidate_v1
- Verification Status: UNVERIFIED hypothesis; no implementation or training
- Scope: one candidate and one controlled diagnostic comparison, pending approval.

## Evidence and non-claims

The frozen factorial found best interior-small recall A126/370, B77/370, C92/370,
D105/370. Current capped inverse-sqrt weighting increases small foreground weighted
mass share from .82% to 1.87%, but this is not measured gradient share or proof of
the cause of missed detections. Backbone stride4 and labels may also limit detection.
We propose testing a more direct per-component objective, not claiming a diagnosed
optimization defect. Retain all negative results and the original protocol unchanged.

## One candidate R

Use A architecture, boundary head OFF, original size weights OFF. No additional
inference layers, no sampler changes, no postprocessing. Train-only auxiliary loss:

L_R = .4 L_BCE + .4 L_Tversky + .1 L_local.

Global BCE and Tversky are exactly A's unweighted objectives, FP=.3, FN=.7. Do not
renormalize coefficients. Added objective may interact with clipping; log gradients.
Train ground truth only defines connected components and local neighborhoods.

For each minibatch, label valid-masked GT using 8-connectivity. Eligible C_k have
1<=|C_k|<=35 and do not touch image exterior or an 8-neighbor invalid pixel, matching
the already-defined interior small-object evaluation population. No predictions,
validation labels, coherence, or manufactured uncertainty masks determine eligibility.

Let R_k = (Chebyshev dilation(C_k, radius3) minus C_k) intersect valid intersect
background GT. Exclude other foreground from R_k. Compute with logits z using stable
softplus, not clamped probability logarithms:

positive_k = mean_{i in C_k} softplus(-z_i)
negative_k = mean_{j in R_k} softplus(z_j), or 0 if R_k empty
local_k = .5 positive_k + .5 negative_k
L_local = mean over eligible components in the entire minibatch, or differentiable
zero if none exist. Log empty-ring count and empty-eligible-batch fraction. Rings may
overlap and repeat supervision; retain this explicit rule rather than deduplicating
silently. No changes to valid mask, segmentation targets, inference or global loss.

Each component has equal nominal auxiliary coefficient regardless of area, and local
background is explicitly penalized. This does not guarantee equal actual gradients,
correct instance matching, or improved recall/precision. Eligibility focuses training
on a restricted subset; also report clipped-small recall and all component metrics.
Tiny noisy annotations can be amplified; never interpret all labeled fragments as
physical buildings, roads or complete flood objects.

## Prior art and novelty gate

Instance imbalance is not new. Blob loss (preprint 2022, revised2023) addresses
instance-sensitive segmentation; ICI loss (2023) includes instance-wise supervision.
Sources inspected on 2026-09-28:
- https://arxiv.org/abs/2205.08209 (v3 dated2023-06-06)
- https://arxiv.org/abs/2304.06229 (2023-04-13)

This candidate is an adaptation hypothesis, NOT a claimed novel loss. Abstract-level
source checks establish prior-art overlap, not a comprehensive novelty clearance.
Before claiming a method contribution, compare full equations and implementations
against blob/ICI and related size-balanced losses, and justify the validity-aware
local-background formulation with direct controls. A successful pilot alone is not
publication evidence. No claim that these medical-image results transfer to SAR.

## Fixed two-run budget (not authorized to launch)

Fresh A reference and fresh R, each15 epochs, batch8, seed42, FP32, same AdamW .0003,
weight decay .0001, clip1, no scheduler/early stop, unchanged train/validation split.
Same common initial weights and every batch-order hash; same RNG reset convention as
factorial_v1. Two full runs total, no lambda/radius sweep or automatic retries.
Lambda=.1 and radius3 are proposed fixed engineering choices, not tuned values or
proven optima. Preflight must check loss/gradient finiteness, valid masking, empty
cases and deterministic pairing before any full run; no test arrays.

Keep event-macro F1 at threshold .5 as checkpoint selection, earlier wins ties. Review
best and last for both cells. Existing A historical run is context, not substituted
for the fresh control. Do not retrospectively select checkpoint by small recall.

## Predefined screening gate (engineering decision, not significance)

Advance only if all hold at best selected checkpoint:
1. Pooled interior-small recall R-A >=5 percentage points (>=19 extra matched GT if
   denominator remains370; verify denominator before analysis).
2. Event-macro pixel F1 loss <=1 percentage point.
3. Overall component precision loss <=2 percentage points (not small-only precision).
4. Interior-small recall does not decrease in either defined validation event.
And at last epoch, pooled interior-small recall difference must be nonnegative.
Report all other metrics/trade-offs including boundaries, clipped-small recall and
per-event denominators; no p-values from dependent pixels/components.

These margins are proposed project stop/go criteria, not clinical/operational
requirements or statistical noninferiority bounds. They require user approval before
training. Failure means stop this candidate without more weight/epoch/seed searches;
reconsider direction. Passing permits consideration of paired repeated-seed confirmation
under a separately approved budget, not a claim of novelty, robustness or acceptance.

## Deliverables and boundaries

If approved next: implement isolated auxiliary loss, synthetic unit tests and train-only
eligibility audit (including counts per event and no-small batch rate); then bounded
GPU preflight. Keep current model/loss code paths versioned and unchanged. Checkpoint
and log outputs must distinguish R from the original Proposed.
No entry command exists yet. No source implementation, config, training, test access,
new baseline run, or 135-matrix execution in this design stage.
Inference graph/parameter count remains A's by construction; actual device latency,
memory, serialized size and deployment budget still require measurement.
