# Proposed implementation audit

## Material Passport

- Origin: academic-research-suite / experiment-agent, diagnostic validation.
- Date: 2026-09-25.
- Verification Status: ANALYZED, local source and synthetic CPU forward/backward only.
- Inputs: current source; imported Proposed pilot resolved config and existing metrics.
- No production implementation changes, optimizer steps, remote training or test access.

## Wiring

The pilot config has use_size_aware=true, boundary_head=true and loss coefficients
0.4 BCE / 0.4 Tversky / 0.2 boundary. next_steps.pilot passes the same loss config to
_dataset and constructs SmallFloodLoss with these coefficients. _dataset creates a
connected-component cache and ManifestDataset generates component_weights. Trainer
moves them to the active device and passes them to the composite loss. A missing or
non-tensor field silently falls back to unit weights: this is a fail-open risk, not
proof that the actual pilot omitted weights.

For component area A, reference area R:
w(A)=1+alpha*min(cap,(R/(A+1e-6))^gamma); background weight=1.
Pilot alpha=1, gamma=0.5, cap=4 means maximum total weight 5, not 4. All foreground
components receive some enhancement, not only those below evaluation threshold 35.
The train reference area R is separate from that threshold. Its real numeric value
and the distribution of actual weights were not available locally.

BCE is sum(valid*w*BCE)/sum(valid*w). Weighted TP and FN enter Tversky, while FP is
unweighted; loss=1-(TPw+eps)/(TPw+0.3FP+0.7FNw+eps). Terms are pooled over the batch,
not averaged per component. Larger objects can still dominate: uncapped total
foreground weight is approximately A+alpha*sqrt(R*A), not constant per object.

Boundary target is a radius-1 morphological gradient (inner and outer pixels), not
the evaluation's one-pixel inner contour. Boundary loss is half BCE plus half Dice.
The feature head predicts boundary logits; sigmoid probabilities join decoder
features to predict a residual. Refined logit=preliminary+clamp(s,0,1)*residual,
then bilinear upsampling from stride 4. Both segmentation and boundary losses can
backpropagate to the boundary branch. Inference uses refined logits.

## Executed checks

- 11 existing tests passed: weights, cache, Tversky, composite loss, boundary supervision,
  model.
- 4 new diagnostic tests passed in tests/unit/test_proposed_contribution_audit.py:
  weights alter BCE/Tversky logit gradients; small components receive larger per-pixel
  gradients in the controlled example; actual Proposed forward/backward at 256x256
  passes gradients through boundary/features/residual/scale; out-of-range scale has
  zero refinement gradient; invalid-neighbor boundary remains supervised.
- These checks establish local wiring, not learned checkpoint behavior or remote
  training-time gradient magnitudes. No model/data modifications were made.

## Risks, not established explanations of pilot ranking

1. The learned scale uses hard clamp. When s<0 or s>1 its gradient is zero. Below zero
   refinement is disabled although auxiliary boundary training can continue. Need
   checkpoint scalar inspection before attributing the pilot result to this issue.
2. Boundary supervision uses valid mask without the 3-pixel erosion used by the
   evaluation. Mask-cut boundaries adjacent to invalid regions can receive training
   supervision. Reproduced with a synthetic example; real affected count is unknown.
3. Component labels are generated from target before applying valid/uncertain masks.
   Invalid positive labels could alter component sizes or bridge objects even though
   their loss is masked. Real-data audit is needed; no assertion that these exist.
4. Trainer's boundary radius uses function default 1, not config forwarding. Pilot
   radius is 1, so no demonstrated mismatch in this run; future radius ablations need
   explicit wiring verification.
5. decoder_channels is present in config but registry/decoder use projection_channels.
   It is not an active knob in this implementation. It does not explain pilot ranking,
   but must not be advertised as an implemented decoder-width ablation.

## Next evidence gate

Inspect boundary_head.residual_scale in trusted Proposed best/last checkpoints on
server; report full value and effective clamp, without modifying weights. Then obtain
train-only weight quantiles, fraction capped, component sizes after valid masking,
and invalid-neighbor boundary counts using a predefined sampling/full-scan protocol.
Do not silently change training code or launch new experiments. The present evidence
does not establish that size-aware loss or boundary refinement improves accuracy;
controlled within-Proposed ablations remain necessary.
