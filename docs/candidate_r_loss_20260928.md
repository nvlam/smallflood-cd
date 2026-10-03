# Candidate R loss implementation, isolated v1

## Material Passport

Origin: academic-research-suite, approved implementation of bounded redesign loss/tests.
Status: implemented only; effectiveness and novelty UNVERIFIED. No full-data/GPU training.
Original model, SmallFloodLoss, trainer, runners, metrics, checkpoints and configs unchanged.

## API

New module smallflood_cd.losses.local_component contains:
- build_local_supervision(target, valid): detached label-derived constant weight maps
  and component counts per image, with empty-ring count.
- local_loss(logits, supervision): stable softplus reduction on logits device.
- LocalComponentLoss: preparation plus reduction.
- CandidateRLoss: .4 unweighted BCE + .4 unweighted Tversky + .1 local auxiliary;
  rejects a boundary-on output. Deliberately not exported through existing loss factory.

Eligibility: 8-connected valid-masked GT area<=35, no component pixel touching patch
exterior or 8-neighbor invalid. Radius3 Chebyshev ring intersects valid background,
excluding all foreground. Each positive component mean and background ring mean has
factor .5; average over all eligible components in the minibatch, not over images.
Empty ring contributes zero negative term. Empty eligible batch yields differentiable
zero auxiliary term while retaining original global loss. Overlapping rings accumulate.
CPU SciPy labels/dilates only GT, bounding-box cropped to include full ring; no cache
or external files written. SciPy is an explicit runtime requirement already used by
object_boundary_v2, but not yet declared in the project's base dependencies. Missing
SciPy should stop the check; do not auto-upgrade server dependencies.

Dense positive/background coefficient maps are algebraically equivalent to explicit
per-component means. They are cast to logits dtype/device for GPU loss computation.
CPU preparation overhead and full-data gradient balance remain unmeasured. FP32 is
the intended training precision; no AMP deployment/training claim is made.

## Tests and limitations

Local complete targeted run: 18 passed, 1 skipped (CUDA unavailable). Expected on
the CUDA server: 19 passed. These totals include three existing base-loss tests.

Tests cover zero-logit analytic value, gradient signs, equal component mass, averaging
across batch components, size threshold35, invalid/edge exclusions, foreground removal
from rings, overlapping-ring value/gradient agreement with explicit formula, absent
eligible components, artificial empty-ring reduction, stable extreme logits, rejected
bad masks/shapes, input preservation, deterministic maps, random eligibility agreement
with evaluation, exact preservation of A base terms, rejection of boundary head and
real A-model synthetic backward without optimizer. Optional CUDA test compares loss
and gradient with CPU at atol1e-6 (not bitwise reproducibility).

Server command from /data/fit.lamnv/smallflood_cd with existing .venv:

```bash
python scripts/test_candidate_r_loss.py
```

Only synthetic tests: no real train/validation/test arrays, optimizer steps, checkpoint
loading or GPU preflight. CUDA availability enables only a tiny loss/gradient check.
Next step after approval/test evidence: train-only eligibility audit and isolated
runner integration/preflight. Existing factorial runners do NOT use this loss.
Adding source files invalidates old preflight fingerprints intentionally; do not rerun
old factorial jobs or bypass fingerprints to use R. A new runner/gate is required.
