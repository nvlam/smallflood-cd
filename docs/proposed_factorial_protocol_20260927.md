# Proposed factorial diagnostic protocol v1

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan; Origin Date: 2026-09-27
- Verification Status: configuration checks only; experiments NOT RUN
- Version Label: proposed_factorial_v1
- Scope: approved protocol/configuration preparation, not execution authorization.

## Question and factors

On the same lightweight Proposed backbone, do size-aware weighting and the boundary
refinement/supervision package improve validation small-region recall and boundary
metrics under a fixed 15-epoch diagnostic training budget?

| Cell | Size weights | Boundary head + loss | Total loss |
|---|---|---|---|
| A | off | off | .4 BCE + .4 Tversky |
| B | on | off | .4 weighted BCE + .4 weighted Tversky |
| C | off | on | .4 BCE + .4 Tversky + .2 boundary |
| D | on | on | .4 weighted BCE + .4 weighted Tversky + .2 boundary |

Keep segmentation coefficients fixed, NOT .5/.5 when disabling boundary. Coefficients
need not sum to one. The boundary-on total coefficient sum is 1 vs .8 off; this is an
additive auxiliary objective, not a globally rescaled segmentation objective.
Global clipping can interact with the extra gradient; log pre-clip gradient norm and
clipping frequency. Effects describe the package under this optimizer, not intrinsic
module utility independently of optimization. Size weighting affects BOTH BCE and
Tversky, including weighted-BCE normalization, not Tversky in isolation.

Boundary-off actually removes the head and residual pathway, not just its loss.
Boundary-on combines extra parameters, residual inference and auxiliary supervision.
This 2x2 cannot separate those three effects; do not claim otherwise.

## Frozen controls

- Same existing train/validation manifest, 19166/4767 patches; no test arrays.
- Manifest SHA256 51eba373aef67e05b768bfc3c402d15c9fb9c8417c7026c6079a8fc0bea9b3ae.
- Component stats SHA256 525e4e1354e991983ebd2e3d3f68caad3f9894308a6eae46256a74405c9f2247.
- Normalization SHA256 33ef9a286340e3a58b8fc3623da210d83624c8a5901e9bcd0799e660a6ff625d.
- SAR pre 5/6, post 7/8, VH/VV; frozen train normalization, no coherence.
- Seed 42; 15 epochs, batch 8, FP32, workers 0, strict determinism, math attention.
- Fresh model/optimizer; AdamW LR .0003, weight decay .0001, clip 1.0; no scheduler,
  no warmup, no early stopping (patience 16), no augmentation changes or resampling.
- Tversky FP .3/FN .7; size R=65, alpha=1, gamma=.5, enhancement cap=4 (total cap=5).
- Boundary target and masking unchanged, initial residual scale .1, no postprocessing.
- Common backbone/decoder/change-head initial state must match exactly across A–D;
  C/D entire initial state must match. Current local constructor test passes for seed
  42; runner must verify on server and save initialization hashes.
- Independent seeded DataLoader generator, same recorded batch-order hash each epoch.
  Reset training RNG after model creation consistently across all four cells so extra
  head initialization does not shift subsequent random streams. Record this choice.
- Old D pilot is historical reference only; run all four fresh through the same future
  runner. Do not substitute an old checkpoint for D or claim exact repeat in advance.

## Analysis and checkpoint selection

Select best by event_pixel_v2 event-macro pixel F1 at fixed threshold .5, earlier wins
ties. Also report epoch-15 last for every cell. Never choose checkpoint by small recall
or boundary quality after seeing results.

Report primary diagnostic axes: pooled interior-small recall (<=35 pixels) and pooled
Boundary IoU. Report companion component P/R/F1, tolerance-based boundary P/R/F1,
pixel P/R/F1/IoU and event macro metrics, denominators and per-event results. Use
unchanged object_boundary_patch_v2; no cross-patch merging, no threshold sweeps.

For each metric M, report B-A and D-C (size effects), C-A and D-B (boundary package
effects), and interaction D-C-B+A. Positive differences are observations, not proof
of significance. Report all cells and trade-offs, no selective success criterion.
The historical evaluation has only 370 interior-small GT objects across two defined
events; patches/components are not independent experimental replicates. One seed
supports diagnosis only. Repeated-seed confirmation requires a separately approved
budget and frozen follow-up protocol; no p-values or broad generalization claims now.

## Required outputs and execution gate

Per run: resolved config, source/data hashes, environment, parameter count, initial-state
and batch-order hashes, best/last checkpoints, epoch train/validation metrics, component
loss values (unweighted and coefficient-scaled), component-weight summary, residual
scale, pre-clip gradient diagnostics, timing/memory and completion flag. Logs of losses
are diagnostics, not directly comparable quality scores across weighted/unweighted cells.
Post-training review produces the existing pixel/object/boundary reports and previews.

NOT ready to launch via generic run_experiment: it constructs and evaluates test data.
Existing next_steps pilot CLI does not accept these four config paths directly. Protocol
metadata flags in YAML are declarations, NOT security enforcement in existing runners.
A train/validation-only factorial runner, server preflight, safety tests and user launch
approval are required next. Do not add these files to the 135-experiment matrix.
Runner should stop on errors with no automatic retries or altered hyperparameters.
Monitor only new run logs; no unattended kill timeout chosen here.

## Configuration check outcome

Four YAML files are isolated under configs/experiment/proposed_factorial_v1/.
Loader resolves them through existing defaults convention. Three local tests verify
factor-only differences, fixed loss coefficients/budget, actual head presence and common
initial tensors, and D's loss recipe. No production model/training code was modified.
