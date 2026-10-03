# Pilot15 training/configuration audit

## Material Passport

Status: ANALYZED. Inputs: user-provided pilot15_training_details_20260924.tar.gz,
previous pilot15 review archive and its source_snapshot.tar.gz. No training or test
evaluation performed. One CPU zero-input forward used to inspect BIT tensor shapes;
this is not a performance or accuracy benchmark. No implementation changes made.

## Integrity

All 45 metrics.jsonl records match their corresponding console validation records,
including all event metrics. Every run has 15 epochs, 2,396 steps/epoch, 35,940 final
steps. Epoch numbering in metrics.jsonl is zero-based. Earlier selected checkpoints
remain Proposed epoch 13, FC epoch 12 and BIT epoch 7 (one-based).
BIT source, trainer and launcher in server source snapshot match current local files.

## Learning trajectories

| Model | Train loss epoch 1 | Train loss epoch 15 | Best event F1 (%) | Last event F1 (%) |
|---|---:|---:|---:|---:|
| Proposed | 0.452408 | 0.228416 | 83.6456 | 80.5683 |
| FC-Siam-Diff | 0.373985 | 0.184824 | 85.5226 | 85.1712 |
| BIT implementation | 0.543781 | 0.295419 | 65.9520 | 57.6012 |

All models optimize training loss. BIT loss continues decreasing after epoch 7 while
validation F1 does not surpass its epoch-7 maximum; consistent with a generalization
or optimization mismatch, not definitive proof of overfitting. Proposed also fluctuates;
FC recovers after large validation drops. No validation loss or training F1 is logged,
so a directly comparable train/validation generalization gap cannot be calculated.
Loss magnitudes must not be ranked across Proposed and baselines: objectives differ.

## Configuration comparability

Shared: train/validation manifest and normalization hashes, 19,166/4,767 patches,
seed 42, batch 8, 15 epochs, FP32, zero loader workers, strict deterministic settings,
AdamW launcher, weight decay 1e-4, clipping 1.0, no scheduler, threshold 0.5,
event-macro F1 selection, intensity-only two channels per date, no test use.
Proposed and BIT explicitly pretrained=false; FC uses fresh initialization.

Differences: Proposed/FC learning rate 3e-4 versus BIT 1e-4. Proposed objective
0.4 BCE + 0.4 size-aware Tversky + 0.2 boundary, with size-aware weights enabled;
baselines 0.5 BCE + 0.5 Tversky without size-aware weights or boundary supervision.
This compares complete method packages, not isolated architecture or loss effects.
Different learning rates are not automatically unfair but need a documented, comparable
validation-only tuning budget before final baseline claims.

Parameters: Proposed 1,080,805; FC 487,857; BIT 11,360,133. Proposed has about 90.49%
fewer parameters than this BIT implementation, but about 2.22 times FC's parameters.
Neither establishes inference-latency superiority. Config timing=20 warmups/100 timed
runs is unused by the pilot and does not match the frozen deployment 100/500 protocol;
correct the deployment configuration separately before actual deployment benchmarking.

## Concrete BIT architecture concern

Current backbone includes standard ResNet18 layer4. A CPU forward confirms projected
features [1,64,8,8] and decoder logits [1,1,8,8] for [1,2,256,256] inputs. Final output
is bilinear interpolation to [1,1,256,256]. No high-resolution skip or refinement path
is present. This coarse output grid is a plausible contributor to weak small-object
and boundary metrics, not a demonstrated sole cause. Full-resolution output shape
alone was insufficient as a smoke-test criterion. Official BIT fidelity has not been
established by this audit; do not label these results as a faithful official reproduction.

## Next gate

Before more training: read-only comparison with official BIT implementation and verify
spatial resolution, backbone stages, token/decoder structure and training recipe.
Then propose explicitly versioned adaptations for two-channel SAR and seek approval.
Keep existing runs immutable and label them engineering pilot/custom BIT implementation.
After baseline definition is fixed, specify controlled Proposed ablations for size-aware
loss and boundary head; do not change architecture, loss, LR and scheduler simultaneously.
Do not launch the 135-run matrix or open test yet.

## Statistical screen

11/11 categories considered, continuing the previous review: grouped/pooled ranking and
ecological inference remain limited to three events; selection bias from small-interior
filter remains; collider and reverse-causality analyses not applicable; rare-positive base
rate remains 1.076%; best-epoch optimism/regression-to-mean risk remains; all three runs
completed (no observed completion filtering); no significance testing or cherry-picked
new threshold; future tuning remains exploratory (forking-path risk); no module-level
causal attribution without ablations. No p-values, confidence intervals or long-run
reproducibility claims inferred from this single-seed evidence.
