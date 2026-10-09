# recipe_confirmation_v1: phase-2 results and closure

## Material Passport

Protocol: `docs/recipe_confirmation_protocol_v1_20261008.md` (branch
`direction-b-confirmation`, commit `668f450`). Runners, reviewer, tests and launcher: commit
`3ca416c`, copied to the server by sha256-verified file transfer (the server project is not a
git checkout).
Execution: server run `runs/recipe_confirmation_v1/20261007T234436Z`, logs
`artifacts/recipe_confirmation_v1/20261007T234436Z`, reviews
`artifacts/recipe_confirmation_review/20261007T234436Z/{s201,s202,s203}`. Started
2026-10-07T23:44Z, completed 2026-10-09T04:09Z (about 28.4 h).
Scope: validation only. The held-out test set stayed sealed.
Status: **closed.** Under the §8 rules, H0 returns **stable** for arm A, and H1 (both rules)
and H3 return **inconsistent**. Per protocol this triggers no further runs. **The H0 outcome
must not be read as "R1 fixed the instability"**: arm A is stable because it fails on Lumberton
in every seed (§3). Sections 5–6 are post-hoc descriptive diagnostics, not preregistered
analyses.

## 1. Execution and integrity

- Source identity: the six new files on the server match commit `3ca416c` by sha256. Every
  other tracked source, script, config and test file shared by both trees was identical
  before launch (only installation `egg-info` metadata differed).
- Pre-launch checks, all passed:
  - synthetic tests: factorial runner 27, pilot runner 29, reviewer 28 passed, none skipped
    (the two fork-worker tests skipped on macOS ran on the server);
  - seed-42 regression (`regress_s42/REGRESSION_PASS.json`): initial-state anchors A/B
    `71c04062…54b8` and C/D `db03a4cf…e476` and all 15 batch-order hashes reproduced;
    `r1_matches_recipe_stability: true`; workers 0 and 4 equivalent for A **and** C, so the
    factorial runs used `num_workers = 4`;
  - per seed: factorial 1-epoch preflight, BIT-SAR v2 batch-8 gate, FC-Siam-Diff 1-epoch
    preflight (full training data, see §7).
- All 18 runs completed 15 epochs and 35,940 optimizer steps at learning rate 1e-4, constant.
  Parameter counts: FC-Siam-Diff 487,857; BIT-SAR v2 3,492,642.
- Factorial pairing held within each seed: common initial state and identical batch order
  across A–D (seed 201 `e69b7a42…`, 202 `63808d87…`, 203 `5886248f…`); A/B and C/D share
  full initial states.
- Review `COMPLETE.json` (top level and each seed): 36 evaluations; `test_used: false`;
  `training_performed: false`; `pixel_counts_match_training: true`;
  `denominators_match_frozen: true`. Reviewer SHA256 `b9b5754b…a3f4`. `DECISION.json`
  SHA256 `17a86c4f…d79c`.
- Elapsed per run: factorial cells 1.33–1.48 h, FC-Siam-Diff 1.28–1.30 h, BIT-SAR v2
  2.11–2.15 h. The server was lightly loaded (load average about 2) throughout.
- Storage: 1.3 GB, within the 1–1.5 GB estimate.

## 2. Decision rules (§8, epoch-15 checkpoint, seeds 201 / 202 / 203)

Percentage points.

| Rule | Seed 201 | Seed 202 | Seed 203 | Outcome |
|---|---|---|---|---|
| H0 recipe transfer, arm A: macro range ≤ 5 pp and Lumberton ≥ 50% of median | macro 58.74, Lumberton 8.8 | macro 59.03, Lumberton 3.8 | macro 59.68, Lumberton 6.1 | **Stable** (range 0.94; median 6.1, floor 3.05) |
| H1 size weighting: B−A ≤ 0 and D−C ≤ 0, interior-small recall | B−A +6.22; D−C +4.86 | B−A −0.81; D−C −1.08 | B−A +5.68; D−C +10.54 | **Inconsistent** |
| H1 boundary package: C−A ≤ 0 and D−B ≤ 0, inner-band IoU | C−A +5.79; D−B +2.96 | C−A −0.39; D−B +6.71 | C−A +2.14; D−B +0.47 | **Inconsistent** |
| H3: FC ≥ A and FC ≥ BIT-SAR v2, event-macro F1 | FC−A +3.59; FC−BIT −21.11 | FC−A +24.38; FC−BIT +1.96 | FC−A −3.00; FC−BIT −27.79 | **Inconsistent** |

Because H0 is "stable", no H1/H3 outcome carries the "not stable" label. That label would
have been informative here; see §3.

**H0 for the other arms (reported without a rule):**

| Arm | Macro F1 s201 / s202 / s203 | Range | Lumberton F1 s201 / s202 / s203 | Stable by the H0 criteria |
|---|---|---:|---|---|
| A | 58.74 / 59.03 / 59.68 | 0.94 | 8.8 / 3.8 / 6.1 | yes |
| B | 56.51 / 57.06 / 62.08 | 5.57 | 3.5 / 5.7 / 20.9 | no (range) |
| C | 75.45 / 59.44 / 59.91 | 16.01 | 61.5 / 6.8 / 8.3 | no (range) |
| D | 59.17 / 62.42 / 62.05 | 3.24 | 17.1 / 16.7 / 13.7 | yes |
| FC-Siam-Diff | 62.32 / 83.41 / 56.68 | 26.73 | 19.0 / 82.0 / 7.2 | no (range) |
| BIT-SAR v2 | 83.43 / 81.45 / 84.47 | 3.02 | 78.0 / 74.1 / 79.9 | yes |

## 3. Limitation of the H0 rule (identified during the run, rule not changed)

The Lumberton floor is 50% of the arm's **own** median Lumberton F1. If an arm fails on
Lumberton in every seed, the median is low and the floor is lower still, so a consistent
failure passes. The same applies to phase 1 (`recipe_stability_v1` §5), whose Japan floor
had the same form.

This happened for arm A and arm D: both are "stable" at 3.8–17.1% Lumberton F1. The rule
therefore measured **consistency across seeds**, not **absence of the failure**. It was first
flagged on 2026-10-08, before seed 203 finished, and the frozen rule was applied unchanged.
Future protocols that use a stability rule should add an absolute floor or a reference arm.
Choosing such a floor now, after seeing these results, would be post hoc.

## 4. Per-checkpoint results (validation, percent)

Columns: event-macro pixel F1; global pixel F1; component precision; component F1;
interior-small recall (n = 370); clipped-small recall (n = 1,551); boundary F1; inner-band
IoU; predicted-positive fraction (truth 1.08%); per-event pixel F1 for Lumberton / Coraki /
Hebei.

### 4.1 Last checkpoint (epoch 15, primary)

| Seed | Arm | Event F1 | Pixel F1 | Comp. P | Comp. F1 | Small int. | Small clip. | Boundary F1 | Inner-band IoU | Pred. pos. | Lumb. / Cor. / Heb. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 201 | A | 58.74 | 53.61 | 36.53 | 39.30 | 16.49 | 7.35 | 61.71 | 25.00 | 1.92 | 8.8 / 93.2 / 74.2 |
| 201 | B | 56.51 | 32.87 | 32.65 | 38.06 | 22.70 | 11.09 | 50.35 | 18.75 | 3.87 | 3.5 / 91.6 / 74.4 |
| 201 | C | 75.45 | 79.04 | 51.39 | 45.47 | 13.78 | 6.71 | 70.55 | 30.79 | 0.87 | 61.5 / 91.8 / 73.1 |
| 201 | D | 59.17 | 62.61 | 30.32 | 35.37 | 18.65 | 9.16 | 54.61 | 21.71 | 1.34 | 17.1 / 92.7 / 67.7 |
| 201 | FC | 62.32 | 65.72 | 13.65 | 21.62 | 27.84 | 20.63 | 58.40 | 32.04 | 1.38 | 19.0 / 94.2 / 73.8 |
| 201 | BIT | 83.43 | 84.55 | 62.71 | 55.32 | 24.86 | 15.60 | 76.50 | 31.12 | 1.19 | 78.0 / 90.5 / 81.8 |
| 202 | A | 59.03 | 35.67 | 28.34 | 35.32 | 23.24 | 11.41 | 46.21 | 17.73 | 3.87 | 3.8 / 92.8 / 80.5 |
| 202 | B | 57.06 | 42.65 | 21.32 | 29.13 | 22.43 | 13.28 | 45.95 | 17.32 | 2.77 | 5.7 / 90.0 / 75.5 |
| 202 | C | 59.44 | 47.20 | 19.00 | 27.41 | 27.84 | 14.06 | 45.57 | 17.34 | 2.64 | 6.8 / 92.2 / 79.3 |
| 202 | D | 62.42 | 66.16 | 31.02 | 37.95 | 26.76 | 15.54 | 60.49 | 24.03 | 1.58 | 16.7 / 91.8 / 78.8 |
| 202 | FC | 83.41 | 82.33 | 34.55 | 38.25 | 12.16 | 15.41 | 73.97 | 42.56 | 0.83 | 82.0 / 95.1 / 73.1 |
| 202 | BIT | 81.45 | 82.76 | 63.92 | 48.73 | 8.92 | 6.38 | 71.61 | 30.40 | 0.95 | 74.1 / 93.5 / 76.7 |
| 203 | A | 59.68 | 47.15 | 15.80 | 23.60 | 20.27 | 9.22 | 43.58 | 17.44 | 2.60 | 6.1 / 93.1 / 79.8 |
| 203 | B | 62.08 | 67.59 | 29.59 | 36.80 | 25.95 | 13.86 | 59.13 | 23.25 | 1.33 | 20.9 / 92.2 / 73.2 |
| 203 | C | 59.91 | 53.37 | 21.62 | 29.36 | 18.92 | 9.99 | 48.03 | 19.58 | 2.08 | 8.3 / 92.4 / 79.0 |
| 203 | D | 62.05 | 63.75 | 30.11 | 37.52 | 29.46 | 14.83 | 58.00 | 23.72 | 1.67 | 13.7 / 92.2 / 80.3 |
| 203 | FC | 56.68 | 46.05 | 10.97 | 17.25 | 12.16 | 10.96 | 42.51 | 20.81 | 2.15 | 7.2 / 95.1 / 67.7 |
| 203 | BIT | 84.47 | 85.18 | 67.70 | 56.16 | 21.08 | 12.38 | 77.65 | 33.47 | 1.11 | 79.9 / 91.7 / 81.8 |

### 4.2 Best checkpoint (`event_pixel_v2`, descriptive only)

| Seed | Arm | Epoch | Event F1 | Pixel F1 | Comp. P | Comp. F1 | Small int. | Small clip. | Boundary F1 | Inner-band IoU | Pred. pos. | Lumb. / Cor. / Heb. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 201 | A | 8 | 58.86 | 42.40 | 17.76 | 25.30 | 19.73 | 7.99 | 44.05 | 16.53 | 3.09 | 4.6 / 92.3 / 79.7 |
| 201 | B | 12 | 56.77 | 33.19 | 25.65 | 32.66 | 21.62 | 11.15 | 50.25 | 18.48 | 4.00 | 3.8 / 89.5 / 77.0 |
| 201 | C | 15 | 75.45 | 79.04 | 51.39 | 45.47 | 13.78 | 6.71 | 70.55 | 30.79 | 0.87 | 61.5 / 91.8 / 73.1 |
| 201 | D | 15 | 59.17 | 62.61 | 30.32 | 35.37 | 18.65 | 9.16 | 54.61 | 21.71 | 1.34 | 17.1 / 92.7 / 67.7 |
| 201 | FC | 11 | 84.00 | 87.40 | 31.23 | 39.53 | 38.11 | 26.24 | 80.66 | 46.80 | 1.08 | 73.4 / 92.4 / 86.2 |
| 201 | BIT | 13 | 84.12 | 84.35 | 62.79 | 55.42 | 24.05 | 15.02 | 76.32 | 31.61 | 1.16 | 81.0 / 90.1 / 81.2 |
| 202 | A | 8 | 60.30 | 53.76 | 23.25 | 31.40 | 28.65 | 13.60 | 48.48 | 17.97 | 2.26 | 8.7 / 92.0 / 80.2 |
| 202 | B | 13 | 72.10 | 78.15 | 40.76 | 44.45 | 29.73 | 15.60 | 71.22 | 27.78 | 1.11 | 50.0 / 90.4 / 75.9 |
| 202 | C | 9 | 66.91 | 74.75 | 28.74 | 34.86 | 18.92 | 8.90 | 63.18 | 25.85 | 1.08 | 33.4 / 93.0 / 74.3 |
| 202 | D | 12 | 76.79 | 79.68 | 46.51 | 46.66 | 22.43 | 13.80 | 71.69 | 28.64 | 0.99 | 64.9 / 91.7 / 73.7 |
| 202 | FC | 13 | 83.59 | 86.16 | 27.12 | 35.05 | 19.46 | 20.25 | 78.54 | 45.86 | 0.97 | 74.0 / 94.3 / 82.4 |
| 202 | BIT | 5 | 83.81 | 85.85 | 66.48 | 56.41 | 29.46 | 13.02 | 77.68 | 31.31 | 1.13 | 76.5 / 91.1 / 83.8 |
| 203 | A | 2 | 64.44 | 72.40 | 46.88 | 41.03 | 11.08 | 5.54 | 53.47 | 14.33 | 1.35 | 33.1 / 85.4 / 74.8 |
| 203 | B | 13 | 64.65 | 71.59 | 29.24 | 36.03 | 27.84 | 13.15 | 62.45 | 23.84 | 1.13 | 32.4 / 91.1 / 70.4 |
| 203 | C | 3 | 65.13 | 73.42 | 38.62 | 38.93 | 12.70 | 7.09 | 57.60 | 18.28 | 1.27 | 29.8 / 88.9 / 76.7 |
| 203 | D | 3 | 69.43 | 77.48 | 36.95 | 39.41 | 22.16 | 11.03 | 60.95 | 18.30 | 1.30 | 41.2 / 88.6 / 78.5 |
| 203 | FC | 7 | 83.97 | 85.94 | 23.19 | 32.23 | 35.14 | 24.50 | 79.65 | 45.10 | 0.98 | 76.3 / 93.3 / 82.3 |
| 203 | BIT | 15 | 84.47 | 85.18 | 67.70 | 56.16 | 21.08 | 12.38 | 77.65 | 33.47 | 1.11 | 79.9 / 91.7 / 81.8 |

Best checkpoints again include early epochs (2, 3, 3, 5) and a large best-vs-last gap for
FC-Siam-Diff (s201 84.00 vs 62.32; s203 83.97 vs 56.68).

## 5. Post-hoc diagnostic: Lumberton under R1 (read-only, training logs)

From each seed's `epochs.csv` (per-epoch validation from `metrics.jsonl`). "Low" means
Lumberton F1 < 20%.

| Arm | Low epochs of 15 (s201 / s202 / s203) | Low epochs in 11–15 | Mean pred. pos. % in epochs 11–15 | Train loss, epoch 15 |
|---|---|---|---|---|
| A | 15 / 15 / 12 | 5 / 5 / 5 | 2.24 / 2.65 / 2.29 | .173 / .171 / .176 |
| B | 15 / 13 / 10 | 5 / 3 / 2 | 4.37 / 1.74 / 1.72 | .169 / .166 / .169 |
| C | 14 / 13 / 8 | 4 / 5 / 5 | 2.80 / 2.09 / 2.43 | .234 / .230 / .237 |
| D | 15 / 10 / 12 | 5 / 3 / 4 | 1.90 / 1.52 / 1.85 | .229 / .223 / .230 |
| FC | 11 / 7 / 9 | 3 / 0 / 4 | 2.01 / 0.90 / 1.64 | .186 / .184 / .184 |
| BIT | 0 / 0 / 0 | 0 / 0 / 0 | 1.20 / 1.15 / 1.18 | .188 / .188 / .187 |

**Observations, descriptive only:**
1. **Under R1, every SmallFlood-CDNet arm fails on Lumberton in most epochs of every seed.**
   Arm A is below 20% in 42 of 45 epochs. Coraki (85–95%) and Hebei (68–86%) stay stable
   across arms and seeds, so the event-macro spread is again almost entirely Lumberton.
2. **The failure is over-prediction.** The SmallFlood-CDNet arms predict 1.5–4.4% positive pixels in
   epochs 11–15 against a 1.08% truth. BIT-SAR v2 stays at 1.15–1.20%.
3. **Training loss is again nearly identical across seeds within each arm.** The failure is
   not a failure to fit the training objective.
4. **BIT-SAR v2 never fell below 20% Lumberton F1 in any of its 45 epochs.** Under the old
   recipe it also never did (seeds 42, 1337, 2026). That is 6 seeds, 90 epochs, without a
   Lumberton collapse.
5. **FC-Siam-Diff alternates.** Its best checkpoints reach 73–76% Lumberton F1 in every seed,
   but its last checkpoint fails in two of three seeds.

## 6. Comparison with the old recipe (descriptive; different seeds, not paired)

Last checkpoint (epoch 15), event-macro F1, percent. Old recipe values are from
`docs/direction_b_confirmation_results_20261006.md` §4 (learning rate 3e-4 for A–D and FC;
BIT-SAR v2 already 1e-4).

| Arm | Old recipe, seeds 42 / 1337 / 2026 | R1, seeds 201 / 202 / 203 |
|---|---|---|
| A | 82.57 / 80.20 / 57.68 | 58.74 / 59.03 / 59.68 |
| B | 80.56 / 80.29 / 54.88 | 56.51 / 57.06 / 62.08 |
| C | 76.21 / 73.50 / 56.94 | 75.45 / 59.44 / 59.91 |
| D | 80.20 / 80.70 / 57.87 | 59.17 / 62.42 / 62.05 |
| FC | 85.17 / 66.97 / 62.85 | 62.32 / 83.41 / 56.68 |
| BIT | 85.00 / 78.91 / 79.71 | 83.43 / 81.45 / 84.47 |

Under the old recipe, arm A collapsed on Lumberton in one of three seeds (2026). Under R1 it
collapsed in all three. **Lowering the learning rate made the CNN arms consistently worse on
the validation events, not more stable at a good level.** Seeds differ between the two
columns, so this is an observation, not a paired effect. It also contradicts the phase-1
internal-dev result, where R1 gave the best and most balanced arm A. The phase-1 dev failure
mode (Canada under-prediction) differed from this one (Lumberton over-prediction), which
`docs/recipe_stability_results_20261006.md` §4 flagged as a risk.

**What this does not establish.** The cause of the Lumberton over-prediction is still
unknown. BIT-SAR v2's robustness cannot be attributed to its architecture: it also differs in
initialization (normal 0.02, BN reset), loss coefficients (0.5 + 0.5) and output adapter.
Nothing here licenses re-thresholding, dropping Lumberton, per-event reweighting, more
recipes, more seeds or test-set use.

## 7. Deviations and implementation notes

1. §11 asks for per-run per-epoch tables in `artifacts/recipe_confirmation_v1/<UTC>/`. They
   were written by the reviewer to `artifacts/recipe_confirmation_review/<UTC>/s*/epochs.csv`
   instead. Raw `metrics.jsonl` stay in each run directory.
2. The FC-Siam-Diff 1-epoch preflight trains on the full training set (about 5 minutes),
   because `next_steps.pilot` has no subset mode. It is never promoted to a run.
3. `decide` stops on any undefined metric a rule needs (for example a null Lumberton F1)
   rather than substituting zero. This did not trigger.
4. Results remain on the server; no run or review tree was copied into this checkout.

## 8. Conclusions permitted by this protocol

- Under recipe R1 and seeds 201–203, the confirmation rules neither support nor contradict:
  - "size weighting does not improve interior-small recall";
  - "the boundary package gives no persistent inner-band IoU gain";
  - "FC-Siam-Diff is not worse than the heavier arms".
- R1 makes arm A **consistent** across seeds on the validation events, at about 59%
  event-macro F1 with Lumberton F1 below 10%. It does **not** remove the Lumberton failure.
- Across two recipes and six seeds, factorial contrasts and whole-method rankings for the
  CNN arms are not resolvable against seed variation concentrated on one low-flood-fraction
  validation event. BIT-SAR v2 is the only arm that avoided that failure throughout.

**Not permitted:**
- reporting H0 "stable" as evidence that the training recipe is solved;
- p-values or confidence intervals from pixels, patches or components;
- choosing a seed, checkpoint, threshold or event weighting after the fact;
- any test-set use.

## 9. Next decisions (require user approval)

No follow-up run is triggered. The open choice is the paper's framing. Candidates, without
commitment:
- a controlled negative result plus a seed- and event-sensitivity analysis of lightweight SAR
  flood change detectors (the evidence above supports this most directly);
- a bounded study of why BIT-SAR v2 resists the Lumberton failure, which would need a new
  frozen protocol separating architecture, initialization and loss coefficients.

Test stays sealed.
