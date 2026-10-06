# direction_b_confirmation_v1: results and closure

## Material Passport

Protocol: `docs/direction_b_confirmation_protocol_v1_20261003.md` (branch
`direction-b-confirmation`, commit `6a0c3fa`).
Execution: server run `runs/direction_b_confirmation_v1/20261003T135430Z`, reviews
`artifacts/direction_b_confirmation_review/20261003T135430Z/{s1337,s2026}`, logs
`artifacts/direction_b_confirmation/20261003T135430Z`. Started 2026-10-03T13:54Z, completed
2026-10-05T09:45Z.
Scope: validation only. The held-out test set stayed sealed. No protocol deviation.
Status: **closed.** All three §7 decision rules return **inconsistent**. Per protocol this
triggers no further runs. Sections 6–7 are post-hoc descriptive diagnostics, not
preregistered analyses.

## 1. Execution and integrity

- Pre-launch checks:
  - Synthetic tests on the server: 14 + 16 passed.
  - The seed-42 regression reproduced the recorded factorial anchors exactly: initial
    state A/B `71c04062…54b8`, C/D `db03a4cf…e476`, common `71c04062…54b8`, and all 15
    batch-order hashes. Recorded in `regress_s42/REGRESSION_PASS.json`.
- Every run passed its gate before training:
  - factorial preflight for seeds 1337 and 2026;
  - BIT-SAR v2 batch-8 gates.
- All 12 runs completed 15 epochs and 35,940 optimizer steps.
- Factorial pairing held within each seed: common initial state and identical batch order
  across A–D.
- Both review `COMPLETE.json` files record:
  - `evaluations`: 12;
  - `test_used`: false;
  - `training_performed`: false;
  - `pixel_counts_match_training`: true;
  - `denominators_match_frozen`: true (4,767 patches; 6,745 components; 1,551 clipped-small;
    370 interior-small).
- Reviewer SHA256 `1e11875f…0637`.
- **Engineering note.** The server was heavily shared (load average about 28, 16 users,
  another process using the GPU). Epochs took 1–6× longer than the seed-42 runs. Strict
  determinism makes results independent of load. Elapsed times are not efficiency evidence.

## 2. Decision rules (§7, confirmation seeds only)

Percentage points.

| Rule | Seed 1337 | Seed 2026 | Outcome |
|---|---|---|---|
| H1 size weighting: B−A ≤ 0 and D−C ≤ 0 at best, interior-small recall | B−A +1.62; D−C +6.76 | B−A +7.03; D−C −10.00 | **Inconsistent** |
| H1 boundary package: C−A ≤ 0 and D−B ≤ 0 at last, inner-band IoU | C−A −1.12; D−B +2.08 | C−A −1.77; D−B −3.19 | **Inconsistent** |
| H3: FC ≥ A and FC ≥ BIT at best, event-macro F1 | FC−A +2.65; FC−BIT +2.84 | FC−A +5.50; FC−BIT −5.60 | **Inconsistent** |

**Discovery seed 42 (not used for decisions):**
- B−A −13.24, D−C +3.51.
- At last: C−A −0.31, D−B −1.33.
- FC−A +2.68, FC−BIT +0.52.

The size-weighting contrast B−A changes sign across the three seeds: −13.24, +1.62, +7.03.

## 3. Best checkpoints (validation, percent)

| Seed | Arm | Epoch | Event F1 | Pixel F1 | Comp. P | Comp. F1 | Small interior | Small clipped | Boundary F1 | Inner-band IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | A | 12 | 82.84 | 84.64 | 57.93 | 52.85 | 34.05 | 15.22 | 76.92 | 30.46 |
| 42 | B | 13 | 82.82 | 83.56 | 63.95 | 53.82 | 20.81 | 12.57 | 75.70 | 31.92 |
| 42 | C | 12 | 82.95 | 83.97 | 62.71 | 53.87 | 24.86 | 12.64 | 77.07 | 33.24 |
| 42 | D | 14 | 82.64 | 83.17 | 61.43 | 54.15 | 28.38 | 15.93 | 76.64 | 33.17 |
| 42 | FC | 12 | 85.52 | 85.47 | 43.53 | 47.63 | 30.54 | 21.41 | 78.49 | 47.80 |
| 42 | BIT | 15 | 85.00 | 86.30 | 67.15 | 57.88 | 29.19 | 15.86 | 78.95 | 34.02 |
| 1337 | A | 14 | 83.95 | 84.84 | 61.25 | 54.98 | 26.22 | 14.64 | 78.72 | 33.57 |
| 1337 | B | 14 | 84.15 | 85.26 | 64.32 | 55.05 | 27.84 | 17.15 | 77.26 | 31.47 |
| 1337 | C | 14 | 83.45 | 83.75 | 65.82 | 55.41 | 25.41 | 12.64 | 77.62 | 32.46 |
| 1337 | D | 14 | 83.09 | 83.23 | 63.20 | 54.76 | 32.16 | 17.09 | 76.91 | 31.61 |
| 1337 | FC | 9 | 86.60 | 87.98 | 37.59 | 44.59 | 36.22 | 20.95 | 81.13 | 49.08 |
| 1337 | BIT | 11 | 83.76 | 85.10 | 68.10 | 55.08 | 18.38 | 9.80 | 74.78 | 30.43 |
| 2026 | A | 11 | 72.98 | 80.87 | 30.94 | 37.81 | 34.59 | 15.28 | 70.25 | 27.65 |
| 2026 | B | 11 | 76.94 | 82.58 | 46.74 | 48.42 | 41.62 | 20.57 | 74.84 | 29.23 |
| 2026 | C | 11 | 60.66 | 52.11 | 27.98 | 35.53 | 36.49 | 16.76 | 53.76 | 19.88 |
| 2026 | D | 1 | 68.94 | 76.11 | 41.17 | 40.85 | 26.49 | 10.38 | 58.86 | 16.64 |
| 2026 | FC | 14 | 78.48 | 84.39 | 25.50 | 34.19 | 24.59 | 21.66 | 75.01 | 45.07 |
| 2026 | BIT | 12 | 84.08 | 86.09 | 68.47 | 57.32 | 23.51 | 13.67 | 78.11 | 34.46 |

## 4. Last checkpoints (epoch 15, validation, percent)

| Seed | Arm | Event F1 | Pixel F1 | Comp. P | Comp. F1 | Small interior | Small clipped | Boundary F1 | Inner-band IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | A | 82.57 | 83.67 | 61.06 | 53.20 | 22.97 | 10.96 | 76.22 | 32.79 |
| 42 | B | 80.56 | 81.34 | 56.23 | 51.26 | 20.54 | 12.70 | 73.88 | 32.05 |
| 42 | C | 76.21 | 80.72 | 43.63 | 45.32 | 19.46 | 10.77 | 73.07 | 32.49 |
| 42 | D | 80.20 | 80.13 | 62.78 | 53.33 | 22.97 | 13.54 | 74.48 | 30.71 |
| 42 | FC | 85.17 | 84.36 | 40.96 | 45.21 | 23.78 | 20.12 | 77.46 | 47.36 |
| 42 | BIT | 85.00 | 86.30 | 67.15 | 57.88 | 29.19 | 15.86 | 78.95 | 34.02 |
| 1337 | A | 80.20 | 82.54 | 53.93 | 50.31 | 21.62 | 12.31 | 74.93 | 31.76 |
| 1337 | B | 80.29 | 79.91 | 59.69 | 51.89 | 21.08 | 16.12 | 72.32 | 29.62 |
| 1337 | C | 73.50 | 81.11 | 33.49 | 39.05 | 22.16 | 11.73 | 71.74 | 30.65 |
| 1337 | D | 80.70 | 83.65 | 59.64 | 51.71 | 26.49 | 15.15 | 75.52 | 31.70 |
| 1337 | FC | 66.97 | 73.81 | 8.05 | 13.95 | 25.41 | 21.73 | 53.15 | 31.11 |
| 1337 | BIT | 78.91 | 78.99 | 51.33 | 50.02 | 27.30 | 15.86 | 70.33 | 26.69 |
| 2026 | A | 57.68 | 31.50 | 25.75 | 32.97 | 23.24 | 10.44 | 48.12 | 19.44 |
| 2026 | B | 54.88 | 42.43 | 25.48 | 33.79 | 28.92 | 15.41 | 49.94 | 20.43 |
| 2026 | C | 56.94 | 35.63 | 22.40 | 29.67 | 17.84 | 9.35 | 42.92 | 17.67 |
| 2026 | D | 57.87 | 36.64 | 24.90 | 33.18 | 30.00 | 15.47 | 43.28 | 17.24 |
| 2026 | FC | 62.85 | 62.05 | 11.06 | 18.14 | 21.62 | 20.25 | 44.94 | 24.27 |
| 2026 | BIT | 79.71 | 83.20 | 61.60 | 53.11 | 21.08 | 12.44 | 75.12 | 32.76 |

## 5. Seed sensitivity

- **Seed-to-seed spread within one arm can exceed the effects the factorial was meant to
  measure.**
  - Interior-small recall at best (seeds 42 / 1337 / 2026):
    - A: 34.05 / 26.22 / 34.59.
    - B: 20.81 / 27.84 / 41.62.
    - D: 28.38 / 32.16 / 26.49.
  - One interior-small match is 0.27 pp (n = 370; Coraki 42, Hebei 328, Lumberton 0).
- **BIT-SAR v2 was the only arm without a seed with very low event F1:**
  - best 85.00 / 83.76 / 84.08;
  - last 85.00 / 78.91 / 79.71.
- **This cannot be attributed to architecture.** BIT-SAR v2 also differs in learning rate
  (1e-4 vs 3e-4), initialization, loss coefficients and runner (protocol §4).

## 6. Post-hoc diagnostic: where the instability is (read-only, logs only)

Per-epoch validation logs (`metrics.jsonl`) were tabulated for all 18 runs (seeds 42, 1337,
2026). Means are over epochs 11–15.

| Run | Final train loss | Event F1 | Lumberton F1 | Coraki F1 | Hebei F1 | Pred. positive % | Epochs with Lumberton F1 < 20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A s42 / s1337 / s2026 | .172 / .167 / .172 | 82.1 / 82.0 / 60.0 | 76 / 76 / 13 | 92 / 93 / 93 | 79 / 78 / 74 | 0.98 / 0.98 / 2.93 | 1 / 6 / 13 |
| B | .167 / .161 / .167 | 81.4 / 80.7 / 61.1 | 77 / 74 / 17 | 91 / 92 / 91 | 76 / 76 / 75 | 0.93 / 0.96 / 2.34 | 0 / 5 / 12 |
| C | .234 / .229 / .230 | 73.8 / 79.4 / 57.7 | 52 / 68 / 5 | 91 / 93 / 93 | 79 / 78 / 75 | 1.16 / 0.99 / 3.03 | 4 / 5 / 14 |
| D | .226 / .222 / .224 | 80.3 / 81.6 / 58.6 | 77 / 76 / 6 | 91 / 92 / 92 | 74 / 76 / 78 | 0.92 / 0.97 / 3.39 | 1 / 6 / 13 |
| FC | .185 / .186 / .190 | 79.6 / 71.8 / 63.8 | 67 / 49 / 18 | 95 / 95 / 95 | 77 / 72 / 78 | 1.12 / 0.99 / 2.17 | 2 / 4 / 11 |
| BIT | .186 / .187 / .187 | 83.4 / 81.8 / 81.8 | 79 / 75 / 73 | 91 / 91 / 92 | 81 / 79 / 81 | 1.19 / 1.15 / 1.12 | 0 / 0 / 0 |

Validation GT positive fraction is 1.08% (Lumberton about 0.44%).

**Observations, descriptive only:**
1. **Training loss converges to nearly the same value across seeds within each arm.** The
   unstable runs are not failing to fit the training objective.
2. **Coraki and Hebei are stable across seeds and arms.** The event-macro instability is
   almost entirely **Lumberton**.
   - In seed 2026, the CNN arms predict 2–3.4% positive pixels against a 1.08% truth.
   - Lumberton F1 then falls to about 5–18%, because Lumberton has a very low flood
     fraction and F1 there is highly sensitive to false positives.
   - Lumberton carries one third of the event-macro score, which drives checkpoint
     selection. A single Lumberton spike can therefore select the "best" checkpoint, as with
     D s2026 at epoch 1.
3. **BIT-SAR v2 never collapsed on Lumberton** in any epoch of any seed.
4. **A checked and not-supported idea.** Across the 36 reviewed checkpoints, predicted
   positive fraction does not correlate with interior-small recall (r = 0.02). It does
   correlate negatively with component precision (r = −0.59). Interior-small differences
   here are therefore not simply explained by over-prediction. These correlations are
   descriptive and pool dependent checkpoints.

**What this does not establish.** The cause of the Lumberton false-positive regime is
unknown. Candidates include learning rate or schedule, event domain shift, the fixed 0.5
threshold, label characteristics, and selection on a three-event macro. None of these was
tested. Nothing here says Lumberton should be removed, reweighted or re-thresholded. Doing
any of that now would be a post-hoc protocol change.

## 7. Conclusions permitted by this protocol

- The confirmation seeds neither support nor contradict:
  - "size weighting does not improve interior-small recall";
  - "the boundary package gives no persistent inner-band IoU gain";
  - "FC-Siam-Diff is not worse than the heavier arms".
- Under the frozen 15-epoch recipe, factorial contrasts and whole-method rankings are **not
  resolvable against seed-to-seed variation**. That variation is concentrated in
  false-positive behaviour on one validation event.
- Seed-42 conclusions, including the earlier factorial audit, must not be cited as robust
  effects.

**Not permitted:**
- p-values;
- dropping seed 2026;
- choosing a seed or checkpoint after the fact;
- re-thresholding;
- per-event reweighting;
- any test-set use.

## 8. Next decisions (require user approval)

A follow-up must be a new frozen protocol. Candidates, without commitment:
- a training-recipe stability study tuned on an internal split of the training events,
  not the current validation events;
- a seed-sensitivity reporting design.

Test stays sealed.
