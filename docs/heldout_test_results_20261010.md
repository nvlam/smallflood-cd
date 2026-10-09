# heldout_test_v1 (A4): held-out test results and closure

## Material Passport

Protocol: `docs/heldout_test_protocol_v1_20261009.md` (commit `07dbf77`). Evaluator, tests
and launchers: commit `362f6a5`, copied to the server by sha256-verified file transfer.
Execution: the single test evaluation, `artifacts/heldout_test_v1/20261009T153336Z`
(`RESULTS.json`, `denominators.json`, per-seed reports), logs
`artifacts/heldout_test_v1_logs/20261009T153336Z`. Test access approved by the user on
2026-10-09 ("Duyệt mở test"). Started 2026-10-09T15:34Z, completed 18:18Z (about 2.7 h).
Validation-only engineering regression: `artifacts/heldout_test_regression/20261009T145926Z`.
Status: **closed. The test set has now been used once.** Any further test evaluation needs a
new protocol and a stated reason (§11.6 of the protocol). Section 5 is post-hoc description.

## 1. Execution and integrity

- Before test access:
  - 14 synthetic tests passed (CPU locally and on the server);
  - validation-mode regression: the new evaluator reproduced the existing validation
    `report.json` metrics, `events.csv` and object/boundary reports **exactly** for A and
    BIT-SAR v2 at seed 42 and A and FC-Siam-Diff at seed 201.
- `COMPLETE.json`: 72 evaluations; `split: test`; `training_performed: false`;
  `checkpoints_match_validation_records: true`; `denominators_consistent: true`. Evaluator
  SHA256 `3d7d271c…f118a`. Every checkpoint's SHA256 matched its validation review record
  before loading, and every last checkpoint was epoch 15.
- No interruption; `--resume` was not used. No traceback.
- **Test denominators** (identical across all 72 evaluations):

| | Patches | Valid pixels | Positive pixels (%) | GT components | Interior-small | Clipped-small |
|---|---:|---:|---:|---:|---:|---:|
| Weihui | 2,498 | 153,279,548 | 1,576,881 (1.03%) | 4,433 | 360 | 1,033 |
| NovaKakhovka | 2,896 | 181,002,691 | 3,368,972 (1.86%) | 20,073 | 4,379 | 7,904 |
| Jubba | 3,810 | 234,308,120 | 1,994,782 (0.85%) | 2,669 | 160 | 761 |
| **Total** | 9,204 | 568,590,359 | 6,940,635 (1.22%) | 27,175 | 4,899 | 9,698 |

NovaKakhovka holds 89% of the interior-small components, so test interior-small recall is
dominated by that event.

## 2. T1: replication of the validation rules (last checkpoint, pp)

| Rule | Recipe | Seed 1 | Seed 2 | Seed 3 | Outcome |
|---|---|---|---|---|---|
| Size weighting (B−A; D−C, interior-small recall) | old (42/1337/2026) | −0.35; +0.12 | −0.12; −0.12 | +1.20; +2.31 | **Inconsistent** |
| | R1 (201/202/203) | +1.71; +0.88 | +0.43; −0.08 | +1.18; +1.76 | **Inconsistent** |
| Boundary package (C−A; D−B, inner-band IoU) | old | +0.41; +0.03 | +1.62; +2.19 | −1.09; +0.17 | **Inconsistent** |
| | R1 | −0.33; +0.28 | +0.28; +0.54 | −0.42; +0.91 | **Inconsistent** |
| FC ≥ A and FC ≥ BIT-SAR v2 (event-macro F1) | old | +13.01; −2.15 | +6.52; −3.95 | +7.53; −2.50 | **Contradicted** |
| | R1 | +14.14; −8.15 | +1.39; −3.83 | +0.97; −14.25 | **Contradicted** |

"Contradicted" here comes entirely from FC < BIT-SAR v2 in every seed. FC-Siam-Diff exceeds
arm A in all six seeds. The size-weighting and boundary contrasts are all within about
±2.3 pp, at interior-small recall of 2–7%.

## 3. T2: seed sensitivity (last checkpoint)

**Rank stability:** 10 of 15 arm pairs keep their event-macro F1 order in all three seeds under
the old recipe; 9 of 15 under R1.
- Stable under both recipes: A–FC, A–BIT, B–FC, B–BIT, C–BIT, D–FC, D–BIT, FC–BIT.
- Also stable under old only: A–B, C–FC. Under R1 only: B–D.
- The order **between** groups (SmallFlood-CDNet arms < FC-Siam-Diff < BIT-SAR v2) is stable
  except C–FC under R1 (C s202 65.80 > FC s202 64.81). The order **among** A–D is mostly not.

**Per-run values, event-macro F1 and per-event F1 (%):**

| Recipe | Arm | Seeds | Event-macro F1 | Range | Weihui | NovaKakhovka | Jubba |
|---|---|---|---|---:|---|---|---|
| old | A | 42/1337/2026 | 60.31 / 64.40 / 61.32 | 4.09 | 80.8 / 81.7 / 79.0 | 15.6 / 25.8 / 18.0 | 84.5 / 85.6 / 87.1 |
| old | B | | 59.16 / 57.96 / 48.83 | 10.33 | 73.7 / 75.6 / 48.6 | 21.7 / 17.7 / 18.8 | 82.1 / 80.6 / 79.1 |
| old | C | | 59.12 / 68.74 / 58.72 | 10.01 | 77.4 / 81.4 / 74.3 | 21.8 / 38.3 / 18.6 | 78.2 / 86.5 / 83.2 |
| old | D | | 57.31 / 67.76 / 58.87 | 10.45 | 74.3 / 80.9 / 77.4 | 20.0 / 35.0 / 16.1 | 77.6 / 87.4 / 83.1 |
| old | FC | | 73.32 / 70.92 / 68.86 | 4.46 | 82.7 / 86.7 / 84.7 | 49.2 / 33.6 / 29.6 | 88.0 / 92.5 / 92.3 |
| old | BIT | | 75.47 / 74.87 / 71.36 | 4.11 | 86.4 / 83.2 / 83.4 | 51.3 / 54.1 / 42.7 | 88.7 / 87.3 / 88.0 |
| R1 | A | 201/202/203 | 54.84 / 63.42 / 61.60 | 8.57 | 76.6 / 81.7 / 80.8 | 10.0 / 19.9 / 17.4 | 77.9 / 88.7 / 86.7 |
| R1 | B | | 55.65 / 61.89 / 57.39 | 6.23 | 76.7 / 78.4 / 73.0 | 15.5 / 20.3 / 14.6 | 74.8 / 87.0 / 84.6 |
| R1 | C | | 53.95 / 65.80 / 59.54 | 11.86 | 75.9 / 83.4 / 80.7 | 18.5 / 26.3 / 14.6 | 67.4 / 87.7 / 83.3 |
| R1 | D | | 57.51 / 62.43 / 58.36 | 4.92 | 74.7 / 80.4 / 81.2 | 21.7 / 23.9 / 18.9 | 76.2 / 83.1 / 75.0 |
| R1 | FC | | 68.98 / 64.81 / 62.57 | 6.42 | 74.4 / 78.5 / 73.7 | 42.5 / 30.0 / 24.8 | 90.1 / 85.9 / 89.3 |
| R1 | BIT | | 77.14 / 68.64 / 76.82 | 8.49 | 86.1 / 81.0 / 85.2 | 57.2 / 37.4 / 55.2 | 88.2 / 87.6 / 90.0 |

## 4. T3: event sensitivity

- Low-fraction test event (from the denominators): **Jubba**, 0.85% positive.
- Groups where the low-fraction event has the largest across-seed per-event F1 range:
  **4 of 12** (R1: A, B, C, D). NovaKakhovka has the largest range in the other 8 (all six old
  arms, R1 FC and R1 BIT-SAR v2).
- **Outcome: does not generalize** (cut point ≤ 4 of 12).

The validation finding "seed variation concentrates in the lowest-flood-fraction event" is
therefore **not supported** on test under the pre-specified rule.

## 5. Post-hoc description (not pre-specified)

1. **Seed variation on test still concentrates in one event, but not a low-fraction one.**
   NovaKakhovka has the highest flood fraction (1.86%) and almost all small components.
   All 14 runs with any test event below 20% F1 have that event as NovaKakhovka, and all are
   SmallFlood-CDNet arms (old: A s42, B s1337, A/B/C/D s2026; R1: A/B/C s201, A s202,
   A/B/C/D s203). No FC-Siam-Diff or BIT-SAR v2 run falls below 20% on any test event.
2. In 12 of those 14 runs the model predicted more positive pixels on NovaKakhovka than the
   1.86% truth (1.87–3.65%), the same over-prediction direction seen on Lumberton. This is
   description only; the mechanism is not established.
3. **Every model misses almost all interior small floods on test:** interior-small recall is
   2.4–6.7% (seed means, last checkpoint) over 4,899 components. Size weighting does not
   change this.
4. **Old recipe vs R1 on test (different seeds, not paired):** arm A 62.0 vs 60.0 mean
   event-macro F1; FC-Siam-Diff 71.0 vs 65.5; BIT-SAR v2 73.9 vs 74.2. R1 did not help on test
   either.

## 6. Benchmark tables (for direction C)

Mean over three seeds (range in brackets), percent. Parameters: A/B 1,079,865; C/D 1,080,805;
FC-Siam-Diff 487,857; BIT-SAR v2 3,492,642. Costs other than parameters belong to the A3
protocol.

### 6.1 Last checkpoint (primary)

| Recipe | Arm | Event-macro F1 | Pixel F1 | Component F1 | Interior-small recall | Boundary F1 | Inner-band IoU |
|---|---|---|---|---|---|---|---|
| old | A | 62.01 (4.09) | 49.93 (7.84) | 19.28 (1.92) | 2.99 (1.02) | 40.10 (2.33) | 15.46 (1.13) |
| old | B | 55.32 (10.33) | 46.29 (6.25) | 19.13 (0.97) | 3.23 (1.08) | 37.43 (1.19) | 14.28 (0.33) |
| old | C | 62.19 (10.01) | 55.58 (13.93) | 18.86 (4.83) | 2.67 (1.76) | 39.97 (9.75) | 15.78 (3.31) |
| old | D | 61.31 (10.45) | 52.77 (13.57) | 19.20 (1.87) | 3.44 (1.57) | 39.22 (6.73) | 15.07 (2.45) |
| old | FC | 71.03 (4.46) | 61.09 (15.31) | 19.18 (4.10) | 6.72 (0.71) | 50.19 (5.80) | 26.25 (4.47) |
| old | BIT | 73.90 (4.11) | 69.96 (4.16) | 27.14 (4.88) | 5.12 (2.94) | 52.12 (5.98) | 19.83 (1.21) |
| R1 | A | 59.95 (8.57) | 45.55 (6.00) | 17.87 (2.69) | 2.37 (1.12) | 37.94 (7.19) | 14.28 (3.09) |
| R1 | B | 58.31 (6.23) | 44.07 (6.76) | 18.88 (0.85) | 3.48 (0.18) | 38.13 (2.29) | 13.92 (0.94) |
| R1 | C | 59.76 (11.86) | 47.70 (10.98) | 18.02 (4.29) | 2.69 (2.78) | 37.59 (10.05) | 14.13 (3.70) |
| R1 | D | 59.43 (4.92) | 48.09 (3.80) | 19.06 (1.93) | 3.54 (1.82) | 38.74 (2.91) | 14.50 (1.20) |
| R1 | FC | 65.45 (6.42) | 59.52 (11.75) | 15.92 (4.38) | 4.70 (4.39) | 44.66 (7.59) | 23.06 (4.42) |
| R1 | BIT | 74.20 (8.49) | 70.45 (9.19) | 25.56 (10.20) | 4.03 (4.08) | 51.41 (13.63) | 19.51 (3.67) |

### 6.2 Best checkpoint (`event_pixel_v2`, descriptive only)

| Recipe | Arm | Event-macro F1 | Pixel F1 | Component F1 | Interior-small recall | Boundary F1 | Inner-band IoU |
|---|---|---|---|---|---|---|---|
| old | A | 66.42 (4.29) | 56.69 (5.53) | 20.75 (1.69) | 4.30 (1.65) | 43.37 (3.99) | 15.94 (1.61) |
| old | B | 67.40 (9.47) | 60.56 (12.66) | 21.03 (3.53) | 4.13 (3.94) | 43.58 (9.39) | 16.23 (3.50) |
| old | C | 67.79 (9.98) | 62.70 (10.63) | 21.44 (4.61) | 3.84 (2.94) | 44.97 (9.00) | 17.05 (2.52) |
| old | D | 63.34 (5.71) | 55.50 (14.88) | 19.97 (1.78) | 3.50 (1.31) | 40.68 (7.41) | 14.54 (6.20) |
| old | FC | 71.94 (7.09) | 65.56 (5.34) | 18.74 (2.65) | 6.77 (1.59) | 49.25 (3.15) | 25.66 (0.62) |
| old | BIT | 75.18 (3.90) | 71.10 (3.89) | 27.71 (6.31) | 4.99 (2.51) | 52.73 (7.73) | 20.01 (2.73) |
| R1 | A | 61.42 (4.42) | 47.98 (9.74) | 17.84 (4.58) | 2.54 (1.69) | 36.60 (11.61) | 12.19 (6.89) |
| R1 | B | 55.63 (7.94) | 43.23 (9.36) | 18.51 (3.04) | 3.45 (0.47) | 36.80 (6.42) | 13.11 (2.62) |
| R1 | C | 59.00 (8.76) | 48.30 (11.72) | 16.72 (1.24) | 1.88 (0.37) | 34.56 (2.34) | 12.05 (2.21) |
| R1 | D | 61.10 (5.57) | 50.11 (3.37) | 18.71 (1.90) | 3.01 (0.86) | 37.80 (4.85) | 13.04 (4.38) |
| R1 | FC | 73.37 (3.36) | 68.17 (3.51) | 17.20 (3.89) | 7.06 (1.06) | 49.85 (3.47) | 24.75 (2.50) |
| R1 | BIT | 75.67 (2.24) | 72.11 (2.45) | 27.62 (3.11) | 5.14 (2.20) | 53.36 (3.51) | 20.13 (0.85) |

## 7. Implementation note and deviations

1. The protocol says the evaluator "reuses `evaluate()` unchanged, with the split as a
   parameter". The original has no split parameter, so `evaluate_split` is built from the
   source of `review_pilot_validation.evaluate` with four checked edits (function signature,
   dataset split, the validation-count check applied only to validation, and the report's
   `split`/`test_used` fields). A unit test confirms only those lines differ, and the
   validation-mode regression reproduced existing reports exactly. Metric code is unchanged.
2. The report field `validation_patches` keeps its historical name in test reports; it holds
   the test patch count (9,204).
3. Test reports include the evaluator's usual preview images (first two positive and negative
   patches per event). They were not looked at for any analysis.
4. Results remain on the server; no report tree was copied into this checkout.

## 8. Conclusions permitted

- On the held-out test set, under both recipes, size weighting and the boundary package show
  no consistent effect (T1 inconsistent), matching validation.
- BIT-SAR v2 ranks above FC-Siam-Diff, and FC-Siam-Diff above arm A, in every seed of both
  recipes. The ranking among A–D is not stable across seeds.
- Seed variation on test concentrates in one event (NovaKakhovka), but the pre-specified
  "lowest flood fraction" generalization is **not** supported (T3).
- All models recall only 2–7% of interior small floods on test.

**Not permitted:** p-values or confidence intervals from pixels, patches or components;
re-evaluation with other thresholds, checkpoints, seeds or event subsets; claiming that
BIT-SAR v2's advantage comes from its architecture; any further test evaluation without a new
protocol.

## 9. Next steps (require user approval)

- A3 efficiency protocol (cost set for the benchmark).
- Paper outline for direction A + C, revising the "low-flood-fraction event" framing in light
  of T3: the paper should describe single-event concentration of seed variation without
  attributing it to flood fraction.
