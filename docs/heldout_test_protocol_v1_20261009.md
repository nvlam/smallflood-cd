# Held-out test protocol v1 (A4): single pre-specified test evaluation

Protocol ID: `heldout_test_v1`.
Drafted 2026-10-09 for paper direction A + C (`docs/paper_direction_decision_20261009.md`).
The user approved on 2026-10-09 the scope "all 36 runs, last checkpoint primary, best
descriptive" (proposals 1–2 of that day).
Approved 2026-10-09 (user reply "duyệt nguyên 6 lựa chọn, commit protocol"). All six §11
choices were approved as drafted: the 72 checkpoints of §3, T1 as descriptive replication per
recipe group, T3 with the 7-of-12 / 4-of-12 cut points, the validation-hash identity check,
resume-only interruption, and single use.
**Status: protocol approved and frozen; engineering and the test run NOT yet approved.** No
code, regression or test access may start until the user separately approves them (§9).

## 1. Purpose and scope

The held-out test set has never been opened. This protocol evaluates it **once**, on
checkpoints that already exist, with every table, rule and descriptor fixed in advance.

It answers, for the paper:
- **T1 (replication):** do the validation-only rules for size weighting, the boundary package
  and FC-Siam-Diff give the same kind of outcome on test?
- **T2 (seed sensitivity):** how large is seed-to-seed variation on test, and do method
  rankings stay the same across seeds?
- **T3 (event sensitivity):** is seed variation on test also concentrated in the test event
  with the lowest flood fraction, as Lumberton was on validation?
- **Benchmark accuracy** for the lightweight comparison (C), alongside the separate A3 cost
  protocol.

**Not covered:** training of any kind, threshold or calibration changes, checkpoint or seed
selection, new models, H2 (FU/FO), scene reconstruction, efficiency (A3) and any second test
evaluation.

Results are patch-scoped, event-held-out test observations of a fixed set of checkpoints. They
are not significance tests.

## 2. Frozen inputs

- Split `data/splits/split_v1.yaml`, SHA256
  `88d8c3a7d1fd652b870c0dc8c97348eabd5b1b70a71f052568a11e7213431472`, policy
  `event_holdout_v1`.
- **Test events:** `20210727_Weihui`, `20230609_NovaKakhovka`, `20231201_Jubba` (Jubba tracks
  grouped as one event). Only these names were read on 2026-10-09; no test array, label or
  label statistic has been opened.
- Manifest, component statistics and normalization hashes as in
  `docs/recipe_confirmation_protocol_v1_20261008.md` §2. Normalization stays fitted on the
  training events only.
- Server stack as recorded; nothing installed or upgraded.
- **Test denominators are unknown** and are not estimated beforehand. They are recorded by the
  first evaluation and must then be identical across all 72 evaluations (patches, valid
  pixels, positive pixels, GT components, clipped-small and interior-small counts, per event).

## 3. Checkpoints (fixed list, no additions)

Six arms × two recipes × three seeds = **36 runs**. Each run contributes its `last.ckpt`
(epoch 15, primary) and `best_composite.ckpt` (`event_pixel_v2`, descriptive): **72
evaluations**.

| Recipe | Seeds | Factorial A/B/C/D | FC-Siam-Diff | BIT-SAR v2 |
|---|---|---|---|---|
| Old (A–D and FC at 3e-4; BIT-SAR v2 at 1e-4) | 42 | `runs/proposed_factorial_v1/20260927T075837Z/{A,B,C,D}_proposed` | `runs/pilot15/20260924T032911Z/20260924T051127718020Z_fc_siam_diff` | `runs/bit_sar_v2_pilot15/20260925T015023Z/pilot/20260925T015034589216Z_bit_sar_v2` |
| Old | 1337, 2026 | `runs/direction_b_confirmation_v1/20261003T135430Z/factorial_s{S}/{A,B,C,D}_proposed` | `…/pilots_s{S}/*_fc_siam_diff` | `…/pilots_s{S}/*_bit_sar_v2` |
| R1 (all at 1e-4) | 201, 202, 203 | `runs/recipe_confirmation_v1/20261007T234436Z/factorial_s{S}/{A,B,C,D}_proposed` | `…/pilots_s{S}/*_fc_siam_diff` | `…/pilots_s{S}/*_bit_sar_v2` |

**Checkpoint identity.** Before loading, each checkpoint's SHA256 must equal the SHA256 in its
existing validation review report (seed 42: the proposed-factorial, pilot15 and BIT-SAR v2
validation reviews; 1337/2026: `artifacts/direction_b_confirmation_review/20261003T135430Z`;
201–203: `artifacts/recipe_confirmation_review/20261007T234436Z`). This shows the test is run
on exactly the models that were judged on validation. Any mismatch or missing record stops
the work.

Old-recipe caveat: seed 42 used the original runners; FC-Siam-Diff and BIT-SAR v2 never share
initialization or batch order with A–D (as in earlier protocols).

## 4. Evaluation

- Same metric code as every validation review: `review_pilot_validation.evaluate()` with
  `object_boundary_patch_v2`, threshold 0.5, event-macro aggregation `event_pixel_v2`.
- Metrics: global and per-event pixel P/R/F1/IoU; event-macro P/R/F1; predicted-positive
  fraction; component P/R/F1; interior-small and clipped-small recall; boundary P/R/F1;
  inner-band IoU; per-event GT positive fraction.
- No other metric is computed on test.

## 5. Pre-specified analyses

All on the **last** checkpoint unless stated. Percentage points where differences.

### T1. Replication of the validation rules (descriptive outcome labels)

The three sign rules of `recipe_confirmation_v1` §8 are applied to test, separately for the
old-recipe seeds (42, 1337, 2026) and the R1 seeds (201, 202, 203):
- size weighting: supported if B−A ≤ 0 and D−C ≤ 0 (interior-small recall) in all three
  seeds, contradicted if both > 0 in all three, else inconsistent;
- boundary package: the same with C−A and D−B on inner-band IoU;
- FC-Siam-Diff: supported if FC ≥ A and FC ≥ BIT-SAR v2 (event-macro F1) in all three seeds,
  contradicted if FC < A in all three or FC < BIT-SAR v2 in all three, else inconsistent.

These labels are reported as they come out. Their meaning is "does the test set show the
same pattern", not a new confirmatory test. If a rule needs a metric that is undefined on
test (for example interior-small recall with zero interior-small components), it is reported
as **not evaluable**, not substituted.

### T2. Seed sensitivity

For each arm and recipe:
- per-seed values, mean and range (max − min) of event-macro F1 and of each per-event F1;
- **rank stability:** for each of the 15 arm pairs, whether the ordering by event-macro F1 is
  the same in all three seeds of a recipe. Report the number of the 15 pairs with a stable
  ordering, per recipe.

### T3. Event sensitivity (generalization of the Lumberton finding)

- The **low-fraction test event** is the test event with the lowest GT positive fraction,
  determined from the test denominators of §2 (identical for all checkpoints).
- For each of the 12 arm × recipe groups, find the test event with the largest across-seed
  range of per-event F1.
- **Rule (adopted 2026-10-09, before any test access):** the finding *generalizes* if the low-fraction event has the largest range in
  **at least 7 of the 12** groups; *does not generalize* if in at most 4; otherwise
  *partial*.
- Also reported: per run, the number of test events with F1 < 20%, and the predicted-positive
  fraction against truth, per event.

### Benchmark table (for C)

Per arm and recipe: mean and range across seeds of event-macro F1, global pixel F1,
component F1, interior-small recall, boundary F1 and inner-band IoU, with parameter counts.
Best-checkpoint values are reported in a separate descriptive table only.

### Wording limits

- No p-values or confidence intervals from pixels, patches or components.
- No statement that any arm "is best" beyond the pre-specified tables.
- No recomputation with other thresholds, checkpoints, seeds, event subsets or weights.

## 6. Permitted code changes and regression checks

1. **Test reviewer.** A new script, the only code permitted to read the test split. It reuses
   `evaluate()` and `object_boundary_patch_v2` unchanged, with the split as a parameter.
   - **Regression (validation, before any test access):** in validation mode, the new script
     must reproduce the existing validation `report.json` metrics and object/boundary counts
     **exactly** for at least one checkpoint per recipe group (A last, s42 and s201) and one
     FC-Siam-Diff and one BIT-SAR v2 checkpoint.
   - Synthetic CPU tests: the split guard, checkpoint-identity check, denominator-consistency
     check and the T1/T2/T3 calculations.
2. **Single-use guard.** The test mode needs `--approve-test` and refuses to start if any
   `heldout_test_v1` output already exists.
3. **Interruption.** If the run stops part-way (for example a CUDA error), it may resume with
   the same code on the checkpoints not yet evaluated. Completed evaluations are never re-run
   or overwritten. Any code change after test access needs user approval and must not touch
   metric computation.
4. Existing scripts and run directories are never edited.

## 7. Execution sequence

1. Engineering, synthetic tests and the validation-mode regression (no test access).
2. Launch gate (§9).
3. 72 test evaluations, in tmux, one at a time.
4. A `RESULTS.json` with the T1–T3 outcomes and a closure document.

## 8. Stop rules and outputs

**Stop** with no bypass on: a checkpoint hash that differs from its validation record; a
missing checkpoint or record; any change in test denominators between evaluations; a
config, epoch or parameter-count mismatch; a CUDA error (then §6.3); any training call.

**Outputs.** `artifacts/heldout_test_v1/<UTC>/`: per-checkpoint reports, `summary.csv`,
`denominators.json`, `RESULTS.json` and `COMPLETE.json` with `evaluations: 72`,
`split: test`, `training_performed: false` and all match flags true.

## 9. Launch gate and cost

Engineering and the test run each need explicit user approval. At launch, record a read-only
GPU snapshot and script/source hashes.

**Estimated cost.** Validation reviews took about 70 s per checkpoint. The test patch count is
unknown; assuming a test set of similar size, 72 evaluations take about 1.5–2 h. Storage is
small (reports and CSVs). Engineering about 1 day.

## 10. Threshold provenance

| Item | Provenance |
|---|---|
| Checkpoint list, last as primary, best descriptive | User approval 2026-10-09 (proposals 1–2). |
| T1 sign rules | Copied from `recipe_confirmation_v1` §8; applied descriptively. |
| T3 "7 of 12 / 4 of 12" | **Newly proposed here**, before any test access. A majority rule, not a statistical test. |
| Rank stability over 15 pairs | Newly proposed descriptor; no threshold. |

## 11. Choices approved on 2026-10-09 (as drafted)

1. **Checkpoints:** the 36 runs and 72 checkpoints of §3 (already approved in scope).
2. **T1:** apply the three validation sign rules to test descriptively, per recipe group.
3. **T3:** the "lowest flood fraction event" rule with the 7-of-12 / 4-of-12 cut points.
4. **Identity check:** every checkpoint must match its validation review hash.
5. **Interruption rule:** resume only on unevaluated checkpoints, no re-runs.
6. **Single use:** one test evaluation under this protocol; any second look needs a new
   protocol and a stated reason.

Test stays sealed until §9 approval.
