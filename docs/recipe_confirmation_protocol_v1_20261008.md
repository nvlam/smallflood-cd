# Recipe confirmation protocol v1: phase 2 of recipe_stability (validation only)

Protocol ID: `recipe_confirmation_v1`.
Drafted 2026-10-08, after `recipe_stability_v1` selected recipe R1
(`docs/recipe_stability_results_20261006.md`).
Approved 2026-10-08 (user reply "duyệt nguyên các lựa chọn trên"). All six §12 choices
were approved as drafted: all arms at 1e-4 constant, seeds 201–203, last as the primary
checkpoint, 4 workers for the factorial runner subject to the A and C regressions, H0 as a
non-gating rule for arm A, and "all three seeds" rule strength.
**Status: protocol approved and frozen; engineering and launch NOT yet approved.** No code,
preflight or training may start until the user separately approves them (§10).

## 1. Purpose and claim scope

`direction_b_confirmation_v1` could not resolve the factorial contrasts or the whole-method
ranking against seed-to-seed variation under the frozen recipe (learning rate 3e-4). Phase 1
selected a recipe that was stable on an internal dev split of the training events. This
protocol asks:

- **H0 (recipe transfer):** does arm A trained with R1 on all 12 training events stay
  stable across new seeds on the current validation events?
- **H1 (factorial contrasts):** under R1, do size weighting and the boundary package give a
  consistent sign across new seeds?
- **H3 (whole-method):** under R1, is FC-Siam-Diff not worse than factorial A and
  BIT-SAR v2 across new seeds?

**Not covered:** H2 (FU/FO), A3 efficiency, A4 held-out test, H4, Candidate R, the 135-run
matrix, coherence, uncertainty masks, recipes other than R1, epoch counts other than 15,
threshold changes and per-event reweighting.

Results are single-protocol, patch-scoped, event-held-out **validation** observations, not
significance tests or held-out test results.

**Disclosure on validation reuse.** The three validation events (Lumberton, Coraki, Hebei)
are not untouched: they were observed for seeds 42, 1337 and 2026 under the old recipe, and
the Lumberton false-positive regime found there motivated phase 1. Phase 1 itself never
loaded them. This protocol confirms on new seeds and a recipe chosen without them; it is not
a fresh-data test.

## 2. Frozen inputs (unchanged)

- Split `data/splits/split_v1.yaml`; training uses **all 12 training events** (19,166
  patches). The phase-1 internal dev split is not used.
- Manifest SHA256 `51eba373aef67e05b768bfc3c402d15c9fb9c8417c7026c6079a8fc0bea9b3ae`.
- Component stats `525e4e1354e991983ebd2e3d3f68caad3f9894308a6eae46256a74405c9f2247`.
- Normalization `33ef9a286340e3a58b8fc3623da210d83624c8a5901e9bcd0799e660a6ff625d`.
- Server stack as recorded: Python 3.10.12, torch 2.5.1+cu124, CUDA 12.4, cuDNN 90100,
  RTX A6000, the existing `.venv`. Nothing is installed or upgraded.
- Validation denominators, which must reproduce exactly: 4,767 patches; 295,762,355 valid
  pixels; 3,182,289 positive pixels; 6,745 GT components; 1,551 clipped-small; 370
  interior-small (Coraki 42, Hebei 328, Lumberton 0).
- **Test set sealed.** No test arrays, labels or predictions are opened.

## 3. Recipe (§12 choice 1)

All arms train with learning rate **1e-4, constant schedule**, 15 epochs. Everything else
stays as in each arm's frozen recipe:

| | Factorial A/B/C/D | FC-Siam-Diff | BIT-SAR v2 |
|---|---|---|---|
| Learning rate | **3e-4 → 1e-4** | **3e-4 → 1e-4** | 1e-4 (unchanged) |
| Loss | 0.4 BCE + 0.4 Tversky; size weighting B/D; boundary 0.2 C/D | 0.5 BCE + 0.5 Tversky | 0.5 BCE + 0.5 Tversky |
| Initialization | PyTorch default | PyTorch default | normal 0.02, BN reset |

Shared: AdamW, weight decay 1e-4, clip 1.0, batch 8, no scheduler, FP32, strict
determinism, math attention.

**Why FC-Siam-Diff also moves to 1e-4.** Leaving it at 3e-4 would make H3 compare arms at
different learning rates again, which was caveat 2 of the previous H3. With all arms at
1e-4, learning rate is removed as a whole-method difference. BIT-SAR v2 is already at 1e-4
and its runner is unchanged.

**Caveat carried into every H3 table.** The comparison is still not architecture-only. The
arms differ in loss coefficients (total 1.0 vs 0.8), initialization scheme, runner and RNG
convention, and BIT-SAR v2's two-class output adapter. R1 was selected for arm A only; it
was not tuned for FC-Siam-Diff or BIT-SAR v2.

## 4. Seeds (§12 choice 2)

- **Seeds 201, 202 and 203.** No result exists for any of them under any recipe.
- Seeds 42, 1337 and 2026 (old recipe) and 101–103 (phase 1, dev split) are reported only as
  history. They enter no decision rule.
- No seeds are added, whatever the outcome.

## 5. Runs

| Arm | Runs | Runner |
|---|---|---|
| Factorial A/B/C/D | 4 cells × 3 seeds = 12 | Versioned successor of `scripts/direction_b_factorial.py` with `--lr` fixed to 1e-4 |
| FC-Siam-Diff | 3 | Versioned successor of `scripts/direction_b_pilot.py` that loads a new config `configs/experiment/recipe_confirmation_v1/fc_siam_diff.yaml` (identical to `fc_siam_diff.yaml` except `learning_rate: 0.0001`) |
| BIT-SAR v2 | 3 | Same successor, BIT-SAR v2 path unchanged apart from seed list |

**18 new runs in total.** Factorial cells within one seed share the initial state and batch
order, as before.

## 6. Permitted code changes and regression checks

Only the learning rate, the seed list and (optionally) the worker count may change.

1. **Factorial runner.** New versioned file. Learning rate becomes a guarded parameter that
   accepts only 1e-4 for training runs.
   - **Regression, no training:** with seed 42 and learning rate 3e-4 the new runner must
     reproduce the recorded anchors: initial state A/B `71c04062…54b8`, C/D
     `db03a4cf…e476`, and all 15 batch-order hashes.
   - **Regression, short training:** with seed 42, R1 settings, 16 patches and 1 epoch, the
     new runner and `recipe_stability.run_recipe` (R1, called on the same 16 training
     patches) must give identical final weights for cell A. This ties the phase-2 training
     path to the code that produced the phase-1 selection.
2. **FC-Siam-Diff / BIT-SAR v2 runner.** New versioned successor of
   `direction_b_pilot.py`: seed list {201, 202, 203}; FC loads the new config.
   - The new FC config must differ from `fc_siam_diff.yaml` only in `learning_rate`
     (checked by a unit test on the parsed YAML).
   - Parameter counts: FC-Siam-Diff 487,857; BIT-SAR v2 3,492,642.
   - BIT-SAR v2 keeps its batch-8 gate.
3. **Workers (§12 choice 4).** The factorial runner uses `num_workers = 4`, subject to
   the phase-1 style regression (0 vs 4 workers, 16 patches, 1 epoch, identical batch-order
   hashes and final weights) run **separately for A and for C**, because C/D load boundary
   targets. If either fails, use 0. FC-Siam-Diff and BIT-SAR v2 keep 0 workers (their
   runner is `next_steps.pilot`, which is not changed).
4. **Reviewer.** Versioned successor of `scripts/review_direction_b.py`, accepting seeds
   201–203. Reuses `evaluate()` and `object_boundary_patch_v2` unchanged; keeps every guard:
   validation-only loader, checkpoint hash before and after, config/epoch/step checks,
   TP/FP/FN/TN equal to the training log, denominators as in §2. Adds per-epoch
   Lumberton/Coraki/Hebei F1 tabulation from `metrics.jsonl` for the H0 diagnostic.
5. Every new script records its SHA256. Existing scripts and run directories are never
   edited or overwritten.

## 7. Execution sequence and checkpoints

1. Implement §6; run synthetic tests (CPU and CUDA) and the regressions.
2. 1-epoch preflight per runner and seed in a fresh directory; never promoted.
3. Full 15-epoch runs in a fresh timestamped directory, in tmux, one at a time. No resume,
   no automatic retry.
4. Review best and last checkpoints of all 18 runs (36 evaluations).

**Primary checkpoint: last (epoch 15)** for every decision rule (§12 choice 3). Phase 1
showed best-epoch selection picks transient early peaks (5 of 12 runs peaked at epoch 2–7).
The `event_pixel_v2` best checkpoint is still saved, reviewed and reported descriptively.

**Metrics** unchanged: global and event pixel P/R/F1/IoU; component P/R/F1; interior-small
and clipped-small recall; boundary P/R/F1; inner-band IoU; per-event results and
denominators.

## 8. Decision rules (adopted 2026-10-08, before any phase-2 result exists)

All rules use the **epoch-15 checkpoint** and **all three seeds 201–203**. Contrasts are
paired within a seed. Sign-based; magnitudes reported but not used.

### H0. Recipe transfer (arm A, event-macro F1 and Lumberton F1)

- **Stable** if, across seeds 201–203, the range of validation event-macro F1 is ≤ 5 pp
  **and** every seed has Lumberton F1 ≥ 50% of A's median Lumberton F1.
- Otherwise **not stable**.
- The same quantities are reported for B, C, D, FC-Siam-Diff and BIT-SAR v2 without a rule.
- H0 does not gate H1/H3. All rules are evaluated and reported; if H0 is "not stable", every
  H1/H3 outcome carries that label.

### H1. Factorial contrasts (interior-small recall, n = 370)

- **Size weighting does not improve interior-small recall**
  - **Supported** if B−A ≤ 0 **and** D−C ≤ 0 in all three seeds.
  - **Contradicted** if B−A > 0 **and** D−C > 0 in all three seeds.
  - **Inconsistent** otherwise.
- **Boundary package gives no persistent inner-band IoU gain**
  - **Supported** if C−A ≤ 0 **and** D−B ≤ 0 in all three seeds.
  - **Contradicted** if both > 0 in all three seeds.
  - **Inconsistent** otherwise.

### H3. Whole-method comparison (event-macro pixel F1)

- **"FC-Siam-Diff is not worse than the heavier arms"**
  - **Supported** if FC ≥ A **and** FC ≥ BIT-SAR v2 in all three seeds.
  - **Contradicted** if FC < A in all three seeds, or FC < BIT-SAR v2 in all three seeds.
  - **Inconsistent** otherwise.

### Wording limits

- "Inconsistent" or "not stable" is reported as such and triggers no further runs.
- No p-values or confidence intervals from pixels, patches or components.
- With n = 370 interior-small components and none in Lumberton, one match is 0.27 pp; H1
  size-weighting outcomes say nothing about Lumberton.

## 9. Threshold provenance

| Item | Provenance |
|---|---|
| R1 recipe | Selected by `recipe_stability_v1` §5 on the internal dev split; validation events not loaded. |
| H0 criteria (5 pp range, 50% floor) | Copied from `recipe_stability_v1` §5, with Lumberton replacing Japan as the low-flood-fraction event. Engineering conventions, not statistical tests. |
| H1/H3 sign rules | Same form as `direction_b_confirmation_v1` §7, extended from two to three seeds. Adopted after the old-recipe results were known, applied only to unobserved seeds. |
| Last instead of best checkpoint for H1 size weighting and H3 | **Changed** from `direction_b_confirmation_v1` (best) on the phase-1 evidence in §7. Disclosed as a protocol change. |

## 10. Launch gate (separate user approval required)

Engineering (§6) and launch (§7) each need explicit user approval. At launch, record a
read-only GPU snapshot (`nvidia-smi`, other users' processes), script and source hashes,
and the tmux session.

**Estimated cost** (historical unshared timings at 0 workers: A 1.52 h, B 1.62 h, C 2.59 h,
D 1.69 h, FC 1.27 h, BIT-SAR v2 2.09 h):

| Item | Estimate |
|---|---|
| 18 full runs | ≈ 32 GPU-hours idle (less if 4 workers help the factorial cells) |
| Preflights and regressions | ≈ 1.5 h |
| Reviews (36 evaluations) | ≈ 40 min |
| Storage | ≈ 1–1.5 GB (phase 1 used 458 MB for 12 arm-A runs) |
| Engineering | about 2 days |

Under the heavy sharing seen on 2026-10-04 (about 5× slower), the runs could take 6–7 days.

## 11. Stop rules and outputs

**Stop** with no bypass on: any hash, fingerprint, regression, denominator, config, epoch or
count mismatch; a missing file; a CUDA error; nonfinite values; any attempt to read the test
set.

**Outputs.**
- Runs: `runs/recipe_confirmation_v1/<UTC>/<arm>_s<seed>/`.
- Logs: `artifacts/recipe_confirmation_v1/<UTC>/`, including per-run per-epoch tables.
- Reviews: `artifacts/recipe_confirmation_review/<UTC>/` with `summary.csv`, per-checkpoint
  reports and `COMPLETE.json` containing `evaluations: 36`, `test_used: false`,
  `training_performed: false` and all match flags true.
- A `DECISION.json` with the §8 outcomes, and a closure document reporting every metric at
  best and last per seed, with seeds 42/1337/2026 (old recipe) alongside as history.

## 12. Choices approved on 2026-10-08 (as drafted)

1. **Recipe:** all arms at learning rate 1e-4 constant (FC-Siam-Diff moves from 3e-4), or
   keep FC-Siam-Diff at 3e-4.
2. **Seeds:** three new seeds 201–203 (18 runs), or two seeds (12 runs, cheaper, weaker
   all-seeds rule).
3. **Primary checkpoint:** last for all rules, best reported descriptively.
4. **Workers:** 4 for the factorial runner subject to the A and C regressions; 0 for
   FC-Siam-Diff and BIT-SAR v2.
5. **H0:** keep the transfer rule for arm A as defined in §8, non-gating.
6. **Rule strength:** "all three seeds" for supported/contradicted, as in v1.

Test stays sealed. A4 needs a further separate approval.
