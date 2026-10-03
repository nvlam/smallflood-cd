# Direction (b) confirmation protocol v1: H1 and H3 (validation only)

Protocol ID: `direction_b_confirmation_v1`.
Written 2026-10-03, after the user approved the pre-flight report ("duyệt").
Status: **protocol approved; execution NOT yet approved.** No code change, preflight or
training may start until the user explicitly approves launch (§9).

## 1. Purpose and claim scope

Direction (b) reports a **controlled negative result** together with a lightweight-baseline
benchmark (user decision A0, 2026-10-03). This protocol covers only:

- **H1:** do the frozen factorial contrasts for size weighting and the boundary package
  replicate on two new seeds?
- **H3:** how does a whole-method comparison of FC-Siam-Diff, BIT-SAR v2 and factorial A
  turn out on the same seeds?

**Not covered:**
- H2 (FU/FO). Only its read-only feasibility check was approved (§10).
- A3 efficiency and A4 held-out test. A3 waits until the models are frozen; A4 is not
  approved.
- H4, CPU or INT8 work, extra models, the 135-run matrix, any Candidate R work, coherence,
  uncertainty masks, scene stitching, cross-dataset tests and shift robustness. All
  deferred.

Results are single-protocol, patch-scoped, event-held-out **validation** observations.
They are not significance tests or held-out test results.

## 2. Frozen inputs (unchanged)

- Split `data/splits/split_v1.yaml`.
- Manifest SHA256 `51eba373aef67e05b768bfc3c402d15c9fb9c8417c7026c6079a8fc0bea9b3ae`.
- Component stats `525e4e1354e991983ebd2e3d3f68caad3f9894308a6eae46256a74405c9f2247`.
- Normalization `33ef9a286340e3a58b8fc3623da210d83624c8a5901e9bcd0799e660a6ff625d`.
- Server stack as recorded: Python 3.10.12, torch 2.5.1+cu124, CUDA 12.4, cuDNN 90100,
  RTX A6000, the existing `.venv`. Nothing is installed or upgraded.
- Validation denominators, which must reproduce exactly:
  - 4,767 patches;
  - 295,762,355 valid pixels;
  - 3,182,289 positive pixels;
  - 6,745 GT components;
  - 1,551 clipped-small components;
  - 370 interior-small components (Coraki 42, Hebei 328, Lumberton 0).
- **Test set sealed.** No test arrays, labels or predictions are opened under this protocol.

## 3. Seed design (discovery / confirmation)

- **Seed 42 = discovery.** Existing runs, already observed. They are reported alongside
  the new runs but **do not enter any decision rule**.
- **Seeds 1337 and 2026 = confirmation.** No results exist for either. All decision rules
  in §7 use only these two seeds.
- No further seeds are added, whatever the outcome, including when a result is
  inconsistent.

## 4. Runs

| Arm | Seed 42 (existing, discovery) | New runs (confirmation) | Runner |
|---|---|---|---|
| Factorial A/B/C/D | `runs/proposed_factorial_v1/20260927T075837Z/{A,B,C,D}_proposed` | 4 cells × {1337, 2026} = 8 | Seed-parameterized successor of `scripts/proposed_factorial.py` |
| FC-Siam-Diff | `runs/pilot15/20260924T032911Z/20260924T051127718020Z_fc_siam_diff` | {1337, 2026} = 2 | `scripts/next_steps.py pilot --model fc_siam_diff --seed S --epochs 15 --batch-size 8` (unchanged; SHA256 `c00b1f594545d4e951961adb0c5b72cfa15941dfb3459845f29a7f90bd5911ca`) |
| BIT-SAR v2 | `runs/bit_sar_v2_pilot15/20260925T015023Z/pilot/20260925T015034589216Z_bit_sar_v2` | {1337, 2026} = 2 | Seed-parameterized successor of `scripts/pilot_bit_sar_v2.py` |

**12 new runs in total.** Each run's recipe is its own frozen recipe, unchanged:

| | Factorial cells | FC-Siam-Diff | BIT-SAR v2 |
|---|---|---|---|
| Loss | 0.4 BCE + 0.4 Tversky; size weighting in B/D; boundary 0.2 in C/D | 0.5 BCE + 0.5 Tversky | 0.5 BCE + 0.5 Tversky |
| Learning rate | 3e-4 | 3e-4 | 1e-4 |
| Initialization | PyTorch default | PyTorch default | normal 0.02, BN reset |

Shared by all runs: AdamW, weight decay 1e-4, clip 1.0, 15 epochs, batch 8, no scheduler,
FP32, workers 0, strict determinism, math attention.

**H3 uses factorial A, not Candidate R's A-control.** A-control belongs to the closed R
comparison.

**Whole-method caveat (must appear in every H3 table).** The H3 comparison is not
architecture-only. The arms differ in:
1. loss coefficients (total 1.0 vs 0.8);
2. learning rate (BIT-SAR v2);
3. initialization scheme;
4. runner and RNG convention (cross-architecture batch order is not verified identical);
5. BIT-SAR v2's two-class output adapter.

## 5. Permitted code changes and regression checks

Only seed plumbing may change. Everything else in the training path stays byte-identical
in behaviour.

1. **Factorial runner.** A new versioned file. Replace the hard-coded `42` in
   `strict_seed`, both DataLoader generators, the post-construction reset and the config
   check (`proposed_factorial.py:62,119,124,126,130,266`) with `--seed S`, where
   S ∈ {42, 1337, 2026}. Require `PYTHONHASHSEED == S`. The recipe guards (loss, training,
   switches) stay as they are.
   - **Regression, no training:** with S = 42 the new runner must reproduce the recorded
     anchors.
     - Initial-state SHA256: A/B
       `71c04062460c9639694c4d24f7944b46a909e3fb990340bee8307db5945754b8`, C/D
       `db03a4cfdbee094bf79e2a082bd135b0b5f7c659cce0a3291886d8ceabd7e476`.
     - Common SHA256 `71c04062…54b8`.
     - All 15 recorded epoch batch-order hashes.
   - Any mismatch stops the work.
2. **BIT-SAR v2 runner.** A new versioned file that passes `--seed S` through instead of
   `seed=42`.
   - **Regression:** for S = 42, the initial `state_dict` SHA256 from the new runner must
     equal the one from the unchanged original runner, computed in-process on CUDA without
     training. The original pilot recorded no init hash.
   - Parameter count must be 3,492,642.
3. **FC-Siam-Diff.** No code change. Must report parameter count 487,857.
4. **Reviewer.** A generalized successor of `review_proposed_factorial.py` and the pilot
   reviewers. It takes a run root and seed. It reuses the existing `evaluate()` and
   `object_boundary_patch_v2` unchanged, and keeps the same guards:
   - validation-only loader;
   - checkpoint hash checked before and after evaluation;
   - config, epoch and global step checked;
   - TP/FP/FN/TN must match the training log exactly;
   - denominators as in §2.
   - With synthetic CPU self-tests.
5. Every new script records its own SHA256. Existing scripts and run directories are never
   edited or overwritten.

## 6. Execution sequence and checkpoint selection

1. Implement §5 and run the regression checks and synthetic self-tests (CPU and CUDA).
2. Run a 1-epoch preflight gate per new runner and seed, in a fresh output directory, as
   the factorial gate did. Preflight outputs are never promoted to pilots.
3. Run the full 15-epoch runs in a fresh, timestamped directory, inside tmux on the server,
   one at a time. No resume, no automatic retry.
4. Review both best and last checkpoints of all 12 new runs.

**Checkpoint selection** is unchanged: `event_pixel_v2`, i.e. event-macro pixel F1 at
threshold 0.5, ties keep the earlier epoch. The epoch-15 last checkpoint is always
reported too. No selection by small recall, boundary quality or any other metric.

**Metrics** are unchanged:
- global and event pixel P/R/F1/IoU;
- component P/R/F1;
- interior-small and clipped-small recall;
- boundary P/R/F1;
- inner-band IoU;
- per-event results and denominators.

## 7. Decision rules (adopted 2026-10-03; confirmation seeds only)

All rules are **sign-based and newly adopted** after the seed-42 results were known (see
§8). Magnitudes are reported, but no magnitude margin is used for any decision. "Best" and
"last" refer to the §6 checkpoints. Per-seed contrasts are paired within a seed:
factorial cells in one seed share the initial state and batch order.

### H1. Factorial contrasts (interior-small recall, n = 370)

- **Size weighting does not improve interior-small recall**
  - **Supported** if B−A ≤ 0 **and** D−C ≤ 0 at best, in **both** 1337 and 2026.
  - **Contradicted** if B−A > 0 **and** D−C > 0 at best, in both seeds.
  - **Inconsistent** otherwise.
- **Boundary package gives no persistent inner-band IoU gain**
  - **Supported** if C−A ≤ 0 **and** D−B ≤ 0 at last, in both seeds.
  - **Contradicted** if both > 0 at last, in both seeds.
  - **Inconsistent** otherwise.
- All other frozen contrasts (C−A, D−B, interaction, every metric, best and last) are
  reported descriptively.

### H3. Whole-method comparison (event-macro pixel F1)

- **"FC-Siam-Diff is not worse than the heavier arms"**
  - **Supported** if FC ≥ A **and** FC ≥ BIT-SAR v2 at best, in **both** 1337 and 2026.
  - **Contradicted** if FC < A or FC < BIT-SAR v2 at best, in both seeds, for the same
    comparator.
  - **Inconsistent** otherwise.
- Interior-small recall, component and boundary metrics, and parameter count are reported
  descriptively, without rules.

### Wording limits

- An "inconsistent" outcome is reported as such. It does not trigger more runs.
- No p-values or confidence intervals are computed from pixels, patches or components.

## 8. Threshold provenance (from the 2026-10-03 pre-flight audit)

| Item | Provenance |
|---|---|
| Factorial contrasts, primary axes, selection, metrics | Frozen before any factorial results (`proposed_factorial_protocol_20260927.md`). That protocol set **no success criterion**. |
| Sign-based rules in §7 | **Newly adopted after seed-42 results were known.** They are applied only to unobserved seeds, but they are not preregistered relative to the discovery data. |
| Earlier proposals (5 pp, −1 pp, −5 pp, "≥2/3 seeds", "FU ≥10 pp") | **Not adopted.** The 5 pp and −1 pp margins were inherited from the Candidate R gate (frozen 2026-09-28 for a different question). The rest were new. They must not be cited as criteria. |

## 9. Launch gate (separate user approval required)

Before any step in §6, the user must explicitly approve launch. At launch, record:
- a read-only snapshot of GPU state (`nvidia-smi`, other users' processes);
- script and source hashes, and the tmux session.

On 2026-10-03 the A6000 was shared with another user's process (about 7.9 GB, 75% load).
Under strict determinism this should not change results. Elapsed times under sharing are
engineering records only and cannot be used for any efficiency claim.

**Estimated cost.** Historical, unshared timings: A 1.52 h, B 1.62 h, C 2.59 h, D 1.69 h,
FC 1.27 h, BIT-SAR v2 2.09 h.

| Item | Estimate |
|---|---|
| 12 full runs | ≈ 21.6 GPU-hours |
| Preflights | ≈ 1 h |
| Reviews | ≈ 25 min |
| Storage | ≈ 0.5 GB (checkpoints 13.3 MB factorial, 5.9 MB FC, 42 MB BIT-SAR v2) |
| Engineering | about 2–3 days |

## 10. H2 feasibility record (read-only, 2026-10-03; no predictions evaluated)

Validation semantic labels (4,767 arrays; 0 mismatches against manifest counts):
- NF 292,580,066; FO 3,018,419; FU 163,870 pixels (FU = 5.15% of flood pixels).
- FU by event: Lumberton 194 px in 2 patches; Coraki 1,369 px in 12 patches; Hebei
  162,307 px in 411 patches (99.0% of validation FU).
- 50 patches have ≥1,000 FU pixels, all in Hebei.

**Conclusion:** only one validation event has meaningful FU support. H2 cannot be tested
across events on validation. Any FU analysis is deferred, and needs separate approval and
a protocol that respects the sealed test set.

## 11. Stop rules and outputs

**Stop rules.** Stop and report, with no bypass, if any of these occurs:
- a hash, fingerprint, regression anchor, denominator, config, epoch or count mismatch;
- a missing file;
- a CUDA error;
- any attempt to read the test set.

**Outputs.**
- Runs: `runs/direction_b_confirmation_v1/<UTC>/<arm>_s<seed>/`.
- Reviews: `artifacts/direction_b_confirmation_review/<UTC>/` with `summary.csv`,
  per-checkpoint reports, `COMPLETE.json`, which must contain:
  - `evaluations`: 24;
  - `test_used`: false;
  - `training_performed`: false (in the review);
  - all match flags true.
- After review: a closure document reporting the §7 outcomes, every metric at best and
  last per seed, the seed-42 discovery values alongside, and the §4 caveats.
