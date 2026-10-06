# Recipe stability protocol v1

Protocol ID: `recipe_stability_v1`.
Drafted and approved 2026-10-06 (user reply "duyệt"). All six §10 choices were approved
as drafted: the §3 dev split, recipes R0–R3, the 5 pp / 50% thresholds, last as the
primary checkpoint, `num_workers = 4` subject to its regression, and seeds 101–103.
**Status: protocol approved; engineering and launch NOT yet approved.** No code,
preflight or training may start until the user separately approves them.

## 1. Why this protocol exists

`direction_b_confirmation_v1` returned **inconsistent** on all three decision rules
(`docs/direction_b_confirmation_results_20261006.md`). Read-only diagnostics found:
- training loss converges similarly across seeds;
- the seed sensitivity sits almost entirely in false-positive behaviour on one held-out,
  low-flood-fraction event (Lumberton, about 0.44% flood);
- BIT-SAR v2, which also uses learning rate 1e-4, never collapsed.

The cause is unknown. Before any further comparison of arms, this protocol asks one
bounded question:

> **Q.** Under the frozen data and model contract, does changing only the learning rate
> and/or the learning-rate schedule make factorial arm A stable across seeds on held-out
> events?

It is a **phase-1 screening** study. It does not compare loss variants, architectures or
baselines. It produces one frozen training recipe, or a documented failure to find one.

## 2. Hard constraints

- **The current validation events (Lumberton, Coraki, Hebei) are not loaded at all.** They
  are reserved for the later phase-2 comparison. The runner must assert this.
- **The held-out test set stays sealed.**
- Unchanged:
  - data contract: split v1, manifest, normalization and component-statistics hashes, VH/VV
    order, 256 patches, threshold 0.5;
  - model: SmallFloodCDNet A, boundary off, size weighting off;
  - loss: 0.4 BCE + 0.4 Tversky;
  - optimizer: AdamW, weight decay 1e-4, clip 1.0;
  - batch 8, FP32, strict determinism.
- Normalization and component statistics were fitted on all 12 training events, including
  the two internal-dev events below. They stay frozen. This mild input-statistics overlap is
  disclosed, not corrected; refitting would change frozen inputs.

## 3. Internal development split (label statistics only, no model outputs)

Training events are grouped by location so that no location is in both parts:
- Houston = {2016-04-19, 2017-08-30};
- Sydney = {2021-03-24, 2022-07-05};
- Somalia = {2018-05-08 Somalia, 2023-11-14 Beledweyne};
- the other six events are singletons.

That gives 9 groups.

**Rule:**
1. Order the groups by `sha256("internal_dev_v1:42:" + group)`.
2. Hold out ceil(20% × 9) = 2 groups.
3. At least one held-out group must have every event below 1% flood fraction, mirroring
   Lumberton. If the first two hash-ordered groups contain none, replace the second with
   the first such group in hash order.

**Result** (computed while drafting, from event names and manifest label statistics only):
- Hash order: Canada, Houston, Japan, Beira, PortMacquarie, Niger, Sydney, Iran, Somalia.
- Neither Canada (3.08%) nor Houston (2.33% / 3.39%) is below 1%, so Houston is replaced by
  Japan (0.53%).
- **Internal dev = {20190502_Canada, 20191012_Japan}**: 3,059 patches.
- **Screening train** = the other 10 events: 16,107 patches.

## 4. Design

| Recipe | Learning rate | Schedule | Epochs |
|---|---|---|---|
| R0 (frozen recipe) | 3e-4 | constant | 15 |
| R1 | 1e-4 | constant | 15 |
| R2 | 3e-4 | 1-epoch linear warmup, then cosine to 0 (per step) | 15 |
| R3 | 1e-4 | 1-epoch linear warmup, then cosine to 0 (per step) | 15 |

- A 2×2 design: learning rate × schedule. Epoch count stays at 15, so it is not a factor.
- **Seeds 101, 102 and 103.** These are new; no result exists for any of them.
- **12 runs**, all arm A.
- Each run logs, per epoch on internal dev:
  - per-event and event-macro pixel P/R/F1;
  - predicted-positive fraction;
  - train loss, pre-clip gradient norm, clipping fraction, learning rate.
- At epoch 15, and at the best epoch by dev event-macro F1, also log **per-event average
  precision** computed from logits. AP is threshold-free, so it separates a ranking problem
  from a calibration problem at threshold 0.5. It is diagnostic only.

**Speed (engineering, no semantic change).** `num_workers = 4` instead of 0.
- The dataset has no random augmentation, and batch order comes from the sampler generator
  in the main process. Batches should therefore be identical to `num_workers = 0`.
- **Required regression:** a 1-epoch run on a 16-patch subset with 0 and with 4 workers
  must give identical batch-order hashes and identical final weights. If it does not, use
  0 workers.

## 5. Decision rules (new; adopted 2026-10-06 before any phase-1 result exists)

All rules use the **epoch-15 (last) checkpoint**. Selecting by best epoch was shown to pick
transient spikes. "Dev macro F1" is the equal mean of Canada and Japan event F1.

1. **Stable recipe:** across seeds 101–103, the range (max − min) of dev macro F1 is ≤ 5 pp,
   **and** every seed has Japan event F1 ≥ 50% of that recipe's median Japan F1.
2. **Selection:** among stable recipes, pick the highest mean dev macro F1. Break ties
   (|Δ| < 1 pp) in favour of R0, then the lower learning rate, then constant schedule.
3. **If no recipe is stable:** report that learning rate and schedule do not resolve the
   instability, stop, and return to the user. This would directly support a
   seed-sensitivity paper (Pipeline 2). It does not authorize more recipes or seeds.

**Threshold provenance:** the 5 pp range and the 50%-of-median floor are newly proposed
here. Both are engineering stability conventions, not statistical tests. They are frozen as of
approval.

## 6. Outputs and stop rules

- Runs: `runs/recipe_stability_v1/<UTC>/R{0..3}_s{101..103}/`.
- Summary: `artifacts/recipe_stability_v1/<UTC>/` containing
  - per-run per-epoch tables;
  - the decision-rule evaluation;
  - `COMPLETE.json`, which records `validation_events_loaded: false` and
    `test_used: false`.
- **Stop rules:** stop on any hash, fingerprint, regression or count mismatch; on any load
  of a validation or test record; or on nonfinite values. No retries. No extra seeds or
  recipes.

## 7. Engineering required (after protocol approval)

1. A new runner, `scripts/recipe_stability.py`. It reuses the `direction_b_factorial.run_cell`
   training logic with:
   - a configurable learning rate and per-step warmup + cosine schedule;
   - a dev split built by the §3 rule and checked against the event list above;
   - `num_workers` with the §4 regression;
   - AP on dev for the last and best checkpoints.
2. Unit tests:
   - the schedule values;
   - the dev-split rule reproduces §3 exactly;
   - the runner refuses validation and test records;
   - workers 0 and 4 give equal results on synthetic data;
   - with R0 settings and the full train set, the code path reproduces
     `direction_b_factorial` (seed 42 initial-state and batch-order anchors).
3. A 1-epoch preflight per recipe.

## 8. Cost estimate

- Screening train is 16,107 patches, about 2,013 steps per epoch. Historical A speed with
  an idle server is about 0.15 s per step, so roughly 1.3 h per run.
- 12 runs ≈ **16 GPU-hours idle**.
- Under the load seen on 2026-10-04 (about 5× slower) the same 12 runs would take 3–4 days.
  The 4-worker loading option may recover much of that, because the bottleneck looked like
  CPU.
- Storage under 0.3 GB.
- Engineering about 2 days.

## 9. What follows (outline only, requires its own protocol and approval)

**Phase 2.** Re-run factorial A–D, FC-Siam-Diff and BIT-SAR v2 on all 12 training events
with the selected recipe. Evaluate on the current validation events with sign-based rules,
using new confirmation seeds. Primary checkpoint: last.

The decisions still open for phase 2:
- whether FC-Siam-Diff and BIT-SAR v2 adopt the selected recipe or keep their own;
- how many seeds;
- whether the selection rule changes from `event_pixel_v2` best to last.

A single held-out test evaluation needs a further separate approval.

## 10. Choices approved on 2026-10-06 (as drafted)

1. **Dev split:** approve the §3 rule, including the low-flood-fraction constraint.
2. **Recipe grid:** R0–R3 as listed. An epoch factor (for example 30 epochs) would double
   the cost.
3. **Stability thresholds:** 5 pp range and the 50% Japan floor, or other values.
4. **Primary checkpoint:** last instead of best.
5. **`num_workers = 4`:** accept, subject to the regression.
6. **Seeds:** 101–103.
