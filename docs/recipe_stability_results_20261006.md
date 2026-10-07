# recipe_stability_v1: phase-1 results and closure

## Material Passport

Protocol: `docs/recipe_stability_protocol_v1_20261006.md` (branch
`direction-b-confirmation`, commit `b2865d1`). Runner, tests and launcher: commit `e3892c8`.
Execution: server run `runs/recipe_stability_v1/20261006T033408Z`, logs
`artifacts/recipe_stability_v1/20261006T033408Z`. Started 2026-10-06T03:34Z, completed
2026-10-06T17:02Z (about 13.5 h).
Scope: training events only. The current validation events (Lumberton, Coraki, Hebei) were
never loaded, and the held-out test set stayed sealed.
Status: **closed.** The §5 decision rule returns **selected: R1** (learning rate 1e-4,
constant schedule). Sections 4–5 are post-hoc descriptive diagnostics, not preregistered
analyses. Section 6 lists minor deviations in outputs, none of which affect the decision.

## 1. Execution and integrity

- Source identity: server `script_sha256.txt` matches commit `e3892c8` for
  `recipe_stability.py` (`5e03eb8d…aece`), `direction_b_factorial.py`,
  `proposed_factorial.py` and `tests/unit/test_recipe_stability.py`.
- Pre-launch checks, all passed:
  - 8 synthetic tests;
  - seed-42 regression (`regress_s42/REGRESSION_PASS.json`): initial state
    `71c04062…54b8` and full-train batch order match the factorial anchor;
    `workers_equivalent: true` for 0 vs 4 workers on the 16-patch subset;
  - 1-epoch preflight R0–R3 (`preflight/COMPLETE.json`), checkpoint reload exact.
- All 12 runs report `status: COMPLETE`, 15 epochs, 30,210 optimizer steps (2,014 per epoch),
  `checkpoint_reload_exact: true`, `validation_events_loaded: false`, `test_used: false`.
  No traceback in any log.
- Pairing held within each seed: R0–R3 share one initial state and identical per-epoch
  batch orders (seed 101 `c0a7b69e…`, 102 `7e778c49…`, 103 `1946a44b…`).
- Each run took 1.05–1.13 h with 4 workers, close to the idle-server estimate in §8.
- `DECISION.json` SHA256 `dd39bed1…6b70`; it records `validation_events_loaded: false`
  and `test_used: false`.

## 2. Decision rule (§5, epoch-15 checkpoint, internal dev)

Dev = {20190502_Canada, 20191012_Japan}. Percent.

| Recipe | Macro F1 s101 / s102 / s103 | Mean | Range | Japan F1 s101 / s102 / s103 | Japan floor (50% of median) | Stable |
|---|---|---:|---:|---|---:|---|
| R0 (3e-4, constant) | 34.77 / 38.90 / 38.47 | 37.38 | 4.13 | 62.35 / 66.92 / 52.09 | 31.18 | yes |
| **R1 (1e-4, constant)** | 52.40 / 47.99 / 50.18 | **50.19** | 4.41 | 45.58 / 54.15 / 51.30 | 25.65 | **yes** |
| R2 (3e-4, warmup + cosine) | 41.90 / 42.03 / 36.27 | 40.07 | 5.76 | 52.50 / 59.50 / 45.61 | 26.25 | no (range) |
| R3 (1e-4, warmup + cosine) | 37.03 / 46.57 / 48.69 | 44.10 | 11.66 | 17.65 / 49.43 / 47.35 | 23.68 | no (range, Japan s101) |

- Stable recipes: R0, R1.
- Selection: R1 has the highest mean, 12.81 pp above R0, far outside the 1 pp tie band.
- **Outcome: R1 selected.** Recomputed independently from the per-seed values; it agrees
  with `DECISION.json`.

## 3. Per-run detail (epoch 15, internal dev, percent)

Pred. pos. = predicted-positive pixel fraction over both dev events; ground truth is 1.94%.
AP is binned (10,000 bins) from logits, diagnostic only.

| Recipe | Seed | Canada P / R / F1 | Japan P / R / F1 | Pred. pos. | Canada AP | Japan AP | Best epoch (macro F1) |
|---|---:|---|---|---:|---:|---:|---|
| R0 | 101 | 63.6 / 3.8 / 7.19 | 75.2 / 53.3 / 62.35 | 0.27 | 11.5 | 62.2 | 7 (53.02) |
| R0 | 102 | 77.5 / 5.9 / 10.89 | 65.1 / 68.9 / 66.92 | 0.38 | 17.1 | 69.7 | 14 (60.74) |
| R0 | 103 | 60.9 / 15.6 / 24.84 | 39.8 / 75.5 / 52.09 | 0.89 | 29.7 | 66.1 | 2 (59.87) |
| R1 | 101 | 87.9 / 44.6 / 59.22 | 36.2 / 61.6 / 45.58 | 1.27 | 72.9 | 44.1 | 15 (52.40) |
| R1 | 102 | 76.2 / 28.8 / 41.82 | 41.8 / 76.9 / 54.15 | 1.08 | 46.9 | 56.8 | 14 (55.02) |
| R1 | 103 | 71.6 / 37.3 / 49.06 | 37.9 / 79.6 / 51.30 | 1.39 | 54.3 | 66.6 | 11 (58.67) |
| R2 | 101 | 69.1 / 20.2 / 31.31 | 41.0 / 73.0 / 52.50 | 0.92 | 35.5 | 61.2 | 3 (55.82) |
| R2 | 102 | 81.2 / 14.5 / 24.56 | 49.4 / 74.7 / 59.50 | 0.66 | 37.8 | 69.0 | 3 (52.74) |
| R2 | 103 | 60.4 / 17.3 / 26.92 | 38.0 / 57.0 / 45.61 | 0.84 | 31.6 | 52.2 | 3 (44.91) |
| R3 | 101 | 63.5 / 50.8 / 56.42 | 10.5 / 55.6 / 17.65 | 2.62 | 63.6 | 9.3 | 3 (46.00) |
| R3 | 102 | 78.5 / 30.3 / 43.71 | 46.5 / 52.8 / 49.43 | 0.93 | 49.8 | 44.8 | 5 (57.60) |
| R3 | 103 | 68.5 / 39.4 / 50.03 | 35.9 / 69.4 / 47.35 | 1.44 | 56.4 | 54.8 | 8 (55.12) |

Final train loss: R0 0.167 (all seeds); R1 0.173–0.176; R2 0.165–0.170; R3 0.176–0.187.
Clipping fraction in epoch 15: R0 0.25–0.26, R1 0.40–0.45, R2 0.32–0.33, R3 0.49–0.53.

## 4. Post-hoc diagnostic: failure mode on dev (read-only, logs only)

**Observations, descriptive only:**
1. **The R0 failure on dev is under-prediction on Canada, not false positives.** R0
   Canada recall is 3.8–15.6% at epoch 15, with 0.27–0.89% predicted positive against
   1.94% truth. Canada AP is also low (11.5–29.7), so this is not only a 0.5-threshold
   effect. This is the opposite direction from the Lumberton false-positive regime seen
   in `direction_b_confirmation_v1`. R0's "stable" verdict rests on the macro range; its
   Canada F1 ranges 7.2–24.8.
2. **R1 is the only recipe with balanced events in all three seeds** (Canada F1 41.8–59.2,
   Japan F1 45.6–54.2). The cost is lower Japan precision (36–42%) than R0.
3. **Best-epoch selection would again pick transient early peaks.** Best macro F1 occurs at
   epoch 2–7 for 5 of 12 runs (all R2 runs at epoch 3), supporting the protocol choice of
   the last checkpoint.
4. **R1 is not flat within a run.** Over epochs 11–15, R1 s101 averages 44.4 macro F1
   against 52.4 at epoch 15, and its Japan F1 dips to 25.9 in that window (below the floor
   if that epoch had been last). The §5 rule judges one checkpoint per run; epoch-to-epoch
   fluctuation is not part of it.
5. Both stable recipes pass the 5 pp range rule with modest margins (R0 0.87 pp, R1 0.59 pp).

**What this does not establish.** Three seeds on two dev events cannot show that R1
removes the Lumberton false-positive regime on the current validation events. The dev
failure mode differs from the one that motivated the study. Learning rate and schedule are
not separated from their interaction with gradient clipping (R1 clips 40–45% of steps
versus 25% for R0). Nothing here licenses other recipes, more seeds, re-thresholding or
any test-set use.

## 5. Conclusions permitted by this protocol

- Under the frozen arm-A contract and the §3 internal dev split, R1 (learning rate 1e-4,
  constant schedule, 15 epochs) is the selected training recipe for phase 2.
- Lowering the learning rate from 3e-4 to 1e-4 with a constant schedule raised mean dev
  macro F1 from 37.4 to 50.2 and stayed within the stability criteria. Adding warmup and
  cosine did not help at either learning rate under these criteria.

**Not permitted:**
- p-values or claims of a statistically significant recipe effect;
- reporting dev F1 as a generalization estimate (dev events also contributed to
  normalization statistics, protocol §2);
- choosing a different recipe, seed or checkpoint after the fact;
- any validation-event or test-set use under this protocol.

## 6. Deviations from §6 outputs (no effect on the decision)

1. §6 asks for `COMPLETE.json` in `artifacts/recipe_stability_v1/<UTC>/`. The run wrote
   `COMPLETE.json` only for the preflight; the final record is
   `runs/…/DECISION.json`, which carries the same two flags. Per-run completion is in each
   `summary.json`.
2. Per-run per-epoch tables are in `runs/…/R*_s*/metrics.jsonl`, not copied into the
   artifacts directory. Tables in §2–3 were computed read-only from those files.
3. Storage is 458 MB (checkpoints included), above the "under 0.3 GB" estimate in §8.
4. Results remain server-only; no run or artifact tree was copied into this checkout.

## 7. Next decisions (require user approval)

Phase 2 (§9) needs its own frozen protocol. Open choices:
- whether FC-Siam-Diff and BIT-SAR v2 adopt R1 or keep their own recipes (BIT-SAR v2 already
  uses 1e-4);
- number of confirmation seeds;
- whether selection moves from `event_pixel_v2` best to last;
- whether to pre-register a diagnostic for within-run epoch fluctuation (§4 item 4).

Test stays sealed.
