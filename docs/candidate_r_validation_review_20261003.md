# Candidate R validation review and closure: four checkpoints

## Material Passport

Origin: Claude Code continuation of the Codex handoff (`RESEARCH_HANDOFF.md`).
Status: closure record written from an already completed server review; no inference,
training, tests or remote access were run while writing this document.
Inputs: `artifacts/candidate_r_validation_review/20261003T111633Z/` (tracked in Git
except `images/`), the frozen gate in `docs/proposed_bounded_redesign_plan_20260928.md`
and the training audit `docs/candidate_r_training_audit_20261003.md`.
Verdict: **Candidate R fails the predefined screening gate (4 of 5 criteria) and is
closed.** No further coefficient, radius, epoch or seed search under this plan.

## Review provenance

- Pilot root: `runs/candidate_r_v1/20261002T150633Z` on the server.
- Review output: `artifacts/candidate_r_validation_review/20261003T111633Z/results`;
  log one level above. Log ends with `CANDIDATE R VALIDATION REVIEW COMPLETE`.
- `SHA256SUMS` (81 files) verified locally on 2026-10-03: all OK.
- `COMPLETE.json`: evaluations 4, `test_used` false, `training_performed` false,
  `pixel_counts_match_training`, `global_and_event_metrics_match_training` and
  `gt_denominators_identical` all true.
- Reviewer `scripts/review_candidate_r.py` SHA256 `309dba4c…7484` and anchor
  `configs/review/candidate_r_20261003.json` SHA256 `c14b51eb…d704` match the
  recorded `review_protocol.json` values and the current local files.
- Audited training archive SHA256 `a70c3afe…210d` as recorded in the protocol.
- Source import check passed against `/data/fit.lamnv/smallflood_cd/src`.
- Validation only, threshold 0.5, batch 8, `object_boundary_patch_v2`
  (8-connectivity, match IoU 0.1, interior small = GT area ≤35 px not touching patch
  edge or invalid neighbours, boundary tolerance 2, inner-band IoU). Checkpoint
  selection unchanged (event-macro pixel F1); diagnostics did not select checkpoints.

| Cell/checkpoint | Epoch (one-based) | Checkpoint SHA256 |
|---|---:|---|
| A-control best | 12 | `ccf0df21a0518c79f37501913e33c683c844809bbfdf0a960c44b76b20254fe1` |
| A-control last | 15 | `43e0ba3ab3018dd5c8d1f18b583fcbe70ea70f8cf683efa7de36b79107c9e19b` |
| R best | 2 | `332b961cdac4a5baf1fada2c6cb1a612bd9f63319b38a3fc2eebba794aaad085` |
| R last | 15 | `9db40933d37103171d62afba76eafadf248d8506ece77d6175492ced27d7de69` |

All four re-evaluations reproduce the training-log TP/FP/FN/TN and global/event
pixel metrics exactly, so the pixel numbers in the training audit are confirmed.

## Denominators (identical across all four evaluations)

| Event | Patches | GT components | Small clipped | Small interior | Boundary target px |
|---|---:|---:|---:|---:|---:|
| 20161011_Lumberton | 763 | 897 | 110 | **0** | 87,093 |
| 20220302_Coraki_Australia | 479 | 984 | 218 | 42 | 184,098 |
| 20230805_Hebei | 3,525 | 4,864 | 1,223 | 328 | 630,041 |
| Pooled | 4,767 | 6,745 | 1,551 | 370 | 901,232 |

The interior-small denominator remains 370, so criterion 1 needs ≥19 extra matches.
Lumberton has no interior-small GT; its interior-small recall is undefined (null), so
criterion 4 is evaluated on the two defined events, Coraki and Hebei.

## Frozen gate assessment

| # | Criterion (frozen 2026-09-28) | A-control | R | Difference | Result |
|---|---|---:|---:|---:|---|
| 1 | Best pooled interior-small recall R−A ≥ +5 pp | 34.05 (126/370) | 38.65 (143/370) | +4.59 pp, +17 | **Fail** |
| 2 | Best event-macro pixel F1 loss ≤ 1 pp | 82.84 | 66.71 | −16.14 pp | **Fail** |
| 3 | Best overall component precision loss ≤ 2 pp | 57.93 | 24.36 | −33.57 pp | **Fail** |
| 4 | Best interior-small recall not lower in any defined event | Coraki 24/42, Hebei 102/328 | Coraki 23/42, Hebei 120/328 | Coraki −1, Hebei +18 | **Fail** |
| 5 | Last pooled interior-small recall R−A ≥ 0 | 22.97 (85/370) | 36.22 (134/370) | +13.24 pp | Pass |

The gate is conjunctive; failure of any one criterion closes the candidate. Criterion 2
had already failed from training logs; the review adds failures of criteria 1, 3 and 4.
The criterion-1 shortfall (17 versus 19 matches) is reported as measured; it is not to
be reinterpreted as "near pass".

## Other predefined metrics (percent)

| Cell/checkpoint | Event F1 | Pixel F1 | Pred. components | Comp. precision | Comp. recall | Comp. F1 | Small clipped recall | Boundary F1 | Inner-band IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A-control best | 82.84 | 84.64 | 5,659 | 57.93 | 48.60 | 52.85 | 15.22 (236/1551) | 76.92 | 30.46 |
| R best | 66.71 | 74.99 | 12,590 | 24.36 | 45.47 | 31.72 | 14.83 (230/1551) | 57.82 | 18.62 |
| A-control last | 82.57 | 83.67 | 5,206 | 61.06 | 47.13 | 53.20 | 10.96 (170/1551) | 76.22 | 32.79 |
| R last | 59.40 | 51.85 | 19,741 | 17.27 | 50.54 | 25.74 | 14.12 (219/1551) | 38.43 | 15.15 |

Per-event interior-small recall at last: Coraki A 25/42, R 25/42; Hebei A 60/328,
R 109/328.

## Interpretation (descriptive only)

- R's interior-small gains come with large over-prediction: predicted components rise
  from 5,659 to 12,590 at best and from 5,206 to 19,741 at last; pixel FP rises from
  485,780 to 1,194,324 (best) and from 370,862 to 3,929,422 (last). Component recall
  barely changes, so the extra small matches are not accompanied by better object
  detection overall.
- At best, the entire pooled small-recall gain is in Hebei (+18); Coraki loses one
  match. With 42 Coraki interior targets, per-event estimates are coarse.
- Clipped-small recall, boundary F1 and inner-band IoU are lower for R at best; at last,
  R's boundary F1 is roughly half of A-control's.
- R's best checkpoint is epoch 2 of 15; event F1 declined thereafter. Recorded gradient
  ratio (6.94–8.48) and clipping fraction (45.08% vs 29.63%) remain observations from
  the training audit; this review does not establish the causal mechanism.
- Single seed, 15 epochs, patch-scoped validation only. These are engineering screening
  results, not significance tests, scene-level metrics, held-out test results or a
  general statement about local auxiliary losses.

## Closure decisions

- Candidate R is closed as a negative result. Do not promote R, select its checkpoints
  by small recall, relax thresholds, or launch more coefficient/radius/epoch/seed runs
  under the bounded plan.
- The original size-aware/boundary claim remains unsupported (factorial audit), and R
  does not supply a replacement claim.
- Any next step (negative-result reporting, a lightweight-baseline deployment study, or
  a new method protocol with prior-art review) requires an explicit user decision and
  its own frozen protocol.

## Evidence to preserve

- Tracked in Git: this review's reports, CSVs, `COMPLETE.json`, `review.log`,
  `SHA256SUMS`. Not tracked: `results/*/*/images/` (ignored by `.gitignore`).
- Server only: the four checkpoints and full run directory
  `runs/candidate_r_v1/20261002T150633Z`, eligibility audit
  `artifacts/candidate_r_eligibility/20260928T134626505807Z`, and the review images.
  Keep them immutable.
- Outside the repository: training archive
  `/Users/nguyenvulam/Downloads/candidate_r_training_review_20261003.tar.gz`
  (SHA256 `a70c3afe…210d`). Copy it into durable storage; a fresh clone does not
  contain it.
- `RESEARCH_HANDOFF.md` sections 9–11 and 20 describe this review as pending; this
  document supersedes that status without editing the handoff snapshot.
