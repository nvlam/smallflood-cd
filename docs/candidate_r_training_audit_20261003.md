# Candidate R: audit of completed A-control/R training logs

## Material Passport

- Origin: academic-research-suite / experiment-agent, validate.
- Date: 2026-10-03; version: candidate_r_training_audit_v1.
- Verification Status: ANALYZED. Archive/log consistency checked locally;
  no model inference, checkpoint reload, retraining, or test evaluation performed.
- Source: `/Users/nguyenvulam/Downloads/candidate_r_training_review_20261003.tar.gz`.
- Archive SHA256: `a70c3afe5bb4484aa845ede9a873a9cade6c0489facabf288cccb855d14c210d`.
- Server run: `/data/fit.lamnv/smallflood_cd/runs/candidate_r_v1/20261002T150633Z`.
- Scope: descriptive assessment of the frozen seed42 pilot, not statistical
  significance, mechanism identification, or a general verdict on component losses.

## Outcome

The recorded best-checkpoint event-macro F1 decreases from 0.8284351358718937
(A-control) to 0.6670670929241197 (R): **−16.1368 percentage points**.
This fails the frozen maximum one-percentage-point decrease criterion. Thus this
candidate does not pass the conjunctive engineering screening gate on the supplied
logs, irrespective of any eventual improvement in small-region recall.

Small-component, component precision and boundary results are absent from this
archive. Do not invent them or claim they decreased. Reloading and evaluating the
four saved best/last checkpoints remains useful to confirm the recorded metrics
and document the complete negative result, not to select another checkpoint or
relax the gate. No additional training, coefficient/radius/seed sweep or test-set
evaluation is recommended under the frozen budget.

## Consistency checks performed

- All 11 archive entries are regular files; data parsed without executing archive
  content. No checkpoints or image arrays are included.
- Root COMPLETE stage is `run`, `test_used=false`; its fingerprint matches
  `provenance.json`, and both embedded summaries match their standalone files.
- Both cells contain exactly 15 metric rows, epochs0–14, with 2,396 optimizer
  steps per epoch and 35,940 steps total. Each reports 1,079,865 parameters.
- Initial-state SHA256 is identical:
  `71c04062460c9639694c4d24f7944b46a909e3fb990340bee8307db5945754b8`.
- All 15 batch-order hashes match between cells and their summaries. Every epoch's
  batch count, empty-batch count, eligible count and order hash match the saved
  eligibility audit. Both total 16,562 active batches; 3,521 eligible components
  per epoch; no empty rings. These do not prove that every input array was immutable.
- Resolved configurations match the local frozen runner configuration and full
  selection protocol. The only inter-cell differences are the cell label and local
  coefficient (0 versus .1). Head and old size weighting remain off.
- Recomputed diagnostics hashes match both summaries after excluding timing and
  CUDA memory fields, as specified by the runner.
- Every epoch has 295,762,355 valid validation pixels and 3,182,289 positives.
  Global precision/recall/F1/IoU recompute from confusion counts, event macro
  metrics equal the means of the three supplied event metrics, and scaled losses
  sum to the recorded total within floating-point tolerance.
- Maximum recorded selection scores match summary best scores. Earliest maximum
  is A epoch11 and R epoch1. Epoch numbers are zero-based.
- The two environment records agree: torch2.5.1+cu124, CUDA12.4, cuDNN90100,
  Python3.10.12, NumPy2.2.6, SciPy1.15.3, RTX A6000, 24 torch threads,
  PYTHONHASHSEED42 and CUBLAS_WORKSPACE_CONFIG=:4096:8; 19,166 train/4,767 validation.
- 92 recorded source/config file hashes match local files. One recorded utility,
  `scripts/audit_urbansarfloods.py`, is absent locally and could not be compared.
  The three recorded candidate test-file hashes match. Frozen metadata hashes
  agree with the runner's expected constants; raw metadata/data content on the
  server was not independently re-read in this audit.
- `checkpoint_reload_exact=true` is a runner report. Checkpoint hashes are present,
  but their binary contents were not supplied and therefore were not independently
  verified here. Matching paired initialization/order is not a second full-run
  reproducibility experiment.

## Best and last comparison

Percentages below are descriptive validation results, threshold .5.

| Cell/checkpoint | Logged epoch | Event-macro F1 | Global pixel precision | Global pixel recall | Global pixel F1 | Global pixel IoU |
|---|---:|---:|---:|---:|---:|---:|
| A-control best | 11 (12th epoch) | 82.84 | 84.71 | 84.58 | 84.64 | 73.37 |
| R best | 1 (2nd epoch) | 66.71 | 68.73 | 82.51 | 74.99 | 59.99 |
| A-control last | 14 (15th epoch) | 82.57 | 87.33 | 80.31 | 83.67 | 71.93 |
| R last | 14 (15th epoch) | 59.40 | 38.78 | 78.21 | 51.85 | 35.00 |

Best-to-best global F1 falls 9.6492 points. Last-to-last event-macro F1 falls
23.1671 points. This is not merely a poor final checkpoint: R's best of all15
epochs already fails the frozen F1 gate. Conversely, do not judge the intended
best-checkpoint comparison using only last results.

| Validation event | A best F1 | R best F1 | A last F1 | R last F1 |
|---|---:|---:|---:|---:|
| 20161011_Lumberton | 75.78 | 35.69 | 76.46 | 8.07 |
| 20220302_Coraki_Australia | 91.10 | 87.52 | 92.36 | 92.68 |
| 20230805_Hebei | 81.65 | 76.91 | 78.87 | 77.44 |

At best, all three event F1 values are lower for R. At last, Coraki is slightly
higher but Lumberton is substantially lower. Do not extrapolate this mixed event
behavior to unseen regions/seasons or use event-specific checkpoint selection.

At last, total false positives are A370,862 versus R3,929,422 (10.60 times).
Lumberton R precision is 4.27%, recall74.84%; its false-positive problem is evident
from precision, but per-event confusion counts are not in these training logs.
Foreground prevalence is only1.076% over valid validation pixels; high overall
accuracy is not evidence of useful change detection in this imbalanced setting.

## Training diagnostics: observations, not a causal explanation

R's event-macro F1 peaks at epoch1 (.6671), then varies between .4782 and .6300
over epochs2–14. Its training loss declines overall from .4367 to .2297; lower
training loss is not evidence of improved validation performance. A and R totals
are different objectives and should not be compared as if they were the same loss.

At epoch14, R local loss averaged over all minibatches is .4930, applied term
.04930, and active-batch local mean1.0768. A's diagnostic local loss is computed
but has coefficient0 and is not optimized. Approximately45.78% of epoch14 batches
are active, consistent with the audit, so sparse auxiliary exposure is not an
implementation discrepancy detected by this report.

R's recorded mean applied-auxiliary/base **logit** gradient ratio ranges6.94–8.48
over the15 epochs. At last the clipping fraction is45.08% for R versus29.63% for A.
This supports inspecting optimization interactions, but does not establish that
gradient dominance, clipping, batch normalization or any other mechanism caused
the performance difference. These are means of per-batch ratios, not parameter
gradient shares or ratios of epoch-mean norms. No mechanism is diagnosed here.

Reported elapsed time: A7,108.48s, R6,723.59s, combined3.84h. Peak allocated CUDA
memory: A.4757GiB, R.4629GiB; peak reserved .5176GiB for each. Times include training,
validation and diagnostics on a shared server; do not infer a speed advantage or
Jetson/mobile deployment compliance from these figures.

## Statistical/methodological fallacy screen — 11/11 considered

1. Simpson's paradox: examined pooled and per-event results; no all-event reversal
   in best F1. Last effects differ by event; aggregate alone is insufficient.
2. Ecological inference: pixel F1 is not small-component recall or physical asset
   damage; neither is inferred here.
3. Selection/Berkson bias: fixed development split, three validation events;
   generalization beyond these data unverified. No correlation test performed.
4. Collider bias: no adjusted causal model; not applicable to this descriptive audit.
5. Base-rate neglect: addressed by reporting foreground prevalence, precision,
   recall and confusion counts rather than accuracy alone.
6. Regression to mean: compare fresh paired control, not an extreme historical
   checkpoint; no repeated-seed uncertainty estimate available.
7. Survivorship bias: both prescribed completed runs and all15 epochs supplied;
   prior negative factorial findings remain part of project evidence.
8. Look-elsewhere: no p-values or significance claims; keep best selected by frozen
   macro F1 and last, not a retrospective small-recall winner.
9. Forking paths: frozen controls checked; further lambda/radius/epoch/seed tuning
   would be a new, separately approved exploratory design, not this pilot.
10. Causal overclaim: one controlled seed is insufficient to establish a general
    mechanism for R's failure; gradient observations are not causal proof.
11. Reverse causality: no observational directional model assessed; not applicable.

## Proposed next gate

Subject to approval, implement and test a validation-only wrapper to load exactly
the recorded best and last checkpoints for A-control and R, verify checkpoint
hashes/epochs and pixel counts against these logs, and report component precision,
interior/clipped-small recall, boundary metrics and per-event denominators. No
optimizer steps, new checkpoints, test arrays or threshold tuning. Four evaluations
only; retain results even if they confirm failure. After this closure, reconsider
the research direction separately instead of promoting R or launching more seeds.
