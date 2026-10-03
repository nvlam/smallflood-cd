# Candidate R runner — handoff, 2026-10-02

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: approved implementation and bounded software verification.
- Version Label: candidate_r_runner_v1.
- Verification Status: local synthetic checks completed; real-data GPU preflight
  and full pilots NOT executed by this handoff. No claim of scientific improvement.
- Scope: the previously approved isolated A-control/R comparison. Historical
  factorial code, losses, model, dataset splits and result files remain unchanged.

## Where the project stands

Candidate R loss and train-only eligibility audit were already completed. The user
reported 19,166 train patches, 1,420 with eligible components, 3,521 eligible
components, 95,594 eligible foreground pixels and zero empty rings. Approximately
53–55% of batch-8 minibatches have no eligible component. This is expected: the
global loss remains active; no sampler, radius or coefficient adjustment is made.

This package adds the missing runner, two configurations, tests and a bounded
preflight launcher. It does not run or evaluate the held-out test split. Reading
the frozen manifest's hash is not access to test arrays. It does not repeat any
baseline training or the 135-experiment matrix.

## Local verification performed

- `python scripts/candidate_r_runner.py test`: **37 passed, 2 skipped** (5.44s).
  The two skips require CUDA, unavailable on this Mac.
- Focused regression: factorial configurations/runner, validation protocol v2,
  and base loss: **12 passed** (12.09s).
- `bash -n scripts/run_candidate_r_preflight.sh`: passed.
- Synthetic A/R/R-repeat runs used the actual model on 32×32 generated inputs;
  this is not a dataset experiment or a device benchmark. R repeat model/optimizer
  checkpoints matched exactly; A-control matched the old A optimizer trajectory
  in this bounded check. No real training or validation arrays were loaded.
- The server environment is different; server tests and GPU preflight are still
  mandatory. These results do not establish performance or 15-epoch reproducibility.

## Frozen comparison

| Item | A-control | R |
|---|---|---|
| Backbone | A's shared MobileNetV3-small, fresh initialization | identical |
| Parameters | 1,079,865 | 1,079,865 |
| Boundary head / old size weights | OFF / OFF | OFF / OFF |
| Global loss | .4 BCE + .4 Tversky | identical |
| Applied local loss coefficient | 0 | .1 |
| Local eligibility/ring | audited 8-connected interior components, area ≤35; radius3 | identical |
| Full pilot | 15 epochs, seed42, batch8, FP32 | identical |

AdamW LR .0003, weight decay .0001, clipping1, Tversky FP .3/FN .7,
workers0, no scheduler, early stopping, augmentation or sampling changes. RNG is
reset after model construction; the independent DataLoader generator is seeded
once, not once per epoch. Initial full-state hashes and all epoch-order hashes
must match. Every full-run epoch is checked against the audited metadata schedule.

The local loss is computed on both branches for diagnostic comparability, but is
**not applied to A-control**. The auxiliary has no parameters, consumes no RNG,
and uses detached train labels only. A synthetic regression test compares the
new A-control's model, optimizer and RNG state exactly with the old factorial A
code after two updates. This is a bounded software test, not proof of full-run
reproducibility on the server.

## Checks and output files

- Source pinning rejects the old shadow package outside `src/smallflood_cd`.
- Frozen manifest/component/normalization hashes are required. The audit report
  and per-patch counts must agree with the reported approved totals and current
  loss/audit source hashes. Actual eligible component counts are checked per train
  patch during execution; no cache is used.
- These guards do not rehash every NPY input or prove every SAR raster's band
  semantics. Keep the prepared dataset immutable. Band provenance retains its
  existing metadata-level scope.
- No existing output directory is reused. There is no auto-resume or retry.
- Each cell saves resolved YAML, `metrics.jsonl`, `environment.json`, `summary.json`,
  `checkpoints/last.ckpt` and `checkpoints/best_composite.ckpt`.
- `best_composite.ckpt` remains the legacy filename for best **event-macro pixel
  F1 at threshold .5**; it is not a composite score. Ties keep the earlier epoch.
- Epochs are zero-indexed: 0–14 for a complete 15-epoch pilot.
- Logs include raw/scaled losses, active-batch fraction, eligible counts, empty
  rings, pre-clipping parameter norm, clipped-step fraction, timing and CUDA peak
  allocated/reserved memory. The saved cycle time includes diagnostics and
  validation/checkpoint overhead; this is not a deployment latency benchmark.
- Gradient diagnostics measure L2 norms of derivatives with respect to the
  **output logits**: global loss, unscaled local loss, and applied local loss.
  Ratios are means of per-batch ratios, not ratios of epoch means. They do not
  measure per-parameter gradient shares, equal influence, or causal effectiveness.
  Empty-eligible batches have exactly zero local loss/gradient. `autograd.grad`
  diagnostics do not populate parameter `.grad`; only the chosen total is backpropagated.
- Root `COMPLETE.json` is written only after all prescribed cells, checkpoint
  checks, pairing and unchanged-source checks pass. A crash leaves incomplete
  files for inspection, not a success marker.

## Server prerequisites and step 1: install the additive package and test

No sudo and no dependency upgrade are needed if the earlier loss/audit tests
passed. Existing torch, torchvision, NumPy, SciPy, PyYAML and pytest are required.
The original `pyproject.toml` does not declare SciPy; this package relies on the
already-tested environment and reports its version instead of changing it.
It also relies on the existing factorial scripts/configs and source import guard.

From the **Mac terminal**:

```bash
scp /Users/nguyenvulam/academic-research-skills/academic-research-skills-codex/smallflood-cd/artifacts/updates/candidate_r_runner_20261002.tar.gz fit.lamnv@100.102.158.114:/data/fit.lamnv/smallflood_cd/
```

From the **server SSH terminal**:

```bash
cd /data/fit.lamnv/smallflood_cd
source .venv/bin/activate
tar -xzf candidate_r_runner_20261002.tar.gz
mkdir -p artifacts/candidate_r_runner
set -o pipefail
python scripts/candidate_r_runner.py test 2>&1 | tee artifacts/candidate_r_runner/tests_20261002.log
```

**Stop and send the test output before the next step.** The tests use synthetic
inputs only. Two CUDA-specific tests are skipped on the CPU-only Mac; the GPU
server should exercise them. A passing test suite does not yet authorize a full
pilot. If source imports are wrong or any test fails, stop; do not fall back to
the generic training command.

## Step 2: bounded GPU preflight (manual, after reviewing step 1)

```bash
cd /data/fit.lamnv/smallflood_cd
bash scripts/run_candidate_r_preflight.sh
```

The default audit directory is
`artifacts/candidate_r_eligibility/20260928T134626505807Z`; pass a different directory
as the first argument only if that same completed audit was relocated.

This performs **A-control, R, and an exact R repeat**, each one epoch of only 16
train and eight validation patches: two optimizer steps each, six steps total.
The train subset contains eight audited eligible and eight empty patches to ensure
the auxiliary is exercised. The unchanged seeded sampler may put active patches
in both batches; the all-empty branch is independently covered by unit tests.
Subset IDs are saved. Full training will use the original complete train set with
no enrichment. Repeated R diagnostics (excluding wall time/memory), full model,
optimizer and RNG checkpoint contents must match exactly. This establishes
reproducibility for these two bounded runs only.

Output: `runs/candidate_r_preflight/<UTC>/`; log:
`artifacts/candidate_r_preflight/<UTC>/preflight.log`. Share the final completion
message and summaries. Do not rerun merely because training metrics look poor.

## Step 3: full pilot remains behind an approval gate

The runner's `run` command requires both `--approve-training` and a completed
`--preflight`, plus `--audit` and a fresh `--output`. It refuses a stale preflight
if source/config/test hashes, audit evidence, frozen metadata or recorded runtime
change. Do not upgrade dependencies between preflight and pilot.

Full scope is exactly **two fresh runs**, A-control then R, 15 epochs each. A
preflight checkpoint is never used as full-run initialization. Both have 35,940
optimizer steps. No held-out test evaluation is invoked. This handoff has not
started those runs; obtain the next launch approval after reviewing preflight.
After training, review best and last for both with the frozen object/boundary
metrics. The existing factorial A/B/C/D review launcher does not accept this new
layout; a separately checked A-control/R review wrapper is still needed.

## Scientific decision remains unchanged

Use the predeclared screening criteria in
`docs/proposed_bounded_redesign_plan_20260928.md`: at best selected checkpoint,
R–A interior-small recall ≥5 percentage points, event-macro F1 loss ≤1 point,
overall component precision loss ≤2 points, and no small-recall decrease in either
defined validation event; at last, pooled small-recall difference must be
nonnegative. Verify denominators before comparison. No checkpoint picking by small
recall, no parameter sweep, no significance claims from this single seed.

Passing only motivates separately approved repeated-seed confirmation and prior-art
assessment; it does not establish a novel loss or a publishable result. Failing
means stopping this candidate under the bounded plan, not silently tuning it.
