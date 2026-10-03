# BIT-SAR v2 GPU smoke and repeated-run check

## Material Passport

Stage: preparation and local CPU synthetic integration tests. Actual CUDA run pending user.
This is not model comparison, convergence evidence, test evaluation or a deployment benchmark.
No original model, pilot launcher, dataset or checkpoint is modified by this update.

## Fixed protocol

- Model bit_sar_v2 only; configuration configs/experiment/bit_sar_v2.yaml.
- Two separate Python processes, fresh seed 42, 16 train and 8 validation patches.
- Half positive/half negative subsets, selected reproducibly by existing smoke selector.
- Batch size 2, one epoch, 8 optimizer steps, FP32, zero workers, no scheduler.
- strict_seed with CUBLAS_WORKSPACE_CONFIG=:4096:8 and PYTHONHASHSEED=42 before Python.
- Every retained parameter must have finite gradients; aggregate norm nonzero.
- Verify finite full-resolution logits, finite loss, parameter changes and optimizer steps.
- Select checkpoint with event_pixel_v2; exact output equality on checkpoint reload,
  optimizer state round-trip. No bit legacy checkpoint can be used as initialization.
- Record source and manifest hashes, band metadata, subset IDs, software/GPU information,
  training+validation duration and peak PyTorch allocated/reserved memory.
- Compare config, environment, subset, checks and metrics, full best/last checkpoint state,
  and CPU/CUDA RNG states at end of training. Exclude timing/memory from exact-match criteria.
- PASS applies only to these two smoke runs on this stack, not all future training.

## Run manually on server

From /data/fit.lamnv/smallflood_cd, after installing the preceding bit_sar_v2 update,
inside tmux, run `bash scripts/run_bit_sar_v2_smoke.sh`.
The launcher now invokes bit_sar_v2_entry.py, which preloads the validated src package
before any smoke helpers. It prints source paths and rejects stale loaded modules instead
of deleting/replacing sys.modules entries. Run CPU checks using
`python scripts/bit_sar_v2_entry.py test` (eight tests; one requires CUDA), not direct pytest in the conflicting
server checkout. Both GPU passes and comparison use this same entry point.
This launches a then b then compare; stops immediately on any failure, no automatic retry,
no CPU fallback, no batch-size fallback. Do not weaken determinism automatically if an
unsupported deterministic CUDA operation raises an error; bring the traceback for diagnosis.

New paths: runs/bit_sar_v2_smoke/<UTC>/{a,b} and artifacts/bit_sar_v2_smoke/<UTC>.
Existing output directories are rejected. Logs are a.log, b.log, compare.log; each run
stores config_resolved.yaml, environment.json, subset.json, metrics.jsonl, checks.json,
summary.json, rng_after_training.pt, COMPLETE.json and best/last checkpoints.
Checkpoints are trusted local artifacts; do not use compare on arbitrary untrusted .ckpt files.

Memory/time values cover training and validation, before constructing the second model for
reload. They are not total process peak or Orin/TensorRT performance. Batch 2 smoke success
does not establish batch 8 feasibility. Balanced subsets do not reflect dataset prevalence.

## Local tests

tests/unit/test_bit_sar_v2_smoke.py runs two bounded CPU training passes on synthetic
32×32 tensors through real BIT-SAR/trainer, verifies exact repeatability, no test split access,
overwrite refusal, self-comparison refusal, missing-gradient rejection and deliberate
log/checkpoint corruption detection. This does not replace the server CUDA check.

## Optimizer comparison correction

The first server run 20260924T091305Z completed eight optimizer steps and passed the
checkpoint-output check, then failed at optimizer comparison due to CPU/CUDA tensor
placement. It is an incomplete smoke run, not a reproducibility PASS. Both nested states
are now detached and moved to CPU before exact comparison, without dtype conversion or
tolerance. Checkpoint files and live optimizers are not modified by normalization. A CUDA
regression test covers mixed-device states and rejects changed values. Retain the failed
run; rerun both a/b fresh into the launcher's new timestamp after tests pass.
