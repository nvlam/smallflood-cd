# efficiency_v1 (A3): cost results and closure

## Material Passport

Protocol: `docs/efficiency_protocol_v1_20261010.md` (commit `02c91a9`) with **amendment 1**
(commit `a035f56`, approved by the user on 2026-10-10 before any timing was recorded).
Script, tests and launcher: commits `faa23e4`–`a035f56`, copied to the server by
sha256-verified file transfer (`scripts/efficiency.py` SHA256 `6e0dc4c8…d337`).
Execution: `artifacts/efficiency_v1/20261010T134555Z` (`costs.csv`, `latency_raw/`,
`fidelity.json`, `environment.json`, `COMPLETE.json`), logs
`artifacts/efficiency_v1_logs/20261010T134555Z`. Measured 2026-10-10T13:46Z–14:00Z; fidelity
14:00Z–14:14Z.
Scope: four architectures; validation split only for fidelity; no training; test split not
read.
Status: **closed.** All values hold for the stated RTX A6000 and Xeon Silver 4310 under the
stated settings. Nothing here is a Jetson Orin measurement.

## 1. Execution and integrity

- Synthetic tests: 8 passed on the server. Smoke stage (no timing kept) passed for all four
  architectures.
- `COMPLETE.json`: 16 cost rows; `fidelity_split: validation`; `test_used: false`;
  `training_performed: false`.
- Each representative checkpoint (R1, seed 201, `last.ckpt` of A, C, FC-Siam-Diff,
  BIT-SAR v2) matched its validation review SHA256, and its parameter count matched the
  protocol.
- Environment: torch 2.5.1+cu124, cuDNN 90100, RTX A6000 (driver 560.35.03); onnx 1.23.0,
  onnxruntime 1.23.2 (CPU provider); Intel Xeon Silver 4310 @ 2.10 GHz.
- **Attempts.** Three launches preceded this run; none recorded a timing:
  1. 10:51Z stopped at the synthetic tests (FLOP counter returned zero, §6.1);
  2. 10:52Z stopped at the smoke stage (smoke-only ordering bug, §6.2);
  3. 10:56Z passed tests and smoke, then stopped under the original §6 rule after waiting 30
     minutes for another user's idle GPU process. Its directory
     `artifacts/efficiency_v1/20261010T105621Z` holds no cost data.
- **Load during measurement (amendment 1).** All 48 GPU repetition checks were accepted with
  one other compute process present (314 MiB, another user) and all utilization samples at
  0%. Load average was 0.4–2.4 during GPU timing and 0.4–3.0 during CPU timing. Every GPU row
  is flagged `idle_other_gpu_process_present`.

## 2. Cost table

Input: one pre/post pair, each [1, 2, 256, 256]. FLOPs and MACs come from
`FlopCounterMode` and **count convolutions and matrix multiplications only**. MB = 10^6 bytes.

### 2.1 Size and operation counts

| Architecture | Arms | Parameters | FLOPs | MACs | `state_dict` FP32 | `state_dict` FP16 | ONNX FP32 | ≤ 20 MB |
|---|---|---:|---:|---:|---:|---:|---:|---|
| SmallFlood-CDNet | A, B | 1,079,865 | 0.346 G | 0.173 G | 4.48 MB | 2.29 MB | 4.38 MB | yes |
| SmallFlood-CDNet + boundary head | C, D | 1,080,805 | 0.353 G | 0.176 G | 4.50 MB | 2.31 MB | 4.39 MB | yes |
| FC-Siam-Diff | FC | 487,857 | 7.023 G | 3.512 G | 1.98 MB | 1.00 MB | 1.97 MB | yes |
| BIT-SAR v2 | BIT | 3,492,642 | 21.434 G | 10.717 G | 14.06 MB | 7.07 MB | 14.46 MB | yes |

### 2.2 GPU (PyTorch, RTX A6000)

Latency: batch 1, warmup 100, timed 500, 3 repetitions; median of repetition medians (median
of repetition 90th percentiles in brackets). Throughput: batch 8, pairs per second, median of
3 repetitions. Peak memory: `max_memory_allocated` for one batch-1 forward (minus parameter
memory in brackets).

| Architecture | Latency FP32 (ms) | Latency FP16 (ms) | Throughput FP32 | Throughput FP16 | Peak memory FP32 (MB) | Peak memory FP16 (MB) |
|---|---:|---:|---:|---:|---:|---:|
| SmallFlood-CDNet | 8.70 (9.91) | 10.54 (10.87) | 708 | 566 | 10.0 (5.6) | 5.1 (2.9) |
| SmallFlood-CDNet + boundary head | 9.17 (11.92) | 10.78 (11.11) | 658 | 489 | 10.0 (5.6) | 5.1 (2.9) |
| FC-Siam-Diff | 2.98 (3.05) | 3.84 (3.90) | 1,020 | 1,662 | 56.0 (54.1) | 34.3 (33.3) |
| BIT-SAR v2 | 24.51 (25.23) | 25.66 (26.58) | 233 | 291 | 83.4 (69.4) | 55.3 (48.3) |

### 2.3 CPU (ONNX Runtime FP32, Xeon Silver 4310)

Batch 1, warmup 20, timed 100, 3 repetitions; median of repetition medians (90th percentile in
brackets), milliseconds.

| Architecture | 1 thread | 4 threads |
|---|---:|---:|
| SmallFlood-CDNet | 10.69 (11.12) | 7.49 (7.66) |
| SmallFlood-CDNet + boundary head | 10.65 (11.02) | 8.63 (8.69) |
| FC-Siam-Diff | 81.44 (82.67) | 26.01 (29.47) |
| BIT-SAR v2 | 350.89 (356.51) | 163.29 (166.92) |

## 3. Fidelity (validation split, 4,767 patches)

FP32 PyTorch reproduced each checkpoint's existing validation report exactly.

### 3.1 FP16 vs FP32 (PyTorch, GPU)

| Architecture | Mask agreement | Disagreeing pixels (of 295,762,355) | Max abs logit difference | Largest metric change (pp) |
|---|---:|---:|---:|---|
| SmallFlood-CDNet | 99.9889% | 32,852 | 3.21 | global pixel F1 +0.18 |
| SmallFlood-CDNet + boundary head | 99.9982% | 5,469 | 3.88 | event-macro F1 +0.10 |
| FC-Siam-Diff | 99.9973% | 7,846 | 0.29 | global pixel F1 +0.04 |
| BIT-SAR v2 | 99.9995% | 1,465 | 0.24 | component F1 +0.03 |

Interior-small recall did not change for any architecture. All other metric changes (event-macro
F1, global pixel F1, component F1, boundary F1, inner-band IoU) are at most 0.18 pp in
absolute value.

### 3.2 ONNX Runtime CPU vs PyTorch FP32 (first 64 validation patches)

Mask agreement is 100% for all four architectures (0 of 4,194,304 pixels differ). Maximum
absolute logit difference: SmallFlood-CDNet 7.2e-3, with boundary head 6.8e-2, FC-Siam-Diff
6.4e-5, BIT-SAR v2 4.8e-5. Event-macro F1, pixel F1, component F1, boundary F1 and inner-band
IoU are identical. Interior-small recall is **not evaluable** on these 64 patches (no
interior-small component among them).

## 4. Observations (descriptive)

1. **Parameter count does not track computation.** FC-Siam-Diff has the fewest parameters but
   about 20× the counted FLOPs of SmallFlood-CDNet; BIT-SAR v2 has about 62×.
2. **The efficiency ranking depends on the device.** On the GPU at batch 1, FC-Siam-Diff is
   fastest (3.0 ms), SmallFlood-CDNet is about 2.9× slower despite 20× fewer FLOPs, and
   BIT-SAR v2 about 8× slower than FC-Siam-Diff. On one CPU thread, SmallFlood-CDNet is fastest
   (10.7 ms): FC-Siam-Diff is 7.6× and BIT-SAR v2 32.8× slower. With four threads the ratios
   are 3.5× and 21.8×.
3. **FP16 is not faster at batch 1 on this GPU** for any architecture (0.9–1.8 ms slower). At
   batch 8 it raises throughput for FC-Siam-Diff (1.6×) and BIT-SAR v2 (1.25×) but not for
   SmallFlood-CDNet. FP16 does roughly halve serialized size and reduce peak memory.
4. **Timing noise.** The server was shared. Some repetitions differ visibly, for example
   SmallFlood-CDNet FP16 throughput 566 / 512 / 757 pairs per second and FC-Siam-Diff FP16
   latency medians 3.90 / 2.59 / 3.84 ms. Values are reported as measured; nothing was
   re-run.
5. Read with the A4 test table (`docs/heldout_test_results_20261010.md` §6): the architecture
   that is cheapest on CPU (SmallFlood-CDNet) is the least accurate, and the most accurate
   and seed-stable (BIT-SAR v2) is the most expensive on every measure.

## 5. Claims supported and not supported

**Supported:** the values in §2–3 for the stated hardware, software and settings; that all
four models meet the hardware-independent 20 MB size budget of the frozen deployment YAML;
that FP16 and ONNX conversion preserve the reviewed validation metrics to within 0.18 pp.

**Not supported:** any Jetson Orin latency or memory statement; extrapolation from A6000 or
Xeon numbers to Orin; "real-time" or "onboard" claims; TensorRT results (not installed, not
measured); total operation counts including normalization, activations or attention softmax.

## 6. Engineering findings and deviations

1. **FLOP counter under `inference_mode`.** On torch 2.5.1 `FlopCounterMode` returns zero
   inside `torch.inference_mode()`. Counting now runs under `no_grad` and fails on a zero
   count. Caught by the server synthetic test.
2. **Smoke-stage ordering.** The first smoke implementation counted FLOPs after an FP16
   conversion inside `inference_mode`. It affected only the smoke stage and was fixed.
3. **`export_onnx` leaves the model in training mode.** `torch.onnx.export` restores the
   freshly created wrapper's default (training) mode, which puts the wrapped model in training
   mode after export. The exported graph itself is in eval mode and matches PyTorch (§3.2).
   `src/smallflood_cd/deployment/onnx_export.py` is frozen source and was **not** changed;
   `scripts/efficiency.py` restores eval after export and refuses to measure any model with a
   module in training mode. Anyone reusing a model object after `export_onnx` must call
   `eval()` first.
4. **Amendment 1** (GPU co-tenant rule), as recorded in the protocol §11.
5. Results remain on the server; no output tree was copied into this checkout.

## 7. Next steps (require user approval)

- Paper outline for direction A + C, combining this cost table with the A4 test results.
- Optional and separate: TensorRT FP16 or Jetson Orin measurements would need an installation
  or hardware decision and a new protocol.
