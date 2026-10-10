# Efficiency protocol v1 (A3): cost set for the lightweight benchmark

Protocol ID: `efficiency_v1`.
Drafted 2026-10-10 for paper direction A + C (`docs/paper_direction_decision_20261009.md`),
following the cost set recommended in `docs/literature_review_direction_b_20261003.md` §8.
Approved 2026-10-10 (user reply "duyệt cả cả lựa chọn!"). All six §10 choices were approved
as drafted: no TensorRT, ONNX Runtime CPU with 1 and 4 threads, `FlopCounterMode`, the
100/500/3 and 20/100/3 timing protocols, validation-only fidelity on the four pre-specified
checkpoints, and the load-handling rules.
**Status: protocol approved and frozen; engineering and measurement runs NOT yet approved.**

## 1. Purpose and scope

Report the computational cost of the four architectures in the benchmark under one stated
measurement protocol, with an "edge-oriented efficiency, deployment deferred" framing.

**Not covered:** Jetson Orin measurements, TensorRT, any accuracy claim, training, the test
set (already used once and closed), and changes to the frozen deployment YAML
`configs/deployment/jetson_orin_nano.yaml`. That YAML stays an **unmet, deferred** target.

## 2. Architectures

Cost depends on architecture, not on trained weights, except for the fidelity checks (§5).

| Architecture | Arms | Parameters |
|---|---|---:|
| SmallFlood-CDNet, no boundary head | A, B | 1,079,865 |
| SmallFlood-CDNet, boundary head | C, D | 1,080,805 |
| FC-Siam-Diff | FC | 487,857 |
| BIT-SAR v2 | BIT | 3,492,642 |

Each architecture is built from its frozen config. Inference input: pre and post tensors
of shape [1, 2, 256, 256] (batch 1 unless stated). Output: change logits only, as in
`src/smallflood_cd/deployment/onnx_export.py`.

**Representative checkpoints for §5** (pre-specified, validation only): the R1 seed-201
`last.ckpt` of A, C, FC-Siam-Diff and BIT-SAR v2 from
`runs/recipe_confirmation_v1/20261007T234436Z`.

## 3. Environment (verified read-only on 2026-10-10)

- GPU: NVIDIA RTX A6000, driver 560.35.03; torch 2.5.1+cu124, cuDNN 90100.
- CPU: Intel Xeon Silver 4310 @ 2.10 GHz, 24 cores visible, 2 NUMA nodes, AVX-512.
- onnx 1.23.0; onnxruntime 1.23.2 with **CPU execution provider only**.
- **TensorRT is not installed.** Under the frozen-stack rule nothing is installed, so
  TensorRT FP16 is deferred.
- The server is shared (about 20 logged-in users). §6 sets how load is handled.

## 4. Measurements

All in `torch.inference_mode()`, model in `eval()`.

| Quantity | Method |
|---|---|
| Parameters | Exact count of `model.parameters()`. Must equal §2. |
| FLOPs and MACs | `torch.utils.flop_counter.FlopCounterMode`, batch 1, FP32. MACs = FLOPs / 2. **It counts convolutions and matrix multiplications only**; normalization, activations, softmax and element-wise operations are not counted. Stated in every table. |
| Serialized size | Bytes of the saved `state_dict` in FP32 and FP16 (`torch.save`), and of the FP32 ONNX file (opset 17, existing exporter). Reported in MB (10^6 bytes). |
| Peak GPU memory | `torch.cuda.max_memory_allocated` during one batch-1 forward pass after reset, FP32 and FP16. Reported in total and minus parameter memory (activation estimate). |
| GPU latency | PyTorch, FP32 and FP16 (`model.half()`, half inputs), batch 1. CUDA events around each forward pass, with synchronization. Warmup 100, timed 500, 3 repetitions (the frozen deployment protocol). Median and 90th percentile per repetition; the reported value is the median of the three medians. |
| GPU throughput | Batch 8, FP32 and FP16: patch pairs per second from 3 repetitions of 100 timed batches after 20 warmup batches. |
| CPU latency | ONNX Runtime, CPU provider, FP32, batch 1, `intra_op_num_threads` 1 and 4, `inter_op_num_threads` 1, sequential execution. Warmup 20, timed 100, 3 repetitions. Same reporting as GPU latency. |

**Inference settings for timing:** `cudnn.benchmark = True`, deterministic algorithms off,
TF32 off (so FP32 means FP32), no gradient. These are inference settings, not the training
determinism settings. Model size budget (≤ 20 MB) is checked against the FP32 and FP16
`state_dict` sizes; that is the only frozen Orin budget that is hardware-independent.

## 5. Fidelity checks (validation split only)

On the full validation split (4,767 patches), for each representative checkpoint:

1. **FP16 vs FP32 (PyTorch, GPU):** pixel agreement of thresholded masks (threshold 0.5) over
   valid pixels; maximum absolute logit difference; and the change in every review metric
   (event-macro F1, per-event F1, component F1, interior-small recall, boundary F1, inner-band
   IoU) computed with the same functions as the validation reviews.
2. **ONNX Runtime CPU vs PyTorch FP32:** the same quantities on the first 64 validation
   patches in review order.

Fidelity results are **reported, not thresholded**. The FP32 PyTorch metrics must reproduce
the existing validation report of that checkpoint exactly; if not, stop.

## 6. Load handling and stop rules

- Before each GPU repetition, `nvidia-smi` must show no other compute process. If one is
  present, wait up to 30 minutes, checking every minute; if it persists, stop and report.
  Each repetition records the load average and GPU state.
- CPU timing records the load average before and after each repetition. CPU results are
  reported with that load; they are not filtered or repeated after the fact.
- **Stop** on: a parameter-count mismatch; an FP32 metric that does not reproduce its
  validation report; a nonfinite output; any test-split access; any training call.

## 7. Outputs

`artifacts/efficiency_v1/<UTC>/`: `costs.csv` (one row per architecture, precision and
device), `latency_raw/` (every timed value), `fidelity.json`, `environment.json`
(versions, `nvidia-smi`, `lscpu`, load records) and `COMPLETE.json`. A closure document then
reports the cost table next to the A4 test accuracy table.

## 8. Claims this protocol supports and does not support

**Supports:** parameters, counted FLOPs/MACs, serialized size, peak GPU memory, and measured
latency/throughput on the stated A6000 and Xeon under the stated settings; FP16 and ONNX
fidelity on validation; compliance with the hardware-independent 20 MB size budget.

**Does not support:** any Jetson Orin latency or memory claim, any extrapolation from A6000 or
Xeon numbers to Orin, "real-time" or "onboard" statements, or TensorRT results.

## 9. Cost

Measurements: under 1.5 h in total (BIT-SAR v2 with 1 CPU thread is the slowest part).
Engineering: about 1 day. Storage: small (raw timings, ONNX files of a few MB).

## 10. Choices approved on 2026-10-10 (as drafted)

1. **No TensorRT** (not installed; nothing is installed): FP16 measured with PyTorch on the
   A6000; TensorRT FP16 deferred.
2. **CPU point:** ONNX Runtime FP32 with 1 and 4 threads.
3. **FLOP counter:** `FlopCounterMode` (convolutions and matrix multiplications only),
   disclosed in every table.
4. **Timing protocol:** GPU 100/500/3 (frozen deployment values), CPU 20/100/3, inference
   settings as in §4.
5. **Fidelity:** on validation only, with the four pre-specified R1 seed-201 last
   checkpoints; reported without pass/fail thresholds.
6. **Load handling:** GPU must be free of other compute processes; CPU load recorded, not
   controlled.

## 11. Amendment 1 (approved 2026-10-10, before any timing was recorded)

**Reason.** The first measurement attempt (2026-10-10T10:56Z) stopped under §6 after 30
minutes: another user's idle Jupyter kernel held about 600 MiB on the GPU at 0% utilization.
Such kernels can stay for days. No latency, throughput or memory value had been recorded.
The user approved relaxing the rule ("Duyệt cách 3, nới quy tắc và chạy đo").

**Amended §6 GPU rule.** Before each GPU repetition:
- if no other compute process is present, proceed (unchanged);
- if other compute processes are present, proceed **only if** their total GPU memory is at
  most 1,024 MiB **and** five GPU-utilization samples taken one second apart, with this
  process idle, are all at most 1%;
- otherwise wait as before (checks every minute, stop after 30 minutes).

Every repetition records the other processes, their memory and the utilization samples, and
the cost table marks rows measured while an idle process was present. Everything else in the
protocol is unchanged. Timings are never re-run to obtain better values.
