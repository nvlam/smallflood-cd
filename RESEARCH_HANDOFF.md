# SmallFlood-CD research handoff

Prepared 2026-10-03 (Asia/Ho_Chi_Minh) for continuation in Claude Code.

**Continue this existing project. Do not restart it, redesign it, or reinterpret negative pilots as successes.** The original size-aware/boundary claim is not supported by the current development pilots. The bounded Candidate R comparison has completed according to the latest audit and failed its frozen event-F1 screening criterion. The immediate continuation is evidence preservation and validation-only closure, not more training.

## Evidence scope and how to read this file

- **Verified locally:** current tracked source/configuration/documentation, Git metadata, local dependency package metadata, and available imported JSON/CSV/JSONL/log records. “Verified” here does not mean independent inference reproduction.
- **Recorded:** facts in dated repository audits about remote experiments or user-supplied evidence whose original files are not in this checkout.
- **Unverified:** current server filesystem/process state, absent raw/prepared data, absent checkpoint binaries, hardware performance, and external-source claims not independently revisited during this handoff.
- **Recommendation:** an ordered continuation step, not a new research decision or authorization to launch work.

Inspection covered the tracked repository (186 files before this handoff), README, all 29 documentation files, source/model/metric/loss/data/engine/deployment code, scripts, all configuration families, 36 unit-test files and pinned BIT reference sources, available ignored imported results/logs, update-archive member inventories, local virtual-environment metadata, and all available Git history/reflog. Python files were statically parsed; imported summaries and epoch logs were read. Third-party `.venv` package implementations and generated caches were treated as environment artifacts. Preview PNGs were inventoried; no new visual-quality audit was performed. Update archives were inspected without installing or extracting them over the project.

No training, evaluation, tests, dependency installation, export, remote access, or external literature verification was run for this handoff. The only newly created project file is this document. Local process inspection via `ps` was denied by the execution sandbox, so active jobs cannot be ruled out. No remote process inspection was attempted.

Historical documentation is a sequence of snapshots, not a current task queue. Later audits and current code supersede earlier statements such as “bands unverified,” “v2 pilot pending,” “factorial review wrapper needed,” or “Candidate R runner/review wrapper missing.” Preserve the old documents as provenance.

## 1. Research objective and research question

The README defines lightweight, small-region-aware urban flood change detection from paired SAR inputs, with a sub-5M-parameter Siamese model, size-aware supervision, thin boundary refinement, event-held-out validation and TensorRT FP16 deployment on Jetson Orin Nano.

Research question, synthesized from that contract and the frozen contribution/factorial documents: **Can a lightweight Siamese SAR change detector improve detection of small labeled flood regions and boundary localization while retaining useful overall segmentation quality and meeting edge deployment constraints?** This wording is a synthesis, not a verbatim preregistered title or formal hypothesis statement found in the repository.

The task is binary flood extent/change labeling: original nonflood=0 versus flood-open/flood-urban combined. It is not building/road structural-damage detection, physical flood-object census, or an urban-only FU segmentation benchmark.

## 2. Frozen research claim / hypothesis and present verdict

`docs/research_contributions.md` freezes five methodological mappings: cached connected components; capped inverse-size weights; positive-weighted Tversky; morphological boundary supervision with lightweight residual refinement; supplied uncertain-border masking.

The original intended claim is that size-aware supervision plus boundary refinement improves small-region detection/boundaries within a lightweight deployable model. **It remains a hypothesis, not an established result.** The factorial validation audit explicitly says the strong small-region improvement claim is unsupported. Preserve that finding.

Candidate R is an already implemented, separately bounded development candidate using A's architecture, not a replacement research claim or a novel-loss result. Its frozen engineering gate requires all of:

1. Best-selected pooled interior-small recall R−A ≥5 percentage points (≥19 extra matches if denominator stays 370).
2. Best-selected event-macro pixel F1 decrease ≤1 percentage point.
3. Best-selected overall component precision decrease ≤2 percentage points.
4. No interior-small recall decrease in either defined validation event.
5. Last-checkpoint pooled interior-small recall R−A ≥0.

Recorded best event F1 is A-control 0.8284351358718937 versus R 0.6670670929241197: **−16.1368 points**, already failing criterion 2. Missing small/boundary scores cannot rescue a conjunctive gate. No coefficient/radius/epoch/seed search is justified under the bounded plan. Prior-art overlap with blob/ICI losses is documented; novelty remains unverified.

Sources: `docs/proposed_factorial_validation_audit_20260928.md`, `docs/proposed_bounded_redesign_plan_20260928.md`, `docs/candidate_r_training_audit_20261003.md`.

## 3. Dataset and exact locations

Dataset: **UrbanSARFloods**. Repository-recorded primary source: https://github.com/jie666-6/UrbanSARFloods . The finalized release is documented as `urban_sar_floods.tar.gz`; earlier `train_validation.rar` is not the intended release. These external claims were recorded by earlier source audits, not reverified online here.

| Resource | Exact recorded location | Availability in this inspection |
|---|---|---|
| Current local project | `/Users/nguyenvulam/academic-research-skills/smallflood-cd` | Verified |
| Server project | `/data/fit.lamnv/smallflood_cd` | Recorded; not connected |
| Raw dataset root | `/data/fit.lamnv/UrbanSARFloods/UrbanSARFloods_v1` | Recorded; not locally present |
| Raw archive | `/data/fit.lamnv/UrbanSARFloods/UrbanSARFloods_v1/urban_sar_floods.tar.gz` | Recorded |
| Extracted development | `/data/fit.lamnv/UrbanSARFloods/UrbanSARFloods_v1/extracted_v1/urban_sar_floods` | Preparation code expects `01_NF`, `02_FO`, `03_FU`, each with SAR/GT |
| Original full test scenes | `/data/fit.lamnv/UrbanSARFloods/UrbanSARFloods_v1/testing_case_orig` | Recorded; protected |
| Prepared arrays/cache | `/data/fit.lamnv/smallflood_cd/data/processed/urbansarfloods` | Recorded; absent locally |
| Manifest | `/data/fit.lamnv/smallflood_cd/data/processed/urbansarfloods/patch_manifest.csv` | Recorded hash frozen below |
| Normalization | `/data/fit.lamnv/smallflood_cd/data/metadata/normalization_v1.json` | Recorded; absent locally |
| Component statistics | `/data/fit.lamnv/smallflood_cd/data/metadata/component_stats_v1.json` | Recorded; absent locally |
| Event split | `/data/fit.lamnv/smallflood_cd/data/splits/split_v1.yaml` | Recorded; absent locally |
| Completion marker | `/data/fit.lamnv/smallflood_cd/data/processed/urbansarfloods/PREPARATION_COMPLETE.json` | Expected by preparation; not locally inspectable |

Current YAML uses project-relative paths to those resources. Prepared manifest array paths are absolute server paths; moving files to a new machine does not make them automatically portable. Historical Mac `scp` examples refer to an obsolete nested checkout under `academic-research-skills-codex`; use the current local project root above when locating files. Historical server transfer endpoint is `fit.lamnv@100.102.158.114`; connectivity/access is unverified.

Frozen SHA256 anchors, enforced by factorial/Candidate R tooling:

```text
manifest:
51eba373aef67e05b768bfc3c402d15c9fb9c8417c7026c6079a8fc0bea9b3ae
component_stats:
525e4e1354e991983ebd2e3d3f68caad3f9894308a6eae46256a74405c9f2247
normalization_stats:
33ef9a286340e3a58b8fc3623da210d83624c8a5901e9bcd0799e660a6ff625d
```

Do not bypass these guards to accommodate relocated or changed data. Any path migration requires an explicit provenance-preserving decision; never silently refit data/statistics.

## 4. Dataset preparation and status

Preparation is implemented in `scripts/prepare_urbansar_raw.py` and `scripts/prepare_urbansarfloods.py`. Recorded full-data pilots used **19,166 train and 4,767 validation patches**. Local `data/raw`, `interim`, `processed`, `metadata`, and `splits` directories contain no dataset files; only `data/README.md` is present as a file. Do not rerun preparation because this Mac checkout lacks the server data.

Frozen contract:

- Two channels per date, order **VH,VV**. Rasterio one-based intensity bands pre=[5,6], post=[7,8]; NumPy eight-band indexing equivalents [4,5]/[6,7]. Released intensities are already dB: no second log transform.
- Shared pre/post per-polarization population mean/std fitted using valid **train pixels only**. Saved NPY inputs are already normalized. `load_array` does not normalize again.
- FO=1 and FU=2 map to binary flood=1; NF=0 stays 0. Original semantic arrays retained separately, with invalid/padded pixels=255.
- 256×256 nonoverlapping full/partial windows; retain every patch with valid pixels; pad partial inputs, zero invalid binary targets, mask invalid/padding. Preserve scene IDs, offsets, source geotransform and valid dimensions for future scene reconstruction.
- Enforce paired shapes, CRS and transforms, finite selected SAR bands, raster masks and valid GT values. Grid agreement does not establish physical subpixel registration; undeclared zero margins cannot automatically be identified as NoData.
- Four original test frames remain test; Jubba tracks grouped as one event, Hebei tracks grouped. Development events repartitioned deterministically: ceil(20%) held out using sorted SHA256(seed:event), seed42. This differs from official train/validation partition and is event-held-out, not necessarily location-held-out.
- Validation events: `20161011_Lumberton`, `20220302_Coraki_Australia`, `20230805_Hebei`. Complete train/test event list and test patch count are not available in the local split file, which is absent; do not invent them.
- Train-only 8-connected patch-component median reference area **65 pixels**, small threshold **35 pixels** (floor of train 25th percentile, minimum1). These are patch-clipped areas, not scene objects or physical square metres.
- Coherence and uncertain-mask paths intentionally empty. Coherence feature definition and annotation uncertainty masks are not verified/ready. With/without-uncertainty configurations are not meaningful distinct experiments on the existing manifest.

Recorded train component statistics: 52,516 components, 23,436,067 foreground pixels; 13,497 components ≤35 pixels occupy 192,296 pixels. Original weighting raises small-component foreground weighted-mass share from 0.82051% to 1.87195%; that is neither gradient share nor demonstrated effectiveness.

Recorded R eligibility audit: 1,420 eligible patches, 3,521 eligible components, 95,594 eligible pixels, zero empty rings; roughly53–55% of batch8 minibatches have no eligible object. The source runner enforces these totals. Original audit report/per-patch counts are server-only in this checkout, expected under `artifacts/candidate_r_eligibility/20260928T134626505807Z`.

Published band convention was source-verified in the September24 report. Earlier preparation/protocol notes saying band mapping remains unverified are superseded **only at the published-convention/metadata level**, not by a per-raster binary audit. Labels remove some tiny floods during creation; small-region recall measures agreement with released labels, not completeness for every physical small flood. Do not turn 35 pixels into square metres solely from nominal20m spacing.

## 5. Current model architecture

Registry name `smallflood_cdnet` builds `SmallFloodCDNet`.

- Shared two-channel MobileNetV3-Small encoder, fresh initialization (`pretrained:false`); features captured at indices1,3,8,12, strides4/8/16/32, channels16/24/48/576.
- Per-scale temporal fusion concatenates `abs(post−pre)` and `post*pre`, then convolution/BN/activation projections to24/32/64/96 channels. Optional coherence interpolation exists in code, but current research inputs omit it.
- Lightweight decoder: depthwise-separable bottleneck and three upsample/skip-concatenation stages; final feature grid stride4, 24 channels.
- One-logit preliminary change head; optional small boundary branch produces boundary logits and a residual using decoded features plus sigmoid boundary probability.
- Refined logits = preliminary + clamp(learned residual scale,0,1)×residual. Scale initializes0.1. Change/boundary/preliminary outputs are bilinearly upsampled to input size.
- Parameter counts in recorded runs: boundary-on **1,080,805**, boundary-off A/B/A-control/R **1,079,865**; head adds940. Registry enforces maximum5,000,000.
- `projection_channels` controls actual decoder widths; separate YAML `decoder_channels` is not consumed.

Original loss:0.4 weighted BCE +0.4 size-aware Tversky +0.2 boundary. Foreground weights `1 + alpha*min(cap,(65/(area+1e-6))^gamma)`, defaults alpha1,gamma0.5, enhancement cap4 → **total maximum5**; background1. Weighted BCE divides weighted loss sum by weighted valid-pixel sum. Tversky weights TP/FN positives, not FP, with FP0.3/FN0.7; batch-pooled, not per-component average. Boundary target is radius1 morphological gradient and its objective is equal BCE/Dice.

**Candidate R:** same boundary-off A inference graph, original size weights off,0.4 unweighted BCE +0.4 unweighted Tversky +0.1 local auxiliary. Detached valid-masked train GT defines 8-connected components ≤35 pixels excluding patch/invalid-neighbor rims. For each eligible component, average positive softplus(−logit) and valid-background radius3 Chebyshev ring softplus(logit), each coefficient0.5; average over all eligible components in the minibatch. Other foreground excluded; overlapping rings accumulate; empty ring contributes zero negative term; empty eligible batch gives differentiable zero local term. SciPy CPU preparation; no learnable auxiliary parameters. A-control computes local diagnostics but applies coefficient0. The generic loss factory does not select R; only its isolated runner does.

## 6. Baselines

| Registry | Definition | Parameters | Interpretation |
|---|---|---:|---|
| `fc_siam_diff` | Shared convolutional encoder widths16/32/64/128; absolute feature differences and upsample/skip decoder |487,857| Existing FC-Siam-Diff implementation; no separate official-source parity audit found |
| `bit` | Custom ResNet18 through layer4, stride32; dim64,4 semantic tokens/date,2 encoder layers,4 heads, one bare pixel-token attention decode, coarse logits then interpolation |11,360,133| BIT-inspired historical implementation; **not faithful official BIT** |
| `bit_sar_v2` | Source-pinned official `base_transformer_pos_s4_dd8` architecture adapted to SAR |3,492,642| Versioned SAR adaptation, not reproduction of original RGB training recipe |

BIT v2 pins `justchenhao/BIT_CD` commit `adcd7aea6f234586ffffdd4e9959404f96271711`. Features64×64 for256 input; dim32, four tokens/date, learned positions, one encoder/eight residual decoder blocks,8 heads/head-dim64; full-resolution two-class classification, binary adapter `z_change−z_nochange`. Two SAR input channels, fresh normal initialization, reset BN buffers, no downloads; unused layer4/avgpool/fc removed. Fixed-weight forward/gradient parity tests retain original reference sources. Old `bit` checkpoints cannot initialize v2 and old accuracy/time cannot be paired with v2 parameter count. Research/noncommercial upstream restriction and notices must remain.

Baseline pilots used0.5 BCE +0.5 Tversky, no size/boundary terms; AdamW LR3e-4 for FC,1e-4 for BIT variants. Proposed LR3e-4 with a different objective. This is method-package comparison, not an isolated architectural fairness/contribution proof. No final equivalent validation tuning budget is established.

## 7. Deployment targets

Frozen deployment YAML: **Jetson Orin Nano8GB**, TensorRT FP16, batch1, paired inputs each `[1,2,256,256]`, ONNX opset17. Warmup100, timed500, repetitions3; latency≤30ms, peak memory≤2048MB, model size≤20MB. Accuracy parity/tolerance remains to be specified/verified before claiming compliance.

ONNX export code exists with static two inputs `pre`,`post`, output `change_logits`, eval mode and legacy Torch exporter (`dynamo=False`). No sigmoid is exported. Wrapper omits coherence, so the current intensity-only contract is what it supports. No ONNX artifacts, runtime parity evidence, TensorRT builder, `trtexec` commands, Orin harness or deployment measurements were found. Matrix `deployment_checks` is a declaration; matrix loading/execution does not consume that section. Training RTX A6000 timing/memory is not deployment evidence.

## 8. Completed implementation work

Implemented and tracked:

- Dataset archive integrity/extraction/LFS recovery and resumable CPU raster→NPY preparation; immutable raw inputs, event split, train statistics, compatible component cache.
- Manifest datasets, array loader, event-isolation validation, component weighting, uncertain-border and boundary-supervision helpers.
- Proposed model, two original baselines, independent source-pinned BIT-SAR v2 plus upstream notices/reference parity tests.
- BCE/Tversky/boundary composite loss and isolated local-component Candidate R loss.
- Training/checkpoint/RNG-state logging and `event_pixel_v2` validation selection; legacy metric modules remain separate.
- Generic grid/matrix CLI/CSV collection and ONNX export scaffold.
- Strict smoke/pilot tools, source-import guard, immutable provenance/fingerprint gates and fresh-output enforcement.
- Train-mask and R eligibility audits; paired A/B/C/D factorial runner/reviewer; paired A-control/R runner.
- **Candidate R four-checkpoint validation-only reviewer already exists**: `scripts/review_candidate_r.py`, launcher `scripts/run_candidate_r_review.sh`, hash anchor `configs/review/candidate_r_20261003.json`, tests `tests/unit/test_review_candidate_r.py`. This supersedes the latest training audit's recommendation to implement it. No local result/completion evidence for its real-data execution was found.

Tests cover model shapes/budget, loss gradients/eligibility/empty cases, masks, split/preparation integrity, event aggregation/ties/undefined metrics, checkpoint fidelity, deterministic pairing/gate drift, source shadowing and validation-only review integrity. Historical targeted test pass counts are in dated docs/logs; they are not a fresh whole-suite result for this inspection. No tests were run here.

## 9. Completed experiments and results

All main numerical findings below are **single-seed42,15-epoch development validation pilots**, threshold0.5; not held-out test, scene-level, significant or deployment results. Percentages; epoch numbers in these tables are **one-based** unless specified. JSONL/checkpoint epochs are zero-based. Best always means event-macro pixel F1, never small-recall selection.

### Initial Proposed / FC / custom BIT pilot (locally available imported evidence)

Server run `runs/pilot15/20260924T032911Z`; review `20260924T082109094745Z`. 15 epochs/model,35,940 optimizer steps/model, batch8,FP32,workers0, strict determinism/math attention, no scheduler.

| Model |Best epoch|Event F1|Pixel F1|Component F1|Interior-small recall|Boundary F1|Inner-band IoU|
|---|---:|---:|---:|---:|---:|---:|---:|
|Proposed|13|83.65|84.39|55.89|27.84 (103/370)|77.42|32.45|
|FC-Siam-Diff|12|85.52|85.47|47.63|30.54 (113/370)|78.49|47.80|
|Custom BIT|7|65.95|70.01|19.61|0.00 (0/370)|22.54|5.08|

Last event F1:80.57/85.17/57.60 respectively; last interior-small recall21.35/23.78/0.00. Proposed has higher component F1 than FC but lower small recall, pixel/event F1 and boundary metrics. Do not claim uniform superiority. Training+validation minutes102.00/76.03/58.85 and peak allocated0.486/1.295/0.699GiB are engineering records only.

### BIT-SAR v2 pilot and review (locally available)

Server pilot root `runs/bit_sar_v2_pilot15/20260925T015023Z`, actual pilot child `pilot/20260925T015034589216Z_bit_sar_v2`; review root `artifacts/bit_sar_v2_validation_review/20260925T081025Z`.

Batch8 preflight passed in imported completion/log records; full pilot15 completed. Best=last epoch15. Event F1 **85.00**, pooled pixel F1 **86.30**, IoU75.89, component F1 **57.88**, interior-small recall **29.19 (108/370)**, boundary F1 **78.95**, inner-band IoU **34.02**. Both report metrics match; serialized checkpoint hashes differ, so do not call binaries identical. FC still leads event F1, interior-small recall and inner-band IoU. Source parity does not establish original BIT-paper accuracy reproduction.

### Frozen 2×2 factorial (locally available)

Server `runs/proposed_factorial_v1/20260927T075837Z`; review `artifacts/proposed_factorial_review/20260928T004225Z/results`. All four fresh runs15 epochs,35,940 steps each; common initialization and all15 epoch-order hashes agree in records. A/B weights fully paired; C/D fully paired. Historical Proposed is not substituted for new D (different frozen RNG/order convention).

|Cell|Size|Boundary package|Best epoch|Event F1|Pixel F1|Component precision|Component F1|Small recall|Boundary F1|Inner-band IoU|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|A|off|off|12|82.84|84.64|57.93|52.85|34.05 (126/370)|76.92|30.46|
|B|on|off|13|82.82|83.56|63.95|53.82|20.81 (77/370)|75.70|31.92|
|C|off|on|12|82.95|83.97|62.71|53.87|24.86 (92/370)|77.07|33.24|
|D|on|on|14|82.64|83.17|61.43|54.15|28.38 (105/370)|76.64|33.17|

Last epoch15 event F1 A82.57/B80.56/C76.21/D80.20. Last small recall A22.97 (85)/B20.54 (76)/C19.46 (72)/D22.97 (85), all /370. Last boundary IoU32.79/32.05/32.49/30.71.

Predefined best small-recall contrasts (percentage points): B−A −13.24; D−C +3.51; C−A −9.19; D−B +7.57; interaction D−C−B+A +16.76. D−A −5.68 despite positive interaction. A has highest best small recall in both defined events. Boundary IoU best-checkpoint signals do not persist at last. No causal module explanation or robust improvement established. C/D epoch-end residual scales remained inside(0,1), ending0.815858/0.752798; logs do not establish clamp saturation as the failure cause.

### Candidate R A-control/R pilot (recorded audit; original evidence absent here)

Server `/data/fit.lamnv/smallflood_cd/runs/candidate_r_v1/20261002T150633Z`. Original audited archive `/Users/nguyenvulam/Downloads/candidate_r_training_review_20261003.tar.gz` is outside repository and was not reopened in this handoff. SHA256 `a70c3afe5bb4484aa845ede9a873a9cade6c0489facabf288cccb855d14c210d`; per-file anchors preserved in `configs/review/candidate_r_20261003.json`.

|Cell/checkpoint|Epoch (one-based)|Event F1|Pixel precision|Pixel recall|Pixel F1|Pixel IoU|
|---|---:|---:|---:|---:|---:|---:|
|A-control best|12|82.84|84.71|84.58|84.64|73.37|
|R best|2|66.71|68.73|82.51|74.99|59.99|
|A-control last|15|82.57|87.33|80.31|83.67|71.93|
|R last|15|59.40|38.78|78.21|51.85|35.00|

Both15 epochs/35,940 steps; identical initial state hash `71c04062460c9639694c4d24f7944b46a909e3fb990340bee8307db5945754b8` and epoch-order hashes recorded. Best R worsens all three event F1s. Last FP A370,862 versus R3,929,422. Mean applied-auxiliary/base output-logit gradient ratio6.94–8.48 across epochs; last clipping45.08% R versus29.63% A. These observations do not diagnose the causal mechanism. Recorded elapsed7108.48s A/6723.59s R, about3.84h total, not inference benchmarks.

**Candidate R component/small/boundary metrics are unavailable in this checkout and absent from the audited training archive.** They must not be invented. Recorded checkpoint reload/hash claims await checkpoint-based review. An absent `scripts/audit_urbansarfloods.py` prevented one provenance comparison in the earlier audit; a utility exists at `src/smallflood_cd/smallflood-tools/audit_urbansarfloods.py`, which is not proof of identity with the missing server path.

### Earlier engineering evidence

Docs record earlier strict GPU smoke repeats and a three-epoch pilot, and a BIT v2 batch2 exact-match smoke at `runs/bit_sar_v2_smoke/20260924T091512Z`. Their full original artifacts are not imported here; do not infer unrecorded scores. Candidate R runner docs record synthetic37 passed/2 CUDA skips and regression12 passed, loss18 passed/1 skip locally, and user-reported19 CUDA loss tests. These are historical software checks, not scientific experiments or a new handoff verification run.

## 10. Experiments currently running

**Cannot verify current running jobs.** Local `runs/` is empty; available imported logs show completed historical work. Candidate R training is recorded complete; review completion is not present. Local process listing was sandbox-denied; no SSH/tmux/GPU inspection was performed. “No local artifacts” must not be translated into “no server jobs.” Obtain a read-only server status snapshot before future compute work and do not interrupt any job based on this handoff.

## 11. Pending experiments / reviews

Immediate pending evidence task: four fixed Candidate R A-control/R best/last **validation-only** evaluations using the existing guarded reviewer, after server availability/source/checkpoint verification and applicable user authorization. No optimizer steps or new training checkpoints. Confirm pixel metrics/epochs/counts against logs and collect all component/boundary denominators to close the failed candidate.

Deferred, **not approved for automatic launch**:

- Multi-seed confirmatory work: no established successful candidate currently warrants it; any budget/protocol requires a separate decision.
- Long-run/matrix ablations, final baseline comparisons and held-out test use: blocked by method failure, legacy evaluation integration and protocol/readiness gaps.
- Scene reconstruction/full-scene component metrics; registration robustness and representative visual review.
- Coherence/uncertainty experiments: unavailable input contracts; do not fabricate features/masks.
- ONNX/runtime/TensorRT FP16 parity and Orin deployment benchmark: implemented only at export scaffold/target specification level.

No new experiment is authorized by the existence of YAML or a command below.

## 12. Full ablation plan and status

### Completed controlled diagnostic plan

`configs/experiment/proposed_factorial_v1/{a,b,c,d}.yaml`: A neither, B size, C boundary package, D both. Segmentation coefficients stay0.4/0.4; boundary-on adds0.2, boundary-off0. These are additive objectives, not renormalized losses. Size affects both BCE and Tversky. Boundary package combines head parameters, residual inference and auxiliary supervision, so this design cannot independently isolate all three. Report B−A/D−C/C−A/D−B and interaction for every predefined metric, best and last.

Controls: same immutable split/normalization, seed42,batch8,15 epochs,FP32,workers0,AdamW3e-4/weight decay1e-4/clip1,no scheduler/warmup/early stopping/augmentation/resampling changes, common initial tensors and independent seeded DataLoader order, RNG reset after model construction. Train-only reference65 and small35 fixed. All four and eight reviews completed in imported records.

Candidate R: two fresh A-control/R runs, same controls and inference graph, applied local coefficient0 versus0.1, radius3; completed according to audit; failed gate; four-review closure pending evidence. Do not replace fresh control with historical A merely because recorded scores match.

### Legacy full matrix inventory (deferred)

`configs/experiment/experiment_matrix.yaml`, seeds `[42,1337,2026]`, 45 expanded variants×3=**135 jobs**:

|Entry/config|Variants|Meaning / limitation|
|---|---:|---|
|proposed / full_proposed|1|Full original package|
|fc_siam_diff|1|Existing FC baseline|
|bit|1|Old custom BIT, **not v2**; replacement requires explicit decision|
|plain_bce|1|Boundary off; BCE1, Tversky0, no size|
|standard_tversky|1|Boundary off; BCE/Tversky0.5/0.5, no size|
|size_aware_only|1|Boundary off; BCE/Tversky0.4/0.6, size on; differs from controlled factorial B|
|boundary_only|1|Boundary on;0.4/0.4/0.2, size off|
|morphology_postprocess|1|Boundary off; size on,0.4/0.6; closing; not a pure head-only contrast|
|weight_sensitivity|27|gamma0.25/0.5/1 × enhancement-cap2/4/6 × Tversky FN0.6/0.7/0.8; FP stays0.3|
|intensity_only|1|Existing usable input contract|
|intensity_coherence|1|Blocked: no defined prepared coherence feature|
|no_uncertain_mask|1|Available inputs have no uncertainty masks|
|with_uncertain_mask|1|Blocked/no meaningful contrast absent actual masks; dilation1|
|shift_robustness|5|Post input diagonal shift0/1/2/3/5 pixels; each matrix variant currently retrains rather than reusing one model|
|reduced_decoder|1|Projection widths16/24/32/64; this is the active width knob|

Most configs use120 epochs,batch16,patience15,clip1,LR3e-4 (BIT1e-4); generic runner uses cosine scheduler. These are not the completed15-epoch no-scheduler pilots. Legacy matrix timing20 warmup/100 timed differs from frozen deployment100/500/3. No approved integrated final protocol exists; do not execute the matrix or silently “fix” its inventory.

## 13. Validation protocol

Current checkpoint selection: `event_pixel_v2`, implemented in `engine/validator.py` and used by dedicated runners.

- Fixed probability threshold0.5, validation only, unshifted/no morphology for selection.
- Apply valid masks and supplied uncertainty exclusions where supported; present dedicated SAR reviews reject coherence/uncertainty inputs rather than manufacturing them.
- Sum integer TP/FP/FN/TN over patches **within each event**, then average defined event F1s equally. Also report pooled global pixel metrics and all event metrics.
- Zero denominator→null; omit undefined values from macro and report defined counts. No absent/nonfinite score fallback. Strictly higher wins, so ties keep earlier epoch.
- `best_composite.ckpt` is only a legacy filename, not composite selection. Epochs saved0-based, review CSVs1-based.
- Protocol history matters: v2 was adopted after exploratory earlier pilot validation. Do not relabel earlier pilots as prospectively executed under later decisions.

Diagnostic object/boundary protocol: `object_boundary_patch_v2`, authoritative source `metrics/object_boundary_v2.py` and `docs/object_boundary_protocol_v2.md`.

- Mask thresholded predictions/GT with valid mask,8-connectivity; patch-local components.
- Sparse overlap IoU≥0.1; descending IoU, then predicted ID/GT ID ascending; deterministic greedy one-to-one matching. Not optimal assignment; splits/merges receive at most one match/component.
- Component P=M/Ppred,R=M/Tgt,F1=2M/(Ppred+Tgt), matched mean IoU. Small recall conditioned on GT area≤35; report clipped and interior. Interior excludes component touching patch exterior or8-neighbor invalid rim. Match all components before small stratification. No separately defined small-prediction precision/F1 exists.
- Boundary contour `X−erosion1(X)`, band `X−erosion2(X)`, Chebyshev morphology. Restrict to valid mask eroded3, outside image invalid. Tolerance F1 uses2-pixel matching; inner-band IoU is a distinct measure, not dilated-contour IoU.
- Pool counts globally/within event before ratios; macro separately with undefined counts. Report raw patch/event counts and boundary coverage.
- Identical validation denominators in existing reports:4,767 patches,295,762,355 valid pixels,3,182,289 positive pixels (~1.076%),6,745 GT components,1,551 clipped-small,370 interior-small. Interior Coraki42/Hebei328/Lumberton0 (undefined recall);1,181 small components excluded. No independent-replicate significance inference from pixels/patches/components.

Earlier `docs/evaluation_metrics.md` and legacy metric modules use morphological-gradient boundaries and old aggregation. Do not equate their output with v2 inner-band IoU. Full matrix `collect_metrics` still averages per-patch results and automatically evaluates test after training; it is **not approved** for the135-run study. Scene reconstruction is missing; all current object conclusions are patch-scoped. Deterministic preview selection is first examples by event/category, not representative sampling.

## 14. Important design decisions and rationale

|Decision|Recorded rationale / consequence|
|---|---|
|Lightweight shared MobileNet encoder, depthwise decoder, small residual boundary head|Meet parameter/edge budget while preserving multiscale temporal features; latency benefit still unmeasured|
|Train-only cache/statistics and event holdout|Avoid repeated component analysis and statistical/event leakage; preserve track grouping|
|Inverse-sqrt capped positive weighting, FN0.7/FP0.3|Prioritize missed small changes while limiting tiny-label domination; does not equalize object contribution|
|Fixed threshold, macro-event best and last reporting|Prevent patch-size weighting and post-hoc checkpoint/threshold cherry-picking; expose checkpoint sensitivity|
|Separate old BIT and source-pinned v2|Correct architecture fidelity without erasing historical evidence or mixing counts/results|
|Controlled factorial with unchanged segmentation coefficients|Measure size and boundary packages under a common budget; disclose extra-gradient/clipping interaction|
|Separate R runner/control and frozen stop gate|Test one bounded auxiliary hypothesis with identical inference/initialization/order; avoid open-ended tuning after failure|
|Fresh run directories and fingerprint/import guards|Prevent overwrites, stale package execution, accidental resumes and provenance drift|
|No invented coherence or uncertainty|Existing manifest lacks verified features/masks; distinct configs do not create distinct data|
|Protected test and separate diagnostics|Development validation is not confirmation; no final claim before readiness/protocol gates|

## 15. Known bugs, blockers and unresolved questions

**Source-verified risks/gaps (not all established causes of poor scores):**

- Generic `cli/train.py` constructs `ManifestDataset` without component cache/reference weights; trainer falls back to unit weights. Thus its command is not equivalent to size-aware Proposed pilots. Use dedicated protocol tooling for faithful continuation, not the generic shortcut.
- Generic matrix `run_experiment` builds test loader and evaluates test automatically, despite protocol metadata flags. Final collection uses legacy patch averages and legacy boundaries. YAML `test_allowed:false` is not a security gate in generic code.
- Generic `seed_everything` uses deterministic `warn_only=True`; strict dedicated launchers use hard failures, TF32 off/math attention and env vars set before Python. Do not substitute one for the other.
- Boundary residual scale hard clamp has zero refinement gradient outside[0,1]. Synthetic tests demonstrate risk; factorial logs show no epoch-end saturation. No evidence establishes this as the failure cause.
- Boundary training supervision differs from evaluation's eroded safe interior; valid-neighbor edges may be supervised. Component cache derives raw binary target before valid/uncertain masking. Real audit scripts exist; detailed train-mask report is not present locally, so affected real-data counts are unverified.
- Generic trainer does not forward configured boundary radius and training uncertainty dilation to helper calls (default1). Current primary pilot uses default settings, so no demonstrated mismatch there; future radius/dilation ablations need explicit verification.
- `decoder_channels` inactive; `warmup_epochs`, AMP and logging declarations are not evidence of implemented training behavior. Current generic fit has no AMP/autocast or warmup and no TensorBoard writing path.
- SciPy is required for R/preparation/v2 metrics but undeclared in base dependencies. Pillow is imported by reviewer and normally comes through torchvision. No lockfile; local and server stacks differ.
- Tracked egg-info is stale: dependency list lacks einops and entry-points file lacks the matrix entry. Treat `pyproject.toml` as authoritative packaging source; do not regenerate metadata during handoff.
- Virtual environment was created at an older nested Mac path; editable installation/script paths can be stale after relocation. Dedicated import guard must point to current `src/smallflood_cd`.
- Current checkpoint saver stores CPU torch RNG only (plus model/optimizer/scheduler/config/epoch/score). It does not implement complete restart replay of Python/NumPy/CUDA/DataLoader RNG states. Smoke tools have separate broader repeat checks; exact bounded smoke does not establish15-epoch reproducibility or resumability.
- Trusted checkpoints use `torch.load(...,weights_only=False)`; dedicated R reviewer verifies recorded hash before unpickling. Only self-produced trusted checkpoints should be used.
- Current R reviewer tools are present but server deployment/test pass/full review completion is not evidenced here. Hash/source/count/epoch drift must stop review rather than be bypassed. Its four jobs require actual server checkpoints and anchored11 evidence files.
- Research failure cause, prior-art novelty clearance, independent multi-seed uncertainty, final fair baseline protocol, scene-level performance and Orin compliance remain unresolved. Do not conflate engineering implementation tests with these questions.

## 16. Environment and dependencies

Authoritative `pyproject.toml`: Python≥3.10; numpy≥1.26,PyYAML≥6,torch≥2.2,torchvision≥0.17,einops==0.8.1. Extras:geo rasterio≥1.3/pandas≥2.1;dev pytest≥8/ruff≥0.5;deploy onnx≥1.16/onnxruntime≥1.18. SciPy currently must already be supplied for the relevant tools; do not silently upgrade the frozen server stack.

Verified local installed package metadata (not imported backend/device smoke):Python3.14.0,torch2.13.0,torchvision0.28.0,numpy2.5.1,scipy1.18.1,PyYAML6.0.3,pytest9.1.1,ruff0.15.21,einops0.8.1,rasterio1.5.1. pandas/onnx/onnxruntime not installed. `.venv/pyvenv.cfg` points to Homebrew Python3.14. Local versions do not reproduce CUDA-server runtime.

Recorded Candidate R server runtime:Python3.10.12,torch2.5.1+cu124,CUDA12.4,cuDNN90100,numpy2.2.6,scipy1.15.3,NVIDIA RTX A6000,24 torch threads;PYTHONHASHSEED42,CUBLAS_WORKSPACE_CONFIG=:4096:8. Earlier imported environment records confirm key torch/CUDA/GPU stack for pilots. Live driver, remaining disk/memory, CUDA availability, TensorRT/JetPack versions and competing jobs are unverified.

Install recipe exists (`python -m pip install -e '.[geo,dev,deploy]'`) but is **not** an instruction to change the historical server environment. Confirm exact package metadata/import source first; preserve dependencies between preflight and pilot. All strict tools run from the project root with `.venv` active; use tmux for authorized server work and stop on failures, no automatic retries.

## 17. Important files and directories

Paths below are relative to local `/Users/nguyenvulam/academic-research-skills/smallflood-cd` or recorded server `/data/fit.lamnv/smallflood_cd`.

|Path|Purpose|
|---|---|
|README.md / data/README.md|Top-level model and manifest contracts|
|pyproject.toml|Authoritative package requirements/CLI definitions|
|configs/base.yaml; data/; model/|Default environment/data/model settings; merge via utils/config.py|
|configs/experiment/main.yaml, ablations/, experiment_matrix.yaml|Legacy120-epoch plan and135-job inventory, deferred|
|configs/experiment/proposed_factorial_v1/|Frozen controlled A/B/C/D configs|
|configs/experiment/candidate_r_v1/|Frozen isolated A-control/R configs|
|configs/review/candidate_r_20261003.json|Audited R training-file hash anchors|
|configs/deployment/jetson_orin_nano.yaml|Unmeasured deployment targets|
|src/smallflood_cd/data/|Manifest/masks/components/cache/normalization/split helpers|
|src/smallflood_cd/models/|Proposed, fusion/decoder/heads and baseline registry|
|models/baselines/bit_upstream/; tests/reference/bit_upstream/|Adapted runtime and original pinned BIT sources/notices|
|src/smallflood_cd/losses/|Original composite/Tversky plus isolated R local loss|
|src/smallflood_cd/engine/|Generic training/selection/checkpoints/grid/legacy final collection|
|src/smallflood_cd/metrics/object_boundary_v2.py|Authoritative current patch-object/boundary diagnostic protocol|
|Other metrics modules|Legacy metrics; do not mix definitions silently|
|src/smallflood_cd/cli/; deployment/|Generic CLI and static ONNX export|
|scripts/prepare_urbansar_raw.py; prepare_urbansarfloods.py|Raw integrity/recovery and prepared dataset conversion|
|scripts/next_steps.py; smoke_train.py|Original strict pilot orchestration/integration smoke|
|scripts/bit_sar_v2_entry.py|Source pinning and v2 dispatch; protects against stale shadow package|
|scripts/smoke_bit_sar_v2.py; pilot_bit_sar_v2.py|V2 exact smoke and batch8/pilot gates|
|scripts/proposed_factorial.py; review_proposed_factorial.py|Paired factorial execution/review|
|scripts/audit_proposed_train_masks.py; audit_candidate_r_eligibility.py|Read-only train mask diagnostics and metadata schedule|
|scripts/candidate_r_runner.py|Isolated R tests/preflight/full-run gates, completed pilot capability|
|scripts/review_pilot_validation.py; review_candidate_r.py|Shared validation-only reports and frozen four-checkpoint R closure|
|scripts/run_*.sh; preflight_proposed_factorial.sh|Manual server launchers, strict env/logging/fresh UTC outputs|
|tests/unit/|36 test files; synthetic CPU/CUDA guards, numerical parity and integrity checks|
|docs/|Frozen protocols, dated implementation/research audits and superseded preparation notes|
|artifacts/imported/|Ignored local evidence, not included in Git clone|
|artifacts/updates/|15 ignored update tarballs; historical packages, not guaranteed latest source|
|runs/|Empty locally; server run/checkpoint trees must be retained separately|
|src/smallflood_cd.egg-info/|Tracked historical installed metadata; stale versus pyproject|
|.venv/, .pytest_cache/, .ruff_cache/, __pycache__|Local generated environment/caches, not experiment evidence|

### Available imported evidence inventory

All under local `artifacts/imported/` (ignored by Git):

- `pilot15_20260924/` —125 files: three console logs, metadata/preflight log, six pixel/object reports, CSVs, previews/indexes, completion and source snapshot.
- `pilot15_training_details_20260924/` —15 files: three runs' resolved configs/environment/summary/metrics/timing.
- `bit_sar_v2_pilot15_20260925/` —20 files: batch8 check, pilot logs/metadata/metrics/timing and source snapshot.
- `bit_sar_v2_validation_20260925/` —41 files: best/last reports/CSVs/previews/review log/completion.
- `proposed_factorial_20260928/` —19 files: four run configs/environment/metrics/summary, root provenance/completion and training log.
- `proposed_factorial_validation_20260928/` —155 files: eight checkpoint report/CSV/preview sets and review completion/log.

No `.ckpt`, `.onnx` or `.engine` files found outside `.venv`; no Candidate R imported run/review folder, no train-mask report or eligibility report locally. A fresh Git clone will lose all ignored evidence unless separately transferred. Preserve the imported trees, original server run/checkpoint directories and metadata/audit files before handoff cleanup. Do not confuse an update tarball containing code with a result archive.

### Documentation reading order

Start with this file, `docs/candidate_r_training_audit_20261003.md`, current R review source/tests/anchor, then `docs/proposed_bounded_redesign_plan_20260928.md` and `docs/candidate_r_runner_20261002.md`. Read `docs/proposed_factorial_validation_audit_20260928.md` before any further method claim. For reproducibility:factorial protocol/runner/review docs, validation/object-boundary v2 docs, dataset preparation/band verification. For baselines:BIT official audit, BIT v2 implementation and validation audit. Preserve initial pilot review/training audit, Proposed implementation/train-mask/component-histogram audits, and contribution contract as research history.

## 18. Exact commands available in the repository

**Reference commands only: none was executed during this handoff.** Run from the proper project root. Do not treat historical launch examples as renewed approval; no new experiments, tests, export or deployment are requested by this handoff. Placeholder paths must be replaced with actual verified artifacts, not invented gates.

### Read-only identity / server readiness

```bash
cd /Users/nguyenvulam/academic-research-skills/smallflood-cd
git status --short --branch
git remote -v
git rev-parse HEAD
# On an already authorized server session:
cd /data/fit.lamnv/smallflood_cd
source .venv/bin/activate
nvidia-smi
tmux ls
ps -ef
```

### Tests (synthetic, may write temporary fixtures/cache; no dataset pilot)

```bash
python -m pytest -q
python scripts/test_candidate_r_loss.py
python scripts/candidate_r_runner.py test
python scripts/proposed_factorial.py test
python scripts/review_candidate_r.py --self-test
python scripts/review_proposed_factorial.py --self-test
python scripts/audit_candidate_r_eligibility.py --self-test
python scripts/audit_proposed_train_masks.py --self-test
python scripts/bit_sar_v2_entry.py test
python scripts/bit_sar_v2_entry.py test-pilot
python scripts/bit_sar_v2_entry.py test-review
python -m pytest -q tests/unit/test_bit_sar_v2.py tests/unit/test_baselines.py
```

A successful synthetic run does not authorize training/review launches or establish dataset effectiveness. Local CUDA skips are expected; do not report them as server passes.

### Candidate R validation-only closure (existing tools, completion not verified)

```bash
cd /data/fit.lamnv/smallflood_cd
source .venv/bin/activate
python scripts/review_candidate_r.py --self-test
# Once server/tooling/evidence is checked and review scope is authorized:
bash scripts/run_candidate_r_review.sh
```

Launcher exports required strict env and evaluates exactly A-control/R best/last from `runs/candidate_r_v1/20261002T150633Z`. Output `artifacts/candidate_r_validation_review/<UTC>/results`, log one level above. Success marker `CANDIDATE R VALIDATION REVIEW COMPLETE`; root COMPLETE must contain evaluations4,test_used=false,training_performed=false,pixel_counts_match_training=true,global_and_event_metrics_match_training=true,gt_denominators_identical=true. It verifies anchored training files, source dependencies, metadata, checkpoint hashes before load, configurations/epochs/best scores, global/event pixel metrics and invariant GT denominators. Stop on discrepancy; no bypass or automatic retry.

### Historical evaluation tools

```bash
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/bit_sar_v2_entry.py review \
  --pilot-root runs/pilot15/20260924T032911Z \
  --output-root artifacts/validation_review --object-boundary
bash scripts/run_bit_sar_v2_validation_review.sh
bash scripts/run_proposed_factorial_review.sh
# Generic pixel-only CLI, not equivalent to guarded full review:
python -m smallflood_cd.cli.validate \
  --config configs/experiment/main.yaml \
  --checkpoint /ABSOLUTE/TRUSTED/CHECKPOINT.ckpt --split validation --threshold 0.5
```

The generic CLI exposes `--split test` but test use is protected; do not choose it. Dedicated reviews use config saved inside trusted checkpoints. Historical reviews are already completed and should not be rerun without a reason/authorization.

### Training / preflight capabilities — historical, do not launch again

```bash
bash scripts/run_pilot15.sh
bash scripts/run_bit_sar_v2_smoke.sh
bash scripts/run_bit_sar_v2_pilot15.sh
bash scripts/preflight_proposed_factorial.sh
python -u scripts/proposed_factorial.py run \
  --preflight runs/proposed_factorial_preflight/REPLACE_WITH_COMPLETED_UTC \
  --output runs/proposed_factorial_v1/REPLACE_WITH_NEW_UTC --approve-training
bash scripts/run_candidate_r_preflight.sh
python -u scripts/candidate_r_runner.py run \
  --audit artifacts/candidate_r_eligibility/20260928T134626505807Z \
  --preflight runs/candidate_r_preflight/REPLACE_WITH_COMPLETED_UTC \
  --output runs/candidate_r_v1/REPLACE_WITH_NEW_UTC --approve-training
```

Full-run examples require `PYTHONHASHSEED=42` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported before Python, unchanged fingerprinted stack, completed exact protocol gate, fresh output and explicit launch approval. Historical R budget is already consumed; its gate failed, so these capabilities are not a recommendation to repeat it.

Generic commands exist but are unsafe substitutes for research continuation:

```bash
python -m smallflood_cd.cli.train --config configs/experiment/main.yaml --run-dir runs/NEW_RUN
bash scripts/run_main_experiments.sh
bash scripts/run_ablations.sh
python -m smallflood_cd.cli.run_experiments --matrix configs/experiment/experiment_matrix.yaml --dry-run
```

Generic train has size-weight wiring gap; matrix launch automatically uses test and legacy collection. Even matrix `--dry-run` creates an output directory/execution_plan.json; it is not strictly read-only. Do not invoke any of these as part of this handoff.

### Dataset preparation commands — implemented, do not redo existing preparation

```bash
python scripts/prepare_urbansar_raw.py /data/fit.lamnv/UrbanSARFloods/UrbanSARFloods_v1
python scripts/prepare_urbansarfloods.py \
  --raw-root /data/fit.lamnv/UrbanSARFloods/UrbanSARFloods_v1 --plan-only
python -u scripts/prepare_urbansarfloods.py \
  --raw-root /data/fit.lamnv/UrbanSARFloods/UrbanSARFloods_v1
```

Preparation reads original test rasters as part of approved original data construction; that is distinct from authorizing model test evaluation now. Exact-input resume rules, disk checks and preparation lock are in dataset-preparation docs; never clear locks or overwrite outputs without checking state.

### ONNX export — available implementation, unverified export result

```bash
python -m smallflood_cd.cli.export_onnx \
  --config /ABSOLUTE/VERIFIED/config_resolved.yaml \
  --checkpoint /ABSOLUTE/TRUSTED/CHECKPOINT.ckpt \
  --output /ABSOLUTE/NEW/model.onnx
```

Use the checkpoint's matching architecture config; A-control/R need boundary-off config. CLI builds model on CPU, loads state, exports static batch1,two channels, configured patch size,opset17. ONNX/onnxruntime absent locally; do not install/export without a separate task. No exact TensorRT build/deployment command exists in the inspected repository, so none is fabricated here. No end-to-end parity or benchmark harness is implemented.

## 19. Current Git state

Inspection baseline before creating this document:

```text
absolute root: /Users/nguyenvulam/academic-research-skills/smallflood-cd
origin fetch/push: https://github.com/nvlam/smallflood-cd.git
branch: main
upstream: origin/main
HEAD: 38416651c2a22afb497988c3d3238946b84951b8
HEAD subject: Initial commit for SmallFlood-CD research project
commit time: 2026-10-03 16:41:18 +0700
tracked state: clean; main and local origin/main at same commit
history: one initial commit; 186 files, 13,194 inserted lines
```

No earlier development commits/branches are visible; dated docs/imported results supply pre-initial-commit chronology. Remote live tip was not fetched, so local origin/main equality is not independent live remote verification. Ignored data/artifacts/runs/environment are excluded from Git. Git state after this handoff should differ only by untracked `RESEARCH_HANDOFF.md`; no commit/staging/push is requested or performed.

## 20. Next five recommended actions, in priority order

1. **Preserve and locate evidence.** Confirm server access/current jobs read-only; retain immutable Candidate R run, four checkpoints,11 anchored metadata/log files, train metadata/audits and all ignored local imported evidence. Confirm current `src` imports and exact server stack. Do not overwrite historical outputs or recreate missing local data.
2. **Check the already implemented R reviewer on the server.** Verify current source/anchor/test files are installed; run synthetic review tests only under the intended next task scope. Do not spend time implementing a duplicate wrapper because the dated audit predates current code. Any source/hash discrepancy is an investigation, not permission to loosen guards.
3. **Complete the four fixed validation-only evaluations** when authorized and available. Require best/last checkpoint hashes/epochs and pixel results to match the audited logs; collect clipped/interior-small, component/boundary/per-event denominators and completion evidence. No retraining, threshold sweeps or test arrays.
4. **Close and preserve the negative R result.** Assess all frozen criteria with the new reports, while retaining the already-failed event-F1 gate and factorial negatives. Record missing evidence honestly; diagnose optimization only to the extent supported by logs. Do not promote R, infer causality or launch more seeds.
5. **Present the research decision gate to the user.** Use the complete existing evidence to decide whether to retain a negative-result direction or authorize a separately scoped next step. Any new research direction/protocol, final evaluation integration, prior-art study or deployment work must be an explicit continuation decision; do not autonomously redesign the model or change the frozen claim.

## 21. Things the next agent MUST NOT change without explicit approval

- Research objective/claim, original methodological mapping, bounded R hypothesis/stop criteria, or what constitutes success/failure. Never relax the conjunctive gate after seeing results.
- Data labels, event partition/grouping, normalization, VH/VV/date band order,256 patch policy, manifest/statistics anchors, cache version/connectivity,65 reference or35 small threshold. No double normalization/logging, validation/test fitting, invented masks/features or unit conversion claims.
- Protected held-out test access/evaluation or use of the generic full matrix to circumvent validation-only scope.
- Model architectures, pretrained policy, boundary path/residual clamp, loss formulas/coefficients/normalization, local coefficient0.1/radius3/eligibility/ring overlap, optimizer/LR/clipping/scheduler, seed/batch/epoch budget, precision/workers, sampler/augmentation or RNG/order convention.
- Selection threshold0.5, macro-event selection/tie rule, best/last reporting, object match IoU0.1,8-connectivity, interior eligibility, boundary erosion/tolerance/IoU definition or pooling/null policy. No post-hoc small-recall checkpoint selection or small-only F1 invention.
- Old BIT identity/results/checkpoints or v2 source pin/adaptation/license/reference files; no silent v2 substitution into the135-run matrix.
- Existing dataset/checkpoints/results/logs/audit provenance, hash anchors or failure records. Never reuse outputs, implicitly resume, auto-retry, erase negatives, bypass stale gates or delete shadow packages to conceal source mismatch.
- Frozen server dependencies between gate and pilot, deployment target/budgets/benchmark protocol, or unimplemented knobs presented as working ablations.
- Scientific evidence status: single-seed validation pilots are not statistical confirmation, full-scene accuracy, original-paper reproduction, real-world damage detection, novelty proof or Jetson deployment compliance.

This document grants no new experiment authorization. The handoff itself changes documentation only; continuation must preserve the existing research and its complete evidence trail.
