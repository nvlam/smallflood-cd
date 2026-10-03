# BIT-SAR v2: batch-8 gate and 15-epoch pilot

## Material Passport

Approved: user approved batch-8 check and a fresh BIT-SAR v2-only 15-epoch pilot.
Status: package prepared; no remote training launched by assistant.
Prior evidence: server GPU batch-2 smoke EXACT MATCH, runs/bit_sar_v2_smoke/20260924T091512Z.
Local targeted tests: 18 passed, 1 CUDA-only test skipped. No GPU feasibility claim yet.

## Run contract

1. Batch-8 gate: 16 train + 8 validation patches, two optimizer steps, strict seed 42,
   finite gradients, changed parameters, exact checkpoint-output/optimizer state reload.
2. Only after success: new Python process, new model/optimizer, full train/validation,
   15 epochs, batch 8, FP32, AdamW LR 1e-4, weight decay 1e-4, clipping 1.0,
   no scheduler, BCE/Tversky 0.5/0.5, no size-aware or boundary loss.
3. Best checkpoint: fixed 0.5 threshold, event_pixel_v2 event-macro F1; legacy filename
   best_composite.ckpt. No test, no resume, no reuse of gate weights, no other models.
4. Stop on any failure; do not change batch, precision, determinism or settings automatically.

Source/config/manifest/normalization hashes must agree between preflight and pilot.
Gate covers metadata consistency, not a full integrity scan of all NPY arrays.
New registry mapping exists only in the pilot process: old bit CLI, scripts and results
are untouched. The source import guard precedes both stages. Server CUDA test/gate
remains necessary; CPU synthetic integration also covers batch 8 at 32×32, not real 256×256.

## Manual installation

On Mac:
```bash
scp /Users/nguyenvulam/academic-research-skills/academic-research-skills-codex/smallflood-cd/artifacts/updates/bit_sar_v2_pilot15_20260925.tar.gz fit.lamnv@100.102.158.114:/data/fit.lamnv/smallflood_cd/
```

On server:
```bash
cd /data/fit.lamnv/smallflood_cd
source .venv/bin/activate
mkdir -p artifacts/code_backups
tar -czf "artifacts/code_backups/before_v2_pilot15_$(date -u +%Y%m%dT%H%M%SZ).tar.gz" scripts/bit_sar_v2_entry.py scripts/smoke_bit_sar_v2.py tests/unit/test_bit_sar_v2_smoke.py
tar -xzf bit_sar_v2_pilot15_20260925.tar.gz
python scripts/bit_sar_v2_entry.py test-pilot
```
Expected server CUDA result: 19 passed. Stop on errors/skips and inspect before continuing.

Inside tmux, from project root:
```bash
bash scripts/run_bit_sar_v2_pilot15.sh
```

Outputs: runs/bit_sar_v2_pilot15/<UTC>/batch8_preflight and
runs/bit_sar_v2_pilot15/<UTC>/pilot/<UTC>_bit_sar_v2.
Logs/source snapshot: artifacts/bit_sar_v2_pilot15/<UTC>/{batch8.log,pilot.log,...}.
Success marker: ALL BIT-SAR v2 PILOT15 CHECKS COMPLETE.
Duration is not yet measured for v2 full-data batch 8; do not reuse old custom BIT timing
as a promise. Observe early epoch times. Detach with Ctrl-b then d, not Ctrl-c.
After completion request validation/object-boundary review integration for v2 separately;
the existing reviewer CLI may not yet accept the new model name. Do not run test/matrix.
