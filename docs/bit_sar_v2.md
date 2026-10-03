# BIT-SAR v2: implementation and validation

## Material Passport

Stage: approved implementation and local unit tests; no remote GPU training or test-set use.
Name: bit_sar_v2. Old registry name bit and old model/checkpoints remain unchanged.
Upstream: justchenhao/BIT_CD commit adcd7aea6f234586ffffdd4e9959404f96271711,
base_transformer_pos_s4_dd8. AI-assisted adaptation and verification.

## Architecture

Use upstream code with narrowly documented compatibility edits in bit_upstream/NOTICE.md.
At 256×256 input: stem/layer1 64×64; layer2/layer3 32×32; nearest ×2 then
256-to-32 channel projection produces 64×64 features. Four semantic tokens per date,
learned token positions, one encoder block, eight shared decoder blocks with residual
cross-attention and feed-forward sublayers. Dimension 32, eight heads, dimension per
head 64, upstream attention scaling retained. Difference features are upsampled ×4
before two-layer convolutional classification into two logits at full resolution.

Official ResNet18 stride-replacement behavior is preserved, including its BasicBlock
choice not to use dilation >1. No high-resolution skip branch is invented.

## Explicit adaptations

- Replace hard-coded RGB stem with two SAR input channels [VH,VV].
- Keep two-class head. Project one binary logit as z_change minus z_nochange, so
  sigmoid(binary) equals softmax(two-class)[change]. This is NOT argmax conversion.
- Existing BCE/Tversky loss can consume this adapter, but is not the original CE recipe.
- Fresh normal(0,0.02) Conv/Linear initialization, normal(1,0.02) BN scales, zero biases;
  BN running mean/variance reset. Token position parameters retain constructor initialization.
  No pretrained weights are downloaded; pretrained=true is rejected. This does not reproduce
  any pretrained BN buffers surviving the upstream initialization call chain.
- Remove unexecuted resnet.layer4, avgpool and fc; parity checks verify unchanged outputs.
  Parameter count is 3,492,642 retained trainable parameters, not upstream's stored dead weights.
- Remove transient visualization token attributes after each call to avoid retaining graphs.
- Require equal N×2×H×W inputs, H/W>=32 divisible by 8; reject coherence explicitly.

The old custom BIT is 11,360,133 parameters; do not compare its accuracy/time to this new
parameter count. Proposed's count is 1,080,805. New efficiency comparisons require new
measurements; no latency or GPU memory estimates are established by these CPU checks.

## Tests and scope

tests/unit/test_bit_sar_v2.py checks registry separation/config loading, feature and classifier
resolution at 256, decoder depth and identity behavior with zeroed residual branches,
fixed-weight original-source forward parity at 32/64/256, input-output probability adapter,
all retained parameter gradient parity under CE at 32, finite project-loss gradients and
an optimizer step, strict checkpoint reload, and invalid input/initialization rejection.

Reference files under tests/reference/bit_upstream are pinned original sources with SHA256
checks. In-memory test shims adapt old imports and disable downloads; only input conv is
replaced for SAR. Fixed weights are copied with a checked missing-key allowlist limited
to unused layer4/fc. Reference model retains those unused modules.
Forward/gradient parity tolerance: atol=1e-6, rtol=1e-5. Checkpoint-output comparison exact.
CPU tests establish implementation equivalence under tested conditions, not official
paper accuracy, CUDA determinism, 15-epoch reproducibility, or deployment performance.
Reference code may emit SyntaxWarning for old string-identity expressions; vendored runtime
uses equality instead. Source reference remains untouched for provenance.

## Integration boundaries

Added model and experiment YAMLs. Engineering YAML preserves baseline BCE/Tversky and
LR settings for later controlled checks; it is not approval for long training and does
not claim original SGD/200-epoch reproduction. Existing next_steps.py choices, smoke_train.py,
pilot15 shell launcher and 135-run matrix are intentionally unchanged. Do not substitute
the new YAML silently into an old run or attempt to load old bit checkpoints into v2.

New dependency: einops==0.8.1. Upstream code permits research/non-commercial use only;
retain notices and cite Chen, Qi, Shi, DOI 10.1109/TGRS.2021.3095166.

## Server validation (no dataset or GPU job)

After backing up registry.py and pyproject.toml and extracting the update at the project root:

```bash
python -m pip install -e .
python -m pytest -q tests/unit/test_bit_sar_v2.py tests/unit/test_baselines.py
```

These tests use CPU synthetic tensors and temporary test checkpoints only. Never invoke
the upstream main_cd.py or run_cd.sh as part of this update. A separately approved GPU
smoke and reproducibility step is still required before any longer pilot.
