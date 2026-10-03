# BIT official-source audit — 2026-09-24

## Material Passport

Status: source/code comparison completed; no implementation or configuration changed,
no upstream code executed, no training/test evaluation. AI-assisted source verification.
Local target: src/smallflood_cd/models/baselines/bit.py and pilot15 resolved BIT config.
Upstream: justchenhao/BIT_CD, master resolved through GitHub API to
adcd7aea6f234586ffffdd4e9959404f96271711. Audit target is the supplied
base_transformer_pos_s4_dd8 recipe, not every possible BIT variant.

## Verdict

Current project baseline is BIT-inspired, not a faithful implementation of that official
configuration. Earlier engineering runs remain valid records of the model actually run,
but must not be presented as accuracy/efficiency results of an official BIT reproduction.
The missing structural features are not merely adaptations needed for two-channel SAR.

## Architecture findings

| Item | Official dd8 configuration | Current project |
|---|---|---|
| Backbone forward | Through layer3; output stride 8 | Through layer4; stride 32 |
| Pixel features for transformer, 256 input | 64×64 after ×2 upsampling | 8×8 |
| Feature dimension | 32 | 64 |
| Tokens per date | 4 | 4 |
| Token positional embedding | Learned | Absent |
| Encoder depth | 1 | 2 |
| Attention heads | 8 | 4 |
| Decoder | 8 residual cross-attention + FFN blocks | One bare MultiheadAttention |
| Classification | Upsample features, then classify | Classify 8×8, then upsample logits |

Official spatial dimensions above are inferred from source strides and forward logic,
not an upstream runtime test. Local 8×8 dimensions were confirmed by the previous CPU
shape check. Sources: [networks.py](https://github.com/justchenhao/BIT_CD/blob/adcd7aea6f234586ffffdd4e9959404f96271711/models/networks.py#L111-L334),
[resnet.py](https://github.com/justchenhao/BIT_CD/blob/adcd7aea6f234586ffffdd4e9959404f96271711/models/resnet.py#L117-L189),
[local implementation](../src/smallflood_cd/models/baselines/bit.py).

Official decoder preserves the input pixel representation via residual additions,
with normalization and feed-forward transformations. Local _decode_tokens returns only
attention output. The official attention also uses custom projections and scaling;
generic torch MultiheadAttention is not a drop-in numerical equivalent. Sources:
[help_funcs.py](https://github.com/justchenhao/BIT_CD/blob/adcd7aea6f234586ffffdd4e9959404f96271711/models/help_funcs.py#L15-L172).

Important porting nuance: official ResNet18 BasicBlock suppresses dilation >1 and
constructs ordinary 3×3 convolutions, while stride replacement still removes downsampling.
Do not describe it simply as a standard torchvision dilated ResNet18 or assume the same
constructor flags reproduce its behavior. Source:
[BasicBlock and layer construction](https://github.com/justchenhao/BIT_CD/blob/adcd7aea6f234586ffffdd4e9959404f96271711/models/resnet.py#L32-L49).

## Training and adaptation findings

Official supplied launch recipe: RGB LEVIR data, 256 patches, batch 8, 200 epochs,
learning rate 0.01, linear schedule. Trainer uses SGD, momentum 0.9, weight decay 5e-4;
CLI loss defaults to cross-entropy and model output has two classes. The project pilot
uses two SAR channels, AdamW 1e-4, weight decay 1e-4, 15 epochs, no scheduler and
one-logit BCE/Tversky. These are substantial declared protocol adaptations, not proof
that one optimizer is intrinsically more fair. Sources:
[run_cd.sh](https://github.com/justchenhao/BIT_CD/blob/adcd7aea6f234586ffffdd4e9959404f96271711/scripts/run_cd.sh),
[trainer.py](https://github.com/justchenhao/BIT_CD/blob/adcd7aea6f234586ffffdd4e9959404f96271711/models/trainer.py#L32-L76),
[main_cd.py](https://github.com/justchenhao/BIT_CD/blob/adcd7aea6f234586ffffdd4e9959404f96271711/main_cd.py#L29-L70).

Do not infer retained ImageNet initialization merely from pretrained=True: upstream
define_G subsequently invokes init_net/init_weights over Conv/Linear/BatchNorm modules.
That call path resets parameters, although BN running buffers need separate attention.
Initialization must therefore be explicitly specified in any adaptation, not copied
from a constructor flag. Source: networks.py lines 62–127 and 142–144 linked above.

Data-split warning: run_cd.sh sets split=trainval and split_val=test; main_cd.py invokes
test after training. Do not run that orchestration unchanged in this project. Preserve
our frozen train/validation split and protected test set. This observation concerns the
published sample script, not a claim about every experiment in the original paper.

## Implications and next approval

The coarse local logits and absence of pixel residual decoder are plausible contributors
to poor small-object/boundary metrics; the audit cannot establish their separate causal
effects or guarantee that a faithful port will improve performance. SAR domain, loss,
initialization, training duration and learning rate remain alternative explanations.

Recommend a separately named, source-pinned BIT SAR adaptation, leaving old code and
checkpoints intact. Preserve official spatial/token/decoder architecture; document input
channel and output/loss adapters. For a two-class head, a one-logit interface can use
z_change - z_nochange (sigmoid equals two-class softmax probability), but BCE/Tversky
training is still an explicit departure from original CE training.

Before GPU work, require structural shape tests, decoder-depth/residual tests, fixed-weight
numerical comparison against a compatibility-wrapped upstream model, gradient/checkpoint
round-trip tests, and then a strict-reproducibility smoke run. Compare parameter counts
and latency only after defining the repaired architecture; 11.36M is the old custom model's
count, not an established official-BIT count. Handle unused upstream backbone modules
transparently rather than silently changing the parameter-count convention.

These are recommendations only. No port, dependency installation, experiment launch or
hyperparameter change has been performed. Research gap remains unproven: correcting BIT
does not resolve Proposed's current deficits against FC-Siam-Diff.

## Retrieval and evidence limits

Located official repository via search, then read author-owned raw source files and live
GitHub commit metadata. [Paper record](https://arxiv.org/abs/2103.00208) identifies this
repository. Source code is primary evidence for implementation behavior, not independent
replication of reported accuracy. No full-paper numerical reproduction, license audit,
runtime compatibility certification, or official GPU benchmark was performed.
