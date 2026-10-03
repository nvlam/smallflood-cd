# Experiment execution contract

The matrix at `configs/experiment/experiment_matrix.yaml` is the single inventory of
models, mandatory ablations, grids, and fixed seeds. The runner expands grids
deterministically by sorted dotted key, then executes every seed in listed order.

Each run writes:

- `config_resolved.yaml`: exact post-grid, post-seed configuration;
- `environment.json`: Python, platform, PyTorch, and seed;
- `metrics.jsonl`: append-only epoch training/validation log;
- `checkpoints/last.ckpt` and `checkpoints/best_composite.ckpt`;
- `metrics.csv`: one machine-readable final metric row;
- `summary.json`: the same row with artifact paths.

The matrix root writes `execution_plan.json` before execution and updates
`all_results.csv` after every successful run. No result analysis, ranking, or model
selection occurs in this layer.

The CSV schema includes pixel IoU/precision/recall/F1, strict and tolerance-aware
boundary metrics, component metrics, small-component recall, inference timing,
parameter count, checkpoint path, and run directory.

Reproducibility controls include fixed seeds, deterministic PyTorch mode, immutable
event splits, resolved configuration capture, content-addressed component caches, and
checkpointed RNG state. Running into an existing run directory is refused to prevent
silent overwrite.
