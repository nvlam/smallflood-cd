"""Bounded BIT-SAR v2 smoke: one epoch, 16 train/8 validation patches; never test."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from next_steps import band_provenance, exact_equal, strict_seed
from smoke_train import select_indices
from smallflood_cd.engine.checkpointing import load_model_state
from smallflood_cd.engine.experiment import _dataset
from smallflood_cd.engine.trainer import fit
from smallflood_cd.engine.validator import SELECTION_PROTOCOL
from smallflood_cd.losses import SmallFloodLoss
from smallflood_cd.models.registry import build_model
from smallflood_cd.utils.config import load_experiment_config

from bit_sar_v2_entry import verify_loaded_modules

# Stop before any training if a helper script imported a stale package.
verify_loaded_modules()


def source_hashes():
    paths = [Path(__file__), Path('scripts/next_steps.py'), Path('scripts/smoke_train.py'),
             Path('scripts/bit_sar_v2_entry.py')]
    paths += sorted(Path('src/smallflood_cd').rglob('*.py'))
    return {str(p.relative_to(Path.cwd()) if p.is_absolute() else p):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def state_on_cpu(value):
    """Compare checkpoint values without device-placement differences or dtype casts."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: state_on_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [state_on_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(state_on_cpu(item) for item in value)
    return value


def check_gradients(model):
    grads = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not parameter.grad.isfinite().all():
            raise RuntimeError(f'Missing/nonfinite gradient: {name}')
        grads.append(parameter.grad.detach().float().norm())
    norm = torch.stack(grads).norm().item() if grads else 0.0
    if not math.isfinite(norm) or norm == 0:
        raise RuntimeError('Zero/nonfinite aggregate gradient')
    return norm


def run_smoke(directory, *, config=None, device=None, batch_size=2):
    if batch_size not in (2, 8):
        raise ValueError('Supported smoke batch sizes: 2 or 8')
    expected_steps = 16 // batch_size
    # CPU injection is for unit tests only; CLI always requests CUDA.
    device = torch.device('cuda') if device is None else torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no automatic CPU fallback')
    strict_seed(42)
    config = config or load_experiment_config('configs/experiment/bit_sar_v2.yaml')
    if config['model']['name'] != 'bit_sar_v2':
        raise ValueError('This smoke accepts bit_sar_v2 only')
    provenance = band_provenance(config)
    config['base'].update(seed=42, device=str(device), num_workers=0, amp=False)
    config['training'].update(epochs=1, batch_size=batch_size, patience=1)
    config['selection'] = dict(SELECTION_PROTOCOL)
    config['smoke'] = {'version': 'bit_sar_v2_smoke_v1', 'train_patches': 16,
                       'validation_patches': 8, 'test_used': False,
                       'strict_deterministic': True, 'band_provenance': provenance}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'config_resolved.yaml').write_text(yaml.safe_dump(config))
    train, _ = _dataset(config, 'train', config['loss'])
    val, _ = _dataset(config, 'validation', config['loss'])
    ti = select_indices(train, config['data']['manifest'], 16, 42)
    vi = select_indices(val, config['data']['manifest'], 8, 42)
    subset = {'train': [train.records[i].patch_id for i in ti],
              'validation': [val.records[i].patch_id for i in vi],
              'selection': 'half positive/half negative, seed 42', 'test_used': False}
    if set(subset['train']) & set(subset['validation']):
        raise ValueError('Train/validation patch IDs overlap')
    (directory / 'subset.json').write_text(json.dumps(subset, indent=2))
    train_loader = DataLoader(Subset(train, ti), batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(Subset(val, vi), batch_size=batch_size, shuffle=False, num_workers=0)
    environment = {'python': sys.version, 'torch': str(torch.__version__),
                   'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                   'device': str(device),
                   'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
                   'manifest_sha256': hashlib.sha256(Path(config['data']['manifest']).read_bytes()).hexdigest(),
                   'source_sha256': source_hashes(), 'band_provenance': provenance}
    (directory / 'environment.json').write_text(json.dumps(environment, indent=2))
    model = build_model(config['model']).to(device)
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']),
                                  weight_decay=float(config['training']['weight_decay']))
    steps = []
    def inspect(opt, args, kwargs):
        norm = check_gradients(model)
        steps.append(norm)
        print(f'bit_sar_v2: step {len(steps)}/{expected_steps}; gradient norm={norm:.6f}', flush=True)
    def check_output(module, inputs, output):
        logits = output.change_logits
        if logits.shape != (inputs[0].shape[0], 1, *inputs[0].shape[-2:]):
            raise RuntimeError('Unexpected output shape')
        if not logits.isfinite().all():
            raise RuntimeError('Nonfinite model output')
    hook = optimizer.register_step_pre_hook(inspect)
    output_hook = model.register_forward_hook(check_output)
    c = config['loss']
    loss = SmallFloodLoss(c['bce_weight'], c['tversky_weight'], c['boundary_weight'],
                         c['tversky_fp'], c['tversky_fn'])
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        result = fit(model, train_loader, val_loader, loss, optimizer, None, device,
                     1, directory, config, patience=1,
                     gradient_clip_norm=float(config['training']['gradient_clip_norm']))
    finally:
        hook.remove()
        output_hook.remove()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    allocated = torch.cuda.max_memory_allocated()/1024**3 if device.type == 'cuda' else None
    reserved = torch.cuda.max_memory_reserved()/1024**3 if device.type == 'cuda' else None
    torch.save({'cpu': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []},
               directory / 'rng_after_training.pt')
    changed = sum(not torch.equal(initial[n], p.detach().cpu()) for n,p in model.named_parameters())
    rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
    if len(steps) != expected_steps or not changed or len(rows) != 1 or not math.isfinite(rows[0]['train_loss']):
        raise RuntimeError('Training smoke verification failed')
    model.eval()
    batch = next(iter(val_loader))
    pre, post = batch['pre'].to(device), batch['post'].to(device)
    with torch.no_grad():
        expected = model(pre, post).change_logits.cpu()
    restored = build_model(config['model']).to(device).eval()
    checkpoint = load_model_state(result.best_checkpoint, restored, device)
    with torch.no_grad():
        actual = restored(pre, post).change_logits.cpu()
    exact_equal(expected, actual, 'checkpoint_output')
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    restored_optimizer.load_state_dict(checkpoint['optimizer'])
    # AdamW step counters can reside on CPU while moment estimates reside on CUDA.
    # map_location/load_state_dict can change placement. Preserve dtype and exact values.
    exact_equal(state_on_cpu(optimizer.state_dict()),
                state_on_cpu(restored_optimizer.state_dict()), 'optimizer_reload')
    # Timings/memory are descriptive only and are excluded from the equality comparison.
    checks = {'status': 'PASS', 'model': 'bit_sar_v2', 'batch_size': batch_size,
              'optimizer_steps': len(steps),
              'gradient_norms': steps, 'changed_parameter_tensors': changed,
              'parameter_count': model.parameter_count(), 'checkpoint_reload_exact': True,
              'test_used': False, 'train_loss': rows[0]['train_loss']}
    (directory / 'checks.json').write_text(json.dumps(checks, indent=2))
    summary = {**checks, 'elapsed_train_validation_seconds': elapsed,
               'peak_allocated_gib': allocated, 'peak_reserved_gib': reserved,
               'memory_scope': 'training/validation before second model for reload',
               'not_a_deployment_benchmark': True}
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    (directory / 'COMPLETE.json').write_text(json.dumps({'status': 'PASS', 'test_used': False}))
    print(json.dumps(summary, indent=2), flush=True)


def compare(left, right):
    left, right = Path(left), Path(right)
    if left.resolve() == right.resolve():
        raise ValueError('Comparison requires two different run directories')
    for directory in (left, right):
        if json.loads((directory / 'COMPLETE.json').read_text()) != {'status': 'PASS', 'test_used': False}:
            raise ValueError('Both runs must have completed successfully without test use')
    for filename in ('COMPLETE.json', 'subset.json', 'checks.json', 'environment.json',
                     'config_resolved.yaml', 'metrics.jsonl'):
        exact_equal((left / filename).read_text(), (right / filename).read_text(), filename)
    for filename in ('best_composite.ckpt', 'last.ckpt'):
        a = torch.load(left / 'checkpoints' / filename, map_location='cpu', weights_only=False)
        b = torch.load(right / 'checkpoints' / filename, map_location='cpu', weights_only=False)
        exact_equal(a, b, filename)
    exact_equal(torch.load(left / 'rng_after_training.pt', weights_only=True),
                torch.load(right / 'rng_after_training.pt', weights_only=True), 'rng_after_training')
    print('BIT-SAR v2: EXACT MATCH', flush=True)
    print('REPRODUCIBILITY CHECK PASSED (these two smoke runs only)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--output-dir', type=Path, required=True)
    comp = sub.add_parser('compare')
    comp.add_argument('left', type=Path)
    comp.add_argument('right', type=Path)
    args = parser.parse_args()
    if args.command == 'run':
        run_smoke(args.output_dir)
    else:
        compare(args.left, args.right)


if __name__ == '__main__':
    main()
