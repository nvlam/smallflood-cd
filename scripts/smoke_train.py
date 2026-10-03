"""One-epoch integration test on a small train/validation subset; never loads test."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from smallflood_cd.engine.checkpointing import load_model_state
from smallflood_cd.engine.experiment import _dataset
from smallflood_cd.engine.reproducibility import seed_everything
from smallflood_cd.engine.trainer import fit
from smallflood_cd.losses import SmallFloodLoss
from smallflood_cd.models.registry import build_model
from smallflood_cd.utils.config import load_experiment_config


CONFIGS = {
    'proposed': 'configs/experiment/ablations/full_proposed.yaml',
    'fc_siam_diff': 'configs/experiment/fc_siam_diff.yaml',
    'bit': 'configs/experiment/bit.yaml',
}


def select_indices(dataset, manifest, count, seed):
    with Path(manifest).open(newline='') as handle:
        meta = {r['patch_id']: r for r in csv.DictReader(handle)}
    positives, negatives = [], []
    for index, record in enumerate(dataset.records):
        (positives if int(meta[record.patch_id]['flood_pixels']) > 0 else negatives).append(index)
    if len(positives) < count // 2 or len(negatives) < count - count // 2:
        raise ValueError('Not enough positive and negative patches for the smoke subset')
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    chosen = positives[:count // 2] + negatives[:count - count // 2]
    rng.shuffle(chosen)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-patches', type=int, default=32)
    parser.add_argument('--val-patches', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--output-root', type=Path, default=Path('runs/smoke'))
    args = parser.parse_args()
    if args.batch_size < 2 or min(args.train_patches, args.val_patches) < 2:
        parser.error('Batch size and subset sizes must be at least 2')
    if args.train_patches % args.batch_size:
        parser.error('Train patches must be divisible by batch size')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    device = torch.device(args.device)
    output = args.output_root / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for name in args.models:
        seed_everything(args.seed)
        config = load_experiment_config(CONFIGS[name])
        config['base'].update(seed=args.seed, device=args.device, num_workers=0, amp=False)
        config['training'].update(epochs=1, batch_size=args.batch_size, patience=1)
        train, _ = _dataset(config, 'train', config['loss'])
        validation, _ = _dataset(config, 'validation', config['loss'])
        selected_train = select_indices(train, config['data']['manifest'], args.train_patches, args.seed)
        selected_val = select_indices(validation, config['data']['manifest'], args.val_patches, args.seed)
        directory = output / name
        directory.mkdir()
        (directory / 'config_resolved.yaml').write_text(yaml.safe_dump(config))
        (directory / 'subset.json').write_text(json.dumps({
            'purpose': 'integration_smoke_not_scientific_results',
            'selection': 'half patches with flood pixels, half without; fixed seed',
            'train': [train.records[i].patch_id for i in selected_train],
            'validation': [validation.records[i].patch_id for i in selected_val],
            'test_used': False,
        }, indent=2))
        train_loader = DataLoader(Subset(train, selected_train), batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)
        val_loader = DataLoader(Subset(validation, selected_val), batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
        model = build_model(config['model']).to(device)
        initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        optimizer = torch.optim.AdamW(model.parameters(),
                                     lr=float(config['training']['learning_rate']),
                                     weight_decay=float(config['training']['weight_decay']))
        step_checks = []

        def inspect_gradients(opt, _args, _kwargs):
            grads = [p.grad for group in opt.param_groups for p in group['params'] if p.grad is not None]
            if not grads or not torch.stack([g.isfinite().all() for g in grads]).all().item():
                raise RuntimeError('Missing or nonfinite gradients')
            norm = torch.stack([g.detach().float().norm() for g in grads]).norm().item()
            if norm == 0:
                raise RuntimeError('All gradients are zero')
            step_checks.append(norm)
            print(f'{name}: optimizer step {len(step_checks)}, gradient norm={norm:.6f}', flush=True)

        hook = optimizer.register_step_pre_hook(inspect_gradients)
        loss_cfg = config['loss']
        loss = SmallFloodLoss(float(loss_cfg['bce_weight']), float(loss_cfg['tversky_weight']),
                              float(loss_cfg['boundary_weight']), float(loss_cfg['tversky_fp']),
                              float(loss_cfg['tversky_fn']))
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        print(f'\nSTART {name}: {args.train_patches} train, {args.val_patches} validation patches', flush=True)
        trained = fit(model, train_loader, val_loader, loss, optimizer, None, device,
                      1, directory, config, patience=1,
                      gradient_clip_norm=float(config['training']['gradient_clip_norm']))
        hook.remove()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == 'cuda' else None
        changed = sum(not torch.equal(initial[n], p.detach().cpu())
                      for n, p in model.named_parameters() if p.requires_grad)
        if not changed or len(step_checks) != args.train_patches // args.batch_size:
            raise RuntimeError('Optimizer updates were not verified')
        logs = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
        if not all(math.isfinite(r['train_loss']) for r in logs):
            raise RuntimeError('Nonfinite training loss')
        model.eval()
        batch = next(iter(val_loader))
        pre, post = batch['pre'].to(device), batch['post'].to(device)
        with torch.inference_mode():
            expected = model(pre, post).change_logits.detach().cpu()
        restored = build_model(config['model']).to(device).eval()
        checkpoint = load_model_state(trained.best_checkpoint, restored, device)
        with torch.inference_mode():
            actual = restored(pre, post).change_logits.detach().cpu()
        if not torch.isfinite(actual).all():
            raise RuntimeError('Nonfinite checkpoint output')
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        if not checkpoint['optimizer']['state']:
            raise RuntimeError('Optimizer state was not saved')
        result = {'model': name, 'status': 'PASS', 'optimizer_steps': len(step_checks),
                  'changed_parameter_tensors': changed, 'train_loss': logs[-1]['train_loss'],
                  'elapsed_train_and_validation_seconds': elapsed,
                  'pytorch_peak_allocated_gib': peak, 'checkpoint': str(trained.best_checkpoint),
                  'checkpoint_reload_verified': True, 'test_used': False,
                  'not_for_model_comparison_or_deployment_benchmark': True}
        (directory / 'smoke_summary.json').write_text(json.dumps(result, indent=2))
        results.append(result)
        (output / 'summary.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        del model, restored, optimizer, checkpoint, initial, batch, pre, post
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print(f'\nALL SMOKE TRAINING CHECKS PASSED: {output}', flush=True)


if __name__ == '__main__':
    main()
