"""Isolated, gated A-control/R pilot. Never constructs a test dataset."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from audit_candidate_r_eligibility import batch_schedule
from bit_sar_v2_entry import pin_source, verify_loaded_modules
from next_steps import band_provenance, strict_seed
import proposed_factorial as factorial
from smoke_train import select_indices
from smallflood_cd.engine.checkpointing import save_checkpoint
from smallflood_cd.engine.validator import SELECTION_PROTOCOL, checkpoint_score, validate
from smallflood_cd.losses.local_component import CandidateRLoss
from smallflood_cd.models.registry import build_model
from smallflood_cd.utils.config import load_experiment_config

CELLS = ('A_control', 'R')
PROTOCOL = {
    'id': 'candidate_r_v1', 'stage': 'diagnostic_validation_only',
    'execution_approved': False, 'test_allowed': False, 'scheduler': 'none',
    'shared_initialization': 'exact_full_state_required',
}
LOCAL = {'small_area_threshold': 35, 'radius': 3, 'connectivity': 8,
         'aggregation': 'eligible_component_mean_across_minibatch'}
sha, dump, tensor_hash = factorial.sha, factorial.dump, factorial.tensor_hash


def configs():
    """Permit exactly one difference: auxiliary coefficient 0 versus .1."""
    reference = factorial.configs()['A']
    result = {}
    for cell in CELLS:
        expected = deepcopy(reference)
        expected['protocol'] = {**PROTOCOL, 'cell': cell}
        expected['local_supervision'] = {**LOCAL, 'coefficient': 0. if cell == 'A_control' else .1}
        actual = load_experiment_config(ROOT / f'configs/experiment/candidate_r_v1/{cell.lower()}.yaml')
        if actual != expected:
            raise RuntimeError(f'Frozen Candidate R configuration changed: {cell}')
        result[cell] = actual
    base = reference['base']
    if any(base[k] != v for k, v in dict(seed=42, deterministic=True, device='cuda',
                                       amp=False, num_workers=0).items()):
        raise RuntimeError('Frozen execution settings changed')
    if reference['model']['pretrained'] or reference['model']['include_coherence']:
        raise RuntimeError('Only fresh SAR-only A backbone is allowed')
    return result


def load_audit(path):
    """Validate saved train-only audit; never scan validation/test labels here."""
    report = json.loads((path / 'report.json').read_text())
    details = json.loads((path / 'patch_counts.json').read_text())
    if report.get('status') != 'COMPLETE' or report.get('train_patches') != 19166:
        raise RuntimeError('Incomplete eligibility audit')
    for key in ('test_used', 'validation_arrays_used', 'training_performed', 'model_loaded'):
        if report.get(key) is not False:
            raise RuntimeError(f'Eligibility audit scope mismatch: {key}')
    if report.get('metadata_sha256') != {
        'manifest': factorial.DATA_HASHES['manifest'],
        'stats': factorial.DATA_HASHES['component_stats'],
    }:
        raise RuntimeError('Eligibility metadata mismatch')
    if report.get('loss_source_sha256') != sha(ROOT / 'src/smallflood_cd/losses/local_component.py'):
        raise RuntimeError('Eligibility loss implementation changed; audit again')
    if report.get('script_sha256') != sha(ROOT / 'scripts/audit_candidate_r_eligibility.py'):
        raise RuntimeError('Eligibility audit implementation changed')
    if len(details) != 19166 or len({r['patch_id'] for r in details}) != len(details):
        raise RuntimeError('Invalid eligibility patch IDs/count')
    if details != sorted(details, key=lambda r: (r['event_id'], r['patch_id'])):
        raise RuntimeError('Eligibility record order changed')
    for row in details:
        for key in ('eligible', 'empty_rings', 'small_clipped', 'eligible_pixels'):
            if type(row[key]) is not int or row[key] < 0:
                raise RuntimeError('Invalid eligibility count')
        if row['eligible'] > row['small_clipped'] or row['empty_rings'] > row['eligible']:
            raise RuntimeError('Inconsistent eligibility count')
    observed = {
        'patches': len(details),
        'patches_with_eligible': sum(r['eligible'] > 0 for r in details),
        **{k: sum(r[k] for r in details)
           for k in ('eligible', 'small_clipped', 'empty_rings', 'eligible_pixels')},
    }
    approved = dict(patches=19166, patches_with_eligible=1420, eligible=3521,
                    small_clipped=13497, empty_rings=0, eligible_pixels=95594)
    if observed != approved or any(report['counts'].get(k) != v for k, v in observed.items()):
        raise RuntimeError('Eligibility counts differ from approved audit')
    if report['counts'].get('foreground_pixels') != 23436067:
        raise RuntimeError('Audit foreground count changed')
    if report.get('prior_A_batch_order_matches') is not True:
        raise RuntimeError('Eligibility audit did not verify historical A order')
    if batch_schedule(details) != report.get('epochs'):
        raise RuntimeError('Eligibility batch schedule mismatch')
    return report, details


def fingerprint(cs, audit_path):
    # Reuse the conservative full source/config fingerprint, without changing it.
    result = factorial.fingerprint({'A': cs['A_control'], 'R': cs['R']})
    result['audit'] = {name: sha(audit_path / name) for name in ('report.json', 'patch_counts.json')}
    result['tests'] = {name: sha(ROOT / 'tests/unit' / name) for name in (
        'test_candidate_r_runner.py', 'test_local_component_loss.py', 'test_candidate_r_eligibility.py')}
    result['runtime'] = {'numpy': np.__version__, 'scipy': scipy.__version__,
                         'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
                         'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                         'torch_num_threads': torch.get_num_threads()}
    return result


def datasets(config, details, preflight):
    train, val = factorial.datasets(config, False)  # only train and validation
    ids = [(r.event_id, r.patch_id) for r in train.records]
    if ids != [(r['event_id'], r['patch_id']) for r in details]:
        raise RuntimeError('Train records differ from eligibility audit')
    if preflight:
        # Deterministic enriched subset, diagnostic only; full-run sampler unchanged.
        active = [i for i, r in enumerate(details) if r['eligible'] > 0][:8]
        empty = [i for i, r in enumerate(details) if r['eligible'] == 0][:8]
        if len(active) != 8 or len(empty) != 8:
            raise RuntimeError('Need eight eligible and eight empty preflight patches')
        indices = sorted(active + empty)
        train = Subset(train, indices)
        val = Subset(val, select_indices(val, config['data']['manifest'], 8, 42))
    return train, val


def gradient_diagnostics(base, local, logits, coefficient):
    """Output-logit gradients only; autograd.grad must not populate parameter .grad."""
    gb = torch.autograd.grad(base, logits, retain_graph=True)[0]
    gl = torch.autograd.grad(local, logits, retain_graph=True)[0]
    if not torch.isfinite(gb).all() or not torch.isfinite(gl).all():
        raise RuntimeError('Nonfinite diagnostic logit gradient')
    nb, nl = float(gb.norm()), float(gl.norm())
    return {'base_logit_grad_l2': nb, 'local_logit_grad_l2': nl,
            'applied_aux_logit_grad_l2': coefficient * nl,
            'applied_aux_base_logit_grad_ratio': coefficient * nl / nb if nb else None}


def run_cell(cell, config, train, val, output, device, epochs, *, eligible=None,
             expected_schedule=None):
    """CPU use is for injected synthetic tests only; the CLI requires CUDA."""
    if cell not in CELLS or config['protocol']['cell'] != cell:
        raise ValueError('Unknown or mismatched cell')
    coefficient = config['local_supervision']['coefficient']
    if coefficient != (0. if cell == 'A_control' else .1):
        raise ValueError('Frozen auxiliary coefficient changed')
    if config['model']['boundary_head'] or config['loss']['use_size_aware']:
        raise ValueError('Candidate requires boundary/size weights OFF')
    output.mkdir(parents=True, exist_ok=False)
    config = deepcopy(config)
    config['selection'] = dict(SELECTION_PROTOCOL)
    config['training']['epochs'] = epochs
    config['protocol'].update(execution_approved=True, preflight=epochs == 1)
    (output / 'config_resolved.yaml').write_text(yaml.safe_dump(config))
    strict_seed(42)
    model = build_model(config['model']).to(device)
    initial = tensor_hash(model.state_dict())
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != 1079865:
        raise RuntimeError('Frozen A-backbone parameter count changed')
    loader = DataLoader(train, batch_size=8, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(42))
    vloader = DataLoader(val, batch_size=8, shuffle=False, num_workers=0,
                         generator=torch.Generator().manual_seed(42))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    criterion = CandidateRLoss()
    strict_seed(42)
    best, step, order_hashes, diagnostic_history = float('-inf'), 0, [], []
    active_total = 0
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(epochs):
        tick = time.perf_counter()
        model.train()
        acc, order, batches = defaultdict(float), hashlib.sha256(), 0
        for batch in loader:
            order.update(json.dumps(batch['patch_id'], separators=(',', ':')).encode())
            if any(k in batch for k in ('coherence', 'uncertain_mask', 'component_weights')):
                raise RuntimeError('Unexpected optional supervision/input')
            pre, post, target, valid = [batch[k].to(device).float()
                                       for k in ('pre', 'post', 'target', 'valid_mask')]
            if not all(torch.isfinite(x).all() for x in (pre, post, target, valid)):
                raise RuntimeError('Nonfinite training input')
            optimizer.zero_grad(set_to_none=True)
            out = model(pre, post)
            if out.change_logits.shape != target.shape:
                raise RuntimeError('Invalid logits shape')
            losses = criterion(out, target, valid)
            supervision = losses.supervision
            if eligible is not None:
                expected = tuple(eligible[p] for p in batch['patch_id'])
                if supervision.components_per_image != expected:
                    raise RuntimeError('Actual mask eligibility differs from audit')
            base = .4 * losses.bce + .4 * losses.tversky
            total = base if cell == 'A_control' else losses.total
            if not torch.isfinite(total):
                raise RuntimeError('Nonfinite training loss')
            dg = gradient_diagnostics(base, losses.local, out.change_logits, coefficient)
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError('Diagnostics unexpectedly populated parameter gradients')
            if supervision.components and dg['local_logit_grad_l2'] <= 0:
                raise RuntimeError('Eligible batch has zero local logit gradient')
            if not supervision.components and (float(losses.local.detach()) != 0 or
                                                dg['local_logit_grad_l2'] != 0):
                raise RuntimeError('Empty batch auxiliary must be exactly zero')
            total.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad):
                raise RuntimeError('Missing/nonfinite parameter gradient')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if norm <= 0:
                raise RuntimeError('Zero aggregate parameter gradient')
            optimizer.step()
            step += 1
            batches += 1
            active = int(supervision.components > 0)
            acc['active_batches'] += active
            acc['eligible_components'] += supervision.components
            acc['empty_rings'] += supervision.empty_rings
            acc['preclip_norm'] += float(norm)
            acc['clipped_steps'] += int(norm > 1)
            for name, value in dict(total=total, bce=losses.bce, tversky=losses.tversky,
                                    local=losses.local).items():
                acc[name] += float(value.detach())
            for name, value in dg.items():
                if value is not None:
                    acc[name] += value
                    acc[name + '_defined'] += 1
            if step % 100 == 0:
                print(f'{cell}: step {step}/{len(loader) * epochs}', flush=True)
        if not batches:
            raise RuntimeError('Empty training loader')
        schedule = dict(epoch=epoch, batches=batches, empty_batches=batches-int(acc['active_batches']),
                        eligible_total=int(acc['eligible_components']), batch_order_sha256=order.hexdigest())
        if expected_schedule is not None and any(expected_schedule[epoch][k] != v
                                                 for k, v in schedule.items()):
            raise RuntimeError('Epoch differs from frozen audit batch schedule')
        active_total += int(acc['active_batches'])
        order_hashes.append(order.hexdigest())
        metrics = validate(model, vloader, device)
        score = checkpoint_score(metrics)
        row = {
            'epoch': epoch, 'global_step': step, 'train_loss': acc['total'] / batches,
            'batch_order_sha256': order.hexdigest(),
            **{f'validation_{k}': v for k, v in metrics.items()}, 'validation_selection_score': score,
            'loss_terms_mean': {k: acc[k] / batches for k in ('bce', 'tversky', 'local')},
            'scaled_loss_terms_mean': {'bce': .4 * acc['bce'] / batches,
                                      'tversky': .4 * acc['tversky'] / batches,
                                      'local': coefficient * acc['local'] / batches},
            'local_loss_active_batch_mean': acc['local'] / acc['active_batches'] if acc['active_batches'] else None,
            'eligibility': {**schedule, 'active_batch_fraction': acc['active_batches'] / batches,
                            'empty_rings': int(acc['empty_rings'])},
            'logit_gradient_means': {k: acc[k] / acc[k + '_defined'] if acc[k + '_defined'] else None
                                    for k in dg},
            'gradient_scope': 'output logits; means over minibatches; not parameter-gradient shares',
            'preclip_norm_mean': acc['preclip_norm'] / batches,
            'clipping_fraction': acc['clipped_steps'] / batches,
        }
        save_checkpoint(output / 'checkpoints/last.ckpt', model, optimizer, None,
                        epoch, step, max(best, score), config)
        if score > best:  # Earlier checkpoint wins ties.
            best = score
            save_checkpoint(output / 'checkpoints/best_composite.ckpt', model, optimizer, None,
                            epoch, step, best, config)
        diagnostic_history.append(deepcopy(row))  # Excludes timing and memory from exact repeat check.
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            row.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
                       peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3)
        row['cycle_seconds'] = time.perf_counter() - tick
        with (output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)
    if not active_total:
        raise RuntimeError('No eligible supervision exercised; cannot pass preflight/run')
    final = tensor_hash(model.state_dict())
    if final == initial:
        raise RuntimeError('Model state unchanged')
    restored = torch.load(output / 'checkpoints/last.ckpt', map_location='cpu', weights_only=False)
    if tensor_hash(restored['model']) != final:
        raise RuntimeError('Checkpoint model state mismatch')
    exact_cpu(optimizer.state_dict(), restored['optimizer'])
    # Actual strict load plus hash, not just a successful torch.load call.
    model.load_state_dict(restored['model'], strict=True)
    if tensor_hash(model.state_dict()) != final:
        raise RuntimeError('Checkpoint model reload mismatch')
    summary = dict(status='COMPLETE', cell=cell, epochs=epochs, optimizer_steps=step,
                   initial_sha256=initial, final_sha256=final, batch_order_sha256=order_hashes,
                   parameter_count=parameter_count, test_used=False, checkpoint_reload_exact=True,
                   active_batches=active_total, best_score=best,
                   elapsed_seconds=time.perf_counter()-started, not_a_deployment_benchmark=True,
                   diagnostics_sha256=hashlib.sha256(json.dumps(diagnostic_history, sort_keys=True,
                                                               allow_nan=False).encode()).hexdigest(),
                   checkpoint_sha256={n: sha(output / 'checkpoints' / n)
                                      for n in ('last.ckpt', 'best_composite.ckpt')})
    dump(output / 'summary.json', summary)
    return summary


def exact_cpu(left, right):
    """Exact dtype/shape/value comparison, portable across checkpoint CPU/GPU storage."""
    if isinstance(left, torch.Tensor):
        if (not isinstance(right, torch.Tensor) or left.dtype != right.dtype or
                left.shape != right.shape or not torch.equal(left.detach().cpu(), right.detach().cpu())):
            raise RuntimeError('Checkpoint tensor mismatch')
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise RuntimeError('Checkpoint mapping mismatch')
        for key in left:
            exact_cpu(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise RuntimeError('Checkpoint sequence mismatch')
        for a, b in zip(left, right):
            exact_cpu(a, b)
    elif left != right:
        raise RuntimeError('Checkpoint scalar mismatch')


def check_pairing(summaries):
    for summary in summaries[1:]:
        for key in ('initial_sha256', 'batch_order_sha256', 'parameter_count'):
            if summary[key] != summaries[0][key]:
                raise RuntimeError(f'Paired {key} mismatch')


def check_repeat(first, second):
    for key in ('initial_sha256', 'final_sha256', 'batch_order_sha256', 'diagnostics_sha256'):
        if first[key] != second[key]:
            raise RuntimeError(f'R repeat mismatch: {key}')


def check_repeat_checkpoints(path):
    for filename in ('last.ckpt', 'best_composite.ckpt'):
        left = torch.load(path / 'R/checkpoints' / filename, map_location='cpu', weights_only=False)
        right = torch.load(path / 'R_repeat/checkpoints' / filename, map_location='cpu', weights_only=False)
        exact_cpu(left, right)


def verify_gate(path, current):
    gate = json.loads((path / 'COMPLETE.json').read_text())
    if (gate.get('stage') != 'preflight' or gate.get('fingerprint') != current or
            gate.get('test_used') is not False or gate.get('repeat_exact') is not True):
        raise RuntimeError('Invalid/stale preflight gate')
    summaries = gate.get('summaries', [])
    if [s.get('cell') for s in summaries] != ['A_control', 'R', 'R']:
        raise RuntimeError('Incomplete preflight')
    for directory, summary in zip(('A_control', 'R', 'R_repeat'), summaries):
        if (summary.get('status') != 'COMPLETE' or summary.get('epochs') != 1 or
                summary.get('optimizer_steps') != 2 or summary.get('active_batches', 0) <= 0 or
                summary.get('checkpoint_reload_exact') is not True or summary.get('test_used') is not False):
            raise RuntimeError('Preflight did not pass')
        if json.loads((path / directory / 'summary.json').read_text()) != summary:
            raise RuntimeError('Preflight summary changed')
        for filename in ('last.ckpt', 'best_composite.ckpt'):
            if sha(path / directory / 'checkpoints' / filename) != summary['checkpoint_sha256'][filename]:
                raise RuntimeError('Preflight checkpoint changed')
    check_pairing(summaries)
    check_repeat(summaries[1], summaries[2])


def main():
    pin_source()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('test', 'preflight', 'run'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--approve-training', action='store_true')
    args = parser.parse_args()
    if args.stage == 'test':
        import pytest
        code = pytest.main(['--import-mode=importlib', '-q',
                           'tests/unit/test_candidate_r_runner.py',
                           'tests/unit/test_local_component_loss.py',
                           'tests/unit/test_candidate_r_eligibility.py'])
        verify_loaded_modules()
        raise SystemExit(code)
    if args.output is None or args.output.exists() or args.audit is None:
        parser.error('Provide a new --output directory and completed --audit directory')
    if args.stage == 'run' and (not args.approve_training or args.preflight is None):
        parser.error('Full training needs --approve-training and completed --preflight')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    strict_seed(42)
    cs = configs()
    report, details = load_audit(args.audit)
    before = fingerprint(cs, args.audit)
    if args.stage == 'run':
        verify_gate(args.preflight, before)
    provenance = band_provenance(cs['A_control'])
    train, val = datasets(cs['A_control'], details, args.stage == 'preflight')
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / 'provenance.json', before)
    dump(args.output / 'eligibility_report.json', report)
    if args.stage == 'preflight':
        dump(args.output / 'subset.json', {
            'train': [train.dataset.records[i].patch_id for i in train.indices],
            'validation': [val.dataset.records[i].patch_id for i in val.indices],
            'selection': 'first eight audited eligible and first eight empty; sorted original order',
        })
    summaries = []
    for directory, cell in [('A_control', 'A_control'), ('R', 'R')] + (
            [('R_repeat', 'R')] if args.stage == 'preflight' else []):
        destination = args.output / directory
        summary = run_cell(cell, cs[cell], train, val, destination, torch.device('cuda'),
                           1 if args.stage == 'preflight' else 15,
                           eligible={r['patch_id']: r['eligible'] for r in details},
                           expected_schedule=None if args.stage == 'preflight' else report['epochs'])
        dump(destination / 'environment.json', {
            **{k: before[k] for k in ('torch', 'cuda', 'cudnn', 'python', 'gpu', 'runtime')},
            'manifest_sha256': factorial.DATA_HASHES['manifest'], 'band_provenance': provenance,
            'train_patches': len(train), 'validation_patches': len(val), 'test_used': False,
        })
        summaries.append(summary)
        check_pairing(summaries)
        torch.cuda.empty_cache()
    if args.stage == 'preflight':
        check_repeat(summaries[1], summaries[2])
        check_repeat_checkpoints(args.output)
    verify_loaded_modules()
    if fingerprint(cs, args.audit) != before:
        raise RuntimeError('Source/config/audit/data metadata changed during run')
    dump(args.output / 'COMPLETE.json', {
        'stage': args.stage, 'fingerprint': before, 'summaries': summaries,
        'repeat_exact': args.stage == 'preflight', 'test_used': False,
        'engineering_only': True, 'not_a_deployment_benchmark': True,
    })
    print(f'CANDIDATE R {args.stage.upper()} COMPLETE: {args.output}', flush=True)


if __name__ == '__main__':
    main()
