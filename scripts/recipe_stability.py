"""recipe_stability_v1: learning-rate x schedule screening for factorial arm A.

docs/recipe_stability_protocol_v1_20261006.md. Trains on 10 training events and evaluates
on an internal dev split (Canada, Japan) built from training events only. The current
validation events and the test set are never loaded.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from torch.utils.data import DataLoader, Subset
import yaml
from bit_sar_v2_entry import pin_source
from next_steps import strict_seed, band_provenance
from smoke_train import select_indices
import proposed_factorial as pf
from direction_b_factorial import ANCHOR_INITIAL, ANCHOR_RUN, PatchIds, order_hashes
from smallflood_cd.engine.experiment import _dataset
from smallflood_cd.engine.checkpointing import save_checkpoint
from smallflood_cd.engine.validator import validate, checkpoint_score, SELECTION_PROTOCOL
from smallflood_cd.data.boundary_supervision import build_boundary_supervision
from smallflood_cd.models.registry import build_model
from smallflood_cd.losses import SmallFloodLoss

PROTOCOL_ID = 'recipe_stability_v1'
RECIPES = {'R0': (3e-4, 'constant'), 'R1': (1e-4, 'constant'),
           'R2': (3e-4, 'warmup_cosine'), 'R3': (1e-4, 'warmup_cosine')}
SEEDS = (101, 102, 103)
EPOCHS = 15
NUM_WORKERS = 4
AP_BINS = 10000
VALIDATION_EVENTS = {'20161011_Lumberton', '20220302_Coraki_Australia', '20230805_Hebei'}
LOCATION_GROUPS = {'Houston': ('20160419_Houston', '20170830_Houston'),
                   'Sydney': ('20210324_Sydney', '20220705_Sydney'),
                   'Somalia': ('20180508_Somalia', '20231114_Beledweyne')}
EXPECTED_DEV = ('20190502_Canada', '20191012_Japan')
EXPECTED_COUNTS = {'train_all': 19166, 'screen_train': 16107, 'dev': 3059}
# Stability rules (protocol section 5), frozen at approval.
MAX_RANGE_PP = 5.0
JAPAN_FLOOR = 0.5
TIE_PP = 1.0


def train_event_flood_fractions(manifest):
    """Label statistics of training rows only (manifest columns; no arrays)."""
    valid, flood = defaultdict(int), defaultdict(int)
    with Path(manifest).open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['split'] != 'train':
                continue
            valid[row['event_id']] += int(row['valid_pixels'])
            flood[row['event_id']] += int(row['flood_pixels'])
    return {e: flood[e] / valid[e] for e in valid}


def dev_split(fractions):
    """Protocol section 3 rule; returns the held-out events."""
    groups = {g: tuple(es) for g, es in LOCATION_GROUPS.items()}
    grouped = {e for es in groups.values() for e in es}
    for event in fractions:
        if event not in grouped:
            groups[event] = (event,)
    if set(e for es in groups.values() for e in es) != set(fractions):
        raise RuntimeError('Location groups do not match training events')
    order = sorted(groups, key=lambda g: hashlib.sha256(f'internal_dev_v1:42:{g}'.encode()).hexdigest())
    k = math.ceil(0.2 * len(groups))
    chosen = order[:k]
    low = lambda g: all(fractions[e] < 0.01 for e in groups[g])
    if not any(low(g) for g in chosen):
        chosen[-1] = next(g for g in order if low(g))
    return tuple(sorted(e for g in chosen for e in groups[g]))


def datasets(config, preflight):
    """Training split only; screening train and internal dev by event."""
    train, _ = _dataset(config, 'train', config['loss'])
    if len(train) != EXPECTED_COUNTS['train_all']:
        raise RuntimeError('Unexpected training count')
    train.records.sort(key=lambda r: (r.event_id, r.patch_id))
    events = {r.event_id for r in train.records}
    if events & VALIDATION_EVENTS or any(r.split != 'train' for r in train.records):
        raise RuntimeError('Validation/test records reached the screening loader')
    if any(r.coherence_path or r.uncertain_mask_path for r in train.records):
        raise RuntimeError('Unexpected coherence/uncertainty inputs')
    dev_events = dev_split(train_event_flood_fractions(config['data']['manifest']))
    if dev_events != EXPECTED_DEV:
        raise RuntimeError(f'Dev split differs from protocol: {dev_events}')
    screen, dev = copy.copy(train), copy.copy(train)
    screen.records = [r for r in train.records if r.event_id not in dev_events]
    dev.records = [r for r in train.records if r.event_id in dev_events]
    if (len(screen), len(dev)) != (EXPECTED_COUNTS['screen_train'], EXPECTED_COUNTS['dev']):
        raise RuntimeError('Unexpected screening/dev counts')
    if preflight:
        manifest = config['data']['manifest']
        screen = Subset(screen, select_indices(screen, manifest, 16, 42))
        dev = Subset(dev, select_indices(dev, manifest, 8, 42))
    return screen, dev


def lr_lambda(schedule, steps_per_epoch, total_steps):
    if schedule == 'constant':
        return None
    if schedule != 'warmup_cosine':
        raise ValueError(schedule)
    warmup = steps_per_epoch

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
    return factor


def _worker_init(worker_id):
    import random
    import numpy as np
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)


def loader(dataset, shuffle, seed, workers):
    extra = dict(worker_init_fn=_worker_init, multiprocessing_context='fork') if workers else {}
    return DataLoader(dataset, batch_size=8, shuffle=shuffle, num_workers=workers,
                      generator=torch.Generator().manual_seed(seed), **extra)


def run_recipe(name, config, train, dev, output, device, epochs, seed, lr, schedule, workers):
    """direction_b_factorial.run_cell with learning rate, schedule and workers as inputs;
    `dev` replaces the validation loader. CPU only for injected synthetic tests."""
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(json.dumps(config))
    config['selection'] = dict(SELECTION_PROTOCOL)
    config['training'].update(epochs=epochs, learning_rate=lr, schedule=schedule, num_workers=workers)
    config['base']['seed'] = seed
    config['protocol'].update(execution_approved=True, preflight=epochs == 1,
                              recipe_protocol=PROTOCOL_ID, recipe=name,
                              evaluation='internal_dev', dev_events=list(EXPECTED_DEV))
    (output/'config_resolved.yaml').write_text(yaml.safe_dump(config))
    strict_seed(seed)
    model = build_model(config['model']).to(device)
    initial = pf.tensor_hash(model.state_dict())
    common = pf.tensor_hash({k:v for k,v in model.state_dict().items() if not k.startswith('boundary_head.')})
    tloader = loader(train, True, seed, workers)
    dloader = loader(dev, False, seed, workers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.0001)
    factor = lr_lambda(schedule, len(tloader), len(tloader) * epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor) if factor else None
    lc = config['loss']
    loss_fn = SmallFloodLoss(lc['bce_weight'], lc['tversky_weight'], lc['boundary_weight'], .3, .7)
    strict_seed(seed)  # Equal post-construction RNG, as in the factorial runner.
    best, step, hashes = float('-inf'), 0, []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(epochs):
        tick = time.perf_counter(); model.train(); acc = defaultdict(float)
        order = hashlib.sha256(); batches = 0
        for batch in tloader:
            order.update(json.dumps(batch['patch_id'], separators=(',', ':')).encode())
            pre, post, target, valid = [batch[k].to(device).float() for k in ('pre','post','target','valid_mask')]
            if not all(torch.isfinite(x).all() for x in (pre,post,target,valid)):
                raise RuntimeError('Nonfinite batch')
            if not torch.all((target==0)|(target==1)) or not torch.all((valid==0)|(valid==1)):
                raise RuntimeError('Nonbinary target/valid')
            if lc['use_size_aware']:
                raise RuntimeError('Arm A only: size weighting must be off')
            weights = torch.ones_like(target)
            boundary = build_boundary_supervision(target, valid)
            opt.zero_grad(set_to_none=True)
            out = model(pre,post)
            if out.change_logits.shape != target.shape or not torch.isfinite(out.change_logits).all():
                raise RuntimeError('Invalid model logits')
            losses = loss_fn(out,target,weights,valid,boundary.target,boundary.valid_mask)
            if not torch.isfinite(losses.total):
                raise RuntimeError('Nonfinite loss')
            losses.total.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
                raise RuntimeError('Missing/nonfinite parameter gradient')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if norm <= 0:
                raise RuntimeError('Zero aggregate gradient')
            acc['lr'] += opt.param_groups[0]['lr']
            opt.step(); step += 1; batches += 1
            if sched is not None:
                sched.step()
            for k in ('total','weighted_bce','weighted_tversky','boundary'):
                acc[k] += float(getattr(losses,k).detach())
            acc['preclip_norm'] += float(norm); acc['clipped_steps'] += int(norm > 1)
            if step % 100 == 0:
                print(f'{name} s{seed}: step {step}/{len(tloader)*epochs}', flush=True)
        if not batches:
            raise RuntimeError('Empty training loader')
        metrics = validate(model,dloader,device)
        score = checkpoint_score(metrics)
        if not math.isfinite(score):
            raise RuntimeError('Nonfinite dev score')
        hashes.append(order.hexdigest())
        row = {'epoch':epoch, 'global_step':step, 'train_loss':acc['total']/batches,
               'batch_order_sha256':order.hexdigest(), 'lr_mean':acc['lr']/batches,
               **{f'dev_{k}':v for k,v in metrics.items()}, 'dev_selection_score':score,
               'preclip_norm_mean':acc['preclip_norm']/batches,
               'clipping_fraction':acc['clipped_steps']/batches}
        save_checkpoint(output/'checkpoints/last.ckpt', model,opt,sched,epoch,step,max(best,score),config)
        if score > best:
            best=score
            save_checkpoint(output/'checkpoints/best_composite.ckpt',model,opt,sched,epoch,step,best,config)
        if device.type=='cuda':
            torch.cuda.synchronize(device)
            row.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3)
        row['cycle_seconds']=time.perf_counter()-tick
        with (output/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)
    restored=torch.load(output/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
    if pf.tensor_hash(restored['model']) != pf.tensor_hash(model.state_dict()):
        raise RuntimeError('Checkpoint state reload mismatch')
    final = pf.tensor_hash(model.state_dict())
    if initial == final:
        raise RuntimeError('Model state unchanged')
    summary={'status':'COMPLETE','recipe':name,'seed':seed,'protocol':PROTOCOL_ID,'epochs':epochs,
             'learning_rate':lr,'schedule':schedule,'num_workers':workers,'optimizer_steps':step,
             'initial_sha256':initial,'common_initial_sha256':common,'final_sha256':final,
             'batch_order_sha256':hashes,'parameter_count':sum(p.numel() for p in model.parameters()),
             'validation_events_loaded':False,'test_used':False,'checkpoint_reload_exact':True,
             'best_score':best,'elapsed_seconds':time.perf_counter()-started}
    pf.dump(output/'summary.json',summary)
    return summary


def binned_ap(pos_hist, neg_hist):
    """Average precision from per-bin counts, highest-probability bin first."""
    total = int(pos_hist.sum())
    if total == 0:
        return None
    tp = torch.cumsum(pos_hist.flip(0), 0).double()
    fp = torch.cumsum(neg_hist.flip(0), 0).double()
    precision = tp / (tp + fp).clamp(min=1)
    delta = pos_hist.flip(0).double() / total
    return float((precision * delta).sum())


@torch.inference_mode()
def dev_average_precision(model, dloader, device):
    pos = defaultdict(lambda: torch.zeros(AP_BINS, dtype=torch.int64))
    neg = defaultdict(lambda: torch.zeros(AP_BINS, dtype=torch.int64))
    model.eval()
    for batch in dloader:
        prob = model(batch['pre'].to(device), batch['post'].to(device)).change_logits.sigmoid().cpu()
        valid, truth = batch['valid_mask'] > .5, batch['target'] > .5
        bins = (prob * AP_BINS).long().clamp(0, AP_BINS - 1)
        for i, event in enumerate(batch['event_id']):
            v = valid[i]
            pos[event] += torch.bincount(bins[i][v & truth[i]], minlength=AP_BINS)
            neg[event] += torch.bincount(bins[i][v & ~truth[i]], minlength=AP_BINS)
    events = {e: binned_ap(pos[e], neg[e]) for e in sorted(pos)}
    defined = [v for v in events.values() if v is not None]
    return {'events': events, 'macro': sum(defined)/len(defined) if defined else None,
            'bins': AP_BINS, 'note': 'binned AP, diagnostic only; not used for decisions'}


def ap_report(run_dir, config, dev, device, workers):
    report = {}
    dloader = loader(dev, False, config['base']['seed'], workers)
    for name in ('last.ckpt', 'best_composite.ckpt'):
        ckpt = torch.load(run_dir/'checkpoints'/name, map_location='cpu', weights_only=False)
        model = build_model(dict(config['model'], pretrained=False)).to(device)
        model.load_state_dict(ckpt['model'], strict=True)
        report[name] = {'epoch_1based': int(ckpt['epoch'])+1, **dev_average_precision(model, dloader, device)}
    pf.dump(run_dir/'dev_ap.json', report)
    return report


def regression(cs, output):
    """Seed 42 anchors (no validation loading) and the workers 0 vs 4 equivalence."""
    c = cs['A']
    strict_seed(42)
    initial = pf.tensor_hash(build_model(c['model']).state_dict())
    if initial != ANCHOR_INITIAL['A']:
        raise RuntimeError('Regression: initial state mismatch')
    train, _ = _dataset(c, 'train', c['loss'])
    train.records.sort(key=lambda r: (r.event_id, r.patch_id))
    recorded = json.loads((ANCHOR_RUN/'A_proposed/summary.json').read_text())['batch_order_sha256']
    if order_hashes(PatchIds(train.records), 42, 15) != recorded:
        raise RuntimeError('Regression: full-train batch order mismatch')
    screen, dev = datasets(c, True)
    runs = {}
    for w in (0, NUM_WORKERS):
        runs[w] = run_recipe('R0', c, screen, dev, output/f'workers{w}', torch.device('cuda'), 1, 42,
                             *RECIPES['R0'], w)
    keys = ('initial_sha256', 'final_sha256', 'batch_order_sha256', 'best_score', 'optimizer_steps')
    if any(runs[0][k] != runs[NUM_WORKERS][k] for k in keys):
        raise RuntimeError('Regression: workers 0 and 4 differ; use workers 0')
    return {'initial_sha256': initial, 'full_train_batch_order_matches_anchor': True,
            'workers_equivalent': True, 'final_sha256': runs[0]['final_sha256']}


def decide(root):
    """Protocol section 5 on the 12 completed runs (last checkpoint)."""
    rows = {}
    for name in RECIPES:
        for seed in SEEDS:
            d = root/f'{name}_s{seed}'
            s = json.loads((d/'summary.json').read_text())
            if s['status'] != 'COMPLETE' or s['epochs'] != EPOCHS or s['validation_events_loaded']:
                raise RuntimeError(f'Incomplete or invalid run {d}')
            last = [json.loads(x) for x in (d/'metrics.jsonl').read_text().splitlines()][-1]
            rows[(name, seed)] = {'macro': 100*last['dev_event_macro_f1'],
                                  'japan': 100*last['dev_event/20191012_Japan/f1'],
                                  'canada': 100*last['dev_event/20190502_Canada/f1']}
    table = {}
    for name in RECIPES:
        macro = [rows[(name, s)]['macro'] for s in SEEDS]
        japan = sorted(rows[(name, s)]['japan'] for s in SEEDS)
        median = japan[1]
        stable = (max(macro) - min(macro) <= MAX_RANGE_PP
                  and all(j >= JAPAN_FLOOR * median for j in japan))
        table[name] = {'per_seed': {s: rows[(name, s)] for s in SEEDS}, 'mean_macro': sum(macro)/3,
                       'range_macro': max(macro)-min(macro), 'japan_median': median, 'stable': stable}
    stable = [n for n in RECIPES if table[n]['stable']]
    selected = None
    if stable:
        top = max(table[n]['mean_macro'] for n in stable)
        tied = [n for n in stable if top - table[n]['mean_macro'] < TIE_PP]
        preference = ['R0', 'R1', 'R3', 'R2']  # R0, then lower LR, then constant schedule
        selected = next(n for n in preference if n in tied)
    return {'protocol': PROTOCOL_ID, 'checkpoint': 'last (epoch 15)', 'recipes': table,
            'stable_recipes': stable, 'selected_recipe': selected,
            'outcome': 'selected' if selected else 'no stable recipe: stop and report',
            'validation_events_loaded': False, 'test_used': False}


def main():
    pin_source()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['test', 'regress', 'preflight', 'run', 'decide'])
    p.add_argument('--recipe', choices=list(RECIPES))
    p.add_argument('--seed', type=int)
    p.add_argument('--output', type=Path)
    p.add_argument('--preflight', type=Path)
    p.add_argument('--approve-training', action='store_true')
    a = p.parse_args()
    if a.stage == 'test':
        import pytest
        raise SystemExit(pytest.main(['--import-mode=importlib', '-q', 'tests/unit/test_recipe_stability.py']))
    if a.output is None:
        p.error('--output is required')
    if a.stage == 'decide':
        result = decide(a.output)
        pf.dump(a.output/'DECISION.json', result)
        print(json.dumps(result, indent=2), flush=True)
        return
    if a.output.exists():
        p.error('Provide a new --output path')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    cs = pf.configs(); before = pf.fingerprint(cs); c = cs['A']
    band_provenance(c)
    if a.stage == 'regress':
        a.output.mkdir(parents=True)
        result = {'protocol': PROTOCOL_ID, 'stage': 'regress', 'seed': 42, **regression(cs, a.output),
                  'validation_events_loaded': False, 'test_used': False}
        pf.dump(a.output/'REGRESSION_PASS.json', result)
        print(f'RECIPE REGRESSION PASSED: {a.output}', flush=True)
        return
    if a.stage == 'preflight':
        a.output.mkdir(parents=True)
        screen, dev = datasets(c, True)
        summaries = [run_recipe(n, c, screen, dev, a.output/n, torch.device('cuda'), 1, SEEDS[0], lr, s, NUM_WORKERS)
                     for n, (lr, s) in RECIPES.items()]
        if any(s['optimizer_steps'] != 2 for s in summaries) or pf.fingerprint(cs) != before:
            raise RuntimeError('Preflight did not pass')
        pf.dump(a.output/'COMPLETE.json', {'stage': 'preflight', 'protocol': PROTOCOL_ID, 'fingerprint': before,
                                         'summaries': summaries, 'test_used': False})
        print(f'RECIPE PREFLIGHT COMPLETE: {a.output}', flush=True)
        return
    if a.recipe is None or a.seed not in SEEDS or not a.approve_training or a.preflight is None:
        p.error('run needs --recipe, --seed (101|102|103), --approve-training and --preflight')
    gate = json.loads((a.preflight/'COMPLETE.json').read_text())
    if gate.get('stage') != 'preflight' or gate.get('fingerprint') != before or gate.get('protocol') != PROTOCOL_ID:
        raise RuntimeError('Invalid or stale preflight gate')
    screen, dev = datasets(c, False)
    lr, schedule = RECIPES[a.recipe]
    summary = run_recipe(a.recipe, c, screen, dev, a.output, torch.device('cuda'), EPOCHS, a.seed,
                         lr, schedule, NUM_WORKERS)
    if summary['optimizer_steps'] != EPOCHS * math.ceil(EXPECTED_COUNTS['screen_train'] / 8):
        raise RuntimeError('Unexpected optimizer step count')
    ap_report(a.output, yaml.safe_load((a.output/'config_resolved.yaml').read_text()), dev,
              torch.device('cuda'), NUM_WORKERS)
    if pf.fingerprint(cs) != before:
        raise RuntimeError('Source/config/data metadata changed during run')
    print(f'RECIPE RUN COMPLETE: {a.recipe} s{a.seed} {a.output}', flush=True)


if __name__ == '__main__':
    main()
