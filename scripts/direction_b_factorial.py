"""Seed-parameterized successor of proposed_factorial.py for direction_b_confirmation_v1.

Only the training seed changes. Recipe guards, data guards, pairing checks, fingerprinting
and checkpointing are reused from proposed_factorial.py unchanged. Validation only; never test.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from torch.utils.data import DataLoader
import yaml
from bit_sar_v2_entry import pin_source
from next_steps import strict_seed, band_provenance
import proposed_factorial as pf
from smallflood_cd.engine.checkpointing import save_checkpoint
from smallflood_cd.engine.validator import validate, checkpoint_score, SELECTION_PROTOCOL
from smallflood_cd.data.boundary_supervision import build_boundary_supervision
from smallflood_cd.models.registry import build_model
from smallflood_cd.losses import SmallFloodLoss

PROTOCOL_ID = 'direction_b_confirmation_v1'
SEEDS = (42, 1337, 2026)
CONFIRMATION_SEEDS = (1337, 2026)
ANCHOR_RUN = Path('runs/proposed_factorial_v1/20260927T075837Z')
# Recorded seed-42 anchors (summary.json of the 2026-09-27 factorial run).
ANCHOR_INITIAL = {
    'A': '71c04062460c9639694c4d24f7944b46a909e3fb990340bee8307db5945754b8',
    'B': '71c04062460c9639694c4d24f7944b46a909e3fb990340bee8307db5945754b8',
    'C': 'db03a4cfdbee094bf79e2a082bd135b0b5f7c659cce0a3291886d8ceabd7e476',
    'D': 'db03a4cfdbee094bf79e2a082bd135b0b5f7c659cce0a3291886d8ceabd7e476',
}
ANCHOR_COMMON = '71c04062460c9639694c4d24f7944b46a909e3fb990340bee8307db5945754b8'


def check_seed(seed):
    if seed not in SEEDS:
        raise ValueError(f'Seed {seed} is not in the frozen protocol {SEEDS}')
    return seed


def initial_hashes(config, seed):
    """Initial state exactly as run_cell builds it, without data or training."""
    strict_seed(seed)
    model = build_model(config['model'])
    state = model.state_dict()
    return (pf.tensor_hash(state),
            pf.tensor_hash({k: v for k, v in state.items() if not k.startswith('boundary_head.')}))


class PatchIds(torch.utils.data.Dataset):
    """Same length/order as the training dataset, yielding patch IDs only."""

    def __init__(self, records):
        self.ids = [r.patch_id for r in records]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {'patch_id': self.ids[i]}


def order_hashes(dataset, seed, epochs):
    """Batch-order hashes that run_cell would record; each epoch's iterator is created
    from the same generator, exactly as in training."""
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    hashes = []
    for _ in range(epochs):
        h = hashlib.sha256()
        for batch in loader:
            h.update(json.dumps(batch['patch_id'], separators=(',', ':')).encode())
        hashes.append(h.hexdigest())
    return hashes


def regression(cs, anchor_root=ANCHOR_RUN):
    """Seed 42 through the new code must reproduce the recorded factorial anchors."""
    recorded = {cell: json.loads((anchor_root / f'{cell}_proposed/summary.json').read_text())
                for cell in 'ABCD'}
    result = {}
    for cell, c in cs.items():
        initial, common = initial_hashes(c, 42)
        if initial != ANCHOR_INITIAL[cell] or initial != recorded[cell]['initial_sha256']:
            raise RuntimeError(f'Regression: initial state mismatch for {cell}')
        if common != ANCHOR_COMMON or common != recorded[cell]['common_initial_sha256']:
            raise RuntimeError(f'Regression: common initial state mismatch for {cell}')
        result[cell] = {'initial_sha256': initial, 'common_initial_sha256': common}
    train, _ = pf.datasets(cs['A'], False)
    orders = order_hashes(PatchIds(train.records), 42, 15)
    for cell in 'ABCD':
        if orders != recorded[cell]['batch_order_sha256']:
            raise RuntimeError(f'Regression: batch order mismatch for {cell}')
    result['batch_order_sha256'] = orders
    return result


def run_cell(cell, config, train, val, output, device, epochs, seed):
    """proposed_factorial.run_cell with the hard-coded 42 replaced by `seed`.
    CPU permitted only for injected synthetic unit tests; CLI requires CUDA."""
    check_seed(seed)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(json.dumps(config))
    config['selection'] = dict(SELECTION_PROTOCOL)
    config['training']['epochs'] = epochs
    config['base']['seed'] = seed
    config['protocol'].update(execution_approved=True, preflight=epochs == 1,
                              confirmation_protocol=PROTOCOL_ID)
    (output/'config_resolved.yaml').write_text(yaml.safe_dump(config))
    strict_seed(seed)
    model = build_model(config['model']).to(device)
    initial = pf.tensor_hash(model.state_dict())
    common = pf.tensor_hash({k:v for k,v in model.state_dict().items() if not k.startswith('boundary_head.')})
    loader = DataLoader(train, batch_size=8, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    vloader = DataLoader(val, batch_size=8, shuffle=False, num_workers=0,
                         generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    lc = config['loss']
    loss_fn = SmallFloodLoss(lc['bce_weight'], lc['tversky_weight'], lc['boundary_weight'], .3, .7)
    strict_seed(seed)  # Equal post-construction RNG, regardless of optional head.
    best, step, order_hashes_ = float('-inf'), 0, []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(epochs):
        tick = time.perf_counter(); model.train(); acc = defaultdict(float)
        order = hashlib.sha256(); batches = 0
        for batch in loader:
            order.update(json.dumps(batch['patch_id'], separators=(',', ':')).encode())
            pre, post, target, valid = [batch[k].to(device).float() for k in ('pre','post','target','valid_mask')]
            if not all(torch.isfinite(x).all() for x in (pre,post,target,valid)):
                raise RuntimeError('Nonfinite batch')
            if not torch.all((target==0)|(target==1)) or not torch.all((valid==0)|(valid==1)):
                raise RuntimeError('Nonbinary target/valid')
            if lc['use_size_aware']:
                if 'component_weights' not in batch:
                    raise RuntimeError('Missing size weights (no fallback allowed)')
                weights = batch['component_weights'].to(device)
                if not torch.isfinite(weights).all() or weights.min()<1 or weights.max()>5:
                    raise RuntimeError('Invalid component weights')
            else:
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
            opt.step(); step += 1; batches += 1
            for name in ('total','weighted_bce','weighted_tversky','boundary'):
                acc[name] += float(getattr(losses,name).detach())
            acc['preclip_norm'] += float(norm); acc['clipped_steps'] += int(norm > 1)
            pos = (target>.5)&(valid>.5)
            acc['positive_pixels'] += int(pos.sum())
            acc['positive_weight_mass'] += float(weights[pos].sum())
            acc['positive_pixels_at_cap'] += int((weights[pos]>=5).sum())
            if step % 100 == 0:
                print(f'{cell} s{seed}: step {step}/{len(loader)*epochs}', flush=True)
        if not batches:
            raise RuntimeError('Empty training loader')
        metrics = validate(model,vloader,device)
        score = checkpoint_score(metrics)
        if not torch.isfinite(torch.tensor(score)):
            raise RuntimeError('Nonfinite validation selection score')
        order_hashes_.append(order.hexdigest())
        row = {'epoch':epoch, 'global_step':step, 'train_loss':acc['total']/batches,
               'batch_order_sha256':order.hexdigest(),
               **{f'validation_{k}':v for k,v in metrics.items()}, 'validation_selection_score':score,
               'loss_terms_mean':{k:acc[k]/batches for k in ('weighted_bce','weighted_tversky','boundary')},
               'scaled_loss_terms_mean':{'bce':lc['bce_weight']*acc['weighted_bce']/batches,
                    'tversky':lc['tversky_weight']*acc['weighted_tversky']/batches,
                    'boundary':lc['boundary_weight']*acc['boundary']/batches},
               'preclip_norm_mean':acc['preclip_norm']/batches,
               'clipping_fraction':acc['clipped_steps']/batches,
               'weight_totals':{k:acc[k] for k in ('positive_pixels','positive_weight_mass','positive_pixels_at_cap')}}
        if model.boundary_head is not None:
            s=float(model.boundary_head.residual_scale.detach())
            row.update(residual_scale=s, effective_residual_scale=max(0.,min(1.,s)))
        save_checkpoint(output/'checkpoints/last.ckpt', model,opt,None,epoch,step,max(best,score),config)
        if score > best:
            best=score
            save_checkpoint(output/'checkpoints/best_composite.ckpt',model,opt,None,epoch,step,best,config)
        if device.type=='cuda':
            torch.cuda.synchronize(device)
            row.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
                       peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3)
        row['cycle_seconds']=time.perf_counter()-tick
        with (output/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)
    restored=torch.load(output/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
    if pf.tensor_hash(restored['model']) != pf.tensor_hash(model.state_dict()):
        raise RuntimeError('Checkpoint state reload mismatch')
    summary={'status':'COMPLETE','cell':cell,'seed':seed,'protocol':PROTOCOL_ID,'epochs':epochs,
             'optimizer_steps':step,'initial_sha256':initial,'common_initial_sha256':common,
             'batch_order_sha256':order_hashes_,
             'parameter_count':sum(p.numel() for p in model.parameters()),
             'test_used':False,'checkpoint_reload_exact':True,'best_score':best,
             'elapsed_seconds':time.perf_counter()-started,'not_a_deployment_benchmark':True}
    if initial==pf.tensor_hash(model.state_dict()):
        raise RuntimeError('Model state unchanged')
    pf.dump(output/'summary.json',summary)
    return summary


def verify_gate(path, current, seed):
    pf.verify_gate(path, current)
    gate = json.loads((path/'COMPLETE.json').read_text())
    if gate.get('seed') != seed or gate.get('protocol') != PROTOCOL_ID:
        raise RuntimeError('Preflight gate is for a different seed/protocol')
    if any(s.get('seed') != seed for s in gate['summaries']):
        raise RuntimeError('Preflight cell seed mismatch')


def main():
    pin_source()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['test','regress','preflight','run'])
    p.add_argument('--seed',type=int)
    p.add_argument('--output',type=Path)
    p.add_argument('--preflight',type=Path)
    p.add_argument('--approve-training',action='store_true')
    args=p.parse_args()
    if args.stage=='test':
        import pytest
        raise SystemExit(pytest.main(['--import-mode=importlib','-q','tests/unit/test_direction_b.py',
            'tests/unit/test_proposed_factorial_config.py','tests/unit/test_proposed_factorial_runner.py']))
    if args.seed is None:
        p.error('--seed is required')
    check_seed(args.seed)
    if args.stage=='regress':
        if args.seed != 42 or args.output is None or args.output.exists():
            p.error('regress needs --seed 42 and a new --output path')
        strict_seed(42)
        cs=pf.configs()
        result={'protocol':PROTOCOL_ID,'stage':'regress','seed':42,'test_used':False,
                'training_performed':False,'anchor_run':str(ANCHOR_RUN),**regression(cs)}
        args.output.mkdir(parents=True,exist_ok=False)
        pf.dump(args.output/'REGRESSION_PASS.json',result)
        print(f'FACTORIAL SEED REGRESSION PASSED: {args.output}',flush=True)
        return
    if args.seed not in CONFIRMATION_SEEDS:
        p.error('preflight/run are limited to confirmation seeds 1337 and 2026')
    if args.output is None or args.output.exists():
        p.error('Provide a new --output directory')
    if args.stage=='run' and (not args.approve_training or args.preflight is None):
        p.error('Full training needs --approve-training and a completed --preflight')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    strict_seed(args.seed)
    cs=pf.configs(); before=pf.fingerprint(cs)
    if args.stage=='run':
        verify_gate(args.preflight,before,args.seed)
    args.output.mkdir(parents=True,exist_ok=False)
    pf.dump(args.output/'provenance.json',{**before,'seed':args.seed,'protocol':PROTOCOL_ID})
    summaries=[]
    for cell,c in cs.items():
        band_provenance(c)
        train,val=pf.datasets(c,args.stage=='preflight')
        directory=args.output/f'{cell}_proposed'
        env={'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(),
             'manifest_sha256':pf.DATA_HASHES['manifest'],'validation_patches':len(val),
             'train_patches':len(train),'test_used':False,'seed':args.seed}
        summary=run_cell(cell,c,train,val,directory,torch.device('cuda'),
                         1 if args.stage=='preflight' else 15,args.seed)
        pf.dump(directory/'environment.json',env)
        summaries.append(summary); pf.check_pairing(summaries)
        del train,val
        torch.cuda.empty_cache()
    if pf.fingerprint(cs)!=before:
        raise RuntimeError('Source/config/data metadata changed during run')
    pf.dump(args.output/'COMPLETE.json',{'stage':args.stage,'seed':args.seed,'protocol':PROTOCOL_ID,
                                       'fingerprint':before,'summaries':summaries,'test_used':False})
    print(f'FACTORIAL {args.stage.upper()} s{args.seed} COMPLETE: {args.output}',flush=True)


if __name__=='__main__':
    main()
