"""Explicit preflight or separately authorized train/validation-only 2x2 pilot."""
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
from torch.utils.data import DataLoader, Subset
import yaml
from bit_sar_v2_entry import pin_source
from next_steps import strict_seed, band_provenance
from smoke_train import select_indices
from smallflood_cd.engine.experiment import _dataset
from smallflood_cd.engine.checkpointing import save_checkpoint
from smallflood_cd.engine.validator import validate, checkpoint_score, SELECTION_PROTOCOL
from smallflood_cd.data.boundary_supervision import build_boundary_supervision
from smallflood_cd.models.registry import build_model
from smallflood_cd.losses import SmallFloodLoss
from smallflood_cd.utils.config import load_experiment_config

DATA_HASHES = {
    'manifest': '51eba373aef67e05b768bfc3c402d15c9fb9c8417c7026c6079a8fc0bea9b3ae',
    'component_stats': '525e4e1354e991983ebd2e3d3f68caad3f9894308a6eae46256a74405c9f2247',
    'normalization_stats': '33ef9a286340e3a58b8fc3623da210d83624c8a5901e9bcd0799e660a6ff625d',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def tensor_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode()); h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def configs():
    cs = {c:load_experiment_config(ROOT / f'configs/experiment/proposed_factorial_v1/{c.lower()}.yaml')
          for c in 'ABCD'}
    normalized=[]
    for cell,c in cs.items():
        expected_loss=dict(bce_weight=.4,tversky_weight=.4,boundary_weight=.2 if cell in 'CD' else 0.,
            use_size_aware=cell in 'BD',tversky_fp=.3,tversky_fn=.7,component_alpha=1.,
            component_gamma=.5,component_weight_cap=4.)
        expected_training=dict(epochs=15,batch_size=8,learning_rate=.0003,weight_decay=.0001,
                               patience=16,gradient_clip_norm=1.)
        if c['loss']!=expected_loss or c['training']!=expected_training:
            raise RuntimeError('Frozen factorial loss/training recipe changed')
        if c['model']['boundary_head'] != (cell in 'CD') or c['base']['seed']!=42 or c['base']['amp']:
            raise RuntimeError('Frozen factorial switches changed')
        x=json.loads(json.dumps(c))
        del x['loss']['use_size_aware'],x['loss']['boundary_weight'],x['model']['boundary_head'],x['protocol']['cell']
        normalized.append(x)
    if any(x!=normalized[0] for x in normalized):
        raise RuntimeError('Non-factor configuration differences')
    return cs


def fingerprint(cs):
    hashes = {}
    for folder in ('src', 'scripts', 'configs'):
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and p.suffix in ('.py', '.yaml'):
                hashes[str(p.relative_to(ROOT))] = sha(p)
    for k, expected in DATA_HASHES.items():
        actual = sha(cs['A']['data'][k])
        if actual != expected:
            raise RuntimeError(f'Frozen {k} changed')
        hashes[k] = actual
    return {'files': hashes, 'configs': cs, 'torch': str(torch.__version__),
            'cuda': torch.version.cuda, 'cudnn':torch.backends.cudnn.version(),
            'python':sys.version,'gpu': torch.cuda.get_device_name()}


def datasets(config, preflight):
    train, _ = _dataset(config, 'train', config['loss'])
    val, _ = _dataset(config, 'validation', config['loss'])
    if (len(train), len(val)) != (19166, 4767):
        raise RuntimeError('Unexpected train/validation counts')
    for ds in (train, val):
        ds.records.sort(key=lambda r: (r.event_id, r.patch_id))
        if len({r.patch_id for r in ds.records}) != len(ds):
            raise RuntimeError('Duplicate patch IDs')
        if any(r.coherence_path or r.uncertain_mask_path for r in ds.records):
            raise RuntimeError('Unexpected coherence/uncertainty inputs')
    if {r.event_id for r in train.records} & {r.event_id for r in val.records}:
        raise RuntimeError('Train/validation event overlap')
    if {r.patch_id for r in train.records} & {r.patch_id for r in val.records}:
        raise RuntimeError('Train/validation patch overlap')
    if preflight:
        manifest = config['data']['manifest']
        train = Subset(train, select_indices(train, manifest, 16, 42))
        val = Subset(val, select_indices(val, manifest, 8, 42))
    return train, val


def run_cell(cell, config, train, val, output, device, epochs):
    """CPU permitted only for injected synthetic unit tests; CLI requires CUDA."""
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(json.dumps(config))
    config['selection'] = dict(SELECTION_PROTOCOL)
    config['training']['epochs'] = epochs
    config['protocol']['execution_approved'] = True
    config['protocol']['preflight'] = epochs == 1
    (output/'config_resolved.yaml').write_text(yaml.safe_dump(config))
    strict_seed(42)
    model = build_model(config['model']).to(device)
    initial = tensor_hash(model.state_dict())
    common = tensor_hash({k:v for k,v in model.state_dict().items() if not k.startswith('boundary_head.')})
    loader = DataLoader(train, batch_size=8, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(42))
    vloader = DataLoader(val, batch_size=8, shuffle=False, num_workers=0,
                         generator=torch.Generator().manual_seed(42))
    opt = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    lc = config['loss']
    loss_fn = SmallFloodLoss(lc['bce_weight'], lc['tversky_weight'], lc['boundary_weight'], .3, .7)
    strict_seed(42)  # Equal post-construction RNG, regardless of optional head.
    best, step, order_hashes = float('-inf'), 0, []
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
                print(f'{cell}: step {step}/{len(loader)*epochs}', flush=True)
        if not batches:
            raise RuntimeError('Empty training loader')
        metrics = validate(model,vloader,device)
        score = checkpoint_score(metrics)
        if not torch.isfinite(torch.tensor(score)):
            raise RuntimeError('Nonfinite validation selection score')
        order_hashes.append(order.hexdigest())
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
    # Trusted self-produced checkpoint; compare state exactly, without a second GPU model.
    restored=torch.load(output/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
    if tensor_hash(restored['model']) != tensor_hash(model.state_dict()):
        raise RuntimeError('Checkpoint state reload mismatch')
    summary={'status':'COMPLETE','cell':cell,'epochs':epochs,'optimizer_steps':step,
             'initial_sha256':initial,'common_initial_sha256':common,'batch_order_sha256':order_hashes,
             'parameter_count':sum(p.numel() for p in model.parameters()),
             'test_used':False,'checkpoint_reload_exact':True,'best_score':best,
             'elapsed_seconds':time.perf_counter()-started,'not_a_deployment_benchmark':True}
    if initial==tensor_hash(model.state_dict()):
        raise RuntimeError('Model state unchanged')
    dump(output/'summary.json',summary)
    return summary


def check_pairing(summaries):
    for s in summaries[1:]:
        if s['common_initial_sha256'] != summaries[0]['common_initial_sha256']:
            raise RuntimeError('Common initialization mismatch')
        if s['batch_order_sha256'] != summaries[0]['batch_order_sha256']:
            raise RuntimeError('Batch order mismatch')
    by_cell={s['cell']:s for s in summaries}
    for a,b in [('A','B'),('C','D')]:
        if a in by_cell and b in by_cell and by_cell[a]['initial_sha256'] != by_cell[b]['initial_sha256']:
            raise RuntimeError('Paired full initialization mismatch')


def verify_gate(path, current):
    gate=json.loads((path/'COMPLETE.json').read_text())
    if gate.get('stage')!='preflight' or gate.get('fingerprint')!=current or gate.get('test_used') is not False:
        raise RuntimeError('Invalid/stale preflight gate')
    if [s.get('cell') for s in gate.get('summaries',[])] != list('ABCD'):
        raise RuntimeError('Incomplete preflight')
    if any(s.get('epochs')!=1 or s.get('optimizer_steps')!=2 or s.get('status')!='COMPLETE'
           or not s.get('checkpoint_reload_exact') for s in gate['summaries']):
        raise RuntimeError('Preflight did not pass')
    check_pairing(gate['summaries'])


def main():
    pin_source()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['test','preflight','run'])
    p.add_argument('--output',type=Path)
    p.add_argument('--preflight',type=Path)
    p.add_argument('--approve-training',action='store_true')
    args=p.parse_args()
    if args.stage=='test':
        import pytest
        raise SystemExit(pytest.main(['--import-mode=importlib','-q',
            'tests/unit/test_proposed_factorial_config.py','tests/unit/test_proposed_factorial_runner.py']))
    if args.output is None or args.output.exists():
        p.error('Provide a new --output directory')
    if args.stage=='run' and (not args.approve_training or args.preflight is None):
        p.error('Full training needs --approve-training and a completed --preflight')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    strict_seed(42)
    cs=configs(); before=fingerprint(cs)
    if args.stage=='run':
        verify_gate(args.preflight,before)
    args.output.mkdir(parents=True,exist_ok=False)
    dump(args.output/'provenance.json',before)
    summaries=[]
    for cell,c in cs.items():
        band_provenance(c)
        train,val=datasets(c,args.stage=='preflight')
        directory=args.output/f'{cell}_proposed'
        dump_env={'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(),
                  'manifest_sha256':DATA_HASHES['manifest'],'validation_patches':len(val),
                  'train_patches':len(train),'test_used':False}
        summary=run_cell(cell,c,train,val,directory,torch.device('cuda'),
                         1 if args.stage=='preflight' else 15)
        dump(directory/'environment.json',dump_env)
        summaries.append(summary); check_pairing(summaries)
        del train,val
        torch.cuda.empty_cache()
    if fingerprint(cs)!=before:
        raise RuntimeError('Source/config/data metadata changed during run')
    dump(args.output/'COMPLETE.json',{'stage':args.stage,'fingerprint':before,
                                    'summaries':summaries,'test_used':False})
    print(f'FACTORIAL {args.stage.upper()} COMPLETE: {args.output}',flush=True)


if __name__=='__main__':
    main()
