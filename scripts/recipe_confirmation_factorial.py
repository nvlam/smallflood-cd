"""recipe_confirmation_v1 factorial runner (phase 2 of recipe_stability).

docs/recipe_confirmation_protocol_v1_20261008.md. direction_b_factorial.run_cell with the
learning rate fixed to recipe R1 (1e-4, constant) and num_workers chosen by the seed-42
regression. Trains on all 12 training events; validation only; never test.
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
import yaml
from bit_sar_v2_entry import pin_source
from next_steps import strict_seed, band_provenance
import proposed_factorial as pf
import direction_b_factorial as dbf
import recipe_stability as rs
from smallflood_cd.engine.checkpointing import save_checkpoint
from smallflood_cd.engine.validator import validate, checkpoint_score, SELECTION_PROTOCOL
from smallflood_cd.data.boundary_supervision import build_boundary_supervision
from smallflood_cd.models.registry import build_model
from smallflood_cd.losses import SmallFloodLoss

PROTOCOL_ID = 'recipe_confirmation_v1'
SEEDS = (201, 202, 203)
REGRESSION_SEED = 42
LR = 1e-4                # recipe R1 (recipe_stability_v1 DECISION.json)
HISTORICAL_LR = 3e-4     # frozen factorial recipe; seed-42 regressions only
EPOCHS = 15
FULL_STEPS = 35940       # 15 x ceil(19166 / 8)
WORKERS = 4              # used only if the A and C workers regressions both pass
WORKER_CELLS = ('A', 'C')


def check_seed_lr(seed, lr):
    """Training seeds use R1 only; seed 42 is allowed for regressions (either rate)."""
    if seed in SEEDS and lr == LR:
        return
    if seed == REGRESSION_SEED and lr in (LR, HISTORICAL_LR):
        return
    raise ValueError(f'Seed {seed} with learning rate {lr} is outside protocol {PROTOCOL_ID}')


def run_cell(cell, config, train, val, output, device, epochs, seed, lr, workers):
    """direction_b_factorial.run_cell with learning rate and workers as inputs.
    CPU permitted only for injected synthetic unit tests; CLI requires CUDA."""
    check_seed_lr(seed, lr)
    if workers not in (0, WORKERS):
        raise ValueError(f'num_workers must be 0 or {WORKERS}')
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(json.dumps(config))
    config['selection'] = dict(SELECTION_PROTOCOL)
    config['training'].update(epochs=epochs, learning_rate=lr, num_workers=workers)
    config['base']['seed'] = seed
    config['protocol'].update(execution_approved=True, preflight=epochs == 1,
                              recipe_protocol=PROTOCOL_ID, recipe='R1' if lr == LR else 'R0')
    (output/'config_resolved.yaml').write_text(yaml.safe_dump(config))
    strict_seed(seed)
    model = build_model(config['model']).to(device)
    initial = pf.tensor_hash(model.state_dict())
    common = pf.tensor_hash({k:v for k,v in model.state_dict().items() if not k.startswith('boundary_head.')})
    loader = rs.loader(train, True, seed, workers)
    vloader = rs.loader(val, False, seed, workers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.0001)
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
               'batch_order_sha256':order.hexdigest(), 'learning_rate':lr,
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
    final = pf.tensor_hash(model.state_dict())
    if initial == final:
        raise RuntimeError('Model state unchanged')
    summary={'status':'COMPLETE','cell':cell,'seed':seed,'protocol':PROTOCOL_ID,'epochs':epochs,
             'learning_rate':lr,'schedule':'constant','num_workers':workers,
             'optimizer_steps':step,'initial_sha256':initial,'common_initial_sha256':common,
             'final_sha256':final,'batch_order_sha256':order_hashes_,
             'parameter_count':sum(p.numel() for p in model.parameters()),
             'test_used':False,'checkpoint_reload_exact':True,'best_score':best,
             'elapsed_seconds':time.perf_counter()-started,'not_a_deployment_benchmark':True}
    pf.dump(output/'summary.json',summary)
    return summary


SAME_RUN_KEYS = ('initial_sha256', 'final_sha256', 'batch_order_sha256', 'best_score', 'optimizer_steps')


def same_run(a, b):
    return all(a[k] == b[k] for k in SAME_RUN_KEYS)


def regression(cs, output, device=torch.device('cuda')):
    """Protocol section 6: seed-42 anchors (stop on mismatch), the R1 tie to the phase-1
    code path (stop on mismatch) and the 0 vs 4 workers check for A and C (fallback to 0)."""
    anchors = dbf.regression(cs)
    train, val = pf.datasets(cs['A'], True)
    mine = run_cell('A', cs['A'], train, val, output/'r1_new', device, 1, REGRESSION_SEED, LR, 0)
    phase1 = rs.run_recipe('R1', cs['A'], train, val, output/'r1_phase1', device, 1, REGRESSION_SEED,
                           *rs.RECIPES['R1'], 0)
    if not same_run(mine, phase1):
        raise RuntimeError('Regression: R1 path differs from recipe_stability.run_recipe')
    equivalent = {}
    for cell in WORKER_CELLS:
        train, val = pf.datasets(cs[cell], True)
        runs = [run_cell(cell, cs[cell], train, val, output/f'{cell}_workers{w}', device, 1,
                         REGRESSION_SEED, LR, w) for w in (0, WORKERS)]
        equivalent[cell] = same_run(*runs)
    return {**anchors, 'r1_matches_recipe_stability': True,
            'r1_final_sha256': mine['final_sha256'], 'workers_equivalent': equivalent,
            'num_workers': WORKERS if all(equivalent.values()) else 0}


def read_regression(path, current):
    record = json.loads((path/'REGRESSION_PASS.json').read_text())
    if (record.get('protocol') != PROTOCOL_ID or record.get('fingerprint') != current
            or record.get('r1_matches_recipe_stability') is not True or record.get('test_used') is not False):
        raise RuntimeError('Invalid or stale regression record')
    if record['num_workers'] != (WORKERS if all(record['workers_equivalent'].values()) else 0):
        raise RuntimeError('Regression worker choice inconsistent')
    return record['num_workers']


def verify_gate(path, current, seed, workers):
    pf.verify_gate(path, current)
    gate = json.loads((path/'COMPLETE.json').read_text())
    if gate.get('seed') != seed or gate.get('protocol') != PROTOCOL_ID:
        raise RuntimeError('Preflight gate is for a different seed/protocol')
    if gate.get('num_workers') != workers or gate.get('learning_rate') != LR:
        raise RuntimeError('Preflight gate used different workers/learning rate')
    if any(s.get('seed') != seed or s.get('learning_rate') != LR or s.get('num_workers') != workers
           for s in gate['summaries']):
        raise RuntimeError('Preflight cell seed/recipe mismatch')


def main():
    pin_source()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['test','regress','preflight','run'])
    p.add_argument('--seed',type=int)
    p.add_argument('--output',type=Path)
    p.add_argument('--regression',type=Path,help='Completed REGRESSION_PASS directory')
    p.add_argument('--preflight',type=Path)
    p.add_argument('--approve-training',action='store_true')
    args=p.parse_args()
    if args.stage=='test':
        import pytest
        raise SystemExit(pytest.main(['--import-mode=importlib','-q','tests/unit/test_recipe_confirmation.py',
            'tests/unit/test_direction_b.py','tests/unit/test_recipe_stability.py']))
    if args.output is None or args.output.exists():
        p.error('Provide a new --output directory')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    if args.stage=='regress':
        if args.seed != REGRESSION_SEED:
            p.error('regress needs --seed 42')
        strict_seed(REGRESSION_SEED)
        cs=pf.configs(); before=pf.fingerprint(cs)
        band_provenance(cs['A'])
        args.output.mkdir(parents=True,exist_ok=False)
        result={'protocol':PROTOCOL_ID,'stage':'regress','seed':REGRESSION_SEED,'test_used':False,
                'anchor_run':str(dbf.ANCHOR_RUN),**regression(cs,args.output)}
        if pf.fingerprint(cs)!=before:
            raise RuntimeError('Source/config/data metadata changed during regression')
        pf.dump(args.output/'REGRESSION_PASS.json',{**result,'fingerprint':before})
        print(f'RECIPE CONFIRMATION REGRESSION PASSED: {args.output} '
              f'(num_workers={result["num_workers"]}, equivalent={result["workers_equivalent"]})',flush=True)
        return
    if args.seed not in SEEDS:
        p.error('preflight/run are limited to seeds 201, 202 and 203')
    if args.regression is None:
        p.error('preflight/run need --regression')
    if args.stage=='run' and (not args.approve_training or args.preflight is None):
        p.error('Full training needs --approve-training and a completed --preflight')
    strict_seed(args.seed)
    cs=pf.configs(); before=pf.fingerprint(cs)
    workers=read_regression(args.regression,before)
    if args.stage=='run':
        verify_gate(args.preflight,before,args.seed,workers)
    args.output.mkdir(parents=True,exist_ok=False)
    pf.dump(args.output/'provenance.json',{**before,'seed':args.seed,'protocol':PROTOCOL_ID})
    epochs=1 if args.stage=='preflight' else EPOCHS
    summaries=[]
    for cell,c in cs.items():
        band_provenance(c)
        train,val=pf.datasets(c,args.stage=='preflight')
        directory=args.output/f'{cell}_proposed'
        env={'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(),
             'manifest_sha256':pf.DATA_HASHES['manifest'],'validation_patches':len(val),
             'train_patches':len(train),'test_used':False,'seed':args.seed,
             'learning_rate':LR,'num_workers':workers}
        summary=run_cell(cell,c,train,val,directory,torch.device('cuda'),epochs,args.seed,LR,workers)
        if args.stage=='run' and summary['optimizer_steps']!=FULL_STEPS:
            raise RuntimeError('Unexpected optimizer step count')
        pf.dump(directory/'environment.json',env)
        summaries.append(summary); pf.check_pairing(summaries)
        del train,val
        torch.cuda.empty_cache()
    if pf.fingerprint(cs)!=before:
        raise RuntimeError('Source/config/data metadata changed during run')
    pf.dump(args.output/'COMPLETE.json',{'stage':args.stage,'seed':args.seed,'protocol':PROTOCOL_ID,
                                       'learning_rate':LR,'num_workers':workers,
                                       'fingerprint':before,'summaries':summaries,'test_used':False})
    print(f'RECIPE CONFIRMATION FACTORIAL {args.stage.upper()} s{args.seed} COMPLETE: {args.output}',flush=True)


if __name__=='__main__':
    main()
