"""Read train masks; simulate loader order on metadata, not model training."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import ndimage as ndi
from bit_sar_v2_entry import pin_source
from smallflood_cd.data.datasets import read_manifest
from smallflood_cd.losses.local_component import build_local_supervision


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scan(records):
    details=[]; events=defaultdict(Counter)
    for i,r in enumerate(records,1):
        if r.split!='train' or r.uncertain_mask_path:
            raise ValueError('Only train without uncertainty masks is supported')
        arrays=[]
        for path in (r.label_path,r.valid_mask_path):
            if Path(path).suffix!='.npy': raise ValueError('Expected prepared NPY masks')
            a=np.load(path,allow_pickle=False)
            if a.shape==(256,256): a=a[None]
            if a.shape!=(1,256,256): raise ValueError(f'Unexpected shape {a.shape}')
            arrays.append(torch.from_numpy(a.copy())[None])
        target,valid=arrays
        s=build_local_supervision(target,valid)
        labels,n=ndi.label(((target>0)&(valid>0)).numpy()[0,0],np.ones((3,3),bool))
        areas=np.bincount(labels.ravel())[1:]
        small=int((areas<=35).sum())
        eligible_pixels=int((s.positive_weights>0).sum())
        row=dict(patch_id=r.patch_id,event_id=r.event_id,eligible=s.components,
                 empty_rings=s.empty_rings,small_clipped=small,eligible_pixels=eligible_pixels)
        details.append(row)
        events[r.event_id].update(dict(patches=1,patches_with_eligible=int(s.components>0),
            eligible=s.components,small_clipped=small,empty_rings=s.empty_rings,
            eligible_pixels=eligible_pixels,foreground_pixels=int(((target>0)&(valid>0)).sum())))
        if i%500==0: print(f'Checked {i}/{len(records)} TRAIN masks',flush=True)
    return details,{k:dict(v) for k,v in sorted(events.items())}


def batch_schedule(details, epochs=15):
    # Same generator consumption as factorial DataLoader: seeded once, not every epoch.
    loader=DataLoader(details,batch_size=8,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(42))
    rows=[]
    for epoch in range(epochs):
        h=hashlib.sha256(); numbers=[]
        for batch in loader:
            h.update(json.dumps(batch['patch_id'],separators=(',',':')).encode())
            numbers.append(int(batch['eligible'].sum()))
        if not numbers: raise ValueError('No batches')
        rows.append(dict(epoch=epoch,batches=len(numbers),empty_batches=numbers.count(0),
            empty_fraction=numbers.count(0)/len(numbers),eligible_total=sum(numbers),
            eligible_per_batch_min=min(numbers),eligible_per_batch_max=max(numbers),
            eligible_per_batch_mean=sum(numbers)/len(numbers),batch_order_sha256=h.hexdigest()))
    return rows


def main():
    pin_source();torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--output',type=Path,default=Path('artifacts/candidate_r_eligibility')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    a=p.parse_args()
    if a.self_test:
        import pytest
        raise SystemExit(pytest.main(['--import-mode=importlib','-q','tests/unit/test_candidate_r_eligibility.py']))
    manifest=Path('data/processed/urbansarfloods/patch_manifest.csv')
    stats=Path('data/metadata/component_stats_v1.json')
    hashes={'manifest':sha(manifest),'stats':sha(stats)}
    if hashes!={'manifest':'51eba373aef67e05b768bfc3c402d15c9fb9c8417c7026c6079a8fc0bea9b3ae',
                'stats':'525e4e1354e991983ebd2e3d3f68caad3f9894308a6eae46256a74405c9f2247'}:
        raise RuntimeError('Frozen metadata changed')
    records=sorted(read_manifest(manifest,'train'),key=lambda r:(r.event_id,r.patch_id))
    if len(records)!=19166 or len({r.patch_id for r in records})!=19166:
        raise RuntimeError('Train count/IDs differ')
    a.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    details,events=scan(records);schedule=batch_schedule(details)
    totals=Counter()
    for e in events.values(): totals.update(e)
    if hashes!={'manifest':sha(manifest),'stats':sha(stats)}:
        raise RuntimeError('Metadata changed during scan')
    old=Path('runs/proposed_factorial_v1/20260927T075837Z/A_proposed/summary.json')
    order_match=None
    if old.is_file():
        order_match=[e['batch_order_sha256'] for e in schedule]==json.loads(old.read_text())['batch_order_sha256']
        if not order_match: raise RuntimeError('Batch order differs from factorial A; investigate')
    report=dict(status='COMPLETE',test_used=False,validation_arrays_used=False,
        training_performed=False,cache_used=False,model_loaded=False,
        train_patches=len(records),counts=dict(totals),events=events,epochs=schedule,
        prior_A_batch_order_matches=order_match,metadata_sha256=hashes,script_sha256=sha(__file__),
        loss_source_sha256=sha(ROOT/'src/smallflood_cd/losses/local_component.py'),
        scope='all train masks; metadata-only 15-epoch batch schedule; no gradient or performance evidence',
        elapsed_seconds=time.perf_counter()-start)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (a.output/'patch_counts.json').write_text(json.dumps(details,allow_nan=False))
    print(json.dumps({'output':str(a.output),'counts':dict(totals),
        'empty_batch_fraction_min':min(e['empty_fraction'] for e in schedule),
        'empty_batch_fraction_max':max(e['empty_fraction'] for e in schedule),
        'prior_A_batch_order_matches':order_match},indent=2),flush=True)
    print('CANDIDATE R ELIGIBILITY AUDIT COMPLETE',flush=True)


if __name__=='__main__': main()
