import importlib.util
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
import pytest


@pytest.fixture
def audit(monkeypatch):
    scripts=Path(__file__).parents[2]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('eligibility_audit',scripts/'audit_candidate_r_eligibility.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_scan_reads_only_label_valid_and_preserves_inputs(audit,tmp_path):
    y=np.zeros((1,256,256),np.float32);y[:,10:12,10:12]=1;y[:,0,50]=1
    label=tmp_path/'label.npy';valid=tmp_path/'valid.npy'
    np.save(label,y);np.save(valid,np.ones_like(y))
    before=[audit.sha(p) for p in (label,valid)]
    r=SimpleNamespace(split='train',uncertain_mask_path=None,label_path=str(label),
        valid_mask_path=str(valid),patch_id='p',event_id='e',pre_path='/NEVER_READ',post_path='/NEVER_READ')
    details,events=audit.scan([r])
    assert details[0]['eligible']==1 and details[0]['small_clipped']==2
    assert details[0]['eligible_pixels']==4
    assert before==[audit.sha(p) for p in (label,valid)]
    r.split='test'
    with pytest.raises(ValueError):audit.scan([r])


def test_schedule_repeatable_and_counts(audit):
    d=[dict(patch_id=f'p{i}',eligible=int(i==0),event_id='e') for i in range(17)]
    a=audit.batch_schedule(d);b=audit.batch_schedule(d)
    assert a==b and len(a)==15
    assert all(e['batches']==3 and e['eligible_total']==1 and e['empty_batches']==2 for e in a)


def test_all_empty_batches(audit):
    d=[dict(patch_id=f'p{i}',eligible=0,event_id='e') for i in range(16)]
    assert audit.batch_schedule(d,1)[0]['empty_fraction']==1
