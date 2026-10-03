import importlib.util
from pathlib import Path
import pytest


@pytest.fixture
def review(monkeypatch):
    scripts=Path(__file__).parents[2]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('review_factorial_test',scripts/'review_proposed_factorial.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def rows():
    return [dict(epoch=i,global_step=(i+1)*2396,validation_selection_score=1 if i in [4,7] else .5)
            for i in range(15)]


def test_best_tie_earlier_and_last(review):
    e=review.expected_epochs(rows())
    assert e['best_composite.ckpt']['epoch']==4
    assert e['last.ckpt']['epoch']==14


def test_incomplete_rejected(review):
    with pytest.raises(RuntimeError): review.expected_epochs(rows()[:-1])


def test_wrong_checkpoint_rejected(review):
    row=rows()[0]; cfg={'model':{'name':'smallflood_cdnet'}}
    checkpoint={'epoch':0,'global_step':2396,'config':cfg}
    review.verify_checkpoint(checkpoint,row,cfg)
    checkpoint['epoch']=1
    with pytest.raises(RuntimeError): review.verify_checkpoint(checkpoint,row,cfg)
    checkpoint['epoch']=0
    with pytest.raises(RuntimeError): review.verify_checkpoint(checkpoint,row,{})


def test_count_mismatch_rejected(review):
    report={'checkpoint_epoch_1based':1,'global_pixel':dict(tp=1,fp=2,fn=3,tn=4)}
    expected={'epoch':0,**{f'validation_global_{k}':v for k,v in report['global_pixel'].items()}}
    review.verify_counts(report,expected)
    report['global_pixel']['tp']=2
    with pytest.raises(RuntimeError): review.verify_counts(report,expected)
