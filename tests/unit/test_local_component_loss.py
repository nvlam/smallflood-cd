import math
import numpy as np
import pytest
import torch
from scipy import ndimage as ndi

from smallflood_cd.losses.local_component import (
    build_local_supervision, local_loss, LocalComponentLoss, CandidateRLoss, LocalSupervision,
)
from smallflood_cd.losses import SmallFloodLoss
from smallflood_cd.models.smallflood_cdnet import ChangeDetectionOutput
from smallflood_cd.models.registry import build_model
from smallflood_cd.metrics.object_boundary_v2 import patch_counts


def masks(batch=1):
    y=torch.zeros(batch,1,32,32); y[:,:,8:10,8:10]=1
    return y,torch.ones_like(y)


def test_zero_logits_formula_and_gradient_signs():
    y,v=masks(); z=torch.zeros_like(y,requires_grad=True)
    loss,s=LocalComponentLoss()(z,y,v)
    assert s.components==1 and s.empty_rings==0
    assert loss.item()==pytest.approx(math.log(2))
    assert s.positive_weights.sum()==pytest.approx(.5)
    assert s.negative_weights.sum()==pytest.approx(.5)
    loss.backward()
    assert (z.grad[y.bool()]<0).all()
    assert (z.grad[s.negative_weights>0]>0).all()
    assert (z.grad[(s.negative_weights+s.positive_weights)==0]==0).all()


def test_equal_component_mass_not_pixel_mass():
    y,v=masks(); y[:,:,20:23,20:23]=1
    s=build_local_supervision(y,v)
    assert s.components==2
    assert s.positive_weights[:,:,8:10,8:10].sum()==pytest.approx(.25)
    assert s.positive_weights[:,:,20:23,20:23].sum()==pytest.approx(.25)


def test_batch_component_mean_not_image_mean():
    y,v=masks(2); y[1,:,20:23,20:23]=1
    s=build_local_supervision(y,v)
    assert s.components_per_image==(1,2)
    assert s.positive_weights[0].sum()==pytest.approx(1/6)
    assert s.positive_weights[1].sum()==pytest.approx(1/3)


def test_eligibility_matches_evaluation_and_size_cutoff():
    y,v=masks(); y[:,:,0,20]=1; y[:,:,20:26,20:26]=1
    y[:,:,15,8]=1; v[:,:,15,9]=0
    s=build_local_supervision(y,v)
    counts=patch_counts(np.zeros((32,32),bool),y[0,0].numpy(),v[0,0].numpy(),35)
    assert s.components==counts['small_target_interior']==1
    y=torch.zeros_like(y);v=torch.ones_like(v);y[:,:,10:15,10:17]=1
    assert build_local_supervision(y,v).components==1  # exactly 35


def test_invalid_and_other_foreground_excluded_from_ring():
    y,v=masks(); y[:,:,11:17,11:17]=1; v[:,:,6,6]=0
    s=build_local_supervision(y,v)
    assert s.components==1
    assert (s.negative_weights[y.bool()]==0).all()
    assert (s.negative_weights[~v.bool()]==0).all()
    assert (s.positive_weights[~v.bool()]==0).all()


def test_overlapping_rings_equal_explicit_component_average():
    y,v=masks(); y[:,:,8:10,12:14]=1
    z=torch.linspace(-2,2,y.numel(),dtype=torch.float64).reshape(y.shape).requires_grad_()
    actual,s=LocalComponentLoss()(z,y,v)
    fg=y[0,0].numpy().astype(bool); labels,n=ndi.label(fg,np.ones((3,3)))
    terms=[]
    for i in range(1,n+1):
        c=labels==i; ring=ndi.binary_dilation(c,np.ones((3,3)),iterations=3)&~fg
        terms.append(.5*torch.nn.functional.softplus(-z[0,0][c]).mean()
                     +.5*torch.nn.functional.softplus(z[0,0][ring]).mean())
    expected=torch.stack(terms).mean()
    assert torch.allclose(actual,expected,atol=1e-12,rtol=1e-12)
    ga=torch.autograd.grad(actual,z,retain_graph=True)[0]
    ge=torch.autograd.grad(expected,z)[0]
    assert torch.allclose(ga,ge,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('case',['background','invalid','large'])
def test_no_eligible_differentiable_zero(case):
    y,v=masks()
    if case=='background':y.zero_()
    if case=='invalid':v.zero_()
    if case=='large':y[:,:,5:20,5:20]=1
    z=torch.randn_like(y,requires_grad=True)
    loss,s=LocalComponentLoss()(z,y,v)
    assert s.components==0 and loss.item()==0
    loss.backward(); assert torch.count_nonzero(z.grad)==0


def test_nonbinary_invalid_shapes_and_nonfinite_rejected():
    y,v=masks()
    with pytest.raises(ValueError): build_local_supervision(y.squeeze(),v)
    y[:,:,0,0]=float('nan')
    with pytest.raises(ValueError): build_local_supervision(y,v)
    y,v=masks()
    with pytest.raises(ValueError): LocalComponentLoss()(torch.full_like(y,float('inf')),y,v)


def test_large_logits_finite_and_no_input_mutation():
    y,v=masks();before=y.clone(),v.clone()
    z=torch.full_like(y,1000,requires_grad=True)
    loss,_=LocalComponentLoss()(z,y,v)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    assert torch.equal(y,before[0]) and torch.equal(v,before[1])


def test_empty_ring_reduction_and_deterministic_maps():
    y,v=masks();s=build_local_supervision(y,v);again=build_local_supervision(y,v)
    assert torch.equal(s.positive_weights,again.positive_weights)
    assert torch.equal(s.negative_weights,again.negative_weights)
    # Artificial prepared empty-ring case tests specified fallback without division by zero.
    empty=LocalSupervision(s.positive_weights,torch.zeros_like(s.negative_weights),(1,),1)
    z=torch.zeros_like(y,requires_grad=True)
    assert local_loss(z,empty).item()==pytest.approx(.5*math.log(2))


def test_random_eligibility_matches_metric():
    rng=np.random.default_rng(42)
    for _ in range(8):
        y=torch.from_numpy((rng.random((2,1,17,19))<.12).astype(np.float32))
        v=torch.from_numpy((rng.random(y.shape)>.08).astype(np.float32))
        s=build_local_supervision(y,v)
        expected=tuple(patch_counts(np.zeros((17,19),bool),y[b,0].numpy(),v[b,0].numpy(),35)
                       ['small_target_interior'] for b in range(2))
        assert s.components_per_image==expected


def test_composite_preserves_A_terms_and_rejects_boundary():
    y,v=masks();z=torch.randn_like(y,requires_grad=True)
    out=ChangeDetectionOutput(z,None,z)
    new=CandidateRLoss()(out,y,v)
    old=SmallFloodLoss(.4,.4,0,.3,.7)(out,y,torch.ones_like(y),v,y,v)
    assert torch.equal(new.bce,old.weighted_bce)
    assert torch.equal(new.tversky,old.weighted_tversky)
    assert torch.equal(new.total,old.total+.1*new.local)
    with pytest.raises(ValueError): CandidateRLoss()(ChangeDetectionOutput(z,z,z),y,v)


def test_real_A_model_backward_without_optimizer():
    torch.manual_seed(42)
    model=build_model({'name':'smallflood_cdnet','boundary_head':False,'pretrained':False}).eval()
    y,v=masks(2)
    out=model(torch.randn(2,2,32,32),torch.randn(2,2,32,32))
    result=CandidateRLoss()(out,y,v)
    result.total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_matches_cpu_loss_and_gradient():
    y,v=masks();z=torch.randn_like(y,requires_grad=True)
    cpu,_=LocalComponentLoss()(z,y,v);cpu.backward()
    zg=z.detach().cuda().requires_grad_()
    gpu,_=LocalComponentLoss()(zg,y.cuda(),v.cuda());gpu.backward()
    assert torch.allclose(cpu,gpu.cpu(),atol=1e-6)
    assert torch.allclose(z.grad,zg.grad.cpu(),atol=1e-6)
