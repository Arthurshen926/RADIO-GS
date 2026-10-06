import pytest
import torch
from torch.nn import functional as F
from radio_gs.training.native_semantic_mixture import fit_weighted_native_two_means, native_mixture_scores


def test_component_permutation_is_exact_same_query_function():
    torch.manual_seed(2)
    c=torch.randn(7,2,3);w=torch.rand(7,2);w/=w.sum(-1,keepdim=True)
    b=torch.eye(5)[:3];q=torch.randn(4,5)
    assert torch.allclose(native_mixture_scores(c,w,q,b,.1),native_mixture_scores(c.flip(1),w.flip(1),q,b,.1),atol=1e-6)


def test_identical_components_reduce_to_native_baseline_for_any_prior():
    torch.manual_seed(3)
    v=torch.randn(5,3);c=v[:,None].expand(-1,2,-1);w=torch.tensor([[1.,0.],[.1,.9],[.5,.5],[.01,.99],[0.,1.]])
    b=torch.eye(5)[:3];q=torch.randn(4,5)
    truth=F.normalize(v,dim=-1)@(F.normalize(q,dim=-1)@b.T).T
    assert torch.allclose(native_mixture_scores(c,w,q,b,.03),truth,atol=1e-6)


def test_native_query_norm_is_not_normalized_after_projection():
    c=torch.tensor([[[1.,0.],[1.,0.]]]);b=torch.eye(3)[:2]
    score=native_mixture_scores(c,torch.tensor([[.5,.5]]),torch.tensor([[1.,0.,2.]]),b,.1)
    assert torch.allclose(score,torch.tensor([[1/5**.5]]))


def test_source_masses_change_priors_and_rare_mode_query_response():
    x=torch.eye(2);i=torch.tensor([0,0]);r=torch.tensor([0,1])
    fit=fit_weighted_native_two_means(x,i,r,torch.tensor([9.,1.]),1)
    assert torch.allclose(fit['mixture_weights'].sort().values,torch.tensor([[.1,.9]]))
    assert torch.allclose((fit['centers']*fit['mixture_weights'][...,None]).sum(1),torch.tensor([[.9,.1]]))
    scores=native_mixture_scores(fit['centers'],fit['mixture_weights'],x,torch.eye(2),.1)
    assert scores[0,0]>scores[0,1]


def test_single_evidence_direction_collapses_without_nan():
    fit=fit_weighted_native_two_means(torch.tensor([[.6,.8]]),torch.tensor([0,0]),torch.tensor([0,0]),torch.tensor([1.,2.]),2)
    assert fit['two_nonempty_modes']==0 and torch.isfinite(fit['centers']).all()
    assert fit['temperature']>0 and fit['weighted_mean_reconstruction_max_error']<1e-6


def test_invalid_mass_fails_closed():
    with pytest.raises(ValueError,match='positive'):
        fit_weighted_native_two_means(torch.eye(2),torch.tensor([0]),torch.tensor([0]),torch.tensor([-1.]),1)
