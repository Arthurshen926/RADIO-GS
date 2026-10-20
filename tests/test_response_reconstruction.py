import pytest
import torch

from radio_gs.querying.response_reconstruction import AdmittedResponseSystem
from radio_gs.querying.typed_evidence import solve_rendered_evidence


def problem(matrix, target, initial, known=None, **kwargs):
    a=torch.tensor(matrix,dtype=torch.float64)
    y=torch.tensor(target,dtype=torch.float64).reshape(a.shape[0],-1)
    x=torch.tensor(initial,dtype=torch.float64).reshape(a.shape[1],-1)
    k=torch.ones_like(y) if known is None else torch.tensor(known,dtype=torch.float64).reshape_as(y)
    return AdmittedResponseSystem(x,[(a.to_sparse(),y,k,1.)],**kwargs)


def test_mixture_is_not_an_element_label():
    # One pure sighting and one mixed pixel identify (1,0). Assigning each
    # pixel's response directly to both elements blurs the boundary.
    p=problem([[1.,0.],[.5,.5]],[1.,.5],[.5,.5])
    diagonal=p.diagonal();rendered=p.rendered(residual_clip=None)
    assert diagonal[:,0].tolist()==pytest.approx([5/6,.5])
    torch.testing.assert_close(rendered,torch.tensor([[1.],[0.]],dtype=torch.float64),atol=1e-6,rtol=0)
    assert p.objective(rendered)<p.objective(diagonal)*1e-8


def test_identity_transport_agrees_with_precision_weighted_consensus():
    prior=torch.tensor([[.2],[.8]],dtype=torch.float64)
    strength=torch.tensor([[2.],[3.]],dtype=torch.float64)
    p=problem([[1.,0.],[0.,1.]],[1.,0.],[.5,.5],prior=prior,strength=strength)
    expected=torch.tensor([[1.4/3],[2.4/4]],dtype=torch.float64)
    torch.testing.assert_close(p.diagonal(),expected)
    torch.testing.assert_close(p.rendered(residual_clip=None),expected)


def test_unknown_and_unsupported_never_create_updates():
    p=problem([[1.,0.,0.],[0.,1.,0.]],[0.,1.],[.9,.2,.7],known=[0.,1.],
        prior=torch.zeros(3,1,dtype=torch.float64),strength=torch.ones(3,1,dtype=torch.float64))
    for value in (p.diagonal(),p.rendered()):
        assert value[0,0]==.9 and value[2,0]==.7
        assert value[1,0]>.2


def test_licensed_counterevidence_can_revise_high_native_prior():
    p=problem([[1.]],[0.],[.95])
    assert p.rendered()[0,0]<1e-8


def test_extracted_entry_matches_dense_huber_gradient():
    a=torch.tensor([[.8,.2],[.1,.9]],dtype=torch.float64)
    y=torch.tensor([[.9,.1],[.2,.8]],dtype=torch.float64)
    k=torch.tensor([[.3,0.],[1.,.8]],dtype=torch.float64)
    x=torch.tensor([[.1,.9],[.7,.3]],dtype=torch.float64)
    expected=x.clone();mass=a.T@k
    for _ in range(17):
        gradient=a.T@((y-a@expected).clamp(-.25,.25)*k)
        expected=(expected+.95*gradient/mass.clamp_min(1e-8)).clamp(0,1)
    actual,actual_mass=solve_rendered_evidence(x,[(a.to_sparse(),y,k,1.)],steps=17)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(actual_mass,mass)


def test_prior_strength_rejects_invalid_shape_or_negative_precision():
    with pytest.raises(ValueError,match='shape'):
        problem([[1.]],[.2],[.5],strength=torch.ones(2,1))
    with pytest.raises(ValueError,match='nonnegative'):
        problem([[1.]],[.2],[.5],strength=-torch.ones(1,1))


def test_category_capture_preserves_two_priors_and_default_output(monkeypatch):
    from radio_gs.querying.local_element_fusion import compile_local_evidence
    record=dict(queries=['a','b'],boundary_radius=0,instances=[
        dict(scores=torch.ones(1),foreground_margin=torch.ones(1),masks=torch.tensor([[[1.,0.]]])),
        dict(scores=torch.ones(1),foreground_margin=torch.ones(1),masks=torch.tensor([[[0.,1.]]]))])
    hits=dict(pixel_ids=torch.tensor([0,1]),gaussian_ids=torch.tensor([0,1]),base_weights=torch.ones(2))
    monkeypatch.setattr(torch,'load',lambda path,**kwargs:record if path=='identity' else hits)
    scene=dict(source_evidence=dict(raster_hw=[1,2],views=[dict(identity='identity',contributions='hits',frame=0)]))
    prior=torch.tensor([[.1,.9],[.1,.9],[.1,.9]]).logit()
    auxiliary=torch.tensor([[.9,.1],[.9,.1],[.9,.1]]).logit()
    observed=torch.ones(3,dtype=torch.bool)
    default=compile_local_evidence(prior,observed,scene,['a','b'],auxiliary)
    captured=compile_local_evidence(prior,observed,scene,['a','b'],auxiliary,capture_observations=True)
    assert 'reconstruction' not in default
    torch.testing.assert_close(default['average_logits'],captured['average_logits'])
    r=captured['reconstruction']
    p=AdmittedResponseSystem(r['initial'],r['observations'],prior=r['prior'],strength=r['strength'])
    torch.testing.assert_close(p.diagonal(),default['average_logits'].sigmoid())
    confidence=torch.tensor(1.).sigmoid()
    assert p.diagonal()[0,0].item()==pytest.approx(float((confidence+.05)/(confidence+.1)))
    torch.testing.assert_close(p.diagonal()[2],prior.sigmoid()[2])
