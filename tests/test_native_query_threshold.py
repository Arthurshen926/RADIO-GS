import pytest
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_query_threshold import NativeQueryThreshold, fit_threshold_ridge


def make_field():
    torch.manual_seed(7)
    field = ElementRetrievalField(11, 6, {'text':8,'image':8,'prompt':4},
        shared_query_groups=[['text','image']], calibrated_readout=True,
        native_siglip_readout=True, native_semantic_dim=4,
        native_query_thresholds={'text':{'lower':-.4,'upper':.4}})
    field.observed[:] = True
    return field


def test_query_threshold_preserves_native_ranks_and_other_modalities():
    field=make_field();q=torch.randn(3,8);prompt=torch.randn(2,4)
    cosine=field.native_cosine(q).detach().clone()
    old_image=field.logits(q,'image').detach().clone()
    old_prompt=field.logits(prompt,'prompt').detach().clone()
    logits=field.logits(q,'text');scale=field.readout_log_scale['text'].exp()
    torch.testing.assert_close(logits[1:]-logits[:1],scale*(cosine[1:]-cosine[:1]))
    with torch.no_grad():field.native_query_thresholds['text'].linear.weight.add_(10)
    assert torch.equal(cosine,field.native_cosine(q))
    assert torch.equal(old_image,field.logits(q,'image'))
    assert torch.equal(old_prompt,field.logits(prompt,'prompt'))


def test_query_threshold_is_bounded_batch_and_row_independent_and_cold_loads():
    field=make_field();q=torch.randn(3,8);rows=torch.tensor([1,5,7])
    torch.testing.assert_close(field.logits(q,'text',rows),field.logits(q,'text')[rows])
    torch.testing.assert_close(field.logits(q[:1],'text'),field.logits(q,'text')[:,:1])
    threshold=field.native_query_thresholds['text'](q)
    assert ((threshold>=-.4)&(threshold<=.4)).all()
    other=make_field();other.load_state_dict(field.state_dict(),strict=True)
    assert torch.equal(field.logits(q,'text'),other.logits(q,'text'))


def test_ridge_generalizes_linear_boundary_and_constant_targets():
    torch.manual_seed(5);q=torch.nn.functional.normalize(torch.randn(40,5),dim=-1)
    w=torch.randn(5)*.1;t=q@w+.2
    fit,b=fit_threshold_ridge(q[:30],t[:30],.001)
    torch.testing.assert_close(q[30:]@fit+b,t[30:],atol=1e-4,rtol=1e-3)
    fit,b=fit_threshold_ridge(q,torch.full((40,),.25),.1)
    torch.testing.assert_close(fit,torch.zeros(5));assert b==.25


def test_bad_bounds_and_conflicting_readouts_fail():
    with pytest.raises(ValueError):NativeQueryThreshold(5,.7,.1)
    with pytest.raises(ValueError):fit_threshold_ridge(torch.randn(1,5),torch.zeros(1),.1)
    with pytest.raises(ValueError,match='plain native'):
        ElementRetrievalField(10,6,{'text':8,'image':8},shared_query_groups=[['text','image']],
            native_siglip_readout=True,standardize_native_text=True,
            native_query_thresholds={'text':{'lower':0.,'upper':.5}})
