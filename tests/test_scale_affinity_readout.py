import torch
import pytest
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.scale_affinity_readout import ScaleAffinityReadout


def make_field():
    torch.manual_seed(3)
    field=ElementRetrievalField(12,10,{'text':16,'image':16,'prompt':8},shared_query_groups=[['text','image']],native_siglip_readout=True,native_semantic_dim=6,scale_affinity={'scales':[.2,.5,1.], 'default_scale':1})
    field.observed[:]=True
    xyz=torch.stack([torch.arange(12).float(),torch.zeros(12),torch.zeros(12)],dim=1)
    normal={'minimum':torch.zeros(3),'extent':torch.tensor([11.,1.,1.])}
    field.affinity.bind_geometry(xyz,normal)
    q=torch.tensor([[4.,5.,0.,0.,0.,1.,0.,0.]])
    return field,q


def test_registered_prompt_reads_map_not_linear_adapter_or_dino():
    field,q=make_field();before=field.logits(q,'prompt').detach()
    with torch.no_grad():field.query_adapters['prompt'].weight.normal_(100,10)
    q[:,:2]*=100
    torch.testing.assert_close(before,field.logits(q,'prompt'))
    assert field.affinity.seed_rows(q).tolist()==[[0,11]]
    assert before[0]>0 and before[11]<0


def test_structural_changes_cannot_change_native_text_image_scores():
    field,q=make_field();native=torch.randn(2,16)
    before={m:field.logits(native,m).detach() for m in ('text','image')}
    with torch.no_grad():field.latent[:,6:].normal_()
    for m in before:assert torch.equal(before[m],field.logits(native,m))


def test_row_and_batch_invariance_and_unknown_seed_abstention():
    field,q=make_field();rows=torch.tensor([0,3,11]);scores=field.logits(q,'prompt')
    torch.testing.assert_close(field.logits(q,'prompt',rows),scores[rows])
    torch.testing.assert_close(field.logits(q.repeat(2,1),'prompt'),scores.repeat(1,2))
    field.observed[0]=False
    assert not (field.retrieve(q,'prompt')['scores']>.5).any()
    assert not any('xyz' in k for k in field.state_dict())
    field.affinity.xyz=None
    with pytest.raises(ValueError,match='bound'):field.logits(q,'prompt')


def test_scale_changes_affinity_features_and_has_gradients():
    layer=ScaleAffinityReadout(8,[.1,1.,10.]);z=torch.randn(6,8,requires_grad=True)
    small=layer.features(z,.1);large=layer.features(z,10.)
    assert not torch.allclose(small,large)
    layer.logits_at_scale(z,0,5,.5).square().mean().backward()
    assert z.grad.abs().sum()>0 and layer.cutoffs.grad.abs().sum()>0


def test_full_vector_control_is_ungated_and_preserves_native_path():
    field,q=make_field();native=torch.randn(2,16)
    before=field.logits(native,'text').detach().clone()
    field.affinity.full_prompt_features=True
    expected=field.affinity.logits_at_scale(field.latent,0,11,.1)
    torch.testing.assert_close(field.logits(q,'prompt')[:,0],expected)
    with torch.no_grad():field.affinity.cutoffs.add_(100)
    field.affinity.default_scale.fill_(0)
    torch.testing.assert_close(field.logits(q,'prompt')[:,0],expected)
    assert torch.equal(before,field.logits(native,'text'))


def test_native_seed_grouping_uses_full_map_matching_and_preserves_cosine():
    field,_=make_field();q=torch.randn(2,16)
    native=field.native_cosine(q).detach().clone()
    field.affinity.native_seed_queries=True
    scores=field.logits(q,'text');rows=torch.tensor([2,4,7])
    torch.testing.assert_close(field.logits(q,'text',rows),scores[rows])
    torch.testing.assert_close(field.logits(q[:1],'text'),scores[:,:1])
    assert torch.equal(native,field.native_cosine(q))
    field.observed[0]=False
    assert not field.retrieve(q,'image')['scores'][0].any()
