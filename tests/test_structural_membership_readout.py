import pytest
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField


def make_field(use_structure=True):
    torch.manual_seed(14)
    field=ElementRetrievalField(40,12,{'text':16,'image':16,'prompt':7},
        shared_query_groups=[['text','image']],native_siglip_readout=True,
        native_semantic_dim=8,structural_membership={'use_structure':use_structure})
    field.observed[:30]=True
    field.membership.bind_geometry(torch.randn(40,3),field.observed)
    return field


def test_rows_and_batches_use_same_complete_map_seeds():
    field=make_field();query=torch.randn(3,16);rows=torch.tensor([2,5,13])
    scores=field.logits(query,'text')
    torch.testing.assert_close(field.logits(query,'text',rows),scores[rows])
    torch.testing.assert_close(field.logits(query[:1],'text'),scores[:,:1])
    assert not field.retrieve(query,'text')['scores'][30:].any()
    for cosine in field.native_cosine(query).T:
        assert field.observed[field.membership.seed_indices(cosine,field.observed)].all()


def test_structure_changes_membership_without_rewriting_native_scores():
    field=make_field();query=torch.randn(2,16)
    native=field.native_cosine(query).detach().clone();before=field.logits(query,'image').detach()
    with torch.no_grad():field.latent[:,8:].normal_()
    assert torch.equal(native,field.native_cosine(query))
    assert not torch.equal(before,field.logits(query,'image'))


def test_semantic_control_ignores_structural_features():
    field=make_field(False);query=torch.randn(2,16);before=field.logits(query,'text').detach()
    with torch.no_grad():field.latent[:,8:].normal_()
    assert torch.equal(before,field.logits(query,'text'))


def test_head_optimizer_preserves_parent_state():
    field=make_field();field.requires_grad_(False);field.membership.requires_grad_(True)
    before={k:v.clone() for k,v in field.state_dict().items() if not k.startswith('membership.')}
    optimizer=torch.optim.Adam(field.membership.parameters())
    field.logits(torch.randn(1,16),'image').square().mean().backward();optimizer.step()
    assert all(torch.equal(v,field.state_dict()[k]) for k,v in before.items())
    assert not any('xyz' in k for k in field.state_dict())
    field.membership.xyz=None
    with pytest.raises(ValueError,match='bound frozen geometry'):field.logits(torch.randn(1,16),'text')
