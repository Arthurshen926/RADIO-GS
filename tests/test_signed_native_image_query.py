import pytest
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.querying.signed_native_image_query import condition_signed_native_image


def test_signed_reference_conditioning_updates_only_temporary_query():
    torch.manual_seed(2)
    field=ElementRetrievalField(4,4,{'text':4,'image':4},shared_query_groups=[['text','image']],native_siglip_readout=True)
    with torch.no_grad():
        field.latent.copy_(torch.eye(4));field.observed[:]=True
        field.native_decoder.weight.copy_(torch.eye(4));field.native_decoder.bias.zero_()
    before={k:v.clone() for k,v in field.state_dict().items()}
    def render(scores):return scores[:,0],torch.tensor([True,True,True,False])
    result=condition_signed_native_image(field,torch.tensor([[0.,0.,1.,0.]]),render,
        torch.tensor([True,True,False,True]),torch.tensor([False,False,True,False]))
    assert result['report']['all_supported_signs_satisfied']
    assert result['report']['unsupported_positive_pixels']==1
    assert result['scores'][0]>.5 and result['scores'][1]>.5 and result['scores'][2]<.5
    assert all(torch.equal(v,field.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in field.parameters())


def test_conflicting_signed_input_rejected():
    f=ElementRetrievalField(2,2,{'text':2,'image':2},shared_query_groups=[['text','image']],native_siglip_readout=True)
    with pytest.raises(ValueError,match='disjoint'):
        condition_signed_native_image(f,torch.ones(1,2),None,torch.ones(2,dtype=torch.bool),torch.ones(2,dtype=torch.bool))
