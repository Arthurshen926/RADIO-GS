import pytest
import torch
from radio_gs.models.bounded_extent_readout import bounded_extent_logits
from radio_gs.models.element_retrieval_field import ElementRetrievalField


def test_bound_zero_and_abstention():
    native=torch.tensor([[-5.,1.],[5.,-1.]])
    extent=torch.tensor([[1.,0.],[0.,1.]])
    overlap=torch.tensor([.7,0.])
    assert bounded_extent_logits(native,extent,overlap,0) is native
    output=bounded_extent_logits(native,extent,overlap,2)
    assert (output-native).abs().max() <= 1.4+1e-6
    assert torch.equal(output[:,1],native[:,1])
    assert output[0,0] < output[1,0]  # strong native margin survives
    with pytest.raises(ValueError):bounded_extent_logits(native,extent,overlap,3)


@pytest.mark.parametrize('modality', ['text', 'image'])
def test_complete_field_query_and_element_batches_and_structure_gradient(modality):
    torch.manual_seed(5)
    f=ElementRetrievalField(12,6,{'text':8,'image':8},shared_query_groups=[['text','image']],calibrated_readout=True,native_siglip_readout=True,native_semantic_dim=3,grounded_transfer={'num_groups':2,'num_proposals':4},bounded_extent={modality:1.})
    f.observed.fill_(True)
    with torch.no_grad():
        f.readout_bias[modality].fill_(1.)
        f.grounded_transfer.decoder.bias.fill_(.55)
        f.grounded_transfer.decoder.weight.mul_(.1)
    q=torch.randn(2,8)
    value=f.logits(q,modality)
    torch.testing.assert_close(value,torch.cat([f.logits(x,modality) for x in q.split(1)],1))
    torch.testing.assert_close(value[[2,5]],f.logits(q,modality,torch.tensor([2,5])))
    value.sum().backward()
    assert f.latent.grad[:,3:].abs().sum()>0
    assert f.grounded_transfer.decoder.weight.grad.abs().sum()>0
