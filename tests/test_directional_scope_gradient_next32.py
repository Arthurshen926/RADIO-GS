import pytest
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.training.directional_scope_gradient import directional_probability_clamp, recovered_source_concept_probability


def test_forward_is_exact_existing_clamp():
    raw=torch.tensor([-2.,0.,1e-6,.2,.8,1-1e-6,1.,3.],requires_grad=True)
    assert torch.equal(directional_probability_clamp(raw),raw.clamp(1e-6,1-1e-6))


def test_wrong_saturated_outputs_receive_recovery_gradient():
    raw=torch.tensor([-.2,1.2],requires_grad=True)
    torch.nn.functional.binary_cross_entropy(directional_probability_clamp(raw),torch.tensor([1.,0.])).backward()
    assert raw.grad[0]<0 and raw.grad[1]>0 and torch.isfinite(raw.grad).all()


def test_outward_gradient_blocked_and_interior_gradient_unchanged():
    raw=torch.tensor([-.2,1.2,.2,.8],requires_grad=True)
    directional_probability_clamp(raw).backward(torch.tensor([1.,-1.,-3.,4.]))
    assert torch.equal(raw.grad,torch.tensor([0.,0.,-3.,4.]))


def model():
    torch.manual_seed(7)
    m=ElementRetrievalField(3,4,{'text':2,'image':2},shared_query_groups=[['text','image']],
       calibrated_readout=True,native_siglip_readout=True,native_semantic_dim=2,
       native_query_thresholds={'text':{'lower':-.5,'upper':.5}},extent_benefit_admission=True,
       grounded_transfer=dict(num_groups=2,num_proposals=4,residual_dim=3,
        identity_selection=dict(metric='support_iou',trusted_concept_start=2,
          require_native_support=False,completion_mode='native_bounded_extent',scope_logit_bound=4.)))
    with torch.no_grad():
        m.observed.copy_(torch.tensor([True,True,False]));m.grounded_transfer.prototypes.copy_(torch.eye(2))
        m.grounded_transfer.acceptance.zero_();m.extent_benefit_admission.coefficient[-1]=1
        m.grounded_transfer.decoder.weight.zero_();m.grounded_transfer.decoder.bias.copy_(torch.tensor([.3,.7,-.2,1.2]))
    m.requires_grad_(False);m.grounded_transfer.decoder.requires_grad_(True)
    m.grounded_transfer.residual.requires_grad_(True)
    return m


def test_real_model_exact_forward_and_only_active_concept_gradient_recovers():
    m=model();query=torch.eye(2);old=m.logits(query,'text').sigmoid()*m.observed[:,None]
    result,audit=recovered_source_concept_probability(m,query)
    assert torch.equal(result,old) and audit['replaced_concepts']==2
    torch.nn.functional.binary_cross_entropy(result[:2],torch.tensor([[1.,0.],[1.,0.]])).backward()
    assert m.grounded_transfer.decoder.bias.grad[2]<0 and m.grounded_transfer.decoder.bias.grad[3]>0
    assert torch.equal(m.grounded_transfer.decoder.bias.grad[:2],torch.zeros(2))


def test_rejected_admission_does_not_gain_scope_gradient():
    m=model()
    with torch.no_grad():m.extent_benefit_admission.coefficient[-1]=-1
    result,audit=recovered_source_concept_probability(m,torch.eye(2))
    assert audit['replaced_concepts']==0
    result.sum().backward()
    assert torch.equal(m.grounded_transfer.decoder.bias.grad,torch.zeros(4))


def test_hard_native_fallback_keeps_exact_output_and_zero_scope_gradient():
    m=model()
    with torch.no_grad():
        m.grounded_transfer.acceptance.fill_(2)
        m.native_decoder.weight.copy_(torch.eye(2));m.native_decoder.bias.zero_()
        m.latent[:,:2].copy_(torch.tensor([[1.,0.],[0.,1.],[1.,1.]]))
    query=torch.eye(2);old=m.logits(query,'text').sigmoid()*m.observed[:,None]
    result,audit=recovered_source_concept_probability(m,query)
    assert audit['replaced_concepts']==0 and audit['hard_or_rejected_queries']==2
    assert torch.equal(result,old)
    result.sum().backward()
    assert torch.equal(m.grounded_transfer.decoder.bias.grad,torch.zeros(4))


def test_nonfinite_cannot_hide_inside_forward_zero_identity():
    with pytest.raises(ValueError,match='Finite'):
        directional_probability_clamp(torch.tensor([float('inf')]))
