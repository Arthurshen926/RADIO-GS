"""Independent check through the actual balanced source-mask loss."""
import copy
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.training.directional_scope_gradient import recovered_source_concept_probability
from scripts.audits.train_rendered_extent import balanced_mask_loss


def field(bias):
    torch.manual_seed(81)
    m=ElementRetrievalField(3,4,{'text':2,'image':2},shared_query_groups=[['text','image']],
        calibrated_readout=True,native_siglip_readout=True,native_semantic_dim=2,
        native_query_thresholds={'text':{'lower':-.5,'upper':.5}},extent_benefit_admission=True,
        grounded_transfer=dict(num_groups=2,num_proposals=4,residual_dim=3,
            identity_selection=dict(metric='support_iou',trusted_concept_start=2,
                require_native_support=False,completion_mode='native_bounded_extent',scope_logit_bound=4.)))
    with torch.no_grad():
        m.observed.copy_(torch.tensor([True,True,False]))
        m.grounded_transfer.prototypes.copy_(torch.eye(2));m.grounded_transfer.acceptance.zero_()
        m.extent_benefit_admission.coefficient[-1]=1
        m.grounded_transfer.decoder.weight.zero_()
        m.grounded_transfer.decoder.bias.copy_(torch.tensor([.3,.7,*bias]))
    m.requires_grad_(False)
    m.grounded_transfer.decoder.requires_grad_(True)
    m.grounded_transfer.residual.requires_grad_(True)
    return m


def loss(probability):
    target=torch.tensor([[1.,0.],[1.,0.],[0.,0.]])
    return balanced_mask_loss(probability,target,torch.tensor([True,True,False]))


def test_saturated_gradient_recovers_through_real_loss_with_frozen_native_ownership():
    baseline=field([-.2,1.2]);candidate=copy.deepcopy(baseline);q=torch.eye(2)
    old=baseline.logits(q,'text').sigmoid()*baseline.observed[:,None]
    new,_=recovered_source_concept_probability(candidate,q)
    assert torch.equal(old,new)
    loss(old).backward();loss(new).backward()
    assert torch.equal(baseline.grounded_transfer.decoder.bias.grad,torch.zeros(4))
    grad=candidate.grounded_transfer.decoder.bias.grad
    assert grad[2]<0 and grad[3]>0 and torch.isfinite(grad).all()
    for name,p in candidate.named_parameters():
        if not name.startswith(('grounded_transfer.decoder.','grounded_transfer.residual.')):
            assert not p.requires_grad and p.grad is None,name


def test_interior_real_loss_gradient_matches_original_head():
    baseline=field([.2,.8]);candidate=copy.deepcopy(baseline);q=torch.eye(2)
    old=baseline.logits(q,'text').sigmoid()*baseline.observed[:,None]
    new,_=recovered_source_concept_probability(candidate,q)
    assert torch.equal(old,new)
    loss(old).backward();loss(new).backward()
    for (name,p),(other_name,other) in zip(baseline.named_parameters(),candidate.named_parameters()):
        assert name==other_name
        if p.grad is not None:
            assert torch.allclose(p.grad,other.grad,atol=1e-6,rtol=1e-5),name
