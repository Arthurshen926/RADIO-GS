import pytest
import torch
from radio_gs.models.extent_benefit_gate import scope_semantic_features
from radio_gs.models.element_retrieval_field import ElementRetrievalField


def test_rejected_concept_does_not_supply_fallback_semantics():
    query=torch.eye(3)[:2]
    result=scope_semantic_features(query,query,query*.25,torch.tensor([True,False]))
    torch.testing.assert_close(result,torch.tensor([[1.,.25],[0.,0.]]))


def test_semantic_gate_uses_matching_query_and_preserves_row_batch_cold_contracts():
    torch.manual_seed(19)
    kwargs=dict(num_elements=12,latent_dim=10,query_dims={'text':16,'image':16},
        shared_query_groups=[['text','image']],native_siglip_readout=True,native_semantic_dim=6,
        calibrated_readout=True,native_query_thresholds={'text':{'lower':-1.,'upper':1.}},
        grounded_transfer={'num_groups':1,'num_proposals':2,'routing':{'scope_gate':True},'identity_selection':{
            'metric':'support_iou','trusted_concept_start':1,'completion_mode':'native_within_extent'}})
    parent=ElementRetrievalField(**kwargs);parent.observed[:]=True
    with torch.no_grad():
        parent.grounded_transfer.acceptance.fill_(-2)
        parent.grounded_transfer.routing_scope_bias.fill_(10)
        parent.grounded_transfer.prototypes.zero_();parent.grounded_transfer.prototypes[:,0]=1
    field=ElementRetrievalField(**kwargs,extent_benefit_admission={'kind':'semantic_linear'})
    field.load_state_dict(parent.state_dict(),strict=False)
    # Admit a matching native text prototype, reject the orthogonal query.
    field.extent_benefit_admission.coefficient[8]=1
    field.extent_benefit_admission.coefficient[-1]=-.5
    q=torch.eye(16)[:2]
    native=field.readout_log_scale['text'].clamp(-4,6).exp()*(field.native_cosine(q)-field.native_query_thresholds['text'](q))
    result=field.logits(q,'text')
    torch.testing.assert_close(result[:,:1],parent.logits(q,'text')[:,:1])
    torch.testing.assert_close(result[:,1:],native[:,1:])
    rows=torch.tensor([0,5,8]);torch.testing.assert_close(field.logits(q,'text',rows),result[rows])
    torch.testing.assert_close(field.logits(q[:1],'text'),result[:,:1])
    torch.testing.assert_close(field.logits(q[1:],'text'),result[:,1:])
    assert torch.equal(field.logits(q,'image'),parent.logits(q,'image'))
    cold=ElementRetrievalField(**kwargs,extent_benefit_admission={'kind':'semantic_linear'});cold.load_state_dict(field.state_dict())
    assert torch.equal(cold.logits(q,'text'),result)
    kwargs['grounded_transfer']['routing']=None
    with pytest.raises(ValueError,match='scope prototypes'):
        ElementRetrievalField(**kwargs,extent_benefit_admission={'kind':'semantic_linear'})
