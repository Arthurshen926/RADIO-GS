import torch
from radio_gs.models.extent_benefit_gate import extent_agreement_features
from radio_gs.models.element_retrieval_field import ElementRetrievalField


def test_scope_features_distinguish_parent_from_part_and_ignore_unobserved():
    c=torch.tensor([[.8,.8],[.8,.8],[-.3,-.3],[100.,100.]])
    extent=torch.tensor([[1.,1.],[1.,-1.],[1.,-1.],[-100.,100.]])
    x=extent_agreement_features(c,0.,extent,torch.tensor([1,1,1,0],dtype=torch.bool),torch.ones(2,dtype=torch.bool))
    torch.testing.assert_close(x[:,1],torch.tensor([2/3,1.]))
    torch.testing.assert_close(x[:,2],torch.tensor([1.,.5]))
    assert (x[:,6]==.8).all()


def test_gate_accept_reject_preserves_native_contract_rows_and_cold_state():
    torch.manual_seed(17)
    kwargs=dict(num_elements=12,latent_dim=10,query_dims={'text':16,'image':16},
        shared_query_groups=[['text','image']],native_siglip_readout=True,native_semantic_dim=6,
        calibrated_readout=True,native_query_thresholds={'text':{'lower':-1.,'upper':1.}},
        grounded_transfer={'num_groups':1,'num_proposals':2,'identity_selection':{
            'metric':'support_iou','trusted_concept_start':1,'completion_mode':'native_within_extent'}})
    parent=ElementRetrievalField(**kwargs);parent.observed[:]=True
    with torch.no_grad():parent.grounded_transfer.acceptance.fill_(-2)
    field=ElementRetrievalField(**kwargs,extent_benefit_admission=True)
    field.load_state_dict(parent.state_dict(),strict=False)
    q=torch.eye(16)[:2]
    field.extent_benefit_admission.coefficient[-1]=1
    torch.testing.assert_close(field.logits(q,'text'),parent.logits(q,'text'),rtol=0,atol=0)
    field.extent_benefit_admission.coefficient[-1]=0
    expected=field.readout_log_scale['text'].clamp(-4,6).exp()*(field.native_cosine(q)-field.native_query_thresholds['text'](q))
    torch.testing.assert_close(field.logits(q,'text'),expected)
    rows=torch.tensor([0,5,8]);torch.testing.assert_close(field.logits(q,'text',rows),expected[rows])
    torch.testing.assert_close(field.logits(q[:1],'text'),expected[:,:1])
    assert torch.equal(field.logits(q,'image'),parent.logits(q,'image'))
    cold=ElementRetrievalField(**kwargs,extent_benefit_admission=True);cold.load_state_dict(field.state_dict())
    assert torch.equal(cold.logits(q,'text'),field.logits(q,'text'))
