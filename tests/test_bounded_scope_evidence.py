import pytest
import torch
from radio_gs.models.bounded_scope_evidence import bounded_scope_logits


def test_strong_native_evidence_survives_opposing_scope():
    native=torch.tensor([[5.,-5.],[.1,-.1]])
    mask=torch.tensor([[False,True],[False,True]])
    old=.5*(mask.float()+native.sigmoid())
    assert not bool(old[0,0]>.5) and bool(old[0,1]>.5)
    new=bounded_scope_logits(native,mask,1.)
    assert new[0,0]>0 and new[0,1]<0
    assert new[1,0]<0 and new[1,1]>0
    assert torch.equal(bounded_scope_logits(native,mask,0.),native)


def test_bound_rows_and_invalid_strength():
    torch.manual_seed(0);x=torch.randn(12,3);m=torch.rand(12,3)>.5
    y=bounded_scope_logits(x,m,2.)
    assert ((y-x).abs()<=2.000001).all()
    assert torch.equal(y[2:5,1:],bounded_scope_logits(x[2:5,1:],m[2:5,1:],2.))
    for bad in [-1.,float('nan'),float('inf')]:
        with pytest.raises(ValueError):bounded_scope_logits(x,m,bad)


@pytest.mark.parametrize("positive_only", [False, True])
def test_field_opt_in_preserves_state_modalities_and_global_row_contract(positive_only):
    from radio_gs.models.element_retrieval_field import ElementRetrievalField
    torch.manual_seed(17)
    args=dict(num_elements=12,latent_dim=10,query_dims={'text':16,'image':16,'prompt':8},
              shared_query_groups=[['text','image']],native_siglip_readout=True,native_semantic_dim=6,calibrated_readout=True,
              native_query_thresholds={'text':{'lower':-1.,'upper':1.}},
              grounded_transfer={'num_groups':1,'num_proposals':2,'identity_selection':{
                  'metric':'support_iou','trusted_concept_start':1,'completion_mode':'native_within_extent'}})
    old=ElementRetrievalField(**args);old.observed[:]=True
    with torch.no_grad():
        old.grounded_transfer.acceptance.fill_(2.)
        old.grounded_transfer.decoder.weight.zero_();old.grounded_transfer.decoder.bias.fill_(.8)
    config=dict(args['grounded_transfer']);config['identity_selection']=dict(config['identity_selection'],completion_mode='native_bounded_extent',scope_logit_bound=1.,preserve_native_positive=positive_only)
    new=ElementRetrievalField(**dict(args,grounded_transfer=config));new.load_state_dict(old.state_dict(),strict=True)
    q=torch.eye(16)[:2];actual=new.logits(q,'text')
    torch.testing.assert_close(actual[[0,5,8]],new.logits(q,'text',torch.tensor([0,5,8])))
    torch.testing.assert_close(actual[:,1:],new.logits(q[1:],'text'))
    assert torch.equal(old.logits(q,'image'),new.logits(q,'image'))
    prompt=torch.eye(8)[:2];assert torch.equal(old.logits(prompt,'prompt'),new.logits(prompt,'prompt'))
    with torch.no_grad():
        old.grounded_transfer.acceptance.fill_(-2);new.grounded_transfer.acceptance.fill_(-2)
    assert torch.equal(old.logits(q,'text'),new.logits(q,'text'))
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())


def test_positive_only_scope_cannot_erase_native_membership():
    x=torch.tensor([[-.2,.1,2.],[-2.,-.1,5.]])
    scope=torch.tensor([[True,False,True],[False,True,False]])
    y=bounded_scope_logits(x,scope,1.,suppress_outside=False)
    assert torch.equal(y[~scope],x[~scope])
    assert (y>=x).all() and ((y>0)[x>0]).all()
    assert y[0,0]>0 and y[1,1]>0
