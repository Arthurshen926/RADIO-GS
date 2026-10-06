import pytest
import torch
from radio_gs.querying.query_relative_scope import add_query_difference


def test_same_scope_can_distinguish_queries_without_changing_prototype():
    scope=torch.tensor([[2.,2.,2.],[-2.,-2.,-2.]])
    prototype=torch.tensor([[1.,1.,1.],[-1.,-1.,-1.]])
    query=torch.tensor([[1.,-3.,9.],[-1.,3.,9.]])
    actual=add_query_difference(scope,query,prototype,torch.tensor([True,True,False]))
    assert torch.equal(actual[:,0],scope[:,0])
    assert torch.equal(actual[:,2],scope[:,2])
    assert not torch.equal(actual[:,0]>0,actual[:,1]>0)


def test_rows_and_query_subsets_do_not_change_differences():
    torch.manual_seed(12);x=torch.randn(9,4);q=torch.randn(9,4);p=torch.randn(9,4);allow=torch.tensor([True,False,True,True]);full=add_query_difference(x,q,p,allow)
    assert torch.equal(full[:3,1:],add_query_difference(x[:3,1:],q[:3,1:],p[:3,1:],allow[1:]))
    with pytest.raises(ValueError):add_query_difference(x,q,p,torch.ones(4))


def test_real_field_intervention_keeps_state_image_and_rejected_routes():
    from radio_gs.models.element_retrieval_field import ElementRetrievalField
    from radio_gs.querying.query_relative_scope import install_query_relative_scope_control
    torch.manual_seed(17)
    field=ElementRetrievalField(num_elements=12,latent_dim=10,query_dims={'text':16,'image':16},shared_query_groups=[['text','image']],native_siglip_readout=True,native_semantic_dim=6,calibrated_readout=True,native_query_thresholds={'text':{'lower':-1.,'upper':1.}},grounded_transfer={'num_groups':1},extent_benefit_admission=True)
    field.observed[:]=True
    with torch.no_grad():
        field.grounded_transfer.prototypes[0,0]=1
        field.grounded_transfer.acceptance.fill_(-2)
        field.extent_benefit_admission.coefficient[-1]=1
    query=torch.eye(16)[:2];before=field.logits(query,'text');image=field.logits(query,'image');state={k:v.clone() for k,v in field.state_dict().items()}
    undo=install_query_relative_scope_control(field)
    actual=field.logits(query,'text')
    assert torch.equal(actual[:,0],before[:,0])
    assert torch.equal(image,field.logits(query,'image'))
    torch.testing.assert_close(actual[[0,5,8]],field.logits(query,'text',torch.tensor([0,5,8])))
    torch.testing.assert_close(actual[:,1:],field.logits(query[1:],'text'))
    assert all(torch.equal(state[k],v) for k,v in field.state_dict().items())
    undo();assert torch.equal(before,field.logits(query,'text'))
    with torch.no_grad():field.extent_benefit_admission.coefficient[-1]=-1
    native=field.logits(query,'text');undo=install_query_relative_scope_control(field)
    assert torch.equal(native,field.logits(query,'text'));undo()
