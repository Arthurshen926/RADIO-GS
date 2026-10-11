from types import SimpleNamespace
import torch
from radio_gs.query_field.scope_confidence import confident_scope


def test_weak_admission_retains_original_view_confidence_not_full_certainty():
    class Evidence:
        views=torch.tensor([0]);view_count=1
        def matrix(self,key):return torch.ones(2,1).to_sparse_csr()
    class Constraints:
        def paired(self,accepted,permissions):
            p=torch.ones(2,1);n=torch.zeros(2,1)
            return {False:(p,n,{}),True:(p,n,{})}
    f=SimpleNamespace(observed=torch.ones(2,dtype=torch.bool),semantic_measure=SimpleNamespace(supported=torch.ones(2,dtype=torch.bool)),
        evidence=Evidence(),scope_constraints=Constraints(),scope_permissions=None)
    prior=torch.tensor([.3,.8]);result,a=confident_scope(f,prior,torch.tensor([.51]))
    assert a.item() and result[True][0][0]<.5
    torch.testing.assert_close(result[True][0],(prior+.02)/1.02)
    strong,_=confident_scope(f,prior,torch.tensor([1.]));torch.testing.assert_close(strong[True][0],(prior+1)/2)


def test_positive_witness_uses_visible_region_incidence_not_ray_coverage():
    from radio_gs.query_field.scope_confidence import native_positive_witness
    class Evidence:
        views=torch.tensor([0,0])
        def matrix(self,key):return torch.tensor([[0.,0.],[.3,0.],[0.,.4]]).to_sparse_csr()
    f=SimpleNamespace(evidence=Evidence())
    assert native_positive_witness(f,torch.tensor([True,False])).tolist()==[False,True,False]
    assert not native_positive_witness(f,torch.tensor([False,False])).any()
