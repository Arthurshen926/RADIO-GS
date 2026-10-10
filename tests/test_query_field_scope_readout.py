from types import SimpleNamespace
import torch
from radio_gs.query_field.scope_readout import query_text_scope


def test_readout_computes_current_semantics_and_keeps_member_set_distinct():
    class Semantic:
        def score(self,q):return torch.ones(3,len(q) if q.ndim==2 else 1)*.2
    class Measure:
        supported=torch.ones(3,dtype=torch.bool)
        def union_probabilities(self,q,neg,prior,*args):
            assert torch.equal(prior,torch.full((3,),.5))
            return torch.tensor([.8,.6,.3])
    class Evidence:
        region_semantic=torch.tensor([[1.,0.]])
        def matrix(self,key):return torch.ones(3,1).to_sparse()
    class Constraints:
        def __call__(self,accepted,permissions,expanded=True):
            assert accepted.tolist()==[True]
            return torch.zeros(3,1),torch.tensor([[1.],[0.],[0.]]) if expanded else torch.zeros(3,1),{}
    field=SimpleNamespace(observed=torch.ones(3,dtype=torch.bool),semantic=Semantic(),semantic_measure=Measure(),
        evidence=Evidence(),scope_constraints=Constraints(),scope_permissions=object())
    result=query_text_scope(field,torch.tensor([1.,0.]),torch.tensor([[0.,1.]]))
    assert result.semantic_probability.tolist()==torch.tensor([.8,.6,.3]).tolist()
    assert result.selected.tolist()==[False,True,False]
    assert torch.equal(result.selected,result.membership_score>.5)
    assert not result.diagnostics['cached_query_response']
