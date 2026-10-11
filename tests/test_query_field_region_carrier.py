import builtins
from types import SimpleNamespace
import numpy as np
from scipy import sparse
import torch
from radio_gs.query_field.region_carrier import RegionCarrier,query_text_region_scope
from scripts.research.compile_query_field_region_carrier import binary_state
from test_query_field_functional_training import small_field


def carrier():
    q=dict(roots=torch.tensor([0]),feature=torch.tensor([0]),left=torch.tensor([0]),right=torch.tensor([0]),
           cut=torch.tensor([-2.],dtype=torch.float64),value=torch.tensor([[.9,.9,.9]],dtype=torch.float64),
           depth=torch.tensor(0),dimensions=torch.tensor(16),addition_threshold=torch.tensor(.8,dtype=torch.float64))
    b=sparse.csr_matrix(np.array([[1,0],[1,1],[0,1],[0,0]],np.float32))
    v=sparse.csr_matrix(np.array([[1,1],[1,1],[0,1],[0,0]],np.float32))
    state=dict(frames=torch.tensor([0,1]),view_count=torch.tensor(2),descriptors=torch.ones(2,768),
               **binary_state(b,'membership'),**binary_state(v,'visibility'))
    state.update({'quality.'+k:v for k,v in q.items()})
    return RegionCarrier(state,4)


def test_visible_nonmember_counts_but_unobserved_abstains():
    c=carrier();d=c.decoder(torch.tensor([[0.,0,0],[1.,0,0],[2.,0,0],[3.,0,0]]))
    (scope,known),detail=d.scope(torch.tensor([.9,.6]))
    assert detail['applied'] and detail['seed']==0
    torch.testing.assert_close(scope,torch.tensor([1/1.9,1.,1.,0.]))
    assert known.tolist()==[True,True,True,False]
    assert d.scope(torch.tensor([.4,.3]))[0] is None


def test_optional_field_state_roundtrip_and_no_runtime_files(tmp_path,monkeypatch):
    from radio_gs.query_field.storage import save_field,load_field
    f=small_field(4);f.region_carrier=carrier();save_field(f,tmp_path/'new',{})
    loaded,_=load_field(tmp_path/'new')
    for k,v in f.state_dict().items():assert torch.equal(v,loaded.state_dict()[k])
    d=loaded.region_carrier.decoder(loaded.xyz)
    def denied(*a,**k):raise AssertionError('file read')
    monkeypatch.setattr(builtins,'open',denied)
    assert d.scope(torch.tensor([.9,.6]))[1]['applied']


def test_new_entry_retains_unknown_and_can_complete_weak_initial(monkeypatch):
    import radio_gs.query_field.scope_confidence as module
    f=small_field(4);f.region_carrier=carrier();f.semantic_measure=SimpleNamespace(supported=torch.ones(4,dtype=torch.bool))
    prior=torch.tensor([.8,.8,.2,.9]);prob=torch.tensor([.9,.6])
    monkeypatch.setattr(module,'native_text_evidence',lambda *a:(prior,prob))
    monkeypatch.setattr(module,'native_scope_confidence',lambda *a:(prob>.5,torch.ones(2)))
    class Fixed:
        field=f
        def decode(self,*a,**k):return prior.clone(),{}
    values,_=query_text_region_scope(f,None,None,exclusion_decoder=Fixed())
    assert values['conditional_addition'][2]>.5
    assert values['conditional_replace'][3]==prior[3]
    assert torch.equal(values['semantic_initial'],prior)


def test_seed_only_entry_never_executes_quality_or_pair_features(monkeypatch):
    from radio_gs.query_field.seed_scope import query_text_seed_scope
    import radio_gs.query_field.seed_scope as module
    f=small_field(4);f.region_carrier=carrier();f.semantic_measure=SimpleNamespace(supported=torch.ones(4,dtype=torch.bool))
    prior=torch.tensor([.8,.2,.1,.9]);prob=torch.tensor([.9,.6])
    monkeypatch.setattr(module,'native_text_evidence',lambda *a:(prior,prob))
    monkeypatch.setattr(module,'native_scope_confidence',lambda *a:(prob>.5,torch.ones(2)))
    def denied(*a,**k):raise AssertionError('unused supervised quality/appearance path')
    monkeypatch.setattr(f.region_carrier.quality,'forward',denied)
    monkeypatch.setattr(f.region_carrier,'decoder',denied)
    class Fixed:
        field=f
        def decode(self,*a,**k):return prior.clone(),{}
    result=query_text_seed_scope(f,None,None,exclusion_decoder=Fixed())
    assert result.selected.tolist()==[True,True,False,True]
    assert result.diagnostics['added']==1 and not result.diagnostics['conditional_quality_model_used']
    assert result.membership_score[3]==prior[3]
