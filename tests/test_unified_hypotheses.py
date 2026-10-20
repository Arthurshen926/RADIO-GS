import torch
import pytest

from radio_gs.query_field.contracts import QueryPacket, QueryResult, Support
from radio_gs.query_field.unified_hypotheses import HypothesisRuntime


class Fixture:
    def __init__(self):
        self.xyz=torch.tensor([[0.,0,0],[.1,0,0],[1.,0,0],[1.1,0,0],[2.,0,0]])
        self.observed=torch.tensor([True,True,True,True,False]);self.threshold=.5
        self.default_scale=.5;self.world_units_per_metre=1.
        self.semantic=type("Semantic",(),dict(native_dim=2,score=lambda _,q:torch.tensor([.99,.01,.8,.8,-5.])[:,None]))()
        self.local=type("Local",(),dict(native_dim=2))()
    def support_membership(self,support,*args):
        return torch.tensor([.9,.9,.1,.1,0.]) if int(support.indices[0])<2 else torch.tensor([.1,.1,.9,.9,0.])
    def query(self,packet):
        return "exact existing explicit path"


def test_pooled_evidence_rejects_single_point_outlier_without_extra_state():
    f=Fixture();q=QueryPacket("text",semantic=torch.ones(2))
    peak=HypothesisRuntime(f,"peak",count=2).query(q)
    pooled=HypothesisRuntime(f,"pooled",count=2).query(q)
    assert torch.equal(peak.candidate_indices,pooled.candidate_indices)
    assert torch.equal(peak.selected,torch.tensor([True,True,False,False,False]))
    assert torch.equal(pooled.selected,torch.tensor([False,False,True,True,False]))
    assert pooled.diagnostics["persistent_proposal_bank"] is False


def test_union_keeps_separate_hypotheses_and_never_forces_anchor_membership():
    f=Fixture();q=QueryPacket("text",task="union",semantic=torch.ones(2))
    r=HypothesisRuntime(f,"peak",count=2,margin=1).query(q)
    assert r.selected[:4].all() and not r.selected[4]
    assert r.candidate_indices.tolist()==[0,2]
    assert not r.diagnostics["hard_positive_constraint"]
    f.support_membership=lambda *args:torch.zeros(5)
    r=HypothesisRuntime(f,"pooled",count=2).query(q)
    assert not r.selected.any()


def test_explicit_inputs_preserve_exact_existing_path():
    f=Fixture();q=QueryPacket("point_3d",positive=Support(torch.tensor([0]),torch.ones(1)),exact_positive=True)
    assert HypothesisRuntime(f,"pooled").query(q)==f.query(q)


def test_no_observed_support_is_not_silently_fabricated():
    f=Fixture();f.observed[:]=False
    with pytest.raises(ValueError,match="no observed"):
        HypothesisRuntime(f,"pooled").query(QueryPacket("text",semantic=torch.ones(2)))


def test_real_compact_cold_roundtrip_needs_no_teacher_io_or_extra_state(tmp_path,monkeypatch):
    import builtins
    import io
    from tests.test_unified_compact import field_fixture
    from radio_gs.query_field.shared_storage import save_field_v2,load_field_v2
    f=field_fixture();manifest=save_field_v2(f,tmp_path/"field",{},store_root=tmp_path/"blobs",reserve_bytes=1)
    cold,_=load_field_v2(tmp_path/"field",store_root=tmp_path/"blobs")
    before={k:v.clone() for k,v in cold.state_dict().items()}
    packet=QueryPacket("image",semantic=torch.randn(7),local=torch.randn(2,6),local_weights=torch.tensor([1.,0.]))
    expected=HypothesisRuntime(f,"pooled").query(packet)
    def denied(*args,**kwargs):raise AssertionError("query opened source files")
    with monkeypatch.context() as context:
        context.setattr(builtins,"open",denied);context.setattr(io,"open",denied);context.setattr(torch,"load",denied)
        actual=HypothesisRuntime(cold,"pooled").query(packet)
    assert torch.equal(expected.selected,actual.selected)
    assert torch.equal(expected.membership_scores,actual.membership_scores)
    assert set(before)==set(cold.state_dict())
    assert all(torch.equal(v,cold.state_dict()[k]) for k,v in before.items())
