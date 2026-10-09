import torch
from radio_gs.query_field.contracts import Support,QueryPacket
from radio_gs.query_field.confidence_mixture_evidence import ConfidenceMixtureEvidence,enable
from radio_gs.query_field.signed_regions import signed_coefficients,query
from radio_gs.query_field.storage import save_field
from radio_gs.query_field.runtime import QueryFieldRuntime
from test_query_field_mixture_evidence import data
from test_query_preserving_field import example


def test_signed_source_region_selection_preserves_unknown_and_nested_exclusion():
    e=ConfidenceMixtureEvidence(data([[1,1],[1,0],[0,0]],[[0,0],[0,1],[0,0]],[0,0]),3)
    p=Support(torch.tensor([0]),torch.ones(1));n=Support(torch.tensor([1]),torch.ones(1));u=Support(torch.tensor([2]),torch.ones(1))
    c,positive,negative=signed_coefficients(e,p,n,1.)
    torch.testing.assert_close(c,torch.tensor([0.,.5]))
    torch.testing.assert_close(signed_coefficients(e,p,u,1.)[0],positive)
    torch.testing.assert_close(signed_coefficients(e,p,Support(torch.tensor([1,1]),torch.tensor([.4,.6])),1.)[0],c)
    # Independent dense source composition equals exported-coordinate contraction.
    actual=torch.sparse.mm(e.matrix('positive'),c[:,None]).flatten()
    torch.testing.assert_close(actual,torch.tensor([.5,0,0]))
    assert c.sum()<positive.sum()  # no resurrection of vetoed evidence confidence


def test_signed_runtime_export_and_unsigned_identity(tmp_path):
    f=example();enable(f,data([[1,1],[1,0]]+[[0,0]]*10,[[0,0],[0,1]]+[[0,0]]*10,[0,0]))
    p=Support(torch.tensor([0]),torch.ones(1));n=Support(torch.tensor([1]),torch.ones(1))
    packet=QueryPacket('point_2d',positive=p,negative=n,scale=1.)
    result=query(f,packet);save_field(f,tmp_path/'field',{})
    loaded=QueryFieldRuntime.load(tmp_path/'field').signed_regions(packet)
    assert torch.equal(result.membership_scores,loaded.membership_scores)
    one=QueryPacket('point_2d',positive=p,scale=1.)
    assert torch.equal(query(f,one).membership_scores,f.query(one).membership_scores)
    assert result.diagnostics['shared_member_core'] and not result.diagnostics['source_access']
