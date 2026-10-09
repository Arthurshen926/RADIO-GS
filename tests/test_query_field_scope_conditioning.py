import pytest
import torch
from radio_gs.query_field.contracts import Support, QueryPacket
from radio_gs.query_field.confidence_mixture_evidence import ConfidenceMixtureEvidence, enable
from radio_gs.query_field.positive_rays import query, mixture_logits
from radio_gs.query_field.negative_evidence import competitive_membership
from radio_gs.query_field.storage import save_field, load_field
from test_query_field_mixture_evidence import data
from test_query_preserving_field import example


def evidence(count=6):
    s=data([[1,1],[1,0],[0,1]]+[[0,0]]*(count-3),
           [[0,0],[0,1],[1,0]]+[[0,0]]*(count-3),[0,0])
    s['scales']=torch.tensor([1.,10.])
    return s


def test_all_scope_preserves_view_budget_and_unknown_without_reweighting_scale():
    e=ConfidenceMixtureEvidence(evidence(),6);p=Support(torch.tensor([0]),torch.ones(1))
    torch.testing.assert_close(e.coefficients(p,1.),torch.tensor([1.,0.]))
    torch.testing.assert_close(e.coefficients(p,1.,scope_policy='all'),torch.tensor([.5,.5]))
    assert e.coefficients(p,1.,scope_policy='all').sum()==1
    logits=torch.tensor([0.,0.,0.,-100.,100.,-1.])
    actual=e.refine_scope(logits,p,1.,scope_policy='all')
    torch.testing.assert_close(actual[:3].sigmoid(),torch.tensor([.75,.5,.5]))
    assert torch.equal(actual[3:],logits[3:])
    assert torch.equal(e.refine_scope(logits,p,1.),e.refine(logits,p,1.))
    with pytest.raises(ValueError):e.coefficients(p,1.,scope_policy='oracle')


def test_scope_intervention_keeps_kernel_and_signed_semantics_after_export(tmp_path):
    f=example();enable(f,evidence(len(f.xyz)))
    p=Support(torch.tensor([0]),torch.ones(1),torch.tensor([0]))
    n=Support(torch.tensor([2]),torch.ones(1),torch.tensor([0]))
    packet=QueryPacket('point_2d',positive=p,negative=n,scale=1.)
    pos=mixture_logits(f,p,1.,aggregation='expected_union');neg=mixture_logits(f,n,1.,aggregation='expected_union')
    save_field(f,tmp_path/'field',{});cold,_=load_field(tmp_path/'field','cpu')
    for policy in ['local','positive_all','signed_all']:
        expected=competitive_membership(f.evidence.refine_scope(pos,p,1.,scope_policy='local' if policy=='local' else 'all'),
            f.evidence.refine_scope(neg,n,1.,scope_policy='all' if policy=='signed_all' else 'local'))
        actual=query(f,packet,aggregation='expected_union',negative_aggregation='expected_competition',region_scope=policy)
        restored=query(cold,packet,aggregation='expected_union',negative_aggregation='expected_competition',region_scope=policy)
        torch.testing.assert_close(actual.membership_scores,expected)
        assert torch.equal(actual.membership_scores,restored.membership_scores)
        assert actual.scale==1. and actual.diagnostics['local_kernel_scale_unchanged']
