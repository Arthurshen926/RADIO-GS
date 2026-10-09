import pytest
import torch
from torch.nn import functional as F
from radio_gs.query_field.semantic_measure import SemanticMeasure
from radio_gs.query_field.runtime import QueryFieldRuntime
from radio_gs.query_field.storage import save_field
from test_query_preserving_field import example


def measure(rows=12,dim=9):
    # Every disconnected site is semantically positive, regardless of top-k.
    descriptors=torch.zeros(rows,dim);descriptors[:,0]=torch.linspace(.6,1.,rows);descriptors[:,1]=.3
    return SemanticMeasure(dict(descriptors=descriptors,pointer=torch.arange(rows+1),index=torch.arange(rows),
        mass=torch.ones(rows),supported=torch.ones(rows,dtype=torch.bool)),rows)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_absolute_union_keeps_all_disconnected_modes_and_rejects_absent_query(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    m=measure().to(device);q=torch.eye(9,device=device)[0];neg=torch.eye(9,device=device)[1:3]
    p=m.union_probabilities(q,neg,torch.zeros(12,device=device))
    assert bool((p>.5).all()) and len(p)>8
    absent=m.union_probabilities(-q,neg,torch.zeros_like(p))
    assert not bool((absent>.5).any())
    assert torch.equal(p,m.union_probabilities(q,neg,torch.zeros_like(p)))


def test_nonlinear_union_matches_direct_region_response_and_not_mean():
    m=SemanticMeasure(dict(descriptors=torch.tensor([[1.,0.],[0.,1.]]),pointer=torch.tensor([0,2]),
        index=torch.arange(2),mass=torch.tensor([.7,.3]),supported=torch.ones(1,dtype=torch.bool)),1)
    q=torch.tensor([1.,0.]);neg=torch.tensor([[0.,1.]])
    expected=.7*torch.sigmoid(torch.tensor(10.))+.3*torch.sigmoid(torch.tensor(-10.))
    actual=m.union_probabilities(q,neg,torch.zeros(1))
    torch.testing.assert_close(actual,expected[None])
    assert (actual-m.union_probabilities(q,neg,torch.zeros(1),'mean')).abs().item()>.2
    with pytest.raises(ValueError):m.union_probabilities(q,neg[:0],torch.zeros(1))


def test_common_runtime_export_keeps_query_independence(tmp_path):
    f=example();f.semantic_measure=measure();f.semantic_measure_rule='measure'
    runtime=QueryFieldRuntime(f);q=torch.eye(9)[0];neg=torch.eye(9)[1:3]
    first=runtime.text_union(q,neg)
    runtime.text_union(-q,neg)
    again=runtime.text_union(q,neg)
    assert torch.equal(first.membership_scores,again.membership_scores)
    assert bool(first.selected.all()) and first.diagnostics['member_candidate_count']==3
    save_field(f,tmp_path/'field',{});loaded=QueryFieldRuntime.load(tmp_path/'field')
    assert torch.equal(first.membership_scores,loaded.text_union(q,neg).membership_scores)
