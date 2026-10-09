import math
import pytest
import torch
from torch.nn import functional as F
from radio_gs.query_field.semantic_measure import SemanticMeasure
from radio_gs.query_field.storage import save_field,load_field
from radio_gs.query_field.runtime import QueryFieldRuntime
from radio_gs.querying.observation_reliability import reliable_observation_response
from test_query_preserving_field import example


def measure(count=12,dim=9):
    g=torch.Generator().manual_seed(314)
    return SemanticMeasure(dict(descriptors=torch.randn(7,dim,generator=g),
        pointer=torch.tensor([0,2,3]+[3]*(count-2)), index=torch.tensor([0,3,6]),
        mass=torch.tensor([.2,.8,1.]),supported=torch.tensor([True,True]+[False]*(count-2))),count)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_reference_reliability_and_menu_equivariance(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    m=measure().to(device);q=torch.randn(5,9,device=device,generator=torch.Generator(device=device).manual_seed(72))
    fallback=torch.full((12,5),.2,device=device)
    response=(100*(F.normalize(m.descriptors,dim=1)@F.normalize(q,dim=1).T)).softmax(1)
    base=m.operator()@response
    expected,_=reliable_observation_response(m.operator(),response,base,{},'entropy_information')
    actual=m.probabilities(q,fallback)
    torch.testing.assert_close(actual[:2],expected[:2],rtol=0,atol=1e-6)
    assert torch.equal(actual[2:],fallback[2:])
    order=torch.tensor([3,0,4,2,1],device=device)
    torch.testing.assert_close(m.probabilities(q[order],fallback[:,order]),actual[:,order],atol=1e-6,rtol=0)


def test_averaging_and_competition_do_not_commute():
    # Two rows have identical means but distinct distributions.
    m=SemanticMeasure(dict(descriptors=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]),
        pointer=torch.tensor([0,2,4]),index=torch.arange(4),mass=torch.full((4,),.5),
        supported=torch.ones(2,dtype=torch.bool)),2)
    q=torch.tensor([[1.,0.],[-1.,0.],[0.,1.]])
    fallback=torch.full((2,3),1/3)
    mean=m.probabilities(q,fallback,'mean');full=m.probabilities(q,fallback,'measure')
    assert torch.equal(mean[0],mean[1]) and not torch.allclose(full[0],full[1])


def test_zero_information_retains_base_and_checks_mass():
    m=measure();q=torch.ones(3,9);fallback=torch.full((12,3),1/3)
    torch.testing.assert_close(m.probabilities(q,fallback),fallback,atol=1e-6,rtol=0)
    s=dict(m.state_dict());s['mass']=s['mass']*2
    with pytest.raises(ValueError,match='normalized'):SemanticMeasure(s,12)
    with pytest.raises(ValueError,match='complete menu'):m.probabilities(q[:1],fallback[:,:1])


def test_export_runtime_and_other_queries_unchanged(tmp_path):
    f=example();q=torch.randn(4,9)
    from radio_gs.query_field.adapters import point_3d
    packet=point_3d(f,[2,0,0]);old=QueryFieldRuntime(f).query(packet).membership_scores.clone()
    f.semantic_measure=measure();f.semantic_measure_rule='reliable'
    result=QueryFieldRuntime(f).category_partition(q)
    save_field(f,tmp_path/'field',{});loaded=QueryFieldRuntime.load(tmp_path/'field')
    actual=loaded.category_partition(q)
    assert torch.equal(result[0],actual[0]) and torch.equal(result[1],actual[1])
    assert torch.equal(old,loaded.query(packet).membership_scores)
