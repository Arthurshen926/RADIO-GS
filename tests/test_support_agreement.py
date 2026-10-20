import torch
from radio_gs.querying.response_permissions import admit_response_observation
from radio_gs.querying.support_agreement import SupportAgreementRefiner


def evidence(indices, target, size, negative=False):
    op=torch.sparse_coo_tensor(torch.tensor([list(range(len(indices))),indices]),
        torch.ones(len(indices)),(len(indices),size)).coalesce()
    y=torch.tensor(target,dtype=torch.float32)[:,None]
    pos=torch.zeros_like(y,dtype=torch.bool) if negative else torch.ones_like(y,dtype=torch.bool)
    neg=~pos
    return admit_response_observation(op,y,pos,neg,torch.tensor(1.),exclusion=neg)[0]


def test_range_agreement_is_not_just_a_smaller_step():
    prior=torch.full((4,1),.2);guide=torch.tensor([[True],[True],[False],[False]])
    r=SupportAgreementRefiner(prior,guide)
    r.add(evidence([0,1],[1,1],4));r.add(evidence([2,3],[1,1],4))
    result=r.finish();scope,stats=result['scope'];matched,control=result['matched']
    assert torch.allclose(scope[:,0],torch.tensor([1.,1.,.2,.2]))
    assert torch.allclose(matched,torch.full_like(prior,.6))
    assert abs(stats['positive_step_sum'][0]-control['positive_step_sum'][0])<1e-6


def test_partial_observation_can_extend_beyond_frozen_guide():
    prior=torch.tensor([[.8],[.1]]);r=SupportAgreementRefiner(prior,torch.tensor([[True],[False]]))
    r.add(evidence([0,1],[1,1],2))
    value,_=r.finish()['scope']
    assert torch.allclose(value,torch.tensor([[.9],[.55]]))


def test_guide_does_not_create_negative_permission():
    prior=torch.tensor([[.95]])
    y=torch.zeros_like(prior)
    e,_=admit_response_observation(torch.ones(1,1).to_sparse(),y,y.bool(),~y.bool(),torch.tensor(1.))
    r=SupportAgreementRefiner(prior,torch.tensor([[False]]));r.add(e)
    assert torch.equal(r.finish()['scope'][0],prior)
    r.add(evidence([0],[0],1,negative=True))
    assert r.finish()['scope'][0].item()<.95


def test_query_columns_do_not_share_extent_or_control_budget():
    prior=torch.full((2,2),.2);guide=torch.tensor([[True,False],[False,True]])
    y=torch.tensor([[1.,.5],[.5,1.]])
    e,_=admit_response_observation(torch.eye(2).to_sparse(),y,torch.ones_like(guide),
        torch.zeros_like(guide),torch.ones(2))
    r=SupportAgreementRefiner(prior,guide);r.add(e)
    outputs=r.finish()
    assert torch.allclose(outputs['scope'][0],torch.tensor([[.6,.35],[.35,.6]]))
    assert torch.allclose(outputs['scope'][0],outputs['matched'][0])
