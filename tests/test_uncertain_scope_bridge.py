import torch
from radio_gs.querying.uncertain_scope_bridge import candidate_bridge_edges


def test_uncertain_connector_does_not_become_an_output_candidate():
    candidate=torch.tensor([True,False,True]);neighbors=torch.tensor([[1,2],[0,2],[0,1]])
    edges,bridge=candidate_bridge_edges(candidate,torch.tensor([.8,.3,.8]),neighbors,torch.ones_like(neighbors,dtype=torch.bool),enabled=True)
    assert bridge.tolist()==[False,True,False]
    assert edges[1].all() and candidate.tolist()==[True,False,True]


def test_default_and_low_score_do_not_add_connections():
    candidate=torch.tensor([True,False,True]);neighbors=torch.tensor([[1,2],[0,2],[0,1]]);mask=torch.ones_like(neighbors,dtype=torch.bool)
    before,_=candidate_bridge_edges(candidate,torch.tensor([.8,.3,.8]),neighbors,mask)
    after,bridge=candidate_bridge_edges(candidate,torch.tensor([.8,.1,.8]),neighbors,mask,enabled=True)
    assert torch.equal(before,after) and not bridge.any()
