import torch
from radio_gs.querying.volume_matched_region_readout import match_element_count


def test_count_matching_preserves_ranking_and_unknown_elements():
    proposed=torch.tensor([[.1],[.9],[.4],[.7],[.2]])
    parent=torch.tensor([[.6],[.7],[.8],[.1],[.9]])
    observed=torch.tensor([True,True,True,True,False])
    result,record=match_element_count(proposed,parent,observed)
    assert record[0]['parent_count']==3 and record[0]['new_count']==3
    assert torch.equal(torch.argsort(result[:4,0]),torch.argsort(proposed[:4,0]))
    assert result[4,0]==parent[4,0]


def test_ties_are_reported_and_empty_parent_abstains():
    p=torch.tensor([[.5,.2],[.5,.3],[.1,.4]])
    base=torch.tensor([[.8,.1],[.2,.2],[.1,.3]])
    out,record=match_element_count(p,base,torch.ones(3,dtype=torch.bool))
    assert record[0]['tie'] and record[0]['new_count']!=record[0]['parent_count']
    assert record[1]['abstained'] and torch.equal(out[:,1],base[:,1])
