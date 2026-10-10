import itertools
import pytest
import torch
from radio_gs.query_field.scope_permissions import ScopePermissions,compile_view,concatenate
from radio_gs.query_field.scope_exclusion import union_exclusion
from test_query_field_scope_exclusion import fixture


def test_every_subset_matches_raster_permissions_and_views_remain_separate():
    masks,scales=fixture();ids=torch.arange(len(masks));one=compile_view(masks,scales,ids,0)
    graph=ScopePermissions(concatenate([one,compile_view(masks,scales,ids,1)]))
    empty=torch.zeros_like(masks[0]);visible=torch.ones_like(empty)
    for bits in itertools.product([False,True],repeat=len(masks)):
        accepted=torch.tensor(bits);_,expected=union_exclusion(masks,scales,accepted,visible,empty)
        actual=graph(torch.cat([accepted,torch.zeros_like(accepted)]))
        assert torch.equal(actual[:len(masks)],expected) and not actual[len(masks):].any()


def test_graph_rejects_wrong_domains_and_cross_view_edges():
    masks,scales=fixture();state=compile_view(masks,scales,torch.arange(len(masks)),0)
    graph=ScopePermissions(state)
    with pytest.raises(ValueError):graph(torch.ones(len(masks)))
    state['candidate_views'][0]=1
    with pytest.raises(ValueError,match='cross-view'):ScopePermissions(state)


def test_field_roundtrip_embeds_graph_and_keeps_parent_state(tmp_path):
    from test_query_preserving_field import example
    from test_query_field_mixture_evidence import data
    from radio_gs.query_field.confidence_mixture_evidence import enable
    from radio_gs.query_field.storage import save_field,load_field
    field=example()
    enable(field,data([[.8,.2]]*len(field.xyz),[[.2,.8]]*len(field.xyz),[0,0]))
    masks,scales=fixture();state=compile_view(masks,scales,torch.tensor([0,1]),0)
    assert torch.equal(state['accepted_views'],field.evidence.views)
    old={k:v.clone() for k,v in field.state_dict().items()}
    field.scope_permissions=ScopePermissions(state)
    save_field(field,tmp_path/'field',{})
    restored,_=load_field(tmp_path/'field')
    for k,v in old.items():assert torch.equal(v,restored.state_dict()[k])
    accepted=torch.tensor([True,True]);assert torch.equal(field.scope_permissions(accepted),restored.scope_permissions(accepted))
