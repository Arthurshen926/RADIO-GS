import pytest
import torch
from radio_gs.training.scope_vote_abstention import scope_vote_abstention


def test_part_negative_is_unknown_only_where_parent_has_strong_positive():
    p=torch.tensor([[.9,.1,.9],[.1,.1,.9]]);m=torch.ones_like(p)
    result=scope_vote_abstention(p,m,[1,2],[dict(source=1,target=2,relation='source_contains_target')])
    torch.testing.assert_close(result['containment_abstention']['probability'],torch.tensor([.9,.1,.9]))
    torch.testing.assert_close(result['containment_abstention']['mass'],torch.tensor([1.,2.,2.]))
    assert result['incompatible_abstention']['removed_votes']==0


def test_incompatible_conflicts_abstain_symmetrically_and_preserve_other_views():
    p=torch.tensor([[.9,.9],[.1,.1],[.8,.4]]);m=torch.ones_like(p);m[2,1]=0
    result=scope_vote_abstention(p,m,[1,2,3],[dict(source=1,target=2,relation='incompatible')])['incompatible_abstention']
    torch.testing.assert_close(result['probability'],torch.tensor([.8,0.]))
    torch.testing.assert_close(result['mass'],torch.tensor([1.,0.]))


def test_unknown_weak_votes_and_validation_endpoints_are_not_silently_relabeled():
    p=torch.tensor([[.9],[.1]]);m=torch.tensor([[1.],[.01]])
    for relation in ['unknown','support','source_contains_target','incompatible']:
        result=scope_vote_abstention(p,m,[1,2],[dict(source=1,target=2,relation=relation)])
        assert result['combined_abstention']['removed_votes']==0
    with pytest.raises(ValueError,match='outside'):
        scope_vote_abstention(p,m,[1,2],[dict(source=1,target=99,relation='support')])
