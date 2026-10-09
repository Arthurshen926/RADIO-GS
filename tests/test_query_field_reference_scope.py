import numpy as np
import pytest
from radio_gs.query_field.reference_scope import shape_union


def test_shape_union_composes_disjoint_parts_and_preserves_fractional_reference():
    masks=np.array([[[1,1,0,0,0]],[[0,0,1,1,0]],[[0,0,0,0,1]]],bool)
    reference=np.array([[1.,1.,1.,.5,0.]])
    ids,union,trace=shape_union(masks,reference)
    assert ids==[0,1] and trace[-1]==3.5/4
    assert np.array_equal(union,[[1,1,1,1,0]])


def test_shape_union_ties_duplicates_empty_candidates_and_greedy_limitation():
    reference=np.array([[1,1,1,1,0,0]],float)
    parts=np.array([[[1,1,0,0,0,0]],[[0,0,1,1,0,0]],[[1,1,1,1,1,1]]],bool)
    ids,_,trace=shape_union(parts,reference)
    assert ids==[2] and trace==[2/3]  # Feasible greedy output, not global optimum.
    ids,_,_=shape_union(np.concatenate([parts[:2],parts[:1]]),reference)
    assert ids==[0,1]
    ids,union,trace=shape_union(np.zeros((2,1,6),bool),reference)
    assert not ids and not union.any() and not trace
    with pytest.raises(ValueError):shape_union(parts,np.zeros((1,6)))


def test_visible_reference_does_not_remove_occluders_or_promote_unknown_mass():
    import torch
    from radio_gs.query_field.reference_scope import visible_region_scores
    # Pixel0: front occluder owns90% transmittance, rear target only10%.
    # Pixel1: target visible. Pixel2: no field contribution.
    operator=torch.tensor([[.9,.1],[0.,1.],[0.,0.]]).to_sparse()
    region=torch.tensor([[0.],[1.]])
    actual=visible_region_scores(operator,torch.tensor([True,True]),region)
    torch.testing.assert_close(actual,torch.tensor([[.1],[1.],[0.]]))
    # Removing the occluder would make the rear target appear foreground at0.
    removed=torch.tensor([[0.,1.],[0.,1.],[0.,0.]]).to_sparse()
    assert visible_region_scores(removed,torch.ones(2,dtype=torch.bool),region)[0]>.5
    # A tiny observed fraction must not be renormalized to positive certainty.
    unknown=visible_region_scores(operator,torch.tensor([False,True]),region)
    assert unknown[0]==0 and unknown[1]==1
