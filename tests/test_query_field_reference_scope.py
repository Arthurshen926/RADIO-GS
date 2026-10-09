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
