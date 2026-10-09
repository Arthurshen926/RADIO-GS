import numpy as np
from scripts.research.diagnose_query_field_region_composition import greedy_union


def test_disjoint_parts_compose_while_background_extension_is_rejected():
    target=np.array([[1,1,1,1,0,0]],bool)
    parts=np.array([[[1,1,0,0,0,0]],[[0,0,1,1,0,0]],[[1,1,1,1,1,1]]],bool)
    # The broad candidate wins initially (2/3), exposing the greedy limitation:
    # union cannot remove its background. This is NOT a global optimality oracle.
    ids,union,trace=greedy_union(parts,target)
    assert ids==[2] and trace==[2/3]
    ids,union,trace=greedy_union(parts[:2],target)
    assert ids==[0,1] and trace==[.5,1.]
    assert np.array_equal(union,target)


def test_duplicate_and_zero_gain_do_not_add_regions():
    target=np.array([[1,0]],bool);parts=np.array([target,target,[[0,1]]])
    ids,union,trace=greedy_union(parts,target)
    assert ids==[0] and trace==[1.] and np.array_equal(union,target)
