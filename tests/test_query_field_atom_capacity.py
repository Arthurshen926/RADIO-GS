import itertools
import numpy as np
from scripts.research.audit_query_field_atom_capacity import atom_capacity


def test_atomic_iou_optimum_matches_exhaustive_subsets():
    rng=np.random.default_rng(199)
    for _ in range(30):
        center=rng.integers(0,5,30);valid=rng.random(30)>.2;truth=rng.random(30)>.6
        expected=0.
        for mask in itertools.product([False,True],repeat=5):
            p=np.array(mask)[center]&valid;t=truth&valid
            expected=max(expected,(p&t).sum()/max((p|t).sum(),1))
        actual=atom_capacity(center,valid,truth,5)
        if (truth&valid).any():assert abs(actual['iou']-expected)<1e-12
        else:assert actual is None
