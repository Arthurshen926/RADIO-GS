import numpy as np
from scripts.research.audit_query_field_native_scope_order import threshold_capacity


def test_threshold_oracle_matches_exhaustive_cutoffs_without_splitting_ties():
    rng=np.random.default_rng(20261012)
    for _ in range(50):
        scores=rng.choice([.2,.4,.7],size=7);area=rng.integers(1,15,size=7).astype(float)
        positive=np.array([rng.integers(0,int(a)+1) for a in area]);covered=rng.random(7)>.2
        got=threshold_capacity(scores,area,positive,covered)
        if not positive.sum():assert got is None;continue
        values=[]
        for t in [0,.2,.4,.7,1.]:
            selected=covered&(scores>t);tp=positive[selected].sum()
            values.append(tp/(positive.sum()+area[selected].sum()-tp))
        assert abs(got['iou']-max(values))<1e-12
    # Equal query response cannot isolate the pure atom from the impure one.
    result=threshold_capacity(np.array([.6,.6]),np.array([10.,10.]),np.array([10.,0.]),np.ones(2,dtype=bool))
    assert result['iou']==.5 and result['selected_atoms']==2
