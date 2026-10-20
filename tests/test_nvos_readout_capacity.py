import numpy as np
from sklearn.metrics import average_precision_score
from scripts.research.audit_tmm_nvos_readout_capacity import curve_capacity


def test_oracle_curve_groups_ties_and_matches_independent_decisions():
    rng=np.random.default_rng(43)
    scores=rng.integers(0,13,257).astype(np.float32)/12
    truth=rng.random(257)<.3
    actual=curve_capacity(scores,truth)
    exhaustive=max(float((truth&(scores>=t)).sum())/float((truth|(scores>=t)).sum()) for t in np.unique(scores))
    assert abs(actual['oracle_iou']-exhaustive)<1e-12
    assert abs(actual['average_precision']-average_precision_score(truth,scores))<1e-12
    assert actual['unique_decisions']==len(np.unique(scores))


def test_constant_scores_cannot_break_ties_using_labels():
    actual=curve_capacity(np.ones(4),np.array([True,False,False,True]))
    assert actual['oracle_iou']==actual['average_precision']==.5
    assert actual['unique_decisions']==1
