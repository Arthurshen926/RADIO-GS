import pytest
from radio_gs.response_study.residual_diagnosis import decision_attribution


def test_guard_can_prevent_harm_and_cannot_be_blamed_for_all_regret():
    row=dict(scene='room',query_index=0,gains=[0,.2,.1],ious=[.5,.3,.8],evidence_available=True)
    d=decision_attribution([row],.3)['decomposition']
    assert d['available_gain']==pytest.approx(.3)
    assert d['ranking_regret']==pytest.approx(.5)
    assert d['retention_cost']==pytest.approx(-.2)
    assert d['actual_gain']==0


def test_guard_cost_and_rank_error_partition_unrecovered_capacity():
    row=dict(scene='room',query_index=0,gains=[0,.2,.1],ious=[.5,.6,.8],evidence_available=True)
    d=decision_attribution([row],.3)['decomposition']
    assert d['ranking_regret']==pytest.approx(.2)
    assert d['retention_cost']==pytest.approx(.1)
    row['evidence_available']=False
    d=decision_attribution([row],.3)['decomposition']
    # Missing evidence prevents intervention even at zero margin, not a guard loss.
    assert d['ranking_regret']==pytest.approx(.3) and d['retention_cost']==0
