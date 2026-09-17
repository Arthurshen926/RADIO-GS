import pytest
from radio_gs.training.scope_validation_cohort import complete_scope_cohort


def test_scopes_only_in_validation_are_counted_as_misses():
    # Both detections fall in the same heldout fold, so no training scope exists.
    rows,added=complete_scope_cohort([],list(range(8)),[[0,4]],['baseline','new'])
    assert added==2 and {r['frame'] for r in rows}=={0,4}
    assert all(r['new']['iou']==0 and not r['has_training_evidence'] for r in rows)
    same,added=complete_scope_cohort(rows,list(range(8)),[[0,4]],['baseline','new'])
    assert same==rows and added==0


def test_missing_supported_prediction_cannot_be_silently_filled():
    with pytest.raises(ValueError,match='available'):
        complete_scope_cohort([],list(range(8)),[[0,1]],['baseline'])
