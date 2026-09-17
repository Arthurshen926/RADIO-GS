import pytest
from radio_gs.training.source_view_split import source_view_split


def test_explicit_holdout_is_not_silently_truncated_to_last_two():
    train,validation=source_view_split([1,2,3,4,5,6],[2,4,5,6])
    assert train==[1,3] and validation==[2,4,5,6]
    assert source_view_split([1,2,3,4])==([1,2],[3,4])


@pytest.mark.parametrize('frames,validation', [([1,1,2],[2]),([1,2],[2,2]),([1,2],[3]),([1,2],[1,2]),([1,2],[])])
def test_invalid_source_splits_are_rejected(frames,validation):
    with pytest.raises(ValueError):source_view_split(frames,validation)


def test_new_holdout_cohort_requires_query_and_target_to_be_new():
    from radio_gs.training.source_view_split import episode_in_validation_frames
    episode={'split':'validation','query_region':0,'target_view':1}
    assert not episode_in_validation_frames(episode,[0],[31,39],{39})
    assert episode_in_validation_frames(episode,[0],[31,39],{31,39})
    assert not episode_in_validation_frames({**episode,'split':'train'},[0],[31,39],{31,39})
