import pytest
from radio_gs.training.query_encoding_fixture import select_encoding_fixture


def test_missing_validation_is_not_silently_training_validation():
    episodes=[dict(split='train',modality='text',text_index=2)]
    with pytest.raises(ValueError):select_encoding_fixture(episodes,'text')
    episode,scope=select_encoding_fixture(episodes,'text',allow_training_input=True)
    assert episode is episodes[0] and scope=='training_episode_encoding_only_not_validation'
    assert episodes[0]['split']=='train'


def test_existing_validation_remains_preferred_and_missing_modality_fails():
    episodes=[dict(split='train',modality='text'),dict(split='validation',modality='text')]
    assert select_encoding_fixture(episodes,'text',allow_training_input=True)==(episodes[1],'validation_episode_input')
    with pytest.raises(ValueError):select_encoding_fixture(episodes,'image',allow_training_input=True)
