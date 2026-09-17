"""Choose an encoding consistency input without inventing validation labels."""


def select_encoding_fixture(episodes, modality, *, allow_training_input=False):
    validation=[e for e in episodes if e['split']=='validation' and e['modality']==modality]
    if validation:
        return validation[0], 'validation_episode_input'
    if allow_training_input:
        training=[e for e in episodes if e['split']=='train' and e['modality']==modality]
        if training:
            return training[0], 'training_episode_encoding_only_not_validation'
    raise ValueError(f'no validation encoding fixture for {modality}; explicit training-input fallback required')
