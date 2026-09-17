"""Explicit semantic source/validation splits for teacher evidence bundles."""


def source_view_split(frames, validation_frames=None):
    frames=list(frames)
    validation=list(frames[-2:] if validation_frames is None else validation_frames)
    if len(frames)!=len(set(frames)) or len(validation)!=len(set(validation)):
        raise ValueError('duplicate frame in source split')
    if not validation or not set(validation)<=set(frames):
        raise ValueError('validation frames must be a nonempty source subset')
    train=[frame for frame in frames if frame not in set(validation)]
    if not train:
        raise ValueError('at least one training frame is required')
    return train,validation


def episode_in_validation_frames(episode, proposal_views, frames, selected_frames):
    if episode['split'] != 'validation':
        return False
    query_frame=frames[int(proposal_views[episode['query_region']])]
    target_frame=frames[episode['target_view']]
    return query_frame in selected_frames and target_frame in selected_frames
