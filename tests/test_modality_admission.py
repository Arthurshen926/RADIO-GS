import pytest
from radio_gs.training.modality_admission import admitted_training_modalities
from radio_gs.training.retrieval_supervision import balanced_episode_coverage


def test_missing_text_requires_explicit_admission_and_no_fabricated_updates():
    counts = dict(text=0, image=7, prompt=3)
    with pytest.raises(ValueError, match='text'):
        admitted_training_modalities(counts)
    active = admitted_training_modalities(counts, allow_missing_text=True)
    assert active == ['image', 'prompt']
    report = balanced_episode_coverage({m: counts[m] for m in active}, 28)
    assert report['complete'] and report['updates'] == {'image': 14, 'prompt': 14}
    assert 'text' not in report['visited']


def test_empty_image_or_prompt_does_not_silently_pass():
    with pytest.raises(ValueError):
        admitted_training_modalities(dict(text=0, image=2, prompt=0), allow_missing_text=True)
    assert admitted_training_modalities(dict(text=1, image=2, prompt=3)) == ['text', 'image', 'prompt']


def test_empty_grounded_bundle_keeps_axis_and_explicit_permission_checks():
    import torch
    from radio_gs.training.pipeline_contracts import validate_grounded_supervision
    bundle=dict(schema='grounded_text_supervision.v2',negative_policy='same_view_positive_detection_required',queries=['plant'],train_frames=[1],tasks=[],positive_pixels=torch.empty(0,2,dtype=torch.bool),negative_pixels=torch.empty(0,2,dtype=torch.bool),positive_elements=torch.zeros(1,3,dtype=torch.bool),negative_elements=torch.zeros(1,3,dtype=torch.bool),empty_text_supervision_explicitly_allowed=True,direct_text_supervision_available=False)
    kwargs=dict(frames=[1],train_frames=[1],proposal_views=torch.tensor([0]),num_elements=3,num_pixels=2,queries=['plant'])
    with pytest.raises(ValueError):validate_grounded_supervision(bundle,**kwargs)
    result=validate_grounded_supervision(bundle,allow_empty=True,**kwargs)
    assert result['tasks']==0 and not result['direct_text_supervision_available']
    bundle['positive_pixels']=torch.empty(0,3,dtype=torch.bool)
    with pytest.raises(ValueError,match='axes'):validate_grounded_supervision(bundle,allow_empty=True,**kwargs)
