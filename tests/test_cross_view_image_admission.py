import pytest
from radio_gs.training.reliable_correspondence import cross_view_image_tasks_only


def test_only_same_view_image_membership_removed_without_mutating_other_tasks():
    episodes=[(i,dict(split='train',modality=m,query_region=q,target_region=t)) for i,(m,q,t) in enumerate([('image',0,0),('image',0,1),('image',0,2),('prompt',0,0),('text',0,0)])]
    result,report=cross_view_image_tasks_only(episodes,[0,0,1])
    assert [i for i,e in result]==[2,3,4]
    assert report['removed_same_view_image_tasks']==2 and len(episodes)==5
    assert result[1][1] is episodes[3][1]


def test_empty_cross_view_image_modality_fails_closed():
    with pytest.raises(ValueError,match='no cross-view'):
        cross_view_image_tasks_only([(0,dict(split='train',modality='image',query_region=0,target_region=0))],[0])
