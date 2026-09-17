from copy import deepcopy
import pytest
from radio_gs.training.context_text_supervision import context_text_tasks


def episode(q,t,neg=(),split='train'):
    return {'modality':'image','split':split,'query_region':q,'target_region':t,'target_view':1,'negative_regions':list(neg)}


def test_requires_both_endpoints_and_retains_unknown_negatives():
    e=[episode(0,1,[2,3])];before=deepcopy(e)
    tasks=context_text_tasks(e,[0,1,1,1],[4,4,5,6],[True,True,True,False])
    assert tasks[0][1]['negative_regions']==[2]
    assert tasks[0][1]['positive_regions']==[1]
    assert e==before
    assert not context_text_tasks(e,[0,1,1,1],[4,5,5,6],[True]*4)
    assert not context_text_tasks(e,[0,1,1,1],[4,4,5,6],[False,True,True,True])


def test_validation_is_not_used_to_increase_training_coverage():
    assert not context_text_tasks([episode(0,1,split='validation')],[0,1],[1,1],[True,True])


def test_same_text_target_view_unions_only_verified_edges():
    tasks=context_text_tasks([episode(0,1),episode(0,2)],[0,1,1,1],[1,1,1,1],[True]*4)
    assert len(tasks)==1
    assert tasks[0][1]['positive_regions']==[1,2]
    assert tasks[0][1]['source_episode_indices']==[0,1]


def test_reject_axis_mismatch():
    with pytest.raises(ValueError):context_text_tasks([],[],[0],[])
