import pytest
from radio_gs.querying.reference_memory import reference_memory_paths


def test_paths_preserve_authorized_camera_order_and_reference_identity():
    assert reference_memory_paths([0,1,2,4,5],4,3)==dict(pair_memory=[4,3],sequence_memory=[4,3])
    p=reference_memory_paths([0,1,2,4,5],5,3)
    assert p['sequence_memory']==[5,4,3]
    p=reference_memory_paths([0,2,3,4,5],0,1)
    assert p['pair_memory']==p['sequence_memory']==[0,1]
    p=reference_memory_paths([0,2,3,4],0,5)
    assert p['sequence_memory']==[0,2,3,4,5]


def test_reject_contaminated_or_ambiguous_paths():
    for frames,ref,target in [([0,1],0,1),([0,0],0,1),([0,2],1,3),([0,2.],0,1)]:
        with pytest.raises(ValueError):reference_memory_paths(frames,ref,target)
