import numpy as np
import pytest
from radio_gs.querying.partial_response_prompt import partial_response_prompts
from radio_gs.querying.unboxed_response_prompt import unboxed_response_prompts


def test_shared_identity_points_and_only_negative_dense_evidence_changes():
    p = np.full((15, 25), .01, np.float32)
    p[3:12, 7:22] = .99
    variants = partial_response_prompts(p, (15, 25))
    a,b,c = [variants[k] for k in ('signed_core','positive_core','sparse_core')]
    np.testing.assert_array_equal(a['point_coords'], b['point_coords'])
    np.testing.assert_array_equal(a['point_coords'], c['point_coords'])
    assert len(a['point_coords']) == len(set(map(tuple,a['point_coords']))) == 8
    x,y = a['point_coords'].astype(int).T
    assert (p[y,x] >= .5).all()
    assert (a['point_labels'] == 1).all()
    np.testing.assert_array_equal(a['box'], [7,3,22,12])
    positive = a['mask_input'] > 0
    np.testing.assert_array_equal(a['mask_input'][positive], b['mask_input'][positive])
    assert (b['mask_input'][~positive] == 0).all()
    assert 'mask_input' not in c


def test_empty_single_pixel_border_and_determinism():
    assert partial_response_prompts(np.zeros((3,7),np.float32),(4,4)) is None
    p = np.zeros((3,7),np.float32);p[0,6]=1
    result = partial_response_prompts(p,(4,4))['sparse_core']
    np.testing.assert_array_equal(result['point_coords'], [[6,0]])
    np.testing.assert_array_equal(result['box'], [6,0,7,1])
    full = np.ones((6,9),np.float32)
    a = partial_response_prompts(full,(4,4))['sparse_core']
    b = partial_response_prompts(full,(4,4))['sparse_core']
    np.testing.assert_array_equal(a['point_coords'], b['point_coords'])
    assert len(a['point_coords']) == 8
    with pytest.raises(ValueError):partial_response_prompts(full,(4,4),points=0)


def test_unboxed_intervention_preserves_identity_and_signed_prior():
    p=np.full((12,19),.01,np.float32);p[3:9,6:13]=.99
    old=partial_response_prompts(p,(8,8));new=unboxed_response_prompts(p,(8,8))
    assert 'box' in new['boxed_sparse'] and 'box' not in new['unboxed_sparse']
    assert 'box' not in new['unboxed_signed'] and 'mask_input' not in new['unboxed_sparse']
    for variant in new.values():
        np.testing.assert_array_equal(variant['point_coords'],old['sparse_core']['point_coords'])
        np.testing.assert_array_equal(variant['point_labels'],old['sparse_core']['point_labels'])
    np.testing.assert_array_equal(new['unboxed_signed']['mask_input'],old['signed_core']['mask_input'])
    assert unboxed_response_prompts(np.zeros((3,7),np.float32),(8,8)) is None
