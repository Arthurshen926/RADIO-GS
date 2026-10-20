import numpy as np
import pytest
import torch

from radio_gs.querying.response_refinement import PartialResponseEvidence
from radio_gs.response_study.decision_evidence import CompleteResponseAgreement, whole_response_metrics


def item(target, positive, negative):
    n = len(target)
    return PartialResponseEvidence(torch.eye(n).to_sparse(), torch.tensor(target, dtype=torch.float32),
        torch.tensor(positive), torch.tensor(negative), torch.tensor(True), torch.tensor(1.))


def test_unknown_is_not_background_and_empty_retains():
    values = torch.tensor([[[0.], [0.]], [[1.], [1.]]])
    evidence = item([[1.], [0.]], [[True], [False]], [[False], [False]])
    audit = CompleteResponseAgreement(values); audit.add(evidence)
    assert audit.finish()['choices'] == [1]
    assert CompleteResponseAgreement(values).finish()['choices'] == [0]


def test_whole_menu_cannot_stitch_class_winners():
    values = torch.tensor([[[0., 1.]], [[1., 0.]]])
    evidence = item([[1., 1.]], [[True, True]], [[False, False]])
    separate = CompleteResponseAgreement(values); separate.add(evidence)
    assert separate.finish()['choices'] == [1, 0]
    menu = CompleteResponseAgreement(values, whole_menu=True); menu.add(evidence)
    assert menu.finish()['choices'] == [0]


def test_duplicate_pixels_do_not_change_equal_view_weight_and_query_order():
    values = torch.tensor([[[0., .2]], [[1., .8]]])
    audit = CompleteResponseAgreement(values, whole_menu=True)
    audit.add(item([[1., 1.]], [[True, True]], [[False, False]]))
    reverse = CompleteResponseAgreement(values.flip(2), whole_menu=True)
    reverse.add(item([[1., 1.]], [[True, True]], [[False, False]]))
    np.testing.assert_allclose(audit.finish()['gains'], reverse.finish()['gains'])
    repeated = CompleteResponseAgreement(values.repeat(1, 3, 1), whole_menu=True)
    repeated.add(item([[1., 1.]]*3, [[True, True]]*3, [[False, False]]*3))
    np.testing.assert_allclose(audit.finish()['gains'], repeated.finish()['gains'])


def test_same_query_choice_across_frames_and_metrics():
    a = [dict(frame='a', query='q', iou=.1, ap=.9), dict(frame='b', query='q', iou=.8, ap=.1)]
    b = [dict(frame='b', query='q', iou=.2, ap=.6), dict(frame='a', query='q', iou=.9, ap=.4)]
    assert whole_response_metrics([a, b], {'q': 1}, a) == {'iou': .55, 'ap': .5}
    with pytest.raises(ValueError, match='record sets'):
        whole_response_metrics([a, b[:1]], {'q': 1}, a)


def test_permutation_keeps_joint_permissions_and_rejects_bad_indices():
    values = torch.tensor([[[1.], [0.]], [[0.], [1.]]])
    evidence = item([[1.], [0.]], [[True], [False]], [[False], [True]])
    original = CompleteResponseAgreement(values); original.add(evidence)
    permuted = CompleteResponseAgreement(values); permuted.add(evidence, permutation=[1, 0])
    assert original.finish()['choices'] == [0] and permuted.finish()['choices'] == [1]
    with pytest.raises(ValueError, match='permute'):
        permuted.add(evidence, permutation=[1, 1])
