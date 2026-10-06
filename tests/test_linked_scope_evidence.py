import pytest
from radio_gs.training.linked_scope_evidence import linked_phrase_regions


def record(frame, phrases):
    return dict(frame=frame, phrase_indices=phrases, confidence=.9)


def link(a, b):
    return dict(left_frame=a, right_frame=b, left_regions=[a], right_regions=[b],
                relation='mutual_support_candidate')


def test_no_transitive_identity_or_heldout_endpoint_leakage():
    records = [record(0, [7]), record(1, []), record(2, [])]
    original, added = linked_phrase_regions(records, [link(0, 1), link(1, 2)], 7)
    assert original == [0] and set(added) == {1}
    assert not added[1][0]['physical_identity_verified']
    assert linked_phrase_regions(records, [link(0, 1)], 7, [0]) == ([], {})
    assert linked_phrase_regions(records, [link(0, 1)], 7, [1]) == ([0], {})


def test_existing_incompatible_detection_is_not_silently_unioned():
    records = [record(0, [7]), record(1, []), record(1, [7])]
    original, added = linked_phrase_regions(records, [link(0, 1)], 7)
    assert original == [0, 2] and added == {}


def test_invalid_source_frame_cannot_hide_heldout_provenance():
    bad = dict(link(0, 1), left_frame=2)
    with pytest.raises(ValueError, match='endpoint'):
        linked_phrase_regions([record(0, [7]), record(1, [])], [bad], 7, [0])
