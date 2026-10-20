from types import SimpleNamespace

import numpy as np
import pytest

from radio_gs.querying.estimated_anchor_authority import estimated_anchor_patch, apply_patch


def fixture(probability=.2, modality='image'):
    reader = SimpleNamespace(saved=dict(threshold=.35, estimator=SimpleNamespace(
        predict_proba=lambda x: np.tile([1-probability, probability], (len(x), 1)))),
        evidence=SimpleNamespace(encode=lambda anchor, vertices: vertices[:, None]))
    detail = dict(query_id='query', modality=modality, inferred_anchor_membership=dict(
        anchor_vertex=1, anchor_identity_guaranteed=False, threshold=.35))
    return reader, detail, np.array([.8, 1., .1], np.float32), np.array([True, True, False])


def test_one_vertex_changes_without_mutating_the_preserved_provider():
    reader, detail, response, selected = fixture()
    patch = estimated_anchor_patch(reader, detail, response, selected)
    rank, mask = apply_patch(response, selected, patch)
    assert rank.tolist() == pytest.approx([.8, .2, .1])
    assert mask.tolist() == [True, False, False]
    assert response[1] == 1. and selected[1]
    assert patch['rank_entries_changed'] == patch['selection_entries_changed'] == 1


@pytest.mark.parametrize('modality', ['point_2d', 'point_3d', 'text'])
def test_control_cannot_weaken_actual_point_or_text_contract(modality):
    with pytest.raises(ValueError, match='only estimated image'):
        estimated_anchor_patch(*fixture(modality=modality))


def test_classifier_can_itself_reproduce_the_override():
    patch = estimated_anchor_patch(*fixture(probability=1.))
    assert patch['rank_entries_changed'] == patch['selection_entries_changed'] == 0


def test_threshold_is_frozen_and_uses_original_strict_comparison():
    # np.float32(.35) is below the original float64 .35 threshold.
    patch = estimated_anchor_patch(*fixture(probability=.35))
    assert not patch['unforced_selected']
    reader, detail, response, selected = fixture()
    detail['inferred_anchor_membership']['threshold'] = .5
    with pytest.raises(ValueError, match='threshold changed'):
        estimated_anchor_patch(reader, detail, response, selected)


def test_modified_baseline_cannot_receive_the_sealed_patch():
    reader, detail, response, selected = fixture()
    patch = estimated_anchor_patch(reader, detail, response, selected)
    response[1] = .8
    with pytest.raises(ValueError, match='baseline no longer matches'):
        apply_patch(response, selected, patch)
