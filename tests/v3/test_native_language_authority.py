import json

import torch

from radio_gs.v3.evaluation.semantic_mapping_error_ladder import _metrics
from radio_gs.v3.training.build_native_language_authority import (
    _query_names,
    _relevancy,
    _same_components,
)
from radio_gs.v3.training.fit_query_discriminative_semantic_codec import _ternary_losses
from radio_gs.v3.training.fit_masked_semantic_writer import _heldout_mask


def test_query_names_reads_frozen_manifest(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps({"query_ids": ["red cup", "spoon"]}))

    assert _query_names(path) == ["red cup", "spoon"]


def test_relevancy_is_positive_against_hardest_canonical_null():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    text = torch.tensor([[1.0, 0.0]])
    negatives = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])

    values = _relevancy(features, text, negatives)

    assert values[0, 0] > 0.99
    assert values[1, 0] < 0.01


def test_same_components_only_unions_explicit_same_edges():
    left = torch.tensor([0, 1, 2])
    right = torch.tensor([1, 2, 3])
    labels = torch.tensor([1, -1, 0], dtype=torch.int8)

    components = _same_components(4, left, right, labels)

    assert components[0] == components[1]
    assert components[1] != components[2]
    assert components[2] != components[3]


def test_same_components_rejects_transitive_merge_with_repeated_view():
    left = torch.tensor([0, 1])
    right = torch.tensor([1, 2])
    labels = torch.ones(2, dtype=torch.int8)

    components = _same_components(
        3, left, right, labels,
        views=torch.tensor([0, 1, 0]), strength=torch.tensor([0.9, 0.8]),
    )

    assert components[0] == components[1]
    assert components[1] != components[2]


def test_semantic_metrics_exclude_unknown_candidates_from_ranking():
    scores = torch.tensor([[0.8], [0.7], [0.99], [0.1]])
    states = torch.tensor([[1], [0], [-1], [0]], dtype=torch.int8)

    result = _metrics(scores, states)

    assert result["recall_at_1"] == 1.0
    assert result["mrr"] == 1.0
    assert abs(result["margin"] - 0.1) < 1e-6


def test_ternary_codec_loss_ignores_unknown_and_rewards_separation():
    text = torch.eye(2)
    null = torch.tensor([[-1.0, 0.0]])
    states = torch.tensor([[1, 0], [0, 1], [-1, -1]], dtype=torch.int8)
    good = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    bad = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

    good_loss = sum(_ternary_losses(good, text, null, states).values())
    bad_loss = sum(_ternary_losses(bad, text, null, states).values())

    assert good_loss < bad_loss


def test_masked_semantic_writer_split_has_train_and_heldout_per_scene():
    rows = torch.arange(100)

    for scene_index in range(4):
        heldout = _heldout_mask(rows, scene_index)
        assert 15 <= int(heldout.sum()) <= 25
        assert bool((~heldout).any())
