import pytest
import torch
import json

from radio_gs.v4.object_memory.learned_fragment_affinity import (
    PAIR_FEATURE_LAYOUT,
    FragmentAffinityMLP,
    SetConditionedFragmentAffinity,
    balanced_fragment_affinity_loss,
    checkpoint_model,
    fragment_affinity_metrics,
    fragment_pair_features,
    select_fragment_affinity_threshold,
)
from radio_gs.v4.training.train_scannet_fragment_affinity import (
    _cohort_scene_ids,
    _object_equal_pair_weights,
    _source_view_fragments,
    fragment_partition_metrics,
)


def test_fragment_pair_features_have_fixed_cardinality_free_layout():
    features = fragment_pair_features(
        torch.tensor([0.5, 0.2]),
        torch.tensor([0.8, 0.3]),
        torch.tensor([0.4, 0.9]),
        torch.tensor([0.1, 0.7]),
    )
    assert features.shape == (2, 8)
    assert features[0, 4] == pytest.approx(0.4)
    assert features[0, 7] == pytest.approx(0.72)


def test_balanced_affinity_loss_requires_both_edge_classes():
    with pytest.raises(ValueError, match="positive and negative"):
        balanced_fragment_affinity_loss(torch.zeros(2), torch.ones(2, dtype=torch.bool))


def test_affinity_metrics_report_exact_confusion_and_ap():
    metrics = fragment_affinity_metrics(
        torch.tensor([0.9, 0.8, 0.4, 0.1]),
        torch.tensor([1, 0, 1, 0], dtype=torch.bool),
    )
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["average_precision"] == pytest.approx((1 + 2 / 3) / 2)


def test_checkpoint_reconstructs_affinity_model():
    model = FragmentAffinityMLP(7)
    checkpoint = {
        "schema": "radio_gs.surface_object_memory_v4.fragment_affinity_checkpoint.v1",
        "pair_feature_layout": [
            "surface_geometry",
            "calibrated_appearance",
            "spatial_proximity",
            "visibility_conflict",
            "geometry_x_appearance",
            "appearance_x_spatial",
            "geometry_x_nonconflict",
            "appearance_x_nonconflict",
        ],
        "hidden_dimension": 7,
        "model_state_dict": model.state_dict(),
    }
    restored = checkpoint_model(checkpoint)
    assert restored.hidden_dimension == 7


def test_set_conditioned_affinity_is_permutation_equivariant():
    torch.manual_seed(3)
    model = SetConditionedFragmentAffinity(7).eval()
    base = torch.rand(4, 4, 8)
    base = 0.5 * (base + base.transpose(0, 1))
    views = torch.tensor([0, 1, 0, 2])
    permutation = torch.tensor([2, 0, 3, 1])
    expected = model(base, views)
    actual = model(
        base[permutation][:, permutation], views[permutation]
    )
    assert torch.allclose(
        actual,
        expected[permutation][:, permutation],
        atol=1e-6,
    )
    assert torch.allclose(actual, actual.T)


def test_checkpoint_reconstructs_set_conditioned_model():
    model = SetConditionedFragmentAffinity(7)
    checkpoint = {
        "schema": "radio_gs.surface_object_memory_v4.fragment_affinity_checkpoint.v1",
        "pair_feature_layout": list(PAIR_FEATURE_LAYOUT),
        "hidden_dimension": 7,
        "model_kind": "set_context_mlp",
        "model_state_dict": model.state_dict(),
    }
    restored = checkpoint_model(checkpoint)
    assert isinstance(restored, SetConditionedFragmentAffinity)


def test_fragment_affinity_reads_nested_sealed_cohort_split(tmp_path):
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps({
        "schema": "test",
        "scene_ids": ["a", "b", "c"],
        "split": {
            "training_scene_ids": ["a", "b"],
            "validation_scene_ids": ["c"],
        },
    }))
    training, validation, receipt = _cohort_scene_ids(path)
    assert training == ["a", "b"]
    assert validation == ["c"]
    assert receipt["schema"] == "test"


def test_threshold_selection_prefers_precision_for_global_merges():
    selected = select_fragment_affinity_threshold(
        torch.tensor([0.95, 0.8, 0.7, 0.6]),
        torch.tensor([1, 0, 1, 0], dtype=torch.bool),
        beta=0.5,
    )
    assert selected["decision_threshold"] == pytest.approx(0.95)
    assert selected["precision"] == 1.0
    assert selected["recall"] == 0.5


def test_sam_like_fragments_include_parts_and_bounded_light_merges():
    visible = torch.arange(12)
    token_index = torch.tensor([0] * 6 + [1] * 6)
    centres = torch.stack((torch.arange(12), torch.zeros(12), torch.zeros(12)), -1).float()
    fragments = _source_view_fragments(
        visible=visible,
        token_index=token_index,
        centres=centres,
        object_count=2,
        view_id=0,
        noise_mode="sam_like_v1",
        parts_per_object=2,
        retained_fraction=0.85,
        light_merge_fraction=0.15,
    )
    assert len(fragments) == 6
    assert any(kind == "part" for _, _, kind, _ in fragments)
    assert any(kind == "light_merge" for _, _, kind, _ in fragments)
    assert min(purity for _, _, _, purity in fragments) > 0.5


def test_object_equal_pair_weights_equalize_pair_groups_and_classes():
    first = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1])
    second = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2])
    target = torch.tensor([1, 1, 0, 1, 0, 0, 0, 0], dtype=torch.bool)
    weight = _object_equal_pair_weights(first, second, target)
    assert weight[target].sum() == pytest.approx(1.0)
    assert weight[~target].sum() == pytest.approx(1.0)
    # Two pairs for object 0 have the same total mass as one pair for object 1.
    assert weight[:2].sum() == pytest.approx(weight[3])
    # Three negatives for pair (0,1) equal each singleton object-pair group.
    assert weight[2] + weight[4] + weight[5] == pytest.approx(weight[6])


def test_partition_metrics_expose_fragmentation_and_merge_impurity():
    dataset = {
        "pair_first": torch.tensor([0, 0, 1, 1]),
        "pair_second": torch.tensor([2, 3, 2, 3]),
        "fragment_view_index": torch.tensor([0, 0, 1, 1]),
        "fragment_object_index": torch.tensor([0, 1, 0, 1]),
        "fragment_geometry": torch.eye(4),
        "partition_cannot_link_policy": "exact_same_view",
    }
    clean = fragment_partition_metrics(
        torch.tensor([0.9, 0.1, 0.1, 0.9]),
        dataset,
        decision_threshold=0.5,
    )
    assert clean["predicted_hypothesis_count"] == 2
    assert clean["mean_object_fragmentation"] == 1
    assert clean["object_best_hypothesis_recall"] == 1
    assert clean["fragment_weighted_merge_impurity"] == 0

    merged = fragment_partition_metrics(
        torch.tensor([0.1, 0.9, 0.9, 0.1]),
        dataset,
        decision_threshold=0.5,
    )
    assert merged["fragment_weighted_merge_impurity"] > 0


def test_partition_validation_does_not_forbid_partial_same_view_overlap():
    dataset = {
        "pair_first": torch.tensor([0]),
        "pair_second": torch.tensor([1]),
        "fragment_view_index": torch.tensor([0, 0]),
        "fragment_object_index": torch.tensor([0, 0]),
        "fragment_geometry": torch.tensor([[1., 5 / 12], [5 / 12, 1.]]),
        "partition_cannot_link_policy": "disjoint_same_view",
    }
    result = fragment_partition_metrics(torch.tensor([0.9]), dataset, decision_threshold=0.5)
    assert result["predicted_hypothesis_count"] == 1
    assert result["object_best_hypothesis_recall"] == 1
    dataset["fragment_geometry"] = torch.eye(2)
    result = fragment_partition_metrics(torch.tensor([0.9]), dataset, decision_threshold=0.5)
    assert result["predicted_hypothesis_count"] == 2
