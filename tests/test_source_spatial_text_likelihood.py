from pathlib import Path

import torch

from radio_gs.querying.source_spatial_text_likelihood import (
    BoundedSourceSpatialLikelihoodHead,
    FIXED_NEIGHBOR_COUNT,
    MAX_ABS_LOG_ODDS_RESIDUAL,
    SOURCE_SPATIAL_SHARD_SCHEMA,
    SourceSpatialLikelihoodInputs,
    apply_bounded_log_odds_residual,
    fixed_knn_indices,
    fixed_spatial_logit_statistics,
    sha256_file,
    tensor_sha256,
    validate_source_spatial_shard,
)
from radio_gs.scripts.build_source_spatial_text_likelihood_shard import (
    canonical_region_xyz,
)


def test_zero_bounded_residual_is_bitwise_legacy_and_endpoints_absorb() -> None:
    legacy = torch.tensor([[0.0, 0.2, 0.8, 1.0]])
    zero = torch.zeros_like(legacy)
    assert torch.equal(apply_bounded_log_odds_residual(legacy, zero), legacy)
    changed = apply_bounded_log_odds_residual(
        legacy, torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    )
    assert changed[0, 0] == 0 and changed[0, 3] == 1
    assert changed[0, 1] > legacy[0, 1]
    assert changed[0, 2] < legacy[0, 2]


def test_fixed_spatial_statistics_are_query_independent_and_exact() -> None:
    xyz = torch.stack(
        (torch.arange(12, dtype=torch.float32), torch.zeros(12), torch.zeros(12)),
        dim=1,
    )
    indices = fixed_knn_indices(xyz)
    raw = torch.arange(24, dtype=torch.float32).reshape(12, 1, 2)
    valid = torch.ones(12, dtype=torch.bool)
    mean, maximum, contrast = fixed_spatial_logit_statistics(
        raw, indices, valid=valid
    )
    assert indices.shape == (12, FIXED_NEIGHBOR_COUNT)
    assert mean.shape == maximum.shape == contrast.shape == raw.shape
    assert torch.equal(contrast, raw - mean)
    gathered = raw[indices[5]]
    assert torch.equal(mean[5], gathered.mean(dim=0))
    assert torch.equal(maximum[5], gathered.amax(dim=0))


def test_zero_initialized_spatial_head_is_default_off_bitwise() -> None:
    raw = torch.tensor(
        [[[-2.0, 0.0]], [[1.0, 2.0]], [[-0.5, 0.5]]], dtype=torch.float32
    )
    inputs = SourceSpatialLikelihoodInputs(
        raw_logit=raw,
        neighbor_mean_logit=raw * 0.8,
        neighbor_max_logit=raw + 0.2,
        neighbor_contrast_logit=raw * 0.2,
        coverage=torch.tensor([1.0, 0.5, 0.2]),
        reliability=torch.tensor([0.8, 0.5, 0.1]),
    )
    head = BoundedSourceSpatialLikelihoodHead()
    head.reset_parameters_deterministic()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
    probability, residual = head(inputs)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.equal(probability, torch.sigmoid(raw))
    with torch.no_grad():
        head.output.bias.fill_(0.5)
    probability, residual = head(inputs)
    assert bool((residual.abs() <= MAX_ABS_LOG_ODDS_RESIDUAL).all())
    assert bool((probability[residual > 0] > torch.sigmoid(raw)[residual > 0]).all())


def test_region_xyz_uses_only_mpr_membership_and_query_independent_graph() -> None:
    accepted = {
        "region_rows": torch.tensor([[0, 1, -1], [2, 3, -1]]),
        "token_mask": torch.tensor([[True, True, False], [True, True, False]]),
    }
    graph = {
        "num_global_rows": 4,
        "global_rows": torch.tensor([0, 1, 2, 3]),
        "xyz": torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 0.0]]
        ),
    }
    assert torch.equal(
        canonical_region_xyz(accepted, graph),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
    )


def test_source_spatial_shard_is_hash_bound_and_fail_closed(tmp_path: Path) -> None:
    lineage = {}
    for name in (
        "source_text_training_shard",
        "accepted_region_authority",
        "query_independent_support_graph",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        lineage[name] = {"path": str(path), "sha256": sha256_file(path)}
    rows, classes = 12, 2
    xyz = torch.stack(
        (torch.arange(rows, dtype=torch.float32), torch.zeros(rows), torch.zeros(rows)),
        dim=1,
    )
    neighbors = fixed_knn_indices(xyz)
    raw = torch.linspace(-2, 2, rows * classes).reshape(rows, 1, classes)
    valid = torch.ones(rows, dtype=torch.bool)
    mean, maximum, contrast = fixed_spatial_logit_statistics(raw, neighbors, valid=valid)
    tensors = {
        "raw_logit": raw,
        "neighbor_mean_logit": mean,
        "neighbor_max_logit": maximum,
        "neighbor_contrast_logit": contrast,
        "region_xyz": xyz,
        "neighbor_indices": neighbors,
        "semantic_class_distribution": torch.softmax(torch.randn(rows, classes), dim=1),
        "valid": valid,
        "coverage": torch.ones(rows),
        "reliability": torch.full((rows,), 0.8),
        "training_label_weight": torch.ones(rows),
    }
    payload = {
        "schema": SOURCE_SPATIAL_SHARD_SCHEMA,
        "schema_version": 1,
        "scene_id": "scene0001_00",
        "physical_space_id": "scene0001",
        "partition": "source_train",
        "neighbor_count": FIXED_NEIGHBOR_COUNT,
        "class_ids": [1, 2],
        "class_names": ["wall", "floor"],
        **tensors,
        "channel_sha256": {name: tensor_sha256(value) for name, value in tensors.items()},
        "lineage": lineage,
        "source_access": {
            "official_scannet_train_scene": True,
            "source_train_semantic_labels_opened": True,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "lerf_queries_or_ground_truth_opened": False,
            "benchmark_predictions_or_metrics_opened": False,
            "per_scene_or_per_query_metric_tuning": False,
        },
    }
    validate_source_spatial_shard(payload)
    changed = dict(payload)
    changed["raw_logit"] = raw.clone()
    changed["raw_logit"][0, 0, 0] += 0.1
    try:
        validate_source_spatial_shard(changed)
    except ValueError as error:
        assert "channel changed" in str(error)
    else:
        raise AssertionError("source spatial shard accepted a changed tensor")
