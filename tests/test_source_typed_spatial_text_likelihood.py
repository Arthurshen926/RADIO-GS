from pathlib import Path

import torch

from radio_gs.querying.source_spatial_text_likelihood import (
    MAX_ABS_LOG_ODDS_RESIDUAL,
    sha256_file,
    tensor_sha256,
)
from radio_gs.querying.source_typed_spatial_text_likelihood import (
    BoundedTypedSourceSpatialLikelihoodHead,
    FROZEN_EDGE_TYPES,
    SOURCE_TYPED_SPATIAL_SHARD_SCHEMA,
    SourceTypedSpatialLikelihoodInputs,
    normalize_typed_region_edges,
    typed_spatial_logit_statistics,
    validate_source_typed_spatial_shard,
)
from radio_gs.scripts.build_source_typed_spatial_text_likelihood_shard import (
    aggregate_typed_primitive_edges_to_regions,
)


def test_typed_edges_normalize_independently_per_receiver() -> None:
    index = torch.tensor([[0, 0, 1, 1], [1, 2, 0, 2]])
    _, weight = normalize_typed_region_edges(
        index, torch.tensor([1.0, 3.0, 2.0, 2.0]), row_count=3
    )
    assert torch.equal(weight, torch.tensor([0.25, 0.75, 0.5, 0.5]))


def test_typed_mpr_aggregation_preserves_fractional_overlap_and_scales() -> None:
    accepted = {
        "region_rows": torch.tensor(
            [[0, 1, -1], [1, 2, -1], [3, 4, -1], [4, 5, -1]]
        ),
        "token_mask": torch.tensor(
            [[True, True, False], [True, True, False], [True, True, False], [True, True, False]]
        ),
        "scale_indices": torch.tensor([0, 0, 1, 1]),
    }
    support = {
        "num_global_rows": 6,
        "global_rows": torch.arange(6),
        "edge_index": torch.tensor(
            [[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]]
        ),
        "edge_channels": {
            "appearance": torch.tensor([1, 1, 2, 2, 1, 1, 2, 2], dtype=torch.float32),
            "boundary": torch.tensor([2, 2, 1, 1, 2, 2, 1, 1], dtype=torch.float32),
            "geometry": torch.ones(8),
        },
        "metadata": {"edge_channels": list(FROZEN_EDGE_TYPES)},
    }
    result = aggregate_typed_primitive_edges_to_regions(accepted, support)
    assert tuple(result) == FROZEN_EDGE_TYPES
    for edge_type in FROZEN_EDGE_TYPES:
        receiver, neighbor = result[edge_type]["edge_index"]
        assert not bool(((receiver < 2) & (neighbor >= 2)).any())
        assert not bool(((receiver >= 2) & (neighbor < 2)).any())
        sums = torch.zeros(4)
        sums.scatter_add_(0, receiver, result[edge_type]["edge_weight"])
        assert torch.allclose(sums[sums > 0], torch.ones_like(sums[sums > 0]))


def test_typed_statistics_and_bounded_head_keep_legacy_identity() -> None:
    raw = torch.tensor(
        [[[-2.0, 1.0]], [[0.0, 2.0]], [[1.0, -1.0]]], dtype=torch.float32
    )
    typed_edges = {
        name: {
            "edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            "edge_weight": torch.tensor([1.0, 0.4, 0.6, 1.0]),
        }
        for name in FROZEN_EDGE_TYPES
    }
    stats = typed_spatial_logit_statistics(
        raw, typed_edges, valid=torch.ones(3, dtype=torch.bool)
    )
    assert tuple(stats) == FROZEN_EDGE_TYPES
    assert torch.equal(
        stats["appearance"]["neighbor_contrast_logit"],
        raw - stats["appearance"]["neighbor_mean_logit"],
    )
    head = BoundedTypedSourceSpatialLikelihoodHead()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
    inputs = SourceTypedSpatialLikelihoodInputs(
        raw_logit=raw,
        typed_statistics=stats,
        coverage=torch.ones(3),
        reliability=torch.ones(3),
    )
    probability, residual = head(inputs)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.equal(probability, torch.sigmoid(raw))
    with torch.no_grad():
        head.output.bias.fill_(0.5)
    probability, residual = head(inputs)
    assert bool((residual.abs() <= MAX_ABS_LOG_ODDS_RESIDUAL).all())
    assert bool((probability > torch.sigmoid(raw)).all())


def test_typed_shard_is_hash_bound_and_fail_closed(tmp_path: Path) -> None:
    lineage = {}
    for name in (
        "source_text_training_shard",
        "accepted_region_authority",
        "query_independent_support_graph",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        lineage[name] = {"path": str(path), "sha256": sha256_file(path)}
    rows, classes = 4, 2
    raw = torch.linspace(-2, 2, rows * classes).reshape(rows, 1, classes)
    valid = torch.ones(rows, dtype=torch.bool)
    typed_edges = {
        name: {
            "edge_index": torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
            "edge_weight": torch.ones(4),
        }
        for name in FROZEN_EDGE_TYPES
    }
    stats = typed_spatial_logit_statistics(raw, typed_edges, valid=valid)
    base_tensors = {
        "raw_logit": raw,
        "semantic_class_distribution": torch.softmax(torch.randn(rows, classes), dim=1),
        "valid": valid,
        "coverage": torch.ones(rows),
        "reliability": torch.full((rows,), 0.8),
        "training_label_weight": torch.ones(rows),
        "canonical_region_scale_indices": torch.tensor([0, 0, 1, 1]),
    }
    tensors = dict(base_tensors)
    for edge_type in FROZEN_EDGE_TYPES:
        tensors[f"{edge_type}.edge_index"] = typed_edges[edge_type]["edge_index"]
        tensors[f"{edge_type}.edge_weight"] = typed_edges[edge_type]["edge_weight"]
        for name, tensor in stats[edge_type].items():
            tensors[f"{edge_type}.{name}"] = tensor
    payload = {
        "schema": SOURCE_TYPED_SPATIAL_SHARD_SCHEMA,
        "schema_version": 1,
        "scene_id": "scene0001_00",
        "physical_space_id": "scene0001",
        "partition": "source_train",
        "scale_count": 1,
        "edge_types": list(FROZEN_EDGE_TYPES),
        "primitive_to_region_aggregation": "same_scale_fractional_membership_MtAM",
        "class_ids": [1, 2],
        "class_names": ["wall", "floor"],
        **base_tensors,
        "typed_region_edges": typed_edges,
        "typed_statistics": stats,
        "channel_sha256": {name: tensor_sha256(tensor) for name, tensor in tensors.items()},
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
    validate_source_typed_spatial_shard(payload)
    changed = dict(payload)
    changed["raw_logit"] = raw.clone()
    changed["raw_logit"][0, 0, 0] += 0.1
    try:
        validate_source_typed_spatial_shard(changed)
    except ValueError as error:
        assert "channel changed" in str(error)
    else:
        raise AssertionError("typed spatial shard accepted changed logits")
