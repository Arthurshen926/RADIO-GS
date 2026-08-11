from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from radio_gs.querying.query_likelihood_head import MonotoneQueryLikelihoodHead
from radio_gs.querying.source_text_query_likelihood import (
    LEGACY_FIELD_PRIOR_LOGIT_SCALE,
    SOURCE_TEXT_SCENE_INPUT_SCHEMA,
    build_source_text_training_shard,
    confidence_weighted_balanced_bce,
    iter_source_text_likelihood_examples,
    sha256_file,
    validate_source_text_training_shard,
)
from radio_gs.scripts.build_source_text_query_likelihood_dataset import (
    build_scene_shard,
    seal_dataset,
    validate_dataset_manifest,
)
from radio_gs.scripts.train_source_text_query_likelihood_head import (
    fit_source_text_head,
)


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _scene_input(tmp_path: Path) -> dict[str, object]:
    sources = {}
    for name in (
        "descriptor_source",
        "semantic_label_source",
        "class_text_source",
        "canonical_negative_text_source",
        "field_state_source",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(f"frozen-{name}".encode())
        sources[name] = _record(path)
    # Two clean semantic clusters plus small scale/view perturbations.
    descriptors = torch.tensor(
        [
            [[0.98, 0.08, 0.03], [0.93, 0.14, 0.02]],
            [[0.95, 0.12, 0.04], [0.91, 0.18, 0.01]],
            [[0.90, 0.20, 0.02], [0.94, 0.11, 0.03]],
            [[0.92, 0.16, 0.05], [0.89, 0.21, 0.02]],
            [[0.08, 0.98, 0.03], [0.14, 0.93, 0.02]],
            [[0.12, 0.95, 0.04], [0.18, 0.91, 0.01]],
            [[0.20, 0.90, 0.02], [0.11, 0.94, 0.03]],
            [[0.16, 0.92, 0.05], [0.21, 0.89, 0.02]],
        ],
        dtype=torch.float32,
    )
    return {
        "schema": SOURCE_TEXT_SCENE_INPUT_SCHEMA,
        "schema_version": 1,
        "scene_id": "scene_source_00",
        "physical_space_id": "scene_source",
        "partition": "source_train",
        "descriptors": descriptors,
        "semantic_label_ids": torch.tensor([1, 1, 1, 1, 2, 2, 2, 2]),
        "class_ids": [1, 2],
        "class_names": ["first class", "second class"],
        "class_text_embeddings": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        "canonical_negative_text_embeddings": torch.tensor(
            [[0.0, 0.0, 1.0], [-1.0, -1.0, 0.0]]
        ),
        "valid": torch.ones(8, dtype=torch.bool),
        "coverage": torch.tensor([1.0, 0.8, 0.7, 1.0, 1.0, 0.9, 0.6, 1.0]),
        "reliability": torch.tensor([0.9, 1.0, 0.8, 0.7, 1.0, 0.9, 0.8, 0.7]),
        "lineage": sources,
        "source_access": {
            "official_scannet_train_scene": True,
            "source_train_semantic_labels_opened": True,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "lerf_queries_or_ground_truth_opened": False,
            "target_rgb_or_mask_opened": False,
            "benchmark_predictions_or_metrics_opened": False,
            "per_scene_or_per_query_metric_tuning": False,
        },
    }


def test_source_text_shard_preserves_separate_evidence_channels(tmp_path: Path) -> None:
    source = _scene_input(tmp_path)
    shard = validate_source_text_training_shard(
        build_source_text_training_shard(source)
    )
    assert shard["positive_affinity"].shape == (8, 2, 2)
    assert shard["canonical_negative_affinity"].shape == (8, 2, 2)
    assert shard["field_prior_probability"].shape == (8, 2)
    examples = list(iter_source_text_likelihood_examples(shard))
    assert [example.class_id for example in examples] == [1, 2]
    first = examples[0]
    torch.testing.assert_close(first.observations.coverage, source["coverage"])
    torch.testing.assert_close(first.observations.reliability, source["reliability"])
    evidence = MonotoneQueryLikelihoodHead()(
        first.observations, source="source_text_test"
    )
    torch.testing.assert_close(
        evidence.confidence,
        torch.as_tensor(source["coverage"]) * torch.as_tensor(source["reliability"]),
    )


def test_field_prior_exactly_reproduces_frozen_margin_formula(tmp_path: Path) -> None:
    source = _scene_input(tmp_path)
    shard = build_source_text_training_shard(source)
    descriptor = torch.nn.functional.normalize(source["descriptors"], dim=-1)
    classes = torch.nn.functional.normalize(source["class_text_embeddings"], dim=-1)
    negatives = torch.nn.functional.normalize(
        source["canonical_negative_text_embeddings"], dim=-1
    )
    positive = torch.einsum("nsd,cd->nsc", descriptor, classes)
    negative = torch.einsum("nsd,kd->nsk", descriptor, negatives).amax(
        dim=-1, keepdim=True
    )
    expected = torch.sigmoid(
        LEGACY_FIELD_PRIOR_LOGIT_SCALE * (positive - negative)
    ).amax(dim=1)
    torch.testing.assert_close(shard["field_prior_probability"], expected)


def test_source_boundary_rejects_development_or_metric_access(tmp_path: Path) -> None:
    source = _scene_input(tmp_path)
    source["partition"] = "development"
    with pytest.raises(PermissionError, match="source_train"):
        build_source_text_training_shard(source)
    source = _scene_input(tmp_path)
    source["source_access"]["benchmark_predictions_or_metrics_opened"] = True
    with pytest.raises(PermissionError, match="benchmark_predictions_or_metrics_opened"):
        build_source_text_training_shard(source)


def test_source_text_shard_is_channel_hash_bound(tmp_path: Path) -> None:
    shard = build_source_text_training_shard(_scene_input(tmp_path))
    shard["field_prior_probability"] = shard["field_prior_probability"].clone()
    shard["field_prior_probability"][0, 0] *= 0.99
    with pytest.raises(ValueError, match="channel changed"):
        validate_source_text_training_shard(shard)


def test_balanced_loss_is_invariant_to_negative_row_duplication() -> None:
    q = torch.tensor([0.9, 0.7, 0.2, 0.1])
    y = torch.tensor([1.0, 1.0, 0.0, 0.0])
    weight = torch.ones(4)
    original, _ = confidence_weighted_balanced_bce(q, y, weight)
    repeated, _ = confidence_weighted_balanced_bce(
        torch.tensor([0.9, 0.7, 0.2, 0.1, 0.2, 0.1]),
        torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        torch.ones(6),
    )
    torch.testing.assert_close(original, repeated)


def test_balanced_loss_retains_soft_mixed_region_target() -> None:
    loss, diagnostics = confidence_weighted_balanced_bce(
        torch.tensor([0.8, 0.5, 0.1]),
        torch.tensor([1.0, 0.5, 0.0]),
        torch.ones(3),
    )
    assert torch.isfinite(loss)
    torch.testing.assert_close(
        diagnostics["positive_weight"], torch.tensor(1.5)
    )
    torch.testing.assert_close(
        diagnostics["negative_weight"], torch.tensor(1.5)
    )


def test_cpu_source_text_fit_improves_balanced_objective(tmp_path: Path) -> None:
    shard = build_source_text_training_shard(_scene_input(tmp_path))
    head, diagnostics = fit_source_text_head([shard])
    assert diagnostics["cuda_initialized"] is False
    assert diagnostics["final"]["macro_balanced_bce"] < diagnostics["initial"][
        "macro_balanced_bce"
    ]
    assert diagnostics["final"]["macro_positive_minus_negative_probability"] > 0
    assert all(value > 0 for value in diagnostics["parameters"]["positive_weights"])
    assert all(value > 0 for value in diagnostics["parameters"]["negative_weights"])
    assert diagnostics["parameters"]["prior_weight"] > 0
    assert head.training is False


def test_immutable_builder_and_manifest_round_trip(tmp_path: Path) -> None:
    source_path = tmp_path / "scene_input.pt"
    torch.save(_scene_input(tmp_path), source_path)
    shard_path, receipt_path, receipt = build_scene_shard(
        scene_input=source_path,
        output_shard=tmp_path / "scene_shard.pt",
        receipt=tmp_path / "scene_shard.receipt.json",
    )
    assert receipt["partition"] == "source_train"
    assert json.loads(receipt_path.read_text())["shard"]["sha256"] == sha256_file(
        shard_path
    )
    manifest_path, manifest = seal_dataset(
        shards=[shard_path], output=tmp_path / "dataset_manifest.json"
    )
    validated, payloads = validate_dataset_manifest(manifest_path)
    assert validated["scene_count"] == manifest["scene_count"] == 1
    assert len(payloads) == 1
    with pytest.raises(FileExistsError, match="immutable output"):
        build_scene_shard(
            scene_input=source_path,
            output_shard=shard_path,
            receipt=tmp_path / "second.receipt.json",
        )
