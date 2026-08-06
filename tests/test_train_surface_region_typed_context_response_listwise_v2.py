from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces.surface_region_typed_context_training import (
    build_typed_context_normalization_authority,
)
from radio_gs.losses.source_global_response_listwise_loss import (
    FrozenSourceResponseAuthority,
)
from radio_gs.models.surface_region_typed_context_residual import (
    SurfaceRegionAcceptedV2TypedContextResidualV1,
)
from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v2 as trainer,
)


_SHA = "a" * 64


def _normalization(rows: int) -> dict:
    generator = torch.Generator().manual_seed(71)
    values = torch.randn(rows, 30, generator=generator)
    values[:, 25] = torch.linspace(0.3, 0.8, rows)
    values[:, 28] = torch.linspace(-0.2, 0.7, rows)
    records = [
        {
            "scene_id": f"scene{index:04d}_00",
            "training_shard": {"path": f"/s/{index}.pt", "sha256": "b" * 64},
            "adaptive_context": {"path": f"/c/{index}.pt", "sha256": "c" * 64},
        }
        for index in range(24)
    ]
    return build_typed_context_normalization_authority(
        values,
        torch.ones(rows, dtype=torch.bool),
        source_state_cohort_authority_sha256=_SHA,
        train_input_records=records,
    )


def _fixture(rows: int = 4):
    normalization = _normalization(rows)
    generator = torch.Generator().manual_seed(73)
    base = F.normalize(torch.randn(rows, 1536, generator=generator), dim=-1)
    valid = torch.tensor([True, True, False, True], dtype=torch.bool)
    context = torch.zeros(rows, 1280)
    context[valid] = F.normalize(
        torch.randn(int(valid.sum()), 1280, generator=generator), dim=-1
    )
    full_scalar = normalization["median"][:18].repeat(rows, 1)
    statistics = torch.zeros(rows, 12)
    statistics[valid] = normalization["median"][18:].repeat(int(valid.sum()), 1)
    teacher_views = F.normalize(torch.randn(rows, 2, 1536, generator=generator), dim=-1)
    pair_rows = torch.arange(rows).repeat_interleave(2)
    scene = {
        "accepted_v2_e0": base,
        "raw_full_scalar_summary": full_scalar,
        "eligible": torch.ones(rows, dtype=torch.bool),
        "pooled_context_radio_direction": context,
        "typed_context_statistics": statistics,
        "typed_context_valid": valid,
        "official_multiview_siglip2_teacher_pair_region_indices": pair_rows,
        "official_multiview_siglip2_teacher_pair_descriptors": teacher_views.reshape(
            -1, 1536
        ),
        "region_row_ids": [f"region-{index}" for index in range(rows)],
    }
    text = F.normalize(torch.randn(11, 1536, generator=generator), dim=-1)
    anchors = torch.arange(rows)
    negatives = torch.roll(anchors, shifts=-1)
    teacher_consensus = F.normalize(teacher_views.mean(dim=1), dim=-1)
    teacher_cosines = (teacher_consensus[anchors] * teacher_consensus[negatives]).sum(
        dim=-1
    )
    payload = {
        "canonical_region_indices": torch.arange(rows),
        "channels": {
            "anchor_region_indices": anchors,
            "negative_region_indices": negatives,
            "teacher_cosines": teacher_cosines,
        },
    }
    authority = FrozenSourceResponseAuthority(
        payload=payload,
        file_sha256="1" * 64,
        content_authority_sha256="2" * 64,
        scene_id="scene0001_00",
        accepted_v2_file_sha256="3" * 64,
        teacher_file_sha256="4" * 64,
        teacher_pair_descriptors_sha256="5" * 64,
        fit_text_bank_file_sha256="6" * 64,
        fit_text_embedding_tensor_sha256=tensor_sha256(text),
    )
    model = SurfaceRegionAcceptedV2TypedContextResidualV1(
        scalar_median=normalization["median"],
        scalar_robust_scale=normalization["robust_scale"],
        max_angle_radians=0.15,
        max_alpha=0.25,
    )
    return normalization, scene, text, authority, model


def test_complete_scene_zero_init_preserves_fallback_and_has_finite_loss() -> None:
    normalization, scene, text, authority, model = _fixture()
    total, metrics = trainer.complete_scene_objective(
        model,
        scene,
        normalization,
        text,
        authority,
        torch.device("cpu"),
    )
    assert total.ndim == 0 and bool(torch.isfinite(total))
    assert metrics["complete_canonical_rows"] == 4
    assert metrics["active_rows"] == 3
    assert metrics["fallback_bitwise_accepted_v2_e0"] is True
    assert int(metrics["response_valid_profile_queries"]) == 11
    assert int(metrics["response_hard_negative_pairs"]) == 4
    assert trainer.training_contract()["frozen_v1"]["max_angle_radians"] == 0.15


def test_complete_scene_objective_updates_only_existing_residual_model() -> None:
    normalization, scene, text, authority, model = _fixture()
    before = scene["accepted_v2_e0"].clone()
    total, _ = trainer.complete_scene_objective(
        model,
        scene,
        normalization,
        text,
        authority,
        torch.device("cpu"),
    )
    total.backward()
    assert model.residual_projection.weight.grad is not None
    assert float(model.residual_projection.weight.grad.abs().sum()) > 0
    assert torch.equal(scene["accepted_v2_e0"], before)
    assert (
        trainer.training_contract()["frozen_auxiliary"]["new_learnable_parameters"]
        is False
    )


def test_partial_canonical_scene_is_rejected() -> None:
    normalization, scene, text, authority, model = _fixture()
    partial = replace(
        authority,
        payload={
            **authority.payload,
            "canonical_region_indices": torch.arange(3),
        },
    )
    with pytest.raises(ValueError, match="complete canonical scene"):
        trainer.complete_scene_objective(
            model,
            scene,
            normalization,
            text,
            partial,
            torch.device("cpu"),
        )


def test_selection_prefers_lower_response_loss_after_v1_gate() -> None:
    def record(epoch: int, loss: float, *, eligible: bool = True) -> dict:
        return {
            "epoch": epoch,
            "validation": {
                "selection_eligible": eligible,
                "response_listwise": {"scene_macro_auxiliary_loss": loss},
                "v1_non_regression": {
                    "candidate": {
                        "mean_all_view_cosine": 0.5,
                        "p05_row_mean_all_view_cosine": 0.4,
                        "relation_fidelity": 0.3,
                    }
                },
            },
        }

    assert trainer.select_best_epoch([record(0, 0.4), record(1, 0.3)]) == 1
    assert (
        trainer.select_best_epoch([record(0, 0.4), record(1, 0.2, eligible=False)]) == 0
    )


def test_execution_authority_rejects_unapproved_training() -> None:
    record = {"path": "/frozen", "sha256": _SHA}
    value = {
        "schema": trainer.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_after_complete_32_scene_preflight",
        "preregistration": dict(record),
        "implementation": dict(record),
        "fit_text_bank": dict(record),
        "cohort_authority": dict(record),
        "source_state_manifest": dict(record),
        "teacher_manifest": dict(record),
        "benchmark_exclusion_manifest": dict(record),
        "source_train": [],
        "source_validation": [],
        "full_training_authorized": False,
        "benchmark_execution_authorized": False,
        "source_access": trainer.source_access(),
    }
    with pytest.raises(ValueError, match="header"):
        trainer._validate_execution_authority(value)


def test_fit_bank_requires_the_full_declared_query_list(tmp_path) -> None:
    embeddings = F.normalize(torch.randn(806, 3), dim=-1)
    payload = {
        "split": "fit",
        "queries": [f"query-{index}" for index in range(806)],
        "benchmark_vocabulary_opened": False,
        "uses_benchmark_vocabulary_for_construction": False,
        "embedding_tensor_sha256": tensor_sha256(embeddings),
        "embeddings": embeddings,
    }
    path = tmp_path / "fit.pt"
    torch.save(payload, path)
    from radio_gs.utils.immutable_artifacts import sha256_file

    loaded = trainer.load_fit_text_bank(path, expected_sha256=sha256_file(path))
    assert loaded.embeddings.shape == (806, 3)


def test_cli_has_dry_and_authority_only_train_without_benchmark_options() -> None:
    parser = trainer.build_parser()
    text = parser.format_help()
    assert "dry-loss" in text and "train" in text
    assert "benchmark" not in text.lower()


def test_preregistration_binds_the_unchanged_frozen_v1_and_loss() -> None:
    repository = Path(__file__).resolve().parents[1]
    path = (
        repository
        / "paper/artifacts/surface_region_response_listwise_v2_execution_preregistration_20260806.json"
    )
    value = json.loads(path.read_text())
    assert trainer._validate_preregistration(value)["artifact"] == (
        trainer.PREREGISTRATION_ARTIFACT
    )
