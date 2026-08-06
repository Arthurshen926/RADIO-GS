from __future__ import annotations

import copy

import pytest
import torch
from torch.nn import functional as F

from radio_gs.interfaces.surface_region_typed_context_training import (
    apply_typed_context_normalization,
    build_typed_context_normalization_authority,
    build_typed_context_training_certificate,
    load_typed_context_checkpoint,
    typed_context_normalization_authority_sha256,
    validate_typed_context_normalization_authority,
    write_typed_context_checkpoint,
)
from radio_gs.models.surface_region_typed_context_residual import (
    SurfaceRegionAcceptedV2TypedContextResidualV1,
)
from radio_gs.scripts import train_surface_region_typed_context_residual as trainer
from radio_gs.utils.immutable_artifacts import (
    file_record,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


SOURCE_SHA = "a" * 64


def _input_records(count: int, *, prefix: str) -> list[dict]:
    return [
        {
            "scene_id": f"scene{index:04d}_00",
            "training_shard": {
                "path": f"/{prefix}/scene{index:04d}.shard.pt",
                "sha256": "b" * 64,
            },
            "adaptive_context": {
                "path": f"/{prefix}/scene{index:04d}.context.pt",
                "sha256": "c" * 64,
            },
        }
        for index in range(count)
    ]


def _normalization(*, rows: int = 8) -> dict:
    generator = torch.Generator().manual_seed(41)
    values = torch.randn(rows, 30, generator=generator)
    # Stage-B statistics have bounded resultant/cosine coordinates.
    values[:, 18 + 7] = torch.linspace(0.2, 0.9, rows)
    values[:, 18 + 10] = torch.linspace(-0.4, 0.8, rows)
    return build_typed_context_normalization_authority(
        values,
        torch.ones(rows, dtype=torch.bool),
        source_state_cohort_authority_sha256=SOURCE_SHA,
        train_input_records=_input_records(24, prefix="train"),
    )


def _scene(normalization: dict, *, rows: int = 6) -> dict:
    generator = torch.Generator().manual_seed(52)
    base = F.normalize(torch.randn(rows, 1536, generator=generator), dim=-1)
    context_valid = torch.tensor(
        [True, True, False, True, True, False][:rows], dtype=torch.bool
    )
    context = torch.zeros(rows, 1280)
    context[context_valid] = F.normalize(
        torch.randn(int(context_valid.sum()), 1280, generator=generator), dim=-1
    )
    full_scalar = normalization["median"][:18].repeat(rows, 1)
    statistics = torch.zeros(rows, 12)
    statistics[context_valid] = normalization["median"][18:].repeat(
        int(context_valid.sum()), 1
    )
    teachers = F.normalize(
        torch.randn(rows, 2, 1536, generator=generator), dim=-1
    )
    pair_rows = torch.arange(rows).repeat_interleave(2)
    return {
        "accepted_v2_e0": base,
        "raw_full_scalar_summary": full_scalar,
        "eligible": torch.ones(rows, dtype=torch.bool),
        "pooled_context_radio_direction": context,
        "typed_context_statistics": statistics,
        "typed_context_valid": context_valid,
        "official_multiview_siglip2_teacher_pair_region_indices": pair_rows,
        "official_multiview_siglip2_teacher_pair_descriptors": teachers.reshape(
            -1, 1536
        ),
        "region_row_ids": [f"region-{index}" for index in range(rows)],
        "scene_ids": ["scene0001_00"] * rows,
        "teacher_pair_view_ids": [f"view-{index}" for index in range(rows * 2)],
        "lineage": {"source_state_cohort_authority_sha256": SOURCE_SHA},
    }


def _validation_payload() -> dict:
    per_scene = {
        f"scene{index:04d}_00": {
            "base": {},
            "candidate": {},
            "candidate_minus_base": {},
        }
        for index in range(8)
    }
    return {
        "non_regression_passed": True,
        "validation_no_grad": True,
        "per_scene": per_scene,
    }


def test_train_only_normalizer_and_ood_envelope() -> None:
    authority = _normalization()
    frozen = validate_typed_context_normalization_authority(authority)
    original_digest = typed_context_normalization_authority_sha256(frozen)
    # Validation is not an input to the builder and cannot alter its authority.
    validation = torch.full((4, 30), 1e6)
    assert typed_context_normalization_authority_sha256(frozen) == original_digest
    result = apply_typed_context_normalization(validation, frozen)
    assert bool(result.ood_mask.all())
    assert frozen["contract"]["validation_contribution"] is False
    assert frozen["source_access"]["benchmark_queries_opened"] is False


def test_zero_step_identity_and_effective_active_routing() -> None:
    normalization = _normalization()
    scene = _scene(normalization)
    model = SurfaceRegionAcceptedV2TypedContextResidualV1(
        scalar_median=normalization["median"],
        scalar_robust_scale=normalization["robust_scale"],
    )
    declared, ood, active = trainer._routing(scene, normalization)
    output = model(
        scene["accepted_v2_e0"],
        scene["pooled_context_radio_direction"],
        scene["raw_full_scalar_summary"],
        scene["typed_context_statistics"],
        active_mask=declared,
        ood_mask=ood,
    )
    assert torch.equal(output, scene["accepted_v2_e0"])
    assert torch.equal(active, scene["eligible"] & scene["typed_context_valid"] & ~ood)


def test_objective_supports_single_and_batch_with_fixed_weights() -> None:
    generator = torch.Generator().manual_seed(3)
    for rows in (1, 5):
        semantic = F.normalize(torch.randn(rows, 1536, generator=generator), dim=-1)
        teacher = F.normalize(
            torch.randn(rows, 2, 1536, generator=generator), dim=-1
        )
        mask = torch.ones(rows, 2, dtype=torch.bool)
        statistics = torch.zeros(rows, 12)
        statistics[:, 7] = 0.5
        objective, metrics = trainer.typed_context_objective(
            semantic,
            teacher,
            mask,
            statistics,
            boundary_threshold=0.5,
        )
        assert objective.ndim == 0 and bool(torch.isfinite(objective))
        if rows == 1:
            assert float(metrics["relation_gram_smooth_l1_loss"]) == 0.0
            assert float(metrics["boundary_balanced_hard_negative_ranking_loss"]) == 0.0
    contract = trainer.training_contract()
    assert contract["objective"]["unit_direction_all_view_cosine_weight"] == 1.0
    assert contract["objective"]["teacher_relation_gram_smooth_l1_weight"] == 0.1
    assert contract["objective"]["boundary_balanced_hard_negative_ranking_weight"] == 0.25


def test_cpu_synthetic_streaming_epoch_updates_only_residual_model(monkeypatch) -> None:
    normalization = _normalization()
    scene = _scene(normalization)
    binding = trainer.SceneBinding(
        "source_train",
        "scene0001_00",
        {"path": "/shard.pt", "sha256": "1" * 64},
        {"path": "/context.pt", "sha256": "2" * 64},
    )
    monkeypatch.setattr(trainer, "load_scene", lambda _binding: scene)
    model = SurfaceRegionAcceptedV2TypedContextResidualV1(
        scalar_median=normalization["median"],
        scalar_robust_scale=normalization["robust_scale"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=trainer.LEARNING_RATE)
    base_before = scene["accepted_v2_e0"].clone()
    metrics = trainer.train_one_epoch(
        model,
        optimizer,
        [binding],
        normalization,
        torch.device("cpu"),
        epoch=1,
    )
    assert metrics["global_or_split_teacher_densification"] is False
    assert int(torch.count_nonzero(model.residual_projection.weight)) > 0
    assert torch.equal(scene["accepted_v2_e0"], base_before)


def test_sparse_teacher_gather_never_densifies_scene_or_cohort() -> None:
    normalization = _normalization()
    scene = _scene(normalization)
    teachers, mask = trainer.gather_sparse_teacher_batch(
        scene, torch.tensor([1, 4])
    )
    assert teachers.shape == (2, 2, 1536)
    assert mask.shape == (2, 2)
    assert trainer.training_contract()["streaming"][
        "global_or_split_teacher_densification"
    ] is False
    metadata = trainer._lightweight_scene_metadata(scene)
    assert "official_multiview_siglip2_teacher_pair_descriptors" not in metadata
    assert metadata["official_multiview_siglip2_teacher_pair_row_offsets"].shape == (7,)


def test_validation_is_no_grad_and_reports_scene_base_delta() -> None:
    normalization = _normalization()
    scene = _scene(normalization)
    model = SurfaceRegionAcceptedV2TypedContextResidualV1(
        scalar_median=normalization["median"],
        scalar_robust_scale=normalization["robust_scale"],
    )
    result = trainer.evaluate_scene(model, scene, normalization, torch.device("cpu"))
    assert result["validation_no_grad"] is True
    assert result["fallback_bitwise_accepted_v2_e0"] is True
    assert set(result["candidate_minus_base"]) == {
        "mean_all_view_cosine",
        "p05_row_mean_all_view_cosine",
        "relation_fidelity",
    }
    assert result["candidate_minus_base"]["mean_all_view_cosine"] == 0.0


def test_resume_state_roundtrip_and_tensor_hash_tamper() -> None:
    normalization = _normalization()
    model = SurfaceRegionAcceptedV2TypedContextResidualV1(
        scalar_median=normalization["median"],
        scalar_robust_scale=normalization["robust_scale"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=trainer.LEARNING_RATE)
    history = [{"epoch": 0, "validation": {"non_regression_passed": True}}]
    normalization_sha = typed_context_normalization_authority_sha256(normalization)
    state = trainer.build_resume_state(
        model=model,
        optimizer=optimizer,
        next_epoch=1,
        best_epoch=0,
        best_state=model.state_dict(),
        history=history,
        epochs_without_improvement=0,
        normalization_authority_sha256=normalization_sha,
        input_records_sha256="d" * 64,
    )
    restored = SurfaceRegionAcceptedV2TypedContextResidualV1(
        scalar_median=normalization["median"],
        scalar_robust_scale=normalization["robust_scale"],
    )
    restored.load_state_dict(state["model_state_dict"], strict=True)
    assert trainer._state_sha(restored) == state["model_state_dict_sha256"]
    tampered = copy.deepcopy(state)
    tampered["model_state_dict"]["residual_projection.bias"][0] = 1.0
    with pytest.raises(ValueError, match="model hash"):
        trainer.validate_resume_state(
            tampered,
            expected_normalization_authority_sha256=normalization_sha,
            expected_input_records_sha256="d" * 64,
        )


def test_checkpoint_reload_preserves_inactive_and_ood_bitwise_e0(tmp_path) -> None:
    normalization = _normalization()
    normalization_path = write_torch_noclobber(
        tmp_path / "normalization.pt", normalization
    )
    model = SurfaceRegionAcceptedV2TypedContextResidualV1(
        scalar_median=normalization["median"],
        scalar_robust_scale=normalization["robust_scale"],
    )
    with torch.no_grad():
        model.residual_projection.bias.copy_(torch.linspace(-0.2, 0.2, 1536))
    certificate = build_typed_context_training_certificate(
        training_contract=trainer.training_contract(),
        model=model,
        normalization_authority_file=file_record(normalization_path),
        cohort_authority={
            "file": {"path": "/cohort.json", "sha256": "1" * 64},
            "authority_sha256": "5" * 64,
        },
        external_manifests={
            "source_state": {"path": "/source.json", "sha256": "2" * 64},
            "teacher": {"path": "/teacher.json", "sha256": "3" * 64},
            "benchmark_exclusion": {"path": "/exclude.json", "sha256": "4" * 64},
        },
        input_records_by_split={
            "source_train": _input_records(24, prefix="train"),
            "source_validation": _input_records(8, prefix="validation"),
        },
        selected_epoch=1,
        selected_validation=_validation_payload(),
    )
    certificate_path = write_frozen_json(tmp_path / "certificate.json", certificate)
    checkpoint_path, checkpoint_sha = write_typed_context_checkpoint(
        tmp_path / "checkpoint.pt",
        model,
        normalization_authority=normalization,
        normalization_file_sha256=sha256_file(normalization_path),
        certificate=certificate,
        certificate_file_sha256=sha256_file(certificate_path),
    )
    _, loaded = load_typed_context_checkpoint(
        checkpoint_path,
        expected_checkpoint_sha256=checkpoint_sha,
        normalization_path=normalization_path,
        expected_normalization_sha256=sha256_file(normalization_path),
        certificate_path=certificate_path,
        expected_certificate_sha256=sha256_file(certificate_path),
    )
    scene = _scene(normalization)
    declared = scene["typed_context_valid"]
    ood = torch.tensor([False, False, False, True, False, False])
    output = loaded(
        scene["accepted_v2_e0"],
        scene["pooled_context_radio_direction"],
        scene["raw_full_scalar_summary"],
        scene["typed_context_statistics"],
        active_mask=declared,
        ood_mask=ood,
    )
    fallback = ~declared | ood
    assert torch.equal(output[fallback], scene["accepted_v2_e0"][fallback])
    with pytest.raises(ValueError, match="SHA-256 differs"):
        load_typed_context_checkpoint(
            checkpoint_path,
            expected_checkpoint_sha256="0" * 64,
            normalization_path=normalization_path,
            expected_normalization_sha256=sha256_file(normalization_path),
            certificate_path=certificate_path,
            expected_certificate_sha256=sha256_file(certificate_path),
        )
    with pytest.raises(FileExistsError):
        write_typed_context_checkpoint(
            checkpoint_path,
            model,
            normalization_authority=normalization,
            normalization_file_sha256=sha256_file(normalization_path),
            certificate=certificate,
            certificate_file_sha256=sha256_file(certificate_path),
        )
