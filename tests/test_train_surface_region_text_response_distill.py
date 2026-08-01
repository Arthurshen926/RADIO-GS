from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Sequence

import pytest
import torch
import torch.nn.functional as F

from radio_gs.interfaces.surface_region_summary import SurfaceRegionSummaryReadoutV2
from radio_gs.losses.direct_point_query_logit_distill_loss import (
    compute_independent_normalized_cosine_response_smooth_l1_loss,
    compute_scene_wise_text_response_profile_ranking_loss,
)
from radio_gs.scripts.build_target_blind_siglip2_embedding_artifact import (
    MODEL_ID,
    MODEL_REVISION,
    OUTPUT_DIMENSION,
    build_embedding_artifact,
)
from radio_gs.scripts.surface_text_response_distill_authority import (
    IMPLEMENTATION_SOURCES as AUTHORITY_IMPLEMENTATION_SOURCES,
)
import radio_gs.scripts.train_surface_region_text_response_distill as trainer_module
from radio_gs.scripts.train_surface_region_text_response_distill import (
    MAX_COMPLETE_SCENE_BATCH_ROWS,
    RESPONSE_BRANCH_GRADIENT_RATIO,
    SHARED_TRAINING_SEEDS,
    SURFACE_CONTROL_METRICS,
    SURFACE_CONTROL_NONINFERIORITY_TOLERANCE,
    _cache_binding,
    _cache_bound_metadata,
    _fit_bank_binding,
    _implementation_binding,
    _training_config,
    _training_provenance,
    audit_training_artifacts,
    build_parser,
    calibrate_gradient_budgets,
    complete_scene_batches,
    compute_query_free_response_selection_metrics,
    compute_training_losses,
    finalize_response_primary_epoch_selection,
    fixed_calibration_scene_batch,
    load_distill_run_manifest,
    load_fit_text_embedding_bank,
    load_surface_control_checkpoint,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
_TEST_SOURCE_SHA256 = {"mock_imagenet_source.txt": "a" * 64}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_sha(records: list[dict[str, str]], split: str) -> str:
    lines = "".join(
        f"{record['synset']}\t{record['query']}\n"
        for record in records
        if record["split"] == split
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def _write_vocabulary(root: Path) -> tuple[Path, Path, list[dict[str, str]]]:
    records = [
        {"synset": "n00000001", "query": "alpha object", "split": "fit"},
        {"synset": "n00000002", "query": "beta object", "split": "fit"},
        {"synset": "n00000003", "query": "gamma object", "split": "dev"},
        {"synset": "n00000004", "query": "delta object", "split": "audit"},
    ]
    vocabulary = {
        "schema_version": 1,
        "artifact_type": "target_blind_imagenet1k_primary_text_bank",
        "algorithm_version": "imagenet1k-primary-v1",
        "prompt_templates": ["{query}"],
        "benchmark_vocabulary_opened": False,
        "records": records,
    }
    vocabulary_path = root / "vocabulary.json"
    vocabulary_path.write_text(
        json.dumps(
            vocabulary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "target_blind_imagenet1k_primary_text_bank_manifest",
        "algorithm_version": "imagenet1k-primary-v1",
        "benchmark_vocabulary_opened": False,
        "counts": {
            "source_synsets": len(records),
            "deduplicated_queries": len(records),
            **{
                name: sum(record["split"] == name for record in records)
                for name in ("fit", "dev", "audit")
            },
        },
        "sources": {
            name: {"path": f"/mock/{name}", "sha256": digest}
            for name, digest in _TEST_SOURCE_SHA256.items()
        },
        "canonical_json": {
            "path": str(vocabulary_path),
            "sha256": _sha256(vocabulary_path),
        },
        "split_synset_tab_query_lf_sha256": {
            name: _split_sha(records, name) for name in ("fit", "dev", "audit")
        },
    }
    manifest_path = root / "vocabulary.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return vocabulary_path, manifest_path, records


def _write_snapshot(root: Path) -> Path:
    snapshot = root / "snapshots" / MODEL_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "model_type": "siglip",
                "text_config": {
                    "hidden_size": 8,
                    "projection_size": OUTPUT_DIMENSION,
                },
            }
        ),
        encoding="utf-8",
    )
    for name, contents in {
        "tokenizer.json": b"tokenizer-json",
        "tokenizer.model": b"tokenizer-model",
        "tokenizer_config.json": b"{}",
        "special_tokens_map.json": b"{}",
        "preprocessor_config.json": b"{}",
    }.items():
        (snapshot / name).write_bytes(contents)
    shard = "model-00001-of-00001.safetensors"
    (snapshot / shard).write_bytes(b"mock-local-weight-shard")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "text_model.head.weight": shard,
                    "text_model.head.bias": shard,
                    "text_model.embeddings.token_embedding.weight": shard,
                }
            }
        ),
        encoding="utf-8",
    )
    return snapshot


def _write_embedding_artifact(root: Path, split: str) -> tuple[Path, Path, dict, dict]:
    root.mkdir(parents=True, exist_ok=True)
    vocabulary, vocabulary_manifest, records = _write_vocabulary(root)
    snapshot = _write_snapshot(root)
    selected = [record for record in records if record["split"] == split]
    query_to_column = {
        record["query"]: index for index, record in enumerate(selected)
    }

    def encoder(queries: Sequence[str], _: Path) -> torch.Tensor:
        result = torch.zeros(len(queries), OUTPUT_DIMENSION, dtype=torch.float32)
        for row, query in enumerate(queries):
            result[row, query_to_column[query]] = 1.0
        return result

    output = root / f"{split}.pt"
    sidecar = root / f"{split}.manifest.json"
    split_hashes = {
        name: _split_sha(records, name) for name in ("fit", "dev", "audit")
    }
    snapshot_index = json.loads(
        (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    snapshot_files = {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        *(str(value) for value in snapshot_index["weight_map"].values()),
    }
    vocabulary_contract = {
        "canonical_vocabulary_sha256": _sha256(vocabulary),
        "counts": {
            "source_synsets": len(records),
            "deduplicated_queries": len(records),
            **{
                name: sum(record["split"] == name for record in records)
                for name in ("fit", "dev", "audit")
            },
        },
        "source_sha256": _TEST_SOURCE_SHA256,
        "split_sha256": split_hashes,
    }
    snapshot_contract = {
        name: _sha256(snapshot / name) for name in snapshot_files
    }
    build_embedding_artifact(
        vocabulary=vocabulary,
        vocabulary_manifest=vocabulary_manifest,
        split=split,
        snapshot=snapshot,
        output=output,
        sidecar_output=sidecar,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        batch_size=1,
        batch_encoder=encoder,
        _test_vocabulary_contract=vocabulary_contract,
        _test_snapshot_files_sha256=snapshot_contract,
    )
    return output, sidecar, vocabulary_contract, snapshot_contract


def _install_test_bank_contracts(
    monkeypatch: pytest.MonkeyPatch,
    vocabulary_contract: dict,
    snapshot_contract: dict,
) -> None:
    original_vocabulary_validator = trainer_module.bank_builder._validate_vocabulary
    original_snapshot_validator = trainer_module.bank_builder._validate_snapshot
    monkeypatch.setattr(
        trainer_module.bank_builder,
        "_validate_vocabulary",
        lambda vocabulary, manifest, split, **_: original_vocabulary_validator(
            vocabulary,
            manifest,
            split,
            frozen_contract=vocabulary_contract,
        ),
    )
    monkeypatch.setattr(
        trainer_module.bank_builder,
        "_validate_snapshot",
        lambda snapshot, *, model_id, revision, **_: original_snapshot_validator(
            snapshot,
            model_id=model_id,
            revision=revision,
            expected_files_sha256=snapshot_contract,
        ),
    )


def test_fit_loader_accepts_only_builder_bound_fit_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_path, fit_sidecar, vocabulary_contract, snapshot_contract = (
        _write_embedding_artifact(tmp_path / "fit", "fit")
    )
    _install_test_bank_contracts(
        monkeypatch,
        vocabulary_contract,
        snapshot_contract,
    )
    bank = load_fit_text_embedding_bank(fit_path, fit_sidecar)

    assert bank["query_count"] == 2
    assert bank["embeddings"].shape == (2, OUTPUT_DIMENSION)
    assert bank["embeddings"].dtype == torch.float32
    assert bank["file_sha256"] == _sha256(fit_path)

    dev_path, dev_sidecar, _, _ = _write_embedding_artifact(
        tmp_path / "dev", "dev"
    )
    with pytest.raises(ValueError, match="requires the target-blind fit split"):
        load_fit_text_embedding_bank(dev_path, dev_sidecar)


def test_fit_loader_rejects_forged_frozen_formal_producer_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_path, fit_sidecar, vocabulary_contract, snapshot_contract = (
        _write_embedding_artifact(tmp_path, "fit")
    )
    _install_test_bank_contracts(
        monkeypatch,
        vocabulary_contract,
        snapshot_contract,
    )
    sidecar = json.loads(fit_sidecar.read_text(encoding="utf-8"))
    sidecar["builder"] = dict(trainer_module.FROZEN_FORMAL_FIT_BANK_BUILDER)
    fit_sidecar.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="builder binding mismatch"):
        load_fit_text_embedding_bank(fit_path, fit_sidecar)


def test_fit_loader_rejects_benchmark_access_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_path, fit_sidecar, vocabulary_contract, snapshot_contract = (
        _write_embedding_artifact(tmp_path, "fit")
    )
    _install_test_bank_contracts(
        monkeypatch,
        vocabulary_contract,
        snapshot_contract,
    )
    payload = torch.load(fit_path, map_location="cpu")
    payload["benchmark_vocabulary_opened"] = True
    torch.save(payload, fit_path)

    with pytest.raises(ValueError, match="benchmark vocabulary was not opened"):
        load_fit_text_embedding_bank(fit_path, fit_sidecar)


def _tiny_response_inputs() -> tuple[
    torch.nn.Parameter,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(19)
    parameter = torch.nn.Parameter(torch.randn(4, 6))
    target = F.normalize(torch.randn(4, 6), dim=-1)
    alternate = F.normalize(torch.randn(4, 6), dim=-1)
    all_descriptors = torch.stack((target, alternate), dim=1)
    mask = torch.tensor(
        [[True, True], [True, False], [True, True], [False, True]]
    )
    text_bank = F.normalize(torch.randn(5, 6), dim=-1)
    return parameter, target, all_descriptors, mask, text_bank


def test_dual_gradient_budget_limits_each_branch_to_quarter_surface() -> None:
    parameter, target, all_descriptors, mask, text_bank = _tiny_response_inputs()
    projected = F.normalize(parameter, dim=-1)
    first = calibrate_gradient_budgets(
        parameter,
        projected,
        target,
        target,
        all_descriptors,
        mask,
        text_bank,
        ["scene-a", "scene-a", "scene-b", "scene-b"],
        (("descriptor", parameter),),
        token_weight=0.25,
        relation_weight=0.1,
    )
    surface = first["gradient_l2"]["surface"]
    for branch in ("independent_response", "scene_response"):
        assert first["weighted_response_gradient_l2"][branch] == pytest.approx(
            RESPONSE_BRANCH_GRADIENT_RATIO * surface,
            rel=1e-12,
        )
    assert first["combined_response_to_surface_upper_bound_ratio"] == pytest.approx(0.5)
    assert first["trainable_parameters"] == [
        {"name": "descriptor", "shape": [4, 6]}
    ]

    duplicate = torch.nn.Parameter(parameter.detach().clone())
    second = calibrate_gradient_budgets(
        duplicate,
        F.normalize(duplicate, dim=-1),
        target,
        target,
        all_descriptors,
        mask,
        text_bank,
        ["scene-a", "scene-a", "scene-b", "scene-b"],
        (("descriptor", duplicate),),
        token_weight=0.25,
        relation_weight=0.1,
    )
    assert second == first


def test_training_objective_adds_both_calibrated_response_branches() -> None:
    parameter, target, all_descriptors, mask, text_bank = _tiny_response_inputs()
    projected = F.normalize(parameter, dim=-1)
    predicted_token = F.normalize(torch.randn(4, 7), dim=-1).requires_grad_()
    target_token = F.normalize(torch.randn(4, 7), dim=-1)
    independent_lambda = 0.375
    scene_lambda = 0.025
    scene_ids = ["scene-a", "scene-a", "scene-b", "scene-b"]
    losses = compute_training_losses(
        predicted_token,
        projected,
        target_token,
        target,
        all_descriptors,
        mask,
        text_bank,
        scene_ids,
        token_weight=0.25,
        relation_weight=0.1,
        independent_response_lambda=independent_lambda,
        scene_response_lambda=scene_lambda,
    )
    expected_scene, _ = compute_scene_wise_text_response_profile_ranking_loss(
        projected,
        target,
        text_bank,
        scene_ids,
    )
    expected_response = (
        compute_independent_normalized_cosine_response_smooth_l1_loss(
            projected,
            target,
            text_bank,
        )
    )
    expected_total = (
        0.25 * losses["token"]
        + losses["descriptor"]
        + 0.1 * losses["relation"]
        + independent_lambda * expected_response
        + scene_lambda * expected_scene
    )

    torch.testing.assert_close(losses["independent_response"], expected_response)
    torch.testing.assert_close(losses["scene_response"], expected_scene)
    torch.testing.assert_close(losses["total"], expected_total)
    losses["total"].backward()
    assert parameter.grad is not None
    assert predicted_token.grad is not None


def test_complete_scene_batches_never_split_or_drop_rows() -> None:
    scene_ids = [
        *(["scene-d"] * 3),
        *(["scene-a"] * 2),
        *(["scene-c"] * 4),
        *(["scene-b"] * 3),
    ]
    scenes, fixed = fixed_calibration_scene_batch(
        scene_ids, row_count=len(scene_ids)
    )
    assert scenes == ["scene-a", "scene-b", "scene-c", "scene-d"]
    assert len(fixed) == len(scene_ids)
    batches = complete_scene_batches(
        scene_ids,
        row_count=len(scene_ids),
        target_batch_rows=5,
        generator=torch.Generator().manual_seed(7),
    )
    assert max(map(len, batches)) <= MAX_COMPLETE_SCENE_BATCH_ROWS
    assert sorted(row for batch in batches for row in batch.tolist()) == list(
        range(len(scene_ids))
    )
    row_to_batch = {
        row: batch_index
        for batch_index, batch in enumerate(batches)
        for row in batch.tolist()
    }
    for scene in set(scene_ids):
        assert len({row_to_batch[i] for i, value in enumerate(scene_ids) if value == scene}) == 1


def test_response_primary_selection_rejects_high_cosine_support_flip() -> None:
    teacher = F.normalize(
        torch.tensor(
            [
                [1.00, 0.10, 0.0],
                [0.99, 0.14, 0.0],
                [0.20, 0.90, 0.0],
            ]
        ),
        dim=-1,
    )
    text = torch.tensor([[1.0, 0.0, 0.0]])
    # Swapping the two close descriptors keeps row-local cosine very high but
    # moves the text unary peak to the wrong region.
    high_cosine_flip = teacher[[1, 0, 2]]
    # A shared orthogonal component lowers row cosine while preserving every
    # response ordering because the text has no component on that axis.
    rank_preserving = F.normalize(
        teacher + torch.tensor([0.0, 0.0, 0.30]),
        dim=-1,
    )
    flipped = compute_query_free_response_selection_metrics(
        high_cosine_flip,
        teacher,
        text,
        scene_ids=["scene-a"] * 3,
    )
    preserved = compute_query_free_response_selection_metrics(
        rank_preserving,
        teacher,
        text,
        scene_ids=["scene-a"] * 3,
    )
    assert F.cosine_similarity(high_cosine_flip, teacher).mean() > (
        F.cosine_similarity(rank_preserving, teacher).mean()
    )
    assert flipped["text_support_top1_agreement"] == 0.0
    assert preserved["text_support_top1_agreement"] == 1.0

    history, best_epoch, best_score = finalize_response_primary_epoch_selection(
        [
            {
                "epoch": 0,
                "surface_selection_score": 0.94,
                "summary_token_cosine": 0.94,
                "mean_descriptor_cosine": 0.94,
                "all_view_descriptor_cosine": 0.94,
                **flipped,
            },
            {
                "epoch": 1,
                "surface_selection_score": 0.99,
                "summary_token_cosine": 0.99,
                "mean_descriptor_cosine": 0.99,
                "all_view_descriptor_cosine": 0.99,
                **flipped,
            },
            {
                "epoch": 2,
                "surface_selection_score": 0.95,
                "summary_token_cosine": 0.95,
                "mean_descriptor_cosine": 0.95,
                "all_view_descriptor_cosine": 0.95,
                **preserved,
            },
        ]
    )
    assert history[0]["selection_score"] == -1.0
    assert history[1]["selection_score"] == -1.0
    assert best_epoch == 2
    assert best_score == pytest.approx(0.95)


def _selection_row(
    epoch: int,
    *,
    surface: tuple[float, float, float],
    support: float,
    response_error: float,
    relation_error: float,
) -> dict[str, object]:
    summary, mean_descriptor, all_view = surface
    return {
        "epoch": epoch,
        "surface_selection_score": 0.5 * (mean_descriptor + all_view),
        "summary_token_cosine": summary,
        "mean_descriptor_cosine": mean_descriptor,
        "all_view_descriptor_cosine": all_view,
        "text_support_top1_agreement": support,
        "text_response_smooth_l1": response_error,
        "descriptor_relation_smooth_l1": relation_error,
    }


def test_surface_control_feasible_set_precedes_fit_text_selection() -> None:
    rows, best_epoch, _ = finalize_response_primary_epoch_selection(
        [
            _selection_row(
                0,
                surface=(0.90, 0.91, 0.92),
                support=0.20,
                response_error=0.08,
                relation_error=0.04,
            ),
            # Highest text support, but summary is 0.0021 below control.
            _selection_row(
                1,
                surface=(0.8979, 0.93, 0.94),
                support=0.99,
                response_error=0.01,
                relation_error=0.01,
            ),
            # Every Surface component is exactly on/inside the -0.002 bound.
            _selection_row(
                2,
                surface=(0.898, 0.908, 0.918),
                support=0.80,
                response_error=0.02,
                relation_error=0.02,
            ),
        ]
    )
    assert rows[1]["surface_control_feasible"] is False
    assert rows[2]["surface_control_feasible"] is True
    assert rows[2]["surface_control_tolerance"] == 0.002
    assert best_epoch == 2


def test_surface_control_epoch_zero_is_selected_when_training_is_infeasible() -> None:
    rows, best_epoch, best_score = finalize_response_primary_epoch_selection(
        [
            _selection_row(
                0,
                surface=(0.90, 0.91, 0.92),
                support=0.20,
                response_error=0.08,
                relation_error=0.04,
            ),
            _selection_row(
                1,
                surface=(0.89, 0.95, 0.95),
                support=1.0,
                response_error=0.0,
                relation_error=0.0,
            ),
        ]
    )
    assert best_epoch == 0
    assert best_score == pytest.approx(0.915)
    assert rows[0]["selection_score"] == pytest.approx(0.915)
    assert rows[1]["selection_score"] == -1.0


def _dummy_fit_bank(root: Path) -> dict[str, object]:
    artifact = root / "fit.pt"
    manifest = root / "fit.manifest.json"
    artifact.write_bytes(b"fit-bank")
    manifest.write_text("{}", encoding="utf-8")
    return {
        "path": artifact.resolve(),
        "file_sha256": _sha256(artifact),
        "manifest_path": manifest.resolve(),
        "manifest_sha256": _sha256(manifest),
        "query_count": 806,
        "split_sha256": "1" * 64,
        "ordered_records_sha256": "2" * 64,
        "vocabulary_sha256": "3" * 64,
        "vocabulary_manifest_sha256": "4" * 64,
        "embedding_semantic_sha256": "5" * 64,
        "embedding_tensor_sha256": "6" * 64,
        "text_encoder": {"snapshot_files_sha256": "7" * 64},
    }


def _write_surface_control_fixture(
    root: Path,
    *,
    seed: int = 0,
) -> tuple[Path, list[Path], list[Path], dict[str, object], dict[str, object]]:
    train_paths = [root / "train-cache.pt"]
    validation_paths = [root / "validation-cache.pt"]
    train_paths[0].write_bytes(b"train-cache")
    validation_paths[0].write_bytes(b"validation-cache")
    common: dict[str, object] = {
        "split_hashes": ["b" * 64],
        "region_contract_sha256": "c" * 64,
        "region_contract": {"name": "fixed-test-contract"},
        "teacher_region": {"semantics": "fixed-core"},
        "radio_checkpoint_sha256": "d" * 64,
        "excluded_physical_spaces": ["heldout"],
        "exclusion_files": [],
        "physical_space_disjoint": True,
    }
    train_meta = {
        **common,
        "scenes": ["scene-train"],
        "cache_paths": [str(path.resolve()) for path in train_paths],
    }
    validation_meta = {
        **common,
        "scenes": ["scene-validation"],
        "cache_paths": [str(path.resolve()) for path in validation_paths],
    }
    model = SurfaceRegionSummaryReadoutV2(hidden_dim=4)
    payload = {
        "schema_version": 3,
        "architecture": model.architecture("c" * 64),
        "state_dict": model.state_dict(),
        "provenance": {
            "frozen": True,
            "uses_benchmark_scenes": False,
            "uses_benchmark_test_vocabulary": False,
            "train": train_meta,
            "validation": validation_meta,
            "random_seed_contract": {
                "seed": seed,
                "model_initialization": True,
                "data_order": True,
                "canonical_noise": True,
            },
        },
        "history": [{"epoch": 1}],
        "best_epoch": 1,
        "best_selection_score": 0.9,
        "untrained_baseline": {},
        "untrained_baseline_score": 0.0,
        "training_config": {
            "seed": seed,
            "hidden_dim": 4,
            "reliability_attention_mode": "log_prior",
            "context_pooling_mode": "joint_attention_v1",
        },
    }
    checkpoint = root / f"surface-control-seed{seed}.pt"
    torch.save(payload, checkpoint)
    return checkpoint, train_paths, validation_paths, train_meta, validation_meta


def test_surface_control_loader_binds_sha_seed_architecture_and_caches(
    tmp_path: Path,
) -> None:
    checkpoint, train_paths, validation_paths, train_meta, validation_meta = (
        _write_surface_control_fixture(tmp_path)
    )
    model, binding = load_surface_control_checkpoint(
        checkpoint,
        expected_sha256=_sha256(checkpoint),
        seed=0,
        train_paths=train_paths,
        validation_paths=validation_paths,
        train_meta=train_meta,
        validation_meta=validation_meta,
        hidden_dim=4,
        reliability_attention_mode="log_prior",
        context_pooling_mode="joint_attention_v1",
    )
    assert binding["sha256"] == _sha256(checkpoint)
    assert binding["seed"] == 0
    assert binding["train_caches"] == _cache_binding(train_paths)
    assert set(model.state_dict())

    with pytest.raises(ValueError, match="SHA-256 differs"):
        load_surface_control_checkpoint(
            checkpoint,
            expected_sha256="0" * 64,
            seed=0,
            train_paths=train_paths,
            validation_paths=validation_paths,
            train_meta=train_meta,
            validation_meta=validation_meta,
            hidden_dim=4,
            reliability_attention_mode="log_prior",
            context_pooling_mode="joint_attention_v1",
        )
    with pytest.raises(ValueError, match="seed/training provenance"):
        load_surface_control_checkpoint(
            checkpoint,
            expected_sha256=_sha256(checkpoint),
            seed=1,
            train_paths=train_paths,
            validation_paths=validation_paths,
            train_meta=train_meta,
            validation_meta=validation_meta,
            hidden_dim=4,
            reliability_attention_mode="log_prior",
            context_pooling_mode="joint_attention_v1",
        )
    with pytest.raises(ValueError, match="architecture differs"):
        load_surface_control_checkpoint(
            checkpoint,
            expected_sha256=_sha256(checkpoint),
            seed=0,
            train_paths=train_paths,
            validation_paths=validation_paths,
            train_meta=train_meta,
            validation_meta=validation_meta,
            hidden_dim=8,
            reliability_attention_mode="log_prior",
            context_pooling_mode="joint_attention_v1",
        )
    drifted_validation = dict(validation_meta)
    drifted_validation["scenes"] = ["another-scene"]
    with pytest.raises(ValueError, match="validation-cache provenance"):
        load_surface_control_checkpoint(
            checkpoint,
            expected_sha256=_sha256(checkpoint),
            seed=0,
            train_paths=train_paths,
            validation_paths=validation_paths,
            train_meta=train_meta,
            validation_meta=drifted_validation,
            hidden_dim=4,
            reliability_attention_mode="log_prior",
            context_pooling_mode="joint_attention_v1",
        )


def _write_distill_run_manifest_fixture(
    root: Path,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    train_path = root / "train.pt"
    validation_path = root / "validation.pt"
    radio_path = root / "radio.pt"
    train_path.write_bytes(b"train-cache")
    validation_path.write_bytes(b"validation-cache")
    radio_path.write_bytes(b"radio-checkpoint")
    fit_bank = _dummy_fit_bank(root)
    outputs = []
    output_paths: dict[int, Path] = {}
    calibrations = []
    calibration_paths: dict[int, Path] = {}
    for seed in SHARED_TRAINING_SEEDS:
        calibration_path = root / f"calibration_seed{seed}.json"
        calibration_path.write_text("{}", encoding="utf-8")
        calibration_paths[seed] = calibration_path
        calibrations.append(
            {
                "seed": seed,
                "manifest": {
                    "path": str(calibration_path.resolve()),
                    "sha256": _sha256(calibration_path),
                },
                "audit": {"path": str(root / f"audit{seed}.json"), "sha256": "a" * 64},
            }
        )
        output = root / f"seed_{seed}.pt"
        output_paths[seed] = output
        outputs.append(
            {
                "seed": seed,
                "checkpoint": str(output.resolve()),
                "report": str(
                    output.resolve().with_suffix(output.suffix + ".json")
                ),
            }
        )
    implementation_sources = {
        relative: _sha256(REPO_ROOT / relative)
        for relative in AUTHORITY_IMPLEMENTATION_SOURCES
    }
    training_contract = {"frozen_training_contract": "test-v1"}
    payload: dict[str, object] = {
        "schema_version": trainer_module.DISTILL_RUN_MANIFEST_SCHEMA_VERSION,
        "artifact_type": trainer_module.DISTILL_RUN_MANIFEST_ARTIFACT_TYPE,
        "authority_status": "query_free_three_seed_gpu1_run_frozen",
        "candidate": "context_c1024_geometric",
        "train_caches": _cache_binding([train_path]),
        "validation_caches": _cache_binding([validation_path]),
        "fit_text_bank": {
            "artifact": {
                "path": str(fit_bank["path"]),
                "sha256": str(fit_bank["file_sha256"]),
            },
            "manifest": {
                "path": str(fit_bank["manifest_path"]),
                "sha256": str(fit_bank["manifest_sha256"]),
            },
        },
        "radio_checkpoint": {
            "path": str(radio_path.resolve()),
            "sha256": _sha256(radio_path),
        },
        "calibrations": calibrations,
        "training_contract": training_contract,
        "outputs": outputs,
        "implementation_sources": implementation_sources,
        "runtime_closure": {"digest": "a" * 64},
        "authority_contract": {
            "seed_resume": "skip_only_exact_guarded_terminal_v1"
        },
    }
    manifest_path = root / "run_manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    arguments: dict[str, object] = {
        "train_paths": [train_path],
        "validation_paths": [validation_path],
        "fit_bank": fit_bank,
        "radio_path": radio_path,
        "calibration_path": calibration_paths[0],
        "output_path": output_paths[0],
        "seed": 0,
        "training_contract": training_contract,
    }
    return manifest_path, payload, arguments


def test_distill_manifest_accepts_authority_implementation_source_superset(
    tmp_path: Path,
) -> None:
    manifest_path, payload, arguments = _write_distill_run_manifest_fixture(
        tmp_path
    )
    implementation = payload["implementation_sources"]
    assert isinstance(implementation, dict)
    assert {
        "radio_gs/scripts/surface_attention_pooling_screen.py",
        "radio_gs/scripts/surface_gpu1_lock_supervisor.py",
    }.issubset(implementation)

    loaded = load_distill_run_manifest(manifest_path, **arguments)

    assert loaded["candidate"] == "context_c1024_geometric"
    assert loaded["sha256"] == _sha256(manifest_path)


def test_distill_manifest_rejects_missing_trainer_required_source(
    tmp_path: Path,
) -> None:
    manifest_path, payload, arguments = _write_distill_run_manifest_fixture(
        tmp_path
    )
    implementation = payload["implementation_sources"]
    assert isinstance(implementation, dict)
    implementation.pop(
        "radio_gs/scripts/train_surface_region_text_response_distill.py"
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="implementation source set differs"):
        load_distill_run_manifest(manifest_path, **arguments)


def test_distill_manifest_rejects_trainer_required_source_sha_drift(
    tmp_path: Path,
) -> None:
    manifest_path, payload, arguments = _write_distill_run_manifest_fixture(
        tmp_path
    )
    relative = "radio_gs/scripts/train_surface_region_text_response_distill.py"
    implementation = payload["implementation_sources"]
    assert isinstance(implementation, dict)
    implementation[relative] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=f"distill implementation source changed: {relative}",
    ):
        load_distill_run_manifest(manifest_path, **arguments)


def test_cli_exposes_no_tunable_lambda_or_calibration_batch() -> None:
    parser = build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    calibrate_destinations = {
        action.dest for action in subparsers.choices["calibrate"]._actions
    }
    train_destinations = {
        action.dest for action in subparsers.choices["train"]._actions
    }

    assert "response_lambda" not in train_destinations
    assert "run_manifest" in train_destinations
    assert {
        "surface_control_checkpoint",
        "surface_control_checkpoint_sha256",
    }.issubset(train_destinations)
    assert {"audit-calibration", "audit-checkpoint"}.issubset(subparsers.choices)
    assert "calibration_batch_size" not in calibrate_destinations
    seed_action = next(
        action
        for action in subparsers.choices["train"]._actions
        if action.dest == "seed"
    )
    assert tuple(seed_action.choices) == SHARED_TRAINING_SEEDS


def test_checkpoint_remains_v2_inference_compatible(tmp_path: Path) -> None:
    model = SurfaceRegionSummaryReadoutV2(feature_dim=8, hidden_dim=4)
    architecture = model.architecture("c" * 64)
    checkpoint = tmp_path / "readout.pt"
    torch.save(
        {
            "schema_version": 3,
            "architecture": architecture,
            "state_dict": model.state_dict(),
            "provenance": {
                "text_response_distillation": {"fit_split_only": True}
            },
        },
        checkpoint,
    )

    restored, payload = SurfaceRegionSummaryReadoutV2.from_checkpoint(checkpoint)
    assert payload["architecture"] == architecture
    assert set(restored.state_dict()) == set(model.state_dict())
    assert not any("text" in name or "response" in name for name in restored.state_dict())
    output = restored(
        torch.randn(2, 3, 8),
        torch.randn(2, 3, 14),
        anchor_index=torch.tensor([0, 1]),
        token_mask=torch.ones(2, 3, dtype=torch.bool),
        reliability=torch.ones(2, 3, 1),
    )
    assert output.shape == (2, 8)


def test_cache_bound_metadata_adds_exact_hashes_and_rejects_path_drift(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "validation.pt"
    cache.write_bytes(b"immutable-validation-cache")
    metadata = {"cache_paths": [str(cache.resolve())], "scenes": ["scene-a"]}

    bound = _cache_bound_metadata(metadata, [cache])
    assert bound["cache_bindings"] == [
        {"path": str(cache.resolve()), "sha256": _sha256(cache)}
    ]
    with pytest.raises(ValueError, match="paths differ"):
        _cache_bound_metadata(
            {"cache_paths": [str((tmp_path / "other.pt").resolve())]},
            [cache],
        )


def test_companion_runner_is_gpu1_thermal_guarded_and_query_blind() -> None:
    runner = (
        REPO_ROOT
        / "radio_gs"
        / "scripts"
        / "run_surface_region_text_response_distill.sh"
    )
    subprocess.run(["bash", "-n", str(runner)], check=True)
    source = runner.read_text(encoding="utf-8")

    assert 'if [[ "$GPU" != "1" ]]' in source
    assert 'for seed in 0 1 2' in source
    assert "run_with_gpu_thermal_guard.sh" in source
    assert 'GPU_MAX_TEMP_C="${GPU_MAX_TEMP_C:-78}"' in source
    assert 'GPU_START_MAX_TEMP_C="${GPU_START_MAX_TEMP_C:-65}"' in source
    assert 'GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-3}"' in source
    assert 'GPU_SOFT_PAUSE_TEMP_C="${GPU_SOFT_PAUSE_TEMP_C:-75}"' in source
    assert 'GPU_SOFT_RESUME_TEMP_C="${GPU_SOFT_RESUME_TEMP_C:-70}"' in source
    assert 'GPU_PEER_INDEX=""' in source
    assert "GPU_PEER_PAUSE_TEMP_C=0" in source
    assert "GPU_PEER_RESUME_TEMP_C=0" in source
    assert 'GPU_PEER_QUIET_SECONDS="${GPU_PEER_QUIET_SECONDS:-0}"' in source
    assert "GPU_PEER_MAX_POWER_W=0" in source
    assert "GPU_PEER_MAX_MEMORY_MIB=0" in source
    assert "GPU_PEER_MAX_UTIL_PCT=100" in source
    assert 'GPU_MAX_POWER_LIMIT_W="${GPU_MAX_POWER_LIMIT_W:-300.5}"' in source
    assert 'RUN_MANIFEST="$OUTPUT_ROOT/run_manifest.json"' in source
    assert "canonical_seed_calibration_manifest()" in source
    assert (
        source.count(
            'calibration_manifest="$(canonical_seed_calibration_manifest "$seed")"'
        )
        == 2
    )
    assert (
        '--calibration-manifest '
        '"$seed=$(canonical_seed_calibration_manifest "$seed")"'
        in source
    )
    authority_source = (
        REPO_ROOT
        / "radio_gs/scripts/surface_text_response_distill_authority.py"
    ).read_text(encoding="utf-8")
    assert "surface_region_text_response_distill_run" in authority_source
    assert "audit-calibration" in source
    assert "audit-checkpoint" in source
    assert 'LOCK_ROOT="/root/RADIO-GS/output"' in source
    assert "surface_text_response_distill_authority.py" in source
    assert "prepare-command" in source and "finalize-seed" in source
    assert "journalctl -k" in source
    assert "distill candidate is not the frozen query-free selection" in authority_source
    assert "surface_region_text_response_distill_completion" in authority_source
    assert "date -Iseconds" not in source
    assert "--response-lambda" not in source
    assert "PFIR" not in source
    assert "benchmark" not in source.lower()


def test_runner_calibrate_arguments_are_accepted_by_calibrate_parser() -> None:
    runner = (
        REPO_ROOT
        / "radio_gs"
        / "scripts"
        / "run_surface_region_text_response_distill.sh"
    )
    source = runner.read_text(encoding="utf-8")
    command_start = source.index("common_calibration_args=(")
    command_end = source.index("\ndone", command_start)
    runner_options = set(
        re.findall(r"--[a-z][a-z0-9-]*", source[command_start:command_end])
    )
    runner_options.discard("--calibration-manifest")  # audit-only in this block

    parser = trainer_module.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    calibrate_parser = subparsers.choices["calibrate"]
    parser_options = {
        option
        for action in calibrate_parser._actions
        for option in action.option_strings
    }

    assert "--radio-checkpoint" in runner_options
    assert runner_options <= parser_options
    parsed = parser.parse_args(
        [
            "calibrate",
            "--train-caches",
            "/immutable/train_shard*.pt",
            "--validation-caches",
            "/immutable/validation_shard*.pt",
            "--fit-text-bank",
            "/immutable/fit.pt",
            "--fit-text-bank-manifest",
            "/immutable/fit.pt.json",
            "--surface-control-checkpoint",
            "/immutable/surface-seed0.pt",
            "--surface-control-checkpoint-sha256",
            "a" * 64,
            "--gradient-diagnostic",
            "/immutable/diagnostic.json",
            "--gradient-diagnostic-sha256",
            "b" * 64,
            "--hidden-dim",
            "256",
            "--reliability-attention-mode",
            "log_prior",
            "--context-pooling-mode",
            "joint_attention_v1",
            "--radio-checkpoint",
            "/immutable/radio.pt",
            "--output",
            "/output/calibration.json",
            "--token-weight",
            "0.25",
            "--relation-weight",
            "0.1",
            "--seed",
            "0",
            "--device",
            "cpu",
        ]
    )
    assert parsed.command == "calibrate"
    assert parsed.radio_checkpoint == "/immutable/radio.pt"
