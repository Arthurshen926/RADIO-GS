from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v21_pilot as trainer,
)
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as pilot_shard
from radio_gs.scripts import train_surface_region_full_scalar_residual as base_trainer
from radio_gs.scripts import build_surface_region_v21_pilot_execution_authority as builder
from radio_gs.utils.immutable_artifacts import (
    file_record,
    sha256_file,
    write_torch_noclobber,
)


SHA = "a" * 64


def _record() -> dict[str, str]:
    return {"path": "/frozen", "sha256": SHA}


def _scene(scene_id: str) -> dict:
    return {
        "scene_id": scene_id,
        "training_shard": _record(),
        "adaptive_context": _record(),
        "hard_negative_authority": _record(),
        "hard_negative_content_authority_sha256": SHA,
    }


def _authority() -> dict:
    banks = {
        name: {**_record(), "loss_weight": weight}
        for name, weight in trainer.COMPONENT_WEIGHTS.items()
    }
    return {
        "schema": trainer.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_source_only_4train_2validation_v21_pilot",
        "implementation": _record(),
        "active_pair_addendum": _record(),
        "cohort_authority": _record(),
        "pilot_cohort_region_view_registry": _record(),
        "benchmark_exclusion_manifest": _record(),
        "fit_text_bank": _record(),
        "canonical_negative_bank": _record(),
        "compositional_banks": banks,
        "typed_relation_authority": {
            **_record(),
            "content_authority_sha256": SHA,
        },
        "source_train": [_scene(scene) for scene in trainer.TRAIN_SCENES],
        "source_validation": [_scene(scene) for scene in trainer.VALIDATION_SCENES],
        "training_authorized": True,
        "benchmark_execution_authorized": False,
        "source_access": trainer.source_access(),
    }


def test_execution_authority_fixes_cohort_weights_and_typed_relations() -> None:
    frozen = trainer.validate_execution_authority(_authority())
    assert tuple(row["scene_id"] for row in frozen["source_train"]) == (
        trainer.TRAIN_SCENES
    )
    assert tuple(row["scene_id"] for row in frozen["source_validation"]) == (
        trainer.VALIDATION_SCENES
    )
    assert {
        name: row["loss_weight"] for name, row in frozen["compositional_banks"].items()
    } == trainer.COMPONENT_WEIGHTS


@pytest.mark.parametrize("tamper", ["cohort", "weight", "relation", "benchmark"])
def test_execution_authority_fails_closed(tamper: str) -> None:
    value = deepcopy(_authority())
    if tamper == "cohort":
        value["source_validation"][0]["scene_id"] = "scene0013_00"
    elif tamper == "weight":
        value["compositional_banks"]["high_precision_part_of"]["loss_weight"] = 0.20
    elif tamper == "relation":
        value["typed_relation_authority"].pop("content_authority_sha256")
    else:
        value["benchmark_execution_authorized"] = True
    with pytest.raises(ValueError):
        trainer.validate_execution_authority(value)


def test_contract_and_synthetic_dry_run_are_exact() -> None:
    contract = trainer.training_contract()
    assert contract["v21"]["teacher_multiview_temperature"] == 0.1
    assert contract["v21"]["component_weights"] == {
        "object_noun_primary": 0.25,
        **trainer.COMPONENT_WEIGHTS,
    }
    assert contract["selection"]["split"] == "source_validation"
    assert contract["selection"]["benchmark_read"] is False
    assert contract["input_authority"] == {
        "pilot_training_shard_schema": pilot_shard.PILOT_TRAINING_SHARD_SCHEMA,
        "pilot_training_shard_contract_sha256": (
            pilot_shard.PILOT_TRAINING_SHARD_CONTRACT_SHA256
        ),
        "pilot_cohort_region_view_registry_schema": (
            pilot_shard.PILOT_COHORT_REGISTRY_SCHEMA
        ),
        "pilot_cohort_region_view_registry_contract_sha256": (
            trainer.canonical_json_sha256(
                pilot_shard.pilot_cohort_registry_contract()
            )
        ),
        "one_registry_file_and_content_authority_for_all_six_shards": True,
        "legacy_24plus8_shard_or_registry_schema_accepted": False,
    }
    dry = trainer.synthetic_dry_run()
    assert dry["training_objective_pairs"] == 2
    assert dry["both_immutable_pairs_excluded"] == 2
    assert dry["validation_objective_pairs"] == 4
    assert dry["selected_epoch"] == 1
    assert dry["benchmark_opened"] is False


def test_cli_exposes_only_source_authority_workflows() -> None:
    parser = trainer.build_parser()
    assert parser.parse_args(["synthetic-dry-run"]).command == "synthetic-dry-run"
    actions = parser._subparsers._group_actions[0].choices
    assert set(actions) == {"synthetic-dry-run", "validate-authority", "train"}


def test_pilot_routing_uses_train4_envelope_without_validation_fit() -> None:
    normalization = {
        "median": torch.zeros(30),
        "robust_scale": torch.ones(30),
        "constant_coordinate_mask": torch.zeros(30, dtype=torch.bool),
        "source_max_robust_linf": 2.0,
    }
    scene = {
        "raw_full_scalar_summary": torch.zeros(3, 18),
        "typed_context_statistics": torch.zeros(3, 12),
        "eligible": torch.tensor([True, True, False]),
        "typed_context_valid": torch.tensor([True, True, True]),
    }
    scene["typed_context_statistics"][1, 0] = 3.0
    declared, ood, active = trainer._pilot_routing(scene, normalization)
    assert declared.tolist() == [True, True, True]
    assert ood.tolist() == [False, True, True]
    assert active.tolist() == [True, False, False]


def _pilot_training_shard() -> dict:
    rows = 2
    base = F.normalize(torch.eye(rows, base_trainer.DESCRIPTOR_DIM), dim=-1)
    teachers = base.clone()
    pair_rows = torch.tensor([0, 1], dtype=torch.long)
    selection = {
        "sampling_contract_sha256": base_trainer.SAMPLING_CONTRACT_SHA256,
        "canonical_candidate_region_count": rows,
        "exact_overlap_candidate_count": rows,
        "teacher_visible_candidate_count": rows,
        "selected_region_count": rows,
        "selected_count_by_scale": [rows],
    }
    payload = {
        "schema": pilot_shard.PILOT_TRAINING_SHARD_SCHEMA,
        "schema_version": pilot_shard.SCHEMA_VERSION,
        "contract": pilot_shard.pilot_training_shard_contract(),
        "contract_sha256": pilot_shard.PILOT_TRAINING_SHARD_CONTRACT_SHA256,
        "split": "source_validation",
        "accepted_v2_e0": base.float(),
        "raw_full_scalar_summary": torch.zeros(rows, 18, dtype=torch.float32),
        "eligible": torch.ones(rows, dtype=torch.bool),
        "official_multiview_siglip2_teacher_pair_region_indices": pair_rows,
        "official_multiview_siglip2_teacher_pair_descriptors": teachers.float(),
        "scene_ids": ["scene0004_00"] * rows,
        "region_row_ids": ["scene0004_00:region:0", "scene0004_00:region:1"],
        "teacher_pair_view_ids": ["scene0004_00:view:0", "scene0004_00:view:1"],
        "sampling_audit": {
            "scene_id": "scene0004_00",
            "sampling_contract_sha256": base_trainer.SAMPLING_CONTRACT_SHA256,
            "canonical_region_indices_sha256": "a" * 64,
            "accepted_selection_audit": selection,
            "selected_region_count": rows,
            "pair_count": rows,
            "maximum_views_per_region": 1,
        },
        "lineage": {
            "accepted_v2_authority": base_trainer._accepted_v2_authority(),
            "source_state_cohort_authority_sha256": "b" * 64,
            "source_state_manifest_file_sha256": "c" * 64,
            "cohort_authority_sha256": "d" * 64,
            "cohort_authority_file_sha256": "e" * 64,
            "teacher_authority_sha256": "f" * 64,
            "teacher_manifest_file_sha256": "1" * 64,
            "pilot_cohort_region_view_registry_authority_sha256": "2" * 64,
            "pilot_cohort_region_view_registry_file_sha256": "3" * 64,
        },
        "source_access": base_trainer._source_access("source_validation"),
    }
    payload["channel_sha256"] = base_trainer.training_shard_channel_sha256(payload)
    return payload


def test_pilot_shard_loader_accepts_only_independent_schema(tmp_path) -> None:
    pilot = _pilot_training_shard()
    path = tmp_path / "pilot.pt"
    write_torch_noclobber(path, pilot)
    loaded, record = trainer.load_pilot_training_shard(
        path,
        expected_sha256=sha256_file(path),
        expected_split="source_validation",
    )
    assert loaded["schema"] == pilot_shard.PILOT_TRAINING_SHARD_SCHEMA
    assert record["sha256"] == sha256_file(path)

    legacy = deepcopy(pilot)
    legacy["schema"] = base_trainer.TRAINING_SHARD_SCHEMA
    legacy["schema_version"] = base_trainer.TRAINING_SHARD_SCHEMA_VERSION
    legacy["contract"] = base_trainer.training_shard_contract()
    legacy["contract_sha256"] = base_trainer.TRAINING_SHARD_CONTRACT_SHA256
    legacy["lineage"].pop(
        "pilot_cohort_region_view_registry_authority_sha256"
    )
    legacy["lineage"].pop("pilot_cohort_region_view_registry_file_sha256")
    legacy_path = tmp_path / "legacy.pt"
    write_torch_noclobber(legacy_path, legacy)
    with pytest.raises(ValueError, match="pilot full-scalar.*contract"):
        trainer.load_pilot_training_shard(
            legacy_path,
            expected_sha256=sha256_file(legacy_path),
            expected_split="source_validation",
        )

    wrong_scene = deepcopy(pilot)
    wrong_scene["scene_ids"] = ["scene0024_00"] * 2
    wrong_scene["sampling_audit"]["scene_id"] = "scene0024_00"
    wrong_scene["channel_sha256"] = base_trainer.training_shard_channel_sha256(
        wrong_scene
    )
    wrong_scene_path = tmp_path / "wrong_scene.pt"
    write_torch_noclobber(wrong_scene_path, wrong_scene)
    with pytest.raises(ValueError, match=r"outside exact 4\+2"):
        trainer.load_pilot_training_shard(
            wrong_scene_path,
            expected_sha256=sha256_file(wrong_scene_path),
            expected_split="source_validation",
        )


def test_base_binding_rejects_a_different_pilot_registry(monkeypatch) -> None:
    item = _scene("scene0004_00")
    shard = _pilot_training_shard()
    shard["lineage"][
        "pilot_cohort_region_view_registry_file_sha256"
    ] = "4" * 64
    monkeypatch.setattr(
        trainer,
        "load_pilot_training_shard",
        lambda *args, **kwargs: (shard, dict(item["training_shard"])),
    )
    monkeypatch.setattr(
        trainer.v1_trainer,
        "_load_context",
        lambda *args, **kwargs: ({}, dict(item["adaptive_context"])),
    )
    monkeypatch.setattr(
        trainer.v1_trainer,
        "_validate_scene_pair",
        lambda shard_value, context_value: "scene0004_00",
    )
    with pytest.raises(ValueError, match="scene binding differs"):
        trainer._base_bindings(
            [item],
            split="source_validation",
            expected_lineage={
                **shard["lineage"],
                "pilot_cohort_region_view_registry_file_sha256": "3" * 64,
            },
            registry_scene_records={
                "scene0004_00": {
                    "accepted_region_authority_file_sha256": "8" * 64,
                    "factorized_state_file_sha256": "9" * 64,
                }
            },
        )


@pytest.mark.parametrize(
    "tamper", ["teacher_manifest", "parent_cohort", "context_accepted_file"]
)
def test_base_binding_rejects_cross_authority_lineage(monkeypatch, tamper) -> None:
    item = _scene("scene0004_00")
    shard = _pilot_training_shard()
    context = {
        "input_authority": {
            "accepted_v2_canonical_region_authority": {"sha256": "8" * 64},
            "factorized_primitive_state": {"sha256": "9" * 64},
        }
    }
    expected_lineage = dict(shard["lineage"])
    if tamper == "teacher_manifest":
        expected_lineage["teacher_manifest_file_sha256"] = "a" * 64
    elif tamper == "parent_cohort":
        expected_lineage["cohort_authority_sha256"] = "a" * 64
    else:
        context["input_authority"]["accepted_v2_canonical_region_authority"][
            "sha256"
        ] = "a" * 64
    monkeypatch.setattr(
        trainer,
        "load_pilot_training_shard",
        lambda *args, **kwargs: (shard, dict(item["training_shard"])),
    )
    monkeypatch.setattr(
        trainer.v1_trainer,
        "_load_context",
        lambda *args, **kwargs: (context, dict(item["adaptive_context"])),
    )
    monkeypatch.setattr(
        trainer.v1_trainer,
        "_validate_scene_pair",
        lambda shard_value, context_value: "scene0004_00",
    )
    with pytest.raises(ValueError):
        trainer._base_bindings(
            [item],
            split="source_validation",
            expected_lineage=expected_lineage,
            registry_scene_records={
                "scene0004_00": {
                    "accepted_region_authority_file_sha256": "8" * 64,
                    "factorized_state_file_sha256": "9" * 64,
                }
            },
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "teacher_channel",
        "accepted_file",
        "teacher_file",
        "canonical_channel",
        "canonical_tensor",
    ],
)
def test_hard_negative_must_bind_pilot_shard_and_registry(tamper: str) -> None:
    canonical = torch.tensor([7, 11, 19], dtype=torch.int64)
    canonical_channel = base_trainer._tensor_channel_sha256(canonical)
    summary = {
        "accepted_v2_e0": "a" * 64,
        "teacher_pair_descriptors": "b" * 64,
        "teacher_pair_region_indices": "c" * 64,
        "canonical_region_indices": canonical_channel,
    }
    payload = {
        "canonical_region_indices": canonical,
        # The HN authority deliberately uses another hash namespace here;
        # cross-file comparison must use the AcceptedV2 channel hash.
        "canonical_region_indices_sha256": "d" * 64,
        "input_authority": {
            "accepted_v2": {
                "sha256": "e" * 64,
                "channel_sha256": {
                    "accepted_v2_e0": "a" * 64,
                    "canonical_region_indices": canonical_channel,
                },
            },
            "official_multiview_siglip2_teacher": {
                "sha256": "f" * 64,
                "channel_sha256": {
                    "pair_descriptors": "b" * 64,
                    "pair_region_indices": "c" * 64,
                },
            },
        },
    }
    registry_scene = {
        "accepted_region_authority_file_sha256": "e" * 64,
        "teacher_observation_authority_file_sha256": "f" * 64,
    }
    binding = SimpleNamespace(
        scene_id="scene0004_00",
        response_authority=SimpleNamespace(payload=payload),
    )
    trainer._validate_response_shard_channel_summary(
        binding, summary, registry_scene
    )
    if tamper == "teacher_channel":
        payload["input_authority"]["official_multiview_siglip2_teacher"][
            "channel_sha256"
        ]["pair_descriptors"] = "0" * 64
    elif tamper == "accepted_file":
        registry_scene["accepted_region_authority_file_sha256"] = "0" * 64
    elif tamper == "teacher_file":
        registry_scene["teacher_observation_authority_file_sha256"] = "0" * 64
    elif tamper == "canonical_channel":
        payload["input_authority"]["accepted_v2"]["channel_sha256"][
            "canonical_region_indices"
        ] = "0" * 64
    else:
        payload["canonical_region_indices"] = torch.tensor(
            [7, 11, 23], dtype=torch.int64
        )
    with pytest.raises(ValueError, match="hard-negative channels"):
        trainer._validate_response_shard_channel_summary(
            binding, summary, registry_scene
        )


def _build_spec(record: dict[str, str]) -> dict:
    scene = lambda scene_id: {
        "scene_id": scene_id,
        "training_shard": dict(record),
        "adaptive_context": dict(record),
        "hard_negative_authority": dict(record),
        "hard_negative_content_authority_sha256": "1" * 64,
    }
    return {
        "schema": builder.BUILD_SPEC_SCHEMA,
        "schema_version": builder.SCHEMA_VERSION,
        "cohort_authority": dict(record),
        "pilot_cohort_region_view_registry": dict(record),
        "benchmark_exclusion_manifest": dict(record),
        "fit_text_bank": dict(record),
        "canonical_negative_bank": dict(record),
        "compositional_banks": {
            name: {**record, "loss_weight": weight}
            for name, weight in trainer.COMPONENT_WEIGHTS.items()
        },
        "typed_relation_authority": {
            **record,
            "content_authority_sha256": "1" * 64,
        },
        "source_train": [scene(value) for value in trainer.TRAIN_SCENES],
        "source_validation": [
            scene(value) for value in trainer.VALIDATION_SCENES
        ],
    }


def test_execution_authority_builder_is_fixed_and_no_clobber(tmp_path) -> None:
    dependency = tmp_path / "dependency.bin"
    dependency.write_bytes(b"caller-sha-bound source dependency")
    spec = _build_spec(file_record(dependency))
    authority = builder.build(spec)
    assert authority["implementation"] == file_record(Path(trainer.__file__).resolve())
    assert authority["training_authorized"] is True
    assert authority["benchmark_execution_authorized"] is False
    output = tmp_path / "execution.json"
    builder.write_authority(spec, output)
    with pytest.raises(FileExistsError, match="refuses to clobber"):
        builder.write_authority(spec, output)

    changed = deepcopy(spec)
    changed["source_validation"][0]["scene_id"] = "scene0013_00"
    with pytest.raises(ValueError, match="fixed pilot cohort"):
        builder.build(changed)
