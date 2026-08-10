from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from radio_gs.interfaces.surface_region_summary import (
    surface_region_state_dict_sha256,
)
from radio_gs.interfaces.surface_region_typed_context_training import (
    accepted_v2_authority,
)
from radio_gs.interfaces.surface_region_v21_source_gate import (
    validate_source_pilot_chain,
    validate_source_promotion_evidence,
)
import radio_gs.interfaces.surface_region_v21_source_gate as source_gate
from radio_gs.models.surface_region_typed_context_residual import (
    SurfaceRegionAcceptedV2TypedContextResidualV1,
)
from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v21_pilot as pilot,
)
from radio_gs.scripts import train_surface_region_typed_context_residual as v1_trainer
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as pilot_shard
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    sha256_file,
)


SHA = "1" * 64


@pytest.fixture
def complete_source_loader_stub(monkeypatch):
    """Keep chain-shape tests light; a separate test exercises real failure."""

    def prepare(path, *, expected_sha256):
        assert sha256_file(path) == expected_sha256
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        execution = pilot.validate_execution_authority(raw)
        registry_record = execution["pilot_cohort_region_view_registry"]
        registry_raw = json.loads(
            Path(registry_record["path"]).read_text(encoding="utf-8")
        )
        registry = pilot_shard.validate_pilot_cohort_region_view_registry(
            registry_raw
        )
        execution["verified_path"] = str(Path(path).resolve())
        execution["verified_sha256"] = expected_sha256
        def bindings(split):
            result = []
            for row in execution[split]:
                base = SimpleNamespace(
                    training_shard=dict(row["training_shard"]),
                    adaptive_context=dict(row["adaptive_context"]),
                )
                result.append(
                    SimpleNamespace(
                        scene_id=row["scene_id"],
                        base=base,
                        hard_negative=dict(row["hard_negative_authority"]),
                        hard_negative_content_authority_sha256=row[
                            "hard_negative_content_authority_sha256"
                        ],
                    )
                )
            return tuple(result)

        source_manifest, _teacher_manifest = (
            pilot_shard.derive_pilot_global_manifests(registry)
        )
        return SimpleNamespace(
            execution=execution,
            registry=registry,
            source_state_manifest=source_manifest,
            train=bindings("source_train"),
            validation=bindings("source_validation"),
        )

    monkeypatch.setattr(source_gate.pilot, "prepare_inputs", prepare)


def _v1_validation() -> dict:
    names = (
        "mean_all_view_cosine",
        "p05_row_mean_all_view_cosine",
        "relation_fidelity",
    )
    base = dict(zip(names, (0.5, 0.4, 0.3)))
    candidate = dict(base)
    delta = {name: 0.0 for name in names}
    per_scene = {
        scene: {
            "base": dict(base),
            "candidate": dict(candidate),
            "candidate_minus_base": dict(delta),
            "active_rows": 2,
            "inactive_fallback_rows": 0,
            "fallback_bitwise_accepted_v2_e0": True,
            "relation_evaluation_rows": 2,
            "validation_no_grad": True,
        }
        for scene in pilot.VALIDATION_SCENES
    }
    checks = {
        "macro_mean_all_view_cosine": True,
        "macro_p05_row_mean_all_view_cosine": True,
        "macro_relation_fidelity": True,
        "paired_scene_worst_mean_delta": True,
        "every_scene_two_active_rows": True,
        "fallback_bitwise_accepted_v2_e0": True,
    }
    return {
        "aggregation": "scene_macro",
        "base": base,
        "candidate": candidate,
        "candidate_minus_base": delta,
        "paired_scene_mean_delta": {
            "minimum": 0.0,
            "p05": 0.0,
            "maximum": 0.0,
        },
        "per_scene": per_scene,
        "non_regression_checks": checks,
        "non_regression_passed": True,
        "validation_no_grad": True,
        "global_or_split_teacher_densification": False,
    }


def _validation(aux: float, absolute: tuple[float, float], pairwise: float) -> dict:
    per_scene = {}
    for index, scene in enumerate(pilot.VALIDATION_SCENES):
        per_scene[scene] = {
            "response_auxiliary_loss": aux,
            "response_absolute_relevance_loss": absolute[index],
            "response_continuous_pairwise_relevance_loss": pairwise,
            "response_objective_hard_negative_pairs": 4,
            "response_authority_hard_negative_pairs": 4,
            "response_pair_trainable_endpoint_coverage": 1.0,
        }
    return {
        "v1_non_regression": _v1_validation(),
        "response_listwise_v21": {
            "scene_count": 2,
            "scene_macro_auxiliary_loss": aux,
            "scene_macro_pair_trainable_endpoint_coverage": 1.0,
            "all_authority_pairs_retained": True,
            "per_scene": per_scene,
        },
        "selection_eligible": True,
        "validation_no_grad": True,
        "benchmark_opened": False,
    }


def _write_json(path: Path, value: object) -> dict[str, str]:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return file_record(path)


def _write_torch(path: Path, value: object) -> dict[str, str]:
    torch.save(value, path)
    return file_record(path)


def _scene(scene_id: str, record: dict[str, str]) -> dict:
    return {
        "scene_id": scene_id,
        "training_shard": dict(record),
        "adaptive_context": dict(record),
        "hard_negative_authority": dict(record),
        "hard_negative_content_authority_sha256": SHA,
    }


def _pilot_registry(tmp_path: Path) -> tuple[dict[str, str], str, str]:
    teacher_model_sha = canonical_json_sha256(
        pilot_shard.official_teacher_model_authority()
    )
    records = []
    for split, scenes in (
        ("source_train", pilot.TRAIN_SCENES),
        ("source_validation", pilot.VALIDATION_SCENES),
    ):
        for scene in scenes:
            count = 2 if split == "source_validation" else 1
            regions = []
            for index in range(count):
                fingerprint = hashlib.sha256(
                    f"{scene}-region-{index}".encode()
                ).hexdigest()
                view_sha = hashlib.sha256(
                    f"{scene}-view-{index}".encode()
                ).hexdigest()
                regions.append(
                    {
                        "region_fingerprint": fingerprint,
                        "region_row_id": pilot_shard.stable_region_id(
                            scene, fingerprint
                        ),
                        "teacher_view_ids": [f"{scene}:source-rgb:{view_sha}"],
                        "eligible_overlap_teacher": True,
                    }
                )
            regions.sort(key=lambda row: row["region_row_id"])
            accepted_sha = hashlib.sha256(f"{scene}-accepted".encode()).hexdigest()
            state_sha = hashlib.sha256(f"{scene}-state".encode()).hexdigest()
            records.append(
                {
                    "scene_id": scene,
                    "physical_space_id": pilot_shard.trainer.canonical_physical_space_id(
                        scene
                    ),
                    "split": split,
                    "accepted_region_authority_file_sha256": accepted_sha,
                    "factorized_state_file_sha256": state_sha,
                    "teacher_observation_authority_file_sha256": hashlib.sha256(
                        f"{scene}-teacher".encode()
                    ).hexdigest(),
                    "source_state_artifact_sha256": (
                        pilot_shard.source_state_artifact_sha256(
                            accepted_region_file_sha256=accepted_sha,
                            factorized_state_file_sha256=state_sha,
                        )
                    ),
                    "teacher_model_authority_sha256": teacher_model_sha,
                    "eligible_overlap_teacher_row_count": count,
                    "region_records": regions,
                }
            )
    contract = pilot_shard.pilot_cohort_registry_contract()
    registry = {
        "schema": pilot_shard.PILOT_COHORT_REGISTRY_SCHEMA,
        "schema_version": pilot_shard.SCHEMA_VERSION,
        "contract": contract,
        "contract_sha256": canonical_json_sha256(contract),
        "cohort_authority_sha256": "a" * 64,
        "cohort_authority_file_sha256": "b" * 64,
        "pilot_splits": {
            "source_train": list(pilot.TRAIN_SCENES),
            "source_validation": list(pilot.VALIDATION_SCENES),
        },
        "teacher_model_authority_sha256": teacher_model_sha,
        "scene_records": sorted(records, key=lambda row: row["scene_id"]),
        "source_access": pilot_shard._authority_access(source_rgb_used=True),
    }
    registry["authority_sha256"] = pilot_shard._authority_content_sha256(registry)
    source_manifest, _teacher_manifest = pilot_shard.derive_pilot_global_manifests(
        registry
    )
    return (
        _write_json(tmp_path / "pilot_registry.json", registry),
        registry["authority_sha256"],
        source_manifest["authority_sha256"],
    )


def _chain(
    tmp_path: Path,
    *,
    selected_epoch: int = 1,
    selected_absolute: tuple[float, float] = (0.25, 0.35),
    bad_scale: bool = False,
    bad_certificate_execution: bool = False,
    bad_certificate_registry: bool = False,
    bad_normalization_source_lineage: bool = False,
    tamper_checkpoint_state: bool = False,
) -> tuple[Path, str]:
    dependency = tmp_path / "source-dependency.bin"
    dependency.write_bytes(b"immutable source dependency")
    dependency_record = file_record(dependency)
    (
        registry_record,
        registry_authority_sha256,
        pilot_source_manifest_authority_sha256,
    ) = _pilot_registry(tmp_path)
    implementation_record = file_record(Path(pilot.__file__).resolve())
    addendum_record = file_record(
        Path(pilot.__file__).resolve().parents[2] / pilot.ACTIVE_PAIR_ADDENDUM
    )
    execution = {
        "schema": pilot.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_source_only_4train_2validation_v21_pilot",
        "implementation": implementation_record,
        "active_pair_addendum": addendum_record,
        "cohort_authority": dict(dependency_record),
        "pilot_cohort_region_view_registry": dict(registry_record),
        "benchmark_exclusion_manifest": dict(dependency_record),
        "fit_text_bank": dict(dependency_record),
        "canonical_negative_bank": dict(dependency_record),
        "compositional_banks": {
            name: {**dependency_record, "loss_weight": weight}
            for name, weight in pilot.COMPONENT_WEIGHTS.items()
        },
        "typed_relation_authority": {
            **dependency_record,
            "content_authority_sha256": SHA,
        },
        "source_train": [
            _scene(scene, dependency_record) for scene in pilot.TRAIN_SCENES
        ],
        "source_validation": [
            _scene(scene, dependency_record) for scene in pilot.VALIDATION_SCENES
        ],
        "training_authorized": True,
        "benchmark_execution_authorized": False,
        "source_access": pilot.source_access(),
    }
    execution_record = _write_json(tmp_path / "execution.json", execution)
    train_inputs = [
        {
            "scene_id": row["scene_id"],
            "training_shard": row["training_shard"],
            "adaptive_context": row["adaptive_context"],
        }
        for row in execution["source_train"]
    ]
    normalization = {
        "schema": "radio_gs.v21_pilot_train4_normalization.v1",
        "schema_version": 1,
        "fit_split": "fixed_source_train_four_only",
        "source_state_cohort_authority_sha256": (
            "1" * 64
            if bad_normalization_source_lineage
            else pilot_source_manifest_authority_sha256
        ),
        "train_input_records": train_inputs,
        "source_count": 8,
        "median": torch.zeros(30, dtype=torch.float32),
        "mad": torch.zeros(30, dtype=torch.float32),
        "robust_scale": torch.full(
            (30,), 2.0 if bad_scale else 1.0, dtype=torch.float32
        ),
        "constant_coordinate_mask": torch.zeros(30, dtype=torch.bool),
        "source_max_robust_linf": 2.0,
        "source_boundary_score_median": 0.5,
        "validation_contribution": False,
        "source_access": pilot.source_access(),
    }
    normalization_record = _write_torch(
        tmp_path / "model.pt.normalization.pt", normalization
    )
    model = SurfaceRegionAcceptedV2TypedContextResidualV1(
        scalar_median=normalization["median"],
        scalar_robust_scale=normalization["robust_scale"],
        max_angle_radians=v1_trainer.MAX_ANGLE_RADIANS,
        max_alpha=v1_trainer.MAX_ALPHA,
    )
    state = {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }
    state_sha = surface_region_state_dict_sha256(state)
    epoch0 = _validation(0.5, (0.30, 0.40), 0.30)
    epoch1 = _validation(0.4, selected_absolute, 0.20)
    history = [
        {
            "epoch": 0,
            "training": None,
            "validation": epoch0,
            "model_state_dict_sha256": state_sha,
        },
        {
            "epoch": 1,
            "training": {},
            "validation": epoch1,
            "model_state_dict_sha256": state_sha,
        },
    ]
    if selected_epoch == 0:
        history = history[:1]
    selected_validation = history[selected_epoch]["validation"]
    certificate_execution = (
        dependency_record if bad_certificate_execution else execution_record
    )
    certificate = {
        "schema": (
            "radio_gs.surface_region_typed_context_response_listwise_v21_"
            "pilot_certificate.v1"
        ),
        "schema_version": 1,
        "training_contract": pilot.training_contract(),
        "training_contract_sha256": pilot.TRAINING_CONTRACT_SHA256,
        "execution_authority": dict(certificate_execution),
        "cohort_authority": dict(execution["cohort_authority"]),
        "pilot_cohort_region_view_registry": dict(
            dependency_record
            if bad_certificate_registry
            else execution["pilot_cohort_region_view_registry"]
        ),
        "pilot_cohort_region_view_registry_authority_sha256": (
            registry_authority_sha256
        ),
        "benchmark_exclusion_manifest": dict(execution["benchmark_exclusion_manifest"]),
        "input_records_by_split": {
            "source_train": copy.deepcopy(execution["source_train"]),
            "source_validation": copy.deepcopy(execution["source_validation"]),
        },
        "selected_epoch": selected_epoch,
        "selected_validation": copy.deepcopy(selected_validation),
        "model_state_dict_sha256": state_sha,
        "normalization_authority": dict(normalization_record),
        "normalization_content_authority_sha256": (
            pilot.pilot_normalization_authority_sha256(normalization)
        ),
        "source_access": pilot.source_access(),
        "benchmark_opened": False,
    }
    certificate["content_sha256"] = canonical_json_sha256(certificate)
    certificate_record = _write_json(
        tmp_path / "model.pt.certificate.json", certificate
    )
    checkpoint_state = copy.deepcopy(state)
    if tamper_checkpoint_state:
        checkpoint_state["residual_projection.bias"][0] = 1.0
    checkpoint = {
        "schema": pilot.CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "model_class": type(model).__name__,
        "model_architecture": model.architecture(),
        "accepted_v2_authority": accepted_v2_authority(),
        "model_state_dict": checkpoint_state,
        "model_state_dict_sha256": state_sha,
        "normalization_authority": dict(normalization_record),
        "certificate": dict(certificate_record),
        "selected_epoch": selected_epoch,
        "source_access": pilot.source_access(),
    }
    checkpoint_record = _write_torch(tmp_path / "model.pt", checkpoint)
    result = {
        "schema": (
            "radio_gs.surface_region_typed_context_response_listwise_v21_"
            "pilot_result.v1"
        ),
        "schema_version": 1,
        "status": "source_only_pilot_complete_no_benchmark_execution",
        "training_contract": pilot.training_contract(),
        "training_contract_sha256": pilot.TRAINING_CONTRACT_SHA256,
        "execution_authority": dict(execution_record),
        "checkpoint": dict(checkpoint_record),
        "normalization_authority": dict(normalization_record),
        "certificate": dict(certificate_record),
        "selected_epoch": selected_epoch,
        "automatic_fallback_to_epoch_zero": selected_epoch == 0,
        "selected_validation": copy.deepcopy(selected_validation),
        "history": history,
        "source_access": pilot.source_access(),
        "benchmark_opened": False,
    }
    result_path = tmp_path / "model.pt.json"
    _write_json(result_path, result)
    return result_path, sha256_file(result_path)


def test_full_source_chain_passes_and_authorizes_only_source_promotion(
    tmp_path: Path, complete_source_loader_stub,
) -> None:
    path, digest = _chain(tmp_path)
    gate = validate_source_pilot_chain(path, expected_sha256=digest)
    assert gate["selected_epoch"] == 1
    assert gate["source_promotion_authorized"] is True
    assert gate["target_execution_authorized"] is False
    assert gate["benchmark_opened"] is False


def test_lightweight_gate_rejects_cross_scene_absolute_relevance_regression(
    tmp_path: Path,
) -> None:
    path, _ = _chain(tmp_path, selected_absolute=(0.25, 0.41))
    result = json.loads(path.read_text(encoding="utf-8"))
    gate = validate_source_promotion_evidence(result)
    assert gate["passed"] is False
    assert gate["checks"]["absolute_relevance_every_scene_non_regression"] is False


def test_lightweight_gate_accepts_ineligible_intermediate_history_epoch(
    tmp_path: Path,
) -> None:
    path, _ = _chain(tmp_path)
    result = json.loads(path.read_text(encoding="utf-8"))
    selected = copy.deepcopy(result["history"][1])
    selected["epoch"] = 2
    selected["validation"] = _validation(0.35, (0.24, 0.34), 0.19)
    regressed = result["history"][1]["validation"]
    v1 = regressed["v1_non_regression"]
    for scene in pilot.VALIDATION_SCENES:
        row = v1["per_scene"][scene]
        row["candidate"]["mean_all_view_cosine"] = 0.49
        row["candidate_minus_base"]["mean_all_view_cosine"] = -0.01
    v1["candidate"]["mean_all_view_cosine"] = 0.49
    v1["candidate_minus_base"]["mean_all_view_cosine"] = -0.01
    v1["paired_scene_mean_delta"] = {
        "minimum": -0.01,
        "p05": -0.01,
        "maximum": -0.01,
    }
    v1["non_regression_checks"]["macro_mean_all_view_cosine"] = False
    v1["non_regression_checks"]["paired_scene_worst_mean_delta"] = False
    v1["non_regression_passed"] = False
    regressed["selection_eligible"] = False
    result["history"].append(selected)
    result["selected_epoch"] = 2
    result["selected_validation"] = selected["validation"]
    gate = validate_source_promotion_evidence(result)
    assert gate["passed"] is True
    assert gate["selected_epoch"] == 2


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"selected_epoch": 0}, "selected_epoch_positive"),
        (
            {"selected_absolute": (0.25, 0.41)},
            "absolute_relevance_every_scene_non_regression",
        ),
        ({"bad_scale": True}, "robust scale"),
        ({"bad_certificate_execution": True}, "chain differs"),
        ({"bad_certificate_registry": True}, "chain differs"),
        ({"bad_normalization_source_lineage": True}, "chain differs"),
        ({"tamper_checkpoint_state": True}, "state SHA-256 differs"),
    ],
)
def test_full_gate_fails_closed_on_epoch_scene_and_cross_file_tampering(
    tmp_path: Path, kwargs: dict, match: str, complete_source_loader_stub
) -> None:
    path, digest = _chain(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=match):
        validate_source_pilot_chain(path, expected_sha256=digest)


def test_full_gate_rejects_result_to_checkpoint_sha_tampering(
    tmp_path: Path, complete_source_loader_stub
) -> None:
    path, _ = _chain(tmp_path)
    result = json.loads(path.read_text(encoding="utf-8"))
    result["checkpoint"]["sha256"] = "f" * 64
    _write_json(path, result)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        validate_source_pilot_chain(path, expected_sha256=sha256_file(path))


def test_full_gate_opens_and_rejects_existing_fake_source_dependencies(
    tmp_path: Path,
) -> None:
    path, digest = _chain(tmp_path)
    with pytest.raises(ValueError):
        validate_source_pilot_chain(path, expected_sha256=digest)


@pytest.mark.parametrize("fake_kind", ["shard", "context", "hard_negative"])
def test_source_replay_rejects_existing_fake_bound_scene_payloads(
    tmp_path: Path, monkeypatch, fake_kind: str
) -> None:
    path, digest = _chain(tmp_path)
    fake = tmp_path / f"fake_{fake_kind}.pt"
    torch.save({}, fake)
    fake_sha = sha256_file(fake)

    def reject_during_complete_source_reload(*args, **kwargs):
        if fake_kind == "shard":
            return pilot.load_pilot_training_shard(
                fake,
                expected_sha256=fake_sha,
                expected_split="source_train",
            )
        if fake_kind == "context":
            return pilot.v1_trainer._load_context(
                fake, expected_sha256=fake_sha
            )
        return pilot.v2_trainer._load_response_authority(
            fake,
            expected_file_sha256=fake_sha,
            expected_content_authority_sha256="1" * 64,
            expected_scene_id="scene0001_00",
        )

    monkeypatch.setattr(
        source_gate.pilot,
        "prepare_inputs",
        reject_during_complete_source_reload,
    )
    with pytest.raises(ValueError):
        validate_source_pilot_chain(path, expected_sha256=digest)
