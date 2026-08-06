from __future__ import annotations

from argparse import Namespace
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from radio_gs.field import FeatureSpaceSignature
from radio_gs.field.factorized_radio_contract import FactorizedRadioFieldSignature
from radio_gs.interfaces.factorized_primitive_state import FactorizedPrimitiveState
import radio_gs.interfaces.factorized_primitive_state as state_module
from radio_gs.scripts import (
    materialize_full_scalar_clean_training_shard as materializer,
)
from radio_gs.scripts import (
    seal_full_scalar_clean_cohort_region_view_registry as registry_sealer,
)
from radio_gs.scripts import train_surface_region_full_scalar_residual as trainer
from radio_gs.scripts.materialize_official_multiview_siglip2_teacher_authority import (
    build_source_rgb_scene_authority,
)
from radio_gs.training.factorized_radio_cache import (
    FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


def _cohort(tmp_path: Path) -> tuple[Path, dict]:
    train = [f"scene{index:04d}_00" for index in range(24)]
    validation = [f"scene{index:04d}_00" for index in range(24, 32)]
    payload = {
        "schema": trainer.COHORT_AUTHORITY_SCHEMA,
        "schema_version": trainer.COHORT_AUTHORITY_SCHEMA_VERSION,
        "contract": trainer.cohort_authority_contract(),
        "contract_sha256": trainer.COHORT_AUTHORITY_CONTRACT_SHA256,
        "source_train_scene_ids": train,
        "source_validation_scene_ids": validation,
        "source_train_physical_space_ids": [
            trainer.canonical_physical_space_id(scene) for scene in train
        ],
        "source_validation_physical_space_ids": [
            trainer.canonical_physical_space_id(scene) for scene in validation
        ],
        "benchmark_exclusion": {
            "manifest_authority_sha256": "a" * 64,
            "manifest_file_sha256": "b" * 64,
        },
        "source_access": trainer._cohort_authority_access(),
    }
    payload["authority_sha256"] = trainer.cohort_authority_content_sha256(payload)
    path = tmp_path / "cohort.json"
    write_frozen_json(path, payload)
    return path, payload


def _state() -> FactorizedPrimitiveState:
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    valid = torch.tensor([True, True, False, True])
    rows = torch.where(valid)[0]
    directions = torch.zeros(3, 1280, dtype=torch.float16)
    directions[0, 0] = directions[1, 1] = directions[2, 2] = 1
    base = FeatureSpaceSignature(
        radio_version="c-radio_v4-h",
        radio_checkpoint_sha256=materializer.OFFICIAL_RADIO_CHECKPOINT_SHA256,
        raw_feature_dim=1280,
        token_type="primitive",
        normalization="radio_raw_full",
        crop_policy="training_views_canonical_factorized_radio_v1",
    )
    signature = FactorizedRadioFieldSignature.create(base)
    registration = "c" * 64
    metadata = {
        "source": "factorized_primitive_state_v2",
        "field_checkpoint": "/frozen/field.pth",
        "field_checkpoint_sha256": "d" * 64,
        "factorized_radio_cache": "/frozen/cache.pt",
        "factorized_radio_cache_sha256": "e" * 64,
        "factorized_radio_field_signature": signature.to_dict(),
        "factorized_radio_field_signature_sha256": signature.digest,
        "factorized_radio_reliability_scalar_names": [
            "directional_resultant",
            "directional_dispersion",
            "log_amplitude_std",
            "observation_evidence",
            "visibility_purity",
        ],
        "factorized_radio_reliability_scalar_names_sha256": canonical_json_sha256(
            [
                "directional_resultant",
                "directional_dispersion",
                "log_amplitude_std",
                "observation_evidence",
                "visibility_purity",
            ]
        ),
        "geometry_fingerprint": {
            "num_gaussians": len(xyz),
            "xyz_sha256": state_module._float32_rows_sha256(xyz),
        },
        "visibility_purity_authority": {
            **FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY,
            "registration_responsibility_cache_sha256": registration,
        },
        "registration_responsibility_cache_sha256": registration,
        "feature_output_bundle_sha256": "f" * 64,
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    # Use the exact exported names rather than trusting the spelling above.
    from radio_gs.field.factorized_radio_contract import (
        FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES,
        FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256,
    )

    metadata["factorized_radio_reliability_scalar_names"] = list(
        FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES
    )
    metadata["factorized_radio_reliability_scalar_names_sha256"] = (
        FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256
    )
    return FactorizedPrimitiveState(
        xyz=xyz,
        valid=valid,
        global_rows=rows,
        semantic_direction=directions,
        predicted_log_amplitude=torch.tensor([0.0, 0.1, 0.2]),
        directional_dispersion=torch.tensor([0.1, 0.2, 0.3]),
        log_amplitude_std=torch.tensor([0.1, 0.1, 0.1]),
        observation_evidence=torch.tensor([0.8, 0.7, 0.6]),
        visibility_purity_value=torch.tensor([0.9, 0.8, 0.7]),
        visibility_purity_known=torch.ones(3, dtype=torch.bool),
        metadata=metadata,
    )


def _accepted(scene: str, geometry: dict) -> dict:
    e0 = torch.zeros(2, trainer.DESCRIPTOR_DIM, dtype=torch.float32)
    e0[0, 0] = e0[1, 1] = 1
    payload = {
        "schema": materializer.ACCEPTED_REGION_SCHEMA,
        "schema_version": materializer.ACCEPTED_REGION_SCHEMA_VERSION,
        "contract": materializer.accepted_region_authority_contract(),
        "contract_sha256": canonical_json_sha256(
            materializer.accepted_region_authority_contract()
        ),
        "scene_id": scene,
        "physical_space_id": trainer.canonical_physical_space_id(scene),
        "accepted_v2_authority": trainer._accepted_v2_authority(),
        "geometry_fingerprint": geometry,
        "accepted_base_valid": torch.tensor([True, True, True, False]),
        "canonical_region_indices": torch.tensor([0, 1], dtype=torch.long),
        "region_rows": torch.tensor([[0, 3], [1, 0]], dtype=torch.long),
        "token_mask": torch.ones(2, 2, dtype=torch.bool),
        "anchor_index": torch.zeros(2, dtype=torch.long),
        "scale_indices": torch.zeros(2, dtype=torch.long),
        "accepted_v2_e0": e0,
        "input_authority": {
            "geometry_authority": {
                "kind": "factorized_primitive_state_v2",
                "factorized_primitive_state_file_sha256": "1" * 64,
                "factorized_primitive_state_contract_sha256": (
                    materializer.FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256
                ),
                "factorized_field_checkpoint_file_sha256": "2" * 64,
                "factorized_radio_cache_file_sha256": "3" * 64,
                "primitive_row_authority_sha256": "4" * 64,
                "geometry_fingerprint": geometry,
            },
            "support_graph_authority": {
                "kind": "canonical_query_free_support_graph_v1",
                "support_graph_file_sha256": "5" * 64,
                "primitive_row_authority_sha256": "4" * 64,
            },
            "selection_authority": {
                "kind": "exact_marginal_anchor_visibility_sparse_selection_v1",
                "exact_marginal_responsibility_authority_file_sha256": "6" * 64,
                "exact_marginal_formula_sha256": "7" * 64,
                "responsibility_view_records_sha256": "8" * 64,
                "sampling_contract_sha256": materializer.SAMPLING_CONTRACT_SHA256,
            },
            "accepted_v2_checkpoint_authority": trainer._accepted_v2_authority(),
            "official_summary_head_authority": (
                materializer.accepted_region_official_head_authority()
            ),
        },
        "source_access": materializer._authority_access(source_rgb_used=False),
    }
    payload["region_fingerprints"] = materializer.stable_region_fingerprints(payload)
    payload["selection_audit"] = {
        "sampling_contract_sha256": materializer.SAMPLING_CONTRACT_SHA256,
        "canonical_candidate_region_count": 6,
        "exact_overlap_candidate_count": 6,
        "teacher_visible_candidate_count": 2,
        "selected_region_count": 2,
        "selected_count_by_scale": [2],
    }
    payload["channel_sha256"] = materializer.accepted_region_channel_sha256(payload)
    return payload


def _teacher(scene: str, accepted: dict, *, accepted_file_sha256: str) -> dict:
    fingerprints = materializer.stable_region_fingerprints(accepted)
    views = [
        {
            "frame_id": f"{index:05d}",
            "source_relative_path": f"color/{index:05d}.jpg",
            "source_image_sha256": hashlib.sha256(f"rgb-{index}".encode()).hexdigest(),
            "field_frame_authority_sha256": hashlib.sha256(
                f"frame-{index}".encode()
            ).hexdigest(),
            "source_image_height": 16,
            "source_image_width": 16,
            "feature_grid_height": 4,
            "feature_grid_width": 4,
            "responsibility_view_index": index,
            "responsibility_view_file_sha256": hashlib.sha256(
                f"responsibility-{index}".encode()
            ).hexdigest(),
        }
        for index in range(2)
    ]
    descriptors = torch.zeros(3, trainer.DESCRIPTOR_DIM, dtype=torch.float32)
    descriptors[0, 0] = 1
    descriptors[1, 1] = 1
    descriptors[2, 3] = 1
    pair_rows = torch.tensor([0, 1, 1], dtype=torch.long)
    pair_views = torch.tensor([0, 0, 1], dtype=torch.long)
    boxes = torch.tensor(
        [
            [0, 1, 8, 9],
            [2, 3, 10, 11],
            [4, 5, 12, 13],
        ],
        dtype=torch.long,
    )
    hit_counts = torch.tensor([4, 5, 3], dtype=torch.long)
    model = materializer.official_teacher_model_authority()
    payload = {
        "schema": materializer.TEACHER_OBSERVATION_SCHEMA,
        "schema_version": materializer.TEACHER_OBSERVATION_SCHEMA_VERSION,
        "contract": materializer.teacher_observation_authority_contract(),
        "contract_sha256": canonical_json_sha256(
            materializer.teacher_observation_authority_contract()
        ),
        "scene_id": scene,
        "physical_space_id": trainer.canonical_physical_space_id(scene),
        "source_rgb_scene_authority_sha256": "9" * 64,
        "teacher_model_authority": model,
        "teacher_model_authority_sha256": canonical_json_sha256(model),
        "canonical_region_indices": accepted["canonical_region_indices"].clone(),
        "region_fingerprints": fingerprints,
        "view_records": views,
        "pair_region_indices": pair_rows,
        "pair_view_indices": pair_views,
        "pair_descriptors": descriptors,
        "pair_crop_boxes_tlbr": boxes,
        "pair_support_hit_counts": hit_counts,
        "pair_visible_primitive_counts": torch.tensor([2, 2, 1]),
        "selection_audit": {
            "accepted_selection_audit": copy.deepcopy(accepted["selection_audit"]),
            "pair_count": 3,
            "maximum_views_per_region": 2,
        },
        "input_authority": {
            "source_rgb_scene_authority_file_sha256": "6" * 64,
            "source_rgb_scene_authority_content_sha256": "9" * 64,
            "factorized_primitive_state_file_sha256": "1" * 64,
            "accepted_region_authority_file_sha256": accepted_file_sha256,
            "accepted_region_channel_sha256": canonical_json_sha256(
                accepted["channel_sha256"]
            ),
            "accepted_region_fingerprints_sha256": canonical_json_sha256(fingerprints),
            "exact_marginal_responsibility_authority_file_sha256": "7" * 64,
            "official_radio_checkpoint_file_sha256": (
                materializer.OFFICIAL_RADIO_CHECKPOINT_SHA256
            ),
            "descriptor_definition": (
                materializer.official_teacher_descriptor_definition()
            ),
        },
        "source_access": materializer._authority_access(source_rgb_used=True),
    }
    payload["channel_sha256"] = materializer.teacher_observation_channel_sha256(payload)
    return payload


def _registry(
    cohort: dict,
    cohort_file_sha: str,
    *,
    current_scene: str,
    accepted_sha: str,
    state_sha: str,
    teacher_sha: str,
    accepted: dict,
    teacher: dict,
) -> dict:
    teacher_model_sha = canonical_json_sha256(
        materializer.official_teacher_model_authority()
    )
    current_fingerprints = materializer.stable_region_fingerprints(accepted)
    current_view_ids = [
        materializer.stable_teacher_view_id(current_scene, record)
        for record in teacher["view_records"]
    ]
    current_pair_rows = teacher["pair_region_indices"]
    current_pair_views = teacher["pair_view_indices"]
    scene_records = []
    for split, scenes in (
        ("source_train", cohort["source_train_scene_ids"]),
        ("source_validation", cohort["source_validation_scene_ids"]),
    ):
        for scene in scenes:
            if scene == current_scene:
                region_records = sorted(
                    [
                        {
                            "region_fingerprint": fingerprint,
                            "region_row_id": materializer.stable_region_id(
                                scene, fingerprint
                            ),
                            "teacher_view_ids": [
                                current_view_ids[int(view)]
                                for view in current_pair_views[current_pair_rows == row]
                            ],
                            "eligible_overlap_teacher": row < 2,
                        }
                        for row, fingerprint in enumerate(current_fingerprints)
                    ],
                    key=lambda item: item["region_row_id"],
                )
                accepted_file = accepted_sha
                factorized_file = state_sha
                teacher_file = teacher_sha
            else:
                region_records = []
                for index in range(2):
                    fingerprint = hashlib.sha256(
                        f"{scene}-region-{index}".encode()
                    ).hexdigest()
                    view_hash = hashlib.sha256(
                        f"{scene}-view-{index}".encode()
                    ).hexdigest()
                    region_records.append(
                        {
                            "region_fingerprint": fingerprint,
                            "region_row_id": materializer.stable_region_id(
                                scene, fingerprint
                            ),
                            "teacher_view_ids": [f"{scene}:source-rgb:{view_hash}"],
                            "eligible_overlap_teacher": True,
                        }
                    )
                region_records.sort(key=lambda item: item["region_row_id"])
                accepted_file = hashlib.sha256(f"{scene}-accepted".encode()).hexdigest()
                factorized_file = hashlib.sha256(f"{scene}-state".encode()).hexdigest()
                teacher_file = hashlib.sha256(f"{scene}-teacher".encode()).hexdigest()
            scene_records.append(
                {
                    "scene_id": scene,
                    "physical_space_id": trainer.canonical_physical_space_id(scene),
                    "split": split,
                    "accepted_region_authority_file_sha256": accepted_file,
                    "factorized_state_file_sha256": factorized_file,
                    "teacher_observation_authority_file_sha256": teacher_file,
                    "source_state_artifact_sha256": (
                        materializer.source_state_artifact_sha256(
                            accepted_region_file_sha256=accepted_file,
                            factorized_state_file_sha256=factorized_file,
                        )
                    ),
                    "teacher_model_authority_sha256": teacher_model_sha,
                    "eligible_overlap_teacher_row_count": 2,
                    "region_records": region_records,
                }
            )
    scene_records.sort(key=lambda item: item["scene_id"])
    return materializer.build_cohort_region_view_registry(
        cohort_authority=cohort,
        cohort_authority_file_sha256=cohort_file_sha,
        scene_records=scene_records,
    )


def _inputs(tmp_path: Path) -> tuple[Namespace, dict]:
    cohort_path, cohort = _cohort(tmp_path)
    scene = "scene0024_00"
    state = _state()
    state_path = tmp_path / "state.pt"
    write_torch_noclobber(state_path, state.to_payload())
    accepted = _accepted(scene, dict(state.metadata["geometry_fingerprint"]))
    accepted_path = tmp_path / "accepted.pt"
    write_torch_noclobber(accepted_path, accepted)
    teacher = _teacher(
        scene,
        accepted,
        accepted_file_sha256=sha256_file(accepted_path),
    )
    teacher_path = tmp_path / "teacher.pt"
    write_torch_noclobber(teacher_path, teacher)
    registry = _registry(
        cohort,
        sha256_file(cohort_path),
        current_scene=scene,
        accepted_sha=sha256_file(accepted_path),
        state_sha=sha256_file(state_path),
        teacher_sha=sha256_file(teacher_path),
        accepted=accepted,
        teacher=teacher,
    )
    registry_path = tmp_path / "registry.json"
    write_frozen_json(registry_path, registry)
    args = Namespace(
        cohort_authority=str(cohort_path),
        expected_cohort_authority_sha256=sha256_file(cohort_path),
        cohort_region_view_registry=str(registry_path),
        expected_cohort_region_view_registry_sha256=sha256_file(registry_path),
        accepted_region_authority=str(accepted_path),
        expected_accepted_region_authority_sha256=sha256_file(accepted_path),
        factorized_state=str(state_path),
        expected_factorized_state_sha256=sha256_file(state_path),
        expected_field_checkpoint_sha256="d" * 64,
        expected_factorized_radio_cache_sha256="e" * 64,
        teacher_observation_authority=str(teacher_path),
        expected_teacher_observation_authority_sha256=sha256_file(teacher_path),
        output_shard=str(tmp_path / "validation_shard.pt"),
        output_source_state_manifest=str(tmp_path / "source_manifest.json"),
        output_teacher_manifest=str(tmp_path / "teacher_manifest.json"),
        output_receipt=str(tmp_path / "receipt.json"),
        preflight_only=False,
    )
    return args, {
        "cohort": cohort,
        "registry": registry,
        "accepted": accepted,
        "teacher": teacher,
    }


def test_materializer_builds_one_valid_shard_and_global_manifests(
    tmp_path: Path,
) -> None:
    args, values = _inputs(tmp_path)
    prepared = materializer.preflight(args)
    assert prepared["split"] == "source_validation"
    assert prepared["nonvacuous_prerequisite"] == {
        "eligible_overlap_teacher_rows": 2,
        "minimum_required": 2,
        "passed": True,
        "training_certificate_claimed": False,
        "in_domain_after_normalization_pending": True,
    }

    receipt = materializer.materialize(args)
    shard, _record = trainer.load_training_shard(
        args.output_shard,
        expected_sha256=sha256_file(args.output_shard),
        expected_split="source_validation",
    )
    assert int(shard["eligible"].sum()) == 2
    assert shard["raw_full_scalar_summary"].shape == (2, 18)
    assert shard["official_multiview_siglip2_teacher_pair_descriptors"].shape == (
        3,
        trainer.DESCRIPTOR_DIM,
    )
    source = trainer._load_json_manifest(
        args.output_source_state_manifest,
        expected_sha256=sha256_file(args.output_source_state_manifest),
        label="source-state manifest",
        validator=trainer.validate_source_state_manifest,
    )[0]
    teacher = trainer._load_json_manifest(
        args.output_teacher_manifest,
        expected_sha256=sha256_file(args.output_teacher_manifest),
        label="teacher manifest",
        validator=trainer.validate_teacher_manifest,
    )[0]
    assert len(source["scene_records"]) == 32
    assert len(teacher["region_view_records"]) == sum(
        len(record["region_records"]) for record in values["registry"]["scene_records"]
    )
    assert receipt["source_access"]["online_model_execution"] is False

    with pytest.raises(FileExistsError, match="refuses to clobber"):
        materializer.materialize(args)


def test_registry_rejects_partial_or_vacuous_validation_coverage(
    tmp_path: Path,
) -> None:
    args, values = _inputs(tmp_path)
    partial = copy.deepcopy(values["registry"])
    partial["scene_records"].pop()
    partial["authority_sha256"] = materializer._authority_content_sha256(partial)
    with pytest.raises(ValueError, match=r"24\+8"):
        materializer.validate_cohort_region_view_registry(partial)

    vacuous = copy.deepcopy(values["registry"])
    current = next(
        record
        for record in vacuous["scene_records"]
        if record["scene_id"] == "scene0024_00"
    )
    current["region_records"][1]["eligible_overlap_teacher"] = False
    current["region_records"][1]["teacher_view_ids"] = []
    current["eligible_overlap_teacher_row_count"] = 1
    vacuous["authority_sha256"] = materializer._authority_content_sha256(vacuous)
    with pytest.raises(ValueError, match="nonvacuous-certificate"):
        materializer.validate_cohort_region_view_registry(vacuous)


def test_formal_authorities_reject_semantic_cache_and_radio_summary_impostors(
    tmp_path: Path,
) -> None:
    _args, values = _inputs(tmp_path)
    semantic_cache = {
        "features": values["accepted"]["accepted_v2_e0"].half(),
        "metadata": {"schema_version": 5},
    }
    with pytest.raises(ValueError, match="fields differ"):
        materializer.validate_accepted_region_authority(semantic_cache)

    radio_summary = copy.deepcopy(values["teacher"])
    radio_summary["pair_descriptors"] = torch.zeros(3, 2560)
    radio_summary["channel_sha256"] = materializer.teacher_observation_channel_sha256(
        radio_summary
    )
    with pytest.raises(ValueError, match="tensor or row alignment"):
        materializer.validate_teacher_observation_authority(radio_summary)


@pytest.mark.parametrize("drift", ["canonical", "fingerprint", "audit"])
def test_teacher_rejects_every_accepted_sampling_drift(
    tmp_path: Path, drift: str
) -> None:
    _args, values = _inputs(tmp_path)
    accepted = values["accepted"]
    teacher = copy.deepcopy(values["teacher"])
    if drift == "canonical":
        teacher["canonical_region_indices"] = torch.tensor([0, 2])
    elif drift == "fingerprint":
        teacher["region_fingerprints"][0] = "0" * 64
    else:
        teacher["selection_audit"]["accepted_selection_audit"][
            "canonical_candidate_region_count"
        ] += 1
    with pytest.raises(ValueError, match="teacher rows differ"):
        materializer.validate_teacher_accepted_sampling_alignment(teacher, accepted)


def test_wrong_caller_sha_fails_before_any_output(tmp_path: Path) -> None:
    args, _values = _inputs(tmp_path)
    args.expected_teacher_observation_authority_sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 differs"):
        materializer.materialize(args)
    assert not Path(args.output_shard).exists()
    assert not Path(args.output_source_state_manifest).exists()
    assert not Path(args.output_teacher_manifest).exists()
    assert not Path(args.output_receipt).exists()


def test_preflight_only_validates_every_authority_without_writes(
    tmp_path: Path,
) -> None:
    args, _values = _inputs(tmp_path)
    args.preflight_only = True
    result = materializer.materialize(args)
    assert result["outputs_written"] is False
    assert result["nonvacuous_prerequisite"]["passed"] is True
    assert not Path(args.output_shard).exists()
    assert not Path(args.output_source_state_manifest).exists()
    assert not Path(args.output_teacher_manifest).exists()
    assert not Path(args.output_receipt).exists()


def _registry_sealer_scene_inputs(tmp_path: Path) -> tuple[Namespace, dict]:
    cohort_path, cohort = _cohort(tmp_path)
    scene = "scene0024_00"
    state = _state()
    state_path = tmp_path / "sealer_state.pt"
    write_torch_noclobber(state_path, state.to_payload())
    state_sha = sha256_file(state_path)

    accepted = _accepted(scene, dict(state.metadata["geometry_fingerprint"]))
    geometry_input = accepted["input_authority"]["geometry_authority"]
    geometry_input["factorized_primitive_state_file_sha256"] = state_sha
    geometry_input["factorized_field_checkpoint_file_sha256"] = "d" * 64
    geometry_input["factorized_radio_cache_file_sha256"] = "e" * 64
    accepted_path = tmp_path / "sealer_accepted.pt"
    write_torch_noclobber(accepted_path, accepted)
    accepted_sha = sha256_file(accepted_path)

    teacher = _teacher(
        scene,
        accepted,
        accepted_file_sha256=accepted_sha,
    )
    source_frame_inputs = [
        {
            "frame_id": record["frame_id"],
            "source_relative_path": record["source_relative_path"],
            "source_image_sha256": record["source_image_sha256"],
            "source_image_height": record["source_image_height"],
            "source_image_width": record["source_image_width"],
        }
        for record in teacher["view_records"]
    ]
    source = build_source_rgb_scene_authority(
        scene_id=scene,
        field_source_contract_file_sha256="a" * 64,
        field_frame_manifest_sha256="b" * 64,
        feature_frame_manifest_file_sha256="c" * 64,
        frame_records=source_frame_inputs,
    )
    source_path = tmp_path / "sealer_source_rgb.json"
    write_frozen_json(source_path, source)
    source_sha = sha256_file(source_path)
    source_by_frame = {record["frame_id"]: record for record in source["frame_records"]}
    for record in teacher["view_records"]:
        record["field_frame_authority_sha256"] = source_by_frame[record["frame_id"]][
            "field_frame_authority_sha256"
        ]
    teacher["source_rgb_scene_authority_sha256"] = source["authority_sha256"]
    teacher_input = teacher["input_authority"]
    teacher_input["source_rgb_scene_authority_file_sha256"] = source_sha
    teacher_input["source_rgb_scene_authority_content_sha256"] = source[
        "authority_sha256"
    ]
    teacher_input["factorized_primitive_state_file_sha256"] = state_sha
    teacher_input["accepted_region_authority_file_sha256"] = accepted_sha
    teacher_input["exact_marginal_responsibility_authority_file_sha256"] = accepted[
        "input_authority"
    ]["selection_authority"]["exact_marginal_responsibility_authority_file_sha256"]
    teacher["channel_sha256"] = materializer.teacher_observation_channel_sha256(teacher)
    teacher_path = tmp_path / "sealer_teacher.pt"
    write_torch_noclobber(teacher_path, teacher)

    args = Namespace(
        cohort_authority=str(cohort_path),
        expected_cohort_authority_sha256=sha256_file(cohort_path),
        accepted_region_authority=str(accepted_path),
        expected_accepted_region_authority_sha256=accepted_sha,
        factorized_state=str(state_path),
        expected_factorized_state_sha256=state_sha,
        teacher_observation_authority=str(teacher_path),
        expected_teacher_observation_authority_sha256=sha256_file(teacher_path),
        source_rgb_scene_authority=str(source_path),
        expected_source_rgb_scene_authority_sha256=source_sha,
        output=str(tmp_path / "scene_declaration.json"),
        preflight_only=False,
    )
    return args, {
        "cohort": cohort,
        "cohort_path": cohort_path,
        "accepted": accepted,
        "teacher": teacher,
        "source": source,
    }


def _synthetic_scene_declaration(
    *, cohort: dict, cohort_file_sha: str, scene_record: dict
) -> dict:
    payload = {
        "schema": registry_sealer.SCENE_DECLARATION_SCHEMA,
        "schema_version": registry_sealer.SCHEMA_VERSION,
        "contract": registry_sealer.scene_declaration_contract(),
        "contract_sha256": canonical_json_sha256(
            registry_sealer.scene_declaration_contract()
        ),
        "cohort_authority_sha256": cohort["authority_sha256"],
        "cohort_authority_file_sha256": cohort_file_sha,
        "artifact_file_sha256": {
            "accepted_region_authority": scene_record[
                "accepted_region_authority_file_sha256"
            ],
            "factorized_state": scene_record["factorized_state_file_sha256"],
            "teacher_observation_authority": scene_record[
                "teacher_observation_authority_file_sha256"
            ],
            "source_rgb_scene_authority": hashlib.sha256(
                f"{scene_record['scene_id']}-source-rgb-file".encode()
            ).hexdigest(),
        },
        "source_rgb_scene_authority_content_sha256": hashlib.sha256(
            f"{scene_record['scene_id']}-source-rgb-content".encode()
        ).hexdigest(),
        "scene_record": copy.deepcopy(scene_record),
        "source_access": registry_sealer._source_access(
            source_rgb_authority_opened=True
        ),
    }
    payload["authority_sha256"] = registry_sealer._content_sha256(payload)
    return payload


def test_scene_registry_sealer_binds_real_authority_chain_and_refuses_clobber(
    tmp_path: Path,
) -> None:
    args, values = _registry_sealer_scene_inputs(tmp_path)
    result = registry_sealer.seal_scene(args)
    assert result["outputs_written"] is True
    declaration = registry_sealer.validate_scene_declaration(
        json.loads(Path(args.output).read_text(encoding="utf-8")),
        cohort_authority=values["cohort"],
        cohort_authority_file_sha256=sha256_file(values["cohort_path"]),
    )
    assert declaration["scene_record"]["scene_id"] == "scene0024_00"
    assert declaration["scene_record"]["split"] == "source_validation"
    assert declaration["scene_record"]["eligible_overlap_teacher_row_count"] == 2
    assert declaration["artifact_file_sha256"]["source_rgb_scene_authority"] == (
        sha256_file(Path(args.source_rgb_scene_authority))
    )
    with pytest.raises(FileExistsError, match="refuses to clobber"):
        registry_sealer.seal_scene(args)


def test_scene_registry_sealer_rejects_cross_artifact_source_drift(
    tmp_path: Path,
) -> None:
    args, _values = _registry_sealer_scene_inputs(tmp_path)
    wrong_source = build_source_rgb_scene_authority(
        scene_id="scene0024_00",
        field_source_contract_file_sha256="f" * 64,
        field_frame_manifest_sha256="b" * 64,
        feature_frame_manifest_file_sha256="c" * 64,
        frame_records=[
            {
                key: record[key]
                for key in (
                    "frame_id",
                    "source_relative_path",
                    "source_image_sha256",
                    "source_image_height",
                    "source_image_width",
                )
            }
            for record in _values["source"]["frame_records"]
        ],
    )
    wrong_path = tmp_path / "wrong_source.json"
    write_frozen_json(wrong_path, wrong_source)
    args.source_rgb_scene_authority = str(wrong_path)
    args.expected_source_rgb_scene_authority_sha256 = sha256_file(wrong_path)
    with pytest.raises(ValueError, match="source RGB authority lineage differs"):
        registry_sealer.seal_scene(args)
    assert not Path(args.output).exists()


def test_global_registry_sealer_requires_exact_32_and_seals_complete_set(
    tmp_path: Path,
) -> None:
    scene_args, values = _registry_sealer_scene_inputs(tmp_path)
    registry_sealer.seal_scene(scene_args)
    real_declaration = json.loads(Path(scene_args.output).read_text(encoding="utf-8"))
    synthetic_registry = _registry(
        values["cohort"],
        sha256_file(values["cohort_path"]),
        current_scene="scene0024_00",
        accepted_sha=real_declaration["scene_record"][
            "accepted_region_authority_file_sha256"
        ],
        state_sha=real_declaration["scene_record"]["factorized_state_file_sha256"],
        teacher_sha=real_declaration["scene_record"][
            "teacher_observation_authority_file_sha256"
        ],
        accepted=values["accepted"],
        teacher=values["teacher"],
    )
    declaration_paths = []
    declaration_shas = []
    for record in synthetic_registry["scene_records"]:
        if record["scene_id"] == "scene0024_00":
            path = Path(scene_args.output)
        else:
            payload = _synthetic_scene_declaration(
                cohort=values["cohort"],
                cohort_file_sha=sha256_file(values["cohort_path"]),
                scene_record=record,
            )
            path = tmp_path / f"{record['scene_id']}.declaration.json"
            write_frozen_json(path, payload)
        declaration_paths.append(str(path))
        declaration_shas.append(sha256_file(path))
    global_args = Namespace(
        cohort_authority=str(values["cohort_path"]),
        expected_cohort_authority_sha256=sha256_file(values["cohort_path"]),
        scene_declaration=declaration_paths,
        expected_scene_declaration_sha256=declaration_shas,
        output_registry=str(tmp_path / "global_registry.json"),
        output_receipt=str(tmp_path / "global_registry_receipt.json"),
        preflight_only=False,
    )
    result = registry_sealer.seal_registry(global_args)
    assert result["scene_count"] == 32
    assert result["outputs_written"] is True
    global_value = json.loads(
        Path(global_args.output_registry).read_text(encoding="utf-8")
    )
    materializer.validate_cohort_region_view_registry(
        global_value,
        cohort_authority=values["cohort"],
        cohort_authority_file_sha256=sha256_file(values["cohort_path"]),
    )
    with pytest.raises(FileExistsError, match="refuses to clobber"):
        registry_sealer.seal_registry(global_args)

    partial_args = copy.copy(global_args)
    partial_args.scene_declaration = declaration_paths[:-1]
    partial_args.expected_scene_declaration_sha256 = declaration_shas[:-1]
    partial_args.output_registry = str(tmp_path / "partial_registry.json")
    partial_args.output_receipt = str(tmp_path / "partial_receipt.json")
    with pytest.raises(ValueError, match="exactly 32"):
        registry_sealer.seal_registry(partial_args)
    assert not Path(partial_args.output_registry).exists()
    assert not Path(partial_args.output_receipt).exists()
