#!/usr/bin/env python3
"""Seal real scene records and the complete clean 24/8 region/view registry.

This CPU-only two-stage boundary closes the gap between the per-scene
AcceptedV2/state/official-teacher authorities and the global registry consumed
by ``materialize_full_scalar_clean_training_shard``.

``scene`` validates and seals one portable scene declaration.  It binds the
caller-trusted file SHA-256 of the frozen cohort, AcceptedV2 authority,
factorized primitive state, official multi-view teacher, and clean source-RGB
authority.  ``registry`` accepts exactly the 32 independently sealed scene
declarations named by the frozen 24-train/8-validation cohort and then emits
the one global registry.  A partial set can never produce a global artifact.

Neither command opens source images, benchmark data, labels, masks, queries,
or metrics, and neither command executes a model.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces.factorized_primitive_state import (
    load_factorized_primitive_state,
)
from radio_gs.interfaces.surface_region_full_scalar_contract import (
    aggregate_surface_region_full_scalars,
)
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.scripts import train_surface_region_full_scalar_residual as trainer
from radio_gs.scripts.materialize_official_multiview_siglip2_teacher_authority import (
    validate_source_rgb_scene_authority,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    write_frozen_json,
)


SCENE_DECLARATION_SCHEMA = (
    "radio_gs.surface_region_full_scalar_clean_cohort_scene_declaration.v1"
)
REGISTRY_SEAL_RECEIPT_SCHEMA = (
    "radio_gs.surface_region_full_scalar_clean_cohort_registry_seal_receipt.v1"
)
SCHEMA_VERSION = 1


def _source_access(*, source_rgb_authority_opened: bool) -> dict[str, bool]:
    return {
        "source_rgb_authority_opened": bool(source_rgb_authority_opened),
        "source_rgb_bytes_opened": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "target_heldout_opened": False,
        "text_queries_opened": False,
        "online_model_execution": False,
        "per_scene_hyperparameters": False,
    }


def scene_declaration_contract() -> dict[str, Any]:
    return {
        "schema": SCENE_DECLARATION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "cohort": "one_exact_member_of_caller_sha_bound_clean_24_train_8_validation",
        "artifacts": (
            "caller_sha_bound_accepted_v2_exact_factorized_state_official_"
            "multiview_teacher_and_clean_source_rgb_scene_authority"
        ),
        "cross_artifact_binding": {
            "accepted_state": "exact_file_sha_geometry_field_and_radio_cache",
            "teacher_accepted": (
                "exact_file_sha_channel_fingerprints_and_sampling_selection"
            ),
            "teacher_state": "exact_file_sha",
            "teacher_source_rgb": (
                "exact_file_sha_content_sha_and_shared_frame_identity"
            ),
            "teacher_responsibility": (
                "same_exact_marginal_authority_as_accepted_selection"
            ),
        },
        "region_records": ("stable_region_ids_with_sparse_official_teacher_view_ids"),
        "partial_progress_allowed": True,
        "global_registry_claimed": False,
        "query_independent": True,
    }


def registry_seal_receipt_contract() -> dict[str, Any]:
    return {
        "schema": REGISTRY_SEAL_RECEIPT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "input": "exact_complete_set_of_32_caller_sha_bound_scene_declarations",
        "cohort": "exactly_24_source_train_and_8_source_validation_scenes",
        "output": shard.COHORT_REGISTRY_SCHEMA,
        "no_partial_global_seal": True,
        "query_independent": True,
    }


def _content_sha256(value: Mapping[str, Any]) -> str:
    content = dict(value)
    content.pop("authority_sha256", None)
    return canonical_json_sha256(content)


def _require_absent(paths: Sequence[str | Path]) -> None:
    existing = [
        str(Path(path).expanduser().resolve())
        for path in paths
        if Path(path).expanduser().exists() or Path(path).expanduser().is_symlink()
    ]
    if existing:
        raise FileExistsError(
            "clean cohort registry sealer refuses to clobber outputs: "
            + ", ".join(existing)
        )


def _load_torch_authority(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
    validator: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    expected = shard._require_sha256(expected_sha256, label=label)
    value, observed, source = load_torch_mapping(
        path,
        expected_sha256=expected,
        map_location="cpu",
        label=label,
    )
    return validator(value), {"path": str(source), "sha256": observed}


def _load_source_authority(
    path: str | Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, str]]:
    expected = shard._require_sha256(
        expected_sha256, label="clean source RGB scene authority"
    )
    value, observed, source = load_json_object(
        path,
        expected_sha256=expected,
        label="clean source RGB scene authority",
    )
    return validate_source_rgb_scene_authority(value), {
        "path": str(source),
        "sha256": observed,
    }


def _cohort_split(cohort: Mapping[str, Any], scene_id: str) -> str:
    in_train = scene_id in cohort["source_train_scene_ids"]
    in_validation = scene_id in cohort["source_validation_scene_ids"]
    if in_train == in_validation:
        raise ValueError("scene is not one unique member of the frozen clean cohort")
    return "source_train" if in_train else "source_validation"


def _validate_teacher_source_binding(
    *,
    scene_id: str,
    teacher: Mapping[str, Any],
    source: Mapping[str, Any],
    source_file_sha256: str,
) -> None:
    inputs = teacher["input_authority"]
    if (
        source["scene_id"] != scene_id
        or teacher["source_rgb_scene_authority_sha256"] != source["authority_sha256"]
        or inputs["source_rgb_scene_authority_content_sha256"]
        != source["authority_sha256"]
        or inputs["source_rgb_scene_authority_file_sha256"] != source_file_sha256
    ):
        raise ValueError("official teacher and source RGB authority lineage differs")
    source_frames = {
        str(record["frame_id"]): record for record in source["frame_records"]
    }
    if len(source_frames) != len(source["frame_records"]):
        raise ValueError("source RGB authority repeats a frame ID")
    for teacher_view in teacher["view_records"]:
        source_view = source_frames.get(str(teacher_view["frame_id"]))
        if source_view is None or any(
            teacher_view[key] != source_view[key]
            for key in (
                "source_relative_path",
                "source_image_sha256",
                "source_image_height",
                "source_image_width",
                "field_frame_authority_sha256",
            )
        ):
            raise ValueError("official teacher/source RGB frame identity differs")


def _scene_region_records(
    *, teacher: Mapping[str, Any], accepted: Mapping[str, Any]
) -> list[dict[str, Any]]:
    scene = str(accepted["scene_id"])
    view_ids = [
        shard.stable_teacher_view_id(scene, record)
        for record in teacher["view_records"]
    ]
    pair_rows = teacher["pair_region_indices"]
    pair_views = teacher["pair_view_indices"]
    return sorted(
        [
            {
                "region_fingerprint": fingerprint,
                "region_row_id": shard.stable_region_id(scene, fingerprint),
                "teacher_view_ids": [
                    view_ids[int(view)]
                    for view in pair_views[pair_rows == row].tolist()
                ],
                "eligible_overlap_teacher": True,
            }
            for row, fingerprint in enumerate(accepted["region_fingerprints"])
        ],
        key=lambda item: item["region_row_id"],
    )


def prepare_scene_declaration(args: argparse.Namespace) -> dict[str, Any]:
    cohort, cohort_file = trainer.load_cohort_authority(
        args.cohort_authority,
        expected_sha256=args.expected_cohort_authority_sha256,
    )
    accepted, accepted_file = _load_torch_authority(
        args.accepted_region_authority,
        args.expected_accepted_region_authority_sha256,
        label="AcceptedV2 canonical region authority",
        validator=shard.validate_accepted_region_authority,
    )
    teacher, teacher_file = _load_torch_authority(
        args.teacher_observation_authority,
        args.expected_teacher_observation_authority_sha256,
        label="official multi-view SigLIP2 teacher authority",
        validator=shard.validate_teacher_observation_authority,
    )
    source, source_file = _load_source_authority(
        args.source_rgb_scene_authority,
        args.expected_source_rgb_scene_authority_sha256,
    )
    state_expected = shard._require_sha256(
        args.expected_factorized_state_sha256, label="factorized primitive state"
    )
    geometry_input = accepted["input_authority"]["geometry_authority"]
    if geometry_input["factorized_primitive_state_file_sha256"] != state_expected:
        raise ValueError("AcceptedV2 and caller-bound factorized state file differ")
    state = load_factorized_primitive_state(
        args.factorized_state,
        expected_sha256=state_expected,
        expected_field_checkpoint_sha256=geometry_input[
            "factorized_field_checkpoint_file_sha256"
        ],
        expected_factorized_radio_cache_sha256=geometry_input[
            "factorized_radio_cache_file_sha256"
        ],
    )
    state_file = file_record(args.factorized_state)
    if state_file["sha256"] != state_expected:
        raise ValueError("factorized state changed after validated loading")

    scene = str(accepted["scene_id"])
    split = _cohort_split(cohort, scene)
    teacher_inputs = teacher["input_authority"]
    if (
        teacher["scene_id"] != scene
        or state.metadata["geometry_fingerprint"] != accepted["geometry_fingerprint"]
        or state.valid.shape != accepted["accepted_base_valid"].shape
        or teacher_inputs["factorized_primitive_state_file_sha256"]
        != state_file["sha256"]
        or teacher_inputs["accepted_region_authority_file_sha256"]
        != accepted_file["sha256"]
        or teacher_inputs["accepted_region_channel_sha256"]
        != canonical_json_sha256(accepted["channel_sha256"])
        or teacher_inputs["accepted_region_fingerprints_sha256"]
        != canonical_json_sha256(accepted["region_fingerprints"])
        or teacher_inputs["exact_marginal_responsibility_authority_file_sha256"]
        != accepted["input_authority"]["selection_authority"][
            "exact_marginal_responsibility_authority_file_sha256"
        ]
    ):
        raise ValueError("AcceptedV2/state/official-teacher lineage differs")
    shard.validate_teacher_accepted_sampling_alignment(teacher, accepted)
    _validate_teacher_source_binding(
        scene_id=scene,
        teacher=teacher,
        source=source,
        source_file_sha256=source_file["sha256"],
    )

    summary = aggregate_surface_region_full_scalars(
        state,
        accepted["accepted_base_valid"],
        accepted["region_rows"],
        accepted["token_mask"],
        accepted["anchor_index"],
    )
    eligible = summary.use_full_scalar_mask.bool()
    pair_counts = torch.bincount(
        teacher["pair_region_indices"], minlength=int(eligible.numel())
    )
    if (
        eligible.shape != (len(accepted["region_fingerprints"]),)
        or not bool(eligible.all())
        or not bool((pair_counts > 0).all())
    ):
        raise ValueError(
            "scene declaration requires every sampled row to have exact-state "
            "overlap and an official teacher observation"
        )
    minimum = 2 if split == "source_validation" else 1
    if int(eligible.sum()) < minimum:
        raise ValueError("scene is nonvacuous-certificate insufficient")

    teacher_model_sha = canonical_json_sha256(shard.official_teacher_model_authority())
    region_records = _scene_region_records(teacher=teacher, accepted=accepted)
    scene_record = {
        "scene_id": scene,
        "physical_space_id": trainer.canonical_physical_space_id(scene),
        "split": split,
        "accepted_region_authority_file_sha256": accepted_file["sha256"],
        "factorized_state_file_sha256": state_file["sha256"],
        "teacher_observation_authority_file_sha256": teacher_file["sha256"],
        "source_state_artifact_sha256": shard.source_state_artifact_sha256(
            accepted_region_file_sha256=accepted_file["sha256"],
            factorized_state_file_sha256=state_file["sha256"],
        ),
        "teacher_model_authority_sha256": teacher_model_sha,
        "eligible_overlap_teacher_row_count": int(eligible.sum()),
        "region_records": region_records,
    }
    declaration: dict[str, Any] = {
        "schema": SCENE_DECLARATION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "contract": scene_declaration_contract(),
        "contract_sha256": canonical_json_sha256(scene_declaration_contract()),
        "cohort_authority_sha256": cohort["authority_sha256"],
        "cohort_authority_file_sha256": cohort_file["sha256"],
        "artifact_file_sha256": {
            "accepted_region_authority": accepted_file["sha256"],
            "factorized_state": state_file["sha256"],
            "teacher_observation_authority": teacher_file["sha256"],
            "source_rgb_scene_authority": source_file["sha256"],
        },
        "source_rgb_scene_authority_content_sha256": source["authority_sha256"],
        "scene_record": scene_record,
        "source_access": _source_access(source_rgb_authority_opened=True),
    }
    declaration["authority_sha256"] = _content_sha256(declaration)
    return validate_scene_declaration(
        declaration,
        cohort_authority=cohort,
        cohort_authority_file_sha256=cohort_file["sha256"],
    )


def _validate_region_records(
    scene_record: Mapping[str, Any], *, teacher_model_sha256: str
) -> dict[str, Any]:
    required = {
        "scene_id",
        "physical_space_id",
        "split",
        "accepted_region_authority_file_sha256",
        "factorized_state_file_sha256",
        "teacher_observation_authority_file_sha256",
        "source_state_artifact_sha256",
        "teacher_model_authority_sha256",
        "eligible_overlap_teacher_row_count",
        "region_records",
    }
    if set(scene_record) != required:
        raise ValueError("clean cohort scene record fields differ")
    record = dict(scene_record)
    scene = str(record["scene_id"])
    accepted_sha = shard._require_sha256(
        record["accepted_region_authority_file_sha256"],
        label="scene declaration AcceptedV2 file",
    )
    state_sha = shard._require_sha256(
        record["factorized_state_file_sha256"],
        label="scene declaration factorized state file",
    )
    teacher_sha = shard._require_sha256(
        record["teacher_observation_authority_file_sha256"],
        label="scene declaration teacher file",
    )
    if (
        record["physical_space_id"] != trainer.canonical_physical_space_id(scene)
        or record["split"] not in {"source_train", "source_validation"}
        or record["source_state_artifact_sha256"]
        != shard.source_state_artifact_sha256(
            accepted_region_file_sha256=accepted_sha,
            factorized_state_file_sha256=state_sha,
        )
        or record["teacher_model_authority_sha256"] != teacher_model_sha256
    ):
        raise ValueError("clean cohort scene record authority differs")
    regions = record["region_records"]
    if not isinstance(regions, list) or not regions:
        raise ValueError("clean cohort scene declaration has no regions")
    frozen: list[dict[str, Any]] = []
    for raw in regions:
        if not isinstance(raw, Mapping) or set(raw) != {
            "region_fingerprint",
            "region_row_id",
            "teacher_view_ids",
            "eligible_overlap_teacher",
        }:
            raise ValueError("clean cohort scene declaration region fields differ")
        fingerprint = shard._require_sha256(
            raw["region_fingerprint"], label="scene declaration region fingerprint"
        )
        views = raw["teacher_view_ids"]
        if (
            raw["region_row_id"] != shard.stable_region_id(scene, fingerprint)
            or raw["eligible_overlap_teacher"] is not True
            or not isinstance(views, list)
            or not views
            or len(set(views)) != len(views)
            or any(
                not isinstance(view, str)
                or not view.startswith(f"{scene}:source-rgb:")
                or shard._SHA256.fullmatch(view.rsplit(":", 1)[-1]) is None
                for view in views
            )
        ):
            raise ValueError("clean cohort scene declaration stable IDs differ")
        frozen.append(
            {
                "region_fingerprint": fingerprint,
                "region_row_id": str(raw["region_row_id"]),
                "teacher_view_ids": list(views),
                "eligible_overlap_teacher": True,
            }
        )
    if (
        [item["region_row_id"] for item in frozen]
        != sorted(item["region_row_id"] for item in frozen)
        or len({item["region_row_id"] for item in frozen}) != len(frozen)
        or int(record["eligible_overlap_teacher_row_count"]) != len(frozen)
    ):
        raise ValueError("clean cohort scene declaration region order/count differs")
    minimum = 2 if record["split"] == "source_validation" else 1
    if len(frozen) < minimum:
        raise ValueError("scene declaration is nonvacuous-certificate insufficient")
    return {
        **record,
        "accepted_region_authority_file_sha256": accepted_sha,
        "factorized_state_file_sha256": state_sha,
        "teacher_observation_authority_file_sha256": teacher_sha,
        "region_records": frozen,
    }


def validate_scene_declaration(
    value: object,
    *,
    cohort_authority: Mapping[str, Any] | None = None,
    cohort_authority_file_sha256: str = "",
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("clean cohort scene declaration must be a mapping")
    declaration = dict(value)
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "cohort_authority_sha256",
        "cohort_authority_file_sha256",
        "artifact_file_sha256",
        "source_rgb_scene_authority_content_sha256",
        "scene_record",
        "source_access",
        "authority_sha256",
    }
    if (
        set(declaration) != required
        or declaration.get("schema") != SCENE_DECLARATION_SCHEMA
        or declaration.get("schema_version") != SCHEMA_VERSION
        or declaration.get("contract") != scene_declaration_contract()
        or declaration.get("contract_sha256")
        != canonical_json_sha256(scene_declaration_contract())
        or declaration.get("source_access")
        != _source_access(source_rgb_authority_opened=True)
    ):
        raise ValueError("clean cohort scene declaration contract differs")
    cohort_content_sha = shard._require_sha256(
        declaration["cohort_authority_sha256"],
        label="scene declaration cohort content",
    )
    cohort_file_sha = shard._require_sha256(
        declaration["cohort_authority_file_sha256"],
        label="scene declaration cohort file",
    )
    source_content_sha = shard._require_sha256(
        declaration["source_rgb_scene_authority_content_sha256"],
        label="scene declaration source RGB content",
    )
    artifacts = declaration["artifact_file_sha256"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "accepted_region_authority",
        "factorized_state",
        "teacher_observation_authority",
        "source_rgb_scene_authority",
    }:
        raise ValueError("clean cohort scene declaration artifact fields differ")
    artifact_shas = {
        key: shard._require_sha256(value, label=f"scene declaration {key}")
        for key, value in artifacts.items()
    }
    teacher_model_sha = canonical_json_sha256(shard.official_teacher_model_authority())
    scene_record = declaration["scene_record"]
    if not isinstance(scene_record, Mapping):
        raise ValueError("clean cohort scene declaration record must be a mapping")
    frozen_record = _validate_region_records(
        scene_record, teacher_model_sha256=teacher_model_sha
    )
    if (
        artifact_shas["accepted_region_authority"]
        != frozen_record["accepted_region_authority_file_sha256"]
        or artifact_shas["factorized_state"]
        != frozen_record["factorized_state_file_sha256"]
        or artifact_shas["teacher_observation_authority"]
        != frozen_record["teacher_observation_authority_file_sha256"]
    ):
        raise ValueError("clean cohort scene declaration artifact binding differs")
    del source_content_sha  # Validated and retained verbatim in the declaration.
    if cohort_authority is not None:
        cohort = trainer.validate_cohort_authority_payload(cohort_authority)
        expected_split = _cohort_split(cohort, frozen_record["scene_id"])
        if (
            frozen_record["split"] != expected_split
            or cohort_content_sha != cohort["authority_sha256"]
            or cohort_file_sha
            != shard._require_sha256(
                cohort_authority_file_sha256, label="cohort authority file"
            )
        ):
            raise ValueError("scene declaration and frozen cohort authority differ")
    if declaration["authority_sha256"] != _content_sha256(declaration):
        raise ValueError("clean cohort scene declaration content SHA-256 differs")
    return {
        **declaration,
        "artifact_file_sha256": artifact_shas,
        "scene_record": frozen_record,
    }


def seal_scene(args: argparse.Namespace) -> dict[str, Any]:
    if not bool(args.preflight_only):
        _require_absent([args.output])
    declaration = prepare_scene_declaration(args)
    result = {
        "status": "ready" if bool(args.preflight_only) else "sealed",
        "scene_id": declaration["scene_record"]["scene_id"],
        "split": declaration["scene_record"]["split"],
        "eligible_overlap_teacher_row_count": declaration["scene_record"][
            "eligible_overlap_teacher_row_count"
        ],
        "authority_sha256": declaration["authority_sha256"],
        "outputs_written": False,
    }
    if bool(args.preflight_only):
        return result
    write_frozen_json(args.output, declaration)
    return {**result, "output": file_record(args.output), "outputs_written": True}


def _paired_scene_inputs(args: argparse.Namespace) -> list[tuple[str, str]]:
    paths = list(args.scene_declaration)
    shas = list(args.expected_scene_declaration_sha256)
    required = trainer.TRAIN_SCENE_COUNT + trainer.VALIDATION_SCENE_COUNT
    if len(paths) != len(shas):
        raise ValueError("scene declaration paths and expected SHA-256 counts differ")
    if len(paths) != required:
        raise ValueError(
            f"global registry requires exactly {required} scene declarations"
        )
    return list(zip(paths, shas))


def prepare_registry(args: argparse.Namespace) -> dict[str, Any]:
    cohort, cohort_file = trainer.load_cohort_authority(
        args.cohort_authority,
        expected_sha256=args.expected_cohort_authority_sha256,
    )
    records: list[dict[str, Any]] = []
    declaration_files: list[dict[str, str]] = []
    for path, expected_sha in _paired_scene_inputs(args):
        expected = shard._require_sha256(
            expected_sha, label="clean cohort scene declaration"
        )
        value, observed, source = load_json_object(
            path,
            expected_sha256=expected,
            label="clean cohort scene declaration",
        )
        declaration = validate_scene_declaration(
            value,
            cohort_authority=cohort,
            cohort_authority_file_sha256=cohort_file["sha256"],
        )
        records.append(declaration["scene_record"])
        declaration_files.append(
            {
                "scene_id": declaration["scene_record"]["scene_id"],
                "path": str(source),
                "sha256": observed,
                "authority_sha256": declaration["authority_sha256"],
            }
        )
    expected_scenes = sorted(
        cohort["source_train_scene_ids"] + cohort["source_validation_scene_ids"]
    )
    observed_scenes = sorted(str(record["scene_id"]) for record in records)
    if observed_scenes != expected_scenes or len(set(observed_scenes)) != len(records):
        raise ValueError("scene declarations do not cover the exact frozen cohort")
    registry = shard.build_cohort_region_view_registry(
        cohort_authority=cohort,
        cohort_authority_file_sha256=cohort_file["sha256"],
        scene_records=records,
    )
    declaration_files.sort(key=lambda item: item["scene_id"])
    receipt: dict[str, Any] = {
        "schema": REGISTRY_SEAL_RECEIPT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "contract": registry_seal_receipt_contract(),
        "contract_sha256": canonical_json_sha256(registry_seal_receipt_contract()),
        "cohort_authority": {
            "file_sha256": cohort_file["sha256"],
            "authority_sha256": cohort["authority_sha256"],
        },
        "scene_declaration_files": declaration_files,
        "registry_authority_sha256": registry["authority_sha256"],
        "source_access": _source_access(source_rgb_authority_opened=False),
    }
    receipt["authority_sha256"] = _content_sha256(receipt)
    return {"registry": registry, "receipt": receipt}


def seal_registry(args: argparse.Namespace) -> dict[str, Any]:
    if not bool(args.preflight_only):
        _require_absent([args.output_registry, args.output_receipt])
    prepared = prepare_registry(args)
    registry = prepared["registry"]
    result = {
        "status": "ready" if bool(args.preflight_only) else "sealed",
        "scene_count": len(registry["scene_records"]),
        "train_scene_count": trainer.TRAIN_SCENE_COUNT,
        "validation_scene_count": trainer.VALIDATION_SCENE_COUNT,
        "registry_authority_sha256": registry["authority_sha256"],
        "outputs_written": False,
    }
    if bool(args.preflight_only):
        return result
    write_frozen_json(args.output_registry, registry)
    write_frozen_json(args.output_receipt, prepared["receipt"])
    return {
        **result,
        "registry": file_record(args.output_registry),
        "receipt": file_record(args.output_receipt),
        "outputs_written": True,
    }


def _cohort_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cohort-authority", required=True)
    parser.add_argument("--expected-cohort-authority-sha256", required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    scene = commands.add_parser("scene", help="seal one strict scene declaration")
    _cohort_arguments(scene)
    scene.add_argument("--accepted-region-authority", required=True)
    scene.add_argument("--expected-accepted-region-authority-sha256", required=True)
    scene.add_argument("--factorized-state", required=True)
    scene.add_argument("--expected-factorized-state-sha256", required=True)
    scene.add_argument("--teacher-observation-authority", required=True)
    scene.add_argument("--expected-teacher-observation-authority-sha256", required=True)
    scene.add_argument("--source-rgb-scene-authority", required=True)
    scene.add_argument("--expected-source-rgb-scene-authority-sha256", required=True)
    scene.add_argument("--output", required=True)
    scene.add_argument("--preflight-only", action="store_true")
    scene.set_defaults(handler=seal_scene)

    registry = commands.add_parser(
        "registry", help="seal the exact complete 24/8 global registry"
    )
    _cohort_arguments(registry)
    registry.add_argument("--scene-declaration", action="append", required=True)
    registry.add_argument(
        "--expected-scene-declaration-sha256", action="append", required=True
    )
    registry.add_argument("--output-registry", required=True)
    registry.add_argument("--output-receipt", required=True)
    registry.add_argument("--preflight-only", action="store_true")
    registry.set_defaults(handler=seal_registry)

    args = parser.parse_args()
    print(json.dumps(args.handler(args), indent=2))


if __name__ == "__main__":
    main()
