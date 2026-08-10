"""Formal target contract for the source-promoted native-V3 relation head.

The source promotion is validated before any target record is opened.  Target
use is query-free and keeps the frozen V2 feature/inference chain as a bitwise
fallback whenever an AcceptedV2 canonical anchor lacks exact factorized state.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from radio_gs.scripts import infer_region_comembership_v2 as parent_inference
from radio_gs.scripts import materialize_region_comembership_features_v2 as parent_feature
from radio_gs.scripts import train_source_region_comembership_native_v3 as trainer
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
)


TARGET_EXECUTION_SCHEMA = (
    "radio_gs.region_comembership_native_v3_target_execution_authority.v1"
)
TARGET_EXECUTION_STATUS = (
    "authorized_after_source_only_native_v3_exact4train_2validation_promotion"
)
FEATURE_SCHEMA = "radio_gs.region_comembership_feature_authority.native_v3"
INFERENCE_SCHEMA = "radio_gs.region_comembership_inference_authority.native_v3"
SCHEMA_VERSION = 1
TARGET_INPUT_NAMES = (
    "parent_v2_feature_authority",
    "parent_v2_inference_authority",
    "accepted_v2",
    "factorized_state",
)
ROOT = Path(__file__).resolve().parents[2]
TARGET_IMPLEMENTATION_PATHS = {
    "formal_interface": Path(__file__).resolve(),
    "authority_builder": ROOT
    / "radio_gs/scripts/build_region_comembership_native_v3_target_execution_authority.py",
    "feature_materializer": ROOT
    / "radio_gs/scripts/materialize_region_comembership_features_native_v3.py",
    "inference_runner": ROOT
    / "radio_gs/scripts/infer_region_comembership_native_v3.py",
    "native_relation_interface": ROOT
    / "radio_gs/interfaces/factorized_native_region_relation.py",
    "native_model": ROOT / "radio_gs/models/region_comembership_native_v3.py",
    "source_trainer": Path(trainer.__file__).resolve(),
    "parent_v2_feature_validator": Path(parent_feature.__file__).resolve(),
    "parent_v2_inference_validator": Path(parent_inference.__file__).resolve(),
}


def access_audit(*, target_opened: bool) -> dict[str, bool]:
    return {
        "source_promotion_validated_before_target_files": True,
        "source_instance_labels_opened": False,
        "target_feature_authorities_opened": bool(target_opened),
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "text_queries_opened": False,
        "target_metrics_computed": False,
    }


def fallback_contract() -> dict[str, Any]:
    return {
        "anchor_eligibility": (
            "AcceptedV2 canonical anchor row has exact FactorizedPrimitiveState.valid"
        ),
        "native_pair_eligibility": "both endpoint anchors are exact",
        "ineligible_native_feature_sentinel": "nine exact zeros never evaluated",
        "ineligible_pair_probability": "bitwise parent V2 pair probability",
        "ineligible_pair_edge_decision": "bitwise parent V2 accepted-edge decision",
        "alternate_anchor_substitution": False,
        "legacy_v2_default_changed": False,
    }


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path = str(value["path"])
    digest = str(value["sha256"])
    if (
        not Path(path).is_absolute()
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} file record differs")
    return {"path": path, "sha256": digest}


def _canonical_output(value: object, *, label: str) -> str:
    raw = str(value)
    resolved = str(Path(raw).expanduser().resolve())
    if raw != resolved:
        raise ValueError(f"{label} must be a canonical absolute path")
    return resolved


def validate_source_promotion(
    result_record: object, checkpoint_record: object
) -> dict[str, Any]:
    """Validate the complete promoted source chain without touching target files."""

    result_bound = _record(result_record, label="native V3 source result")
    checkpoint_bound = _record(checkpoint_record, label="native V3 checkpoint")
    result_raw, result_sha, result_path = load_json_object(
        result_bound["path"],
        expected_sha256=result_bound["sha256"],
        label="native V3 source result",
    )
    checkpoint_raw, checkpoint_sha, checkpoint_path = load_torch_mapping(
        checkpoint_bound["path"],
        expected_sha256=checkpoint_bound["sha256"],
        map_location="cpu",
        label="native V3 promoted checkpoint",
    )
    if (
        result_bound != {"path": str(result_path), "sha256": result_sha}
        or checkpoint_bound
        != {"path": str(checkpoint_path), "sha256": checkpoint_sha}
    ):
        raise ValueError("native V3 source records differ")
    checkpoint = trainer.validate_checkpoint(checkpoint_raw)
    required_result = {
        "schema",
        "schema_version",
        "status",
        "checkpoint",
        "selected_validation",
        "singleton_validation",
        "promotion_gate",
        "exact_candidate_count",
        "exact_candidates",
        "calibration_by_epoch",
        "proxy_audit",
        "history",
        "source_access",
        "target_execution_performed",
    }
    if (
        not isinstance(result_raw, Mapping)
        or set(result_raw) != required_result
        or result_raw.get("schema") != trainer.RESULT_SCHEMA
        or result_raw.get("schema_version") != 1
        or result_raw.get("status")
        != "source_only_native_v3_exact4train_2validation_complete"
        or result_raw.get("checkpoint") != checkpoint_bound
        or result_raw.get("selected_validation")
        != checkpoint["selected_validation"]
        or result_raw.get("singleton_validation")
        != checkpoint["singleton_validation"]
        or result_raw.get("promotion_gate") != checkpoint["promotion_gate"]
        or result_raw.get("source_access") != trainer.source_access()
        or result_raw.get("target_execution_performed") is not False
        or checkpoint["promotion_gate"].get("passed") is not True
        or int(checkpoint["selected_epoch"]) <= 0
        or any(
            checkpoint["promotion_gate"].get(name) is not True
            for name in (
                "topology_strictly_exceeds_singleton",
                "iou_strictly_exceeds_singleton",
                "f1_strictly_exceeds_singleton",
                "macro_brier_strictly_improves_epoch_zero",
                "macro_log_loss_strictly_improves_epoch_zero",
                "every_validation_scene_brier_non_regression",
            )
        )
    ):
        raise ValueError("native V3 source promotion differs")

    execution_record = checkpoint["execution_authority"]
    execution_path = validate_file_record(
        execution_record, label="native V3 source execution authority"
    )
    execution_raw, _, _ = load_json_object(
        execution_path,
        expected_sha256=execution_record["sha256"],
        label="native V3 source execution authority",
    )
    execution = trainer.validate_execution_authority(execution_raw)
    expected_paths = {
        "implementation": Path(trainer.__file__).resolve(),
        "model_implementation": ROOT
        / "radio_gs/models/region_comembership_native_v3.py",
        "source_materializer_implementation": ROOT
        / "radio_gs/scripts/materialize_source_region_comembership_native_v3.py",
        "native_interface_implementation": ROOT
        / "radio_gs/interfaces/factorized_native_region_relation.py",
        "preregistration": ROOT / trainer.PREREGISTRATION,
        "parent_v2_source_result": ROOT / trainer.PARENT_V2_SOURCE_RESULT,
    }
    for name, expected in expected_paths.items():
        if (
            validate_file_record(execution[name], label=f"native V3 source {name}")
            != expected.resolve()
        ):
            raise ValueError(f"native V3 source execution binds another {name}")
    for split in ("source_train", "source_validation"):
        for row in execution[split]:
            validate_file_record(
                row["authority"], label=f"native V3 source {split} authority"
            )
    return {
        "result": dict(result_raw),
        "result_record": result_bound,
        "checkpoint": checkpoint,
        "checkpoint_record": checkpoint_bound,
        "execution": execution,
    }


def validate_target_execution_authority(
    path: str | Path,
    *,
    expected_sha256: str,
    scene_id: str,
    expected_feature_output: str | Path | None = None,
    expected_inference_output: str | Path | None = None,
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="native V3 target execution authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "source_result",
        "promoted_checkpoint",
        "builder_implementation",
        "implementation_dependencies",
        "target_inputs",
        "target_feature_output",
        "target_inference_output",
        "feature_materialization_authorized",
        "checkpoint_inference_authorized",
        "query_execution_authorized",
        "metric_execution_authorized",
        "fallback_contract",
        "access_audit",
    }
    authority = dict(raw)
    if (
        set(authority) != required
        or authority.get("schema") != TARGET_EXECUTION_SCHEMA
        or authority.get("schema_version") != SCHEMA_VERSION
        or authority.get("status") != TARGET_EXECUTION_STATUS
        or authority.get("scene_id") != str(scene_id)
        or authority.get("feature_materialization_authorized") is not True
        or authority.get("checkpoint_inference_authorized") is not True
        or authority.get("query_execution_authorized") is not False
        or authority.get("metric_execution_authorized") is not False
        or authority.get("fallback_contract") != fallback_contract()
        or authority.get("access_audit") != access_audit(target_opened=True)
    ):
        raise ValueError("native V3 target execution header differs")

    # This call must stay before every target record validation.
    source_gate = validate_source_promotion(
        authority["source_result"], authority["promoted_checkpoint"]
    )
    builder = validate_file_record(
        authority["builder_implementation"], label="native V3 target builder"
    )
    if builder != TARGET_IMPLEMENTATION_PATHS["authority_builder"].resolve():
        raise ValueError("native V3 target authority binds another builder")
    dependencies = authority.get("implementation_dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(
        TARGET_IMPLEMENTATION_PATHS
    ):
        raise ValueError("native V3 target dependency set differs")
    for name, expected in TARGET_IMPLEMENTATION_PATHS.items():
        if (
            validate_file_record(dependencies[name], label=f"native V3 target {name}")
            != expected.resolve()
        ):
            raise ValueError(f"native V3 target dependency differs: {name}")

    inputs = authority.get("target_inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(TARGET_INPUT_NAMES):
        raise ValueError("native V3 target input set differs")
    records: dict[str, dict[str, str]] = {}
    for name in TARGET_INPUT_NAMES:
        record = _record(inputs[name], label=f"native V3 target {name}")
        verified = validate_file_record(record, label=f"native V3 target {name}")
        records[name] = {"path": str(verified), "sha256": record["sha256"]}
    parent_feature_raw, _, _ = load_torch_mapping(
        records["parent_v2_feature_authority"]["path"],
        expected_sha256=records["parent_v2_feature_authority"]["sha256"],
        map_location="cpu",
        label="native V3 parent V2 feature authority",
    )
    parent_inference_raw, _, _ = load_torch_mapping(
        records["parent_v2_inference_authority"]["path"],
        expected_sha256=records["parent_v2_inference_authority"]["sha256"],
        map_location="cpu",
        label="native V3 parent V2 inference authority",
    )
    feature = parent_feature.validate_feature_authority(parent_feature_raw)
    inference = parent_inference.validate_inference_authority(parent_inference_raw)
    if (
        feature["domain"] != "target"
        or inference["domain"] != "target"
        or feature["scene_id"] != str(scene_id)
        or inference["scene_id"] != str(scene_id)
        or inference["feature_authority"]
        != records["parent_v2_feature_authority"]
        or feature["input_authority"]["accepted_v2"] != records["accepted_v2"]
        or feature["input_authority"]["factorized_state"]
        != records["factorized_state"]
        or feature["region_fingerprints"] != inference["region_fingerprints"]
        or not torch.equal(
            feature["canonical_region_indices"],
            inference["canonical_region_indices"],
        )
        or not torch.equal(feature["pair_indices"], inference["pair_indices"])
    ):
        raise ValueError("native V3 target parent V2 chain differs")
    feature_output = _canonical_output(
        authority["target_feature_output"], label="native V3 feature output"
    )
    inference_output = _canonical_output(
        authority["target_inference_output"], label="native V3 inference output"
    )
    if feature_output == inference_output:
        raise ValueError("native V3 target outputs must differ")
    if expected_feature_output is not None and feature_output != str(
        Path(expected_feature_output).expanduser().resolve()
    ):
        raise ValueError("native V3 target feature output differs")
    if expected_inference_output is not None and inference_output != str(
        Path(expected_inference_output).expanduser().resolve()
    ):
        raise ValueError("native V3 target inference output differs")
    authority["verified_record"] = {"path": str(source), "sha256": digest}
    authority["verified_source_gate"] = source_gate
    authority["verified_target_inputs"] = records
    authority["verified_parent_feature"] = feature
    authority["verified_parent_inference"] = inference
    return authority


__all__ = [
    "FEATURE_SCHEMA",
    "INFERENCE_SCHEMA",
    "SCHEMA_VERSION",
    "TARGET_EXECUTION_SCHEMA",
    "TARGET_EXECUTION_STATUS",
    "TARGET_IMPLEMENTATION_PATHS",
    "TARGET_INPUT_NAMES",
    "access_audit",
    "fallback_contract",
    "validate_source_promotion",
    "validate_target_execution_authority",
]
