"""Strict formal-authority validation for RegionCoMembershipV2 target use.

The target execution authority is intentionally separate from V1.  Validation
of the source-only V2 result and its promoted checkpoint happens before any
target input file record is opened.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.models.region_comembership_v2 import (
    HIDDEN_DIMENSIONS,
    PAIR_FEATURE_NAMES,
    RegionCoMembershipV2,
)
from radio_gs.scripts import train_source_region_comembership_v2 as trainer
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
)


TARGET_EXECUTION_SCHEMA = (
    "radio_gs.region_comembership_v2_target_execution_authority.v1"
)
TARGET_EXECUTION_STATUS = (
    "authorized_after_source_only_v2_4train_2validation_promotion"
)
RESULT_SCHEMA = "radio_gs.region_comembership_v2_pilot_result.v1"
TARGET_INPUT_NAMES = (
    "accepted_v2",
    "typed_context",
    "support_graph",
    "factorized_state",
    "capability_descriptor",
)
METRIC_NAMES = (
    "iou",
    "f1",
    "contamination",
    "giant_excess",
    "selected_units",
    "selected_regions",
    "topology_score",
)
STATE_SHAPES = {
    "feature_median": (len(PAIR_FEATURE_NAMES),),
    "feature_robust_scale": (len(PAIR_FEATURE_NAMES),),
    "network.0.weight": (HIDDEN_DIMENSIONS[0], len(PAIR_FEATURE_NAMES)),
    "network.0.bias": (HIDDEN_DIMENSIONS[0],),
    "network.2.weight": (HIDDEN_DIMENSIONS[1], HIDDEN_DIMENSIONS[0]),
    "network.2.bias": (HIDDEN_DIMENSIONS[1],),
    "network.4.weight": (1, HIDDEN_DIMENSIONS[1]),
    "network.4.bias": (1,),
}


def _state_sha(state: Mapping[str, torch.Tensor]) -> str:
    return canonical_json_sha256(
        {name: tensor_sha256(value) for name, value in sorted(state.items())}
    )


def _finite(value: object, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _canonical_output_path(value: object, *, label: str) -> str:
    raw = str(value)
    resolved = str(Path(raw).expanduser().resolve())
    if raw != resolved:
        raise ValueError(f"{label} must be an absolute canonical path")
    return resolved


def _validate_metric(value: object, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(METRIC_NAMES):
        raise ValueError(f"{label} metric fields differ")
    metric = {name: _finite(value[name], label=f"{label} {name}") for name in METRIC_NAMES}
    if (
        not 0.0 <= metric["iou"] <= 1.0
        or not 0.0 <= metric["f1"] <= 1.0
        or not 0.0 <= metric["contamination"] <= 1.0
        or metric["giant_excess"] < 0.0
        or metric["selected_units"] <= 0.0
        or metric["selected_regions"] <= 0.0
        or not math.isclose(
            metric["topology_score"],
            metric["iou"] - metric["contamination"] - metric["giant_excess"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"{label} metric values differ")
    return metric


def validate_selection_candidate(value: object, *, label: str) -> dict[str, Any]:
    required = {
        "epoch",
        "method",
        "maximum_regions",
        "threshold",
        "scene_macro",
        "per_scene",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{label} selection fields differ")
    candidate = dict(value)
    epoch = candidate.get("epoch")
    maximum = candidate.get("maximum_regions")
    threshold = _finite(candidate.get("threshold"), label=f"{label} threshold")
    if (
        not isinstance(epoch, int)
        or epoch not in trainer.SNAPSHOT_EPOCHS
        or candidate.get("method") not in trainer.METHODS
        or not isinstance(maximum, int)
        or maximum not in trainer.MAXIMUM_REGIONS
        or threshold not in trainer.THRESHOLDS
    ):
        raise ValueError(f"{label} selection rule differs")
    candidate["scene_macro"] = _validate_metric(
        candidate["scene_macro"], label=f"{label} scene macro"
    )
    per_scene = candidate.get("per_scene")
    if not isinstance(per_scene, Mapping) or set(per_scene) != set(
        trainer.VALIDATION_SCENES
    ):
        raise ValueError(f"{label} validation scene axis differs")
    candidate["per_scene"] = {
        scene: _validate_metric(per_scene[scene], label=f"{label} {scene}")
        for scene in trainer.VALIDATION_SCENES
    }
    recomputed = {
        name: sum(candidate["per_scene"][scene][name] for scene in trainer.VALIDATION_SCENES)
        / len(trainer.VALIDATION_SCENES)
        for name in METRIC_NAMES
    }
    if any(
        not math.isclose(
            candidate["scene_macro"][name], recomputed[name], rel_tol=0.0, abs_tol=1e-12
        )
        for name in METRIC_NAMES
    ):
        raise ValueError(f"{label} scene macro does not match its scene axis")
    return candidate


def _validate_promotion(
    value: object,
    *,
    selected: Mapping[str, Any],
    singleton: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "selected_epoch_positive",
        "topology_strictly_exceeds_singleton",
        "iou_strictly_exceeds_singleton",
        "f1_strictly_exceeds_singleton",
        "singleton",
        "selected",
        "passed",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("V2 promotion fields differ")
    promotion = dict(value)
    expected_flags = {
        "selected_epoch_positive": int(selected["epoch"]) > 0,
        "topology_strictly_exceeds_singleton": selected["scene_macro"][
            "topology_score"
        ]
        > singleton["scene_macro"]["topology_score"],
        "iou_strictly_exceeds_singleton": selected["scene_macro"]["iou"]
        > singleton["scene_macro"]["iou"],
        "f1_strictly_exceeds_singleton": selected["scene_macro"]["f1"]
        > singleton["scene_macro"]["f1"],
    }
    if (
        promotion.get("singleton") != singleton["scene_macro"]
        or promotion.get("selected") != selected["scene_macro"]
        or any(promotion.get(name) is not expected for name, expected in expected_flags.items())
        or promotion.get("passed") is not all(expected_flags.values())
    ):
        raise ValueError("V2 promotion does not match exact validation")
    return promotion


def _selection_key(value: Mapping[str, Any]) -> tuple[float, ...]:
    """Mirror the preregistered trainer's exact global selection order."""

    metric = value["scene_macro"]
    return (
        float(metric["topology_score"]),
        float(metric["iou"]),
        float(metric["f1"]),
        -float(metric["contamination"]),
        -float(metric["giant_excess"]),
        -float(value["maximum_regions"]),
        -float(value["threshold"]),
        -float(value["epoch"]),
        -float(trainer.METHODS.index(str(value["method"]))),
    )


def _validate_training_execution(record: object) -> dict[str, Any]:
    path = validate_file_record(record, label="V2 source training execution authority")
    raw, _, _ = load_json_object(
        path,
        expected_sha256=record["sha256"],
        label="V2 source training execution authority",
    )
    authority = trainer.validate_execution_authority(raw)
    implementation = validate_file_record(
        authority["implementation"], label="V2 trainer implementation"
    )
    if implementation != Path(trainer.__file__).resolve():
        raise ValueError("V2 checkpoint source execution binds another trainer")
    for name in ("preregistration", "efficiency_addendum"):
        validate_file_record(authority[name], label=f"V2 source execution {name}")
    for split in ("source_train", "source_validation"):
        for row in authority[split]:
            validate_file_record(row["authority"], label=f"V2 source execution {split}")
    return authority


def validate_checkpoint(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "training_contract",
        "training_contract_sha256",
        "execution_authority",
        "feature_names",
        "normalization",
        "model_state_dict",
        "model_state_dict_sha256",
        "selected_epoch",
        "selected_rule",
        "selected_validation",
        "singleton_validation",
        "promotion_gate",
        "source_access",
        "target_execution_performed",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("RegionCoMembership V2 checkpoint fields differ")
    checkpoint = dict(value)
    contract = trainer.training_contract()
    state = checkpoint.get("model_state_dict")
    normalization = checkpoint.get("normalization")
    if (
        checkpoint.get("schema") != trainer.CHECKPOINT_SCHEMA
        or checkpoint.get("schema_version") != 1
        or checkpoint.get("training_contract") != contract
        or checkpoint.get("training_contract_sha256")
        != canonical_json_sha256(contract)
        or checkpoint.get("feature_names") != list(PAIR_FEATURE_NAMES)
        or checkpoint.get("source_access") != trainer.source_access()
        or checkpoint.get("target_execution_performed") is not False
        or not isinstance(normalization, Mapping)
        or set(normalization) != {"median", "robust_scale"}
        or not isinstance(state, Mapping)
        or set(state) != set(STATE_SHAPES)
        or checkpoint.get("model_state_dict_sha256") != _state_sha(state)
    ):
        raise ValueError("RegionCoMembership V2 checkpoint identity differs")
    _validate_training_execution(checkpoint["execution_authority"])
    for name, shape in STATE_SHAPES.items():
        tensor = torch.as_tensor(state[name])
        if (
            tensor.dtype != torch.float32
            or tuple(tensor.shape) != shape
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(f"RegionCoMembership V2 checkpoint state differs: {name}")
    median = torch.as_tensor(normalization["median"])
    scale = torch.as_tensor(normalization["robust_scale"])
    if (
        not torch.equal(median, state["feature_median"])
        or not torch.equal(scale, state["feature_robust_scale"])
        or bool((scale <= 0).any())
    ):
        raise ValueError("RegionCoMembership V2 normalization differs from state")
    selected = validate_selection_candidate(
        checkpoint["selected_validation"], label="selected validation"
    )
    singleton = validate_selection_candidate(
        checkpoint["singleton_validation"], label="singleton validation"
    )
    expected_rule = {
        name: selected[name] for name in ("method", "maximum_regions", "threshold")
    }
    if (
        checkpoint.get("selected_epoch") != selected["epoch"]
        or checkpoint.get("selected_rule") != expected_rule
        or singleton["epoch"] != 0
        or singleton["method"] != trainer.METHODS[0]
        or singleton["maximum_regions"] != 1
        or singleton["threshold"] != max(trainer.THRESHOLDS)
    ):
        raise ValueError("RegionCoMembership V2 checkpoint selection differs")
    checkpoint["promotion_gate"] = _validate_promotion(
        checkpoint["promotion_gate"], selected=selected, singleton=singleton
    )
    return checkpoint


def validate_result(value: object, *, require_promotion: bool) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "status",
        "checkpoint",
        "selected_validation",
        "singleton_validation",
        "promotion_gate",
        "exact_candidate_count",
        "exact_candidates",
        "proxy_audit",
        "history",
        "source_access",
        "target_execution_performed",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("RegionCoMembership V2 result fields differ")
    result = dict(value)
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("schema_version") != 1
        or result.get("status") != "source_only_v2_4train_2validation_complete"
        or result.get("source_access") != trainer.source_access()
        or result.get("target_execution_performed") is not False
    ):
        raise ValueError("RegionCoMembership V2 result identity differs")
    validate_file_record(result["checkpoint"], label="promoted V2 checkpoint")
    selected = validate_selection_candidate(
        result["selected_validation"], label="result selected validation"
    )
    singleton = validate_selection_candidate(
        result["singleton_validation"], label="result singleton validation"
    )
    promotion = _validate_promotion(
        result["promotion_gate"], selected=selected, singleton=singleton
    )
    candidates = result.get("exact_candidates")
    if not isinstance(candidates, list) or int(result.get("exact_candidate_count", -1)) != len(candidates):
        raise ValueError("RegionCoMembership V2 exact candidate count differs")
    normalized_candidates = [
        validate_selection_candidate(row, label=f"exact candidate {index}")
        for index, row in enumerate(candidates)
    ]
    if (
        selected not in normalized_candidates
        or singleton not in normalized_candidates
        or selected != max(normalized_candidates, key=_selection_key)
    ):
        raise ValueError(
            "RegionCoMembership V2 selected candidate is absent or not the frozen global maximum"
        )
    history = result.get("history")
    if not isinstance(history, list) or len(history) != trainer.EPOCHS:
        raise ValueError("RegionCoMembership V2 training history differs")
    for expected_epoch, row in enumerate(history, start=1):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"epoch", "train_scene_macro_balanced_bce"}
            or row.get("epoch") != expected_epoch
        ):
            raise ValueError("RegionCoMembership V2 training history row differs")
        _finite(row["train_scene_macro_balanced_bce"], label="V2 training loss")
    if not isinstance(result.get("proxy_audit"), Mapping) or set(result["proxy_audit"]) != {
        str(epoch) for epoch in trainer.SNAPSHOT_EPOCHS
    }:
        raise ValueError("RegionCoMembership V2 proxy audit axis differs")
    if require_promotion and (
        promotion["passed"] is not True or int(selected["epoch"]) <= 0
    ):
        raise ValueError("RegionCoMembership V2 source promotion did not pass")
    return result


def validate_target_execution_authority(
    path: str | Path,
    *,
    expected_sha256: str,
    scene_id: str,
    expected_feature_output: str | Path | None = None,
    expected_inference_output: str | Path | None = None,
) -> dict[str, Any]:
    """Validate V2 promotion before opening any target input record."""

    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="RegionCoMembership V2 target execution authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "preregistration",
        "efficiency_addendum",
        "four_plus_two_result",
        "promoted_checkpoint",
        "target_feature_inputs",
        "target_feature_output",
        "target_inference_output",
        "target_feature_materialization_authorized",
        "target_checkpoint_inference_authorized",
        "target_metric_authorized",
        "access_audit",
    }
    authority = dict(raw)
    if (
        set(authority) != required
        or authority.get("schema") != TARGET_EXECUTION_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != TARGET_EXECUTION_STATUS
        or authority.get("scene_id") != str(scene_id)
        or authority.get("target_feature_materialization_authorized") is not True
        or authority.get("target_checkpoint_inference_authorized") is not True
        or authority.get("target_metric_authorized") is not False
        or authority.get("access_audit")
        != {
            "benchmark_images_opened": False,
            "benchmark_labels_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_queries_opened": False,
            "target_metrics_computed": False,
        }
    ):
        raise ValueError("RegionCoMembership V2 target execution header differs")
    root = Path(trainer.__file__).resolve().parents[2]
    for name, expected in (
        ("preregistration", root / trainer.PREREGISTRATION),
        ("efficiency_addendum", root / trainer.EFFICIENCY_ADDENDUM),
    ):
        verified = validate_file_record(authority[name], label=f"V2 target {name}")
        if verified != expected.resolve():
            raise ValueError(f"V2 target authority binds another {name}")

    # Source promotion and checkpoint identity are proven before target records.
    result_path = validate_file_record(
        authority["four_plus_two_result"], label="V2 target formal result"
    )
    result_raw, _, _ = load_json_object(
        result_path,
        expected_sha256=authority["four_plus_two_result"]["sha256"],
        label="V2 target formal result",
    )
    result = validate_result(result_raw, require_promotion=True)
    if result["checkpoint"] != authority["promoted_checkpoint"]:
        raise ValueError("V2 target authority checkpoint differs from result")
    checkpoint_path = validate_file_record(
        authority["promoted_checkpoint"], label="V2 target promoted checkpoint"
    )
    checkpoint_raw, checkpoint_sha, checkpoint_source = load_torch_mapping(
        checkpoint_path,
        expected_sha256=authority["promoted_checkpoint"]["sha256"],
        map_location="cpu",
        label="V2 target promoted checkpoint",
    )
    checkpoint = validate_checkpoint(checkpoint_raw)
    if (
        checkpoint["selected_validation"] != result["selected_validation"]
        or checkpoint["singleton_validation"] != result["singleton_validation"]
        or checkpoint["promotion_gate"] != result["promotion_gate"]
        or checkpoint["selected_epoch"] != result["selected_validation"]["epoch"]
        or checkpoint["promotion_gate"]["passed"] is not True
        or int(checkpoint["selected_epoch"]) <= 0
    ):
        raise ValueError("V2 target result/checkpoint promotion chain differs")

    inputs = authority.get("target_feature_inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(TARGET_INPUT_NAMES):
        raise ValueError("V2 target feature input authority differs")
    target_inputs = {}
    for name in TARGET_INPUT_NAMES:
        verified = validate_file_record(inputs[name], label=f"V2 target {name}")
        target_inputs[name] = {
            "path": str(verified),
            "sha256": str(inputs[name]["sha256"]),
        }
    feature_output = _canonical_output_path(
        authority["target_feature_output"], label="V2 target feature output"
    )
    inference_output = _canonical_output_path(
        authority["target_inference_output"], label="V2 target inference output"
    )
    if feature_output == inference_output:
        raise ValueError("V2 target feature and inference outputs must differ")
    if expected_feature_output is not None and feature_output != str(
        Path(expected_feature_output).expanduser().resolve()
    ):
        raise ValueError("V2 target feature output differs from execution authority")
    if expected_inference_output is not None and inference_output != str(
        Path(expected_inference_output).expanduser().resolve()
    ):
        raise ValueError("V2 target inference output differs from execution authority")
    authority["target_feature_inputs"] = target_inputs
    authority["verified_record"] = {"path": str(source), "sha256": digest}
    authority["verified_result"] = result
    authority["verified_checkpoint"] = checkpoint
    authority["verified_checkpoint_record"] = {
        "path": str(checkpoint_source),
        "sha256": checkpoint_sha,
    }
    return authority


__all__ = [
    "RESULT_SCHEMA",
    "TARGET_EXECUTION_SCHEMA",
    "TARGET_EXECUTION_STATUS",
    "TARGET_INPUT_NAMES",
    "validate_checkpoint",
    "validate_result",
    "validate_selection_candidate",
    "validate_target_execution_authority",
]
