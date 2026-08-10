"""Fail-closed source promotion gate for triggered V2.1C Stage-II runs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import surface_region_v21b_source_gate as v21b_gate
from radio_gs.scripts import (
    train_surface_region_v21c_two_stage_constrained_adamw as trainer,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
)


PROMOTION_SCHEMA = "radio_gs.surface_region_v21c_source_promotion_evidence.v1"
PROMOTION_CHAIN_SCHEMA = "radio_gs.surface_region_v21c_source_promotion_chain.v1"


def _record(value: object, *, label: str) -> dict[str, str]:
    return trainer._record(value, label=label)


def _optional_record(value: object, *, label: str) -> dict[str, str] | None:
    return None if value is None else _record(value, label=label)


def _validation(
    value: object,
    *,
    epoch_zero: Mapping[str, Any] | None,
    label: str,
) -> dict[str, Any]:
    required = {
        "v1_non_regression",
        "response_listwise_v21b",
        "validation_no_grad",
        "benchmark_opened",
        "promotion",
        "selection_eligible",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{label} validation fields differ")
    result = dict(value)
    if (
        result["validation_no_grad"] is not True
        or result["benchmark_opened"] is not False
        or not isinstance(result["selection_eligible"], bool)
    ):
        raise ValueError(f"{label} validation protocol differs")
    v21b_gate.baseline_gate._validate_v1_non_regression(
        result["v1_non_regression"], label=f"{label}.v1"
    )
    result["response_listwise_v21b"] = v21b_gate._validate_response(
        result["response_listwise_v21b"], label=f"{label}.response"
    )
    if epoch_zero is not None:
        recomputed = trainer.promotion_checks(result, epoch_zero)
        if (
            result["promotion"] != recomputed
            or result["selection_eligible"] is not bool(recomputed["passed"])
        ):
            raise ValueError(f"{label} V2.1C promotion differs")
    return result


def _history(value: object) -> list[dict[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != trainer.OPTIMIZER_STEPS + 1
    ):
        raise ValueError("V2.1C history must contain step zero plus 30 steps")
    result: list[dict[str, Any]] = []
    epoch_zero: dict[str, Any] | None = None
    for expected_step, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != {
            "step",
            "training",
            "validation",
            "model_state_dict_sha256",
        }:
            raise ValueError("V2.1C history row fields differ")
        if raw["step"] != expected_step or (expected_step == 0) != (
            raw["training"] is None
        ):
            raise ValueError("V2.1C history step axis differs")
        validation = _validation(
            raw["validation"],
            epoch_zero=epoch_zero if expected_step > 0 else None,
            label=f"V2.1C step {expected_step}",
        )
        if expected_step == 0:
            epoch_zero = validation
            recomputed = trainer.promotion_checks(validation, validation)
            if (
                validation["promotion"] != recomputed
                or validation["selection_eligible"] is not False
            ):
                raise ValueError("V2.1C step-zero promotion differs")
        else:
            training = raw["training"]
            if not isinstance(training, Mapping) or set(training) != {
                "scene_count",
                "equal_scene_weight",
                "per_scene",
                "gradient_evidence",
                "projected_adamw_evidence",
            }:
                raise ValueError("V2.1C projected training evidence differs")
            projected = training["projected_adamw_evidence"]
            if (
                not isinstance(projected, Mapping)
                or projected.get("kkt", {}).get("passed") is not True
                or float(projected.get("projected_dot", {}).get("absolute", -1.0))
                < -2e-7
                or float(projected.get("projected_dot", {}).get("pairwise", -1.0))
                < -2e-7
                or projected.get("adamw_moments_advanced_before_projection")
                is not True
                or projected.get("decoupled_weight_decay_in_candidate") is not True
            ):
                raise ValueError("V2.1C projected update certificate differs")
            trainer._finite_tree(training, label=f"V2.1C step {expected_step}")
        digest = str(raw["model_state_dict_sha256"])
        if trainer._SHA256.fullmatch(digest) is None:
            raise ValueError("V2.1C model-state digest differs")
        result.append({**dict(raw), "validation": validation})
    return result


def validate_source_promotion_evidence(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "status",
        "training_contract",
        "training_contract_sha256",
        "execution_authority",
        "stage_i_audit_result",
        "normalization_authority",
        "state_archive",
        "checkpoint",
        "certificate",
        "selected_step",
        "selected_validation",
        "promotion_candidate_available",
        "history",
        "source_access",
        "benchmark_opened",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("V2.1C source result fields differ")
    result = dict(value)
    if (
        result["schema"] != trainer.STAGE_II_RESULT_SCHEMA
        or result["schema_version"] != 1
        or result["training_contract"] != trainer.training_contract()
        or result["training_contract_sha256"] != trainer.TRAINING_CONTRACT_SHA256
        or result["source_access"] != trainer.source_access()
        or result["benchmark_opened"] is not False
        or not isinstance(result["promotion_candidate_available"], bool)
    ):
        raise ValueError("V2.1C source result identity differs")
    for name in (
        "execution_authority",
        "stage_i_audit_result",
        "normalization_authority",
        "state_archive",
    ):
        result[name] = _record(result[name], label=name)
    result["checkpoint"] = _optional_record(result["checkpoint"], label="checkpoint")
    result["certificate"] = _optional_record(
        result["certificate"], label="certificate"
    )
    history = _history(result["history"])
    selected = trainer.select_promotion_step(history)
    passed = selected is not None
    expected_status = (
        "source_only_v21c_promotion_candidate_complete"
        if passed
        else "source_only_v21c_complete_no_eligible_promotion"
    )
    if (
        result["status"] != expected_status
        or result["promotion_candidate_available"] is not passed
        or result["selected_step"] != selected
        or result["selected_validation"]
        != (history[selected]["validation"] if selected is not None else None)
        or (result["checkpoint"] is not None) is not passed
        or (result["certificate"] is not None) is not passed
    ):
        raise ValueError("V2.1C selected promotion state differs")
    return {
        "schema": PROMOTION_SCHEMA,
        "schema_version": 1,
        "selected_step": selected,
        "checks": (
            history[selected]["validation"]["promotion"]["checks"]
            if selected is not None
            else {}
        ),
        "passed": passed,
        "normalized_history": history,
        "normalized_result": result,
        "checkpoint_opened": False,
        "target_execution_authorized": False,
        "benchmark_opened": False,
    }


def validate_normalization(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("V2.1C normalization must be a mapping")
    original = dict(value)
    if (
        original.get("schema") != trainer.NORMALIZATION_SCHEMA
        or original.get("source_access") != trainer.source_access()
    ):
        raise ValueError("V2.1C normalization identity differs")
    legacy = dict(original)
    legacy["schema"] = v21b_gate.trainer.NORMALIZATION_SCHEMA
    legacy["source_access"] = v21b_gate.trainer.source_access()
    v21b_gate.validate_normalization_authority(legacy)
    return original


def validate_state_archive(
    value: object,
    *,
    normalization: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "training_contract_sha256",
        "execution_authority",
        "stage_i_audit_result",
        "normalization_authority",
        "saved_steps",
        "promotion_eligible_steps",
        "model_state_dict_sha256_by_step",
        "model_state_dict_by_step",
        "source_access",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("V2.1C state archive fields differ")
    archive = dict(value)
    eligible = [
        int(row["step"])
        for row in history
        if row["validation"]["selection_eligible"] is True
    ]
    saved = [0, *eligible]
    expected_keys = {str(step) for step in saved}
    if (
        archive["schema"] != trainer.STAGE_II_STATE_ARCHIVE_SCHEMA
        or archive["schema_version"] != 1
        or archive["training_contract_sha256"] != trainer.TRAINING_CONTRACT_SHA256
        or archive["source_access"] != trainer.source_access()
        or archive["saved_steps"] != saved
        or archive["promotion_eligible_steps"] != eligible
        or set(archive["model_state_dict_sha256_by_step"]) != expected_keys
        or set(archive["model_state_dict_by_step"]) != expected_keys
    ):
        raise ValueError("V2.1C state archive identity differs")
    for name in (
        "execution_authority",
        "stage_i_audit_result",
        "normalization_authority",
    ):
        archive[name] = _record(archive[name], label=f"archive {name}")
    frozen: dict[str, dict[str, torch.Tensor]] = {}
    for step in saved:
        state, digest = v21b_gate._validate_model_state(
            archive["model_state_dict_by_step"][str(step)],
            normalization=normalization,
        )
        if (
            digest != archive["model_state_dict_sha256_by_step"][str(step)]
            or digest != history[step]["model_state_dict_sha256"]
        ):
            raise ValueError(f"V2.1C state archive step {step} differs")
        frozen[str(step)] = state
    archive["model_state_dict_by_step"] = frozen
    return archive


def _actual(path: Path, digest: str) -> dict[str, str]:
    return {"path": str(path), "sha256": digest}


def validate_source_pilot_chain(
    path: str | Path,
    *,
    expected_sha256: str,
    require_promotion: bool = True,
) -> dict[str, Any]:
    raw, result_sha, result_path = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="V2.1C source result",
    )
    promotion = validate_source_promotion_evidence(raw)
    result = promotion.pop("normalized_result")
    history = promotion.pop("normalized_history")
    execution = result["execution_authority"]
    inputs = trainer.prepare_inputs(
        validate_file_record(execution, label="V2.1C execution authority"),
        expected_sha256=execution["sha256"],
    )
    if (
        inputs.execution["stage"] != trainer.STAGE_II
        or inputs.execution["stage_i_audit_result"]
        != result["stage_i_audit_result"]
    ):
        raise ValueError("V2.1C result execution/audit chain differs")
    normalization_raw, normalization_sha, normalization_path = load_torch_mapping(
        result["normalization_authority"]["path"],
        expected_sha256=result["normalization_authority"]["sha256"],
        map_location="cpu",
        label="V2.1C normalization",
    )
    normalization = validate_normalization(normalization_raw)
    normalization_record = _actual(normalization_path, normalization_sha)
    archive_raw, archive_sha, archive_path = load_torch_mapping(
        result["state_archive"]["path"],
        expected_sha256=result["state_archive"]["sha256"],
        map_location="cpu",
        label="V2.1C state archive",
    )
    archive = validate_state_archive(
        archive_raw, normalization=normalization, history=history
    )
    archive_record = _actual(archive_path, archive_sha)
    if (
        result["normalization_authority"] != normalization_record
        or result["state_archive"] != archive_record
        or archive["execution_authority"] != execution
        or archive["stage_i_audit_result"] != result["stage_i_audit_result"]
        or archive["normalization_authority"] != normalization_record
    ):
        raise ValueError("V2.1C result/archive chain differs")

    checkpoint_record = None
    certificate_record = None
    selected_sha = None
    if promotion["passed"]:
        certificate_raw, certificate_sha, certificate_path = load_json_object(
            result["certificate"]["path"],
            expected_sha256=result["certificate"]["sha256"],
            label="V2.1C certificate",
        )
        certificate = dict(certificate_raw)
        declared = certificate.pop("content_sha256", None)
        if (
            declared is None
            or canonical_json_sha256(certificate) != declared
            or certificate.get("schema") != trainer.STAGE_II_CERTIFICATE_SCHEMA
            or certificate.get("benchmark_opened") is not False
        ):
            raise ValueError("V2.1C certificate identity differs")
        certificate["content_sha256"] = declared
        certificate_record = _actual(certificate_path, certificate_sha)
        checkpoint_raw, checkpoint_sha, checkpoint_path = load_torch_mapping(
            result["checkpoint"]["path"],
            expected_sha256=result["checkpoint"]["sha256"],
            map_location="cpu",
            label="V2.1C checkpoint",
        )
        checkpoint = dict(checkpoint_raw)
        if (
            checkpoint.get("schema") != trainer.STAGE_II_CHECKPOINT_SCHEMA
            or checkpoint.get("source_access") != trainer.source_access()
        ):
            raise ValueError("V2.1C checkpoint identity differs")
        state, selected_sha = v21b_gate._validate_model_state(
            checkpoint.get("model_state_dict"), normalization=normalization
        )
        selected_step = int(promotion["selected_step"])
        if (
            selected_sha != checkpoint.get("model_state_dict_sha256")
            or selected_sha != history[selected_step]["model_state_dict_sha256"]
            or checkpoint.get("selected_step") != selected_step
            or checkpoint.get("normalization_authority") != normalization_record
            or checkpoint.get("certificate") != certificate_record
            or checkpoint.get("state_archive") != archive_record
            or certificate.get("execution_authority") != execution
            or certificate.get("stage_i_audit_result")
            != result["stage_i_audit_result"]
            or certificate.get("selected_step") != selected_step
            or certificate.get("selected_validation")
            != history[selected_step]["validation"]
            or certificate.get("model_state_dict_sha256") != selected_sha
            or certificate.get("normalization_authority") != normalization_record
            or certificate.get("state_archive") != archive_record
            or archive["model_state_dict_sha256_by_step"][str(selected_step)]
            != selected_sha
        ):
            raise ValueError("V2.1C promoted checkpoint/certificate chain differs")
        checkpoint_record = _actual(checkpoint_path, checkpoint_sha)
        if (
            result["checkpoint"] != checkpoint_record
            or result["certificate"] != certificate_record
        ):
            raise ValueError("V2.1C promoted output record differs")
    if require_promotion and promotion["passed"] is not True:
        raise ValueError("V2.1C source promotion has no eligible state")
    return {
        "schema": PROMOTION_CHAIN_SCHEMA,
        "schema_version": 1,
        "source_result": _actual(result_path, result_sha),
        "execution_authority": execution,
        "stage_i_audit_result": result["stage_i_audit_result"],
        "normalization_authority": normalization_record,
        "state_archive": archive_record,
        "certificate": certificate_record,
        "checkpoint": checkpoint_record,
        "selected_step": promotion["selected_step"],
        "model_state_dict_sha256": selected_sha,
        "promotion": promotion,
        "source_promotion_authorized": promotion["passed"] is True,
        "target_execution_authorized": False,
        "benchmark_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", required=True)
    parser.add_argument("--expected-source-result-sha256", required=True)
    parser.add_argument("--allow-failed-diagnostic", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = validate_source_pilot_chain(
        args.source_result,
        expected_sha256=args.expected_source_result_sha256,
        require_promotion=not args.allow_failed_diagnostic,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "PROMOTION_CHAIN_SCHEMA",
    "PROMOTION_SCHEMA",
    "build_parser",
    "validate_normalization",
    "validate_source_pilot_chain",
    "validate_source_promotion_evidence",
    "validate_state_archive",
]
