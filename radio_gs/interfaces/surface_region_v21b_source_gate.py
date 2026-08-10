"""Fail-closed complete source-chain gate for V2.1B exact-4+2 training."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import re
from typing import Any

import torch

from radio_gs.interfaces.surface_region_summary import (
    surface_region_state_dict_sha256,
)
from radio_gs.interfaces.surface_region_typed_context_training import (
    accepted_v2_authority,
)
from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v21_pilot as asset_pilot,
)
from radio_gs.interfaces import surface_region_v21_source_gate as baseline_gate
from radio_gs.interfaces import (
    surface_region_v21b_reliability_conditioned_residual as v21b_interface,
)
from radio_gs.scripts import (
    train_surface_region_v21b_conditioned_rank256_exact4x2 as trainer,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
)


PROMOTION_SCHEMA = "radio_gs.surface_region_v21b_source_promotion_evidence.v1"
PROMOTION_CHAIN_SCHEMA = "radio_gs.surface_region_v21b_source_promotion_chain.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOSS_KEYS = (
    "response_auxiliary_loss",
    "response_absolute_relevance_loss",
    "response_continuous_pairwise_relevance_loss",
)


def _sha(value: object, *, label: str) -> str:
    result = str(value)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _finite(value: object, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _file_record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path = str(value["path"])
    digest = _sha(value["sha256"], label=label)
    if not path.startswith("/"):
        raise ValueError(f"{label} path must be absolute")
    return {"path": path, "sha256": digest}


def _optional_file_record(value: object, *, label: str) -> dict[str, str] | None:
    return None if value is None else _file_record(value, label=label)


def _validate_response(value: object, *, label: str) -> dict[str, Any]:
    required = {
        "scene_count",
        "scene_macro_auxiliary_loss",
        "scene_macro_absolute_relevance_loss",
        "scene_macro_continuous_pairwise_relevance_loss",
        "scene_macro_active_row_coverage",
        "scene_macro_pair_trainable_endpoint_coverage",
        "per_scene",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{label} response fields differ")
    result = dict(value)
    if result["scene_count"] != len(trainer.VALIDATION_SCENES):
        raise ValueError(f"{label} response scene count differs")
    per_scene = result["per_scene"]
    if (
        not isinstance(per_scene, Mapping)
        or tuple(per_scene) != trainer.VALIDATION_SCENES
    ):
        raise ValueError(f"{label} response scene axis differs")
    normalized: dict[str, dict[str, float]] = {}
    for scene in trainer.VALIDATION_SCENES:
        row = per_scene[scene]
        required_row = {
            *_LOSS_KEYS,
            "complete_canonical_rows",
            "active_rows",
            "active_row_coverage",
            "fallback_bitwise_accepted_v2_e0",
            "response_authority_hard_negative_pairs",
            "response_objective_hard_negative_pairs",
            "response_pairwise_objective_hard_negative_pairs",
            "response_triplet_objective_hard_negative_pairs",
            "response_pair_trainable_endpoint_coverage",
        }
        if not isinstance(row, Mapping) or not required_row.issubset(row):
            raise ValueError(f"{label} {scene} response audit differs")
        baseline_gate._finite_scalar_tree(row, label=f"{label}.{scene}")
        losses = {
            name: _finite(row[name], label=f"{label}.{scene}.{name}")
            for name in _LOSS_KEYS
        }
        if any(item < 0 for item in losses.values()):
            raise ValueError(f"{label} {scene} response loss is negative")
        total_rows = int(row["complete_canonical_rows"])
        active_rows = int(row["active_rows"])
        authority_pairs = int(row["response_authority_hard_negative_pairs"])
        pairwise_pairs = int(row["response_pairwise_objective_hard_negative_pairs"])
        triplet_pairs = int(row["response_triplet_objective_hard_negative_pairs"])
        active_coverage = _finite(
            row["active_row_coverage"], label=f"{label}.{scene}.active coverage"
        )
        pair_coverage = _finite(
            row["response_pair_trainable_endpoint_coverage"],
            label=f"{label}.{scene}.pair coverage",
        )
        if (
            total_rows <= 0
            or not 0 < active_rows <= total_rows
            or authority_pairs <= 0
            or not 0 < triplet_pairs <= pairwise_pairs <= authority_pairs
            or int(row["response_objective_hard_negative_pairs"]) != pairwise_pairs
            or not math.isclose(
                active_coverage,
                active_rows / total_rows,
                rel_tol=0.0,
                abs_tol=1e-7,
            )
            or not math.isclose(
                pair_coverage,
                pairwise_pairs / authority_pairs,
                rel_tol=0.0,
                abs_tol=1e-7,
            )
            or row["fallback_bitwise_accepted_v2_e0"] is not True
        ):
            raise ValueError(f"{label} {scene} denominator/coverage differs")
        normalized[scene] = {
            **losses,
            "active_row_coverage": active_coverage,
            "pair_trainable_endpoint_coverage": pair_coverage,
        }

    declarations = {
        "scene_macro_auxiliary_loss": "response_auxiliary_loss",
        "scene_macro_absolute_relevance_loss": "response_absolute_relevance_loss",
        "scene_macro_continuous_pairwise_relevance_loss": (
            "response_continuous_pairwise_relevance_loss"
        ),
        "scene_macro_active_row_coverage": "active_row_coverage",
        "scene_macro_pair_trainable_endpoint_coverage": (
            "pair_trainable_endpoint_coverage"
        ),
    }
    for declared, key in declarations.items():
        expected = sum(normalized[scene][key] for scene in trainer.VALIDATION_SCENES) / len(
            trainer.VALIDATION_SCENES
        )
        if not math.isclose(
            _finite(result[declared], label=f"{label}.{declared}"),
            expected,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{label} {declared} differs")
    return result


def _validation(value: object, *, label: str) -> dict[str, Any]:
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
    baseline_gate._validate_v1_non_regression(
        result["v1_non_regression"], label=f"{label}.v1"
    )
    result["response_listwise_v21b"] = _validate_response(
        result["response_listwise_v21b"], label=f"{label}.response"
    )
    promotion = result["promotion"]
    if not isinstance(promotion, Mapping) or set(promotion) != {
        "relative_improvement_from_step_zero",
        "checks",
        "passed",
    }:
        raise ValueError(f"{label} promotion fields differ")
    baseline_gate._finite_scalar_tree(promotion, label=f"{label}.promotion")
    return result


def _history(value: object) -> list[dict[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != trainer.OPTIMIZER_STEPS + 1
    ):
        raise ValueError("V2.1B history must contain step zero plus 30 steps")
    history: list[dict[str, Any]] = []
    for expected_step, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != {
            "step",
            "training",
            "validation",
            "model_state_dict_sha256",
        }:
            raise ValueError("V2.1B history row fields differ")
        if raw["step"] != expected_step or (expected_step == 0) != (
            raw["training"] is None
        ):
            raise ValueError("V2.1B history step axis differs")
        if expected_step > 0:
            if not isinstance(raw["training"], Mapping):
                raise ValueError("V2.1B training diagnostic differs")
            baseline_gate._finite_scalar_tree(
                raw["training"], label=f"step {expected_step}.training"
            )
        history.append(
            {
                **dict(raw),
                "model_state_dict_sha256": _sha(
                    raw["model_state_dict_sha256"],
                    label=f"step {expected_step} model state",
                ),
                "validation": _validation(
                    raw["validation"], label=f"step {expected_step}"
                ),
            }
        )
    epoch_zero = history[0]["validation"]
    for row in history:
        recomputed = trainer.promotion_checks(row["validation"], epoch_zero)
        if (
            row["validation"]["promotion"] != recomputed
            or row["validation"]["selection_eligible"] is not bool(
                recomputed["passed"]
            )
        ):
            raise ValueError(f"V2.1B step {row['step']} promotion differs")
    return history


def validate_source_promotion_evidence(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "status",
        "training_contract",
        "training_contract_sha256",
        "execution_authority",
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
        raise ValueError("V2.1B source result fields differ")
    result = dict(value)
    if (
        result["schema"] != trainer.RESULT_SCHEMA
        or result["schema_version"] != 1
        or result["training_contract"] != trainer.training_contract()
        or result["training_contract_sha256"] != trainer.TRAINING_CONTRACT_SHA256
        or result["source_access"] != trainer.source_access()
        or result["benchmark_opened"] is not False
        or not isinstance(result["promotion_candidate_available"], bool)
    ):
        raise ValueError("V2.1B source result identity differs")
    result["execution_authority"] = _file_record(
        result["execution_authority"], label="execution authority"
    )
    result["normalization_authority"] = _file_record(
        result["normalization_authority"], label="normalization authority"
    )
    result["state_archive"] = _file_record(
        result["state_archive"], label="state archive"
    )
    result["checkpoint"] = _optional_file_record(
        result["checkpoint"], label="checkpoint"
    )
    result["certificate"] = _optional_file_record(
        result["certificate"], label="certificate"
    )
    history = _history(result["history"])
    selected = trainer.select_promotion_step(history)
    passed = selected is not None
    expected_status = (
        "source_only_v21b_promotion_candidate_complete"
        if passed
        else "source_only_v21b_complete_no_eligible_promotion"
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
        raise ValueError("V2.1B selected promotion state differs")
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
        "checkpoint_opened": False,
        "target_execution_authorized": False,
        "benchmark_opened": False,
        "normalized_history": history,
        "normalized_result": result,
    }


def validate_normalization_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("V2.1B normalization must be a mapping")
    original = dict(value)
    if (
        original.get("schema") != trainer.NORMALIZATION_SCHEMA
        or original.get("source_access") != trainer.source_access()
    ):
        raise ValueError("V2.1B normalization identity differs")
    legacy = dict(original)
    legacy["schema"] = baseline_gate.NORMALIZATION_SCHEMA
    legacy["source_access"] = asset_pilot.source_access()
    validated = baseline_gate.validate_normalization_authority(legacy)
    validated["schema"] = trainer.NORMALIZATION_SCHEMA
    validated["source_access"] = trainer.source_access()
    return validated


def _validate_model_state(
    value: object,
    *,
    normalization: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], str]:
    model = v21b_interface.build_model_from_source_normalization(normalization)
    expected = model.state_dict()
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError("V2.1B model state fields differ")
    state: dict[str, torch.Tensor] = {}
    for name, template in expected.items():
        raw = value[name]
        if not torch.is_tensor(raw):
            raise ValueError(f"V2.1B state {name} is not a tensor")
        tensor = raw.detach().cpu().contiguous()
        if (
            tensor.dtype != template.dtype
            or tuple(tensor.shape) != tuple(template.shape)
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(f"V2.1B state {name} differs")
        state[name] = tensor
    if not torch.equal(state["scalar_median"], normalization["median"]) or not torch.equal(
        state["scalar_robust_scale"], normalization["robust_scale"]
    ):
        raise ValueError("V2.1B state normalization buffers differ")
    model.load_state_dict(state, strict=True)
    return state, surface_region_state_dict_sha256(state)


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
        "normalization_authority",
        "normalization_content_authority_sha256",
        "saved_steps",
        "promotion_eligible_steps",
        "model_state_dict_sha256_by_step",
        "model_state_dict_by_step",
        "source_access",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("V2.1B state archive fields differ")
    archive = dict(value)
    eligible = [
        int(row["step"])
        for row in history
        if row["validation"]["selection_eligible"] is True
    ]
    expected_saved = [0, *eligible]
    hashes = archive["model_state_dict_sha256_by_step"]
    states = archive["model_state_dict_by_step"]
    expected_keys = {str(step) for step in expected_saved}
    if (
        archive["schema"] != trainer.STATE_ARCHIVE_SCHEMA
        or archive["schema_version"] != 1
        or archive["training_contract_sha256"] != trainer.TRAINING_CONTRACT_SHA256
        or archive["source_access"] != trainer.source_access()
        or archive["saved_steps"] != expected_saved
        or archive["promotion_eligible_steps"] != eligible
        or not isinstance(hashes, Mapping)
        or not isinstance(states, Mapping)
        or set(hashes) != expected_keys
        or set(states) != expected_keys
    ):
        raise ValueError("V2.1B state archive identity differs")
    archive["execution_authority"] = _file_record(
        archive["execution_authority"], label="archive execution authority"
    )
    archive["normalization_authority"] = _file_record(
        archive["normalization_authority"], label="archive normalization"
    )
    _sha(
        archive["normalization_content_authority_sha256"],
        label="archive normalization content",
    )
    frozen: dict[str, dict[str, torch.Tensor]] = {}
    for step in expected_saved:
        state, digest = _validate_model_state(
            states[str(step)], normalization=normalization
        )
        if (
            digest != _sha(hashes[str(step)], label=f"archive step {step}")
            or digest != history[step]["model_state_dict_sha256"]
        ):
            raise ValueError(f"V2.1B archive step {step} state digest differs")
        frozen[str(step)] = state
    archive["model_state_dict_by_step"] = frozen
    return archive


def validate_certificate(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "training_contract",
        "training_contract_sha256",
        "execution_authority",
        "cohort_authority",
        "pilot_cohort_region_view_registry",
        "pilot_cohort_region_view_registry_authority_sha256",
        "benchmark_exclusion_manifest",
        "input_records_by_split",
        "selected_step",
        "selected_validation",
        "model_state_dict_sha256",
        "normalization_authority",
        "normalization_content_authority_sha256",
        "state_archive",
        "source_access",
        "benchmark_opened",
        "content_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("V2.1B certificate fields differ")
    result = dict(value)
    content = dict(result)
    declared = _sha(content.pop("content_sha256"), label="certificate content")
    if (
        result["schema"] != trainer.CERTIFICATE_SCHEMA
        or result["schema_version"] != 1
        or result["training_contract"] != trainer.training_contract()
        or result["training_contract_sha256"] != trainer.TRAINING_CONTRACT_SHA256
        or result["source_access"] != trainer.source_access()
        or result["benchmark_opened"] is not False
        or canonical_json_sha256(content) != declared
    ):
        raise ValueError("V2.1B certificate identity differs")
    for name in (
        "execution_authority",
        "cohort_authority",
        "pilot_cohort_region_view_registry",
        "benchmark_exclusion_manifest",
        "normalization_authority",
        "state_archive",
    ):
        result[name] = _file_record(result[name], label=f"certificate {name}")
    _sha(result["model_state_dict_sha256"], label="certificate model state")
    _sha(
        result["pilot_cohort_region_view_registry_authority_sha256"],
        label="certificate registry authority",
    )
    _sha(
        result["normalization_content_authority_sha256"],
        label="certificate normalization content",
    )
    return result


def validate_checkpoint_payload(
    value: object,
    *,
    normalization: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "model_class",
        "model_architecture",
        "accepted_v2_authority",
        "model_state_dict",
        "model_state_dict_sha256",
        "normalization_authority",
        "certificate",
        "state_archive",
        "selected_step",
        "source_access",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("V2.1B checkpoint fields differ")
    checkpoint = dict(value)
    model = v21b_interface.build_model_from_source_normalization(normalization)
    if (
        checkpoint["schema"] != trainer.CHECKPOINT_SCHEMA
        or checkpoint["schema_version"] != 1
        or checkpoint["model_class"] != type(model).__name__
        or checkpoint["model_architecture"] != model.architecture()
        or checkpoint["accepted_v2_authority"] != accepted_v2_authority()
        or checkpoint["source_access"] != trainer.source_access()
        or not isinstance(checkpoint["selected_step"], int)
        or isinstance(checkpoint["selected_step"], bool)
        or not 1 <= checkpoint["selected_step"] <= trainer.OPTIMIZER_STEPS
    ):
        raise ValueError("V2.1B checkpoint identity differs")
    for name in ("normalization_authority", "certificate", "state_archive"):
        checkpoint[name] = _file_record(
            checkpoint[name], label=f"checkpoint {name}"
        )
    state, digest = _validate_model_state(
        checkpoint["model_state_dict"], normalization=normalization
    )
    if digest != _sha(
        checkpoint["model_state_dict_sha256"], label="checkpoint model state"
    ):
        raise ValueError("V2.1B checkpoint state digest differs")
    checkpoint["model_state_dict"] = state
    return checkpoint


def _actual_record(path: Path, digest: str) -> dict[str, str]:
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
        label="V2.1B source result",
    )
    promotion = validate_source_promotion_evidence(raw)
    result = promotion.pop("normalized_result")
    history = promotion.pop("normalized_history")

    execution_record = result["execution_authority"]
    validate_file_record(execution_record, label="V2.1B execution authority")
    inputs = trainer.prepare_inputs(
        execution_record["path"], expected_sha256=execution_record["sha256"]
    )
    if {
        "path": inputs.execution["verified_path"],
        "sha256": inputs.execution["verified_sha256"],
    } != execution_record:
        raise ValueError("V2.1B execution replay record differs")

    normalization_record = result["normalization_authority"]
    normalization_raw, normalization_sha, normalization_path = load_torch_mapping(
        normalization_record["path"],
        expected_sha256=normalization_record["sha256"],
        map_location="cpu",
        label="V2.1B normalization",
    )
    normalization = validate_normalization_authority(normalization_raw)
    actual_normalization = _actual_record(normalization_path, normalization_sha)

    archive_record = result["state_archive"]
    archive_raw, archive_sha, archive_path = load_torch_mapping(
        archive_record["path"],
        expected_sha256=archive_record["sha256"],
        map_location="cpu",
        label="V2.1B state archive",
    )
    archive = validate_state_archive(
        archive_raw, normalization=normalization, history=history
    )
    actual_archive = _actual_record(archive_path, archive_sha)
    normalization_content = trainer.normalization_content_authority_sha256(
        normalization
    )
    normalization_train = [
        {
            "scene_id": row["scene_id"],
            "training_shard": row["training_shard"],
            "adaptive_context": row["adaptive_context"],
        }
        for row in inputs.execution["source_train"]
    ]
    source_manifest, _ = trainer.pilot_shard.derive_pilot_global_manifests(
        inputs.registry
    )
    if (
        result["normalization_authority"] != actual_normalization
        or result["state_archive"] != actual_archive
        or archive["execution_authority"] != execution_record
        or archive["normalization_authority"] != actual_normalization
        or archive["normalization_content_authority_sha256"]
        != normalization_content
        or normalization["train_input_records"] != normalization_train
        or normalization["source_state_cohort_authority_sha256"]
        != source_manifest["authority_sha256"]
    ):
        raise ValueError("V2.1B result/normalization/archive chain differs")

    checkpoint_record = None
    certificate_record = None
    selected_state_sha = None
    if promotion["passed"]:
        if result["certificate"] is None or result["checkpoint"] is None:
            raise ValueError("V2.1B promoted result lacks checkpoint or certificate")
        certificate_raw, certificate_sha, certificate_path = load_json_object(
            result["certificate"]["path"],
            expected_sha256=result["certificate"]["sha256"],
            label="V2.1B certificate",
        )
        certificate = validate_certificate(certificate_raw)
        certificate_record = _actual_record(certificate_path, certificate_sha)
        checkpoint_raw, checkpoint_sha, checkpoint_path = load_torch_mapping(
            result["checkpoint"]["path"],
            expected_sha256=result["checkpoint"]["sha256"],
            map_location="cpu",
            label="V2.1B checkpoint",
        )
        checkpoint = validate_checkpoint_payload(
            checkpoint_raw, normalization=normalization
        )
        checkpoint_record = _actual_record(checkpoint_path, checkpoint_sha)
        selected_step = int(promotion["selected_step"])
        selected_state_sha = history[selected_step]["model_state_dict_sha256"]
        if (
            result["certificate"] != certificate_record
            or result["checkpoint"] != checkpoint_record
            or certificate["execution_authority"] != execution_record
            or certificate["cohort_authority"]
            != inputs.execution["cohort_authority"]
            or certificate["pilot_cohort_region_view_registry"]
            != inputs.execution["pilot_cohort_region_view_registry"]
            or certificate["pilot_cohort_region_view_registry_authority_sha256"]
            != inputs.registry["authority_sha256"]
            or certificate["benchmark_exclusion_manifest"]
            != inputs.execution["benchmark_exclusion_manifest"]
            or certificate["input_records_by_split"]
            != {
                "source_train": trainer._input_records(inputs.train),
                "source_validation": trainer._input_records(inputs.validation),
            }
            or certificate["selected_step"] != selected_step
            or certificate["selected_validation"] != result["selected_validation"]
            or certificate["model_state_dict_sha256"] != selected_state_sha
            or certificate["normalization_authority"] != actual_normalization
            or certificate["normalization_content_authority_sha256"]
            != normalization_content
            or certificate["state_archive"] != actual_archive
            or checkpoint["selected_step"] != selected_step
            or checkpoint["model_state_dict_sha256"] != selected_state_sha
            or checkpoint["normalization_authority"] != actual_normalization
            or checkpoint["certificate"] != certificate_record
            or checkpoint["state_archive"] != actual_archive
            or archive["model_state_dict_sha256_by_step"][str(selected_step)]
            != selected_state_sha
        ):
            raise ValueError("V2.1B promoted checkpoint/certificate chain differs")
    if require_promotion and promotion["passed"] is not True:
        raise ValueError("V2.1B source promotion has no eligible state")
    return {
        "schema": PROMOTION_CHAIN_SCHEMA,
        "schema_version": 1,
        "source_result": _actual_record(result_path, result_sha),
        "execution_authority": execution_record,
        "normalization_authority": actual_normalization,
        "state_archive": actual_archive,
        "certificate": certificate_record,
        "checkpoint": checkpoint_record,
        "selected_step": promotion["selected_step"],
        "model_state_dict_sha256": selected_state_sha,
        "normalization_content_authority_sha256": normalization_content,
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
    "validate_certificate",
    "validate_checkpoint_payload",
    "validate_normalization_authority",
    "validate_source_pilot_chain",
    "validate_source_promotion_evidence",
    "validate_state_archive",
]
