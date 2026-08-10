"""Fail-closed source-heldout promotion evidence for the V2.1 pilot.

The lightweight validator recomputes the semantic promotion decision from a
result mapping.  :func:`validate_source_pilot_chain` is the formal boundary:
it opens and SHA-validates the source result, execution authority, train-only
normalization, certificate and checkpoint, then checks every cross-file
binding before reporting promotion.  Neither path accepts or opens a target
scene, benchmark query, label, mask, image, metric, or renderer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from radio_gs.models.surface_region_typed_context_residual import (
    SurfaceRegionAcceptedV2TypedContextResidualV1,
)
from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v21_pilot as pilot,
)
from radio_gs.scripts import train_surface_region_typed_context_residual as v1_trainer
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
)


RESULT_SCHEMA = (
    "radio_gs.surface_region_typed_context_response_listwise_v21_" "pilot_result.v1"
)
PROMOTION_SCHEMA = "radio_gs.surface_region_v21_source_promotion_evidence.v1"
PROMOTION_CHAIN_SCHEMA = "radio_gs.surface_region_v21_source_promotion_chain.v1"
NORMALIZATION_SCHEMA = "radio_gs.v21_pilot_train4_normalization.v1"
CERTIFICATE_SCHEMA = (
    "radio_gs.surface_region_typed_context_response_listwise_v21_"
    "pilot_certificate.v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESPONSE_KEYS = (
    "response_auxiliary_loss",
    "response_absolute_relevance_loss",
    "response_continuous_pairwise_relevance_loss",
)


def _finite(value: object, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _sha256(value: object, *, label: str) -> str:
    result = str(value)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _finite_scalar_tree(value: object, *, label: str) -> None:
    """Reject non-JSON or non-finite history payloads without changing schema."""

    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        _finite(value, label=label)
        return
    if isinstance(value, Mapping):
        for name, item in value.items():
            if not isinstance(name, str):
                raise ValueError(f"{label} mapping keys differ")
            _finite_scalar_tree(item, label=f"{label}.{name}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _finite_scalar_tree(item, label=f"{label}[{index}]")
        return
    raise ValueError(f"{label} contains a non-JSON value")


def _file_record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path = str(value["path"])
    digest = str(value["sha256"])
    if not path.startswith("/") or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} file record differs")
    return {"path": path, "sha256": digest}


def _validation(value: object, *, label: str) -> dict[str, Any]:
    required = {
        "v1_non_regression",
        "response_listwise_v21",
        "selection_eligible",
        "validation_no_grad",
        "benchmark_opened",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{label} validation fields differ")
    result = dict(value)
    v1 = result["v1_non_regression"]
    response = result["response_listwise_v21"]
    if (
        not isinstance(v1, Mapping)
        or not isinstance(v1.get("non_regression_passed"), bool)
        or v1.get("validation_no_grad") is not True
        or not isinstance(result.get("selection_eligible"), bool)
        or result.get("validation_no_grad") is not True
        or result.get("benchmark_opened") is not False
        or not isinstance(response, Mapping)
        or set(response)
        != {
            "scene_count",
            "scene_macro_auxiliary_loss",
            "scene_macro_pair_trainable_endpoint_coverage",
            "all_authority_pairs_retained",
            "per_scene",
        }
        or response.get("scene_count") != len(pilot.VALIDATION_SCENES)
        or response.get("all_authority_pairs_retained") is not True
    ):
        raise ValueError(f"{label} validation gate differs")
    per_scene = response["per_scene"]
    if (
        not isinstance(per_scene, Mapping)
        or tuple(per_scene) != pilot.VALIDATION_SCENES
    ):
        raise ValueError(f"{label} validation scene axis differs")
    normalized: dict[str, dict[str, float]] = {}
    for scene in pilot.VALIDATION_SCENES:
        row = per_scene[scene]
        if not isinstance(row, Mapping) or not set(_RESPONSE_KEYS).issubset(row):
            raise ValueError(f"{label} {scene} response metrics differ")
        normalized[scene] = {
            name: _finite(row[name], label=f"{label} {scene} {name}")
            for name in _RESPONSE_KEYS
        }
        if any(value < 0 for value in normalized[scene].values()):
            raise ValueError(f"{label} {scene} response loss is negative")
        _finite_scalar_tree(row, label=f"{label}.{scene}")
        objective_pairs = int(row.get("response_objective_hard_negative_pairs", -1))
        authority_pairs = int(row.get("response_authority_hard_negative_pairs", -2))
        if objective_pairs <= 0 or objective_pairs != authority_pairs:
            raise ValueError(f"{label} {scene} hard-negative denominator differs")
    macro = sum(
        normalized[scene]["response_auxiliary_loss"]
        for scene in pilot.VALIDATION_SCENES
    ) / len(pilot.VALIDATION_SCENES)
    declared_macro = _finite(
        response["scene_macro_auxiliary_loss"], label=f"{label} auxiliary macro"
    )
    if not math.isclose(macro, declared_macro, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{label} auxiliary macro differs")
    coverage = sum(
        _finite(
            per_scene[scene].get("response_pair_trainable_endpoint_coverage"),
            label=f"{label} {scene} endpoint coverage",
        )
        for scene in pilot.VALIDATION_SCENES
    ) / len(pilot.VALIDATION_SCENES)
    if not math.isclose(
        coverage,
        _finite(
            response["scene_macro_pair_trainable_endpoint_coverage"],
            label=f"{label} endpoint coverage macro",
        ),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{label} endpoint coverage macro differs")
    _validate_v1_non_regression(v1, label=f"{label}.v1_non_regression")
    if result["selection_eligible"] is not bool(v1["non_regression_passed"]):
        raise ValueError(f"{label} selection eligibility differs")
    result["normalized_response"] = normalized
    return result


def _metric_triplet(value: object, *, label: str) -> dict[str, float]:
    names = (
        "mean_all_view_cosine",
        "p05_row_mean_all_view_cosine",
        "relation_fidelity",
    )
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError(f"{label} metric fields differ")
    return {name: _finite(value[name], label=f"{label}.{name}") for name in names}


def _validate_v1_non_regression(value: object, *, label: str) -> None:
    required = {
        "aggregation",
        "base",
        "candidate",
        "candidate_minus_base",
        "paired_scene_mean_delta",
        "per_scene",
        "non_regression_checks",
        "non_regression_passed",
        "validation_no_grad",
        "global_or_split_teacher_densification",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{label} fields differ")
    if (
        value.get("aggregation") != "scene_macro"
        or value.get("validation_no_grad") is not True
        or value.get("global_or_split_teacher_densification") is not False
    ):
        raise ValueError(f"{label} protocol differs")
    base = _metric_triplet(value["base"], label=f"{label}.base")
    candidate = _metric_triplet(value["candidate"], label=f"{label}.candidate")
    delta = _metric_triplet(
        value["candidate_minus_base"], label=f"{label}.candidate_minus_base"
    )
    if any(
        not math.isclose(
            delta[name], candidate[name] - base[name], rel_tol=0.0, abs_tol=1e-12
        )
        for name in base
    ):
        raise ValueError(f"{label} macro delta differs")
    per_scene = value["per_scene"]
    if (
        not isinstance(per_scene, Mapping)
        or tuple(per_scene) != pilot.VALIDATION_SCENES
    ):
        raise ValueError(f"{label} scene axis differs")
    scene_means: list[float] = []
    scene_base: list[dict[str, float]] = []
    scene_candidate: list[dict[str, float]] = []
    scene_required = {
        "base",
        "candidate",
        "candidate_minus_base",
        "active_rows",
        "inactive_fallback_rows",
        "fallback_bitwise_accepted_v2_e0",
        "relation_evaluation_rows",
        "validation_no_grad",
    }
    for scene in pilot.VALIDATION_SCENES:
        row = per_scene[scene]
        if not isinstance(row, Mapping) or set(row) != scene_required:
            raise ValueError(f"{label}.{scene} fields differ")
        row_base = _metric_triplet(row["base"], label=f"{label}.{scene}.base")
        row_candidate = _metric_triplet(
            row["candidate"], label=f"{label}.{scene}.candidate"
        )
        row_delta = _metric_triplet(
            row["candidate_minus_base"], label=f"{label}.{scene}.delta"
        )
        if any(
            not math.isclose(
                row_delta[name],
                row_candidate[name] - row_base[name],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for name in row_base
        ):
            raise ValueError(f"{label}.{scene} delta differs")
        if (
            not isinstance(row["active_rows"], int)
            or isinstance(row["active_rows"], bool)
            or row["active_rows"] < 2
            or not isinstance(row["inactive_fallback_rows"], int)
            or isinstance(row["inactive_fallback_rows"], bool)
            or row["inactive_fallback_rows"] < 0
            or not isinstance(row["relation_evaluation_rows"], int)
            or isinstance(row["relation_evaluation_rows"], bool)
            or not 2 <= row["relation_evaluation_rows"] <= row["active_rows"]
            or row["fallback_bitwise_accepted_v2_e0"] is not True
            or row["validation_no_grad"] is not True
        ):
            raise ValueError(f"{label}.{scene} audit differs")
        scene_means.append(row_delta["mean_all_view_cosine"])
        scene_base.append(row_base)
        scene_candidate.append(row_candidate)
    for name in base:
        expected_base = sum(row[name] for row in scene_base) / len(scene_base)
        expected_candidate = sum(row[name] for row in scene_candidate) / len(
            scene_candidate
        )
        if not math.isclose(base[name], expected_base, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{label} base macro differs")
        if not math.isclose(
            candidate[name], expected_candidate, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{label} candidate macro differs")
    paired = value["paired_scene_mean_delta"]
    if not isinstance(paired, Mapping) or set(paired) != {"minimum", "p05", "maximum"}:
        raise ValueError(f"{label} paired scene summary differs")
    sorted_means = sorted(scene_means)
    expected_paired = {
        "minimum": sorted_means[0],
        "p05": sorted_means[int(math.floor(0.05 * max(0, len(sorted_means) - 1)))],
        "maximum": sorted_means[-1],
    }
    if any(
        not math.isclose(
            _finite(paired[name], label=f"{label}.{name}"),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for name, expected in expected_paired.items()
    ):
        raise ValueError(f"{label} paired scene values differ")
    expected_checks = {
        "macro_mean_all_view_cosine": (
            delta["mean_all_view_cosine"] >= -v1_trainer.NON_REGRESSION_TOLERANCE
        ),
        "macro_p05_row_mean_all_view_cosine": (
            delta["p05_row_mean_all_view_cosine"]
            >= -v1_trainer.NON_REGRESSION_TOLERANCE
        ),
        "macro_relation_fidelity": (
            delta["relation_fidelity"] >= -v1_trainer.NON_REGRESSION_TOLERANCE
        ),
        "paired_scene_worst_mean_delta": (
            min(scene_means) >= -v1_trainer.NON_REGRESSION_TOLERANCE
        ),
        "every_scene_two_active_rows": True,
        "fallback_bitwise_accepted_v2_e0": True,
    }
    if value["non_regression_checks"] != expected_checks or value[
        "non_regression_passed"
    ] is not all(expected_checks.values()):
        raise ValueError(f"{label} non-regression decision differs")


def _history(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("V2.1 result history differs")
    history: list[dict[str, Any]] = []
    for expected_epoch, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != {
            "epoch",
            "training",
            "validation",
            "model_state_dict_sha256",
        }:
            raise ValueError("V2.1 result history row differs")
        if raw.get("epoch") != expected_epoch or (expected_epoch == 0) != (
            raw.get("training") is None
        ):
            raise ValueError("V2.1 result epoch axis differs")
        digest = str(raw.get("model_state_dict_sha256"))
        if _SHA256.fullmatch(digest) is None:
            raise ValueError("V2.1 history state SHA-256 differs")
        if expected_epoch > 0:
            if not isinstance(raw.get("training"), Mapping):
                raise ValueError("V2.1 history training row differs")
            _finite_scalar_tree(
                raw["training"], label=f"epoch {expected_epoch}.training"
            )
        history.append(
            {
                **dict(raw),
                "validation": _validation(
                    raw["validation"], label=f"epoch {expected_epoch}"
                ),
            }
        )
    return history


def validate_source_promotion_evidence(value: object) -> dict[str, Any]:
    """Validate and recompute the minimum source-only semantic promotion gate."""

    required = {
        "schema",
        "schema_version",
        "status",
        "training_contract",
        "training_contract_sha256",
        "execution_authority",
        "checkpoint",
        "normalization_authority",
        "certificate",
        "selected_epoch",
        "automatic_fallback_to_epoch_zero",
        "selected_validation",
        "history",
        "source_access",
        "benchmark_opened",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("V2.1 source result fields differ")
    result = dict(value)
    contract = pilot.training_contract()
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("schema_version") != 1
        or result.get("status") != "source_only_pilot_complete_no_benchmark_execution"
        or result.get("training_contract") != contract
        or result.get("training_contract_sha256") != canonical_json_sha256(contract)
        or result.get("source_access") != pilot.source_access()
        or result.get("benchmark_opened") is not False
    ):
        raise ValueError("V2.1 source result identity differs")
    for name in (
        "execution_authority",
        "checkpoint",
        "normalization_authority",
        "certificate",
    ):
        result[name] = _file_record(result[name], label=name)
    history = _history(result["history"])
    selected_epoch = int(result.get("selected_epoch", -1))
    if (
        selected_epoch < 0
        or selected_epoch >= len(history)
        or result.get("automatic_fallback_to_epoch_zero") is not (selected_epoch == 0)
        or pilot.select_best_epoch(history) != selected_epoch
        or result.get("selected_validation")
        != result["history"][selected_epoch]["validation"]
    ):
        raise ValueError("V2.1 source selected epoch differs")

    epoch_zero = history[0]["validation"]["normalized_response"]
    selected = history[selected_epoch]["validation"]["normalized_response"]

    def macro(name: str, rows: Mapping[str, Mapping[str, float]]) -> float:
        return sum(rows[scene][name] for scene in pilot.VALIDATION_SCENES) / len(
            pilot.VALIDATION_SCENES
        )

    absolute_scene_non_regression = all(
        selected[scene]["response_absolute_relevance_loss"]
        <= epoch_zero[scene]["response_absolute_relevance_loss"]
        for scene in pilot.VALIDATION_SCENES
    )
    flags = {
        "selected_epoch_positive": selected_epoch > 0,
        "auxiliary_macro_strictly_improved": macro("response_auxiliary_loss", selected)
        < macro("response_auxiliary_loss", epoch_zero),
        "absolute_relevance_macro_strictly_improved": macro(
            "response_absolute_relevance_loss", selected
        )
        < macro("response_absolute_relevance_loss", epoch_zero),
        "absolute_relevance_every_scene_non_regression": (
            absolute_scene_non_regression
        ),
        "continuous_pairwise_macro_strictly_improved": macro(
            "response_continuous_pairwise_relevance_loss", selected
        )
        < macro("response_continuous_pairwise_relevance_loss", epoch_zero),
        "v1_fidelity_non_regression": history[selected_epoch]["validation"][
            "v1_non_regression"
        ]["non_regression_passed"]
        is True,
    }
    return {
        "schema": PROMOTION_SCHEMA,
        "schema_version": 1,
        "selected_epoch": selected_epoch,
        "checks": flags,
        "passed": all(flags.values()),
        "checkpoint_opened": False,
        "target_execution_authorized": False,
        "benchmark_opened": False,
    }


def _validated_record(value: object, *, label: str) -> dict[str, str]:
    path = validate_file_record(value, label=label)
    assert isinstance(value, Mapping)
    if str(value["path"]) != str(path):
        raise ValueError(f"{label} path must be absolute and canonical")
    return {"path": str(path), "sha256": str(value["sha256"])}


def _validate_input_records(
    value: object,
    *,
    split: str,
    response_records: bool,
) -> list[dict[str, Any]]:
    scenes = pilot.TRAIN_SCENES if split == "source_train" else pilot.VALIDATION_SCENES
    required = {"scene_id", "training_shard", "adaptive_context"}
    if response_records:
        required |= {
            "hard_negative_authority",
            "hard_negative_content_authority_sha256",
        }
    if (
        not isinstance(value, list)
        or len(value) != len(scenes)
        or tuple(
            str(row.get("scene_id")) if isinstance(row, Mapping) else ""
            for row in value
        )
        != scenes
    ):
        raise ValueError(f"V2.1 {split} input record axis differs")
    result: list[dict[str, Any]] = []
    for scene, raw in zip(scenes, value):
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError(f"V2.1 {split} {scene} input fields differ")
        row = {
            "scene_id": scene,
            "training_shard": _file_record(
                raw["training_shard"], label=f"{split} {scene} training shard"
            ),
            "adaptive_context": _file_record(
                raw["adaptive_context"], label=f"{split} {scene} adaptive context"
            ),
        }
        if response_records:
            row["hard_negative_authority"] = _file_record(
                raw["hard_negative_authority"],
                label=f"{split} {scene} hard-negative authority",
            )
            row["hard_negative_content_authority_sha256"] = _sha256(
                raw["hard_negative_content_authority_sha256"],
                label=f"{split} {scene} hard-negative content authority",
            )
        result.append(row)
    return result


def _open_execution_authority(
    record: object,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    verified = _validated_record(record, label="V2.1 source execution authority")
    raw, digest, source = load_json_object(
        verified["path"],
        expected_sha256=verified["sha256"],
        label="V2.1 source execution authority",
    )
    authority = pilot.validate_execution_authority(raw)
    implementation = _validated_record(
        authority["implementation"], label="V2.1 pilot implementation"
    )
    if Path(implementation["path"]) != Path(pilot.__file__).resolve():
        raise ValueError("V2.1 execution authority binds another implementation")
    addendum = _validated_record(
        authority["active_pair_addendum"], label="V2.1 active-pair addendum"
    )
    expected_addendum = (
        Path(pilot.__file__).resolve().parents[2] / pilot.ACTIVE_PAIR_ADDENDUM
    )
    if Path(addendum["path"]) != expected_addendum.resolve():
        raise ValueError("V2.1 execution authority binds another active-pair addendum")
    for name in (
        "cohort_authority",
        "pilot_cohort_region_view_registry",
        "benchmark_exclusion_manifest",
        "fit_text_bank",
        "canonical_negative_bank",
    ):
        _validated_record(authority[name], label=f"V2.1 execution {name}")
    for name in pilot.COMPONENT_WEIGHTS:
        bank = authority["compositional_banks"][name]
        _validated_record(
            {"path": bank["path"], "sha256": bank["sha256"]},
            label=f"V2.1 execution {name}",
        )
    relation = authority["typed_relation_authority"]
    _validated_record(
        {"path": relation["path"], "sha256": relation["sha256"]},
        label="V2.1 execution typed relation authority",
    )
    for split in ("source_train", "source_validation"):
        for row in authority[split]:
            scene = str(row["scene_id"])
            for name in (
                "training_shard",
                "adaptive_context",
                "hard_negative_authority",
            ):
                _validated_record(
                    row[name], label=f"V2.1 execution {split} {scene} {name}"
                )
    registry_record = authority["pilot_cohort_region_view_registry"]
    registry_raw, registry_digest, registry_path = load_json_object(
        registry_record["path"],
        expected_sha256=registry_record["sha256"],
        label="V2.1 pilot cohort region/view registry",
    )
    registry = pilot.pilot_shard.validate_pilot_cohort_region_view_registry(
        registry_raw
    )
    if registry_record != {
        "path": str(registry_path),
        "sha256": registry_digest,
    }:
        raise ValueError("V2.1 execution pilot registry record differs")
    # A promotion gate must not trust file-record existence alone.  Re-run the
    # complete source loader once: this opens and validates the exact cohort,
    # exclusion manifest, registry, six pilot shards, six adaptive contexts,
    # six hard-negative authorities and all frozen text/relation banks.  It is
    # deliberately done before any target authority can be constructed.
    prepared = pilot.prepare_inputs(source, expected_sha256=digest)
    prepared_execution = dict(prepared.execution)
    prepared_execution.pop("verified_path", None)
    prepared_execution.pop("verified_sha256", None)
    if (
        prepared_execution != authority
        or prepared.registry != registry
        or tuple(item.scene_id for item in prepared.train) != pilot.TRAIN_SCENES
        or tuple(item.scene_id for item in prepared.validation)
        != pilot.VALIDATION_SCENES
        or pilot._input_records(prepared.train) != authority["source_train"]
        or pilot._input_records(prepared.validation)
        != authority["source_validation"]
    ):
        raise ValueError("V2.1 complete source input reload differs")
    return authority, {"path": str(source), "sha256": digest}, registry


def validate_normalization_authority(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "fit_split",
        "source_state_cohort_authority_sha256",
        "train_input_records",
        "source_count",
        "median",
        "mad",
        "robust_scale",
        "constant_coordinate_mask",
        "source_max_robust_linf",
        "source_boundary_score_median",
        "validation_contribution",
        "source_access",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("V2.1 normalization fields differ")
    normalization = dict(value)
    if (
        normalization.get("schema") != NORMALIZATION_SCHEMA
        or normalization.get("schema_version") != 1
        or normalization.get("fit_split") != "fixed_source_train_four_only"
        or normalization.get("validation_contribution") is not False
        or normalization.get("source_access") != pilot.source_access()
    ):
        raise ValueError("V2.1 normalization identity differs")
    _sha256(
        normalization["source_state_cohort_authority_sha256"],
        label="V2.1 normalization source cohort authority",
    )
    normalization["train_input_records"] = _validate_input_records(
        normalization["train_input_records"],
        split="source_train",
        response_records=False,
    )
    if (
        not isinstance(normalization["source_count"], int)
        or isinstance(normalization["source_count"], bool)
        or normalization["source_count"] < 2
    ):
        raise ValueError("V2.1 normalization source count differs")
    tensors: dict[str, torch.Tensor] = {}
    for name in ("median", "mad", "robust_scale"):
        raw = normalization[name]
        if not torch.is_tensor(raw):
            raise ValueError(f"V2.1 normalization {name} is not a tensor")
        tensor = raw.detach().cpu().contiguous()
        if (
            tensor.dtype != torch.float32
            or tuple(tensor.shape) != (30,)
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(f"V2.1 normalization {name} differs")
        tensors[name] = tensor
        normalization[name] = tensor
    raw_constant = normalization["constant_coordinate_mask"]
    if not torch.is_tensor(raw_constant):
        raise ValueError("V2.1 normalization constant mask is not a tensor")
    constant = raw_constant.detach().cpu().contiguous()
    if constant.dtype != torch.bool or tuple(constant.shape) != (30,):
        raise ValueError("V2.1 normalization constant mask differs")
    normalization["constant_coordinate_mask"] = constant
    expected_scale = torch.where(
        tensors["mad"] > 0,
        tensors["mad"] * 1.4826,
        torch.ones_like(tensors["mad"]),
    )
    if (
        bool((tensors["mad"] < 0).any())
        or bool((tensors["robust_scale"] <= 0).any())
        or not torch.equal(tensors["robust_scale"], expected_scale)
    ):
        raise ValueError("V2.1 normalization robust scale differs from MAD")
    robust_linf = _finite(
        normalization["source_max_robust_linf"],
        label="V2.1 normalization source robust L-infinity",
    )
    boundary = _finite(
        normalization["source_boundary_score_median"],
        label="V2.1 normalization boundary median",
    )
    if robust_linf < 0.0 or not 0.0 <= boundary <= 1.0:
        raise ValueError("V2.1 normalization scalar bounds differ")
    return normalization


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
        "selected_epoch",
        "selected_validation",
        "model_state_dict_sha256",
        "normalization_authority",
        "normalization_content_authority_sha256",
        "source_access",
        "benchmark_opened",
        "content_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("V2.1 certificate fields differ")
    certificate = dict(value)
    contract = pilot.training_contract()
    if (
        certificate.get("schema") != CERTIFICATE_SCHEMA
        or certificate.get("schema_version") != 1
        or certificate.get("training_contract") != contract
        or certificate.get("training_contract_sha256")
        != canonical_json_sha256(contract)
        or certificate.get("source_access") != pilot.source_access()
        or certificate.get("benchmark_opened") is not False
    ):
        raise ValueError("V2.1 certificate identity differs")
    for name in (
        "execution_authority",
        "cohort_authority",
        "pilot_cohort_region_view_registry",
        "benchmark_exclusion_manifest",
        "normalization_authority",
    ):
        certificate[name] = _file_record(certificate[name], label=f"certificate {name}")
    records = certificate["input_records_by_split"]
    if not isinstance(records, Mapping) or set(records) != {
        "source_train",
        "source_validation",
    }:
        raise ValueError("V2.1 certificate input split axis differs")
    certificate["input_records_by_split"] = {
        split: _validate_input_records(
            records[split], split=split, response_records=True
        )
        for split in ("source_train", "source_validation")
    }
    if (
        not isinstance(certificate["selected_epoch"], int)
        or isinstance(certificate["selected_epoch"], bool)
        or certificate["selected_epoch"] < 0
    ):
        raise ValueError("V2.1 certificate selected epoch differs")
    _validation(certificate["selected_validation"], label="certificate selected")
    _sha256(certificate["model_state_dict_sha256"], label="certificate model state")
    _sha256(
        certificate["pilot_cohort_region_view_registry_authority_sha256"],
        label="certificate pilot registry content authority",
    )
    _sha256(
        certificate["normalization_content_authority_sha256"],
        label="certificate normalization content authority",
    )
    declared_content = _sha256(
        certificate["content_sha256"], label="V2.1 certificate content"
    )
    content = dict(value)
    content.pop("content_sha256")
    if canonical_json_sha256(content) != declared_content:
        raise ValueError("V2.1 certificate content SHA-256 differs")
    return certificate


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
        "selected_epoch",
        "source_access",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("V2.1 checkpoint fields differ")
    checkpoint = dict(value)
    model = SurfaceRegionAcceptedV2TypedContextResidualV1(
        scalar_median=normalization["median"],
        scalar_robust_scale=normalization["robust_scale"],
        max_angle_radians=v1_trainer.MAX_ANGLE_RADIANS,
        max_alpha=v1_trainer.MAX_ALPHA,
    )
    if (
        checkpoint.get("schema") != pilot.CHECKPOINT_SCHEMA
        or checkpoint.get("schema_version") != 1
        or checkpoint.get("model_class") != type(model).__name__
        or checkpoint.get("model_architecture") != model.architecture()
        or checkpoint.get("accepted_v2_authority") != accepted_v2_authority()
        or checkpoint.get("source_access") != pilot.source_access()
        or not isinstance(checkpoint.get("selected_epoch"), int)
        or isinstance(checkpoint.get("selected_epoch"), bool)
        or checkpoint["selected_epoch"] < 0
    ):
        raise ValueError("V2.1 checkpoint identity differs")
    checkpoint["normalization_authority"] = _file_record(
        checkpoint["normalization_authority"], label="checkpoint normalization"
    )
    checkpoint["certificate"] = _file_record(
        checkpoint["certificate"], label="checkpoint certificate"
    )
    state = checkpoint["model_state_dict"]
    expected_state = model.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(expected_state):
        raise ValueError("V2.1 checkpoint state fields differ")
    frozen: dict[str, torch.Tensor] = {}
    for name, expected in expected_state.items():
        raw = state[name]
        if not torch.is_tensor(raw):
            raise ValueError(f"V2.1 checkpoint state {name} is not a tensor")
        tensor = raw.detach().cpu().contiguous()
        if (
            tensor.dtype != expected.dtype
            or tuple(tensor.shape) != tuple(expected.shape)
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(f"V2.1 checkpoint state {name} differs")
        frozen[name] = tensor
    state_sha = surface_region_state_dict_sha256(frozen)
    if state_sha != _sha256(
        checkpoint["model_state_dict_sha256"], label="checkpoint model state"
    ):
        raise ValueError("V2.1 checkpoint model state SHA-256 differs")
    if not torch.equal(
        frozen["scalar_median"], normalization["median"]
    ) or not torch.equal(frozen["scalar_robust_scale"], normalization["robust_scale"]):
        raise ValueError("V2.1 checkpoint normalization buffers differ")
    model.load_state_dict(frozen, strict=True)
    checkpoint["model_state_dict"] = frozen
    return checkpoint


def validate_source_pilot_chain(
    path: str | Path,
    *,
    expected_sha256: str,
    require_promotion: bool = True,
) -> dict[str, Any]:
    """Validate the complete source-only chain before any target authority.

    ``require_promotion=True`` rejects epoch zero and any failed semantic
    promotion check.  The returned value is evidence only and deliberately
    does not by itself authorize a target execution.
    """

    raw, result_sha, result_path = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="V2.1 source pilot result",
    )
    promotion = validate_source_promotion_evidence(raw)
    execution, execution_record, pilot_registry = _open_execution_authority(
        raw["execution_authority"]
    )

    normalization_record = _validated_record(
        raw["normalization_authority"], label="V2.1 source normalization"
    )
    normalization_raw, normalization_sha, normalization_path = load_torch_mapping(
        normalization_record["path"],
        expected_sha256=normalization_record["sha256"],
        map_location="cpu",
        label="V2.1 source normalization",
    )
    normalization = validate_normalization_authority(normalization_raw)

    certificate_record = _validated_record(
        raw["certificate"], label="V2.1 source certificate"
    )
    certificate_raw, certificate_sha, certificate_path = load_json_object(
        certificate_record["path"],
        expected_sha256=certificate_record["sha256"],
        label="V2.1 source certificate",
    )
    certificate = validate_certificate(certificate_raw)

    checkpoint_record = _validated_record(
        raw["checkpoint"], label="V2.1 source checkpoint"
    )
    checkpoint_raw, checkpoint_sha, checkpoint_path = load_torch_mapping(
        checkpoint_record["path"],
        expected_sha256=checkpoint_record["sha256"],
        map_location="cpu",
        label="V2.1 source checkpoint",
    )
    checkpoint = validate_checkpoint_payload(
        checkpoint_raw, normalization=normalization
    )

    actual_normalization_record = {
        "path": str(normalization_path),
        "sha256": normalization_sha,
    }
    actual_certificate_record = {
        "path": str(certificate_path),
        "sha256": certificate_sha,
    }
    actual_checkpoint_record = {
        "path": str(checkpoint_path),
        "sha256": checkpoint_sha,
    }
    selected_epoch = int(raw["selected_epoch"])
    selected_state_sha = str(raw["history"][selected_epoch]["model_state_dict_sha256"])
    execution_train = execution["source_train"]
    normalization_train = [
        {
            "scene_id": row["scene_id"],
            "training_shard": row["training_shard"],
            "adaptive_context": row["adaptive_context"],
        }
        for row in execution_train
    ]
    pilot_source_manifest, _pilot_teacher_manifest = (
        pilot.pilot_shard.derive_pilot_global_manifests(pilot_registry)
    )
    if (
        raw["execution_authority"] != execution_record
        or raw["normalization_authority"] != actual_normalization_record
        or raw["certificate"] != actual_certificate_record
        or raw["checkpoint"] != actual_checkpoint_record
        or certificate["execution_authority"] != execution_record
        or certificate["cohort_authority"] != execution["cohort_authority"]
        or certificate["pilot_cohort_region_view_registry"]
        != execution["pilot_cohort_region_view_registry"]
        or certificate["pilot_cohort_region_view_registry_authority_sha256"]
        != pilot_registry["authority_sha256"]
        or certificate["benchmark_exclusion_manifest"]
        != execution["benchmark_exclusion_manifest"]
        or certificate["input_records_by_split"]["source_train"]
        != execution["source_train"]
        or certificate["input_records_by_split"]["source_validation"]
        != execution["source_validation"]
        or normalization["train_input_records"] != normalization_train
        or normalization["source_state_cohort_authority_sha256"]
        != pilot_source_manifest["authority_sha256"]
        or certificate["normalization_authority"] != actual_normalization_record
        or checkpoint["normalization_authority"] != actual_normalization_record
        or checkpoint["certificate"] != actual_certificate_record
        or checkpoint["selected_epoch"] != selected_epoch
        or certificate["selected_epoch"] != selected_epoch
        or certificate["selected_validation"] != raw["selected_validation"]
        or checkpoint["model_state_dict_sha256"] != selected_state_sha
        or certificate["model_state_dict_sha256"] != selected_state_sha
        or certificate["normalization_content_authority_sha256"]
        != pilot.pilot_normalization_authority_sha256(normalization)
    ):
        raise ValueError(
            "V2.1 result/checkpoint/certificate/normalization chain differs"
        )
    if require_promotion and promotion["passed"] is not True:
        failed = [name for name, passed in promotion["checks"].items() if not passed]
        raise ValueError("V2.1 source promotion did not pass: " + ", ".join(failed))
    return {
        "schema": PROMOTION_CHAIN_SCHEMA,
        "schema_version": 1,
        "source_result": {"path": str(result_path), "sha256": result_sha},
        "execution_authority": execution_record,
        "normalization_authority": actual_normalization_record,
        "certificate": actual_certificate_record,
        "checkpoint": actual_checkpoint_record,
        "selected_epoch": selected_epoch,
        "model_state_dict_sha256": selected_state_sha,
        "normalization_content_authority_sha256": certificate[
            "normalization_content_authority_sha256"
        ],
        "promotion": promotion,
        "source_promotion_authorized": promotion["passed"] is True,
        "target_execution_authorized": False,
        "benchmark_opened": False,
    }


__all__ = [
    "CERTIFICATE_SCHEMA",
    "NORMALIZATION_SCHEMA",
    "PROMOTION_CHAIN_SCHEMA",
    "PROMOTION_SCHEMA",
    "RESULT_SCHEMA",
    "validate_certificate",
    "validate_checkpoint_payload",
    "validate_normalization_authority",
    "validate_source_pilot_chain",
    "validate_source_promotion_evidence",
]
