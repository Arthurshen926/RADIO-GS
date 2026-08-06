"""Target-blind signed-scribble cross-validation for propagation selection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import torch

from radio_gs.querying.source_footprint_fold_authority import (
    SourceFoldBaseDecision,
    SourceFootprintFoldAuthority,
    clear_four_source_evidence_for_fold,
    source_fold_population_base_decision,
    splitmix64_source_group_folds,
)


ACTION_STRONG_UNARY = "strong_unary"
ACTION_HASH256_DIFFUSION = "hash256_fixed_f2_g4_k201_diffusion"
REGISTERED_ACTIONS = (ACTION_STRONG_UNARY, ACTION_HASH256_DIFFUSION)

ACTION_SOURCE_UNARY = "unary"
ACTION_SURFACE_SAFE_PROPAGATED = "surface_safe_propagated"
SOURCE_OBSERVATION_ACTIONS = (
    ACTION_SOURCE_UNARY,
    ACTION_SURFACE_SAFE_PROPAGATED,
)


@dataclass(frozen=True)
class SourceObservationOOFFold:
    """Leak-free prompt masses and their immutable evaluation authority."""

    fold_ids: torch.Tensor
    observed: torch.Tensor
    heldout: torch.Tensor
    signed_reference_evidence: torch.Tensor
    reference_weight: torch.Tensor
    training_positive_weight: torch.Tensor
    training_negative_weight: torch.Tensor
    training_raw_positive_mass: torch.Tensor
    training_raw_negative_mass: torch.Tensor


@dataclass(frozen=True)
class SourceObservationOOFGateResult:
    """Strict three-fold decision reconstructed from sealed fold artifacts."""

    selected_action: str
    metrics: dict[str, dict[str, float]]
    fold_reports: list[dict[str, object]]
    fold_ids: torch.Tensor
    observed: torch.Tensor
    oof_predictions: dict[str, torch.Tensor]
    scene_id: str
    protocol_hash: str
    method_contract_sha256: str
    capability_cache_sha256: str
    support_graph_sha256: str
    source_evidence_authority_sha256: str
    source_evidence_authority_content_sha256: str


def prepare_source_observation_oof_fold(
    global_rows: torch.Tensor,
    valid: torch.Tensor,
    positive_weight: torch.Tensor,
    negative_weight: torch.Tensor,
    raw_positive_mass: torch.Tensor,
    raw_negative_mass: torch.Tensor,
    *,
    heldout_fold: int,
    num_folds: int = 3,
) -> SourceObservationOOFFold:
    """Clear one deterministic prompt-evidence fold before query compilation.

    All inputs are aligned to the complete primitive-row domain.  Labels and
    weights are retained from the unmodified raster-adjoint observation, while
    every prompt-derived compiler input is zeroed on held-out valid rows.
    """

    rows = torch.as_tensor(global_rows).detach().long().cpu().reshape(-1)
    valid_cpu = torch.as_tensor(valid).detach().bool().cpu().reshape(-1)
    inputs = [
        torch.as_tensor(value).detach().float().cpu().reshape(-1)
        for value in (
            positive_weight,
            negative_weight,
            raw_positive_mass,
            raw_negative_mass,
        )
    ]
    if not 0 <= int(heldout_fold) < int(num_folds):
        raise ValueError("held-out fold is outside the registered fold assignment")
    if valid_cpu.numel() == 0 or any(value.shape != valid_cpu.shape for value in inputs):
        raise ValueError("OOF prompt masses must align with the global row domain")
    if rows.numel() != int(valid_cpu.sum()) or not torch.equal(
        rows, torch.where(valid_cpu)[0]
    ):
        raise ValueError("OOF global rows must equal the sorted valid-row authority")
    if any(
        not bool(torch.isfinite(value).all()) or bool((value < 0).any())
        for value in inputs
    ):
        raise ValueError("OOF prompt masses must be finite and non-negative")

    valid_folds = stable_primitive_folds(rows, num_folds=int(num_folds))
    fold_ids = torch.full(valid_cpu.shape, -1, dtype=torch.long)
    fold_ids[rows] = valid_folds
    heldout = valid_cpu & (fold_ids == int(heldout_fold))
    positive, negative, raw_positive, raw_negative = inputs
    signed = raw_positive - raw_negative
    reference_weight = raw_positive + raw_negative
    observed = valid_cpu & (reference_weight > 0) & (signed != 0)
    training = [value.clone() for value in inputs]
    for value in training:
        value[heldout] = 0
        if bool((value[heldout] != 0).any()):
            raise RuntimeError("held-out prompt evidence survived OOF clearing")
    return SourceObservationOOFFold(
        fold_ids=fold_ids,
        observed=observed,
        heldout=heldout,
        signed_reference_evidence=signed,
        reference_weight=reference_weight,
        training_positive_weight=training[0],
        training_negative_weight=training[1],
        training_raw_positive_mass=training[2],
        training_raw_negative_mass=training[3],
    )


def prepare_source_observation_footprint_oof_fold(
    footprint_authority: SourceFootprintFoldAuthority,
    global_rows: torch.Tensor,
    valid: torch.Tensor,
    positive_weight: torch.Tensor,
    negative_weight: torch.Tensor,
    raw_positive_mass: torch.Tensor,
    raw_negative_mass: torch.Tensor,
    *,
    heldout_fold: int,
    expected_footprint_authority_sha256: str,
) -> tuple[SourceObservationOOFFold | None, SourceFoldBaseDecision]:
    """Clear one complete source-footprint fold on the global row domain.

    The label-free footprint groups must already be frozen before this
    function is called.  Only then are source prompt masses inspected to
    decide whether every structured training/held-out population contains
    both classes.  A degenerate population returns the registered field-base
    decision without constructing a partially valid OOF fold.
    """

    footprint_authority.validate(
        expected_authority_sha256=expected_footprint_authority_sha256
    )
    rows = torch.as_tensor(global_rows).detach().long().cpu().reshape(-1)
    valid_cpu = torch.as_tensor(valid).detach().bool().cpu().reshape(-1)
    inputs = [
        torch.as_tensor(value).detach().float().cpu().reshape(-1)
        for value in (
            positive_weight,
            negative_weight,
            raw_positive_mass,
            raw_negative_mass,
        )
    ]
    if int(heldout_fold) not in (0, 1, 2):
        raise ValueError("held-out footprint fold must be 0, 1, or 2")
    if valid_cpu.numel() == 0 or any(value.shape != valid_cpu.shape for value in inputs):
        raise ValueError("footprint OOF prompt masses must align with the global row domain")
    valid_rows = torch.where(valid_cpu)[0]
    if (
        rows.numel() != int(valid_cpu.sum())
        or not torch.equal(rows, valid_rows)
        or not torch.equal(footprint_authority.primitive_rows, rows)
    ):
        raise ValueError(
            "footprint rows, capability global_rows, and sorted valid rows must match exactly"
        )
    if any(
        not bool(torch.isfinite(value).all()) or bool((value < 0).any())
        for value in inputs
    ):
        raise ValueError("footprint OOF prompt masses must be finite and non-negative")

    positive, negative, raw_positive, raw_negative = inputs
    decision = source_fold_population_base_decision(
        footprint_authority,
        positive[rows],
        negative[rows],
        expected_authority_sha256=expected_footprint_authority_sha256,
    )
    if not decision.run_source_oof:
        return None, decision

    cleared = clear_four_source_evidence_for_fold(
        footprint_authority,
        positive[rows],
        negative[rows],
        raw_positive[rows],
        raw_negative[rows],
        heldout_fold=int(heldout_fold),
        expected_authority_sha256=expected_footprint_authority_sha256,
    )
    local_fold_ids = splitmix64_source_group_folds(footprint_authority.group_ids)
    if not torch.equal(cleared.fold_ids, local_fold_ids):
        raise RuntimeError("cleared footprint folds differ from frozen group authority")
    fold_ids = torch.full(valid_cpu.shape, -1, dtype=torch.long)
    fold_ids[rows] = local_fold_ids
    heldout = torch.zeros(valid_cpu.shape, dtype=torch.bool)
    heldout[rows] = cleared.heldout_rows

    signed = raw_positive - raw_negative
    reference_weight = raw_positive + raw_negative
    observed = valid_cpu & (reference_weight > 0) & (signed != 0)
    training = [value.clone() for value in inputs]
    local_training = (
        cleared.training_positive_weight,
        cleared.training_negative_weight,
        cleared.training_raw_positive_mass,
        cleared.training_raw_negative_mass,
    )
    for full_value, local_value in zip(training, local_training):
        full_value[rows] = local_value.to(dtype=full_value.dtype)
        if bool((full_value[heldout] != 0).any()):
            raise RuntimeError("held-out footprint evidence survived OOF clearing")

    return (
        SourceObservationOOFFold(
            fold_ids=fold_ids,
            observed=observed,
            heldout=heldout,
            signed_reference_evidence=signed,
            reference_weight=reference_weight,
            training_positive_weight=training[0],
            training_negative_weight=training[1],
            training_raw_positive_mass=training[2],
            training_raw_negative_mass=training[3],
        ),
        decision,
    )


def _select_source_observation_action(
    metrics: Mapping[str, Mapping[str, float]],
    *,
    metric_round_decimals: int,
) -> str:
    if set(metrics) != set(SOURCE_OBSERVATION_ACTIONS):
        raise ValueError("source-observation metrics differ from registered actions")
    ranked: dict[str, tuple[float, float, int]] = {}
    for action in SOURCE_OBSERVATION_ACTIONS:
        values = metrics[action]
        if set(values) != {
            "responsibility_balanced_log_loss",
            "responsibility_weighted_auc",
        }:
            raise ValueError("source-observation metric schema differs")
        loss = float(values["responsibility_balanced_log_loss"])
        auc = float(values["responsibility_weighted_auc"])
        if not np.isfinite(loss) or loss < 0 or not np.isfinite(auc) or not 0 <= auc <= 1:
            raise ValueError("source-observation metric value is invalid")
        ranked[action] = (
            round(loss, int(metric_round_decimals)),
            -round(auc, int(metric_round_decimals)),
            0 if action == ACTION_SOURCE_UNARY else 1,
        )
    return min(SOURCE_OBSERVATION_ACTIONS, key=lambda action: ranked[action])


def evaluate_source_observation_oof_artifacts(
    fold_payloads: Mapping[int, Mapping[str, object]],
    *,
    num_folds: int = 3,
    minimum_class_rows: int = 32,
    metric_round_decimals: int = 12,
    probability_epsilon: float = 1e-7,
) -> SourceObservationOOFGateResult:
    """Validate and combine immutable source-only fold artifacts.

    Each payload must have been produced by an independent compiler execution
    after clearing all four prompt-evidence tensors on its held-out fold.  This
    function deliberately reads predictions only on held-out observed rows;
    full-fit or training-fold predictions can therefore never enter the gate.
    """

    fold_count = int(num_folds)
    if fold_count != 3 or set(fold_payloads) != set(range(fold_count)):
        raise ValueError("source-observation gate requires exactly folds 0, 1, and 2")

    tensor_names = (
        "valid",
        "global_rows",
        "fold_ids",
        "observed",
        "heldout",
        "signed_reference_evidence",
        "reference_weight",
        "unary_probability",
        "surface_safe_propagated_probability",
    )
    invariant_tensor_names = (
        "valid",
        "global_rows",
        "fold_ids",
        "observed",
        "signed_reference_evidence",
        "reference_weight",
    )
    contract_names = (
        "scene_id",
        "protocol_hash",
        "method_contract_sha256",
        "capability_cache_sha256",
        "support_graph_sha256",
        "source_evidence_authority_sha256",
        "source_evidence_authority_content_sha256",
    )
    reference_payload = fold_payloads[0]
    reference_tensors: dict[str, torch.Tensor] = {}
    for name in invariant_tensor_names:
        if name not in reference_payload:
            raise ValueError(f"OOF fold artifact lacks tensor {name!r}")
        reference_tensors[name] = torch.as_tensor(reference_payload[name]).detach().cpu()

    valid = reference_tensors["valid"].bool().reshape(-1)
    rows = reference_tensors["global_rows"].long().reshape(-1)
    fold_ids = reference_tensors["fold_ids"].long().reshape(-1)
    observed = reference_tensors["observed"].bool().reshape(-1)
    signed = reference_tensors["signed_reference_evidence"].float().reshape(-1)
    weight = reference_tensors["reference_weight"].double().reshape(-1)
    full_shape = valid.shape
    if (
        rows.numel() != int(valid.sum())
        or not torch.equal(rows, torch.where(valid)[0])
        or any(value.reshape(-1).shape != full_shape for name, value in reference_tensors.items() if name != "global_rows")
    ):
        raise ValueError("OOF invariant tensors do not align with valid global rows")
    expected_valid_folds = stable_primitive_folds(rows, num_folds=fold_count)
    if not torch.equal(fold_ids[valid], expected_valid_folds) or bool((fold_ids[~valid] != -1).any()):
        raise ValueError("OOF fold ids differ from SplitMix64 global-row authority")
    expected_observed = valid & (weight > 0) & (signed != 0)
    if not torch.equal(observed, expected_observed):
        raise ValueError("OOF observed population differs from signed responsibility evidence")
    if not bool(torch.isfinite(signed).all()) or not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
        raise ValueError("OOF responsibility evidence is invalid")

    _, fold_reports = audit_signed_cv_population(
        rows,
        signed[valid],
        weight[valid],
        num_folds=fold_count,
        minimum_class_rows=int(minimum_class_rows),
    )
    oof = {
        action: torch.full(full_shape, float("nan"), dtype=torch.float32)
        for action in SOURCE_OBSERVATION_ACTIONS
    }
    probability_tensor = {
        ACTION_SOURCE_UNARY: "unary_probability",
        ACTION_SURFACE_SAFE_PROPAGATED: "surface_safe_propagated_probability",
    }

    for heldout_fold in range(fold_count):
        payload = fold_payloads[heldout_fold]
        if payload.get("artifact_type") != "source_observation_surface_safe_oof_fold_v1":
            raise ValueError("source-observation fold artifact type differs")
        if int(payload.get("heldout_fold", -1)) != heldout_fold or int(payload.get("num_folds", -1)) != fold_count:
            raise ValueError("source-observation fold identity differs")
        if any(bool(payload.get(flag, True)) for flag in ("target_rgb_opened", "target_mask_opened", "target_metric_computed")):
            raise ValueError("source-observation fold was not sealed before target access")
        if any(payload.get(name) != reference_payload.get(name) for name in contract_names):
            raise ValueError("source-observation fold provenance differs")
        if any(name not in payload for name in tensor_names):
            raise ValueError("source-observation fold lacks registered tensors")
        tensor_hashes = payload.get("tensor_sha256")
        if not isinstance(tensor_hashes, Mapping) or set(tensor_hashes) != set(tensor_names):
            raise ValueError("source-observation fold tensor hash authority differs")
        for name in tensor_names:
            value = torch.as_tensor(payload[name]).detach().cpu().contiguous()
            digest = hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()
            if str(tensor_hashes[name]) != digest:
                raise ValueError(f"source-observation fold tensor {name!r} changed")
        for name in invariant_tensor_names:
            candidate = torch.as_tensor(payload[name]).detach().cpu()
            if candidate.dtype != reference_tensors[name].dtype or not torch.equal(candidate, reference_tensors[name]):
                raise ValueError(f"source-observation fold invariant {name!r} differs")
        heldout = torch.as_tensor(payload["heldout"]).detach().bool().cpu().reshape(-1)
        expected_heldout = valid & (fold_ids == heldout_fold)
        if not torch.equal(heldout, expected_heldout):
            raise ValueError("source-observation held-out mask differs from fold authority")
        cleared = payload.get("heldout_prompt_evidence_after_clear")
        if not isinstance(cleared, Mapping) or set(cleared) != {
            "positive_weight_sum",
            "negative_weight_sum",
            "raw_positive_mass_sum",
            "raw_negative_mass_sum",
        } or any(float(value) != 0.0 for value in cleared.values()):
            raise ValueError("held-out prompt evidence was not exactly cleared")
        evaluation_rows = observed & heldout
        for action, tensor_name in probability_tensor.items():
            probability = torch.as_tensor(payload[tensor_name]).detach().float().cpu().reshape(-1)
            if probability.shape != full_shape or not bool(torch.isfinite(probability).all()) or bool(((probability < 0) | (probability > 1)).any()):
                raise ValueError(f"source-observation {action} probability is invalid")
            oof[action][evaluation_rows] = probability[evaluation_rows]

    if any(not bool(torch.isfinite(values[observed]).all()) for values in oof.values()):
        raise RuntimeError("source-observation gate lacks held-out predictions")
    labels = signed[observed] > 0
    observed_weight = weight[observed]
    metrics: dict[str, dict[str, float]] = {}
    for action in SOURCE_OBSERVATION_ACTIONS:
        probability = oof[action][observed]
        metrics[action] = {
            "responsibility_balanced_log_loss": responsibility_balanced_log_loss(
                labels,
                probability,
                observed_weight,
                probability_epsilon=float(probability_epsilon),
            ),
            "responsibility_weighted_auc": responsibility_weighted_auc(
                labels,
                probability,
                observed_weight,
            ),
        }
    selected = _select_source_observation_action(
        metrics,
        metric_round_decimals=int(metric_round_decimals),
    )
    return SourceObservationOOFGateResult(
        selected_action=selected,
        metrics=metrics,
        fold_reports=fold_reports,
        fold_ids=fold_ids,
        observed=observed,
        oof_predictions=oof,
        scene_id=str(reference_payload["scene_id"]),
        protocol_hash=str(reference_payload["protocol_hash"]),
        method_contract_sha256=str(reference_payload["method_contract_sha256"]),
        capability_cache_sha256=str(reference_payload["capability_cache_sha256"]),
        support_graph_sha256=str(reference_payload["support_graph_sha256"]),
        source_evidence_authority_sha256=str(
            reference_payload["source_evidence_authority_sha256"]
        ),
        source_evidence_authority_content_sha256=str(
            reference_payload["source_evidence_authority_content_sha256"]
        ),
    )


def evaluate_source_observation_footprint_oof_artifacts(
    fold_payloads: Mapping[int, Mapping[str, object]],
    *,
    footprint_authority: SourceFootprintFoldAuthority,
    footprint_authority_path: str,
    footprint_authority_file_sha256: str,
    metric_round_decimals: int = 12,
    probability_epsilon: float = 1e-7,
) -> SourceObservationOOFGateResult:
    """Validate structured source-footprint OOF artifacts without row CV.

    Unlike the legacy evaluator, fold membership is reconstructed exclusively
    from the frozen source-footprint groups.  Source labels are inspected only
    after that authority has been validated, and the legacy row-population
    audit is deliberately not called.
    """

    if set(fold_payloads) != {0, 1, 2}:
        raise ValueError("source-footprint gate requires exactly folds 0, 1, and 2")
    footprint_authority.validate(
        expected_authority_sha256=footprint_authority.authority_sha256
    )
    expected_lineage = {
        "source_footprint_fold_authority": str(footprint_authority_path),
        "source_footprint_fold_authority_file_sha256": str(
            footprint_authority_file_sha256
        ),
        "source_footprint_fold_authority_sha256": (
            footprint_authority.authority_sha256
        ),
        "source_footprint_fold_authority_tensor_bundle_sha256": (
            footprint_authority.tensor_bundle_sha256
        ),
    }
    tensor_names = (
        "valid",
        "global_rows",
        "fold_ids",
        "observed",
        "heldout",
        "signed_reference_evidence",
        "reference_weight",
        "population_positive_weight",
        "population_negative_weight",
        "unary_probability",
        "surface_safe_propagated_probability",
    )
    invariant_tensor_names = (
        "valid",
        "global_rows",
        "fold_ids",
        "observed",
        "signed_reference_evidence",
        "reference_weight",
        "population_positive_weight",
        "population_negative_weight",
    )
    contract_names = (
        "scene_id",
        "protocol_hash",
        "method_contract_sha256",
        "capability_cache_sha256",
        "support_graph_sha256",
        "source_evidence_authority_sha256",
        "source_evidence_authority_content_sha256",
        *expected_lineage.keys(),
    )
    reference_payload = fold_payloads[0]
    for name, expected in expected_lineage.items():
        if reference_payload.get(name) != expected:
            raise ValueError(f"source-footprint fold lineage {name!r} differs")
    reference_tensors: dict[str, torch.Tensor] = {}
    for name in invariant_tensor_names:
        if name not in reference_payload:
            raise ValueError(f"source-footprint fold lacks tensor {name!r}")
        reference_tensors[name] = torch.as_tensor(reference_payload[name]).detach().cpu()

    valid = reference_tensors["valid"].bool().reshape(-1)
    rows = reference_tensors["global_rows"].long().reshape(-1)
    fold_ids = reference_tensors["fold_ids"].long().reshape(-1)
    observed = reference_tensors["observed"].bool().reshape(-1)
    signed = reference_tensors["signed_reference_evidence"].float().reshape(-1)
    weight = reference_tensors["reference_weight"].double().reshape(-1)
    population_positive = reference_tensors["population_positive_weight"].double().reshape(-1)
    population_negative = reference_tensors["population_negative_weight"].double().reshape(-1)
    full_shape = valid.shape
    if (
        rows.numel() != int(valid.sum())
        or not torch.equal(rows, torch.where(valid)[0])
        or not torch.equal(rows, footprint_authority.primitive_rows)
        or any(
            value.reshape(-1).shape != full_shape
            for name, value in reference_tensors.items()
            if name != "global_rows"
        )
    ):
        raise ValueError(
            "source-footprint rows, capability global_rows, and valid rows differ"
        )
    expected_fold_ids = torch.full(full_shape, -1, dtype=torch.long)
    expected_fold_ids[rows] = splitmix64_source_group_folds(
        footprint_authority.group_ids
    )
    if not torch.equal(fold_ids, expected_fold_ids):
        raise ValueError("OOF fold ids differ from source-footprint group authority")
    expected_observed = valid & (weight > 0) & (signed != 0)
    if not torch.equal(observed, expected_observed):
        raise ValueError("source-footprint observed population differs")
    if any(
        not bool(torch.isfinite(value).all()) or bool((value < 0).any())
        for value in (weight, population_positive, population_negative)
    ) or not bool(torch.isfinite(signed).all()):
        raise ValueError("source-footprint responsibility evidence is invalid")

    population_decision = source_fold_population_base_decision(
        footprint_authority,
        population_positive[rows],
        population_negative[rows],
        expected_authority_sha256=footprint_authority.authority_sha256,
    )
    if not population_decision.run_source_oof:
        raise ValueError("structured OOF artifacts exist for a degenerate population")
    fold_reports = list(population_decision.fold_reports)
    oof = {
        action: torch.full(full_shape, float("nan"), dtype=torch.float32)
        for action in SOURCE_OBSERVATION_ACTIONS
    }
    probability_tensor = {
        ACTION_SOURCE_UNARY: "unary_probability",
        ACTION_SURFACE_SAFE_PROPAGATED: "surface_safe_propagated_probability",
    }

    for heldout_fold in range(3):
        payload = fold_payloads[heldout_fold]
        if payload.get("artifact_type") != "source_observation_surface_safe_footprint_oof_fold_v1":
            raise ValueError("source-footprint fold artifact type differs")
        if payload.get("fold_assignment") != "splitmix64_source_footprint_group_v1":
            raise ValueError("source-footprint fold assignment differs")
        if int(payload.get("heldout_fold", -1)) != heldout_fold or int(payload.get("num_folds", -1)) != 3:
            raise ValueError("source-footprint fold identity differs")
        if any(
            bool(payload.get(flag, True))
            for flag in (
                "target_rgb_opened",
                "target_mask_opened",
                "target_metric_computed",
            )
        ):
            raise ValueError("source-footprint fold was not sealed before target access")
        if any(payload.get(name) != reference_payload.get(name) for name in contract_names):
            raise ValueError("source-footprint fold provenance differs")
        if any(name not in payload for name in tensor_names):
            raise ValueError("source-footprint fold lacks registered tensors")
        tensor_hashes = payload.get("tensor_sha256")
        if not isinstance(tensor_hashes, Mapping) or set(tensor_hashes) != set(tensor_names):
            raise ValueError("source-footprint fold tensor hash authority differs")
        for name in tensor_names:
            value = torch.as_tensor(payload[name]).detach().cpu().contiguous()
            digest = hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()
            if str(tensor_hashes[name]) != digest:
                raise ValueError(f"source-footprint fold tensor {name!r} changed")
        for name in invariant_tensor_names:
            candidate = torch.as_tensor(payload[name]).detach().cpu()
            if candidate.dtype != reference_tensors[name].dtype or not torch.equal(
                candidate, reference_tensors[name]
            ):
                raise ValueError(f"source-footprint invariant {name!r} differs")
        heldout = torch.as_tensor(payload["heldout"]).detach().bool().cpu().reshape(-1)
        expected_heldout = valid & (fold_ids == heldout_fold)
        if not torch.equal(heldout, expected_heldout):
            raise ValueError("source-footprint held-out mask differs from group authority")
        cleared = payload.get("heldout_prompt_evidence_after_clear")
        if not isinstance(cleared, Mapping) or set(cleared) != {
            "positive_weight_sum",
            "negative_weight_sum",
            "raw_positive_mass_sum",
            "raw_negative_mass_sum",
        } or any(float(value) != 0.0 for value in cleared.values()):
            raise ValueError("held-out source-footprint evidence was not exactly cleared")
        evaluation_rows = observed & heldout
        for action, tensor_name in probability_tensor.items():
            probability = torch.as_tensor(payload[tensor_name]).detach().float().cpu().reshape(-1)
            if probability.shape != full_shape or not bool(torch.isfinite(probability).all()) or bool(
                ((probability < 0) | (probability > 1)).any()
            ):
                raise ValueError(f"source-footprint {action} probability is invalid")
            oof[action][evaluation_rows] = probability[evaluation_rows]

    if any(not bool(torch.isfinite(values[observed]).all()) for values in oof.values()):
        raise RuntimeError("source-footprint gate lacks held-out predictions")
    labels = signed[observed] > 0
    observed_weight = weight[observed]
    metrics: dict[str, dict[str, float]] = {}
    for action in SOURCE_OBSERVATION_ACTIONS:
        probability = oof[action][observed]
        metrics[action] = {
            "responsibility_balanced_log_loss": responsibility_balanced_log_loss(
                labels,
                probability,
                observed_weight,
                probability_epsilon=float(probability_epsilon),
            ),
            "responsibility_weighted_auc": responsibility_weighted_auc(
                labels,
                probability,
                observed_weight,
            ),
        }
    selected = _select_source_observation_action(
        metrics,
        metric_round_decimals=int(metric_round_decimals),
    )
    return SourceObservationOOFGateResult(
        selected_action=selected,
        metrics=metrics,
        fold_reports=fold_reports,
        fold_ids=fold_ids,
        observed=observed,
        oof_predictions=oof,
        scene_id=str(reference_payload["scene_id"]),
        protocol_hash=str(reference_payload["protocol_hash"]),
        method_contract_sha256=str(reference_payload["method_contract_sha256"]),
        capability_cache_sha256=str(reference_payload["capability_cache_sha256"]),
        support_graph_sha256=str(reference_payload["support_graph_sha256"]),
        source_evidence_authority_sha256=str(
            reference_payload["source_evidence_authority_sha256"]
        ),
        source_evidence_authority_content_sha256=str(
            reference_payload["source_evidence_authority_content_sha256"]
        ),
    )


def select_registered_action(
    metrics: dict[str, dict[str, float]], *, metric_round_decimals: int = 12
) -> str:
    """Apply the registered log-loss/AUC/complexity lexicographic rule."""

    if set(metrics) != set(REGISTERED_ACTIONS):
        raise ValueError("selection metrics must contain exactly the registered actions")
    rounded = {}
    for action in REGISTERED_ACTIONS:
        values = metrics[action]
        if set(values) != {
            "responsibility_balanced_log_loss",
            "responsibility_weighted_auc",
        }:
            raise ValueError("selection metric schema differs")
        loss = float(values["responsibility_balanced_log_loss"])
        auc = float(values["responsibility_weighted_auc"])
        if not np.isfinite(loss) or not np.isfinite(auc) or loss < 0 or not 0 <= auc <= 1:
            raise ValueError("selection metric value is invalid")
        rounded[action] = (
            round(loss, int(metric_round_decimals)),
            -round(auc, int(metric_round_decimals)),
            0 if action == ACTION_STRONG_UNARY else 1,
        )
    return min(REGISTERED_ACTIONS, key=lambda action: rounded[action])


def stable_primitive_folds(global_rows: torch.Tensor, *, num_folds: int = 3) -> torch.Tensor:
    """Assign global primitive ids with a platform-stable SplitMix64 hash."""

    rows = torch.as_tensor(global_rows).detach().long().cpu().reshape(-1)
    if int(num_folds) < 2 or rows.numel() == 0 or bool((rows < 0).any()):
        raise ValueError("fold assignment requires non-negative rows and >=2 folds")
    if rows.unique().numel() != rows.numel():
        raise ValueError("fold assignment requires unique global rows")
    values = rows.numpy().astype(np.uint64, copy=True)
    values += np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    values = (values ^ (values >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    values ^= values >> np.uint64(31)
    folds = (values % np.uint64(int(num_folds))).astype(np.int64, copy=False)
    return torch.from_numpy(folds.copy())


def training_evidence_for_fold(
    signed_reference_evidence: torch.Tensor,
    fold_ids: torch.Tensor,
    *,
    heldout_fold: int,
) -> torch.Tensor:
    """Remove every held-out signed anchor before classifier and diffusion fit."""

    evidence = torch.as_tensor(signed_reference_evidence).detach().float().cpu().reshape(-1)
    folds = torch.as_tensor(fold_ids).detach().long().cpu().reshape(-1)
    if evidence.shape != folds.shape or not bool(torch.isfinite(evidence).all()):
        raise ValueError("evidence and fold ids must be finite and aligned")
    if int(heldout_fold) < 0 or int(heldout_fold) >= int(folds.max()) + 1:
        raise ValueError("held-out fold is outside the fold assignment")
    training = evidence.clone()
    training[folds == int(heldout_fold)] = 0
    return training


def audit_signed_cv_population(
    global_rows: torch.Tensor,
    signed_reference_evidence: torch.Tensor,
    reference_weight: torch.Tensor,
    *,
    num_folds: int = 3,
    minimum_class_rows: int = 32,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    """Fail closed unless every held-out and training fold has both classes."""

    rows = torch.as_tensor(global_rows).detach().long().cpu().reshape(-1)
    evidence = torch.as_tensor(signed_reference_evidence).detach().float().cpu().reshape(-1)
    weights = torch.as_tensor(reference_weight).detach().double().cpu().reshape(-1)
    if rows.shape != evidence.shape or rows.shape != weights.shape:
        raise ValueError("CV rows, evidence, and weights must align")
    if not bool(torch.isfinite(evidence).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("CV evidence and weights must be finite")
    if bool((weights < 0).any()):
        raise ValueError("CV responsibility weights cannot be negative")
    observed = evidence != 0
    if not bool(observed.any()) or not bool((weights[observed] > 0).all()):
        raise ValueError("every CV anchor needs positive responsibility")
    folds = stable_primitive_folds(rows, num_folds=int(num_folds))
    labels = evidence > 0
    minimum = int(minimum_class_rows)
    if minimum <= 0:
        raise ValueError("minimum_class_rows must be positive")
    reports: list[dict[str, object]] = []
    for fold in range(int(num_folds)):
        heldout = observed & (folds == fold)
        training = observed & (folds != fold)
        report: dict[str, object] = {"fold": fold}
        for population_name, mask in (("heldout", heldout), ("training", training)):
            positive = mask & labels
            negative = mask & ~labels
            positive_count = int(positive.sum())
            negative_count = int(negative.sum())
            positive_weight = float(weights[positive].sum())
            negative_weight = float(weights[negative].sum())
            report[f"{population_name}_positive_rows"] = positive_count
            report[f"{population_name}_negative_rows"] = negative_count
            report[f"{population_name}_positive_weight"] = positive_weight
            report[f"{population_name}_negative_weight"] = negative_weight
            if (
                positive_count < minimum
                or negative_count < minimum
                or positive_weight <= 0
                or negative_weight <= 0
            ):
                raise ValueError(
                    f"fold {fold} {population_name} lacks the registered signed CV population"
                )
        reports.append(report)
    return folds, reports


def responsibility_balanced_log_loss(
    labels: torch.Tensor,
    probability: torch.Tensor,
    reference_weight: torch.Tensor,
    *,
    probability_epsilon: float = 1e-7,
) -> float:
    """Give positive and negative responsibility mass equal total influence."""

    target = torch.as_tensor(labels).detach().bool().cpu().reshape(-1)
    score = torch.as_tensor(probability).detach().double().cpu().reshape(-1)
    weight = torch.as_tensor(reference_weight).detach().double().cpu().reshape(-1)
    if target.shape != score.shape or target.shape != weight.shape:
        raise ValueError("balanced log-loss inputs must align")
    if not bool(torch.isfinite(score).all()) or not bool(torch.isfinite(weight).all()):
        raise ValueError("balanced log-loss inputs must be finite")
    if bool((weight <= 0).any()) or not bool(target.any()) or not bool((~target).any()):
        raise ValueError("balanced log-loss requires positive weight and both classes")
    epsilon = float(probability_epsilon)
    if not 0 < epsilon < 0.5:
        raise ValueError("probability_epsilon must be in (0,0.5)")
    score = score.clamp(epsilon, 1.0 - epsilon)
    positive_weight = weight[target]
    negative_weight = weight[~target]
    positive = -(positive_weight * score[target].log()).sum() / positive_weight.sum()
    negative = -(negative_weight * (1.0 - score[~target]).log()).sum() / negative_weight.sum()
    return float(0.5 * (positive + negative))


def responsibility_weighted_auc(
    labels: torch.Tensor,
    probability: torch.Tensor,
    reference_weight: torch.Tensor,
) -> float:
    from sklearn.metrics import roc_auc_score

    target = torch.as_tensor(labels).detach().bool().cpu().reshape(-1)
    score = torch.as_tensor(probability).detach().double().cpu().reshape(-1)
    weight = torch.as_tensor(reference_weight).detach().double().cpu().reshape(-1)
    if target.shape != score.shape or target.shape != weight.shape:
        raise ValueError("weighted AUC inputs must align")
    if (
        not bool(torch.isfinite(score).all())
        or not bool(torch.isfinite(weight).all())
        or bool((weight <= 0).any())
        or not bool(target.any())
        or not bool((~target).any())
    ):
        raise ValueError("weighted AUC requires finite positive weights and both classes")
    return float(
        roc_auc_score(
            target.numpy().astype(np.int64, copy=False),
            score.numpy(),
            sample_weight=weight.numpy(),
        )
    )


@dataclass(frozen=True)
class SignedScribbleCVResult:
    selected_action: str
    metrics: dict[str, dict[str, float]]
    folds: torch.Tensor
    fold_reports: list[dict[str, object]]
    oof_predictions: dict[str, torch.Tensor]
    observed: torch.Tensor


@torch.inference_mode()
def run_signed_scribble_cross_validation(
    global_rows: torch.Tensor,
    signed_reference_evidence: torch.Tensor,
    reference_weight: torch.Tensor,
    predictor: Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]],
    *,
    num_folds: int = 3,
    minimum_class_rows: int = 32,
    metric_round_decimals: int = 12,
    probability_epsilon: float = 1e-7,
) -> SignedScribbleCVResult:
    """Generate strict OOF predictions and select one registered action."""

    rows = torch.as_tensor(global_rows).detach().long().cpu().reshape(-1)
    evidence = torch.as_tensor(signed_reference_evidence).detach().float().cpu().reshape(-1)
    weights = torch.as_tensor(reference_weight).detach().double().cpu().reshape(-1)
    folds, reports = audit_signed_cv_population(
        rows,
        evidence,
        weights,
        num_folds=int(num_folds),
        minimum_class_rows=int(minimum_class_rows),
    )
    observed = evidence != 0
    oof = {
        ACTION_STRONG_UNARY: torch.full(evidence.shape, float("nan"), dtype=torch.float32),
        ACTION_HASH256_DIFFUSION: torch.full(
            evidence.shape, float("nan"), dtype=torch.float32
        ),
    }
    for fold in range(int(num_folds)):
        training = training_evidence_for_fold(evidence, folds, heldout_fold=fold)
        if bool((training[folds == fold] != 0).any()):
            raise RuntimeError("held-out evidence leaked into the fold predictor")
        unary, diffusion = predictor(training, fold)
        unary = torch.as_tensor(unary).detach().float().cpu().reshape(-1)
        diffusion = torch.as_tensor(diffusion).detach().float().cpu().reshape(-1)
        if unary.shape != evidence.shape or diffusion.shape != evidence.shape:
            raise ValueError("CV predictor outputs must align with primitive rows")
        if (
            not bool(torch.isfinite(unary).all())
            or not bool(torch.isfinite(diffusion).all())
            or bool(((unary < 0) | (unary > 1)).any())
            or bool(((diffusion < 0) | (diffusion > 1)).any())
        ):
            raise ValueError("CV predictor returned an invalid probability")
        heldout = observed & (folds == fold)
        oof[ACTION_STRONG_UNARY][heldout] = unary[heldout]
        oof[ACTION_HASH256_DIFFUSION][heldout] = diffusion[heldout]
    for action in REGISTERED_ACTIONS:
        if not bool(torch.isfinite(oof[action][observed]).all()):
            raise RuntimeError("CV did not produce every held-out prediction")
    labels = evidence[observed] > 0
    observed_weights = weights[observed]
    metrics: dict[str, dict[str, float]] = {}
    for action in REGISTERED_ACTIONS:
        probability = oof[action][observed]
        metrics[action] = {
            "responsibility_balanced_log_loss": responsibility_balanced_log_loss(
                labels,
                probability,
                observed_weights,
                probability_epsilon=float(probability_epsilon),
            ),
            "responsibility_weighted_auc": responsibility_weighted_auc(
                labels, probability, observed_weights
            ),
        }
    selected = select_registered_action(
        metrics, metric_round_decimals=int(metric_round_decimals)
    )
    return SignedScribbleCVResult(
        selected_action=selected,
        metrics=metrics,
        folds=folds,
        fold_reports=reports,
        oof_predictions=oof,
        observed=observed,
    )
