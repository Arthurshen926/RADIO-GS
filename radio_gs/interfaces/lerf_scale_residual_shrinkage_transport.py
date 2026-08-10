"""Source-validatable residual shrinkage after common scale transport.

This v2 interface separates a multiscale frame into its common mean and
centered scale residuals.  The v1 shortest-plane orthogonal map transports
both components coherently.  A monotone, query-free coefficient then shrinks
only the transported residual before the cosine gauge is restored::

    mu = mean_s(x_s)
    delta_s = x_s - mu
    z_s = R_phi(mu) + gamma * R_phi(delta_s)
    y_s = normalize(z_s)

``gamma`` may depend only on source-view reliability and the O0 scale
dispersion.  No scene or query identifier is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.interfaces.lerf_reliability_geodesic_budget import (
    RETAINED_VIEW_CAPACITY,
)
from radio_gs.interfaces.lerf_scale_equivariant_geodesic_transport import (
    CEILING_GRID_RADIANS,
    SCALE_COUNT,
    TRANSPORT_CONTRACT_SHA256,
    scale_equivariant_geodesic_transport,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


GAMMA_POLICY_EXPONENTS: dict[str, float | None] = {
    "rigid": None,
    "k0": 0.0,
    "k0p5": 0.5,
    "k1": 1.0,
    "k2": 2.0,
}
SOURCE_LOO_SCHEMA = "radio_gs.source_only_loo_scale_residual_shrinkage.v1"
SOURCE_GATE_SCHEMA = "radio_gs.source_only_gate_scale_residual_shrinkage.v1"


def candidate_grid() -> list[dict[str, Any]]:
    return [
        {
            "maximum_angle_radians": ceiling,
            "gamma_policy": policy,
            "dispersion_exponent": exponent,
        }
        for ceiling in CEILING_GRID_RADIANS
        for policy, exponent in GAMMA_POLICY_EXPONENTS.items()
    ]


def residual_shrinkage_contract() -> dict[str, Any]:
    return {
        "schema": "radio_gs.lerf_scale_residual_shrinkage_transport.v1",
        "schema_version": 1,
        "upstream_common_transport_contract_sha256": TRANSPORT_CONTRACT_SHA256,
        "decomposition": {
            "mean": "mu=mean_s(x_s_as_stored)",
            "centered_residual": "delta_s=x_s-mu",
            "transport": "same_R_phi_applied_to_mu_and_every_delta_s",
            "reconstruction": "z_s=R_phi(mu)+gamma*R_phi(delta_s)",
            "cosine_gauge": "y_s=l2_normalize(z_s)",
        },
        "source_reliability": (
            "rho=sqrt(clamp(((retained_view_count-1)/3),0,1)"
            "*teacher_view_directional_resultant)"
        ),
        "scale_dispersion": ("d=sqrt(mean_s(||x_s-mu||^2)/mean_s(||x_s||^2))"),
        "gamma_family": {
            "rigid": "gamma=1",
            "monotone": "gamma_k=1-rho*(1-d)^k",
            "dispersion_exponents": [0.0, 0.5, 1.0, 2.0],
            "zero_exponent_definition": "(1-d)^0=1_including_d_equal_one",
            "range": [0.0, 1.0],
            "monotone_nonincreasing_in_reliability": True,
            "monotone_nondecreasing_in_dispersion": True,
        },
        "joint_candidate_grid": candidate_grid(),
        "baseline_candidate": {
            "maximum_angle_radians": 0.15,
            "gamma_policy": "rigid",
        },
        "single_view_policy": "rho_zero_gamma_one_common_0p15_transport",
        "fallback": {
            "teacher_invalid": "bitwise_o0_score_cache",
            "undefined_scale_mean": "bitwise_o0_score_cache",
            "same_direction": "bitwise_o0_score_cache",
            "antipodal": "bitwise_o0_score_cache",
            "degenerate_reconstruction": "bitwise_o0_score_cache",
        },
        "source_only_gate": {
            "statistics": ["mean_cosine", "exact_p05_cosine"],
            "minimum_source_scenes": 2,
            "pooled_mean_strict_improvement": True,
            "every_source_scene_mean_nonregression": True,
            "every_source_scene_p05_nonregression": True,
            "eligibility": (
                "pooled_mean_delta_gt_zero_and_every_scene_mean_delta_ge_zero"
                "_and_every_scene_p05_delta_ge_zero"
            ),
            "selection_primary": "maximum_pooled_mean_cosine",
            "selection_exact_tie_break": "earliest_joint_candidate_grid_index",
        },
        "source_loo_compute": {
            "device": "cpu",
            "common_rotation_evaluations_per_fold": len(CEILING_GRID_RADIANS),
            "common_rotation_reused_across_gamma_policies": True,
        },
        "scene_or_query_id_consumed": False,
        "per_scene_parameters": False,
        "benchmark_labels_or_metrics_consumed": False,
        "formal_target_candidate_authorized": False,
    }


RESIDUAL_SHRINKAGE_CONTRACT_SHA256 = canonical_json_sha256(
    residual_shrinkage_contract()
)


@dataclass(frozen=True)
class ResidualShrinkageTransportOutput:
    descriptor: torch.Tensor
    teacher_applied: torch.Tensor
    fallback_to_o0: torch.Tensor
    gamma: torch.Tensor
    source_reliability: torch.Tensor
    scale_dispersion: torch.Tensor
    angular_budget_radians: torch.Tensor
    angular_step_radians: torch.Tensor
    expanded_budget: torch.Tensor
    reconstruction_valid: torch.Tensor


def residual_shrinkage_gamma(
    source_reliability: torch.Tensor,
    scale_dispersion: torch.Tensor,
    *,
    gamma_policy: str,
) -> torch.Tensor:
    """Return one member of the preregisterable monotone gamma family."""

    rho = torch.as_tensor(source_reliability)
    dispersion = torch.as_tensor(scale_dispersion, device=rho.device)
    if (
        gamma_policy not in GAMMA_POLICY_EXPONENTS
        or rho.shape != dispersion.shape
        or not rho.is_floating_point()
        or not dispersion.is_floating_point()
        or not bool(torch.isfinite(rho).all())
        or not bool(torch.isfinite(dispersion).all())
        or bool((rho < 0).any())
        or bool((rho > 1).any())
        or bool((dispersion < 0).any())
        or bool((dispersion > 1).any())
    ):
        raise ValueError("residual shrinkage gamma inputs differ")
    exponent = GAMMA_POLICY_EXPONENTS[gamma_policy]
    if exponent is None:
        return torch.ones_like(rho)
    if exponent == 0.0:
        # State the k=0 endpoint explicitly instead of relying on 0**0.
        return (1.0 - rho).clamp(0.0, 1.0)
    return (1.0 - rho * torch.pow((1.0 - dispersion).clamp(0.0, 1.0), exponent)).clamp(
        0.0, 1.0
    )


def _source_reliability_and_dispersion(
    base: torch.Tensor,
    count: torch.Tensor,
    agreement: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_float = base.float()
    mu = base_float.mean(dim=-2)
    centered = base_float - mu[..., None, :]
    total_energy = base_float.square().sum(dim=-1).mean(dim=-1)
    residual_energy = centered.square().sum(dim=-1).mean(dim=-1)
    dispersion = torch.sqrt(
        (
            residual_energy / total_energy.clamp_min(torch.finfo(torch.float32).eps)
        ).clamp(0.0, 1.0)
    )
    count_sufficiency = ((count.float() - 1.0) / (RETAINED_VIEW_CAPACITY - 1)).clamp(
        0.0, 1.0
    )
    rho = torch.sqrt((count_sufficiency * agreement.float()).clamp(0.0, 1.0))
    return torch.where(valid, rho, torch.zeros_like(rho)), dispersion


def _reconstruct_shrunk_descriptor(
    base: torch.Tensor,
    transported: torch.Tensor,
    common_applied: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_float = base.float()
    transported_mu = transported.mean(dim=-2)
    transported_residual = transported - transported_mu[..., None, :]
    reconstructed = (
        transported_mu[..., None, :] + gamma[..., None, None] * transported_residual
    )
    reconstructed_norm = torch.linalg.vector_norm(reconstructed, dim=-1)
    eps = 8.0 * torch.finfo(torch.float32).eps
    reconstruction_valid = (reconstructed_norm > eps).all(dim=-1)
    row_applied = common_applied.any(dim=-1) & reconstruction_valid
    normalized = F.normalize(reconstructed, dim=-1)
    descriptor = torch.where(row_applied[..., None, None], normalized, base_float)
    applied = row_applied[..., None].expand(*row_applied.shape, SCALE_COUNT)
    return descriptor.contiguous().float(), applied.contiguous(), reconstruction_valid


def scale_residual_shrinkage_transport(
    o0_descriptor_by_scale: torch.Tensor,
    teacher_mean: torch.Tensor,
    *,
    teacher_valid: torch.Tensor,
    retained_view_count: torch.Tensor,
    teacher_view_directional_resultant: torch.Tensor,
    maximum_angle_radians: float,
    gamma_policy: str,
) -> ResidualShrinkageTransportOutput:
    """Transport the common component and shrink only centered residuals."""

    base = torch.as_tensor(o0_descriptor_by_scale)
    count = torch.as_tensor(retained_view_count, device=base.device)
    agreement = torch.as_tensor(teacher_view_directional_resultant, device=base.device)
    common = scale_equivariant_geodesic_transport(
        base,
        teacher_mean,
        teacher_valid=teacher_valid,
        retained_view_count=count,
        teacher_view_directional_resultant=agreement,
        maximum_angle_radians=maximum_angle_radians,
    )
    valid = torch.as_tensor(teacher_valid, device=base.device)
    rho, dispersion = _source_reliability_and_dispersion(base, count, agreement, valid)
    gamma = residual_shrinkage_gamma(rho, dispersion, gamma_policy=gamma_policy)
    descriptor, applied, reconstruction_valid = _reconstruct_shrunk_descriptor(
        base, common.descriptor, common.teacher_applied, gamma
    )
    return ResidualShrinkageTransportOutput(
        descriptor=descriptor,
        teacher_applied=applied.contiguous(),
        fallback_to_o0=(~applied).contiguous(),
        gamma=gamma.contiguous(),
        source_reliability=rho.contiguous(),
        scale_dispersion=dispersion.contiguous(),
        angular_budget_radians=common.angular_budget_radians,
        angular_step_radians=common.angular_step_radians,
        expanded_budget=common.expanded_budget,
        reconstruction_valid=reconstruction_valid.contiguous(),
    )


def source_only_leave_one_view_out_residual_shrinkage_audit(
    top_descriptors: torch.Tensor,
    top_frame_ids: torch.Tensor,
    o0_descriptor_by_scale: torch.Tensor,
    *,
    row_chunk: int = 2048,
) -> dict[str, Any]:
    """Compute mean and exact p05 cosine for the full joint source grid."""

    descriptors = torch.as_tensor(top_descriptors)
    frame_ids = torch.as_tensor(top_frame_ids)
    base = torch.as_tensor(o0_descriptor_by_scale)
    grid = candidate_grid()
    if (
        descriptors.ndim != 3
        or descriptors.shape[1] != RETAINED_VIEW_CAPACITY
        or frame_ids.shape != descriptors.shape[:2]
        or frame_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
        or base.shape != (descriptors.shape[0], SCALE_COUNT, descriptors.shape[2])
        or descriptors.device.type != "cpu"
        or frame_ids.device.type != "cpu"
        or base.device.type != "cpu"
        or not descriptors.is_floating_point()
        or not base.is_floating_point()
        or not bool(torch.isfinite(descriptors).all())
        or not bool(torch.isfinite(base).all())
        or not isinstance(row_chunk, int)
        or row_chunk < 1
    ):
        raise ValueError("residual shrinkage source-only LOO tensors differ")
    mask = frame_ids >= 0
    counts = mask.sum(dim=1)
    maximum_predictions = int(counts[counts >= 2].sum())
    maximum_observations = maximum_predictions * SCALE_COUNT
    if maximum_observations <= 0:
        raise ValueError("residual shrinkage source-only LOO has no prediction")
    values = torch.empty(len(grid), maximum_observations, dtype=torch.float32)
    cursor = 0
    heldout_predictions = 0
    rows_with_loo = 0
    rows_with_expansion = 0
    eps = 8.0 * torch.finfo(torch.float32).eps
    for start in range(0, descriptors.shape[0], row_chunk):
        stop = min(descriptors.shape[0], start + row_chunk)
        selected = descriptors[start:stop]
        selected_mask = mask[start:stop]
        if bool((selected[~selected_mask] != 0).any()):
            raise ValueError("unretained residual-shrinkage view must be exact zero")
        selected_norm = torch.linalg.vector_norm(selected.float(), dim=-1)
        if bool((selected_norm[selected_mask] <= eps).any()):
            raise ValueError("retained residual-shrinkage view must be nonzero")
        unit_views = F.normalize(selected.float(), dim=-1)
        base_chunk = base[start:stop]
        selected_counts = counts[start:stop]
        view_sum = (unit_views * selected_mask[..., None]).sum(dim=1)
        row_has_loo = torch.zeros(stop - start, dtype=torch.bool)
        for heldout_index in range(RETAINED_VIEW_CAPACITY):
            heldout_valid = selected_mask[:, heldout_index] & (selected_counts >= 2)
            if not bool(heldout_valid.any()):
                continue
            heldout = unit_views[:, heldout_index]
            loo_count = (selected_counts - 1).clamp_min(1)
            loo_sum = view_sum - heldout
            loo_norm = torch.linalg.vector_norm(loo_sum, dim=-1)
            prediction_valid = heldout_valid & (loo_norm > eps)
            if not bool(prediction_valid.any()):
                continue
            row_has_loo |= prediction_valid
            teacher = loo_sum / loo_norm[..., None].clamp_min(eps)
            teacher = torch.where(
                prediction_valid[..., None], teacher, torch.zeros_like(teacher)
            )
            effective_count = torch.where(
                prediction_valid, loo_count, torch.zeros_like(loo_count)
            )
            agreement = (loo_norm / loo_count.float()).clamp(0.0, 1.0)
            agreement = torch.where(
                prediction_valid, agreement, torch.zeros_like(agreement)
            )
            observation_count = int(prediction_valid.sum()) * SCALE_COUNT
            rho, dispersion = _source_reliability_and_dispersion(
                base_chunk,
                effective_count,
                agreement,
                prediction_valid,
            )
            for ceiling_index, ceiling in enumerate(CEILING_GRID_RADIANS):
                common = scale_equivariant_geodesic_transport(
                    base_chunk,
                    teacher,
                    teacher_valid=prediction_valid,
                    retained_view_count=effective_count,
                    teacher_view_directional_resultant=agreement,
                    maximum_angle_radians=ceiling,
                )
                for policy_index, policy in enumerate(GAMMA_POLICY_EXPONENTS):
                    index = ceiling_index * len(GAMMA_POLICY_EXPONENTS) + policy_index
                    gamma = residual_shrinkage_gamma(
                        rho, dispersion, gamma_policy=policy
                    )
                    descriptor, _, _ = _reconstruct_shrunk_descriptor(
                        base_chunk,
                        common.descriptor,
                        common.teacher_applied,
                        gamma,
                    )
                    predicted = (descriptor * heldout[:, None, :]).sum(dim=-1)
                    values[index, cursor : cursor + observation_count] = predicted[
                        prediction_valid
                    ].reshape(-1)
            cursor += observation_count
            heldout_predictions += int(prediction_valid.sum())
        rows_with_loo += int(row_has_loo.sum())
        rows_with_expansion += int((row_has_loo & (selected_counts >= 3)).sum())
    values = values[:, :cursor]
    if cursor != heldout_predictions * SCALE_COUNT or cursor <= 0:
        raise RuntimeError("residual shrinkage LOO observation count differs")
    means = values.double().mean(dim=1)
    p05 = torch.quantile(values, 0.05, dim=1, interpolation="linear").double()
    baseline_mean = float(means[0])
    baseline_p05 = float(p05[0])
    candidates = []
    for spec, mean_value, p05_value in zip(grid, means, p05):
        mean_cosine = float(mean_value)
        p05_cosine = float(p05_value)
        mean_delta = mean_cosine - baseline_mean
        p05_delta = p05_cosine - baseline_p05
        candidates.append(
            {
                **spec,
                "heldout_scale_observations": cursor,
                "mean_cosine": mean_cosine,
                "p05_cosine": p05_cosine,
                "mean_delta_vs_baseline": mean_delta,
                "p05_delta_vs_baseline": p05_delta,
                "mean_nonregression_vs_baseline": mean_delta >= 0.0,
                "p05_nonregression_vs_baseline": p05_delta >= 0.0,
            }
        )
    result = {
        "schema": SOURCE_LOO_SCHEMA,
        "schema_version": 1,
        "residual_shrinkage_contract_sha256": (RESIDUAL_SHRINKAGE_CONTRACT_SHA256),
        "query_independent": True,
        "target_images_labels_masks_metrics_opened": False,
        "candidate_role": "source_only_joint_grid_diagnostic_not_target_authorization",
        "target_candidate_authorized": False,
        "rows": int(descriptors.shape[0]),
        "rows_with_valid_loo_prediction": rows_with_loo,
        "rows_with_expansion_evidence": rows_with_expansion,
        "heldout_predictions": heldout_predictions,
        "heldout_scale_observations": cursor,
        "p05_definition": "exact_torch_quantile_linear_over_transient_observations",
        "candidate_grid": candidates,
        "baseline_candidate_index": 0,
        "cross_scene_gate": residual_shrinkage_contract()["source_only_gate"],
        "transient_storage": {
            "candidate_by_observation_cosine_matrix": True,
            "per_view_descriptors_durable": False,
            "target_data_durable": False,
        },
    }
    validate_source_only_residual_shrinkage_audit(result)
    return result


def validate_source_only_residual_shrinkage_audit(
    value: Mapping[str, Any],
) -> None:
    required = {
        "schema",
        "schema_version",
        "residual_shrinkage_contract_sha256",
        "query_independent",
        "target_images_labels_masks_metrics_opened",
        "candidate_role",
        "target_candidate_authorized",
        "rows",
        "rows_with_valid_loo_prediction",
        "rows_with_expansion_evidence",
        "heldout_predictions",
        "heldout_scale_observations",
        "p05_definition",
        "candidate_grid",
        "baseline_candidate_index",
        "cross_scene_gate",
        "transient_storage",
    }
    rows = value.get("candidate_grid")
    observations = value.get("heldout_scale_observations")
    total_rows = value.get("rows")
    valid_rows = value.get("rows_with_valid_loo_prediction")
    expansion_rows = value.get("rows_with_expansion_evidence")
    predictions = value.get("heldout_predictions")
    if (
        set(value) != required
        or value.get("schema") != SOURCE_LOO_SCHEMA
        or value.get("schema_version") != 1
        or value.get("residual_shrinkage_contract_sha256")
        != RESIDUAL_SHRINKAGE_CONTRACT_SHA256
        or value.get("query_independent") is not True
        or value.get("target_images_labels_masks_metrics_opened") is not False
        or value.get("candidate_role")
        != "source_only_joint_grid_diagnostic_not_target_authorization"
        or value.get("target_candidate_authorized") is not False
        or not isinstance(value.get("rows"), int)
        or not isinstance(value.get("rows_with_valid_loo_prediction"), int)
        or not isinstance(value.get("rows_with_expansion_evidence"), int)
        or not isinstance(value.get("heldout_predictions"), int)
        or not isinstance(observations, int)
        or total_rows < 1
        or not 0 < valid_rows <= total_rows
        or not 0 <= expansion_rows <= valid_rows
        or predictions < valid_rows
        or observations <= 0
        or observations != predictions * SCALE_COUNT
        or value.get("p05_definition")
        != "exact_torch_quantile_linear_over_transient_observations"
        or value.get("baseline_candidate_index") != 0
        or value.get("cross_scene_gate")
        != residual_shrinkage_contract()["source_only_gate"]
        or value.get("transient_storage")
        != {
            "candidate_by_observation_cosine_matrix": True,
            "per_view_descriptors_durable": False,
            "target_data_durable": False,
        }
        or not isinstance(rows, list)
        or len(rows) != len(candidate_grid())
    ):
        raise ValueError("residual shrinkage source-only LOO contract differs")
    keys = {
        "maximum_angle_radians",
        "gamma_policy",
        "dispersion_exponent",
        "heldout_scale_observations",
        "mean_cosine",
        "p05_cosine",
        "mean_delta_vs_baseline",
        "p05_delta_vs_baseline",
        "mean_nonregression_vs_baseline",
        "p05_nonregression_vs_baseline",
    }
    baseline_mean = rows[0].get("mean_cosine") if rows else None
    baseline_p05 = rows[0].get("p05_cosine") if rows else None
    if (
        not isinstance(baseline_mean, float)
        or not math.isfinite(baseline_mean)
        or not isinstance(baseline_p05, float)
        or not math.isfinite(baseline_p05)
    ):
        raise ValueError("residual shrinkage source baseline differs")
    for spec, row in zip(candidate_grid(), rows):
        if not isinstance(row, Mapping) or set(row) != keys:
            raise ValueError("residual shrinkage source candidate fields differ")
        mean = row.get("mean_cosine")
        p05 = row.get("p05_cosine")
        mean_delta = row.get("mean_delta_vs_baseline")
        p05_delta = row.get("p05_delta_vs_baseline")
        if (
            any(row.get(key) != expected for key, expected in spec.items())
            or row.get("heldout_scale_observations") != observations
            or not isinstance(mean, float)
            or not math.isfinite(mean)
            or not -1.000001 <= mean <= 1.000001
            or not isinstance(p05, float)
            or not math.isfinite(p05)
            or not -1.000001 <= p05 <= 1.000001
            or not isinstance(mean_delta, float)
            or mean_delta != mean - baseline_mean
            or not isinstance(p05_delta, float)
            or p05_delta != p05 - baseline_p05
            or row.get("mean_nonregression_vs_baseline") is not (mean_delta >= 0.0)
            or row.get("p05_nonregression_vs_baseline") is not (p05_delta >= 0.0)
        ):
            raise ValueError("residual shrinkage source candidate contract differs")


def select_source_only_residual_shrinkage_candidate(
    scene_audits: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen cross-scene gate to scalar per-scene LOO audits."""

    if not isinstance(scene_audits, Mapping) or len(scene_audits) < 2:
        raise ValueError("residual shrinkage gate needs at least two scenes")
    normalized: list[tuple[str, Mapping[str, Any]]] = []
    for scene_id, audit in scene_audits.items():
        if (
            not isinstance(scene_id, str)
            or not scene_id
            or scene_id.strip() != scene_id
            or not isinstance(audit, Mapping)
        ):
            raise ValueError("residual shrinkage source scene record differs")
        validate_source_only_residual_shrinkage_audit(audit)
        normalized.append((scene_id, audit))
    normalized.sort(key=lambda item: item[0])

    total_observations = sum(
        int(audit["heldout_scale_observations"]) for _, audit in normalized
    )
    baseline_pooled_mean = (
        sum(
            float(audit["candidate_grid"][0]["mean_cosine"])
            * int(audit["heldout_scale_observations"])
            for _, audit in normalized
        )
        / total_observations
    )
    candidate_rows = []
    for index, spec in enumerate(candidate_grid()):
        scene_mean_deltas = [
            float(audit["candidate_grid"][index]["mean_delta_vs_baseline"])
            for _, audit in normalized
        ]
        scene_p05_deltas = [
            float(audit["candidate_grid"][index]["p05_delta_vs_baseline"])
            for _, audit in normalized
        ]
        pooled_mean = (
            sum(
                float(audit["candidate_grid"][index]["mean_cosine"])
                * int(audit["heldout_scale_observations"])
                for _, audit in normalized
            )
            / total_observations
        )
        pooled_delta = pooled_mean - baseline_pooled_mean
        all_mean_nonregression = all(delta >= 0.0 for delta in scene_mean_deltas)
        all_p05_nonregression = all(delta >= 0.0 for delta in scene_p05_deltas)
        candidate_rows.append(
            {
                **spec,
                "pooled_heldout_scale_observations": total_observations,
                "pooled_mean_cosine": pooled_mean,
                "pooled_mean_delta_vs_baseline": pooled_delta,
                "worst_scene_mean_delta_vs_baseline": min(scene_mean_deltas),
                "worst_scene_p05_delta_vs_baseline": min(scene_p05_deltas),
                "every_scene_mean_nonregression": all_mean_nonregression,
                "every_scene_p05_nonregression": all_p05_nonregression,
                "eligible": (
                    pooled_delta > 0.0
                    and all_mean_nonregression
                    and all_p05_nonregression
                ),
            }
        )
    eligible = [
        (index, row) for index, row in enumerate(candidate_rows) if row["eligible"]
    ]
    selected_index = (
        max(eligible, key=lambda item: (item[1]["pooled_mean_cosine"], -item[0]))[0]
        if eligible
        else None
    )
    result = {
        "schema": SOURCE_GATE_SCHEMA,
        "schema_version": 1,
        "residual_shrinkage_contract_sha256": (RESIDUAL_SHRINKAGE_CONTRACT_SHA256),
        "source_scene_ids": [scene_id for scene_id, _ in normalized],
        "source_scene_count": len(normalized),
        "pooled_heldout_scale_observations": total_observations,
        "baseline_candidate_index": 0,
        "candidate_grid": candidate_rows,
        "selected_candidate_index": selected_index,
        "source_gate_passed": selected_index is not None,
        "target_candidate_authorized": False,
        "target_images_labels_masks_metrics_opened": False,
    }
    validate_source_only_residual_shrinkage_gate(result)
    return result


def validate_source_only_residual_shrinkage_gate(
    value: Mapping[str, Any],
) -> None:
    required = {
        "schema",
        "schema_version",
        "residual_shrinkage_contract_sha256",
        "source_scene_ids",
        "source_scene_count",
        "pooled_heldout_scale_observations",
        "baseline_candidate_index",
        "candidate_grid",
        "selected_candidate_index",
        "source_gate_passed",
        "target_candidate_authorized",
        "target_images_labels_masks_metrics_opened",
    }
    scene_ids = value.get("source_scene_ids")
    rows = value.get("candidate_grid")
    observations = value.get("pooled_heldout_scale_observations")
    if (
        set(value) != required
        or value.get("schema") != SOURCE_GATE_SCHEMA
        or value.get("schema_version") != 1
        or value.get("residual_shrinkage_contract_sha256")
        != RESIDUAL_SHRINKAGE_CONTRACT_SHA256
        or not isinstance(scene_ids, list)
        or len(scene_ids) < 2
        or scene_ids != sorted(scene_ids)
        or len(set(scene_ids)) != len(scene_ids)
        or any(
            not isinstance(scene_id, str)
            or not scene_id
            or scene_id.strip() != scene_id
            for scene_id in scene_ids
        )
        or value.get("source_scene_count") != len(scene_ids)
        or not isinstance(observations, int)
        or isinstance(observations, bool)
        or observations <= 0
        or value.get("baseline_candidate_index") != 0
        or not isinstance(rows, list)
        or len(rows) != len(candidate_grid())
        or value.get("target_candidate_authorized") is not False
        or value.get("target_images_labels_masks_metrics_opened") is not False
    ):
        raise ValueError("residual shrinkage source gate contract differs")
    keys = {
        "maximum_angle_radians",
        "gamma_policy",
        "dispersion_exponent",
        "pooled_heldout_scale_observations",
        "pooled_mean_cosine",
        "pooled_mean_delta_vs_baseline",
        "worst_scene_mean_delta_vs_baseline",
        "worst_scene_p05_delta_vs_baseline",
        "every_scene_mean_nonregression",
        "every_scene_p05_nonregression",
        "eligible",
    }
    baseline_mean = rows[0].get("pooled_mean_cosine") if rows else None
    for spec, row in zip(candidate_grid(), rows):
        if not isinstance(row, Mapping) or set(row) != keys:
            raise ValueError("residual shrinkage source gate candidate differs")
        pooled_mean = row.get("pooled_mean_cosine")
        pooled_delta = row.get("pooled_mean_delta_vs_baseline")
        worst_mean = row.get("worst_scene_mean_delta_vs_baseline")
        worst_p05 = row.get("worst_scene_p05_delta_vs_baseline")
        mean_nonregression = row.get("every_scene_mean_nonregression")
        p05_nonregression = row.get("every_scene_p05_nonregression")
        if (
            any(row.get(key) != expected for key, expected in spec.items())
            or row.get("pooled_heldout_scale_observations") != observations
            or not isinstance(pooled_mean, float)
            or not math.isfinite(pooled_mean)
            or not -1.000001 <= pooled_mean <= 1.000001
            or not isinstance(pooled_delta, float)
            or not math.isfinite(pooled_delta)
            or pooled_delta != pooled_mean - baseline_mean
            or not isinstance(worst_mean, float)
            or not math.isfinite(worst_mean)
            or not isinstance(worst_p05, float)
            or not math.isfinite(worst_p05)
            or mean_nonregression is not (worst_mean >= 0.0)
            or p05_nonregression is not (worst_p05 >= 0.0)
            or row.get("eligible")
            is not (pooled_delta > 0.0 and mean_nonregression and p05_nonregression)
        ):
            raise ValueError("residual shrinkage source gate candidate differs")
    eligible = [(index, row) for index, row in enumerate(rows) if row["eligible"]]
    selected = (
        max(eligible, key=lambda item: (item[1]["pooled_mean_cosine"], -item[0]))[0]
        if eligible
        else None
    )
    if value.get("selected_candidate_index") != selected or value.get(
        "source_gate_passed"
    ) is not (selected is not None):
        raise ValueError("residual shrinkage source gate selection differs")


__all__ = [
    "GAMMA_POLICY_EXPONENTS",
    "RESIDUAL_SHRINKAGE_CONTRACT_SHA256",
    "ResidualShrinkageTransportOutput",
    "SOURCE_GATE_SCHEMA",
    "SOURCE_LOO_SCHEMA",
    "candidate_grid",
    "residual_shrinkage_contract",
    "residual_shrinkage_gamma",
    "scale_residual_shrinkage_transport",
    "select_source_only_residual_shrinkage_candidate",
    "source_only_leave_one_view_out_residual_shrinkage_audit",
    "validate_source_only_residual_shrinkage_audit",
    "validate_source_only_residual_shrinkage_gate",
]
