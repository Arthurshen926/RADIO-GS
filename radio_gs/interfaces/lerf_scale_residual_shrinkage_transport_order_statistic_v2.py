"""Versioned exact-source LOO engine for residual-shrinkage transport.

The sealed v1 interface remains untouched.  This v2 engine changes only the
allocation/execution path of its source-only statistic:

* all five gamma policies share one transported mean/residual computation;
* exact linear p05 uses adjacent order statistics instead of a full sort.

Candidate descriptors, means, p05 values, and gate decisions are required to
be numerically equivalent to the sealed v1 implementation.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.interfaces import (
    lerf_scale_residual_shrinkage_transport as _sealed,
)
from radio_gs.interfaces.lerf_reliability_geodesic_budget import (
    RETAINED_VIEW_CAPACITY,
)
from radio_gs.interfaces.lerf_scale_equivariant_geodesic_transport import (
    CEILING_GRID_RADIANS,
    SCALE_COUNT,
    scale_equivariant_geodesic_transport,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


SOURCE_LOO_SCHEMA = "radio_gs.source_only_loo_scale_residual_shrinkage.v2"
SOURCE_GATE_SCHEMA = "radio_gs.source_only_gate_scale_residual_shrinkage.v2"


def source_loo_execution_contract() -> dict[str, Any]:
    return {
        "schema": "radio_gs.lerf_scale_residual_shrinkage_source_loo.v2",
        "schema_version": 2,
        "sealed_math_contract_sha256": (_sealed.RESIDUAL_SHRINKAGE_CONTRACT_SHA256),
        "candidate_grid": _sealed.candidate_grid(),
        "candidate_score_readout": {
            "formula": ("cos_gamma=(a+gamma*b_s)/sqrt(u+2*gamma*v_s+gamma^2*w_s)"),
            "a": "dot(heldout,R_mu)",
            "b_s": "dot(heldout,R_delta_s)",
            "u": "squared_norm(R_mu)",
            "v_s": "dot(R_mu,R_delta_s)",
            "w_s": "squared_norm(R_delta_s)",
            "sealed_descriptor_score_max_abs_error_required": 1e-6,
            "candidate_gate_equivalence_required": True,
        },
        "gamma_grid_execution": (
            "one_transported_mean_residual_reused_by_all_five_gamma_policies"
        ),
        "exact_linear_p05": {
            "rank": "float32_q_times_n_minus_one_matching_input_dtype",
            "lower": "kthvalue_floor_rank_plus_one",
            "upper": "kthvalue_ceil_rank_plus_one",
            "interpolation": "torch_lerp_with_input_dtype_fractional_rank",
            "approximate": False,
            "full_sort": False,
        },
        "source_scene_or_query_specific_parameters": False,
        "query_embeddings_or_text_consumed": False,
        "target_images_labels_masks_metrics_consumed": False,
        "target_candidate_authorized": False,
    }


SOURCE_LOO_EXECUTION_CONTRACT_SHA256 = canonical_json_sha256(
    source_loo_execution_contract()
)


def exact_linear_quantile_by_order_statistic(
    values: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    """Bitwise-match row-wise ``torch.quantile(..., linear)`` without sorting."""

    data = torch.as_tensor(values)
    if (
        data.ndim != 2
        or data.shape[1] < 1
        or data.device.type != "cpu"
        or data.dtype not in (torch.float32, torch.float64)
        or not bool(torch.isfinite(data).all())
        or isinstance(quantile, bool)
        or not isinstance(quantile, (int, float))
        or not math.isfinite(float(quantile))
        or not 0.0 <= float(quantile) <= 1.0
    ):
        raise ValueError("exact linear order-statistic quantile inputs differ")
    rank = torch.as_tensor(float(quantile), dtype=data.dtype) * (data.shape[1] - 1)
    lower_index = int(torch.floor(rank).item())
    upper_index = int(torch.ceil(rank).item())
    lower = torch.kthvalue(data, lower_index + 1, dim=1).values
    if lower_index == upper_index:
        return lower
    upper = torch.kthvalue(data, upper_index + 1, dim=1).values
    return torch.lerp(lower, upper, rank - lower_index)


def score_gamma_grid_from_common_transport(
    base: torch.Tensor,
    transported: torch.Tensor,
    common_applied: torch.Tensor,
    heldout: torch.Tensor,
    source_reliability: torch.Tensor,
    scale_dispersion: torch.Tensor,
) -> torch.Tensor:
    """Return five policy scores from shared analytic scalar terms."""

    gammas = torch.stack(
        [
            _sealed.residual_shrinkage_gamma(
                source_reliability,
                scale_dispersion,
                gamma_policy=policy,
            )
            for policy in _sealed.GAMMA_POLICY_EXPONENTS
        ],
        dim=0,
    )
    base_float = base.float()
    heldout_float = heldout.float()
    transported_mu = transported.mean(dim=-2)
    transported_residual = transported - transported_mu[..., None, :]
    a = (heldout_float * transported_mu).sum(dim=-1)
    b = (heldout_float[:, None, :] * transported_residual).sum(dim=-1)
    u = transported_mu.square().sum(dim=-1)
    v = (transported_mu[:, None, :] * transported_residual).sum(dim=-1)
    w = transported_residual.square().sum(dim=-1)
    numerator = a[None, :, None] + gammas[:, :, None] * b[None, :, :]
    squared_norm = (
        u[None, :, None]
        + 2.0 * gammas[:, :, None] * v[None, :, :]
        + gammas[:, :, None].square() * w[None, :, :]
    )
    reconstructed_norm = torch.sqrt(squared_norm.clamp_min(0.0))
    reconstruction_valid = (
        reconstructed_norm > 8.0 * torch.finfo(torch.float32).eps
    ).all(dim=-1)
    applied = common_applied.any(dim=-1)[None, :] & reconstruction_valid
    analytic = numerator / reconstructed_norm.clamp_min(1e-12)
    base_score = (base_float * heldout_float[:, None, :]).sum(dim=-1)
    return torch.where(applied[..., None], analytic, base_score[None, :, :])


def source_only_leave_one_view_out_residual_shrinkage_audit(
    top_descriptors: torch.Tensor,
    top_frame_ids: torch.Tensor,
    o0_descriptor_by_scale: torch.Tensor,
    *,
    row_chunk: int = 2048,
) -> dict[str, Any]:
    """Compute the sealed 25 candidates with exact order-statistic p05."""

    descriptors = torch.as_tensor(top_descriptors)
    frame_ids = torch.as_tensor(top_frame_ids)
    base = torch.as_tensor(o0_descriptor_by_scale)
    grid = _sealed.candidate_grid()
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
        or isinstance(row_chunk, bool)
        or row_chunk < 1
    ):
        raise ValueError("order-statistic source-only LOO tensors differ")
    mask = frame_ids >= 0
    counts = mask.sum(dim=1)
    maximum_predictions = int(counts[counts >= 2].sum())
    maximum_observations = maximum_predictions * SCALE_COUNT
    if maximum_observations <= 0:
        raise ValueError("order-statistic source-only LOO has no prediction")
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
            raise ValueError("unretained order-statistic view must be exact zero")
        selected_norm = torch.linalg.vector_norm(selected.float(), dim=-1)
        if bool((selected_norm[selected_mask] <= eps).any()):
            raise ValueError("retained order-statistic view must be nonzero")
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
            rho, dispersion = _sealed._source_reliability_and_dispersion(
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
                predicted_grid = score_gamma_grid_from_common_transport(
                    base_chunk,
                    common.descriptor,
                    common.teacher_applied,
                    heldout,
                    rho,
                    dispersion,
                )
                for policy_index in range(len(_sealed.GAMMA_POLICY_EXPONENTS)):
                    index = (
                        ceiling_index * len(_sealed.GAMMA_POLICY_EXPONENTS)
                        + policy_index
                    )
                    values[index, cursor : cursor + observation_count] = predicted_grid[
                        policy_index, prediction_valid
                    ].reshape(-1)
            cursor += observation_count
            heldout_predictions += int(prediction_valid.sum())
        rows_with_loo += int(row_has_loo.sum())
        rows_with_expansion += int((row_has_loo & (selected_counts >= 3)).sum())
    values = values[:, :cursor]
    if cursor != heldout_predictions * SCALE_COUNT or cursor <= 0:
        raise RuntimeError("order-statistic LOO observation count differs")
    means = values.double().mean(dim=1)
    p05 = exact_linear_quantile_by_order_statistic(values, 0.05).double()
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
        "schema_version": 2,
        "residual_shrinkage_contract_sha256": (
            _sealed.RESIDUAL_SHRINKAGE_CONTRACT_SHA256
        ),
        "source_loo_execution_contract_sha256": (SOURCE_LOO_EXECUTION_CONTRACT_SHA256),
        "query_independent": True,
        "target_images_labels_masks_metrics_opened": False,
        "candidate_role": "source_only_joint_grid_diagnostic_not_target_authorization",
        "target_candidate_authorized": False,
        "rows": int(descriptors.shape[0]),
        "rows_with_valid_loo_prediction": rows_with_loo,
        "rows_with_expansion_evidence": rows_with_expansion,
        "heldout_predictions": heldout_predictions,
        "heldout_scale_observations": cursor,
        "p05_definition": (
            "exact_linear_adjacent_order_statistics_over_transient_observations"
        ),
        "candidate_grid": candidates,
        "baseline_candidate_index": 0,
        "cross_scene_gate": _sealed.residual_shrinkage_contract()["source_only_gate"],
        "transient_storage": {
            "candidate_by_observation_cosine_matrix": True,
            "per_view_descriptors_durable": False,
            "target_data_durable": False,
        },
    }
    validate_source_only_residual_shrinkage_audit(result)
    return result


@torch.inference_mode()
def compare_analytic_and_sealed_on_source_chunk(
    top_descriptors: torch.Tensor,
    top_frame_ids: torch.Tensor,
    o0_descriptor_by_scale: torch.Tensor,
    *,
    row_chunk: int = 2048,
) -> dict[str, Any]:
    """Compare every transient scalar against the sealed descriptor path.

    This diagnostic intentionally returns summaries only.  Neither candidate
    descriptors nor the candidate-by-observation matrices leave the process.
    """

    descriptors = torch.as_tensor(top_descriptors)
    frame_ids = torch.as_tensor(top_frame_ids)
    base = torch.as_tensor(o0_descriptor_by_scale)
    analytic_audit = source_only_leave_one_view_out_residual_shrinkage_audit(
        descriptors, frame_ids, base, row_chunk=row_chunk
    )
    sealed_audit = _sealed.source_only_leave_one_view_out_residual_shrinkage_audit(
        descriptors, frame_ids, base, row_chunk=row_chunk
    )
    mask = frame_ids >= 0
    counts = mask.sum(dim=1)
    maximum_predictions = int(counts[counts >= 2].sum())
    maximum_observations = maximum_predictions * SCALE_COUNT
    analytic_values = torch.empty(
        len(_sealed.candidate_grid()), maximum_observations, dtype=torch.float32
    )
    sealed_values = torch.empty_like(analytic_values)
    cursor = 0
    eps = 8.0 * torch.finfo(torch.float32).eps
    for start in range(0, descriptors.shape[0], row_chunk):
        stop = min(descriptors.shape[0], start + row_chunk)
        selected = descriptors[start:stop]
        selected_mask = mask[start:stop]
        unit_views = F.normalize(selected.float(), dim=-1)
        base_chunk = base[start:stop]
        selected_counts = counts[start:stop]
        view_sum = (unit_views * selected_mask[..., None]).sum(dim=1)
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
            rho, dispersion = _sealed._source_reliability_and_dispersion(
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
                analytic_grid = score_gamma_grid_from_common_transport(
                    base_chunk,
                    common.descriptor,
                    common.teacher_applied,
                    heldout,
                    rho,
                    dispersion,
                )
                for policy_index, policy in enumerate(_sealed.GAMMA_POLICY_EXPONENTS):
                    index = (
                        ceiling_index * len(_sealed.GAMMA_POLICY_EXPONENTS)
                        + policy_index
                    )
                    gamma = _sealed.residual_shrinkage_gamma(
                        rho, dispersion, gamma_policy=policy
                    )
                    reconstructed, _, _ = _sealed._reconstruct_shrunk_descriptor(
                        base_chunk,
                        common.descriptor,
                        common.teacher_applied,
                        gamma,
                    )
                    sealed_score = (reconstructed * heldout[:, None, :]).sum(dim=-1)
                    destination = slice(cursor, cursor + observation_count)
                    sealed_values[index, destination] = sealed_score[
                        prediction_valid
                    ].reshape(-1)
                    analytic_values[index, destination] = analytic_grid[
                        policy_index, prediction_valid
                    ].reshape(-1)
            cursor += observation_count
    if cursor != int(analytic_audit["heldout_scale_observations"]):
        raise RuntimeError("analytic/sealed equivalence observation count differs")
    analytic_values = analytic_values[:, :cursor]
    sealed_values = sealed_values[:, :cursor]
    cell_error = (analytic_values - sealed_values).abs()
    analytic_means = analytic_values.double().mean(dim=1)
    sealed_means = sealed_values.double().mean(dim=1)
    analytic_p05 = exact_linear_quantile_by_order_statistic(
        analytic_values, 0.05
    ).double()
    sealed_p05 = torch.quantile(
        sealed_values, 0.05, dim=1, interpolation="linear"
    ).double()

    def eligible_indices(audit: Mapping[str, Any]) -> list[int]:
        return [
            index
            for index, row in enumerate(audit["candidate_grid"])
            if row["mean_delta_vs_baseline"] > 0.0
            and row["p05_nonregression_vs_baseline"] is True
        ]

    analytic_eligible = eligible_indices(analytic_audit)
    sealed_eligible = eligible_indices(sealed_audit)

    def select_single_scene(
        audit: Mapping[str, Any], eligible: list[int]
    ) -> int | None:
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda index: (
                float(audit["candidate_grid"][index]["mean_cosine"]),
                -index,
            ),
        )

    analytic_selected = select_single_scene(analytic_audit, analytic_eligible)
    sealed_selected = select_single_scene(sealed_audit, sealed_eligible)
    candidates = []
    for index, spec in enumerate(_sealed.candidate_grid()):
        candidates.append(
            {
                "candidate_index": index,
                **spec,
                "maximum_cell_abs_error": float(cell_error[index].max()),
                "mean_abs_error": float(
                    (analytic_means[index] - sealed_means[index]).abs()
                ),
                "p05_abs_error": float((analytic_p05[index] - sealed_p05[index]).abs()),
                "mean_gate_identical": (
                    analytic_audit["candidate_grid"][index][
                        "mean_nonregression_vs_baseline"
                    ]
                    is sealed_audit["candidate_grid"][index][
                        "mean_nonregression_vs_baseline"
                    ]
                ),
                "p05_gate_identical": (
                    analytic_audit["candidate_grid"][index][
                        "p05_nonregression_vs_baseline"
                    ]
                    is sealed_audit["candidate_grid"][index][
                        "p05_nonregression_vs_baseline"
                    ]
                ),
            }
        )
    maximum_cell_abs_error = float(cell_error.max())
    maximum_mean_abs_error = float((analytic_means - sealed_means).abs().max())
    maximum_p05_abs_error = float((analytic_p05 - sealed_p05).abs().max())
    candidate_gate_identical = (
        analytic_eligible == sealed_eligible
        and analytic_selected == sealed_selected
        and all(
            row["mean_gate_identical"] and row["p05_gate_identical"]
            for row in candidates
        )
    )
    tolerance = 1e-6
    result = {
        "schema": "radio_gs.lerf_transport_v2_analytic_equivalence.v1",
        "schema_version": 1,
        "rows": int(descriptors.shape[0]),
        "heldout_scale_observations": cursor,
        "per_cell_tolerance": tolerance,
        "maximum_cell_abs_error": maximum_cell_abs_error,
        "maximum_mean_abs_error": maximum_mean_abs_error,
        "maximum_p05_abs_error": maximum_p05_abs_error,
        "candidate_gate_identical": candidate_gate_identical,
        "selected_candidate_index_identical": analytic_selected == sealed_selected,
        "sealed_single_scene_eligible_candidate_indices": sealed_eligible,
        "analytic_single_scene_eligible_candidate_indices": analytic_eligible,
        "sealed_selected_candidate_index": sealed_selected,
        "analytic_selected_candidate_index": analytic_selected,
        "candidates": candidates,
        "equivalence_gate_passed": (
            maximum_cell_abs_error <= tolerance
            and maximum_mean_abs_error <= tolerance
            and maximum_p05_abs_error <= tolerance
            and candidate_gate_identical
        ),
        "transient_storage": {
            "candidate_descriptors_written": False,
            "candidate_by_observation_matrices_written": False,
            "source_view_descriptors_written": False,
        },
    }
    return result


def _sealed_audit(value: Mapping[str, Any]) -> dict[str, Any]:
    surrogate = dict(value)
    surrogate.pop("source_loo_execution_contract_sha256", None)
    surrogate["schema"] = _sealed.SOURCE_LOO_SCHEMA
    surrogate["schema_version"] = 1
    surrogate["p05_definition"] = (
        "exact_torch_quantile_linear_over_transient_observations"
    )
    return surrogate


def validate_source_only_residual_shrinkage_audit(
    value: Mapping[str, Any],
) -> None:
    if (
        value.get("schema") != SOURCE_LOO_SCHEMA
        or value.get("schema_version") != 2
        or value.get("source_loo_execution_contract_sha256")
        != SOURCE_LOO_EXECUTION_CONTRACT_SHA256
        or value.get("p05_definition")
        != "exact_linear_adjacent_order_statistics_over_transient_observations"
    ):
        raise ValueError("order-statistic source-only LOO contract differs")
    _sealed.validate_source_only_residual_shrinkage_audit(_sealed_audit(value))


def select_source_only_residual_shrinkage_candidate(
    scene_audits: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    for audit in scene_audits.values():
        validate_source_only_residual_shrinkage_audit(audit)
    result = _sealed.select_source_only_residual_shrinkage_candidate(
        {scene_id: _sealed_audit(audit) for scene_id, audit in scene_audits.items()}
    )
    result["schema"] = SOURCE_GATE_SCHEMA
    result["schema_version"] = 2
    result["source_loo_execution_contract_sha256"] = (
        SOURCE_LOO_EXECUTION_CONTRACT_SHA256
    )
    return result


__all__ = [
    "SOURCE_GATE_SCHEMA",
    "SOURCE_LOO_EXECUTION_CONTRACT_SHA256",
    "SOURCE_LOO_SCHEMA",
    "compare_analytic_and_sealed_on_source_chunk",
    "exact_linear_quantile_by_order_statistic",
    "score_gamma_grid_from_common_transport",
    "select_source_only_residual_shrinkage_candidate",
    "source_loo_execution_contract",
    "source_only_leave_one_view_out_residual_shrinkage_audit",
    "validate_source_only_residual_shrinkage_audit",
]
