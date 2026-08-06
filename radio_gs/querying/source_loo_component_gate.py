"""Target-blind source-LOO gate for unknown graph components.

The default prediction is the graph-off field prior.  A confidence-zero
component may use a previously materialized harmonic extension only when four
deterministic source-boundary holdouts each improve a proper Brier score over
the graph-off field prior.  Components without enough evidence fail closed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from .observation_clamped_harmonic import (
    ObservationClampedHarmonicConfig,
    solve_observation_clamped_harmonic,
)
from .support_solver import PrimitiveSupportGraph


FOLD_COUNT = 4


def method_contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "method": "source_loo_unknown_component_gate_v1",
        "default": "graph_off_field_prior",
        "candidate": "observation_clamped_harmonic_extension_v1",
        "unknown_component": (
            "positive_affinity_connected_component_induced_by_"
            "source_confidence_equal_zero"
        ),
        "validation_ring": "source_observed_rows_adjacent_to_unknown_component",
        "fold_count": FOLD_COUNT,
        "fold_assignment": (
            "blake2s(scene_id_colon_global_primitive_row)_uint32_mod_4"
        ),
        "heldout_eligibility": (
            "strictly_between_zero_and_one_source_confidence_and_"
            "algebraically_recoverable_in_unit_interval_field_prior"
        ),
        "validation_target": "exact_adjoint_source_observation_probability",
        "validation_weight": (
            "source_observation_confidence_times_boundary_raw_affinity"
        ),
        "proper_score": "weighted_binary_brier",
        "acceptance": (
            "positive_validation_weight_in_every_fold_and_strictly_lower_"
            "candidate_brier_than_field_brier_in_every_fold_and_overall"
        ),
        "no_boundary_or_insufficient_validation": "fail_closed_to_graph_off",
        "observed_rows": "preserve_fused_unary_bitwise",
        "no_numeric_gain_margin": True,
        "scene_specific_parameters": False,
        "uses_target_rgb_mask_or_metric": False,
    }


def deterministic_folds(
    global_rows: torch.Tensor,
    *,
    scene_id: str,
    fold_count: int = FOLD_COUNT,
) -> torch.Tensor:
    rows = torch.as_tensor(global_rows).long().cpu().reshape(-1)
    if int(fold_count) != FOLD_COUNT:
        raise ValueError(f"source-LOO fold count is frozen to {FOLD_COUNT}")
    values = []
    for row in rows.tolist():
        digest = hashlib.blake2s(
            f"{scene_id}:{int(row)}".encode("utf-8"), digest_size=4
        ).digest()
        values.append(int.from_bytes(digest, "little") % FOLD_COUNT)
    return torch.tensor(values, dtype=torch.int64)


def recover_graph_off_field_prior(
    fused_probability: torch.Tensor,
    source_probability: torch.Tensor,
    source_confidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert the registered probability mixture where it is identifiable."""

    fused = torch.as_tensor(fused_probability).double().reshape(-1)
    probability = torch.as_tensor(source_probability).double().reshape(-1)
    confidence = torch.as_tensor(source_confidence).double().reshape(-1)
    if fused.shape != probability.shape or fused.shape != confidence.shape:
        raise ValueError("field-prior recovery vectors must align")
    if (
        not bool(torch.isfinite(fused).all())
        or not bool(torch.isfinite(probability).all())
        or not bool(torch.isfinite(confidence).all())
        or bool(((fused < 0) | (fused > 1)).any())
        or bool(((probability < 0) | (probability > 1)).any())
        or bool(((confidence < 0) | (confidence > 1)).any())
    ):
        raise ValueError("field-prior recovery inputs must be finite probabilities")
    field = fused.clone()
    identifiable = (confidence > 0) & (confidence < 1)
    field[identifiable] = (
        fused[identifiable]
        - confidence[identifiable] * probability[identifiable]
    ) / (1.0 - confidence[identifiable])
    algebraically_valid = identifiable & torch.isfinite(field) & (field >= 0) & (
        field <= 1
    )
    # Invalid inversions are never validated or held out.  Their fallback value
    # remains the sealed fused unary, and confidence-one rows remain boundaries.
    field[identifiable & ~algebraically_valid] = fused[
        identifiable & ~algebraically_valid
    ]
    return field.float().contiguous(), algebraically_valid.contiguous()


def unknown_component_labels(
    graph: PrimitiveSupportGraph,
    observed: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Label only the positive-affinity subgraph induced by unknown rows."""

    observed_cpu = torch.as_tensor(observed).bool().cpu().reshape(-1)
    if observed_cpu.shape != (graph.num_nodes,):
        raise ValueError("observed mask must align with graph")
    unknown_rows = torch.nonzero(~observed_cpu, as_tuple=False).flatten().numpy()
    labels = torch.full((graph.num_nodes,), -1, dtype=torch.int64)
    if unknown_rows.size == 0:
        return labels, 0
    local = np.full(graph.num_nodes, -1, dtype=np.int64)
    local[unknown_rows] = np.arange(unknown_rows.size, dtype=np.int64)
    edges = graph.edge_index.detach().cpu().numpy()
    affinity = graph.raw_affinity.detach().float().cpu().numpy()
    keep = (
        (affinity > 0)
        & (~observed_cpu.numpy()[edges[0]])
        & (~observed_cpu.numpy()[edges[1]])
    )
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    adjacency = coo_matrix(
        (
            np.ones(int(keep.sum()), dtype=np.uint8),
            (local[edges[0, keep]], local[edges[1, keep]]),
        ),
        shape=(unknown_rows.size, unknown_rows.size),
    ).tocsr()
    count, local_labels = connected_components(
        adjacency, directed=False, return_labels=True
    )
    labels[torch.from_numpy(unknown_rows)] = torch.from_numpy(local_labels).long()
    return labels, int(count)


@dataclass(frozen=True)
class BoundaryRingPairs:
    component: torch.Tensor
    observed_row: torch.Tensor
    affinity: torch.Tensor


def boundary_ring_pairs(
    graph: PrimitiveSupportGraph,
    observed: torch.Tensor,
    component_labels: torch.Tensor,
) -> BoundaryRingPairs:
    """Aggregate component/observed-row cross-edge affinity."""

    observed_np = torch.as_tensor(observed).bool().cpu().numpy()
    labels_np = torch.as_tensor(component_labels).long().cpu().numpy()
    edges = graph.edge_index.detach().cpu().numpy()
    affinity = graph.raw_affinity.detach().double().cpu().numpy()
    cross = (affinity > 0) & (observed_np[edges[0]] ^ observed_np[edges[1]])
    if not bool(cross.any()):
        empty_long = torch.empty(0, dtype=torch.int64)
        return BoundaryRingPairs(
            component=empty_long,
            observed_row=empty_long.clone(),
            affinity=torch.empty(0, dtype=torch.float64),
        )
    row = edges[0, cross]
    col = edges[1, cross]
    boundary = np.where(observed_np[row], row, col)
    unknown = np.where(observed_np[row], col, row)
    component = labels_np[unknown]
    if bool((component < 0).any()):
        raise RuntimeError("cross edge does not terminate in an unknown component")
    order = np.lexsort((boundary, component))
    component = component[order]
    boundary = boundary[order]
    weight = affinity[cross][order]
    new_pair = np.ones(component.size, dtype=bool)
    if component.size > 1:
        new_pair[1:] = (component[1:] != component[:-1]) | (
            boundary[1:] != boundary[:-1]
        )
    starts = np.flatnonzero(new_pair)
    return BoundaryRingPairs(
        component=torch.from_numpy(component[starts]).long(),
        observed_row=torch.from_numpy(boundary[starts]).long(),
        affinity=torch.from_numpy(np.add.reduceat(weight, starts)).double(),
    )


def source_loo_predictions(
    graph: PrimitiveSupportGraph,
    *,
    fused_probability: torch.Tensor,
    field_probability: torch.Tensor,
    source_confidence: torch.Tensor,
    validation_eligible: torch.Tensor,
    fold_assignment: torch.Tensor,
    config: ObservationClampedHarmonicConfig,
) -> list[torch.Tensor]:
    """Predict deterministic held-out source boundaries in four folds."""

    fused = torch.as_tensor(fused_probability).float().reshape(-1)
    field = torch.as_tensor(field_probability).float().reshape(-1)
    confidence = torch.as_tensor(source_confidence).float().reshape(-1)
    eligible = torch.as_tensor(validation_eligible).bool().reshape(-1)
    folds = torch.as_tensor(fold_assignment).long().reshape(-1)
    if any(
        value.shape != (graph.num_nodes,)
        for value in (fused, field, confidence, eligible, folds)
    ):
        raise ValueError("source-LOO vectors must align with graph")
    observed = confidence > 0
    results = []
    for fold in range(FOLD_COUNT):
        heldout = observed & eligible & (folds == fold)
        train_confidence = confidence.clone()
        train_confidence[heldout] = 0.0
        fold_prior = field.clone()
        fold_prior[train_confidence > 0] = fused[train_confidence > 0]
        result = solve_observation_clamped_harmonic(
            graph,
            fold_prior,
            train_confidence,
            config=config,
        )
        results.append(result.float().contiguous())
    return results


def component_brier_records(
    *,
    component_count: int,
    rings: BoundaryRingPairs,
    predictions: list[torch.Tensor],
    field_probability: torch.Tensor,
    source_probability: torch.Tensor,
    source_confidence: torch.Tensor,
    validation_eligible: torch.Tensor,
    fold_assignment: torch.Tensor,
) -> tuple[list[dict[str, object]], torch.Tensor]:
    """Attribute source-LOO Brier gains to unknown-component boundary rings."""

    if len(predictions) != FOLD_COUNT:
        raise ValueError("source-LOO requires four prediction folds")
    field = torch.as_tensor(field_probability).double()
    target = torch.as_tensor(source_probability).double()
    confidence = torch.as_tensor(source_confidence).double()
    eligible = torch.as_tensor(validation_eligible).bool()
    folds = torch.as_tensor(fold_assignment).long()
    accepted = torch.zeros(component_count, dtype=torch.bool)
    records: list[dict[str, object]] = []
    for component in range(component_count):
        pair_mask = rings.component == component
        boundary = rings.observed_row[pair_mask]
        ring_affinity = rings.affinity[pair_mask]
        fold_records = []
        total_weight = 0.0
        total_candidate_error = 0.0
        total_field_error = 0.0
        every_fold_improves = True
        for fold in range(FOLD_COUNT):
            take = eligible[boundary] & (folds[boundary] == fold)
            rows = boundary[take]
            weight = ring_affinity[take] * confidence[rows]
            weight_sum = float(weight.sum())
            if weight_sum > 0:
                candidate_error = float(
                    (weight * (predictions[fold][rows].double() - target[rows]).square()).sum()
                )
                field_error = float(
                    (weight * (field[rows] - target[rows]).square()).sum()
                )
                candidate_brier = candidate_error / weight_sum
                field_brier = field_error / weight_sum
                improves = candidate_brier < field_brier
            else:
                candidate_error = field_error = 0.0
                candidate_brier = field_brier = None
                improves = False
            every_fold_improves = every_fold_improves and improves
            total_weight += weight_sum
            total_candidate_error += candidate_error
            total_field_error += field_error
            fold_records.append(
                {
                    "fold": fold,
                    "validation_rows": int(rows.numel()),
                    "validation_weight": weight_sum,
                    "candidate_brier": candidate_brier,
                    "field_brier": field_brier,
                    "strictly_improves": improves,
                }
            )
        overall_candidate = (
            total_candidate_error / total_weight if total_weight > 0 else None
        )
        overall_field = total_field_error / total_weight if total_weight > 0 else None
        overall_improves = bool(
            total_weight > 0
            and overall_candidate is not None
            and overall_field is not None
            and overall_candidate < overall_field
        )
        accept = bool(every_fold_improves and overall_improves)
        accepted[component] = accept
        records.append(
            {
                "component": component,
                "boundary_ring_rows": int(boundary.numel()),
                "validation_weight": total_weight,
                "candidate_brier": overall_candidate,
                "field_brier": overall_field,
                "strictly_improves_every_fold": every_fold_improves,
                "strictly_improves_overall": overall_improves,
                "accepted": accept,
                "folds": fold_records,
            }
        )
    return records, accepted
