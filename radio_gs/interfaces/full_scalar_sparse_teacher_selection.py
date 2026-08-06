"""Frozen query-free row/view selection for sparse full-scalar teachers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

import torch

from radio_gs.utils.immutable_artifacts import canonical_json_sha256


REGION_CAP_PER_SCENE = 4096
VIEW_CAP_PER_REGION = 4
SPARSE_V2_PREREG_FILE_SHA256 = (
    "9e053f64b567298b302d6c39ddf3eb42ac871624b2755f2ce2e16a1923ce9106"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sampling_contract() -> dict[str, Any]:
    return {
        "name": "exact_overlap_anchor_visible_scale_stratified_hash_cap_v1",
        "per_scene_region_cap": REGION_CAP_PER_SCENE,
        "candidate": (
            "accepted_anchor_in_exact_factorized_state_overlap_and_at_least_"
            "one_positive_exact_marginal_anchor_hit"
        ),
        "scale_quota": (
            "floor_cap_over_sorted_scales_plus_one_for_first_remainder_no_"
            "redistribution"
        ),
        "within_scale_order": "ascending_stable_region_fingerprint_sha256",
        "within_scale_tie_break": "ascending_accepted_canonical_region_index",
        "published_order": "ascending_accepted_canonical_region_index",
        "views_per_region_cap": VIEW_CAP_PER_REGION,
        "view_order": (
            "descending_visible_active_primitive_count_then_descending_positive_"
            "hit_count_then_ascending_responsibility_view_index"
        ),
        "per_scene_hyperparameters": False,
        "query_independent": True,
        "preregistration_file_sha256": SPARSE_V2_PREREG_FILE_SHA256,
    }


SAMPLING_CONTRACT_SHA256 = canonical_json_sha256(sampling_contract())


def region_identity(
    *,
    scene_id: str,
    scale_index: int,
    anchor_global_row: int,
    active_global_rows: Sequence[int],
) -> dict[str, Any]:
    return {
        "scene_id": str(scene_id),
        "region_contract": "accepted-v2-canonical-v1",
        "scale_index": int(scale_index),
        "anchor_global_row": int(anchor_global_row),
        "active_global_rows": [int(value) for value in active_global_rows],
    }


def region_fingerprint(**kwargs: Any) -> str:
    return canonical_json_sha256(region_identity(**kwargs))


def scale_quotas(scale_values: Sequence[int]) -> dict[int, int]:
    scales = sorted(set(int(value) for value in scale_values))
    if not scales:
        raise ValueError("sparse teacher selection requires at least one scale")
    base, remainder = divmod(REGION_CAP_PER_SCENE, len(scales))
    return {
        scale: base + (1 if rank < remainder else 0)
        for rank, scale in enumerate(scales)
    }


def validate_sparse_pair_cardinality(
    *, selected_region_count: int, pair_count: int
) -> None:
    selected = int(selected_region_count)
    pairs = int(pair_count)
    if (
        selected <= 0
        or selected > REGION_CAP_PER_SCENE
        or pairs < selected
        or pairs > selected * VIEW_CAP_PER_REGION
    ):
        raise ValueError("sparse teacher pair cardinality exceeds frozen caps")


def select_scale_stratified_indices(
    scale_indices: torch.Tensor,
    fingerprints: Sequence[str],
    candidate_mask: torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    """Select canonical row indices under the single global sampling rule."""

    scales = torch.as_tensor(scale_indices).long().cpu().reshape(-1)
    candidate = torch.as_tensor(candidate_mask).bool().cpu().reshape(-1)
    if (
        scales.shape != candidate.shape
        or len(fingerprints) != scales.numel()
        or any(_SHA256_RE.fullmatch(str(value)) is None for value in fingerprints)
    ):
        raise ValueError("sparse teacher selection inputs differ")
    values = sorted(set(int(value) for value in scales.tolist()))
    quotas = scale_quotas(values)
    selected: list[int] = []
    selected_by_scale: list[int] = []
    for scale in values:
        rows = torch.where(candidate & (scales == scale))[0].tolist()
        rows.sort(key=lambda index: (str(fingerprints[index]), int(index)))
        chosen = rows[: quotas[scale]]
        selected.extend(chosen)
        selected_by_scale.append(len(chosen))
    selected.sort()
    if not selected:
        raise ValueError("sparse teacher selection has no eligible rows")
    return torch.tensor(selected, dtype=torch.long), selected_by_scale


def validate_selection_audit(
    value: object,
    *,
    selected_count: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "sampling_contract_sha256",
        "canonical_candidate_region_count",
        "exact_overlap_candidate_count",
        "teacher_visible_candidate_count",
        "selected_region_count",
        "selected_count_by_scale",
    }:
        raise ValueError("sparse teacher selection audit differs")
    audit = dict(value)
    counts = audit.get("selected_count_by_scale")
    canonical_count = int(audit.get("canonical_candidate_region_count", -1))
    overlap_count = int(audit.get("exact_overlap_candidate_count", -1))
    visible_count = int(audit.get("teacher_visible_candidate_count", -1))
    audit_selected_count = int(audit.get("selected_region_count", -1))
    quotas = scale_quotas(range(len(counts))) if isinstance(counts, list) and counts else {}
    if (
        audit.get("sampling_contract_sha256") != SAMPLING_CONTRACT_SHA256
        or not (
            canonical_count >= overlap_count >= visible_count
            >= audit_selected_count == int(selected_count)
        )
        or audit_selected_count > REGION_CAP_PER_SCENE
        or not isinstance(counts, list)
        or not counts
        or any(not isinstance(count, int) or count < 0 for count in counts)
        or any(count > quotas[index] for index, count in enumerate(counts))
        or sum(counts) != int(selected_count)
    ):
        raise ValueError("sparse teacher selection audit counts differ")
    return audit
