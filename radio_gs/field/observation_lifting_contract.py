"""Versioned, query-free observation-lifting contract for canonical fields."""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from typing import Any, Mapping, Sequence


CANONICAL_OBSERVATION_CONTRACT_NAME = "canonical-mpr-v1"
CANONICAL_FULL_OBSERVATION_CONTRACT_NAME = "canonical-full-observation-mpr-v1"
# Keep the original 240-view full-observation reconstruction frozen as an
# explicit control.  ``v2`` is intentionally a new contract rather than a
# quiet parameter change: a 480-view full-.sens source must be allowed to
# carry its fixed, label-free coverage prefix through MPR as well.
CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME = "canonical-full-observation-mpr-v2"
CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME = "canonical-full-observation-mpr-v3"
CANONICAL_FULL_OBSERVATION_CONTRACT_NAMES = frozenset(
    {
        CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
    }
)


def is_canonical_full_observation_contract(name: str) -> bool:
    """Return whether ``name`` is a versioned coverage-ranked MPR contract."""

    return str(name) in CANONICAL_FULL_OBSERVATION_CONTRACT_NAMES


def canonical_observation_contract(
    name: str = CANONICAL_OBSERVATION_CONTRACT_NAME,
) -> dict[str, Any]:
    """Return a versioned query-free MPR policy used by the canonical field.

    Image resolution, camera calibration, and held-out frame IDs deliberately
    do not live here: they are dataset provenance, not tunable lifting policy.
    The full-observation variant additionally requires a pre-recorded,
    query-free source coverage order rather than a data-set-specific heuristic.
    """

    name = str(name)
    if name == CANONICAL_OBSERVATION_CONTRACT_NAME:
        view_selection = "uniform_temporal_deterministic"
        require_full_source = False
        maximum_views = 120
    elif name in CANONICAL_FULL_OBSERVATION_CONTRACT_NAMES:
        # These are intentionally separate, source-aware contracts rather
        # than mutations of canonical-mpr-v1.  They are used only where a
        # full ScanNet field-source manifest provides a label-free greedy
        # coverage order.  v1 stays immutable as the 240-view control; v2
        # preserves a 480-view source prefix; v3 is the 960-view promotion
        # rung after a label-free v2 support-gate failure.
        view_selection = "field_source_coverage_ranked_deterministic"
        require_full_source = True
        maximum_views = {
            CANONICAL_FULL_OBSERVATION_CONTRACT_NAME: 240,
            CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME: 480,
            CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME: 960,
        }[name]
    else:
        raise ValueError(f"unsupported canonical observation contract: {name}")
    contract = {
        "name": name,
        "schema_version": 1,
        "view_selection": view_selection,
        "maximum_views": maximum_views,
        "aggregation_mode": "raster_gaussian_top1",
        "registration_weight_mode": "alpha_depth",
        "raster_view_fusion": "contribution_mean",
        "normalize_each_view": True,
        "per_view_normalization_stage": "pixel_feature_before_raster_lifting",
        "depth_tolerance": 0.08,
        "relative_depth_tolerance": 0.02,
        "alpha_threshold": 0.02,
        "feature_projection_order": "per_view_before_mpr",
        "responsibility_sharing": "exact_sidecar_across_feature_spaces",
        "query_independent": True,
        "requires_full_observation_source_contract": require_full_source,
    }
    # Promoted declarations carry their source-size lower bound. Do not add
    # this key to v1: v1 cache digests are a frozen 240-view control.
    if name in {
        CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
    }:
        contract["minimum_source_views"] = int(maximum_views)
    return contract


def observation_contract_sha256(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(contract), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def apply_canonical_observation_contract(args: Namespace) -> dict[str, Any]:
    """Apply canonical policy to a cache-builder argument namespace."""

    contract = canonical_observation_contract(
        str(getattr(args, "observation_contract", CANONICAL_OBSERVATION_CONTRACT_NAME))
    )
    args.max_views = int(contract["maximum_views"])
    args.aggregation_mode = str(contract["aggregation_mode"])
    args.registration_weight_mode = str(contract["registration_weight_mode"])
    args.raster_view_fusion = str(contract["raster_view_fusion"])
    args.normalize_each_view = bool(contract["normalize_each_view"])
    args.depth_tolerance = float(contract["depth_tolerance"])
    args.relative_depth_tolerance = float(contract["relative_depth_tolerance"])
    args.alpha_threshold = float(contract["alpha_threshold"])
    args.robust_mpr = False
    return contract


def select_full_observation_coverage_ranked_dataset_indices(
    *,
    dataset_frame_ids: Sequence[int],
    candidate_dataset_indices: Sequence[int],
    ranked_frame_ids: Sequence[int],
    maximum_views: int,
) -> list[int]:
    """Map a source-manifest coverage order into the active dataset rows.

    The source contract is constructed only from registered RGB-D observations.
    Excluded validation/query frames are removed before this routine, so it
    never uses a benchmark target to choose a remaining frame.
    """

    frame_ids = [int(value) for value in dataset_frame_ids]
    candidates = [int(value) for value in candidate_dataset_indices]
    ranked = [int(value) for value in ranked_frame_ids]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("dataset frame IDs must be unique")
    if len(set(ranked)) != len(ranked):
        raise ValueError("coverage-ranked frame IDs must be unique")
    if set(ranked) != set(frame_ids):
        raise ValueError(
            "coverage-ranked source frames must exactly match the active RGB-D dataset"
        )
    if not candidates:
        raise ValueError("full-observation MPR has no candidate frames")
    if any(index < 0 or index >= len(frame_ids) for index in candidates):
        raise IndexError("candidate dataset index is outside the active RGB-D dataset")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate dataset indices must be unique")
    maximum_views = int(maximum_views)
    if maximum_views <= 0:
        raise ValueError("maximum_views must be positive")
    index_by_frame = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    allowed = set(candidates)
    selected = [
        index_by_frame[frame_id]
        for frame_id in ranked
        if index_by_frame[frame_id] in allowed
    ]
    if not selected:
        raise ValueError("coverage-ranked source contains no non-held-out MPR views")
    return selected[:maximum_views]


def validate_observation_contract_metadata(
    metadata: Mapping[str, Any],
    *,
    require_declaration: bool = True,
    contract_name: str | None = None,
) -> dict[str, Any]:
    """Fail closed when an MPR cache does not implement the canonical policy.

    ``require_declaration=False`` is only for certifying old caches whose full
    metadata predates the versioned contract.  It still checks every numerical
    and categorical policy field.
    """

    declared = metadata.get("observation_lifting_contract")
    if contract_name is None and isinstance(declared, Mapping):
        contract_name = str(declared.get("name", CANONICAL_OBSERVATION_CONTRACT_NAME))
    expected = canonical_observation_contract(
        contract_name or CANONICAL_OBSERVATION_CONTRACT_NAME
    )
    if require_declaration and not isinstance(declared, Mapping):
        raise ValueError("MPR cache does not declare an observation-lifting contract")
    if isinstance(declared, Mapping):
        mismatched_declared = [
            key for key, value in expected.items() if declared.get(key) != value
        ]
        if mismatched_declared:
            raise ValueError(
                "MPR observation-lifting declaration differs: "
                f"{sorted(mismatched_declared)}"
            )
        digest = str(metadata.get("observation_lifting_contract_sha256", ""))
        if digest != observation_contract_sha256(expected):
            raise ValueError("MPR observation-lifting contract digest differs")

    direct_fields = {
        "aggregation_mode": expected["aggregation_mode"],
        "registration_weight_mode": expected["registration_weight_mode"],
        "raster_view_fusion": expected["raster_view_fusion"],
        "normalize_each_view": expected["normalize_each_view"],
        "per_view_normalization_applied": True,
        "depth_tolerance": expected["depth_tolerance"],
        "relative_depth_tolerance": expected["relative_depth_tolerance"],
        "alpha_threshold": expected["alpha_threshold"],
    }
    mismatched = [
        key for key, value in direct_fields.items() if metadata.get(key) != value
    ]
    if int(metadata.get("num_declared_views", 0)) > int(expected["maximum_views"]):
        mismatched.append("num_declared_views")
    if mismatched:
        raise ValueError(
            f"MPR metadata violates canonical observation contract: {sorted(mismatched)}"
        )
    if bool(metadata.get("robust_mpr", False)):
        raise ValueError("canonical raster MPR must not declare center-only robust fusion")
    minimum_source_views = int(expected.get("minimum_source_views", 0))
    if minimum_source_views and int(
        metadata.get("full_observation_source_view_count", 0)
    ) < minimum_source_views:
        raise ValueError(
            "MPR cache lacks the full "
            f"{minimum_source_views}-view source prefix required by its observation contract"
        )
    if bool(metadata.get("benchmark_masks_opened", False)) or bool(
        metadata.get("text_queries_opened", False)
    ):
        raise ValueError("canonical observation lifting must be query/label free")
    return expected
