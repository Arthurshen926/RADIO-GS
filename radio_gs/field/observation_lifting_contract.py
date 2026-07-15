"""Versioned, query-free observation-lifting contract for canonical fields."""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from typing import Any, Mapping


CANONICAL_OBSERVATION_CONTRACT_NAME = "canonical-mpr-v1"


def canonical_observation_contract() -> dict[str, Any]:
    """Return the dataset-independent MPR policy used by the canonical field.

    Image resolution, camera calibration, available-view count, and held-out
    frame IDs deliberately do not live here: they are dataset provenance, not
    tunable observation-lifting policy.
    """

    return {
        "name": CANONICAL_OBSERVATION_CONTRACT_NAME,
        "schema_version": 1,
        "view_selection": "uniform_temporal_deterministic",
        "maximum_views": 120,
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
    }


def observation_contract_sha256(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(contract), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def apply_canonical_observation_contract(args: Namespace) -> dict[str, Any]:
    """Apply canonical policy to a cache-builder argument namespace."""

    contract = canonical_observation_contract()
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


def validate_observation_contract_metadata(
    metadata: Mapping[str, Any],
    *,
    require_declaration: bool = True,
) -> dict[str, Any]:
    """Fail closed when an MPR cache does not implement the canonical policy.

    ``require_declaration=False`` is only for certifying old caches whose full
    metadata predates the versioned contract.  It still checks every numerical
    and categorical policy field.
    """

    expected = canonical_observation_contract()
    declared = metadata.get("observation_lifting_contract")
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
    if bool(metadata.get("benchmark_masks_opened", False)) or bool(
        metadata.get("text_queries_opened", False)
    ):
        raise ValueError("canonical observation lifting must be query/label free")
    return expected
