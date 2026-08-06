"""Source-only diagnostics for one sealed sparse full-scalar teacher pair.

The audit is deliberately downstream of the immutable AcceptedV2 and official
multi-view teacher authorities.  It computes descriptive coverage and
consistency statistics without opening RGB, benchmark, target, label, mask, or
query payloads.  Exact core/context roles are reconstructed only when the
caller supplies the SHA-bound support graph; camera geometry is never inferred
from view indices or image dimensions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import torch

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


SCHEMA = "radio_gs.full_scalar_sparse_teacher_quality_audit.v1"
SCHEMA_VERSION = 1
MINIMUM_TOKENS = 24


def audit_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "inputs": (
            "caller_sha_bound_accepted_v2_canonical_region_authority_and_"
            "official_sparse_multiview_siglip2_teacher_authority"
        ),
        "optional_core_context_input": (
            "caller_sha_bound_clean_support_graph_already_bound_by_accepted_"
            "authority"
        ),
        "token_minimum_reference": MINIMUM_TOKENS,
        "crop_area": "integer_source_rgb_bbox_area_and_fraction_of_source_view",
        "support_hit_density_proxy": (
            "exact_marginal_region_gaussian_pixel_hit_pairs_divided_by_"
            "source_bbox_fraction_times_feature_grid_area"
        ),
        "support_occupancy_proxy": (
            "distinct_visible_region_primitives_divided_by_active_region_tokens"
        ),
        "descriptor_dispersion": (
            "one_minus_l2_norm_of_mean_unit_descriptor_for_multiview_regions"
        ),
        "core_context_policy": (
            "exact_frozen_v2_graph_geodesic_reconstruction_or_unavailable"
        ),
        "viewpoint_geometry_policy": (
            "camera_pose_center_or_direction_must_be_explicitly_authoritative_"
            "otherwise_unavailable"
        ),
        "diagnostic_only": True,
        "query_independent": True,
    }


AUDIT_CONTRACT_SHA256 = canonical_json_sha256(audit_contract())


def _distribution(values: torch.Tensor) -> dict[str, Any]:
    tensor = torch.as_tensor(values).detach().double().cpu().reshape(-1)
    if tensor.numel() == 0:
        return {"available": False, "count": 0}
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("quality audit distribution is non-finite")
    quantiles = torch.quantile(
        tensor, torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], dtype=torch.float64)
    )
    return {
        "available": True,
        "count": int(tensor.numel()),
        "minimum": float(tensor.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "maximum": float(tensor.max()),
        "mean": float(tensor.mean()),
    }


def _clean_support_graph(
    graph: object,
    *,
    accepted: Mapping[str, Any],
) -> tuple[PrimitiveSupportGraph, torch.Tensor, torch.Tensor]:
    if not isinstance(graph, Mapping):
        raise ValueError("support graph must be a mapping")
    payload = dict(graph)
    required = {
        "schema_version", "global_rows", "num_global_rows", "xyz",
        "edge_index", "edge_weight", "raw_affinity", "edge_channels",
        "local_sigma", "metadata",
    }
    metadata = payload.get("metadata")
    if (
        set(payload) != required
        or payload.get("schema_version") != 1
        or not isinstance(metadata, Mapping)
        or any(
            metadata.get(key) is not False
            for key in (
                "benchmark_images_opened", "benchmark_masks_opened",
                "text_queries_opened",
            )
        )
    ):
        raise ValueError("quality audit support graph is not clean schema-v1")
    contract = SurfaceRegionContractV2()
    observed_config = dict(metadata.get("graph_config", {}))
    expected_config = asdict(contract.graph_config())
    observed_config.pop("affinity_chunk_size", None)
    expected_config.pop("affinity_chunk_size", None)
    if observed_config != expected_config:
        raise ValueError("quality audit support graph differs from frozen V2 graph")
    global_rows = torch.as_tensor(payload["global_rows"]).long().cpu().contiguous()
    xyz = torch.as_tensor(payload["xyz"]).float().cpu().contiguous()
    global_count = int(payload["num_global_rows"])
    local_count = int(global_rows.numel())
    if (
        global_count != int(accepted["accepted_base_valid"].numel())
        or xyz.shape != (local_count, 3)
        or local_count <= 0
        or not bool(torch.isfinite(xyz).all())
        or global_rows.unique().numel() != local_count
        or bool((global_rows < 0).any())
        or bool((global_rows >= global_count).any())
    ):
        raise ValueError("quality audit support graph rows/xyz differ")
    support = PrimitiveSupportGraph(
        edge_index=payload["edge_index"],
        edge_weight=payload["edge_weight"],
        raw_affinity=payload["raw_affinity"],
        local_sigma=payload["local_sigma"],
        num_nodes=local_count,
        edge_channels=payload["edge_channels"],
    )
    return support, global_rows, xyz


def _core_context_statistics(
    accepted: Mapping[str, Any],
    graph: object | None,
) -> dict[str, Any]:
    if graph is None:
        return {
            "availability": "unavailable",
            "reason": (
                "AcceptedV2 stores token_mask but not core_mask or geodesic_"
                "distance; Euclidean anchor distance alone is not the frozen_"
                "graph-geodesic role authority"
            ),
        }
    support, global_rows, xyz = _clean_support_graph(graph, accepted=accepted)
    contract = SurfaceRegionContractV2()
    if contract.digest != accepted["accepted_v2_authority"]["contract_sha256"]:
        raise ValueError("quality audit frozen V2 region contract differs")
    prepared = contract.prepare_graph(support, xyz)
    canonical = accepted["canonical_region_indices"].long().cpu()
    scales = accepted["scale_indices"].long().cpu()
    rows = accepted["region_rows"].long().cpu()
    token_mask = accepted["token_mask"].bool().cpu()
    anchors = accepted["anchor_index"].long().cpu()
    node_count = int(global_rows.numel())
    expected_scales = torch.div(canonical, node_count, rounding_mode="floor")
    center_local = torch.remainder(canonical, node_count)
    if (
        bool((expected_scales >= len(contract.radii_m)).any())
        or not torch.equal(expected_scales, scales)
        or not torch.equal(
            global_rows[center_local], rows[torch.arange(rows.shape[0]), anchors]
        )
    ):
        raise ValueError("quality audit canonical region/graph anchor differs")
    core_counts = torch.zeros(rows.shape[0], dtype=torch.long)
    context_counts = torch.zeros(rows.shape[0], dtype=torch.long)
    for scale_index, radius in enumerate(contract.radii_m):
        region_indices = torch.where(scales == scale_index)[0]
        for start in range(0, region_indices.numel(), 256):
            batch_indices = region_indices[start : start + 256]
            expansions = contract.expand_batch(
                support,
                xyz,
                center_local[batch_indices].tolist(),
                float(radius),
                prepared_graph=prepared,
            )
            for region_index, expansion in zip(batch_indices.tolist(), expansions):
                selected, core, _distance = expansion
                expected_rows = global_rows[selected]
                observed_rows = rows[region_index][token_mask[region_index]]
                if (
                    not torch.equal(expected_rows, observed_rows)
                    or int(core.numel()) != int(observed_rows.numel())
                ):
                    raise ValueError(
                        "quality audit graph reconstruction differs from AcceptedV2"
                    )
                core_counts[region_index] = int(core.sum())
                context_counts[region_index] = int((~core).sum())
    by_scale = []
    for scale_index, radius in enumerate(contract.radii_m):
        selected = scales == scale_index
        if not bool(selected.any()):
            continue
        by_scale.append(
            {
                "scale_index": scale_index,
                "radius_m": float(radius),
                "region_count": int(selected.sum()),
                "core_token_count": _distribution(core_counts[selected]),
                "context_token_count": _distribution(context_counts[selected]),
                "context_present_region_fraction": float(
                    (context_counts[selected] > 0).double().mean()
                ),
            }
        )
    return {
        "availability": "available",
        "authority": "exact_frozen_v2_graph_geodesic_reconstruction",
        "core_token_count": _distribution(core_counts),
        "context_token_count": _distribution(context_counts),
        "context_present_region_fraction": float(
            (context_counts > 0).double().mean()
        ),
        "by_scale": by_scale,
    }


def _descriptor_statistics(
    pair_rows: torch.Tensor,
    descriptors: torch.Tensor,
    *,
    region_count: int,
) -> dict[str, Any]:
    pairwise: list[torch.Tensor] = []
    region_mean_cosine: list[float] = []
    spherical_dispersion: list[float] = []
    for region in range(region_count):
        values = descriptors[pair_rows == region].double()
        if values.shape[0] < 2:
            continue
        cosine = values @ values.T
        upper = cosine[torch.triu_indices(values.shape[0], values.shape[0], offset=1).unbind()]
        pairwise.append(upper)
        region_mean_cosine.append(float(upper.mean()))
        dispersion = 1.0 - float(torch.linalg.vector_norm(values.mean(dim=0)))
        spherical_dispersion.append(max(0.0, min(1.0, dispersion)))
    combined = torch.cat(pairwise) if pairwise else torch.empty(0, dtype=torch.float64)
    return {
        "multiview_region_count": len(region_mean_cosine),
        "multiview_region_fraction": float(len(region_mean_cosine) / region_count),
        "within_region_unordered_pair_cosine": _distribution(combined),
        "per_region_mean_pairwise_cosine": _distribution(
            torch.tensor(region_mean_cosine, dtype=torch.float64)
        ),
        "per_region_spherical_dispersion": _distribution(
            torch.tensor(spherical_dispersion, dtype=torch.float64)
        ),
    }


def build_quality_audit(
    *,
    accepted_value: object,
    accepted_file_sha256: str,
    teacher_value: object,
    teacher_file_sha256: str,
    support_graph_value: object | None = None,
    support_graph_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate both authorities and build one finite canonical JSON report."""

    accepted_sha = shard._require_sha256(
        accepted_file_sha256, label="quality audit AcceptedV2 file"
    )
    teacher_sha = shard._require_sha256(
        teacher_file_sha256, label="quality audit teacher file"
    )
    accepted = shard.validate_accepted_region_authority(accepted_value)
    teacher = shard.validate_teacher_observation_authority(teacher_value)
    shard.validate_teacher_accepted_sampling_alignment(teacher, accepted)
    if accepted["scene_id"] != teacher["scene_id"]:
        raise ValueError("quality audit scene identities differ")
    expected_accepted_sha = teacher["input_authority"][
        "accepted_region_authority_file_sha256"
    ]
    if expected_accepted_sha != accepted_sha:
        raise ValueError("teacher caller binding to AcceptedV2 file differs")
    expected_graph_sha = accepted["input_authority"]["support_graph_authority"][
        "support_graph_file_sha256"
    ]
    if (support_graph_value is None) != (support_graph_file_sha256 is None):
        raise ValueError("support graph value and SHA must be supplied together")
    graph_sha: str | None = None
    if support_graph_value is not None:
        graph_sha = shard._require_sha256(
            support_graph_file_sha256, label="quality audit support graph file"
        )
        if graph_sha != expected_graph_sha:
            raise ValueError("quality audit support graph caller binding differs")

    token_counts = accepted["token_mask"].sum(dim=1).long().cpu()
    scales = accepted["scale_indices"].long().cpu()
    contract = SurfaceRegionContractV2()
    if contract.digest != accepted["accepted_v2_authority"]["contract_sha256"]:
        raise ValueError("quality audit AcceptedV2 contract is not the frozen V2")
    by_scale = []
    for scale_index in sorted(set(scales.tolist())):
        selected = scales == scale_index
        if scale_index >= len(contract.radii_m):
            raise ValueError("quality audit scale index exceeds frozen radii")
        below = token_counts[selected] < MINIMUM_TOKENS
        by_scale.append(
            {
                "scale_index": int(scale_index),
                "radius_m": float(contract.radii_m[scale_index]),
                "region_count": int(selected.sum()),
                "region_fraction": float(selected.double().mean()),
                "token_count": _distribution(token_counts[selected]),
                "below_minimum_token_region_count": int(below.sum()),
                "below_minimum_token_region_fraction": float(below.double().mean()),
            }
        )

    pair_rows = teacher["pair_region_indices"].long().cpu()
    pair_views = teacher["pair_view_indices"].long().cpu()
    descriptors = teacher["pair_descriptors"].float().cpu()
    boxes = teacher["pair_crop_boxes_tlbr"].long().cpu()
    hits = teacher["pair_support_hit_counts"].double().cpu()
    visible = teacher["pair_visible_primitive_counts"].double().cpu()
    region_count = int(token_counts.numel())
    row_counts = torch.bincount(pair_rows, minlength=region_count)
    histogram = [
        {
            "views": count,
            "region_count": int((row_counts == count).sum()),
            "region_fraction": float((row_counts == count).double().mean()),
        }
        for count in range(1, 5)
    ]

    crop_height = (boxes[:, 2] - boxes[:, 0]).double()
    crop_width = (boxes[:, 3] - boxes[:, 1]).double()
    crop_area = crop_height * crop_width
    image_area = torch.tensor(
        [
            int(teacher["view_records"][view]["source_image_height"])
            * int(teacher["view_records"][view]["source_image_width"])
            for view in pair_views.tolist()
        ],
        dtype=torch.float64,
    )
    grid_area = torch.tensor(
        [
            int(teacher["view_records"][view]["feature_grid_height"])
            * int(teacher["view_records"][view]["feature_grid_width"])
            for view in pair_views.tolist()
        ],
        dtype=torch.float64,
    )
    area_fraction = crop_area / image_area
    grid_crop_area_proxy = (area_fraction * grid_area).clamp_min(1e-12)
    hit_density_proxy = hits / grid_crop_area_proxy
    occupancy_proxy = visible / token_counts[pair_rows].double()
    hit_per_visible = hits / visible
    if bool((occupancy_proxy > 1.0 + 1e-12).any()):
        raise ValueError("quality audit visible primitive count exceeds region tokens")

    below_minimum = token_counts < MINIMUM_TOKENS
    descriptor_stats = _descriptor_statistics(
        pair_rows, descriptors, region_count=region_count
    )
    core_context = _core_context_statistics(accepted, support_graph_value)
    reasons: list[str] = []
    if bool(below_minimum.any()):
        reasons.append("one_or_more_regions_have_fewer_than_24_active_tokens")
    if int((row_counts == 1).sum()) > 0:
        reasons.append("one_or_more_regions_have_only_one_teacher_view")
    overall = "pass" if not reasons else "review_required"
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "contract": audit_contract(),
        "contract_sha256": AUDIT_CONTRACT_SHA256,
        "scene_id": str(accepted["scene_id"]),
        "physical_space_id": str(accepted["physical_space_id"]),
        "input_authority": {
            "accepted_region_authority_file_sha256": accepted_sha,
            "accepted_region_channel_sha256": canonical_json_sha256(
                accepted["channel_sha256"]
            ),
            "teacher_observation_authority_file_sha256": teacher_sha,
            "teacher_observation_channel_sha256": canonical_json_sha256(
                teacher["channel_sha256"]
            ),
            "support_graph_file_sha256": graph_sha,
            "accepted_bound_support_graph_file_sha256": expected_graph_sha,
        },
        "source_access": {
            "accepted_authority_opened": True,
            "teacher_authority_opened": True,
            "support_graph_opened": support_graph_value is not None,
            "source_rgb_opened": False,
            "responsibility_shards_opened": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_queries_opened": False,
            "benchmark_labels_opened": False,
            "target_heldout_opened": False,
            "text_queries_opened": False,
        },
        "statistics": {
            "region_count": region_count,
            "region_view_pair_count": int(pair_rows.numel()),
            "source_view_count": len(teacher["view_records"]),
            "tokens": {
                "minimum_tokens_reference": MINIMUM_TOKENS,
                "active_token_count": _distribution(token_counts),
                "below_minimum_token_region_count": int(below_minimum.sum()),
                "below_minimum_token_region_fraction": float(
                    below_minimum.double().mean()
                ),
                "by_scale": by_scale,
            },
            "core_context": core_context,
            "views_per_region": {
                "histogram": histogram,
                "view_count": _distribution(row_counts),
            },
            "crop_bbox": {
                "height_pixels": _distribution(crop_height),
                "width_pixels": _distribution(crop_width),
                "area_pixels": _distribution(crop_area),
                "area_fraction_of_source_view": _distribution(area_fraction),
            },
            "support_evidence": {
                "support_hit_pair_count": _distribution(hits),
                "visible_primitive_count": _distribution(visible),
                "support_hit_density_proxy": _distribution(hit_density_proxy),
                "support_occupancy_proxy": _distribution(occupancy_proxy),
                "support_hits_per_visible_primitive": _distribution(hit_per_visible),
            },
            "descriptor_consistency": descriptor_stats,
            "viewpoint_geometry": {
                "availability": "unavailable",
                "reason": (
                    "sealed AcceptedV2 and teacher schemas expose view identity, "
                    "image/grid dimensions, and responsibility indices but no "
                    "authoritative camera center, rotation, pose, ray direction, "
                    "or baseline; indices are not treated as geometry"
                ),
            },
        },
        "conclusion_gate": {
            "authority_and_row_alignment": "pass",
            "minimum_24_token_diagnostic": (
                "pass" if not bool(below_minimum.any()) else "review_required"
            ),
            "cross_view_descriptor_diagnostic": (
                "available"
                if descriptor_stats["multiview_region_count"] > 0
                else "unavailable"
            ),
            "core_context_diagnostic": core_context["availability"],
            "viewpoint_geometry_diagnostic": "unavailable",
            "overall": overall,
            "review_reasons": reasons,
        },
    }
    payload["authority_sha256"] = canonical_json_sha256(payload)
    return validate_quality_audit(payload)


def validate_quality_audit(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("sparse teacher quality audit must be a mapping")
    payload = dict(value)
    required = {
        "schema", "schema_version", "contract", "contract_sha256", "scene_id",
        "physical_space_id", "input_authority", "source_access", "statistics",
        "conclusion_gate", "authority_sha256",
    }
    content = dict(payload)
    observed_authority = content.pop("authority_sha256", None)
    if (
        set(payload) != required
        or payload.get("schema") != SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("contract") != audit_contract()
        or payload.get("contract_sha256") != AUDIT_CONTRACT_SHA256
        or observed_authority != canonical_json_sha256(content)
    ):
        raise ValueError("sparse teacher quality audit authority differs")
    # This simultaneously proves that every nested number is finite and every
    # nested value is JSON serializable under the immutable authority hash.
    if not isinstance(payload.get("statistics"), Mapping) or not isinstance(
        payload.get("conclusion_gate"), Mapping
    ):
        raise ValueError("sparse teacher quality audit statistics/gate differ")
    for key, expected in {
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "target_heldout_opened": False,
        "text_queries_opened": False,
    }.items():
        if payload.get("source_access", {}).get(key) is not expected:
            raise ValueError("sparse teacher quality audit source access differs")
    if payload["conclusion_gate"].get("authority_and_row_alignment") != "pass":
        raise ValueError("sparse teacher quality audit alignment gate differs")
    return payload
