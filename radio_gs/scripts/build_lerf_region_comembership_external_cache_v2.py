#!/usr/bin/env python3
"""Build a LERF external score cache from formal V2 co-membership inference.

The globally selected method, K and probability threshold are read only from
the inference authority.  This adapter reuses the frozen O0 positive/negative
relevancy, VALA KNN/remap and AcceptedV2 scale-major alignment, then delegates
bounded expansion to ``bounded_region_union_from_o0``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.querying import bounded_region_comembership_query_readout as bounded_query
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen
from radio_gs.scripts import build_lerf_region_comembership_external_cache_v1 as v1_cache
from radio_gs.scripts.infer_region_comembership_v2 import (
    validate_inference_authority,
)
from radio_gs.scripts.materialize_region_comembership_features_v2 import (
    validate_feature_authority,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.lerf_region_comembership_external_scores.v2"
EXECUTION_AUTHORITY_SCHEMA = (
    "radio_gs.region_comembership_v2_target_query_readout_execution_authority.v1"
)
EXECUTION_AUTHORITY_STATUS = (
    "authorized_after_v2_target_feature_and_inference_complete"
)
IMPLEMENTATION_DEPENDENCIES = {
    "frozen_o0": Path(frozen.__file__).resolve(),
    "bounded_query_readout": Path(bounded_query.__file__).resolve(),
    "scale_major_helpers": Path(v1_cache.__file__).resolve(),
}


def _canonical_output_path(value: object, *, label: str) -> str:
    raw = str(value)
    resolved = str(Path(raw).expanduser().resolve())
    if raw != resolved:
        raise ValueError(f"{label} must be an absolute canonical path")
    return resolved


def _sha256(value: object, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def validate_v2_authority_binding(
    *,
    feature: Mapping[str, Any],
    inference: Mapping[str, Any],
    feature_record: Mapping[str, str],
) -> dict[str, Any]:
    """Require one target scene, canonical axis and exact feature file record."""

    rule = inference.get("selected_rule")
    if (
        feature.get("domain") != "target"
        or feature.get("target_execution_authority") is None
        or inference.get("domain") != "target"
        or inference.get("scene_id") != feature.get("scene_id")
        or inference.get("target_execution_authority")
        != feature.get("target_execution_authority")
        or inference.get("feature_authority") != dict(feature_record)
        or inference.get("source_access") != feature.get("source_access")
        or inference.get("region_fingerprints") != feature.get("region_fingerprints")
        or inference.get("region_fingerprints_sha256")
        != feature.get("region_fingerprints_sha256")
        or inference.get("canonical_axis_sha256")
        != feature.get("canonical_axis_sha256")
        or inference.get("pair_axis_sha256") != feature.get("pair_axis_sha256")
        or not torch.equal(
            torch.as_tensor(inference.get("canonical_region_indices")),
            torch.as_tensor(feature.get("canonical_region_indices")),
        )
        or not torch.equal(
            torch.as_tensor(inference.get("pair_indices")),
            torch.as_tensor(feature.get("pair_indices")),
        )
        or not isinstance(rule, Mapping)
        or set(rule) != {"method", "maximum_regions", "threshold"}
    ):
        raise ValueError("LERF V2 feature/inference authority binding differs")
    return dict(rule)


def validate_renderer_geometry_binding(
    *,
    feature: Mapping[str, Any],
    accepted: Mapping[str, Any],
    accepted_record: Mapping[str, str],
    renderer_geometry_checkpoint_sha256: str,
) -> None:
    """Fail closed unless readout and target regions share one geometry space."""

    inputs = feature.get("input_authority")
    physical = accepted.get("physical_space_authority")
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("accepted_v2") != dict(accepted_record)
        or not isinstance(physical, Mapping)
        or physical.get("geometry_checkpoint_sha256")
        != renderer_geometry_checkpoint_sha256
        or accepted.get("scene_id") != feature.get("scene_id")
        or accepted.get("region_fingerprints") != feature.get("region_fingerprints")
        or not torch.equal(
            torch.as_tensor(accepted.get("canonical_region_indices")),
            torch.as_tensor(feature.get("canonical_region_indices")),
        )
        or not torch.equal(
            torch.as_tensor(accepted.get("region_rows")),
            torch.as_tensor(feature.get("region_rows")),
        )
        or not torch.equal(
            torch.as_tensor(accepted.get("token_mask")),
            torch.as_tensor(feature.get("token_mask")),
        )
    ):
        raise ValueError("LERF V2 renderer/AcceptedV2 geometry binding differs")


def validate_support_geometry_binding(
    *,
    graph_xyz: torch.Tensor,
    global_rows: torch.Tensor,
    full_xyz: torch.Tensor,
    o0_valid: torch.Tensor,
) -> dict[str, int]:
    """Bind one primitive axis without conflating two validity domains.

    ``global_rows`` is the query-independent MPR/support domain, whereas
    ``o0_valid`` is the frozen text-readout domain.  The domains may differ,
    but both must refer to the exact same full primitive geometry and row
    order.
    """

    rows = torch.as_tensor(global_rows).detach().long().cpu()
    graph = torch.as_tensor(graph_xyz).detach().float().cpu()
    xyz = torch.as_tensor(full_xyz).detach().float().cpu()
    valid = torch.as_tensor(o0_valid).detach().bool().cpu()
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or graph.ndim != 2
        or graph.shape != (rows.numel(), 3)
        or rows.ndim != 1
        or valid.shape != (xyz.shape[0],)
        or bool((rows < 0).any())
        or bool((rows >= xyz.shape[0]).any())
        or int(torch.unique(rows).numel()) != int(rows.numel())
        or not torch.equal(graph, xyz[rows])
    ):
        raise ValueError("frozen O0 and V2 support geometry differ")
    graph_active = torch.zeros_like(valid).index_fill_(0, rows, True)
    intersection = graph_active & valid
    return {
        "graph_active_primitives": int(graph_active.sum()),
        "o0_valid_primitives": int(valid.sum()),
        "intersection_primitives": int(intersection.sum()),
        "o0_only_primitives": int((valid & ~graph_active).sum()),
        "mpr_only_primitives": int((graph_active & ~valid).sum()),
    }


def mask_membership_to_o0_valid(
    primitive_membership: torch.Tensor, o0_valid: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Remove selections outside the frozen text-readout validity domain."""

    membership = torch.as_tensor(primitive_membership).detach().float().cpu()
    valid = torch.as_tensor(o0_valid).detach().bool().cpu()
    if (
        membership.ndim != 2
        or valid.shape != (membership.shape[0],)
        or not bool(torch.isfinite(membership).all())
    ):
        raise ValueError("V2 membership/O0 validity axes differ")
    invalid_selected = (membership > 0) & ~valid[:, None]
    masked = membership * valid[:, None].to(membership.dtype)
    return masked.contiguous(), int(invalid_selected.sum())


def bounded_readout_from_v2(
    *,
    feature: Mapping[str, Any],
    inference: Mapping[str, Any],
    region_o0_scores: torch.Tensor,
    num_primitives: int,
):
    """Apply the authority-selected rule; there is no caller rule override."""

    rule = dict(inference["selected_rule"])
    return bounded_query.bounded_region_union_from_o0(
        region_o0_scores=region_o0_scores,
        pair_indices=feature["pair_indices"],
        pair_probabilities=inference["pair_probabilities"],
        probability_threshold=float(rule["threshold"]),
        method=str(rule["method"]),
        maximum_regions=int(rule["maximum_regions"]),
        region_rows=feature["region_rows"],
        token_mask=feature["token_mask"],
        num_primitives=int(num_primitives),
    )


def validate_query_readout_execution_authority(
    path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    """Validate feature/inference before opening either query cache record."""

    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="V2 target query-readout execution authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "implementation",
        "implementation_dependencies",
        "feature_authority",
        "inference_authority",
        "positive_cache",
        "negative_cache",
        "renderer_geometry_checkpoint_sha256",
        "knn_chunk_size",
        "output_cache",
        "output_report",
        "query_readout_authorized",
        "target_metric_authorized",
        "access_audit",
    }
    authority = dict(raw)
    if (
        set(authority) != required
        or authority.get("schema") != EXECUTION_AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != EXECUTION_AUTHORITY_STATUS
        or not isinstance(authority.get("scene_id"), str)
        or not authority["scene_id"]
        or authority.get("query_readout_authorized") is not True
        or authority.get("target_metric_authorized") is not False
        or authority.get("access_audit")
        != {
            "benchmark_queries_opened": True,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_labels_opened": False,
            "target_metrics_computed": False,
        }
        or not isinstance(authority.get("knn_chunk_size"), int)
        or int(authority["knn_chunk_size"]) <= 0
    ):
        raise ValueError("V2 target query-readout execution header differs")
    implementation = validate_file_record(
        authority["implementation"], label="V2 query-readout implementation"
    )
    if implementation != Path(__file__).resolve():
        raise ValueError("V2 query-readout authority binds another implementation")
    dependencies = authority.get("implementation_dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(
        IMPLEMENTATION_DEPENDENCIES
    ):
        raise ValueError("V2 query-readout implementation dependencies differ")
    for name, expected in IMPLEMENTATION_DEPENDENCIES.items():
        verified = validate_file_record(
            dependencies[name], label=f"V2 query-readout dependency {name}"
        )
        if verified != expected:
            raise ValueError(
                f"V2 query-readout authority binds another dependency: {name}"
            )

    # Do not touch positive/negative query cache records before this complete
    # feature/inference promotion and axis validation.
    feature_path = validate_file_record(
        authority["feature_authority"], label="V2 query-readout feature authority"
    )
    feature_raw, feature_sha, feature_source = load_torch_mapping(
        feature_path,
        expected_sha256=authority["feature_authority"]["sha256"],
        map_location="cpu",
        label="V2 query-readout feature authority",
    )
    feature = validate_feature_authority(feature_raw)
    inference_path = validate_file_record(
        authority["inference_authority"], label="V2 query-readout inference authority"
    )
    inference_raw, inference_sha, inference_source = load_torch_mapping(
        inference_path,
        expected_sha256=authority["inference_authority"]["sha256"],
        map_location="cpu",
        label="V2 query-readout inference authority",
    )
    inference = validate_inference_authority(inference_raw)
    selected_rule = validate_v2_authority_binding(
        feature=feature,
        inference=inference,
        feature_record={"path": str(feature_source), "sha256": feature_sha},
    )
    if feature["scene_id"] != authority["scene_id"]:
        raise ValueError("V2 query-readout scene differs from feature authority")

    renderer_sha = _sha256(
        authority["renderer_geometry_checkpoint_sha256"],
        label="V2 query-readout renderer geometry checkpoint",
    )
    accepted_record = feature.get("input_authority", {}).get("accepted_v2")
    accepted_path = validate_file_record(
        accepted_record, label="V2 query-readout target AcceptedV2 authority"
    )
    accepted_raw, accepted_sha, accepted_source = load_torch_mapping(
        accepted_path,
        expected_sha256=accepted_record["sha256"],
        map_location="cpu",
        label="V2 query-readout target AcceptedV2 authority",
    )
    accepted = validate_target_accepted_v2_authority(accepted_raw)
    verified_accepted_record = {
        "path": str(accepted_source),
        "sha256": accepted_sha,
    }
    validate_renderer_geometry_binding(
        feature=feature,
        accepted=accepted,
        accepted_record=verified_accepted_record,
        renderer_geometry_checkpoint_sha256=renderer_sha,
    )

    positive_path = validate_file_record(
        authority["positive_cache"], label="V2 query-readout positive O0 cache"
    )
    negative_path = validate_file_record(
        authority["negative_cache"], label="V2 query-readout negative O0 cache"
    )
    output = _canonical_output_path(
        authority["output_cache"], label="V2 query-readout cache output"
    )
    report = _canonical_output_path(
        authority["output_report"], label="V2 query-readout report output"
    )
    if output == report:
        raise ValueError("V2 query-readout cache and report outputs must differ")
    authority["feature_authority"] = {
        "path": str(feature_source),
        "sha256": feature_sha,
    }
    authority["inference_authority"] = {
        "path": str(inference_source),
        "sha256": inference_sha,
    }
    authority["positive_cache"] = {
        "path": str(positive_path),
        "sha256": str(authority["positive_cache"]["sha256"]),
    }
    authority["negative_cache"] = {
        "path": str(negative_path),
        "sha256": str(authority["negative_cache"]["sha256"]),
    }
    authority["renderer_geometry_checkpoint_sha256"] = renderer_sha
    authority["output_cache"] = output
    authority["output_report"] = report
    authority["verified_record"] = {"path": str(source), "sha256": digest}
    authority["verified_feature"] = feature
    authority["verified_inference"] = inference
    authority["verified_accepted"] = accepted
    authority["verified_accepted_record"] = verified_accepted_record
    authority["selected_rule"] = selected_rule
    return authority


def run(args: argparse.Namespace) -> dict[str, Any]:
    execution = validate_query_readout_execution_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    output = Path(execution["output_cache"])
    report_path = Path(execution["output_report"])
    if (
        output.exists()
        or output.is_symlink()
        or report_path.exists()
        or report_path.is_symlink()
    ):
        raise FileExistsError("LERF V2 external cache outputs must be new")
    feature = execution["verified_feature"]
    inference = execution["verified_inference"]
    feature_record = execution["feature_authority"]
    inference_record = execution["inference_authority"]
    selected_rule = execution["selected_rule"]

    accepted = execution["verified_accepted"]

    positive_payload, positive_sha, positive_path = load_torch_mapping(
        execution["positive_cache"]["path"],
        expected_sha256=execution["positive_cache"]["sha256"],
        map_location="cpu",
        label="positive frozen O0 cache",
    )
    negative_payload, negative_sha, negative_path = load_torch_mapping(
        execution["negative_cache"]["path"],
        expected_sha256=execution["negative_cache"]["sha256"],
        map_location="cpu",
        label="canonical-negative frozen O0 cache",
    )
    query_ids = tuple(str(value) for value in positive_payload["query_ids"])
    positive = frozen.validate_ours_multiscale_query_score_cache(
        positive_payload,
        expected_xyz=torch.as_tensor(positive_payload["xyz"]),
        expected_query_ids=query_ids,
        expected_renderer_geometry_checkpoint_sha256=(
            execution["renderer_geometry_checkpoint_sha256"]
        ),
    )
    negative = frozen.validate_ours_multiscale_query_score_cache(
        negative_payload,
        expected_xyz=positive_payload["xyz"],
        expected_query_ids=frozen.NEGATIVE_PROMPTS,
        expected_renderer_geometry_checkpoint_sha256=(
            execution["renderer_geometry_checkpoint_sha256"]
        ),
    )
    for name in (
        "valid",
        "scale_ids",
        "scale_radii_m",
        "xyz_sha256",
        "field_checkpoint_sha256",
        "readout_checkpoint_sha256",
        "renderer_geometry_checkpoint_sha256",
    ):
        left, right = getattr(positive, name), getattr(negative, name)
        if not bool(
            torch.equal(left, right) if torch.is_tensor(left) else left == right
        ):
            raise ValueError(f"positive/negative frozen O0 cache {name} differs")
    probability = frozen.canonical_negative_relevancy_query_scores(
        positive.query_scores, negative.query_scores, logit_scale=10.0
    )
    count, scales, queries = probability.shape
    smoothed = frozen.vala_knn_smoothed_scores(
        probability.reshape(count, scales * queries),
        positive_payload["xyz"],
        k=10,
        chunk_size=int(execution["knn_chunk_size"]),
        valid_mask=positive.valid,
    ).reshape(count, scales, queries)
    remapped = frozen.vala_minmax_remap_scores(
        smoothed.reshape(count, scales * queries), valid_mask=positive.valid
    ).reshape(count, scales, queries)

    graph_record = feature["input_authority"]["support_graph"]
    graph_path = validate_file_record(graph_record, label="V2 feature support graph")
    graph, _, _ = load_torch_mapping(
        graph_path,
        expected_sha256=graph_record["sha256"],
        map_location="cpu",
        label="V2 feature support graph",
    )
    global_rows = torch.as_tensor(graph["global_rows"]).long().cpu()
    full_xyz = torch.as_tensor(positive_payload["xyz"]).float().cpu()
    validity_domain_audit = validate_support_geometry_binding(
        graph_xyz=graph["xyz"],
        global_rows=global_rows,
        full_xyz=full_xyz,
        o0_valid=positive.valid,
    )
    v1_cache.validate_scale_major_alignment(
        canonical_region_indices=accepted["canonical_region_indices"],
        scale_indices=accepted["scale_indices"],
        anchor_count=int(global_rows.numel()),
        o0_scale_radii_m=tuple(positive.scale_radii_m),
    )
    all_regions = v1_cache._scale_major_anchor_probability(remapped, global_rows)
    canonical = feature["canonical_region_indices"].long()
    if int(canonical.max()) >= all_regions.shape[0]:
        raise ValueError("AcceptedV2 canonical indices exceed frozen O0 axis")
    region_o0 = all_regions[canonical]
    readout = bounded_readout_from_v2(
        feature=feature,
        inference=inference,
        region_o0_scores=region_o0,
        num_primitives=int(full_xyz.shape[0]),
    )
    query_scores, selected_invalid_removed = mask_membership_to_o0_valid(
        readout.primitive_membership, positive.valid
    )
    validity_domain_audit["selected_invalid_memberships_removed"] = int(
        selected_invalid_removed
    )
    cache = {
        "schema": SCHEMA,
        "query_scores": query_scores,
        "valid": positive.valid.contiguous(),
        "xyz": full_xyz.contiguous(),
        "metadata": {
            "query_names": list(query_ids),
            "score_semantics": "binary_AcceptedV2_bounded_region_union_membership",
            "candidate_axis": "AcceptedV2_canonical_4096",
            "seed_rule": "highest_frozen_O0_then_lowest_canonical_index",
            "selected_rule": selected_rule,
            "rule_source": "formal_V2_inference_authority_only",
            "positive_relevancy": "frozen_canonical_negative_logit_scale_10",
            "smoothing": "frozen_VALA_KNN_k10_then_independent_minmax",
            "feature_authority": feature_record,
            "inference_authority": {
                **inference_record,
            },
            "positive_cache": {"path": str(positive_path), "sha256": positive_sha},
            "negative_cache": {"path": str(negative_path), "sha256": negative_sha},
            "producer": file_record(Path(__file__).resolve()),
            "query_readout_execution_authority": execution["verified_record"],
            "validity_domain_audit": validity_domain_audit,
        },
        "selection": {
            "seed_region_indices": readout.seed_region_indices,
            "selected_region_masks": readout.selected_region_masks,
            "canonical_region_indices": canonical,
        },
    }
    written = write_torch_noclobber(output, cache)
    report = {
        "schema": SCHEMA,
        "status": "target_region_comembership_v2_external_cache_complete",
        "cache": file_record(written),
        "query_ids": list(query_ids),
        "selected_rule": selected_rule,
        "seed_region_indices": readout.seed_region_indices.tolist(),
        "selected_region_counts": readout.selected_region_masks.sum(dim=0).tolist(),
        "selected_primitive_counts": query_scores.sum(dim=0)
        .int()
        .tolist(),
        "validity_domain_audit": validity_domain_audit,
        "candidate_axis": "AcceptedV2_canonical_4096",
        "query_readout_execution_authority": execution["verified_record"],
        "benchmark_queries_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_metric_computed": False,
        "target_metric_computed": False,
    }
    write_frozen_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
