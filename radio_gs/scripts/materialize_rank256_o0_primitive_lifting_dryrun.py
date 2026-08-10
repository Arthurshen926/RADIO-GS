#!/usr/bin/env python3
"""Source-gated, query-free rank256-to-O0 primitive lifting dry-run.

This is an independent no-clobber chain.  It never reads benchmark queries,
images, labels, masks, or metrics.  The prefix-limited output is an interface
and geometry validation artifact only; it is explicitly not a formal LERF
candidate or a benchmark result.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.field.rank256_primitive_residual_lifting import (
    CONTRACT_SHA256,
    Rank256PrimitiveLiftingConfig,
    lift_rank256_region_residual_to_o0_multiscale,
    lifting_contract,
)
from radio_gs.interfaces import surface_region_rank256_champion as champion
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.scripts import (
    materialize_lerf_multiscale_query_score_cache as o0_contract,
)
from radio_gs.scripts import run_surface_region_rank256_champion_target as target_pipeline
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_sha_bound_project_checkpoint_mapping,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


IMPLEMENTATION = Path(__file__).resolve()
LIFTING_IMPLEMENTATION = (
    Path(__file__).resolve().parents[1]
    / "field/rank256_primitive_residual_lifting.py"
)
EXECUTION_SCHEMA = "radio_gs.rank256_o0_primitive_lifting_dryrun_authority.v1"
OUTPUT_SCHEMA = "radio_gs.rank256_o0_primitive_lifting_prefix_dryrun.v1"
STATUS = "authorized_query_free_nonformal_prefix_dryrun"


def access_audit() -> dict[str, bool]:
    return {
        "source_promotion_validated_before_target_files": True,
        "o0_query_free_descriptor_opened": True,
        "renderer_geometry_opened": True,
        "benchmark_queries_opened": False,
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "target_metrics_computed": False,
        "formal_candidate_materialized": False,
    }


def _existing(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path) or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be an existing canonical regular file")
    return path


def _new(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path):
        raise ValueError(f"{label} must be an absolute canonical path")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} already exists: {path}")
    return path


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    # Source PASS validation must remain the first artifact access.
    gate = champion.validate_champion_source(
        args.source_variant,
        args.source_result,
        expected_sha256=args.expected_source_result_sha256,
    )
    output_authority = _new(args.output_authority, label="lifting authority")
    output = _new(args.output, label="lifting dry-run output")
    o0_descriptor = _existing(args.o0_descriptor, label="O0 descriptor")
    target_descriptor = _existing(
        args.rank256_target_descriptor, label="rank256 target descriptor"
    )
    accepted = _existing(args.target_accepted_v2, label="target AcceptedV2")
    renderer = _existing(args.renderer_geometry_checkpoint, label="renderer geometry")
    row_limit = int(args.valid_row_prefix_limit)
    if row_limit <= 0:
        raise ValueError("valid_row_prefix_limit must be positive")
    config = Rank256PrimitiveLiftingConfig(
        max_angle_radians=float(args.max_angle_radians),
        minimum_region_reliability=float(args.minimum_region_reliability),
    )
    authority = {
        "schema": EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": STATUS,
        "source_variant": str(args.source_variant),
        "source_result": dict(gate["source_result"]),
        "scene_id": str(args.scene_id),
        "implementation": file_record(IMPLEMENTATION),
        "lifting_implementation": file_record(LIFTING_IMPLEMENTATION),
        "lifting_contract": lifting_contract(),
        "lifting_contract_sha256": CONTRACT_SHA256,
        "input_authority": {
            "o0_descriptor": file_record(o0_descriptor),
            "rank256_target_descriptor": file_record(target_descriptor),
            "target_accepted_v2": file_record(accepted),
            "renderer_geometry_checkpoint": file_record(renderer),
        },
        "configuration": config.to_dict(),
        "scope": {
            "valid_row_prefix_limit": row_limit,
            "prefix_order": "o0_global_rows_ascending_storage_order",
            "dry_run_only": True,
            "formal_candidate_authorized": False,
        },
        "output": str(output),
        "materialization_authorized": True,
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
        "access_audit": access_audit(),
    }
    write_frozen_json(output_authority, authority)
    return {
        "status": "rank256_o0_lifting_dryrun_authority_built",
        "authority": file_record(output_authority),
    }


def validate_authority(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_output: str | Path | None = None,
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="rank256 O0 lifting dry-run authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "source_variant",
        "source_result",
        "scene_id",
        "implementation",
        "lifting_implementation",
        "lifting_contract",
        "lifting_contract_sha256",
        "input_authority",
        "configuration",
        "scope",
        "output",
        "materialization_authorized",
        "query_execution_authorized",
        "metric_execution_authorized",
        "access_audit",
    }
    authority = dict(raw)
    if set(authority) != required:
        raise ValueError("rank256 O0 lifting authority fields differ")
    source_result = champion._record(
        authority["source_result"], label="lifting source result"
    )
    gate = champion.validate_champion_source(
        str(authority["source_variant"]),
        source_result["path"],
        expected_sha256=source_result["sha256"],
    )
    if (
        authority["schema"] != EXECUTION_SCHEMA
        or authority["schema_version"] != 1
        or authority["status"] != STATUS
        or authority["source_variant"] not in champion.SOURCE_VARIANTS
        or source_result != gate["source_result"]
        or authority["lifting_contract"] != lifting_contract()
        or authority["lifting_contract_sha256"] != CONTRACT_SHA256
        or authority["materialization_authorized"] is not True
        or authority["query_execution_authorized"] is not False
        or authority["metric_execution_authorized"] is not False
        or authority["access_audit"] != access_audit()
    ):
        raise ValueError("rank256 O0 lifting authority header differs")
    if (
        validate_file_record(
            authority["implementation"], label="lifting materializer"
        )
        != IMPLEMENTATION
        or validate_file_record(
            authority["lifting_implementation"], label="lifting module"
        )
        != LIFTING_IMPLEMENTATION
    ):
        raise ValueError("rank256 O0 lifting implementation differs")
    names = {
        "o0_descriptor",
        "rank256_target_descriptor",
        "target_accepted_v2",
        "renderer_geometry_checkpoint",
    }
    inputs = authority["input_authority"]
    if not isinstance(inputs, Mapping) or set(inputs) != names:
        raise ValueError("rank256 O0 lifting input authority differs")
    inputs = {
        name: champion._record(inputs[name], label=f"lifting input {name}")
        for name in sorted(names)
    }
    configuration = authority["configuration"]
    if not isinstance(configuration, Mapping) or set(configuration) != {
        "max_angle_radians",
        "minimum_region_reliability",
    }:
        raise ValueError("rank256 O0 lifting configuration differs")
    config = Rank256PrimitiveLiftingConfig(
        max_angle_radians=float(configuration["max_angle_radians"]),
        minimum_region_reliability=float(
            configuration["minimum_region_reliability"]
        ),
    )
    scope = authority["scope"]
    if (
        not isinstance(scope, Mapping)
        or set(scope)
        != {
            "valid_row_prefix_limit",
            "prefix_order",
            "dry_run_only",
            "formal_candidate_authorized",
        }
        or type(scope["valid_row_prefix_limit"]) is not int
        or int(scope["valid_row_prefix_limit"]) <= 0
        or scope["prefix_order"] != "o0_global_rows_ascending_storage_order"
        or scope["dry_run_only"] is not True
        or scope["formal_candidate_authorized"] is not False
    ):
        raise ValueError("rank256 O0 lifting dry-run scope differs")
    output = champion._output(authority["output"], label="lifting output")
    if expected_output is not None and output != str(
        Path(expected_output).expanduser().resolve()
    ):
        raise ValueError("rank256 O0 lifting output differs")
    authority.update(
        {
            "source_result": source_result,
            "verified_source_gate": gate,
            "input_authority": inputs,
            "configuration_object": config,
            "output": output,
            "verified_record": {"path": str(source), "sha256": digest},
        }
    )
    return authority


def _renderer_xyz(value: Mapping[str, Any]) -> torch.Tensor:
    state = value.get("model_state_dict")
    xyz = state.get("_xyz") if isinstance(state, Mapping) else None
    if (
        not torch.is_tensor(xyz)
        or not xyz.is_floating_point()
        or xyz.ndim != 2
        or xyz.shape[1] != 3
        or not bool(torch.isfinite(xyz).all())
    ):
        raise ValueError("renderer geometry checkpoint lacks finite _xyz [N,3]")
    return xyz.detach().float().cpu().contiguous()


def _validate_o0(
    value: Mapping[str, Any], *, renderer_xyz: torch.Tensor
) -> dict[str, Any]:
    metadata = value.get("metadata")
    xyz = value.get("xyz")
    valid = value.get("valid")
    global_rows = value.get("global_rows")
    descriptor = value.get("features_by_scale")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("feature_space")
        != "official_siglip2_summary_descriptor_multiscale"
        or metadata.get("query_set_invariant") is not True
        or metadata.get("text_queries_opened") is not False
        or metadata.get("benchmark_images_opened") is not False
        or metadata.get("benchmark_masks_opened") is not False
    ):
        raise ValueError("O0 descriptor query-free metadata differs")
    if (
        not torch.is_tensor(xyz)
        or not xyz.is_floating_point()
        or xyz.shape != renderer_xyz.shape
        or not torch.equal(xyz.detach().float().cpu(), renderer_xyz)
    ):
        raise ValueError("O0 descriptor and renderer geometry/order differ")
    primitive_count = int(xyz.shape[0])
    if (
        not torch.is_tensor(valid)
        or valid.dtype != torch.bool
        or valid.shape != (primitive_count,)
        or not torch.is_tensor(global_rows)
        or global_rows.dtype != torch.int64
        or global_rows.ndim != 1
        or not torch.equal(global_rows.cpu(), torch.where(valid.cpu())[0])
        or not torch.is_tensor(descriptor)
        or not descriptor.is_floating_point()
        or descriptor.shape
        != (int(global_rows.numel()), o0_contract.SCALE_COUNT, o0_contract.DESCRIPTOR_DIMENSION)
        or not bool(torch.isfinite(descriptor).all())
        or bool((torch.linalg.vector_norm(descriptor.float(), dim=-1) <= 1e-8).any())
    ):
        raise ValueError("O0 sparse multiscale descriptor layout differs")
    return {
        "xyz": xyz.detach().float().cpu().contiguous(),
        "valid": valid.detach().bool().cpu().contiguous(),
        "global_rows": global_rows.detach().long().cpu().contiguous(),
        "descriptor": descriptor.detach().cpu().contiguous(),
    }


def _reindex_regions_to_prefix(
    *,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    prefix_global_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = region_rows.long().cpu()
    tokens = token_mask.bool().cpu()
    safe = rows.clamp_min(0)
    positions = torch.searchsorted(prefix_global_rows, safe)
    bounded = positions.clamp_max(len(prefix_global_rows) - 1)
    matched = tokens & (positions < len(prefix_global_rows)) & (
        prefix_global_rows[bounded] == safe
    )
    keep = matched.any(dim=1)
    if not bool(keep.any()):
        raise ValueError("O0 dry-run prefix does not intersect any rank256 region")
    local_rows = positions[keep].long().contiguous()
    local_mask = matched[keep].bool().contiguous()
    local_rows[~local_mask] = -1
    return torch.where(keep)[0].long(), local_rows, local_mask


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = _new(args.output, label="rank256 O0 lifting dry-run output")
    execution = validate_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        expected_output=output,
    )
    inputs = execution["input_authority"]

    target_raw, _, _ = load_torch_mapping(
        inputs["rank256_target_descriptor"]["path"],
        expected_sha256=inputs["rank256_target_descriptor"]["sha256"],
        map_location="cpu",
        label="rank256 target descriptor",
    )
    target = champion.validate_target_descriptor(target_raw)
    target_execution = target_pipeline.validate_target_authority(
        target["target_execution_authority"]["path"],
        expected_sha256=target["target_execution_authority"]["sha256"],
    )
    if (
        target["source_variant"] != execution["source_variant"]
        or target_execution["source_result"] != execution["source_result"]
        or target["scene_id"] != execution["scene_id"]
        or target["input_authority"]["target_accepted_v2"]
        != inputs["target_accepted_v2"]
    ):
        raise ValueError("rank256 target/source/scene binding differs")

    accepted_raw, _, _ = load_torch_mapping(
        inputs["target_accepted_v2"]["path"],
        expected_sha256=inputs["target_accepted_v2"]["sha256"],
        map_location="cpu",
        label="target AcceptedV2",
    )
    accepted = validate_target_accepted_v2_authority(accepted_raw)
    if (
        accepted["scene_id"] != target["scene_id"]
        or accepted["physical_space_id"] != target["physical_space_id"]
        or not torch.equal(
            accepted["canonical_region_indices"],
            target["canonical_region_indices"],
        )
        or accepted["region_fingerprints"] != target["region_fingerprints"]
    ):
        raise ValueError("rank256 target and AcceptedV2 region axes differ")

    renderer_raw, _, _ = load_sha_bound_project_checkpoint_mapping(
        inputs["renderer_geometry_checkpoint"]["path"],
        expected_sha256=inputs["renderer_geometry_checkpoint"]["sha256"],
        map_location="cpu",
        label="renderer geometry checkpoint",
    )
    renderer_xyz = _renderer_xyz(renderer_raw)
    o0_raw, _, _ = load_torch_mapping(
        inputs["o0_descriptor"]["path"],
        expected_sha256=inputs["o0_descriptor"]["sha256"],
        map_location="cpu",
        label="query-free O0 descriptor",
    )
    o0 = _validate_o0(o0_raw, renderer_xyz=renderer_xyz)
    if (
        accepted["geometry_fingerprint"]["num_gaussians"] != len(renderer_xyz)
        or target["physical_space_authority"]["geometry_checkpoint_sha256"]
        != inputs["renderer_geometry_checkpoint"]["sha256"]
    ):
        raise ValueError("rank256/O0/renderer physical-space binding differs")

    prefix_count = min(
        int(execution["scope"]["valid_row_prefix_limit"]),
        int(o0["global_rows"].numel()),
    )
    prefix_global = o0["global_rows"][:prefix_count].clone()
    prefix_descriptor = o0["descriptor"][:prefix_count].clone()
    # Existing O0 descriptors are already unit up to FP16 roundoff; explicit
    # normalization is deliberately not used because fallback bytes must stay
    # identical to the frozen capability carrier.
    region_positions, local_rows, local_mask = _reindex_regions_to_prefix(
        region_rows=accepted["region_rows"],
        token_mask=accepted["token_mask"],
        prefix_global_rows=prefix_global,
    )
    result = lift_rank256_region_residual_to_o0_multiscale(
        o0_primitive_descriptor=prefix_descriptor,
        primitive_valid_mask=torch.ones(prefix_count, dtype=torch.bool),
        region_base_descriptor=accepted["accepted_v2_e0"][region_positions],
        region_semantic_descriptor=target["semantic_descriptor"][region_positions],
        region_rows=local_rows,
        token_mask=local_mask,
        canonical_region_indices=accepted["canonical_region_indices"][region_positions],
        region_reliability=target["reliability_score"][region_positions],
        region_active_mask=target["active_update_mask"][region_positions],
        region_ood_mask=target["effective_ood_mask"][region_positions],
        config=execution["configuration_object"],
    )
    fallback_equal = torch.equal(
        result.primitive_descriptor[result.fallback_mask].view(torch.uint8),
        prefix_descriptor[result.fallback_mask].view(torch.uint8),
    )
    if not fallback_equal:
        raise RuntimeError("dry-run fallback changed O0 bytes")
    payload = {
        "schema": OUTPUT_SCHEMA,
        "schema_version": 1,
        "status": "complete_nonformal_query_free_prefix_dryrun",
        "scene_id": target["scene_id"],
        "physical_space_id": target["physical_space_id"],
        "source_variant": execution["source_variant"],
        "producer": file_record(IMPLEMENTATION),
        "lifting_implementation": file_record(LIFTING_IMPLEMENTATION),
        "execution_authority": dict(execution["verified_record"]),
        "input_authority": dict(inputs),
        "lifting_contract": lifting_contract(),
        "lifting_contract_sha256": CONTRACT_SHA256,
        "configuration": execution["configuration_object"].to_dict(),
        "scope": dict(execution["scope"]),
        "primitive_global_rows": prefix_global.contiguous(),
        "selected_region_positions": region_positions.contiguous(),
        "lifted_descriptor": result.primitive_descriptor,
        "region_contribution_mask": result.region_contribution_mask,
        "coverage_count": result.coverage_count,
        "aggregate_reliability": result.aggregate_reliability,
        "aggregate_residual_norm": result.aggregate_residual_norm,
        "angular_step_radians": result.angular_step_radians,
        "updated_mask": result.updated_mask,
        "fallback_mask": result.fallback_mask,
        "fallback_bitwise_o0": fallback_equal,
        "tensor_sha256": {
            "primitive_global_rows": tensor_sha256(prefix_global),
            "o0_prefix_descriptor": tensor_sha256(prefix_descriptor),
            "lifted_descriptor": tensor_sha256(result.primitive_descriptor),
            "coverage_count": tensor_sha256(result.coverage_count),
            "angular_step_radians": tensor_sha256(result.angular_step_radians),
        },
        "routing_audit": {
            "valid_prefix_primitives": prefix_count,
            "intersecting_regions": int(region_positions.numel()),
            "contributing_regions": int(result.region_contribution_mask.sum()),
            "covered_primitives": int((result.coverage_count > 0).sum()),
            "updated_primitive_scales": int(result.updated_mask.sum()),
            "fallback_primitive_scales": int(result.fallback_mask.sum()),
            "maximum_angular_step_radians": float(
                result.angular_step_radians.max()
            ),
        },
        "formal_candidate_authorized": False,
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
        "access_audit": access_audit(),
    }
    write_torch_noclobber(output, payload)
    return {
        "status": payload["status"],
        "output": file_record(output),
        "routing_audit": payload["routing_audit"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-authority")
    build.add_argument("--source-variant", choices=champion.SOURCE_VARIANTS, required=True)
    build.add_argument("--source-result", required=True)
    build.add_argument("--expected-source-result-sha256", required=True)
    build.add_argument("--scene-id", required=True)
    build.add_argument("--o0-descriptor", required=True)
    build.add_argument("--rank256-target-descriptor", required=True)
    build.add_argument("--target-accepted-v2", required=True)
    build.add_argument("--renderer-geometry-checkpoint", required=True)
    build.add_argument("--valid-row-prefix-limit", type=int, required=True)
    build.add_argument("--max-angle-radians", type=float, required=True)
    build.add_argument("--minimum-region-reliability", type=float, required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--output-authority", required=True)
    build.set_defaults(func=build_authority)
    run = subparsers.add_parser("materialize")
    run.add_argument("--execution-authority", required=True)
    run.add_argument("--expected-execution-authority-sha256", required=True)
    run.add_argument("--output", required=True)
    run.set_defaults(func=materialize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.func(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "EXECUTION_SCHEMA",
    "OUTPUT_SCHEMA",
    "access_audit",
    "build_authority",
    "materialize",
    "validate_authority",
]
