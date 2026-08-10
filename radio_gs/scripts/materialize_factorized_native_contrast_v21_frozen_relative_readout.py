#!/usr/bin/env python3
"""Materialize the frozen-relative, selected-scale-only contrast V2.1 unary."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import (
    factorized_native_contrast_v21_exact_native_v3_bridge as exact_bridge,
)
from radio_gs.interfaces import (
    factorized_native_contrast_v21_frozen_relative_readout as formal,
)
from radio_gs.interfaces import (
    factorized_native_contrast_v21_target_descriptor as target_formal,
)
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.scripts import (
    materialize_factorized_native_contrast_v21_lerf_exact_relevance as exact_script,
)
from radio_gs.scripts.materialize_lerf_multiscale_query_score_cache import (
    _renderer_checkpoint_xyz,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_sha_bound_project_checkpoint_mapping,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


IMPLEMENTATION = Path(__file__).resolve()
EXECUTION_SCHEMA = (
    "radio_gs.factorized_native_contrast_v21_frozen_relative_execution.v1"
)
EXECUTION_STATUS = "authorized_frozen_relative_query_opaque_unary_only"
DEPENDENCIES = {
    "formal_readout": Path(formal.__file__).resolve(),
    "contrast_exact_bridge": Path(exact_bridge.__file__).resolve(),
    "target_descriptor_formal": Path(target_formal.__file__).resolve(),
    "target_accepted_v2_formal": Path(
        validate_target_accepted_v2_authority.__code__.co_filename
    ).resolve(),
    "renderer_xyz_loader": Path(_renderer_checkpoint_xyz.__code__.co_filename).resolve(),
    "frozen_protocol": Path(formal.FROZEN_PROTOCOL_RECORD["path"]).resolve(),
}


def _float32_rows_sha256(value: torch.Tensor) -> str:
    """Match the geometry fingerprint's header-free little-endian row hash."""

    array = (
        torch.as_tensor(value)
        .detach()
        .float()
        .cpu()
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )
    return hashlib.sha256(array.tobytes()).hexdigest()


def _canonical_new(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path):
        raise ValueError(f"{label} must be canonical absolute")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} already exists: {path}")
    return path


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    record = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    verified = validate_file_record(record, label=label)
    if str(verified) != record["path"]:
        raise ValueError(f"{label} path is not canonical")
    return record


def _load_inputs(
    *,
    exact_relevance_record: Mapping[str, str],
    renderer_geometry_record: Mapping[str, str],
) -> dict[str, Any]:
    relevance_raw, relevance_sha, relevance_path = load_torch_mapping(
        exact_relevance_record["path"],
        expected_sha256=exact_relevance_record["sha256"],
        map_location="cpu",
        label="frozen-relative contrast exact relevance",
    )
    relevance_record = {"path": str(relevance_path), "sha256": relevance_sha}
    dispatched = exact_bridge.dispatch_relevance_schema(relevance_raw)
    execution = exact_script.validate_authority(
        dispatched.payload["query_execution_authority"]["path"],
        expected_sha256=dispatched.payload["query_execution_authority"]["sha256"],
        expected_output=relevance_record["path"],
    )
    relevance = exact_bridge.validate_contrast_exact_lineage(
        dispatched=dispatched,
        relevance_record=relevance_record,
        query_execution=execution,
    )
    gate = execution["verified_prequery_gate"]
    descriptor = target_formal.validate_target_descriptor_authority(
        gate["descriptor"]
    )
    descriptor_record = dict(gate["target_descriptor_record"])
    accepted_record = dict(descriptor["input_authority"]["target_accepted_v2"])
    accepted_raw, accepted_sha, accepted_path = load_torch_mapping(
        accepted_record["path"],
        expected_sha256=accepted_record["sha256"],
        map_location="cpu",
        label="frozen-relative target AcceptedV2",
    )
    accepted = validate_target_accepted_v2_authority(accepted_raw)
    accepted_record = {"path": str(accepted_path), "sha256": accepted_sha}

    renderer_raw, renderer_sha, renderer_path = (
        load_sha_bound_project_checkpoint_mapping(
            renderer_geometry_record["path"],
            expected_sha256=renderer_geometry_record["sha256"],
            map_location="cpu",
            label="frozen-relative renderer geometry checkpoint",
        )
    )
    renderer_record = {"path": str(renderer_path), "sha256": renderer_sha}
    renderer_xyz = _renderer_checkpoint_xyz(renderer_raw)
    rows = accepted["region_rows"]
    anchor = accepted["anchor_index"]
    anchor_rows = rows[torch.arange(rows.shape[0]), anchor].long().contiguous()
    anchor_xyz = renderer_xyz[anchor_rows].float().contiguous()
    fingerprint_sha = canonical_json_sha256(accepted["region_fingerprints"])
    if (
        relevance_record != dict(exact_relevance_record)
        or renderer_record != dict(renderer_geometry_record)
        or relevance["input_authority"]["target_descriptor"] != descriptor_record
        or descriptor["input_authority"]["target_accepted_v2"] != accepted_record
        or relevance["scene_id"] != descriptor["scene_id"]
        or relevance["scene_id"] != accepted["scene_id"]
        or relevance["physical_space_id"] != descriptor["physical_space_id"]
        or relevance["physical_space_id"] != accepted["physical_space_id"]
        or relevance["region_row_ids"] != descriptor["region_row_ids"]
        or relevance["region_fingerprints"] != descriptor["region_fingerprints"]
        or descriptor["region_fingerprints"] != accepted["region_fingerprints"]
        or relevance["channel_sha256"]["region_fingerprints"] != fingerprint_sha
        or not torch.equal(
            relevance["canonical_region_indices"],
            descriptor["canonical_region_indices"],
        )
        or not torch.equal(
            relevance["canonical_region_indices"],
            accepted["canonical_region_indices"],
        )
        or renderer_sha
        != accepted["physical_space_authority"]["geometry_checkpoint_sha256"]
        or descriptor["physical_space_authority"]
        != accepted["physical_space_authority"]
        or renderer_xyz.shape[0] != accepted["accepted_base_valid"].numel()
        or _float32_rows_sha256(renderer_xyz)
        != accepted["geometry_fingerprint"]["xyz_sha256"]
    ):
        raise ValueError("frozen-relative relevance/scale/geometry lineage differs")
    return {
        "relevance_record": relevance_record,
        "query_execution_record": dict(execution["verified_record"]),
        "descriptor_record": descriptor_record,
        "accepted_record": accepted_record,
        "renderer_record": renderer_record,
        "relevance": relevance,
        "descriptor": descriptor,
        "accepted": accepted,
        "anchor_rows": anchor_rows,
        "anchor_xyz": anchor_xyz,
        "region_fingerprints_sha256": fingerprint_sha,
    }


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    authority_output = _canonical_new(
        args.output_authority, label="frozen-relative execution authority"
    )
    output = _canonical_new(args.output, label="frozen-relative readout output")
    exact_record = _record(
        {
            "path": str(Path(args.exact_relevance).expanduser().resolve()),
            "sha256": args.expected_exact_relevance_sha256,
        },
        label="contrast exact relevance",
    )
    renderer_record = _record(
        {
            "path": str(Path(args.renderer_geometry_checkpoint).expanduser().resolve()),
            "sha256": args.expected_renderer_geometry_checkpoint_sha256,
        },
        label="renderer geometry checkpoint",
    )
    inputs = _load_inputs(
        exact_relevance_record=exact_record,
        renderer_geometry_record=renderer_record,
    )
    authority = {
        "schema": EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": EXECUTION_STATUS,
        "scene_id": inputs["relevance"]["scene_id"],
        "physical_space_id": inputs["relevance"]["physical_space_id"],
        "implementation": file_record(IMPLEMENTATION),
        "implementation_dependencies": {
            name: file_record(path) for name, path in sorted(DEPENDENCIES.items())
        },
        "readout_contract": formal.readout_contract(),
        "readout_contract_sha256": formal.READOUT_CONTRACT_SHA256,
        "input_authority": {
            "exact_relevance": inputs["relevance_record"],
            "query_execution": inputs["query_execution_record"],
            "target_descriptor": inputs["descriptor_record"],
            "target_accepted_v2": inputs["accepted_record"],
            "renderer_geometry_checkpoint": inputs["renderer_record"],
        },
        "configuration": {
            "semantic_levels": formal.SEMANTIC_LEVELS,
            "knn_neighbors": formal.KNN_NEIGHBORS,
            "knn_chunk_size": formal.KNN_CHUNK_SIZE,
            "mask_threshold": formal.MASK_THRESHOLD,
            "selected_scale_only": True,
            "graph_or_relation": "forbidden",
        },
        "output": str(output),
        "materialization_authorized": True,
        "metric_execution_authorized": False,
        "access_audit": formal.access_audit(),
    }
    write_frozen_json(authority_output, authority)
    return {
        "status": "frozen_relative_execution_authority_built",
        "authority": file_record(authority_output),
        "output": str(output),
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
        label="frozen-relative execution authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "physical_space_id",
        "implementation",
        "implementation_dependencies",
        "readout_contract",
        "readout_contract_sha256",
        "input_authority",
        "configuration",
        "output",
        "materialization_authorized",
        "metric_execution_authorized",
        "access_audit",
    }
    authority = dict(raw)
    if (
        set(authority) != required
        or authority.get("schema") != EXECUTION_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != EXECUTION_STATUS
        or authority.get("readout_contract") != formal.readout_contract()
        or authority.get("readout_contract_sha256")
        != formal.READOUT_CONTRACT_SHA256
        or authority.get("configuration")
        != {
            "semantic_levels": formal.SEMANTIC_LEVELS,
            "knn_neighbors": formal.KNN_NEIGHBORS,
            "knn_chunk_size": formal.KNN_CHUNK_SIZE,
            "mask_threshold": formal.MASK_THRESHOLD,
            "selected_scale_only": True,
            "graph_or_relation": "forbidden",
        }
        or authority.get("materialization_authorized") is not True
        or authority.get("metric_execution_authorized") is not False
        or authority.get("access_audit") != formal.access_audit()
    ):
        raise ValueError("frozen-relative execution authority differs")
    implementation = _record(
        authority["implementation"], label="frozen-relative implementation"
    )
    if implementation != file_record(IMPLEMENTATION):
        raise ValueError("frozen-relative implementation record differs")
    dependencies = authority.get("implementation_dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(DEPENDENCIES):
        raise ValueError("frozen-relative dependency fields differ")
    dependencies = {
        name: _record(dependencies[name], label=f"frozen-relative {name}")
        for name in sorted(DEPENDENCIES)
    }
    if any(dependencies[name] != file_record(path) for name, path in DEPENDENCIES.items()):
        raise ValueError("frozen-relative dependency record differs")
    input_authority = authority.get("input_authority")
    input_names = {
        "exact_relevance",
        "query_execution",
        "target_descriptor",
        "target_accepted_v2",
        "renderer_geometry_checkpoint",
    }
    if not isinstance(input_authority, Mapping) or set(input_authority) != input_names:
        raise ValueError("frozen-relative input authority fields differ")
    records = {
        name: _record(input_authority[name], label=f"frozen-relative {name}")
        for name in sorted(input_names)
    }
    inputs = _load_inputs(
        exact_relevance_record=records["exact_relevance"],
        renderer_geometry_record=records["renderer_geometry_checkpoint"],
    )
    if (
        records["query_execution"] != inputs["query_execution_record"]
        or records["target_descriptor"] != inputs["descriptor_record"]
        or records["target_accepted_v2"] != inputs["accepted_record"]
        or authority["scene_id"] != inputs["relevance"]["scene_id"]
        or authority["physical_space_id"]
        != inputs["relevance"]["physical_space_id"]
    ):
        raise ValueError("frozen-relative nested input lineage differs")
    output_raw = str(authority["output"])
    output = str(Path(output_raw).expanduser().resolve())
    if output_raw != output:
        raise ValueError("frozen-relative output must be canonical absolute")
    if expected_output is not None and output != str(
        Path(expected_output).expanduser().resolve()
    ):
        raise ValueError("frozen-relative expected output differs")
    authority.update(
        {
            "implementation": implementation,
            "implementation_dependencies": dependencies,
            "input_authority": records,
            "verified_inputs": inputs,
            "output": output,
            "verified_record": {"path": str(source), "sha256": digest},
        }
    )
    return authority


def _audit(readout: formal.FrozenRelativeRegionReadout) -> dict[str, Any]:
    counts = readout.unary_candidate_mask.sum(dim=0)
    return {
        "opaque_query_axes": int(readout.raw_relevance.shape[1]),
        "semantic_levels": formal.SEMANTIC_LEVELS,
        "selected_scale_counts": {
            str(level): int((readout.selected_scale_indices == level).sum())
            for level in range(formal.SEMANTIC_LEVELS)
        },
        "query_gate_passed": int(readout.query_gate.sum()),
        "query_gate_failed": int((~readout.query_gate).sum()),
        "candidate_count_min": int(counts.min()),
        "candidate_count_median": int(counts.float().median()),
        "candidate_count_max": int(counts.max()),
        "outside_selected_scale_nonzero": int(
            readout.relative_relevance[~readout.selected_scale_eligibility]
            .count_nonzero()
        ),
        "outside_selected_scale_candidates": int(
            readout.unary_candidate_mask[~readout.selected_scale_eligibility].sum()
        ),
        "graph_or_relation_applied": False,
        "query_identifiers_consumed_by_readout": False,
        "target_metric_computed": False,
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = _canonical_new(args.output, label="frozen-relative readout output")
    execution = validate_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        expected_output=output,
    )
    inputs = execution["verified_inputs"]
    relevance = inputs["relevance"]
    accepted = inputs["accepted"]
    readout = formal.frozen_relative_region_readout(
        raw_relevance=relevance["region_absolute_relevance"],
        scale_indices=accepted["scale_indices"],
        anchor_xyz=inputs["anchor_xyz"],
        chunk_size=execution["configuration"]["knn_chunk_size"],
    )
    payload = {
        "schema": formal.READOUT_SCHEMA,
        "schema_version": formal.READOUT_SCHEMA_VERSION,
        "contract": formal.readout_contract(),
        "contract_sha256": formal.READOUT_CONTRACT_SHA256,
        "scene_id": relevance["scene_id"],
        "physical_space_id": relevance["physical_space_id"],
        "producer": file_record(IMPLEMENTATION),
        "execution_authority": dict(execution["verified_record"]),
        "input_authority": dict(execution["input_authority"]),
        "region_fingerprints_sha256": inputs["region_fingerprints_sha256"],
        "query_axis_count": int(readout.raw_relevance.shape[1]),
        "canonical_region_indices": relevance["canonical_region_indices"].clone(),
        "scale_indices": accepted["scale_indices"].clone(),
        "anchor_rows": inputs["anchor_rows"].clone(),
        "anchor_xyz": inputs["anchor_xyz"].clone(),
        "raw_relevance": readout.raw_relevance,
        "smoothed_relevance": readout.smoothed_relevance,
        "remapped_relevance": readout.remapped_relevance,
        "raw_smoothed_peaks": readout.raw_smoothed_peaks,
        "selected_scale_indices": readout.selected_scale_indices,
        "selected_scale_eligibility": readout.selected_scale_eligibility,
        "relative_relevance": readout.relative_relevance,
        "query_gate": readout.query_gate,
        "unary_candidate_mask": readout.unary_candidate_mask,
        "audit": _audit(readout),
        "channel_sha256": {},
        "access_audit": formal.access_audit(),
    }
    payload["channel_sha256"] = formal.channel_sha256(payload)
    payload = formal.validate_readout_authority(payload)
    written = write_torch_noclobber(output, payload)
    return {
        "status": "frozen_relative_selected_scale_unary_complete",
        "candidate_status": "query_free_unary_only_no_metric_authority",
        "scene_id": payload["scene_id"],
        "audit": payload["audit"],
        "readout_contract_sha256": formal.READOUT_CONTRACT_SHA256,
        "output": file_record(written),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-authority")
    build.add_argument("--exact-relevance", required=True)
    build.add_argument("--expected-exact-relevance-sha256", required=True)
    build.add_argument("--renderer-geometry-checkpoint", required=True)
    build.add_argument(
        "--expected-renderer-geometry-checkpoint-sha256", required=True
    )
    build.add_argument("--output-authority", required=True)
    build.add_argument("--output", required=True)
    build.set_defaults(handler=build_authority)

    validate = subparsers.add_parser("validate-authority")
    validate.add_argument("--execution-authority", required=True)
    validate.add_argument("--expected-execution-authority-sha256", required=True)
    validate.add_argument("--expected-output")
    validate.set_defaults(
        handler=lambda args: {
            "status": "frozen_relative_execution_authority_valid",
            "authority": validate_authority(
                args.execution_authority,
                expected_sha256=args.expected_execution_authority_sha256,
                expected_output=args.expected_output,
            )["verified_record"],
        }
    )

    run = subparsers.add_parser("materialize")
    run.add_argument("--execution-authority", required=True)
    run.add_argument("--expected-execution-authority-sha256", required=True)
    run.add_argument("--output", required=True)
    run.set_defaults(handler=materialize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.handler(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "DEPENDENCIES",
    "EXECUTION_SCHEMA",
    "EXECUTION_STATUS",
    "IMPLEMENTATION",
    "build_authority",
    "build_parser",
    "materialize",
    "validate_authority",
]
