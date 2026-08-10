#!/usr/bin/env python3
"""Build an authority-bound O0 anchor self-completion candidate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import posixpath
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import lerf_o0_anchor_self_completion as formal
from radio_gs.interfaces.factorized_primitive_state import (
    load_factorized_primitive_state,
)
from radio_gs.scripts.build_lerf_o0_anchored_graph_residual_cache import (
    exact_o0_readout,
)
from radio_gs.scripts.materialize_lerf_multiscale_query_score_cache import (
    _renderer_checkpoint_xyz,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_sha_bound_project_checkpoint_mapping,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


CACHE_AUTHORITY_SCHEMA = "radio_gs.lerf_o0_anchor_self_completion_execution.v2"
METRIC_AUTHORITY_SCHEMA = (
    "radio_gs.lerf_o0_anchor_self_completion_frozen_metric_execution.v2"
)
IMPLEMENTATION = Path(__file__).resolve()
INTERFACE = Path(formal.__file__).resolve()
LAUNCHER = Path(__file__).resolve().parent / "run_lerf_o0_anchor_self_completion_metric.py"
METRIC_PROTOCOL = {
    "protocol_preset": "vala_paper_3d",
    "score_threshold": 0.6,
    "score_postprocess": "none",
    "projection_mode": "selected_only_alpha",
    "threshold_scan": False,
}
KNN_CHUNK_SIZE = 65536


def _record(path: object, digest: object, *, label: str) -> dict[str, str]:
    raw = str(path)
    canonical = str(Path(raw).expanduser().resolve())
    value = {"path": canonical, "sha256": str(digest)}
    if raw != canonical:
        raise ValueError(f"{label} path must be canonical absolute")
    validate_file_record(value, label=label)
    return value


def _new_path(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path) or path.exists() or path.is_symlink():
        raise ValueError(f"{label} must be a new canonical path")
    return path


def _unopened_path(value: object, *, label: str) -> str:
    raw = str(value)
    if not raw.startswith("/") or posixpath.normpath(raw) != raw:
        raise ValueError(f"{label} must be canonical absolute")
    return raw


def _cache_build_access() -> dict[str, bool]:
    return {
        "input_file_records_validated": True,
        "input_payloads_opened": False,
        "query_names_opened": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
        "threshold_scan": False,
    }


def _cache_execute_access() -> dict[str, bool]:
    return {
        "exact_O0_scores_opened": True,
        "query_independent_region_support_opened": True,
        "query_independent_region_reliability_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
        "threshold_scan": False,
    }


def build_cache_authority(args: argparse.Namespace) -> dict[str, Any]:
    output = _new_path(args.output_authority, label="cache execution authority")
    output_cache = _new_path(args.output_cache, label="self-completion cache")
    output_report = _new_path(args.output_report, label="self-completion report")
    inputs = {
        "exact_o0_cache": _record(
            args.exact_o0_cache,
            args.expected_exact_o0_cache_sha256,
            label="exact O0 cache",
        ),
        "positive_o0_cache": _record(
            args.positive_o0_cache,
            args.expected_positive_o0_cache_sha256,
            label="positive frozen O0 cache",
        ),
        "negative_o0_cache": _record(
            args.negative_o0_cache,
            args.expected_negative_o0_cache_sha256,
            label="negative frozen O0 cache",
        ),
        "region_features": _record(
            args.region_features,
            args.expected_region_features_sha256,
            label="region features",
        ),
        "target_descriptor": _record(
            args.target_descriptor,
            args.expected_target_descriptor_sha256,
            label="target descriptor",
        ),
        "factorized_primitive_state": _record(
            args.factorized_primitive_state,
            args.expected_factorized_primitive_state_sha256,
            label="factorized primitive state",
        ),
        "renderer_geometry_checkpoint": _record(
            args.renderer_geometry_checkpoint,
            args.expected_renderer_geometry_checkpoint_sha256,
            label="renderer geometry checkpoint",
        ),
    }
    authority = {
        "schema": CACHE_AUTHORITY_SCHEMA,
        "schema_version": 2,
        "status": "authorized_frozen_O0_anchor_self_completion_cache_only",
        "scene_id": str(args.scene_id),
        "implementation": file_record(IMPLEMENTATION),
        "interface": file_record(INTERFACE),
        "interface_contract_sha256": formal.CONTRACT_SHA256,
        "fixed_rule": formal.completion_contract(),
        "exact_o0_replay": {
            "positive_and_negative_multiscale_caches_bound": True,
            "canonical_negative_logit_scale": 10.0,
            "vala_knn_neighbors": 10,
            "knn_chunk_size": KNN_CHUNK_SIZE,
            "required_proof": (
                "recomputed_exact_o0_query_scores_selected_scales_valid_xyz_"
                "bitwise_equal_bound_reference"
            ),
        },
        "input_authority": inputs,
        "output_cache": str(output_cache),
        "output_report": str(output_report),
        "target_score_cache_authorized": True,
        "target_quality_execution_authorized": False,
        "access_audit": _cache_build_access(),
    }
    write_frozen_json(output, authority)
    return {
        "status": "cache_authority_built_without_opening_inputs",
        "authority": file_record(output),
    }


def _validate_cache_authority(path: str | Path, digest: str) -> dict[str, Any]:
    raw, actual, source = load_json_object(
        path,
        expected_sha256=digest,
        label="self-completion cache authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "implementation",
        "interface",
        "interface_contract_sha256",
        "fixed_rule",
        "exact_o0_replay",
        "input_authority",
        "output_cache",
        "output_report",
        "target_score_cache_authorized",
        "target_quality_execution_authorized",
        "access_audit",
    }
    value = dict(raw)
    if (
        set(value) != required
        or value["schema"] != CACHE_AUTHORITY_SCHEMA
        or value["schema_version"] != 2
        or value["status"]
        != "authorized_frozen_O0_anchor_self_completion_cache_only"
        or value["implementation"] != file_record(IMPLEMENTATION)
        or value["interface"] != file_record(INTERFACE)
        or value["interface_contract_sha256"] != formal.CONTRACT_SHA256
        or value["fixed_rule"] != formal.completion_contract()
        or value["exact_o0_replay"]
        != {
            "positive_and_negative_multiscale_caches_bound": True,
            "canonical_negative_logit_scale": 10.0,
            "vala_knn_neighbors": 10,
            "knn_chunk_size": KNN_CHUNK_SIZE,
            "required_proof": (
                "recomputed_exact_o0_query_scores_selected_scales_valid_xyz_"
                "bitwise_equal_bound_reference"
            ),
        }
        or value["target_score_cache_authorized"] is not True
        or value["target_quality_execution_authorized"] is not False
        or value["access_audit"] != _cache_build_access()
        or set(value["input_authority"])
        != {
            "exact_o0_cache",
            "positive_o0_cache",
            "negative_o0_cache",
            "region_features",
            "target_descriptor",
            "factorized_primitive_state",
            "renderer_geometry_checkpoint",
        }
    ):
        raise ValueError("self-completion cache authority differs")
    for name, record in value["input_authority"].items():
        validate_file_record(record, label=name)
    value["verified_record"] = {"path": str(source), "sha256": actual}
    return value


def _load(record: Mapping[str, str], *, label: str) -> dict[str, Any]:
    value, digest, source = load_torch_mapping(
        record["path"],
        expected_sha256=record["sha256"],
        map_location="cpu",
        label=label,
    )
    if {"path": str(source), "sha256": digest} != dict(record):
        raise ValueError(f"{label} record differs")
    return value


def _validate_region_primitive_lineage(
    *,
    scene_id: str,
    features: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    state_record: Mapping[str, str],
    state_xyz: torch.Tensor,
    renderer_record: Mapping[str, str],
    renderer_xyz: torch.Tensor,
    o0: Mapping[str, Any],
    positive: Mapping[str, Any],
    negative: Mapping[str, Any],
) -> None:
    feature_inputs = features.get("input_authority")
    descriptor_inputs = descriptor.get("input_authority")
    if (
        str(features.get("scene_id")) != str(scene_id)
        or str(descriptor.get("scene_id")) != str(scene_id)
        or features.get("region_fingerprints")
        != descriptor.get("region_fingerprints")
        or not isinstance(feature_inputs, Mapping)
        or not isinstance(descriptor_inputs, Mapping)
        or feature_inputs.get("accepted_v2")
        != descriptor_inputs.get("target_accepted_v2")
        or feature_inputs.get("factorized_state")
        != descriptor_inputs.get("factorized_primitive_state")
        or feature_inputs.get("factorized_state") != dict(state_record)
        or o0.get("metadata", {}).get("renderer_geometry_checkpoint")
        != dict(renderer_record)
        or not torch.equal(torch.as_tensor(positive.get("valid")), torch.as_tensor(negative.get("valid")))
        or not torch.equal(torch.as_tensor(positive.get("xyz")), torch.as_tensor(negative.get("xyz")))
        or not torch.equal(torch.as_tensor(positive.get("xyz")), torch.as_tensor(state_xyz))
        or not torch.equal(torch.as_tensor(positive.get("xyz")), torch.as_tensor(renderer_xyz))
    ):
        raise ValueError("self-completion region/primitive lineage differs")


def _score_change_summary(values: torch.Tensor) -> dict[str, Any]:
    scores = torch.as_tensor(values).detach().float().cpu()
    if scores.ndim != 1 or not bool(torch.isfinite(scores).all()):
        raise ValueError("changed O0 score audit axis differs")
    if scores.numel() == 0:
        return {
            "defined": False,
            "count": 0,
            "minimum": None,
            "median": None,
            "maximum": None,
            "fraction_in_closed_0p4_0p6": None,
        }
    return {
        "defined": True,
        "count": int(scores.numel()),
        "minimum": float(scores.min()),
        "median": float(scores.median()),
        "maximum": float(scores.max()),
        "fraction_in_closed_0p4_0p6": float(
            ((scores >= 0.4) & (scores <= 0.6)).float().mean()
        ),
    }


def execute_cache(args: argparse.Namespace) -> dict[str, Any]:
    authority = _validate_cache_authority(
        args.execution_authority, args.expected_execution_authority_sha256
    )
    cache_path = _new_path(authority["output_cache"], label="self-completion cache")
    report_path = _new_path(
        authority["output_report"], label="self-completion report"
    )
    o0 = _load(authority["input_authority"]["exact_o0_cache"], label="exact O0")
    positive = _load(
        authority["input_authority"]["positive_o0_cache"],
        label="positive frozen O0",
    )
    negative = _load(
        authority["input_authority"]["negative_o0_cache"],
        label="negative frozen O0",
    )
    features = _load(
        authority["input_authority"]["region_features"], label="region features"
    )
    descriptor = _load(
        authority["input_authority"]["target_descriptor"],
        label="target descriptor",
    )
    state_record = authority["input_authority"]["factorized_primitive_state"]
    state = load_factorized_primitive_state(
        state_record["path"], expected_sha256=state_record["sha256"]
    )
    renderer_record = authority["input_authority"]["renderer_geometry_checkpoint"]
    renderer_raw, renderer_sha, renderer_path = (
        load_sha_bound_project_checkpoint_mapping(
            renderer_record["path"],
            expected_sha256=renderer_record["sha256"],
            map_location="cpu",
            label="FIX6 renderer geometry checkpoint",
        )
    )
    renderer_xyz = _renderer_checkpoint_xyz(renderer_raw)
    if (
        o0.get("schema")
        != "radio_gs.lerf_o0_anchored_graph_residual_external_scores.v1"
        or o0.get("metadata", {}).get("canonical_capability")
        != "exact_frozen_O0_canonical_negative_VALA_peak_scale"
        or any(len(rows) for rows in o0.get("selection", {}).get("selected_region_rows", ()))
        or not torch.equal(
            torch.as_tensor(features["canonical_region_indices"]),
            torch.as_tensor(descriptor["canonical_region_indices"]),
        )
        or {"path": str(renderer_path), "sha256": renderer_sha}
        != renderer_record
        or tuple(str(value) for value in positive.get("query_ids", ()))
        != tuple(str(value) for value in o0.get("metadata", {}).get("query_names", ()))
        or tuple(str(value) for value in negative.get("query_ids", ()))
        != ("object", "things", "stuff", "texture")
    ):
        raise ValueError("self-completion exact input lineage differs")
    _validate_region_primitive_lineage(
        scene_id=authority["scene_id"],
        features=features,
        descriptor=descriptor,
        state_record=state_record,
        state_xyz=state.xyz,
        renderer_record=renderer_record,
        renderer_xyz=renderer_xyz,
        o0=o0,
        positive=positive,
        negative=negative,
    )
    replay = exact_o0_readout(
        positive_scores=torch.as_tensor(positive["query_scores"])
        .float()
        .cpu()
        .contiguous(),
        negative_scores=torch.as_tensor(negative["query_scores"])
        .float()
        .cpu()
        .contiguous(),
        xyz=torch.as_tensor(positive["xyz"]).float().cpu().contiguous(),
        valid=torch.as_tensor(positive["valid"]).bool().cpu().contiguous(),
        chunk_size=KNN_CHUNK_SIZE,
    )
    if (
        not torch.equal(replay.final_scores, torch.as_tensor(o0["query_scores"]))
        or not torch.equal(
            replay.selected_scale_indices,
            torch.as_tensor(o0["selection"]["selected_scale_indices"]),
        )
        or not torch.equal(torch.as_tensor(positive["valid"]), torch.as_tensor(o0["valid"]))
        or not torch.equal(torch.as_tensor(positive["xyz"]), torch.as_tensor(o0["xyz"]))
    ):
        raise ValueError("bound O0 cache is not bitwise replayed exact O0")
    eligible = torch.as_tensor(descriptor["active_update_mask"]).bool().cpu() & ~torch.as_tensor(
        descriptor["effective_ood_mask"]
    ).bool().cpu()
    result = formal.o0_anchor_self_completion(
        o0_scores=replay.final_scores,
        region_rows=torch.as_tensor(features["region_rows"]).long().cpu().contiguous(),
        core_mask=torch.as_tensor(features["token_mask"]).bool().cpu().contiguous(),
        primitive_valid_mask=torch.as_tensor(positive["valid"]).bool().cpu().contiguous(),
        region_eligible_mask=eligible.contiguous(),
    )
    external = formal.build_external_query_score_cache(
        result=result,
        o0_valid=positive["valid"],
        o0_xyz=positive["xyz"],
        query_names=positive["query_ids"],
        scene_id=authority["scene_id"],
        input_authority=authority["input_authority"],
    )
    write_torch_noclobber(cache_path, external)
    changed_scores = torch.as_tensor(o0["query_scores"])[result.changed_mask]
    o0_membership = replay.final_scores > formal.O0_SCORE_MINIMUM
    final_membership = result.final_scores > formal.O0_SCORE_MINIMUM
    threshold_flips = (~o0_membership) & final_membership
    report = {
        "schema": formal.SCHEMA,
        "schema_version": 2,
        "status": "O0_anchor_self_completion_cache_complete",
        "execution_authority": authority["verified_record"],
        "output_cache": file_record(cache_path),
        "scene_id": authority["scene_id"],
        "query_axis_count": int(result.final_scores.shape[1]),
        "qualified_anchor_count": int(result.qualified_anchor_mask.sum()),
        "qualified_anchor_counts": result.qualified_anchor_mask.sum(dim=0)
        .long()
        .tolist(),
        "strictly_changed_primitive_query_cells": int(result.changed_mask.sum()),
        "threshold_membership_flips": int(threshold_flips.sum()),
        "changed_exact_O0_score": _score_change_summary(changed_scores),
        "bitwise_invariants": {
            "positive_negative_replay_is_bitwise_bound_exact_O0": True,
            "region_fingerprint_axis_is_exact": True,
            "accepted_v2_record_is_exact": True,
            "factorized_state_record_and_xyz_are_exact__validity_authority_is_frozen_O0": True,
            "renderer_record_and_xyz_are_exact": True,
            "pointwise_non_decreasing": bool(
                (result.final_scores >= torch.as_tensor(o0["query_scores"])).all()
            ),
            "invalid_is_exact_O0": torch.equal(
                result.final_scores[~torch.as_tensor(o0["valid"]).bool()],
                torch.as_tensor(o0["query_scores"])[
                    ~torch.as_tensor(o0["valid"]).bool()
                ],
            ),
            "no_anchor_query_is_exact_O0": True,
        },
        "graph_or_relation": "none",
        "cross_query_argmax": False,
        "access_audit": _cache_execute_access(),
    }
    write_frozen_json(report_path, report)
    return {
        "status": report["status"],
        "cache": file_record(cache_path),
        "report": file_record(report_path),
        "qualified_anchor_count": report["qualified_anchor_count"],
        "strictly_changed_primitive_query_cells": report[
            "strictly_changed_primitive_query_cells"
        ],
        "threshold_membership_flips": report["threshold_membership_flips"],
    }


def build_metric_authority(args: argparse.Namespace) -> dict[str, Any]:
    output = _new_path(args.output_authority, label="metric authority")
    cache_authority = _record(
        args.cache_execution_authority,
        args.expected_cache_execution_authority_sha256,
        label="cache execution authority",
    )
    cache_execution = _validate_cache_authority(
        cache_authority["path"], cache_authority["sha256"]
    )
    cache_record = _record(
        args.external_query_score_cache,
        args.expected_external_query_score_cache_sha256,
        label="external query score cache",
    )
    cache = formal.validate_external_query_score_cache(
        _load(cache_record, label="external query score cache")
    )
    report_record = _record(
        args.cache_report,
        args.expected_cache_report_sha256,
        label="cache report",
    )
    report, _, _ = load_json_object(
        report_record["path"],
        expected_sha256=report_record["sha256"],
        label="cache report",
    )
    if (
        cache_execution["output_cache"] != cache_record["path"]
        or report.get("output_cache") != cache_record
        or cache["metadata"]["scene_id"] != cache_execution["scene_id"]
    ):
        raise ValueError("metric/cache binding differs")
    records = {
        "frozen_evaluator": _record(
            args.frozen_evaluator,
            args.expected_frozen_evaluator_sha256,
            label="frozen evaluator",
        ),
        "frozen_summary_head": _record(
            args.frozen_summary_head,
            args.expected_frozen_summary_head_sha256,
            label="frozen summary head",
        ),
        "config": _record(
            args.config, args.expected_config_sha256, label="frozen config"
        ),
        "renderer_geometry_checkpoint": _record(
            args.renderer_geometry_checkpoint,
            args.expected_renderer_geometry_checkpoint_sha256,
            label="renderer geometry checkpoint",
        ),
        "all_query_text_cache": _record(
            args.all_query_text_cache,
            args.expected_all_query_text_cache_sha256,
            label="all-query text cache",
        ),
        "canonical_negative_text_cache": _record(
            args.canonical_negative_text_cache,
            args.expected_canonical_negative_text_cache_sha256,
            label="canonical-negative text cache",
        ),
    }
    cache_renderer = cache["metadata"]["input_authority"].get(
        "renderer_geometry_checkpoint"
    )
    renderer_raw, renderer_sha, renderer_path = (
        load_sha_bound_project_checkpoint_mapping(
            records["renderer_geometry_checkpoint"]["path"],
            expected_sha256=records["renderer_geometry_checkpoint"]["sha256"],
            map_location="cpu",
            label="FIX6 metric renderer geometry checkpoint",
        )
    )
    if (
        cache_renderer != records["renderer_geometry_checkpoint"]
        or cache_execution["input_authority"]["renderer_geometry_checkpoint"]
        != records["renderer_geometry_checkpoint"]
        or {"path": str(renderer_path), "sha256": renderer_sha}
        != records["renderer_geometry_checkpoint"]
        or not torch.equal(
            _renderer_checkpoint_xyz(renderer_raw), torch.as_tensor(cache["xyz"])
        )
    ):
        raise ValueError("metric renderer/cache geometry binding differs")
    authority = {
        "schema": METRIC_AUTHORITY_SCHEMA,
        "schema_version": 2,
        "status": "preregistered_single_FIX6_anchor_self_completion_metric",
        "scene_id": cache["metadata"]["scene_id"],
        "implementation": file_record(IMPLEMENTATION),
        "interface": file_record(INTERFACE),
        "launcher": file_record(LAUNCHER),
        "cache_execution_authority": cache_authority,
        "external_query_score_cache": cache_record,
        "cache_report": report_record,
        **records,
        "label_root": _unopened_path(args.label_root, label="label root"),
        "output_dir": _unopened_path(args.output_dir, label="metric output"),
        "protocol": METRIC_PROTOCOL,
        "single_candidate_no_sweep": True,
        "metric_execution_authorized": True,
        "access_audit": {
            "cache_and_protocol_records_validated": True,
            "label_root_opened": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_labels_opened": False,
            "target_metrics_computed": False,
            "subprocess_started": False,
            "threshold_scan": False,
        },
    }
    write_frozen_json(output, authority)
    return {
        "status": "single_candidate_metric_preregistered_before_GT_access",
        "authority": file_record(output),
    }


def _add_record(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument(
        "--expected-" + name.replace("_", "-") + "-sha256", required=True
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-cache-authority")
    for name in (
        "exact_o0_cache",
        "positive_o0_cache",
        "negative_o0_cache",
        "region_features",
        "target_descriptor",
        "factorized_primitive_state",
        "renderer_geometry_checkpoint",
    ):
        _add_record(build, name)
    build.add_argument("--scene-id", required=True)
    build.add_argument("--output-cache", required=True)
    build.add_argument("--output-report", required=True)
    build.add_argument("--output-authority", required=True)
    build.set_defaults(handler=build_cache_authority)
    execute = commands.add_parser("execute-cache")
    execute.add_argument("--execution-authority", required=True)
    execute.add_argument("--expected-execution-authority-sha256", required=True)
    execute.set_defaults(handler=execute_cache)
    metric = commands.add_parser("build-metric-authority")
    for name in (
        "cache_execution_authority",
        "external_query_score_cache",
        "cache_report",
        "frozen_evaluator",
        "frozen_summary_head",
        "config",
        "renderer_geometry_checkpoint",
        "all_query_text_cache",
        "canonical_negative_text_cache",
    ):
        _add_record(metric, name)
    metric.add_argument("--label-root", required=True)
    metric.add_argument("--output-dir", required=True)
    metric.add_argument("--output-authority", required=True)
    metric.set_defaults(handler=build_metric_authority)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.handler(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "CACHE_AUTHORITY_SCHEMA",
    "METRIC_AUTHORITY_SCHEMA",
    "METRIC_PROTOCOL",
    "build_cache_authority",
    "build_metric_authority",
    "execute_cache",
]
