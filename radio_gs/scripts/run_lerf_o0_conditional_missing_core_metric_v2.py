#!/usr/bin/env python3
"""Preregister/run one frozen LERF metric for an audited FIX6c v2 cache."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import posixpath
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

from radio_gs.interfaces import lerf_o0_conditional_missing_core_completion_v2 as formal
from radio_gs.scripts import materialize_lerf_o0_conditional_missing_core_completion_v2 as materializer
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
)


SCHEMA = "radio_gs.lerf_o0_conditional_missing_core_frozen_metric.v2"
STATUS = "authorized_single_FIX6c_multisource_v2_scene0003_PASS_metric"
IMPLEMENTATION = Path(__file__).resolve()
PROTOCOL = {
    "protocol_preset": "vala_paper_3d",
    "score_threshold": 0.6,
    "score_postprocess": "none",
    "selection_mode": "score_threshold",
    "projection_mode": "selected_only_alpha",
    "official_frames_only": True,
    "mask_refinement": "none",
    "alpha_binarization": "png_uint8_gt10",
    "silhouette_threshold": 10.0 / 255.0,
    "threshold_scan": False,
}
FROZEN_RECORD_NAMES = (
    "frozen_evaluator",
    "frozen_summary_head",
    "config",
    "renderer_geometry_checkpoint",
    "all_query_text_cache",
    "canonical_negative_text_cache",
)
FROZEN_INPUTS = {
    "frozen_evaluator": {
        "path": "/root/RADIO-GS/radio_gs/scripts/eval_lerf_direct_3d_selection.py",
        "sha256": "8cb39acf08c4f90f6339002ef32022437f67e7cdaae80fc16206b49abbb917d5",
    },
    "frozen_summary_head": {
        "path": "/root/RADIO-GS/checkpoints/siglip2_summary_head.pth",
        "sha256": "41ccc47b2da9b1aed3ee1e80397dc721ec625e083054175c27698e8840b6263c",
    },
    "config": {
        "path": "/root/RADIO-GS/radio_gs/configs/generated/query_consistency/lerf_figurines_radio_verified_pose.yaml",
        "sha256": "a17ada0f1d34cf043f04ddc2f6503c262845d1fed8b4550df8f5d79f2dbd8f11",
    },
    "renderer_geometry_checkpoint": {
        "path": "/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth",
        "sha256": "6900e08d2380d1f1563a2caf01b5846a9ab8df049b2a3c6a3f710452fa96eff2",
    },
    "all_query_text_cache": {
        "path": "/root/RADIO-GS/checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt",
        "sha256": "d0f70797d01cad76e8a12e69c71730fcdfd867e50c3c4b53e3f7bf797e36506d",
    },
    "canonical_negative_text_cache": {
        "path": "/root/RADIO-GS/checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt",
        "sha256": "18d2aac56b50a9670ffe04b397d23a4652dd44fe8f18ed7a309a82b6c1102b67",
    },
}


def _record(path: object, digest: object, *, label: str) -> dict[str, str]:
    raw = str(path)
    canonical = str(Path(raw).expanduser().resolve())
    value = {"path": canonical, "sha256": str(digest)}
    if raw != canonical:
        raise ValueError(f"{label} path must be canonical absolute")
    validate_file_record(value, label=label)
    return value


def _mapping_record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} record differs")
    return _record(value["path"], value["sha256"], label=label)


def _new(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path) or path.exists() or path.is_symlink():
        raise ValueError(f"{label} must be a new canonical absolute path")
    return path


def _unopened(value: object, *, label: str) -> str:
    raw = str(value)
    if not raw.startswith("/") or posixpath.normpath(raw) != raw:
        raise ValueError(f"{label} must be canonical absolute")
    return raw


def _build_access() -> dict[str, bool]:
    return {
        "FIX6c_authority_cache_report_and_O0_lineage_validated": True,
        "multisource_v2_and_scene0003_external_PASS_revalidated": True,
        "frozen_evaluator_inputs_validated": True,
        "label_root_opened": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
        "subprocess_started": False,
        "threshold_scan": False,
    }


def _load_passed_conditional(
    *,
    authority_record: Mapping[str, str],
    cache_record: Mapping[str, str],
    report_record: Mapping[str, str],
    renderer_record: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    authority_raw, authority_digest, authority_path = load_json_object(
        authority_record["path"],
        expected_sha256=authority_record["sha256"],
        label="FIX6c cache execution authority",
    )
    authority = materializer.validate_authority(authority_raw)
    threshold = materializer.validate_source_gate_v2(authority["input_authority"])
    cache_raw, cache_digest, cache_path = load_torch_mapping(
        cache_record["path"],
        expected_sha256=cache_record["sha256"],
        map_location="cpu",
        label="FIX6c external query score cache",
    )
    cache = formal.validate_external_query_score_cache(cache_raw)
    report_raw, report_digest, report_path = load_json_object(
        report_record["path"],
        expected_sha256=report_record["sha256"],
        label="FIX6c no-GT report",
    )
    report = dict(report_raw) if isinstance(report_raw, Mapping) else {}
    gates = report.get("no_GT_safety_gate")
    metadata = cache["metadata"]
    if (
        {"path": str(authority_path), "sha256": authority_digest}
        != dict(authority_record)
        or {"path": str(cache_path), "sha256": cache_digest} != dict(cache_record)
        or {"path": str(report_path), "sha256": report_digest} != dict(report_record)
        or authority.get("outputs")
        != {"cache": cache_record["path"], "report": report_record["path"]}
        or authority.get("target_metric_execution_authorized") is not False
        or report.get("schema") != materializer.REPORT_SCHEMA
        or report.get("schema_version") != 2
        or report.get("status") != "conditional_missing_core_v2_no_GT_gate_passed"
        or report.get("source_external_status")
        != "scene0003_frozen_multiscene_selector_external_gate_passed"
        or report.get("execution_authority") != dict(authority_record)
        or report.get("output_cache") != dict(cache_record)
        or report.get("target_metric_execution_authorized") is not False
        or not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
        or report.get("frozen_threshold_inclusive") != threshold
        or report.get("threshold_source")
        != authority["input_authority"]["multisource_selector_model"]
        or metadata.get("scene_id") != authority.get("scene_id")
        or metadata.get("input_authority") != authority["input_authority"]
        or metadata.get("frozen_threshold_inclusive") != threshold
        or metadata.get("threshold_source")
        != authority["input_authority"]["multisource_selector_model"]
        or metadata.get("candidate_units") != report.get("candidate_units")
        or metadata.get("selected_units") != report.get("selected_units")
        or metadata.get("selected_unique_cells") != report.get("selected_unique_cells")
        or metadata.get("strictly_changed_cells") != report.get("strictly_changed_cells")
        or int(report.get("threshold_membership_flips", -1)) <= 0
    ):
        raise ValueError("FIX6c passed no-GT binding differs")
    o0_record = authority["input_authority"]["exact_o0_cache"]
    o0, _, _ = load_torch_mapping(
        o0_record["path"],
        expected_sha256=o0_record["sha256"],
        map_location="cpu",
        label="FIX6c exact O0 parent",
    )
    o0_metadata = o0.get("metadata", {})
    if (
        o0.get("schema")
        != "radio_gs.lerf_o0_anchored_graph_residual_external_scores.v1"
        or o0_metadata.get("canonical_capability")
        != "exact_frozen_O0_canonical_negative_VALA_peak_scale"
        or o0_metadata.get("renderer_geometry_checkpoint") != dict(renderer_record)
        or list(o0_metadata.get("query_names", ())) != metadata["query_names"]
        or not torch.equal(torch.as_tensor(o0.get("valid")), cache["valid"])
        or not torch.equal(torch.as_tensor(o0.get("xyz")).float(), cache["xyz"])
    ):
        raise ValueError("FIX6c O0/frozen renderer/query lineage differs")
    return authority, cache, report


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    output = _new(args.output_authority, label="FIX6c metric authority")
    authority_record = _record(
        args.cache_execution_authority,
        args.expected_cache_execution_authority_sha256,
        label="FIX6c cache execution authority",
    )
    cache_record = _record(
        args.external_query_score_cache,
        args.expected_external_query_score_cache_sha256,
        label="FIX6c external query cache",
    )
    report_record = _record(
        args.no_gt_report,
        args.expected_no_gt_report_sha256,
        label="FIX6c no-GT report",
    )
    frozen = {
        name: _mapping_record(FROZEN_INPUTS[name], label=f"FIX6c {name}")
        for name in FROZEN_RECORD_NAMES
    }
    authority, _, _ = _load_passed_conditional(
        authority_record=authority_record,
        cache_record=cache_record,
        report_record=report_record,
        renderer_record=frozen["renderer_geometry_checkpoint"],
    )
    if authority["scene_id"] != "figurines":
        raise ValueError("FIX6c metric requires the frozen Figurines cache")
    value = {
        "schema": SCHEMA,
        "schema_version": 2,
        "status": STATUS,
        "scene_id": "figurines",
        "implementation": file_record(IMPLEMENTATION),
        "conditional_materializer": file_record(materializer.IMPLEMENTATION),
        "conditional_interface": file_record(materializer.INTERFACE),
        "cache_execution_authority": authority_record,
        "external_query_score_cache": cache_record,
        "no_GT_report": report_record,
        **frozen,
        "label_root": _unopened(args.label_root, label="FIX6c label root"),
        "output_dir": _unopened(args.output_dir, label="FIX6c metric output"),
        "protocol": PROTOCOL,
        "single_candidate_no_sweep": True,
        "metric_execution_authorized": True,
        "access_audit": _build_access(),
    }
    write_frozen_json(output, value)
    return {
        "status": "single_FIX6c_v2_metric_preregistered_before_GT_access",
        "authority": file_record(output),
    }


def validate_authority(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="FIX6c frozen metric authority",
    )
    value = dict(raw) if isinstance(raw, Mapping) else {}
    required = {
        "schema", "schema_version", "status", "scene_id", "implementation",
        "conditional_materializer", "conditional_interface",
        "cache_execution_authority", "external_query_score_cache", "no_GT_report",
        *FROZEN_RECORD_NAMES, "label_root", "output_dir", "protocol",
        "single_candidate_no_sweep", "metric_execution_authorized", "access_audit",
    }
    if (
        set(value) != required
        or value.get("schema") != SCHEMA
        or value.get("schema_version") != 2
        or value.get("status") != STATUS
        or value.get("scene_id") != "figurines"
        or value.get("implementation") != file_record(IMPLEMENTATION)
        or value.get("conditional_materializer") != file_record(materializer.IMPLEMENTATION)
        or value.get("conditional_interface") != file_record(materializer.INTERFACE)
        or value.get("protocol") != PROTOCOL
        or value.get("single_candidate_no_sweep") is not True
        or value.get("metric_execution_authorized") is not True
        or value.get("access_audit") != _build_access()
    ):
        raise ValueError("FIX6c metric authority header differs")
    frozen = {}
    for name in FROZEN_RECORD_NAMES:
        frozen[name] = _mapping_record(value[name], label=f"FIX6c metric {name}")
        if frozen[name] != FROZEN_INPUTS[name]:
            raise ValueError("FIX6c frozen metric input differs")
    authority_record = _mapping_record(
        value["cache_execution_authority"], label="FIX6c metric cache authority"
    )
    cache_record = _mapping_record(value["external_query_score_cache"], label="FIX6c metric cache")
    report_record = _mapping_record(value["no_GT_report"], label="FIX6c metric report")
    authority, _, _ = _load_passed_conditional(
        authority_record=authority_record,
        cache_record=cache_record,
        report_record=report_record,
        renderer_record=frozen["renderer_geometry_checkpoint"],
    )
    if authority["scene_id"] != value.get("scene_id"):
        raise ValueError("FIX6c metric/cache scene differs")
    value["verified_record"] = {"path": str(source), "sha256": digest}
    return value


def build_command(authority: Mapping[str, Any], *, gpu: int) -> list[str]:
    if isinstance(gpu, bool) or int(gpu) < 0:
        raise ValueError("gpu must be non-negative")
    return [
        sys.executable,
        authority["frozen_evaluator"]["path"],
        "--config", authority["config"]["path"],
        "--checkpoint", authority["renderer_geometry_checkpoint"]["path"],
        "--scene", authority["scene_id"],
        "--protocol_preset", "vala_paper_3d",
        "--label_dir", authority["label_root"],
        "--output_dir", authority["output_dir"],
        "--summary_head_weights", authority["frozen_summary_head"]["path"],
        "--text_embedding_cache", authority["all_query_text_cache"]["path"],
        "--canonical_embedding_cache", authority["canonical_negative_text_cache"]["path"],
        "--external_query_score_cache", authority["external_query_score_cache"]["path"],
        "--gpu", str(int(gpu)),
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    authority = validate_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    command = build_command(authority, gpu=args.gpu)
    if not args.execute:
        return {
            "status": "FIX6c_v2_metric_dry_run",
            "execution_authority": authority["verified_record"],
            "command": command,
        }
    output = Path(authority["output_dir"])
    if output.exists() or output.is_symlink():
        raise FileExistsError("FIX6c metric output must be new")
    subprocess.run(command, check=True)
    result = output / authority["scene_id"] / "lerf_direct_3d_selection_results.json"
    if not result.is_file() or result.is_symlink():
        raise RuntimeError("FIX6c metric result is unavailable")
    return {
        "status": "FIX6c_v2_metric_complete",
        "execution_authority": authority["verified_record"],
        "result": file_record(result),
    }


def _add_record(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--expected-" + name.replace("_", "-") + "-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-authority")
    for name in (
        "cache_execution_authority", "external_query_score_cache", "no_gt_report",
    ):
        _add_record(build, name)
    build.add_argument("--label-root", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--output-authority", required=True)
    build.set_defaults(handler=build_authority)
    execute = commands.add_parser("run")
    execute.add_argument("--execution-authority", required=True)
    execute.add_argument("--expected-execution-authority-sha256", required=True)
    execute.add_argument("--gpu", type=int, default=0)
    execute.add_argument("--execute", action="store_true")
    execute.set_defaults(handler=run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.handler(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "FROZEN_INPUTS", "FROZEN_RECORD_NAMES", "PROTOCOL", "SCHEMA", "STATUS", "build_authority",
    "build_command", "run", "validate_authority",
]
