#!/usr/bin/env python3
"""Seal a target-blind NVOS observation-clamped prediction authority.

This is a source-only boundary.  It binds the completed primitive selector to
its source audit, exact prompt responsibility operator, frozen K16 graph, and
the graph-off unary/source-completion lineage.  It accepts no benchmark,
target-view, target-mask, or metric argument.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch

from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    load_prompt_responsibility_cache,
    tensor_sha256,
)
from radio_gs.querying.observation_clamped_harmonic import (
    _boundary_connected_mask,
)
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
)


ARTIFACT_TYPE = "radio_gs.nvos_observation_clamped_prediction_authority"
SCHEMA_VERSION = 1


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} schema differs")


def _load_selector(args: argparse.Namespace) -> tuple[dict, dict[str, torch.Tensor]]:
    payload, digest, path = load_torch_mapping(
        args.selector,
        expected_sha256=args.selector_sha256,
        label="observation-clamped selector",
    )
    expected = {
        "schema_version",
        "artifact_type",
        "scene_id",
        "method_contract",
        "method_contract_sha256",
        "preregistration_path",
        "preregistration_sha256",
        "numerical_config",
        "base_unary_path",
        "base_unary_sha256",
        "support_graph_path",
        "support_graph_sha256",
        "source_authority",
        "responsibility_authority",
        "tensors",
        "tensor_sha256",
        "tensor_bundle_sha256",
        "target_rgb_opened",
        "target_mask_opened",
        "target_metric_computed",
    }
    _exact_keys(payload, expected, label="selector")
    contract = _mapping(payload["method_contract"], label="selector method contract")
    if (
        payload["schema_version"] != 1
        or payload["artifact_type"]
        != "radio_gs.nvos_observation_clamped_harmonic_selector"
        or payload["scene_id"] != args.scene_id
        or payload["method_contract_sha256"] != canonical_json_sha256(contract)
        or contract.get("method")
        != "observation_clamped_harmonic_extension_v1"
        or contract.get("source_boundary_rewrite") is not False
        or contract.get("unobserved_component_policy")
        != "preserve_field_prior_bitwise"
        or contract.get("uses_target_rgb_mask_or_metric") is not False
        or payload["target_rgb_opened"] is not False
        or payload["target_mask_opened"] is not False
        or payload["target_metric_computed"] is not False
    ):
        raise ValueError("selector scientific authority differs")
    tensors_raw = _mapping(payload["tensors"], label="selector tensors")
    _exact_keys(
        tensors_raw,
        {
            "primitive_probability",
            "source_observation_confidence",
            "source_observed",
            "valid_rows",
        },
        label="selector tensors",
    )
    tensors = {name: torch.as_tensor(value) for name, value in tensors_raw.items()}
    probability = tensors["primitive_probability"]
    confidence = tensors["source_observation_confidence"]
    observed = tensors["source_observed"]
    rows = tensors["valid_rows"]
    if (
        probability.device.type != "cpu"
        or probability.dtype != torch.float32
        or probability.ndim != 1
        or not probability.is_contiguous()
        or not bool(torch.isfinite(probability).all())
        or bool(((probability < 0) | (probability > 1)).any())
        or confidence.device.type != "cpu"
        or confidence.dtype != torch.float32
        or confidence.shape != probability.shape
        or not confidence.is_contiguous()
        or not bool(torch.isfinite(confidence).all())
        or bool(((confidence < 0) | (confidence > 1)).any())
        or observed.device.type != "cpu"
        or observed.dtype != torch.bool
        or observed.shape != probability.shape
        or not torch.equal(observed, confidence > 0)
        or rows.device.type != "cpu"
        or rows.dtype != torch.int64
        or rows.ndim != 1
    ):
        raise ValueError("selector tensors are malformed")
    digests = {name: tensor_sha256(value) for name, value in sorted(tensors.items())}
    if (
        payload["tensor_sha256"] != digests
        or payload["tensor_bundle_sha256"] != canonical_json_sha256(digests)
        or sha256_file(path) != digest
    ):
        raise ValueError("selector tensor or file digest differs")
    return payload, tensors


def _load_source_audit(
    args: argparse.Namespace, *, selector: Mapping[str, object]
) -> tuple[dict, str, Path]:
    audit, digest, path = load_json_object(
        args.source_audit,
        expected_sha256=args.source_audit_sha256,
        label="observation-clamped source audit",
    )
    invariants = _mapping(audit.get("invariants"), label="source audit invariants")
    safety = _mapping(audit.get("safety"), label="source audit safety")
    if (
        audit.get("artifact_type")
        != "nvos_observation_clamped_harmonic_source_audit_v1"
        or audit.get("scene_id") != args.scene_id
        or Path(str(audit.get("output"))).resolve() != Path(args.selector).resolve()
        or audit.get("output_sha256") != args.selector_sha256
        or audit.get("primitive_probability_sha256")
        != selector["tensor_sha256"]["primitive_probability"]
        or audit.get("support_graph_sha256") != selector["support_graph_sha256"]
        or audit.get("method_contract_sha256")
        != selector["method_contract_sha256"]
        or float(invariants.get("observed_primitive_max_absolute_change", -1))
        != 0.0
        or float(invariants.get("source_observed_pixel_max_absolute_change", -1))
        != 0.0
        or int(invariants.get("unknown_changed_rows", 0)) <= 0
        or invariants.get("all_probabilities_in_unit_interval") is not True
        or safety.get("target_rgb_opened") is not False
        or safety.get("target_mask_opened") is not False
        or safety.get("target_metric_computed") is not False
    ):
        raise ValueError("source audit does not authorize prediction")
    return audit, digest, path


def _load_base(
    args: argparse.Namespace,
    *,
    selector: Mapping[str, object],
    tensors: Mapping[str, torch.Tensor],
) -> tuple[dict, torch.Tensor]:
    if (
        Path(str(selector["base_unary_path"])).resolve()
        != Path(args.base_unary).resolve()
        or selector["base_unary_sha256"] != args.base_unary_sha256
    ):
        raise ValueError("base unary CLI and selector lineage differ")
    base, _digest, _path = load_torch_mapping(
        args.base_unary,
        expected_sha256=args.base_unary_sha256,
        label="frozen graph-off base unary",
    )
    if (
        base.get("artifact_type")
        != "nvos_frozen_k16_primitive_unary_probability_v1"
        or base.get("scene_id") != args.scene_id
        or base.get("target_rgb_opened") is not False
        or base.get("target_mask_opened") is not False
        or base.get("written_before_target_ground_truth_open") is not True
    ):
        raise ValueError("base unary is not pre-target authority")
    probability = torch.as_tensor(base.get("primitive_unary_probability"))
    valid = torch.as_tensor(base.get("valid"))
    rows = torch.as_tensor(base.get("valid_rows"))
    if (
        probability.dtype != torch.float32
        or probability.shape != tensors["primitive_probability"].shape
        or valid.dtype != torch.bool
        or valid.shape != probability.shape
        or not torch.equal(rows, tensors["valid_rows"])
        or not torch.equal(rows, torch.nonzero(valid, as_tuple=False).flatten())
    ):
        raise ValueError("base unary row authority differs")
    return base, probability


def _load_graph_and_verify_fail_safe(
    args: argparse.Namespace,
    *,
    selector: Mapping[str, object],
    tensors: Mapping[str, torch.Tensor],
    base_probability: torch.Tensor,
) -> dict[str, object]:
    if (
        Path(str(selector["support_graph_path"])).resolve()
        != Path(args.support_graph).resolve()
        or selector["support_graph_sha256"] != args.support_graph_sha256
    ):
        raise ValueError("K16 CLI and selector lineage differ")
    payload, _digest, _path = load_torch_mapping(
        args.support_graph,
        expected_sha256=args.support_graph_sha256,
        label="frozen K16 support graph",
    )
    rows = tensors["valid_rows"]
    if (
        int(payload.get("num_global_rows", -1)) != int(base_probability.numel())
        or not torch.equal(torch.as_tensor(payload.get("global_rows")), rows)
    ):
        raise ValueError("K16 graph row authority differs")
    graph = PrimitiveSupportGraph(
        edge_index=torch.as_tensor(payload["edge_index"]),
        edge_weight=torch.as_tensor(payload["edge_weight"]).float(),
        raw_affinity=torch.as_tensor(payload["raw_affinity"]).float(),
        local_sigma=payload["local_sigma"],
        num_nodes=int(rows.numel()),
        edge_channels={},
    )
    observed = tensors["source_observed"][rows]
    prior = base_probability[rows]
    completed = tensors["primitive_probability"][rows]
    reachable = _boundary_connected_mask(graph, observed)
    unreachable = ~reachable
    observed_exact = torch.equal(completed[observed], prior[observed])
    unreachable_exact = torch.equal(completed[unreachable], prior[unreachable])
    changed_unknown = (~observed) & (completed != prior)
    if not observed_exact or not unreachable_exact or not bool(changed_unknown.any()):
        raise ValueError("real-graph boundary or no-boundary fail-safe invariant failed")
    return {
        "valid_rows": int(rows.numel()),
        "observed_rows": int(observed.sum()),
        "observed_rows_bitwise_equal": observed_exact,
        "unreachable_no_boundary_rows": int(unreachable.sum()),
        "unreachable_no_boundary_rows_bitwise_equal": unreachable_exact,
        "unknown_changed_rows": int(changed_unknown.sum()),
    }


def _load_exact_w(
    args: argparse.Namespace,
    *,
    selector: Mapping[str, object],
    primitive_count: int,
) -> tuple[PromptResponsibilityAuthority, dict[str, object]]:
    report, report_digest, report_path = load_json_object(
        args.responsibility_report,
        expected_sha256=args.responsibility_report_sha256,
        label="exact-W responsibility report",
    )
    authority = PromptResponsibilityAuthority.from_dict(report["authority"])
    selector_record = _mapping(
        selector["responsibility_authority"],
        label="selector responsibility authority",
    )
    cache_path = Path(str(report["artifact_path"])).resolve()
    if (
        authority.scene_id != args.scene_id
        or authority.num_gaussians != primitive_count
        or authority.digest
        != selector_record.get("responsibility_authority_sha256")
        or report_digest
        != selector_record.get("responsibility_report_sha256")
        or cache_path != Path(args.responsibility_cache).resolve()
        or report.get("file_sha256") != args.responsibility_cache_sha256
        or args.responsibility_cache_sha256
        != selector_record.get("responsibility_cache_sha256")
    ):
        raise ValueError("exact-W responsibility lineage differs")
    cache = load_prompt_responsibility_cache(
        cache_path,
        expected_authority=authority,
        expected_file_sha256=args.responsibility_cache_sha256,
    )
    return authority, {
        "report_path": str(report_path),
        "report_sha256": report_digest,
        "cache_path": str(cache_path),
        "cache_sha256": str(args.responsibility_cache_sha256),
        "tensor_bundle_sha256": str(report["tensor_bundle_sha256"]),
        "authority": authority.to_dict(),
        "authority_sha256": authority.digest,
        "nonzero_responsibility_entries": int(cache.gaussian_ids.numel()),
    }


def seal(args: argparse.Namespace) -> dict[str, object]:
    selector, tensors = _load_selector(args)
    audit, audit_digest, audit_path = _load_source_audit(args, selector=selector)
    base, base_probability = _load_base(
        args, selector=selector, tensors=tensors
    )
    graph_invariants = _load_graph_and_verify_fail_safe(
        args,
        selector=selector,
        tensors=tensors,
        base_probability=base_probability,
    )
    exact_w_authority, exact_w = _load_exact_w(
        args,
        selector=selector,
        primitive_count=int(tensors["primitive_probability"].numel()),
    )
    compiler = _mapping(base["compiler_contract"], label="base compiler contract")
    source_completion = _mapping(
        compiler.get("source_completion_unary"), label="base source completion"
    )
    source_completion_contract = _mapping(
        source_completion.get("contract"), label="base source completion contract"
    )
    source_authority = _mapping(
        selector["source_authority"], label="selector source authority"
    )
    source_files = {
        "base_unary": {
            "path": str(Path(args.base_unary).resolve()),
            "sha256": str(args.base_unary_sha256),
        },
        "source_completion": {
            "path": str(Path(str(source_authority["completion_path"])).resolve()),
            "sha256": str(source_authority["completion_sha256"]),
            "receipt_path": str(
                Path(str(source_authority["completion_receipt_path"])).resolve()
            ),
            "receipt_sha256": str(source_authority["completion_receipt_sha256"]),
        },
        "source_gate": {
            "path": str(Path(str(source_authority["source_gate_path"])).resolve()),
            "sha256": str(source_authority["source_gate_sha256"]),
        },
    }
    for record in source_files.values():
        if sha256_file(record["path"]) != record["sha256"]:
            raise ValueError("pre-GT source-access file digest differs")
        if "receipt_path" in record and (
            sha256_file(record["receipt_path"]) != record["receipt_sha256"]
        ):
            raise ValueError("pre-GT source-access receipt digest differs")
    if (
        source_completion_contract.get("target_rgb_or_mask_used") is not False
        or source_completion.get("target_rgb_opened") is not False
        or source_completion.get("target_mask_opened") is not False
    ):
        raise ValueError("base compiler does not prove pre-GT source-only access")

    authority = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "scene_id": args.scene_id,
        "method": {
            "contract": selector["method_contract"],
            "contract_sha256": selector["method_contract_sha256"],
            "target_readout": "exact_front_to_back_Wu_over_W1",
            "threshold": 0.5,
            "threshold_comparator": "greater_or_equal",
            "connected_selection": False,
            "target_dependent_routing_or_calibration": False,
        },
        "selector": {
            "path": str(Path(args.selector).resolve()),
            "sha256": str(args.selector_sha256),
            "primitive_probability_sha256": selector["tensor_sha256"][
                "primitive_probability"
            ],
            "tensor_bundle_sha256": selector["tensor_bundle_sha256"],
        },
        "source_audit": {
            "path": str(audit_path),
            "sha256": audit_digest,
            "artifact_type": audit["artifact_type"],
        },
        "support_graph": {
            "path": str(Path(args.support_graph).resolve()),
            "sha256": str(args.support_graph_sha256),
            "contract": "frozen_canonical_mpr_v3_shared_support_graph_k16",
        },
        "exact_w": exact_w,
        "geometry_authority": {
            "num_gaussians": exact_w_authority.num_gaussians,
            "geometry_checkpoint_sha256": exact_w_authority.geometry_checkpoint_sha256,
            "geometry_xyz_sha256": exact_w_authority.geometry_xyz_sha256,
            "source_sha256": dict(exact_w_authority.source_sha256 or {}),
        },
        "pre_gt_source_access": {
            "files": source_files,
            "prompt_frame_id": source_authority["prompt_frame_id"],
            "hierarchy_branch": source_authority["hierarchy_branch"],
            "source_only": True,
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_computed": False,
        },
        "real_graph_invariants": graph_invariants,
        "sealed_before_target_score_render": True,
        "sealed_before_target_mask_open": True,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    output = write_frozen_json(args.output, authority)
    output_sha256 = sha256_file(output)
    frozen, digest, path = load_json_object(
        output,
        expected_sha256=output_sha256,
        label="strictly reloaded prediction authority",
    )
    if frozen != authority or digest != output_sha256 or path != output:
        raise RuntimeError("prediction authority changed across strict reload")
    return {
        "scene_id": args.scene_id,
        "prediction_authority": str(output),
        "prediction_authority_sha256": output_sha256,
        "selector_sha256": str(args.selector_sha256),
        "source_audit_sha256": audit_digest,
        "support_graph_sha256": str(args.support_graph_sha256),
        "exact_w_cache_sha256": str(args.responsibility_cache_sha256),
        "real_graph_invariants": graph_invariants,
        "strict_reload_passed": True,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--selector-sha256", required=True)
    parser.add_argument("--source-audit", required=True)
    parser.add_argument("--source-audit-sha256", required=True)
    parser.add_argument("--base-unary", required=True)
    parser.add_argument("--base-unary-sha256", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--support-graph-sha256", required=True)
    parser.add_argument("--responsibility-cache", required=True)
    parser.add_argument("--responsibility-cache-sha256", required=True)
    parser.add_argument("--responsibility-report", required=True)
    parser.add_argument("--responsibility-report-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(seal(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
