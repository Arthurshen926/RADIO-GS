#!/usr/bin/env python3
"""Materialize a target-blind NVOS Dirichlet graph-completion selector.

The input primitive unary must already be sealed by the frozen graph-off
source-completion compiler.  This entrypoint reconstructs its source-view
positive/negative/unknown confidence through the exact prompt responsibility
adjoint.  Non-zero-confidence rows become immutable graph boundaries; only
unknown rows are harmonically completed.  No target path is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch

from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    load_prompt_responsibility_cache,
    tensor_sha256,
)
from radio_gs.querying.nvos_local_positive_completion import (
    local_majority_positive_evidence,
)
from radio_gs.querying.nvos_source_completion_calibration import (
    load_source_completion_loo_gate,
)
from radio_gs.querying.observation_clamped_harmonic import (
    ObservationClampedHarmonicConfig,
    method_contract,
    solve_observation_clamped_harmonic,
)
from radio_gs.querying.sam3_reference_completion import (
    probability_preserving_entropy_observation,
)
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


ARTIFACT_TYPE = "radio_gs.nvos_observation_clamped_harmonic_selector"
SCHEMA_VERSION = 1


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sha256(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _load_base_unary(args: argparse.Namespace) -> tuple[dict, torch.Tensor, torch.Tensor]:
    payload, _digest, _path = load_torch_mapping(
        args.base_unary,
        expected_sha256=args.base_unary_sha256,
        label="frozen graph-off NVOS primitive unary",
    )
    expected_keys = {
        "artifact_type",
        "schema_version",
        "scene_id",
        "protocol_hash",
        "capability_cache",
        "capability_cache_sha256",
        "capability_source_contract",
        "compiler_contract",
        "valid",
        "valid_rows",
        "primitive_unary_probability",
        "target_rgb_opened",
        "target_mask_opened",
        "written_before_target_ground_truth_open",
    }
    if (
        set(payload) != expected_keys
        or payload["artifact_type"]
        != "nvos_frozen_k16_primitive_unary_probability_v1"
        or payload["schema_version"] != 1
        or payload["scene_id"] != args.scene_id
        or payload["target_rgb_opened"] is not False
        or payload["target_mask_opened"] is not False
        or payload["written_before_target_ground_truth_open"] is not True
    ):
        raise ValueError("base graph-off unary authority differs")
    compiler = _mapping(payload["compiler_contract"], label="base compiler contract")
    if (
        compiler.get("graph_disabled") is not True
        or compiler.get("connected_selection_applied") is not False
        or compiler.get("readout") != "unary_prior"
        or compiler.get("registered_observation_fusion") != "probability_mixture"
    ):
        raise ValueError("base unary is not the frozen graph-off probability mixture")
    probability = torch.as_tensor(payload["primitive_unary_probability"])
    valid = torch.as_tensor(payload["valid"])
    rows = torch.as_tensor(payload["valid_rows"])
    if (
        probability.device.type != "cpu"
        or probability.dtype != torch.float32
        or probability.ndim != 1
        or not probability.is_contiguous()
        or not bool(torch.isfinite(probability).all())
        or bool(((probability < 0) | (probability > 1)).any())
        or valid.device.type != "cpu"
        or valid.dtype != torch.bool
        or valid.shape != probability.shape
        or rows.device.type != "cpu"
        or rows.dtype != torch.int64
        or rows.ndim != 1
        or not torch.equal(rows, torch.nonzero(valid, as_tuple=False).flatten())
    ):
        raise ValueError("base unary tensors are malformed")
    return payload, probability, rows


def _source_posterior(
    base: Mapping[str, object],
    *,
    scene_id: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    compiler = _mapping(base["compiler_contract"], label="base compiler contract")
    source = _mapping(
        compiler.get("source_completion_unary"), label="source completion contract"
    )
    completion_path = Path(str(source.get("completion_path"))).resolve()
    completion_sha256 = _sha256(
        source.get("completion_sha256"), label="completion SHA-256"
    )
    receipt_path = Path(str(source.get("receipt_path"))).resolve()
    receipt_sha256 = _sha256(
        source.get("receipt_sha256"), label="completion receipt SHA-256"
    )
    if sha256_file(receipt_path) != receipt_sha256:
        raise ValueError("source completion receipt SHA-256 differs")
    completion, _digest, _path = load_torch_mapping(
        completion_path,
        expected_sha256=completion_sha256,
        label="frozen source completion",
    )
    authority = _mapping(completion.get("authority"), label="completion authority")
    tensors = _mapping(completion.get("tensors"), label="completion tensors")
    if (
        completion.get("artifact_type")
        != "radio_gs.nvos_sam3_reference_completion"
        or completion.get("schema_version") != 1
        or authority.get("scene_id") != scene_id
        or authority.get("target_rgb_opened") is not False
        or authority.get("target_mask_opened") is not False
        or set(tensors)
        != {
            "trial_masks",
            "aggregate_probability",
            "completed_positive",
            "raw_positive",
            "raw_negative",
            "point_coordinates_xy",
            "quality",
        }
    ):
        raise ValueError("source completion payload differs")

    gate_record = _mapping(
        source.get("source_only_calibration_gate"),
        label="source completion calibration gate record",
    )
    gate = load_source_completion_loo_gate(
        gate_record.get("path"),
        expected_gate_sha256=_sha256(
            gate_record.get("sha256"), label="source gate SHA-256"
        ),
        completion_path=completion_path,
        expected_completion_sha256=completion_sha256,
        expected_completion_receipt_sha256=receipt_sha256,
        expected_scene_id=scene_id,
        expected_frame_id=str(authority.get("frame_id")),
    )
    raw_positive = torch.as_tensor(tensors["raw_positive"]).bool().contiguous()
    raw_negative = torch.as_tensor(tensors["raw_negative"]).bool().contiguous()
    aggregate = torch.as_tensor(tensors["aggregate_probability"]).float().contiguous()
    probability_np, confidence_np = probability_preserving_entropy_observation(
        aggregate.numpy(), raw_positive.numpy(), raw_negative.numpy()
    )
    accept_full = bool(gate["decision"]["accept_source_completion"])
    if accept_full:
        probability = torch.from_numpy(probability_np).contiguous()
        confidence = torch.from_numpy(confidence_np).contiguous()
        expected_branch = "v1_full_probability_preserving_completion"
    else:
        probability, confidence = local_majority_positive_evidence(
            torch.from_numpy(probability_np),
            positive_scribble=raw_positive,
            negative_scribble=raw_negative,
        )
        expected_branch = "v2_local_majority_positive_completion"
    hierarchy = _mapping(source.get("hierarchical_trust"), label="hierarchy branch")
    if hierarchy.get("branch") != expected_branch:
        raise ValueError("sealed base unary hierarchy branch differs from source replay")
    return probability, confidence, {
        "completion_path": str(completion_path),
        "completion_sha256": completion_sha256,
        "completion_receipt_path": str(receipt_path),
        "completion_receipt_sha256": receipt_sha256,
        "source_gate_path": str(Path(str(gate_record["path"])).resolve()),
        "source_gate_sha256": str(gate_record["sha256"]),
        "hierarchy_branch": expected_branch,
        "prompt_frame_id": str(authority["frame_id"]),
    }


def _load_graph(
    args: argparse.Namespace,
    *,
    base: Mapping[str, object],
    valid_rows: torch.Tensor,
) -> PrimitiveSupportGraph:
    payload, _digest, _path = load_torch_mapping(
        args.support_graph,
        expected_sha256=args.support_graph_sha256,
        label="frozen canonical support graph",
    )
    required = {
        "global_rows",
        "num_global_rows",
        "edge_index",
        "edge_weight",
        "raw_affinity",
        "local_sigma",
        "metadata",
    }
    if not required.issubset(payload):
        raise ValueError("support graph schema differs")
    graph_rows = torch.as_tensor(payload["global_rows"]).long().cpu()
    metadata = _mapping(payload["metadata"], label="support graph metadata")
    capability_metadata = _mapping(
        metadata.get("capability_metadata"), label="graph capability metadata"
    )
    if (
        int(payload["num_global_rows"])
        != int(torch.as_tensor(base["valid"]).numel())
        or not torch.equal(graph_rows, valid_rows)
        or Path(str(metadata.get("capability_cache"))).resolve()
        != Path(str(base["capability_cache"])).resolve()
        or sha256_file(Path(str(base["capability_cache"])).resolve())
        != str(base["capability_cache_sha256"])
        or not isinstance(capability_metadata.get("field_checkpoint_sha256"), str)
    ):
        raise ValueError("support graph and base-unary row authority differ")
    return PrimitiveSupportGraph(
        edge_index=payload["edge_index"],
        edge_weight=torch.as_tensor(payload["edge_weight"]).float(),
        raw_affinity=torch.as_tensor(payload["raw_affinity"]).float(),
        local_sigma=payload["local_sigma"],
        num_nodes=int(graph_rows.numel()),
        edge_channels={},
    )


def _exact_primitive_observation_confidence(
    args: argparse.Namespace,
    *,
    probability: torch.Tensor,
    confidence: torch.Tensor,
    scene_id: str,
) -> tuple[torch.Tensor, object, PromptResponsibilityAuthority, dict[str, object]]:
    report, report_sha256, report_path = load_json_object(
        args.responsibility_report,
        expected_sha256=args.responsibility_report_sha256,
        label="prompt responsibility report",
    )
    authority = PromptResponsibilityAuthority.from_dict(report["authority"])
    if authority.scene_id != scene_id or tuple(probability.shape) != (
        authority.height,
        authority.width,
    ):
        raise ValueError("source posterior and responsibility authority differ")
    cache_path = Path(str(report["artifact_path"])).resolve()
    if cache_path != Path(args.responsibility_cache).resolve():
        raise ValueError("responsibility cache path differs from report")
    cache = load_prompt_responsibility_cache(
        cache_path,
        expected_authority=authority,
        expected_file_sha256=args.responsibility_cache_sha256,
    )
    if str(report["file_sha256"]) != str(args.responsibility_cache_sha256):
        raise ValueError("responsibility report file digest differs")
    foreground = cache.adjoint(confidence * probability).weighted_sum
    background = cache.adjoint(confidence * (1.0 - probability)).weighted_sum
    labeled = foreground + background
    primitive_confidence = torch.zeros_like(labeled)
    visible = cache.visible_mass > 0
    primitive_confidence[visible] = labeled[visible] / cache.visible_mass[visible]
    primitive_confidence.clamp_(0.0, 1.0)
    return primitive_confidence.float().contiguous(), cache, authority, {
        "responsibility_report_path": str(report_path),
        "responsibility_report_sha256": report_sha256,
        "responsibility_cache_path": str(cache_path),
        "responsibility_cache_sha256": str(args.responsibility_cache_sha256),
        "responsibility_authority_sha256": authority.digest,
    }


def materialize(args: argparse.Namespace) -> dict[str, object]:
    preregistration, preregistration_sha256, preregistration_path = load_json_object(
        args.preregistration,
        expected_sha256=args.preregistration_sha256,
        label="observation-clamped source-audit preregistration",
    )
    if (
        preregistration.get("artifact_type")
        != "nvos_observation_clamped_harmonic_source_audit_preregistration_v1"
        or preregistration.get("status")
        != "registered_before_source_audit_materialization_or_new_target_prediction"
        or preregistration.get("fixed_method", {}).get("boundary")
        != "source_confidence > 0, with no numeric confidence threshold"
    ):
        raise ValueError("source-audit preregistration contract differs")
    base, base_probability, valid_rows = _load_base_unary(args)
    source_probability, source_confidence, source_authority = _source_posterior(
        base, scene_id=args.scene_id
    )
    primitive_confidence, cache, authority, responsibility = (
        _exact_primitive_observation_confidence(
            args,
            probability=source_probability,
            confidence=source_confidence,
            scene_id=args.scene_id,
        )
    )
    if primitive_confidence.shape != base_probability.shape:
        raise ValueError("source observation and base unary row counts differ")
    graph = _load_graph(
        args, base=base, valid_rows=valid_rows
    )
    config = ObservationClampedHarmonicConfig(
        cg_iterations=int(args.cg_iterations),
        cg_tolerance=float(args.cg_tolerance),
        hard_seed_threshold=float(
            _mapping(base["compiler_contract"], label="base compiler").get(
                "hard_seed_threshold"
            )
        ),
        hard_seed_conflict_policy="exclusive_relative",
        hard_seed_conflict_margin=0.0,
    )
    base_valid = base_probability[valid_rows].contiguous()
    confidence_valid = primitive_confidence[valid_rows].contiguous()
    completed_valid = solve_observation_clamped_harmonic(
        graph,
        base_valid,
        confidence_valid,
        config=config,
    ).float().contiguous()
    observed_valid = confidence_valid > 0
    if not torch.equal(completed_valid[observed_valid], base_valid[observed_valid]):
        raise RuntimeError("graph rewrote source-observed primitive unary")

    completed = base_probability.clone()
    completed[valid_rows] = completed_valid
    base_cycle = cache.forward(base_probability.double())
    completed_cycle = cache.forward(completed.double())
    observed_pixels = source_confidence > 0
    if bool(observed_pixels.any()):
        source_pixel_max_change = float(
            (
                base_cycle.normalized_probability[observed_pixels]
                - completed_cycle.normalized_probability[observed_pixels]
            )
            .abs()
            .max()
        )
    else:
        source_pixel_max_change = 0.0

    contract = method_contract()
    contract_sha256 = canonical_json_sha256(contract)
    tensors = {
        "primitive_probability": completed.contiguous(),
        "source_observation_confidence": primitive_confidence.contiguous(),
        "source_observed": (primitive_confidence > 0).contiguous(),
        "valid_rows": valid_rows.contiguous(),
    }
    tensor_digests = {
        name: tensor_sha256(value) for name, value in sorted(tensors.items())
    }
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "scene_id": args.scene_id,
        "method_contract": contract,
        "method_contract_sha256": contract_sha256,
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "numerical_config": {
            "cg_iterations": config.cg_iterations,
            "cg_tolerance": config.cg_tolerance,
            "hard_seed_threshold": config.hard_seed_threshold,
            "hard_seed_conflict_policy": config.hard_seed_conflict_policy,
            "hard_seed_conflict_margin": config.hard_seed_conflict_margin,
        },
        "base_unary_path": str(Path(args.base_unary).resolve()),
        "base_unary_sha256": str(args.base_unary_sha256),
        "support_graph_path": str(Path(args.support_graph).resolve()),
        "support_graph_sha256": str(args.support_graph_sha256),
        "source_authority": source_authority,
        "responsibility_authority": responsibility,
        "tensors": tensors,
        "tensor_sha256": tensor_digests,
        "tensor_bundle_sha256": canonical_json_sha256(tensor_digests),
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    output_path = write_torch_noclobber(args.output, artifact)
    output_sha256 = sha256_file(output_path)
    frozen, _digest, _path = load_torch_mapping(
        output_path,
        expected_sha256=output_sha256,
        label="frozen observation-clamped selector",
    )
    if frozen.get("tensor_sha256") != tensor_digests:
        raise RuntimeError("selector changed across freeze and reload")

    changed_unknown = (~observed_valid) & (completed_valid != base_valid)
    report = {
        "schema_version": 1,
        "artifact_type": "nvos_observation_clamped_harmonic_source_audit_v1",
        "scene_id": args.scene_id,
        "method_contract": contract,
        "method_contract_sha256": contract_sha256,
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "output": str(output_path),
        "output_sha256": output_sha256,
        "primitive_probability_sha256": tensor_digests["primitive_probability"],
        "source_observation_confidence_sha256": tensor_digests[
            "source_observation_confidence"
        ],
        "support_graph_sha256": str(args.support_graph_sha256),
        "source_authority": source_authority,
        "responsibility_authority": responsibility,
        "invariants": {
            "valid_rows": int(valid_rows.numel()),
            "source_observed_valid_rows": int(observed_valid.sum()),
            "source_observed_valid_fraction": float(observed_valid.double().mean()),
            "observed_primitive_max_absolute_change": float(
                (completed_valid[observed_valid] - base_valid[observed_valid])
                .abs()
                .max()
            )
            if bool(observed_valid.any())
            else 0.0,
            "unknown_changed_rows": int(changed_unknown.sum()),
            "unknown_mean_absolute_change": float(
                (completed_valid[~observed_valid] - base_valid[~observed_valid])
                .abs()
                .mean()
            )
            if bool((~observed_valid).any())
            else 0.0,
            "output_minimum": float(completed_valid.min()),
            "output_maximum": float(completed_valid.max()),
            "source_observed_pixel_max_absolute_change": source_pixel_max_change,
            "all_probabilities_in_unit_interval": bool(
                ((completed_valid >= 0) & (completed_valid <= 1)).all()
            ),
        },
        "safety": {
            "source_rgb_opened": False,
            "source_completion_opened": True,
            "source_scribbles_opened_from_completion_tensor_only": True,
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_computed": False,
        },
    }
    report_path = write_frozen_json(args.report, report)
    return {**report, "report": str(report_path), "report_sha256": sha256_file(report_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--base-unary", required=True)
    parser.add_argument("--base-unary-sha256", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--support-graph-sha256", required=True)
    parser.add_argument("--responsibility-cache", required=True)
    parser.add_argument("--responsibility-cache-sha256", required=True)
    parser.add_argument("--responsibility-report", required=True)
    parser.add_argument("--responsibility-report-sha256", required=True)
    parser.add_argument("--cg-iterations", type=int, default=64)
    parser.add_argument("--cg-tolerance", type=float, default=1e-5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(materialize(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
