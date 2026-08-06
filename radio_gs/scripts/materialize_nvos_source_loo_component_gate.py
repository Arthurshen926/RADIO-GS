#!/usr/bin/env python3
"""Materialize a target-blind source-LOO gated NVOS graph completion.

The sealed graph-off unary is always the default.  The observation-clamped
candidate is copied only on confidence-zero connected components whose source
boundary ring passes the fixed four-fold proper-scoring gate.  No target path
or benchmark argument is accepted.
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
    ObservationClampedHarmonicConfig,
)
from radio_gs.querying.source_loo_component_gate import (
    boundary_ring_pairs,
    component_brier_records,
    deterministic_folds,
    method_contract,
    recover_graph_off_field_prior,
    source_loo_predictions,
    unknown_component_labels,
)
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.scripts.materialize_nvos_observation_clamped_harmonic import (
    _source_posterior,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


ARTIFACT_TYPE = "radio_gs.nvos_source_loo_component_gated_selector"


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _load_authority(args: argparse.Namespace) -> tuple[dict, str, Path]:
    authority, digest, path = load_json_object(
        args.prediction_authority,
        expected_sha256=args.prediction_authority_sha256,
        label="sealed observation-clamped prediction authority",
    )
    if (
        authority.get("artifact_type")
        != "radio_gs.nvos_observation_clamped_prediction_authority"
        or authority.get("scene_id") != args.scene_id
        or authority.get("sealed_before_target_score_render") is not True
        or authority.get("sealed_before_target_mask_open") is not True
        or authority.get("target_rgb_opened") is not False
        or authority.get("target_mask_opened") is not False
        or authority.get("target_metric_computed") is not False
    ):
        raise ValueError("prediction authority differs")
    return authority, digest, path


def _load_inputs(
    args: argparse.Namespace, authority: Mapping[str, object]
) -> tuple[dict, dict, dict, PrimitiveSupportGraph, PromptResponsibilityAuthority, object]:
    selector_record = _mapping(authority["selector"], label="selector record")
    selector, _selector_digest, _selector_path = load_torch_mapping(
        selector_record["path"],
        expected_sha256=str(selector_record["sha256"]),
        label="sealed observation-clamped selector",
    )
    source_access = _mapping(
        authority["pre_gt_source_access"], label="pre-GT source access"
    )
    source_files = _mapping(source_access["files"], label="source files")
    base_record = _mapping(source_files["base_unary"], label="base unary record")
    base, _base_digest, _base_path = load_torch_mapping(
        base_record["path"],
        expected_sha256=str(base_record["sha256"]),
        label="sealed graph-off unary",
    )
    graph_record = _mapping(authority["support_graph"], label="support graph record")
    graph_payload, _graph_digest, _graph_path = load_torch_mapping(
        graph_record["path"],
        expected_sha256=str(graph_record["sha256"]),
        label="sealed K16 support graph",
    )
    tensors = _mapping(selector["tensors"], label="selector tensors")
    valid_rows = torch.as_tensor(tensors["valid_rows"]).long().cpu()
    if not torch.equal(torch.as_tensor(graph_payload["global_rows"]), valid_rows):
        raise ValueError("selector and support graph rows differ")
    graph = PrimitiveSupportGraph(
        edge_index=graph_payload["edge_index"],
        edge_weight=torch.as_tensor(graph_payload["edge_weight"]).float(),
        raw_affinity=torch.as_tensor(graph_payload["raw_affinity"]).float(),
        local_sigma=graph_payload["local_sigma"],
        num_nodes=int(valid_rows.numel()),
        edge_channels={},
    )
    exact_w = _mapping(authority["exact_w"], label="exact-W record")
    report, _report_digest, _report_path = load_json_object(
        exact_w["report_path"],
        expected_sha256=str(exact_w["report_sha256"]),
        label="sealed exact-W report",
    )
    responsibility = PromptResponsibilityAuthority.from_dict(report["authority"])
    cache = load_prompt_responsibility_cache(
        exact_w["cache_path"],
        expected_authority=responsibility,
        expected_file_sha256=str(exact_w["cache_sha256"]),
    )
    if (
        responsibility.scene_id != args.scene_id
        or responsibility.target_rgb_opened is not False
        or responsibility.target_mask_opened is not False
        or selector.get("target_rgb_opened") is not False
        or selector.get("target_mask_opened") is not False
        or selector.get("target_metric_computed") is not False
        or base.get("target_rgb_opened") is not False
        or base.get("target_mask_opened") is not False
        or base.get("written_before_target_ground_truth_open") is not True
    ):
        raise ValueError("source-only input flags differ")
    return selector, base, graph_payload, graph, responsibility, cache


def _source_primitive_observation(
    *,
    base: Mapping[str, object],
    cache: object,
    scene_id: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_probability, pixel_confidence, _source_authority = _source_posterior(
        base, scene_id=scene_id
    )
    foreground = cache.adjoint(pixel_confidence * pixel_probability).weighted_sum
    background = cache.adjoint(
        pixel_confidence * (1.0 - pixel_probability)
    ).weighted_sum
    labeled = foreground + background
    confidence = torch.zeros_like(labeled)
    visible = cache.visible_mass > 0
    confidence[visible] = labeled[visible] / cache.visible_mass[visible]
    confidence.clamp_(0.0, 1.0)
    probability = torch.full_like(labeled, 0.5)
    observed = labeled > 0
    probability[observed] = foreground[observed] / labeled[observed]
    probability.clamp_(0.0, 1.0)
    return probability.float().contiguous(), confidence.float().contiguous()


def _aggregate(records: list[dict[str, object]], accepted: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    total_weight = 0.0
    candidate_error = 0.0
    field_error = 0.0
    accepted_weight = 0.0
    accepted_candidate_error = 0.0
    accepted_field_error = 0.0
    positive_validation_components = 0
    rejected_positive_validation_components = 0
    for record in records:
        weight = float(record["validation_weight"])
        if weight > 0:
            positive_validation_components += 1
            candidate_error += weight * float(record["candidate_brier"])
            field_error += weight * float(record["field_brier"])
            total_weight += weight
            if bool(record["accepted"]):
                accepted_candidate_error += weight * float(record["candidate_brier"])
                accepted_field_error += weight * float(record["field_brier"])
                accepted_weight += weight
            else:
                rejected_positive_validation_components += 1
    unknown = labels >= 0
    accepted_rows = torch.zeros_like(unknown)
    accepted_rows[unknown] = accepted[labels[unknown]]
    component_sizes = torch.bincount(labels[unknown], minlength=int(accepted.numel()))
    return {
        "component_count": int(accepted.numel()),
        "accepted_component_count": int(accepted.sum()),
        "positive_validation_component_count": positive_validation_components,
        "rejected_positive_validation_component_count": (
            rejected_positive_validation_components
        ),
        "accepted_component_fraction": float(accepted.double().mean())
        if accepted.numel()
        else 0.0,
        "unknown_rows": int(unknown.sum()),
        "accepted_unknown_rows": int(accepted_rows.sum()),
        "accepted_unknown_row_fraction": float(
            accepted_rows.sum().double() / unknown.sum().clamp_min(1)
        ),
        "largest_component_rows": int(component_sizes.max())
        if component_sizes.numel()
        else 0,
        "all_component_ring_validation_weight": total_weight,
        "all_component_ring_candidate_brier": candidate_error / total_weight
        if total_weight > 0
        else None,
        "all_component_ring_field_brier": field_error / total_weight
        if total_weight > 0
        else None,
        "accepted_component_ring_validation_weight": accepted_weight,
        "accepted_component_ring_candidate_brier": (
            accepted_candidate_error / accepted_weight
            if accepted_weight > 0
            else None
        ),
        "accepted_component_ring_field_brier": (
            accepted_field_error / accepted_weight if accepted_weight > 0 else None
        ),
    }


def materialize(args: argparse.Namespace) -> dict[str, object]:
    prereg, prereg_sha256, prereg_path = load_json_object(
        args.preregistration,
        expected_sha256=args.preregistration_sha256,
        label="source-LOO component-gate preregistration",
    )
    contract = method_contract()
    contract_sha256 = canonical_json_sha256(contract)
    if (
        prereg.get("artifact_type")
        != "nvos_source_loo_component_gate_preregistration_v1"
        or prereg.get("status")
        != "registered_before_source_loo_materialization_or_gated_target_prediction"
        or prereg.get("method_contract") != contract
        or prereg.get("method_contract_sha256") != contract_sha256
    ):
        raise ValueError("source-LOO preregistration differs")
    authority, authority_sha256, authority_path = _load_authority(args)
    selector, base, _graph_payload, graph, _responsibility, cache = _load_inputs(
        args, authority
    )
    tensors = _mapping(selector["tensors"], label="selector tensors")
    valid_rows = torch.as_tensor(tensors["valid_rows"]).long().cpu()
    base_full = torch.as_tensor(base["primitive_unary_probability"]).float().cpu()
    candidate_full = torch.as_tensor(tensors["primitive_probability"]).float().cpu()
    sealed_confidence = torch.as_tensor(
        tensors["source_observation_confidence"]
    ).float().cpu()
    source_probability_full, source_confidence_full = _source_primitive_observation(
        base=base, cache=cache, scene_id=args.scene_id
    )
    if (
        not torch.equal(source_confidence_full, sealed_confidence)
        or source_probability_full.shape != base_full.shape
    ):
        raise ValueError("replayed exact source observation differs from selector")
    fused = base_full[valid_rows].contiguous()
    candidate = candidate_full[valid_rows].contiguous()
    source_probability = source_probability_full[valid_rows].contiguous()
    source_confidence = source_confidence_full[valid_rows].contiguous()
    field, validation_eligible = recover_graph_off_field_prior(
        fused, source_probability, source_confidence
    )
    # Confidence-zero rows are exactly the graph-off field prior by contract.
    unknown = source_confidence == 0
    if not torch.equal(field[unknown], fused[unknown]):
        raise RuntimeError("unknown graph-off field recovery changed the base prior")
    labels, component_count = unknown_component_labels(graph, ~unknown)
    rings = boundary_ring_pairs(graph, ~unknown, labels)
    folds = deterministic_folds(valid_rows, scene_id=args.scene_id)
    numerical = _mapping(selector["numerical_config"], label="selector numerical config")
    config = ObservationClampedHarmonicConfig(
        cg_iterations=int(numerical["cg_iterations"]),
        cg_tolerance=float(numerical["cg_tolerance"]),
        hard_seed_threshold=float(numerical["hard_seed_threshold"]),
        hard_seed_conflict_policy=str(numerical["hard_seed_conflict_policy"]),
        hard_seed_conflict_margin=float(numerical["hard_seed_conflict_margin"]),
    )
    predictions = source_loo_predictions(
        graph,
        fused_probability=fused,
        field_probability=field,
        source_confidence=source_confidence,
        validation_eligible=validation_eligible,
        fold_assignment=folds,
        config=config,
    )
    records, accepted = component_brier_records(
        component_count=component_count,
        rings=rings,
        predictions=predictions,
        field_probability=field,
        source_probability=source_probability,
        source_confidence=source_confidence,
        validation_eligible=validation_eligible,
        fold_assignment=folds,
    )
    accepted_rows = torch.zeros_like(unknown)
    accepted_rows[unknown] = accepted[labels[unknown]]
    gated_full = base_full.clone()
    gated_full[valid_rows[accepted_rows]] = candidate[accepted_rows]
    if (
        not torch.equal(gated_full[sealed_confidence > 0], base_full[sealed_confidence > 0])
        or not torch.equal(
            gated_full[valid_rows[~accepted_rows]],
            base_full[valid_rows[~accepted_rows]],
        )
    ):
        raise RuntimeError("source-LOO gate changed observed or rejected rows")
    summary = _aggregate(records, accepted, labels)
    output_tensors = {
        "primitive_probability": gated_full.contiguous(),
        "valid_rows": valid_rows.contiguous(),
        "unknown_component_labels": labels.contiguous(),
        "component_accepted": accepted.contiguous(),
        "validation_eligible": validation_eligible.contiguous(),
        "fold_assignment": folds.contiguous(),
    }
    tensor_digests = {
        name: tensor_sha256(value) for name, value in sorted(output_tensors.items())
    }
    artifact = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "scene_id": args.scene_id,
        "method_contract": contract,
        "method_contract_sha256": contract_sha256,
        "preregistration_path": str(prereg_path),
        "preregistration_sha256": prereg_sha256,
        "source_prediction_authority_path": str(authority_path),
        "source_prediction_authority_sha256": authority_sha256,
        "base_selector_sha256": authority["selector"]["sha256"],
        "base_unary_sha256": authority["pre_gt_source_access"]["files"][
            "base_unary"
        ]["sha256"],
        "support_graph_sha256": authority["support_graph"]["sha256"],
        "exact_w_cache_sha256": authority["exact_w"]["cache_sha256"],
        "summary": summary,
        "tensors": output_tensors,
        "tensor_sha256": tensor_digests,
        "tensor_bundle_sha256": canonical_json_sha256(tensor_digests),
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    output = write_torch_noclobber(args.output, artifact)
    output_sha256 = sha256_file(output)
    frozen, _digest, _path = load_torch_mapping(
        output,
        expected_sha256=output_sha256,
        label="strictly reloaded source-LOO gated selector",
    )
    if frozen.get("tensor_sha256") != tensor_digests:
        raise RuntimeError("source-LOO selector changed across strict reload")
    report = {
        "schema_version": 1,
        "artifact_type": "radio_gs.nvos_source_loo_component_gate_report",
        "scene_id": args.scene_id,
        "method_contract": contract,
        "method_contract_sha256": contract_sha256,
        "preregistration_path": str(prereg_path),
        "preregistration_sha256": prereg_sha256,
        "source_prediction_authority_path": str(authority_path),
        "source_prediction_authority_sha256": authority_sha256,
        "output": str(output),
        "output_sha256": output_sha256,
        "primitive_probability_sha256": tensor_digests["primitive_probability"],
        "summary": summary,
        "components": records,
        "safety": {
            "source_only": True,
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_computed": False,
        },
    }
    report_path = write_frozen_json(args.report, report)
    return {
        **{key: value for key, value in report.items() if key != "components"},
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--prediction-authority", required=True)
    parser.add_argument("--prediction-authority-sha256", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(materialize(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
