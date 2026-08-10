#!/usr/bin/env python3
"""Build endpoint-safe FIX5 LERF scores with raw-dominant semantic graph gates.

FIX5 retains the exact frozen O0 readout, source-promoted positive-utility edge
confidence, query gate, three-region cap, and monotone FIX4C fusion.  Its only
semantic change is applied before graph support propagation: an O0 anchor and
a graph candidate must each be dominated by that query under the valid-core
mean raw canonical-negative probability.  Raw probabilities are read at each
query's frozen O0-selected scale before independent VALA min/max remapping.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import lerf_raw_unary_region_specificity as unary
from radio_gs.scripts import build_lerf_o0_anchored_positive_utility_residual_cache as fix4b
from radio_gs.scripts import build_lerf_o0_anchored_positive_utility_residual_cache_fix4c as fix4c
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
)


SCHEMA = "radio_gs.lerf_o0_anchored_raw_dominant_positive_utility_external_scores.v1"
EXECUTION_SCHEMA = (
    "radio_gs.lerf_o0_anchored_raw_dominant_positive_utility_execution_fix5.v1"
)
EXECUTION_STATUS = (
    "authorized_source_fixed_raw_dominant_positive_utility_target_score_cache_only"
)
IMPLEMENTATION = Path(__file__).resolve()
DEPENDENCIES = {
    "raw_unary_interface": Path(unary.__file__).resolve(),
    "frozen_fix4b_builder": Path(fix4b.__file__).resolve(),
    "endpoint_safe_fix4c_builder": Path(fix4c.__file__).resolve(),
}


def raw_specificity_with_invalid_region_fallback(
    *,
    raw_query_probabilities: torch.Tensor,
    region_rows: torch.Tensor,
    core_mask: torch.Tensor,
    primitive_valid_mask: torch.Tensor,
) -> unary.RawUnaryRegionSpecificity:
    """Keep all-invalid region cores semantically inactive and exact-fallback."""

    raw = torch.as_tensor(raw_query_probabilities).detach().float().cpu().contiguous()
    rows = torch.as_tensor(region_rows).detach().long().cpu().contiguous()
    core = torch.as_tensor(core_mask).detach().bool().cpu().contiguous()
    valid = torch.as_tensor(primitive_valid_mask).detach().bool().cpu().contiguous()
    if (
        raw.ndim != 2
        or rows.ndim != 2
        or core.shape != rows.shape
        or valid.shape != (raw.shape[0],)
        or bool((rows[core] < 0).any())
        or bool((rows[core] >= raw.shape[0]).any())
    ):
        raise ValueError("FIX5 raw specificity fallback axes differ")
    safe = rows.clamp(min=0, max=raw.shape[0] - 1)
    valid_counts = (core & valid[safe]).sum(dim=1)
    usable = valid_counts > 0
    if not bool(usable.any()):
        raise ValueError("FIX5 target has no region with a valid core primitive")
    subset = unary.raw_unary_region_specificity(
        raw_query_probabilities=raw,
        region_rows=rows[usable],
        core_mask=core[usable],
        primitive_valid_mask=valid,
    )
    region_count, query_count = rows.shape[0], raw.shape[1]
    mean = torch.zeros((region_count, query_count), dtype=torch.float32)
    dominant = torch.zeros((region_count, query_count), dtype=torch.bool)
    fraction = torch.zeros((region_count, query_count), dtype=torch.float32)
    mean[usable] = subset.mean_raw_probability
    dominant[usable] = subset.dominant_query_mask
    fraction[usable] = subset.primitive_top1_fraction
    if bool(dominant[~usable].any()):
        raise RuntimeError("FIX5 invalid region acquired raw dominance")
    return unary.RawUnaryRegionSpecificity(
        mean_raw_probability=mean.contiguous(),
        dominant_query_mask=dominant.contiguous(),
        primitive_top1_fraction=fraction.contiguous(),
        valid_core_counts=valid_counts.long().contiguous(),
    )


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact file record")
    path = validate_file_record(value, label=label)
    return {"path": str(path), "sha256": str(value["sha256"])}


def _load_and_validate_execution(
    path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="FIX5 target execution authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "implementation",
        "dependencies",
        "source_fix5_execution_authority",
        "source_fix5_result",
        "parent_fix4c_execution_authority",
        "parent_fix4c_cache",
        "parent_fix4c_report",
        "fixed_intervention",
        "output_cache",
        "output_report",
        "target_score_cache_authorized",
        "target_quality_execution_authorized",
        "access_audit",
    }
    authority = dict(raw)
    if (
        set(authority) != required
        or authority.get("schema") != EXECUTION_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != EXECUTION_STATUS
        or authority.get("target_score_cache_authorized") is not True
        or authority.get("target_quality_execution_authorized") is not False
        or authority.get("fixed_intervention")
        != {
            "raw_input": "canonical_negative_probability_before_VALA_minmax_at_each_query_frozen_O0_scale",
            "region_statistic": "valid_core_mean_raw_probability_argmax_all_exact_ties_retained",
            "anchor_order": "filter_O0_anchor_before_direct_graph_support_propagation",
            "candidate_gate": "existing_graph_candidate_and_raw_dominant_query",
            "primitive_majority_threshold": None,
            "residual_and_selection": "bitwise_frozen_FIX4B",
            "probability_fusion": "endpoint_safe_monotone_FIX4C",
        }
        or authority.get("access_audit")
        != {
            "query_names_opened": True,
            "raw_query_score_cache_opened": True,
            "target_images_opened": False,
            "target_quality_data_opened": False,
            "target_quality_readout_executed": False,
        }
    ):
        raise ValueError("FIX5 target execution header differs")
    if validate_file_record(authority["implementation"], label="FIX5 implementation") != IMPLEMENTATION:
        raise ValueError("FIX5 target implementation differs")
    dependencies = authority["dependencies"]
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(DEPENDENCIES):
        raise ValueError("FIX5 dependency fields differ")
    for name, dependency in DEPENDENCIES.items():
        if validate_file_record(dependencies[name], label=f"FIX5 {name}") != dependency:
            raise ValueError(f"FIX5 dependency differs: {name}")

    source_authority_record = _record(
        authority["source_fix5_execution_authority"],
        label="FIX5 source execution authority",
    )
    source_result_record = _record(
        authority["source_fix5_result"], label="FIX5 source result"
    )
    source_result, _, _ = load_json_object(
        source_result_record["path"],
        expected_sha256=source_result_record["sha256"],
        label="FIX5 promoted source result",
    )
    if (
        source_result.get("status")
        != "source_only_raw_dominant_FIX5_promoted_target_unopened"
        or source_result.get("execution_authority") != source_authority_record
        or source_result.get("raw_unary_contract_sha256") != unary.CONTRACT_SHA256
        or source_result.get("promotion_gate", {}).get("outcomes", {}).get("passed")
        is not True
        or source_result.get("target_execution_performed") is not False
    ):
        raise ValueError("FIX5 source promotion binding differs")

    parent_record = _record(
        authority["parent_fix4c_execution_authority"],
        label="FIX5 parent FIX4C execution authority",
    )
    parent = fix4c._load_and_validate_execution(
        parent_record["path"], expected_sha256=parent_record["sha256"]
    )
    parent_cache = _record(authority["parent_fix4c_cache"], label="FIX5 parent cache")
    parent_report = _record(authority["parent_fix4c_report"], label="FIX5 parent report")
    if (
        parent["output_cache"] != parent_cache["path"]
        or parent["output_report"] != parent_report["path"]
    ):
        raise ValueError("FIX5 parent FIX4C output binding differs")
    output = fix4b.legacy._output_path(authority["output_cache"], name="FIX5 output cache")
    report = fix4b.legacy._output_path(authority["output_report"], name="FIX5 output report")
    if output == report or output in {
        parent_cache["path"],
        parent_report["path"],
        parent["supersedes_fix4b_cache"]["path"],
        parent["supersedes_fix4b_report"]["path"],
    }:
        raise ValueError("FIX5 outputs must be new and distinct")
    parent.update(
        {
            "output_cache": output,
            "output_report": report,
            "verified_record": {"path": str(source), "sha256": digest},
            "source_fix5_execution_authority": source_authority_record,
            "source_fix5_result": source_result_record,
            "parent_fix4c_execution_authority": parent_record,
            "parent_fix4c_cache": parent_cache,
            "parent_fix4c_report": parent_report,
            "access_audit": authority["access_audit"],
        }
    )
    return parent


def _load_raw_specificity(
    execution: Mapping[str, Any],
) -> tuple[unary.RawUnaryRegionSpecificity, dict[str, Any]]:
    positive_raw, positive_sha, positive_path = load_torch_mapping(
        execution["positive_o0_cache"]["path"],
        expected_sha256=execution["positive_o0_cache"]["sha256"],
        map_location="cpu",
        label="FIX5 positive frozen O0 cache",
    )
    negative_raw, negative_sha, negative_path = load_torch_mapping(
        execution["negative_o0_cache"]["path"],
        expected_sha256=execution["negative_o0_cache"]["sha256"],
        map_location="cpu",
        label="FIX5 negative frozen O0 cache",
    )
    query_ids = tuple(str(item) for item in positive_raw.get("query_ids", ()))
    positive = fix4b.v2.frozen.validate_ours_multiscale_query_score_cache(
        positive_raw,
        expected_xyz=torch.as_tensor(positive_raw.get("xyz")),
        expected_query_ids=query_ids,
        expected_renderer_geometry_checkpoint_sha256=execution[
            "renderer_geometry_checkpoint"
        ]["sha256"],
    )
    negative = fix4b.v2.frozen.validate_ours_multiscale_query_score_cache(
        negative_raw,
        expected_xyz=positive_raw["xyz"],
        expected_query_ids=fix4b.v2.frozen.NEGATIVE_PROMPTS,
        expected_renderer_geometry_checkpoint_sha256=execution[
            "renderer_geometry_checkpoint"
        ]["sha256"],
    )
    for name in ("valid", "scale_ids", "scale_radii_m", "xyz_sha256"):
        left, right = getattr(positive, name), getattr(negative, name)
        if not bool(torch.equal(left, right) if torch.is_tensor(left) else left == right):
            raise ValueError(f"FIX5 positive/negative O0 {name} differs")
    o0 = fix4b.legacy.exact_o0_readout(
        positive_scores=positive.query_scores,
        negative_scores=negative.query_scores,
        xyz=torch.as_tensor(positive_raw["xyz"]).float().cpu().contiguous(),
        valid=positive.valid,
        chunk_size=int(execution["knn_chunk_size"]),
    )
    raw_probability = fix4b.v2.frozen.canonical_negative_relevancy_query_scores(
        positive.query_scores,
        negative.query_scores,
        logit_scale=10.0,
    )
    primitive_count, scale_count, query_count = raw_probability.shape
    if o0.selected_scale_indices.shape != (query_count,):
        raise ValueError("FIX5 selected O0 scale axis differs")
    gather = o0.selected_scale_indices.view(1, 1, query_count).expand(
        primitive_count, 1, query_count
    )
    raw_selected = raw_probability.gather(1, gather).squeeze(1).contiguous()

    feature_raw, feature_sha, feature_path = load_torch_mapping(
        execution["region_feature_authority"]["path"],
        expected_sha256=execution["region_feature_authority"]["sha256"],
        map_location="cpu",
        label="FIX5 target region features",
    )
    region_rows = torch.as_tensor(feature_raw["region_rows"]).long().cpu().contiguous()
    core_mask = torch.as_tensor(feature_raw["token_mask"]).bool().cpu().contiguous()
    specificity = raw_specificity_with_invalid_region_fallback(
        raw_query_probabilities=raw_selected.float().cpu().contiguous(),
        region_rows=region_rows,
        core_mask=core_mask,
        primitive_valid_mask=positive.valid.bool().cpu().contiguous(),
    )
    parent_raw, _, _ = load_torch_mapping(
        execution["parent_fix4c_cache"]["path"],
        expected_sha256=execution["parent_fix4c_cache"]["sha256"],
        map_location="cpu",
        label="FIX5 parent FIX4C cache",
    )
    parent_scale = torch.as_tensor(
        parent_raw.get("selection", {}).get("selected_scale_indices")
    ).long().cpu()
    if (
        parent_scale.shape != o0.selected_scale_indices.shape
        or not torch.equal(parent_scale, o0.selected_scale_indices)
        or parent_raw.get("metadata", {}).get("query_names") != list(query_ids)
    ):
        raise ValueError("FIX5 parent cache O0 scale/query binding differs")
    binding = {
        "query_ids": list(query_ids),
        "positive_o0_cache": {"path": str(positive_path), "sha256": positive_sha},
        "negative_o0_cache": {"path": str(negative_path), "sha256": negative_sha},
        "region_features": {"path": str(feature_path), "sha256": feature_sha},
        "selected_scale_indices": o0.selected_scale_indices.tolist(),
        "selected_scale_count": scale_count,
        "region_rows": region_rows,
        "core_mask": core_mask,
    }
    return specificity, binding


def _eligible_edge_mask(
    *,
    pair_probabilities: torch.Tensor,
    pair_features: torch.Tensor,
    pair_feature_median: torch.Tensor,
    pair_feature_robust_scale: torch.Tensor,
    config: fix4b.PositiveUtilityDeployment,
) -> torch.Tensor:
    probability = torch.as_tensor(pair_probabilities).detach().float().cpu()
    feature = torch.as_tensor(pair_features).detach().float().cpu()
    median = torch.as_tensor(pair_feature_median).detach().float().cpu()
    scale = torch.as_tensor(pair_feature_robust_scale).detach().float().cpu()
    if (
        probability.ndim != 1
        or feature.ndim != 2
        or feature.shape[0] != probability.numel()
        or feature.shape[1] <= 18
        or median.shape != (feature.shape[1],)
        or scale.shape != median.shape
        or not bool(torch.isfinite(probability).all())
        or not bool(torch.isfinite(feature).all())
        or not bool(torch.isfinite(median).all())
        or not bool(torch.isfinite(scale).all())
        or bool((scale <= 0.0).any())
    ):
        raise ValueError("FIX5 target eligible-edge inputs differ")
    reliability = feature[:, [17, 18]].amin(dim=1)
    ood_raw = ((feature - median) / scale).abs().amax(dim=1)
    ood_unit = ood_raw / (ood_raw + float(config.feature_ood_raw_limit))
    return (
        (probability >= float(config.raw_edge_probability_minimum))
        & (reliability >= float(config.minimum_reliability))
        & (ood_unit <= float(config.maximum_feature_ood_score))
    ).contiguous()


def apply_raw_dominant_gate_to_evidence(
    base: fix4b.RegionEvidence,
    *,
    specificity: unary.RawUnaryRegionSpecificity,
    pair_indices: torch.Tensor,
    pair_probabilities: torch.Tensor,
    pair_features: torch.Tensor,
    pair_feature_median: torch.Tensor,
    pair_feature_robust_scale: torch.Tensor,
    config: fix4b.PositiveUtilityDeployment,
) -> fix4b.RegionEvidence:
    """Apply raw specificity before recomputing direct graph support."""

    if specificity.dominant_query_mask.shape != base.anchor_region.shape:
        raise ValueError("FIX5 raw specificity/evidence region-query axes differ")
    edge_eligible = _eligible_edge_mask(
        pair_probabilities=pair_probabilities,
        pair_features=pair_features,
        pair_feature_median=pair_feature_median,
        pair_feature_robust_scale=pair_feature_robust_scale,
        config=config,
    )
    gated = unary.symmetric_raw_dominant_graph_gate(
        base_anchor_region=base.anchor_region,
        dominant_query_mask=specificity.dominant_query_mask,
        pair_indices=pair_indices,
        edge_eligible_mask=edge_eligible,
        region_eligible_mask=base.eligible,
        anchor_quorum=config.anchor_quorum,
    )
    if bool((gated.specific_candidate_region & ~base.candidate_region).any()):
        raise RuntimeError("FIX5 semantic candidate escaped FIX4B candidate set")
    lower = torch.where(
        gated.specific_candidate_region,
        base.lower,
        torch.zeros_like(base.lower),
    )
    query_gate = base.query_gate & gated.specific_candidate_region.any(dim=0)
    diagnostics = dict(base.diagnostics)
    diagnostics.update(
        {
            "raw_dominant_region_count": specificity.dominant_query_mask.sum(dim=0).long(),
            "raw_specific_anchor_count": gated.specific_anchor_region.sum(dim=0).long(),
            "raw_specific_candidate_count": gated.specific_candidate_region.sum(dim=0).long(),
        }
    )
    return fix4b.RegionEvidence(
        lower=lower.contiguous(),
        eligible=base.eligible,
        query_gate=query_gate.contiguous(),
        anchor_region=gated.specific_anchor_region,
        direct_anchor_support=gated.direct_specific_anchor_support,
        candidate_region=gated.specific_candidate_region,
        diagnostics=diagnostics,
        rank256_top_tail=base.rank256_top_tail,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    execution = _load_and_validate_execution(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    specificity, raw_binding = _load_raw_specificity(execution)
    fusion_audit: dict[str, Any] = {}
    selected_specificity: list[list[dict[str, Any]]] = []
    original_loader = fix4b._load_and_validate_execution
    original_evidence = fix4b.build_region_evidence
    original_fusion = fix4b.fuse_exact_o0_probabilities
    original_schema = fix4b.SCHEMA
    original_implementation = fix4b.IMPLEMENTATION
    original_write_torch = fix4b.write_torch_noclobber
    original_write_json = fix4b.write_frozen_json

    def injected_loader(path: str | Path, *, expected_sha256: str):
        del path, expected_sha256
        return execution

    def injected_evidence(**kwargs):
        rows = torch.as_tensor(kwargs["region_rows"]).long().cpu()
        core = torch.as_tensor(kwargs["core_mask"]).bool().cpu()
        if not torch.equal(rows, raw_binding["region_rows"]) or not torch.equal(
            core, raw_binding["core_mask"]
        ):
            raise ValueError("FIX5 raw specificity and consumer region cores differ")
        base = original_evidence(**kwargs)
        return apply_raw_dominant_gate_to_evidence(
            base,
            specificity=specificity,
            pair_indices=kwargs["pair_indices"],
            pair_probabilities=kwargs["pair_probabilities"],
            pair_features=kwargs["pair_features"],
            pair_feature_median=kwargs["pair_feature_median"],
            pair_feature_robust_scale=kwargs["pair_feature_robust_scale"],
            config=kwargs["config"],
        )

    def injected_fusion(o0_scores, result):
        fused, changed, audit = fix4c.fuse_exact_o0_probabilities_monotone(
            o0_scores, result
        )
        fusion_audit.update(audit)
        return fused, changed

    def write_cache(path, payload):
        payload["metadata"]["score_semantics"] = (
            "exact_O0_VALA_plus_source_fixed_raw_dominant_positive_utility_"
            "residual_with_monotone_exact_O0_probability_fusion"
        )
        payload["metadata"]["raw_unary_contract_sha256"] = unary.CONTRACT_SHA256
        payload["metadata"]["source_fix5_result"] = execution["source_fix5_result"]
        payload["metadata"]["supersedes_fix4c_cache"] = execution["parent_fix4c_cache"]
        payload["metadata"]["producer"] = file_record(IMPLEMENTATION)
        selection = payload["selection"]
        selection["raw_dominant_region_counts"] = specificity.dominant_query_mask.sum(dim=0).long()
        selected_specificity.clear()
        for query, selected_rows in enumerate(selection["selected_region_rows"]):
            query_rows: list[dict[str, Any]] = []
            for row in selected_rows:
                mean = specificity.mean_raw_probability[int(row)]
                q_value = float(mean[query])
                competitor = mean.clone()
                competitor[query] = -1.0
                query_rows.append(
                    {
                        "region_row": int(row),
                        "canonical_region_index": int(
                            selection["selected_canonical_region_indices"][query][
                                len(query_rows)
                            ]
                        ),
                        "query_mean_raw_probability": q_value,
                        "runner_up_mean_raw_probability": float(competitor.max()),
                        "winner_margin": q_value - float(competitor.max()),
                        "primitive_top1_fraction_diagnostic_only": float(
                            specificity.primitive_top1_fraction[int(row), query]
                        ),
                    }
                )
            selected_specificity.append(query_rows)
        selection["selected_raw_specificity"] = tuple(
            tuple(tuple(row.items()) for row in query_rows)
            for query_rows in selected_specificity
        )
        return original_write_torch(path, payload)

    def write_report(path, payload):
        if not fusion_audit or len(selected_specificity) != len(raw_binding["query_ids"]):
            raise RuntimeError("FIX5 target audits were not materialized")
        payload["status"] = "o0_anchored_raw_dominant_positive_utility_FIX5_cache_complete"
        payload["raw_dominant_specificity_audit"] = {
            "input": "raw_canonical_probability_before_VALA_minmax_at_frozen_O0_selected_scale",
            "aggregation": "valid_region_core_mean",
            "primitive_majority_threshold": None,
            "query_ids": raw_binding["query_ids"],
            "selected_scale_indices": raw_binding["selected_scale_indices"],
            "raw_dominant_region_counts": specificity.dominant_query_mask.sum(dim=0).tolist(),
            "selected_regions": selected_specificity,
        }
        payload["monotone_fusion_audit"] = dict(fusion_audit)
        payload["bitwise_invariants"].update(
            {
                "selected_updates_non_decreasing": fusion_audit[
                    "selected_updates_non_decreasing"
                ],
                "actual_changes_strictly_increase_exact_O0": fusion_audit[
                    "actual_changes_strictly_increase_exact_O0"
                ],
            }
        )
        payload["source_fix5_result"] = execution["source_fix5_result"]
        payload["parent_fix4c_cache"] = execution["parent_fix4c_cache"]
        payload["parent_fix4c_report"] = execution["parent_fix4c_report"]
        return original_write_json(path, payload)

    try:
        fix4b._load_and_validate_execution = injected_loader
        fix4b.build_region_evidence = injected_evidence
        fix4b.fuse_exact_o0_probabilities = injected_fusion
        fix4b.SCHEMA = SCHEMA
        fix4b.IMPLEMENTATION = IMPLEMENTATION
        fix4b.write_torch_noclobber = write_cache
        fix4b.write_frozen_json = write_report
        return fix4b.run(args)
    finally:
        fix4b._load_and_validate_execution = original_loader
        fix4b.build_region_evidence = original_evidence
        fix4b.fuse_exact_o0_probabilities = original_fusion
        fix4b.SCHEMA = original_schema
        fix4b.IMPLEMENTATION = original_implementation
        fix4b.write_torch_noclobber = original_write_torch
        fix4b.write_frozen_json = original_write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
