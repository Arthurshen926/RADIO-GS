#!/usr/bin/env python3
"""Build a no-GT LERF cache by adding a bounded graph residual to O0.

The canonical capability is the exact frozen O0 canonical-negative/VALA
readout.  A query must expose at least two O0-supermajority anchors; a target
region can then receive a positive residual through one reliable, in-domain,
direct source-promoted edge from any such anchor.  Rank-256 target routing supplies query-independent validity,
reliability, and OOD checks.  Rank-256 text relevance is authority-bound and
reported for drift diagnosis, but it cannot admit or reject a region.

This module has deliberately no target-quality entry point.  It only writes a
row-aligned primitive score cache and an evidence report, both first-writer
only.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import lerf_o0_anchored_conformal_residual as residual
from radio_gs.interfaces import surface_region_rank256_champion as champion
from radio_gs.interfaces.region_comembership_v2_formal import validate_checkpoint
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.querying.bounded_region_comembership_readout import (
    thresholded_adjacency,
)
from radio_gs.scripts import build_lerf_region_comembership_external_cache_v1 as v1
from radio_gs.scripts import build_lerf_region_comembership_external_cache_v2 as v2
from radio_gs.scripts import calibrate_source_only_graph_confidence_v1 as fix2_calibration
from radio_gs.scripts import calibrate_source_only_graph_consumer_exact_fix3 as fix3_calibration
from radio_gs.scripts.infer_region_comembership_v2 import (
    validate_inference_authority,
)
from radio_gs.scripts.materialize_lerf_multiscale_query_score_cache import (
    _renderer_checkpoint_xyz,
)
from radio_gs.scripts.materialize_region_comembership_features_v2 import (
    validate_feature_authority,
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


SCHEMA = "radio_gs.lerf_o0_anchored_graph_residual_external_scores.v1"
EXECUTION_SCHEMA = "radio_gs.lerf_o0_anchored_graph_residual_execution.v1"
EXECUTION_STATUS = "authorized_source_fixed_target_score_cache_only"
CALIBRATION_SCHEMA = "radio_gs.source_only_graph_consumer_exact_calibration_fix3.v1"
IMPLEMENTATION = Path(__file__).resolve()
DEPENDENCIES = {
    "o0_frozen_readout": Path(v2.frozen.__file__).resolve(),
    "o0_target_binding": Path(v2.__file__).resolve(),
    "o0_scale_axis": Path(v1.__file__).resolve(),
    "residual_interface": Path(residual.__file__).resolve(),
    "direct_graph": Path(thresholded_adjacency.__code__.co_filename).resolve(),
    "rank256_authority": Path(champion.__file__).resolve(),
    "target_accepted_authority": Path(
        validate_target_accepted_v2_authority.__code__.co_filename
    ).resolve(),
    "target_feature_authority": Path(
        validate_feature_authority.__code__.co_filename
    ).resolve(),
    "target_inference_authority": Path(
        validate_inference_authority.__code__.co_filename
    ).resolve(),
    "renderer_geometry_loader": Path(
        _renderer_checkpoint_xyz.__code__.co_filename
    ).resolve(),
    "source_calibration_fix2": Path(fix2_calibration.__file__).resolve(),
    "source_calibration_fix3": Path(fix3_calibration.__file__).resolve(),
}
O0_SELECTION_THRESHOLD = 0.6
O0_LOGIT_CLAMP = torch.finfo(torch.float32).eps


@dataclass(frozen=True)
class DeploymentCalibration:
    epsilon_logit: float
    novel_mass_reference: float
    null_step_thresholds: tuple[float, ...]
    minimum_reliability: float
    maximum_feature_ood_score: float
    minimum_anchor_agreement: float
    maximum_null_activation: float
    minimum_stability: float
    raw_edge_probability_minimum: float
    calibrated_region_lower_minimum: float
    graph_method: str
    graph_safety_cap: int
    maximum_selected_regions: int
    anchor_quorum: int
    top_tail_size: int
    query_top_tail_quantile: float
    o0_supermajority_fraction: float
    o0_final_score_minimum: float
    feature_ood_raw_limit: float
    maximum_target_ood_fraction: float
    stability_required_fraction: float
    missing_query_evidence_action: str

    def residual_config(self) -> residual.SourceFixedResidualConfig:
        return residual.SourceFixedResidualConfig(
            epsilon_logit=self.epsilon_logit,
            novel_mass_reference=self.novel_mass_reference,
            null_step_thresholds=self.null_step_thresholds,
            minimum_reliability=self.minimum_reliability,
            maximum_feature_ood_score=self.maximum_feature_ood_score,
            minimum_anchor_agreement=self.minimum_anchor_agreement,
            maximum_null_activation=self.maximum_null_activation,
            minimum_stability=self.minimum_stability,
        )


@dataclass(frozen=True)
class O0Readout:
    final_scores: torch.Tensor
    scores_by_scale: torch.Tensor
    selected_scale_indices: torch.Tensor
    raw_smoothed_peaks: torch.Tensor


@dataclass(frozen=True)
class RegionEvidence:
    lower: torch.Tensor
    eligible: torch.Tensor
    query_gate: torch.Tensor
    anchor_region: torch.Tensor
    direct_anchor_support: torch.Tensor
    candidate_region: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
    rank256_top_tail: torch.Tensor


def _record_shape(value: object, *, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} record differs")
    path = str(value["path"])
    digest = str(value["sha256"])
    if (
        path != str(Path(path).expanduser().resolve())
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} record differs")
    return {"path": path, "sha256": digest}


def _output_path(value: object, *, name: str) -> str:
    raw = str(value)
    resolved = str(Path(raw).expanduser().resolve())
    if raw != resolved:
        raise ValueError(f"{name} must be an absolute canonical path")
    return resolved


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_deployment_config(value: object) -> DeploymentCalibration:
    required = {
        "residual_config", "graph", "anchor", "feature_ood",
        "query_gate_diagnostics", "source_query_bank_requirement",
        "fallback_on_any_failed_gate",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("source deployment config fields differ")
    raw = dict(value)
    residual_raw = raw["residual_config"]
    graph = raw["graph"]
    anchor = raw["anchor"]
    feature_ood = raw["feature_ood"]
    residual_keys = {
        "epsilon_logit", "novel_mass_reference", "null_step_thresholds",
        "minimum_reliability", "maximum_feature_ood_score",
        "minimum_anchor_agreement", "maximum_null_activation",
        "minimum_stability",
    }
    graph_keys = {
        "method", "raw_edge_probability_minimum", "edge_lower_formula",
        "calibrated_edge_lower_minimum", "anchor_quorum",
        "anchor_quorum_definition", "maximum_selected_regions",
        "formal_checkpoint_safety_cap", "candidate_region_lower",
        "null_calibration",
    }
    anchor_keys = {
        "o0_core_supermajority", "o0_final_score_minimum", "definition",
        "rank256_query_top_tail", "top_tail_size",
    }
    ood_keys = {
        "input", "raw_score", "raw_score_limit", "unit_score",
        "unit_score_maximum", "scene_quantile", "maximum_target_ood_fraction",
    }
    if (
        not isinstance(residual_raw, Mapping) or set(residual_raw) != residual_keys
        or not isinstance(graph, Mapping) or set(graph) != graph_keys
        or not isinstance(anchor, Mapping) or set(anchor) != anchor_keys
        or not isinstance(feature_ood, Mapping) or set(feature_ood) != ood_keys
        or not isinstance(raw["query_gate_diagnostics"], Mapping)
        or set(raw["query_gate_diagnostics"])
        != {"reliability", "feature_ood_score", "anchor_agreement", "null_activation", "stability"}
    ):
        raise ValueError("source deployment nested fields differ")
    thresholds = residual_raw["null_step_thresholds"]
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError("source null step thresholds differ")
    config = DeploymentCalibration(
        epsilon_logit=_finite(residual_raw["epsilon_logit"], name="epsilon_logit"),
        novel_mass_reference=_finite(
            residual_raw["novel_mass_reference"], name="novel_mass_reference"
        ),
        null_step_thresholds=tuple(
            _finite(item, name="null_step_threshold") for item in thresholds
        ),
        minimum_reliability=_finite(
            residual_raw["minimum_reliability"], name="minimum_reliability"
        ),
        maximum_feature_ood_score=_finite(
            residual_raw["maximum_feature_ood_score"], name="maximum_feature_ood_score"
        ),
        minimum_anchor_agreement=_finite(
            residual_raw["minimum_anchor_agreement"], name="minimum_anchor_agreement"
        ),
        maximum_null_activation=_finite(
            residual_raw["maximum_null_activation"], name="maximum_null_activation"
        ),
        minimum_stability=_finite(
            residual_raw["minimum_stability"], name="minimum_stability"
        ),
        raw_edge_probability_minimum=_finite(
            graph["raw_edge_probability_minimum"],
            name="raw_edge_probability_minimum",
        ),
        calibrated_region_lower_minimum=_finite(
            graph["calibrated_edge_lower_minimum"],
            name="calibrated_region_lower_minimum",
        ),
        graph_method=str(graph["method"]),
        graph_safety_cap=int(graph["formal_checkpoint_safety_cap"]),
        maximum_selected_regions=int(graph["maximum_selected_regions"]),
        anchor_quorum=int(graph["anchor_quorum"]),
        top_tail_size=int(anchor["top_tail_size"]),
        query_top_tail_quantile=_finite(
            feature_ood["scene_quantile"], name="query_top_tail_quantile"
        ),
        o0_supermajority_fraction=_finite(
            anchor["o0_core_supermajority"],
            name="o0_supermajority_fraction",
        ),
        o0_final_score_minimum=_finite(
            anchor["o0_final_score_minimum"], name="o0_final_score_minimum"
        ),
        feature_ood_raw_limit=_finite(
            feature_ood["raw_score_limit"], name="feature_ood_raw_limit"
        ),
        maximum_target_ood_fraction=_finite(
            feature_ood["maximum_target_ood_fraction"],
            name="maximum_target_ood_fraction",
        ),
        stability_required_fraction=_finite(
            residual_raw["minimum_stability"], name="stability_required_fraction"
        ),
        missing_query_evidence_action=str(raw["source_query_bank_requirement"]),
    )
    unit = (
        config.minimum_reliability,
        config.maximum_feature_ood_score,
        config.minimum_anchor_agreement,
        config.maximum_null_activation,
        config.minimum_stability,
        config.raw_edge_probability_minimum,
        config.calibrated_region_lower_minimum,
        config.query_top_tail_quantile,
        config.o0_supermajority_fraction,
        config.o0_final_score_minimum,
        config.maximum_target_ood_fraction,
        config.stability_required_fraction,
    )
    if (
        any(not 0.0 <= item <= 1.0 for item in unit)
        or config.novel_mass_reference <= 0.0
        or config.epsilon_logit < 0.0
        or any(item < 0.0 for item in config.null_step_thresholds)
        or config.feature_ood_raw_limit <= 0.0
        or config.graph_method != "direct_O0_anchor_edge_residual"
        or config.graph_safety_cap != 8
        or config.maximum_selected_regions != 3
        or config.anchor_quorum != fix2_calibration.ANCHOR_QUORUM
        or config.top_tail_size <= 0
        or config.raw_edge_probability_minimum
        != fix2_calibration.RAW_EDGE_PROBABILITY_MINIMUM
        or config.novel_mass_reference != fix2_calibration.NOVEL_MASS_REFERENCE
        or config.o0_supermajority_fraction
        != fix2_calibration.O0_CORE_SUPERMAJORITY
        or config.o0_final_score_minimum != O0_SELECTION_THRESHOLD
        or config.maximum_null_activation
        != fix2_calibration.MAXIMUM_NULL_ACTIVATION
        or config.minimum_stability != fix2_calibration.MINIMUM_STABILITY
        or len(config.null_step_thresholds) != config.maximum_selected_regions
        or config.missing_query_evidence_action
        != "not_applicable_primary_uses_only_runtime_O0_supermajority_anchors_and_source_calibrated_query_independent_direct_edges"
        or raw["fallback_on_any_failed_gate"] != "bitwise_O0_only_no_graph_residual"
        or anchor["definition"]
        != "mean(valid_core_exact_O0_final_score_strictly_greater_than_0.6)"
        or anchor["rank256_query_top_tail"] != "diagnostic_not_used_by_primary"
        or graph["edge_lower_formula"]
        != "sigmoid(logit(clamp(p,1e-7,1-1e-7))-epsilon_logit)"
        or graph["null_calibration"]
        != "consumer_exact_direct_false_edge_greedy_with_covered_union_and_marginal_primitive_deduplication"
        or abs(
            float(
                torch.sigmoid(
                    torch.logit(torch.tensor(config.raw_edge_probability_minimum))
                    - config.epsilon_logit
                )
            )
            - config.calibrated_region_lower_minimum
        )
        > 1e-6
    ):
        raise ValueError("source deployment config contract differs")
    config.residual_config()
    return config


def validate_source_calibration(
    value: object,
) -> tuple[dict[str, Any], DeploymentCalibration]:
    """Validate the source-only calibration without accepting extra fields."""

    required = {
        "schema", "schema_version", "status", "execution_authority",
        "supersedes_fix2_result", "fix2_issue",
        "consumer_exact_null_calibration", "source_true_edge_selection_audit",
        "deployment_config", "source_access", "benchmark_execution_authorized",
        "target_execution_performed", "content_authority_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("source graph calibration fields differ")
    payload = dict(value)
    if (
        payload["schema"] != CALIBRATION_SCHEMA
        or payload["schema_version"] != 1
        or payload["status"]
        != "source_only_consumer_exact_fix3_complete_target_unopened"
        or payload["benchmark_execution_authorized"] is not False
        or payload["target_execution_performed"] is not False
        or payload["source_access"] != fix3_calibration.source_access()
        or payload["content_authority_sha256"]
        != canonical_json_sha256(
            {key: item for key, item in payload.items() if key != "content_authority_sha256"}
        )
    ):
        raise ValueError("source graph calibration header differs")
    for name in ("execution_authority", "supersedes_fix2_result"):
        payload[name] = _record_shape(payload[name], name=f"calibration {name}")
    return payload, _validate_deployment_config(payload["deployment_config"])


def exact_o0_readout(
    *,
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    xyz: torch.Tensor,
    valid: torch.Tensor,
    chunk_size: int,
) -> O0Readout:
    """Reproduce canonical-negative plus frozen VALA peak-scale readout."""

    probability = v2.frozen.canonical_negative_relevancy_query_scores(
        positive_scores,
        negative_scores,
        logit_scale=10.0,
    )
    if probability.ndim != 3 or probability.shape[1] != 3:
        raise ValueError("frozen O0 probability scale axis differs")
    count, levels, queries = probability.shape
    smoothed = v2.frozen.vala_knn_smoothed_scores(
        probability.reshape(count, levels * queries),
        xyz,
        k=10,
        chunk_size=int(chunk_size),
        valid_mask=valid,
    ).reshape(count, levels, queries)
    valid_cpu = torch.as_tensor(valid).bool().cpu().contiguous()
    peaks = smoothed[valid_cpu].amax(dim=0)
    selected = peaks.argmax(dim=0)
    remapped = v2.frozen.vala_minmax_remap_scores(
        smoothed.reshape(count, levels * queries),
        valid_mask=valid_cpu,
    ).reshape(count, levels, queries)
    gather = selected.view(1, 1, queries).expand(count, 1, queries)
    final = remapped.gather(1, gather).squeeze(1).contiguous()
    return O0Readout(
        final_scores=final,
        scores_by_scale=remapped.contiguous(),
        selected_scale_indices=selected.long().contiguous(),
        raw_smoothed_peaks=peaks.float().contiguous(),
    )


def _rank256_top_tail(
    relevance: torch.Tensor,
    *,
    quantile: float,
    maximum_rows: int,
) -> torch.Tensor:
    """Return a deterministic rank-256 diagnostic; never a selection input."""

    values = torch.as_tensor(relevance).detach().float().cpu().contiguous()
    if values.ndim != 2 or values.numel() == 0:
        raise ValueError("rank-256 diagnostic axis differs")
    rows, queries = values.shape
    result = torch.zeros_like(values, dtype=torch.bool)
    canonical = torch.arange(rows, dtype=torch.long)
    for query in range(queries):
        cutoff = float(torch.quantile(values[:, query], float(quantile)))
        candidates = canonical[values[:, query] >= cutoff]
        ordered = sorted(
            candidates.tolist(), key=lambda row: (-float(values[row, query]), row)
        )[: int(maximum_rows)]
        if ordered:
            result[torch.tensor(ordered, dtype=torch.long), query] = True
    return result.contiguous()


def build_region_evidence(
    *,
    o0_scores: torch.Tensor,
    region_rows: torch.Tensor,
    core_mask: torch.Tensor,
    primitive_valid: torch.Tensor,
    canonical_region_indices: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_probabilities: torch.Tensor,
    pair_features: torch.Tensor,
    pair_feature_median: torch.Tensor,
    pair_feature_robust_scale: torch.Tensor,
    descriptor_reliability: torch.Tensor,
    descriptor_active: torch.Tensor,
    descriptor_ood: torch.Tensor,
    rank256_relevance: torch.Tensor,
    config: DeploymentCalibration,
) -> RegionEvidence:
    """Construct direct two-anchor graph evidence and source-fixed gates."""

    scores = torch.as_tensor(o0_scores).detach().float().cpu().contiguous()
    rows = torch.as_tensor(region_rows).detach().long().cpu().contiguous()
    core = torch.as_tensor(core_mask).detach().bool().cpu().contiguous()
    valid = torch.as_tensor(primitive_valid).detach().bool().cpu().contiguous()
    canonical = (
        torch.as_tensor(canonical_region_indices).detach().long().cpu().contiguous()
    )
    reliability = (
        torch.as_tensor(descriptor_reliability).detach().float().cpu().contiguous()
    )
    active = torch.as_tensor(descriptor_active).detach().bool().cpu().contiguous()
    ood = torch.as_tensor(descriptor_ood).detach().bool().cpu().contiguous()
    rank = torch.as_tensor(rank256_relevance).detach().float().cpu().contiguous()
    if (
        scores.ndim != 2
        or rows.ndim != 2
        or core.shape != rows.shape
        or valid.shape != (scores.shape[0],)
        or canonical.shape != (rows.shape[0],)
        or reliability.shape != canonical.shape
        or active.shape != canonical.shape
        or ood.shape != canonical.shape
        or rank.shape != (rows.shape[0], scores.shape[1])
        or not bool(torch.isfinite(scores).all())
        or not bool(torch.isfinite(reliability).all())
        or bool((rows[core] < 0).any())
        or bool((rows[core] >= scores.shape[0]).any())
        or not bool(core.any(dim=1).all())
    ):
        raise ValueError("O0/region evidence axes differ")
    safe = rows.clamp(min=0, max=scores.shape[0] - 1)
    usable = core & valid[safe]
    denominator = usable.sum(dim=1).clamp_min(1).float()
    support_score = scores[safe]
    positive = (support_score > O0_SELECTION_THRESHOLD) & usable[:, :, None]
    positive_fraction = positive.sum(dim=1).float() / denominator[:, None]
    anchor = (
        (positive_fraction >= float(config.o0_supermajority_fraction))
        & active[:, None]
        & (~ood[:, None])
    )

    adjacency = thresholded_adjacency(
        region_count=int(rows.shape[0]),
        pair_indices=pair_indices,
        pair_probabilities=pair_probabilities,
        threshold=float(config.raw_edge_probability_minimum),
    )
    pair_feature_values = torch.as_tensor(pair_features).detach().float().cpu()
    if (
        pair_feature_values.ndim != 2
        or pair_feature_values.shape[0]
        != torch.as_tensor(pair_probabilities).numel()
        or pair_feature_values.shape[1] <= 18
        or not bool(torch.isfinite(pair_feature_values).all())
    ):
        raise ValueError("target graph reliability feature axis differs")
    edge_reliability = pair_feature_values[:, [17, 18]].amin(dim=1)
    median = torch.as_tensor(pair_feature_median).detach().float().cpu()
    scale = torch.as_tensor(pair_feature_robust_scale).detach().float().cpu()
    if (
        median.shape != (pair_feature_values.shape[1],)
        or scale.shape != median.shape
        or not bool(torch.isfinite(median).all())
        or not bool(torch.isfinite(scale).all())
        or bool((scale <= 0.0).any())
    ):
        raise ValueError("source graph normalization axis differs")
    edge_ood_raw = ((pair_feature_values - median) / scale).abs().amax(dim=1)
    edge_ood_unit = edge_ood_raw / (
        edge_ood_raw + float(config.feature_ood_raw_limit)
    )
    direct_support = torch.zeros(
        (rows.shape[0], scores.shape[1]), dtype=torch.int64
    )
    best_probability = torch.zeros(
        (rows.shape[0], scores.shape[1]), dtype=torch.float32
    )
    best_reliability = torch.zeros_like(best_probability)
    best_ood = torch.ones_like(best_probability)
    for region, neighbors in enumerate(adjacency):
        for neighbor, probability, edge_id in neighbors:
            if (
                float(edge_reliability[int(edge_id)])
                < float(config.minimum_reliability)
                or float(edge_ood_unit[int(edge_id)])
                > float(config.maximum_feature_ood_score)
            ):
                continue
            supported_queries = anchor[neighbor]
            if not bool(supported_queries.any()):
                continue
            direct_support[region, supported_queries] += 1
            stronger = supported_queries & (
                float(probability) > best_probability[region]
            )
            best_probability[region, stronger] = float(probability)
            best_reliability[region, stronger] = edge_reliability[int(edge_id)]
            best_ood[region, stronger] = edge_ood_unit[int(edge_id)]
    eligible = (
        active
        & ~ood
        & usable.any(dim=1)
    )
    enough_anchors = anchor.sum(dim=0) >= int(config.anchor_quorum)
    candidate = (
        (direct_support >= 1)
        & (~anchor)
        & eligible[:, None]
        & enough_anchors[None, :]
    )
    lower = torch.zeros_like(scores[: rows.shape[0]], dtype=torch.float32)
    corrected = torch.sigmoid(
        torch.logit(
            best_probability.clamp(O0_LOGIT_CLAMP, 1.0 - O0_LOGIT_CLAMP)
        )
        - float(config.epsilon_logit)
    )
    if bool(
        (
            corrected[candidate]
            < float(config.calibrated_region_lower_minimum) - 1e-6
        ).any()
    ):
        raise RuntimeError("source-corrected target edge lower bound drifted")
    lower[candidate] = corrected[candidate]

    queries = int(scores.shape[1])
    diagnostic_reliability = torch.zeros(queries, dtype=torch.float32)
    diagnostic_ood = torch.ones(queries, dtype=torch.float32)
    anchor_agreement = torch.zeros(queries, dtype=torch.float32)
    # No target query-null bank exists.  False-edge nulls are already encoded
    # by the source-frozen sequential gain thresholds, so this diagnostic is
    # deliberately not applicable rather than a second target-derived gate.
    null_activation = torch.zeros(queries, dtype=torch.float32)
    stability = torch.zeros(queries, dtype=torch.float32)
    for query in range(queries):
        selected = candidate[:, query]
        if bool(selected.any()):
            diagnostic_reliability[query] = best_reliability[selected, query].amin()
            diagnostic_ood[query] = best_ood[selected, query].amax()
            anchor_agreement[query] = positive_fraction[anchor[:, query], query].amin()
            stability[query] = float(config.stability_required_fraction)
    query_gate = residual.source_fixed_query_gate(
        reliability=diagnostic_reliability,
        feature_ood_score=diagnostic_ood,
        anchor_agreement=anchor_agreement,
        null_activation=null_activation,
        stability=stability,
        config=config.residual_config(),
    ) & candidate.any(dim=0)
    top_tail = _rank256_top_tail(
        rank,
        quantile=float(config.query_top_tail_quantile),
        maximum_rows=int(config.top_tail_size),
    )
    return RegionEvidence(
        lower=lower.contiguous(),
        eligible=eligible.contiguous(),
        query_gate=query_gate.contiguous(),
        anchor_region=anchor.contiguous(),
        direct_anchor_support=direct_support.contiguous(),
        candidate_region=candidate.contiguous(),
        diagnostics={
            "reliability": diagnostic_reliability,
            "feature_ood_score": diagnostic_ood,
            "anchor_agreement": anchor_agreement,
            "null_activation": null_activation,
            "stability": stability,
            "descriptor_reliability_mean": torch.where(
                candidate.any(dim=0),
                (candidate.float() * reliability[:, None]).sum(dim=0)
                / candidate.sum(dim=0).clamp_min(1),
                torch.zeros(queries),
            ),
            "used_edge_ood_fraction_above_source_limit": torch.stack(
                [
                    (
                        (best_ood[candidate[:, query], query] > 0.5).float().mean()
                        if bool(candidate[:, query].any())
                        else torch.tensor(1.0)
                    )
                    for query in range(queries)
                ]
            ),
        },
        rank256_top_tail=top_tail,
    )


def _selected_scale_names(scale_ids: list[str], selected: torch.Tensor) -> list[str]:
    return [scale_ids[int(index)] for index in selected.tolist()]


def fuse_exact_o0_probabilities(
    o0_scores: torch.Tensor,
    residual_result: residual.O0AnchoredConformalResidualResult,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update only positive-residual entries; every other bit remains O0."""

    base = torch.as_tensor(o0_scores).detach()
    delta = torch.as_tensor(residual_result.residual_logits).detach()
    logits = torch.as_tensor(residual_result.fused_logits).detach()
    if (
        base.dtype != torch.float32
        or base.device.type != "cpu"
        or delta.shape != base.shape
        or logits.shape != base.shape
        or not bool(torch.isfinite(base).all())
        or bool((base < 0.0).any())
        or bool((base > 1.0).any())
    ):
        raise ValueError("O0 probability fusion inputs differ")
    changed = delta > 0.0
    result = base.clone()
    result[changed] = torch.sigmoid(logits[changed])
    if (
        not torch.equal(result[~changed], base[~changed])
        or not torch.equal(result[:, ~residual_result.query_gate], base[:, ~residual_result.query_gate])
    ):
        raise RuntimeError("O0 probability fallback changed")
    return result.contiguous(), changed.contiguous()


def _validate_rank256_binding(
    *,
    descriptor: Mapping[str, Any],
    descriptor_record: Mapping[str, str],
    relevance: Mapping[str, Any],
    feature: Mapping[str, Any],
    inference: Mapping[str, Any],
    source_gate: Mapping[str, Any],
) -> None:
    if (
        descriptor["source_variant"] != "v21b"
        or relevance["source_variant"] != "v21b"
        or descriptor["scene_id"] != relevance["scene_id"]
        or descriptor["scene_id"] != feature["scene_id"]
        or feature["scene_id"] != inference["scene_id"]
        or descriptor["physical_space_id"] != relevance["physical_space_id"]
        or descriptor["region_row_ids"] != relevance["region_row_ids"]
        or descriptor["region_fingerprints"] != relevance["region_fingerprints"]
        or descriptor["region_fingerprints"] != feature["region_fingerprints"]
        or relevance["input_authority"]["target_descriptor"]
        != dict(descriptor_record)
        or descriptor["input_authority"]["champion_checkpoint"]
        != source_gate["checkpoint"]
        or descriptor["input_authority"]["champion_normalization"]
        != source_gate["normalization_authority"]
        or not torch.equal(
            descriptor["canonical_region_indices"],
            relevance["canonical_region_indices"],
        )
        or not torch.equal(
            descriptor["canonical_region_indices"],
            feature["canonical_region_indices"],
        )
        or not torch.equal(
            feature["canonical_region_indices"],
            inference["canonical_region_indices"],
        )
        or not torch.equal(feature["pair_indices"], inference["pair_indices"])
    ):
        raise ValueError("rank-256 target/query/graph authority binding differs")


def _load_and_validate_execution(
    path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="O0 anchored residual execution authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "source_variant",
        "implementation",
        "dependencies",
        "source_result",
        "source_calibration",
        "positive_o0_cache",
        "negative_o0_cache",
        "rank256_target_descriptor",
        "rank256_query_relevance",
        "region_feature_authority",
        "region_inference_authority",
        "renderer_geometry_checkpoint",
        "knn_chunk_size",
        "output_cache",
        "output_report",
        "target_score_cache_authorized",
        "target_quality_execution_authorized",
        "access_audit",
    }
    authority = dict(raw)
    if (
        set(authority) != required
        or authority["schema"] != EXECUTION_SCHEMA
        or authority["schema_version"] != 1
        or authority["status"] != EXECUTION_STATUS
        or authority["source_variant"] != "v21b"
        or authority["target_score_cache_authorized"] is not True
        or authority["target_quality_execution_authorized"] is not False
        or not isinstance(authority["knn_chunk_size"], int)
        or int(authority["knn_chunk_size"]) <= 0
        or authority["access_audit"]
        != {
            "query_names_opened": True,
            "target_images_opened": False,
            "target_quality_data_opened": False,
            "target_quality_readout_executed": False,
        }
    ):
        raise ValueError("O0 anchored residual execution header differs")
    if (
        validate_file_record(authority["implementation"], label="implementation")
        != IMPLEMENTATION
    ):
        raise ValueError("O0 anchored residual implementation differs")
    dependencies = authority["dependencies"]
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(DEPENDENCIES):
        raise ValueError("O0 anchored residual dependency fields differ")
    for name, dependency in DEPENDENCIES.items():
        if validate_file_record(dependencies[name], label=name) != dependency:
            raise ValueError(f"O0 anchored residual dependency differs: {name}")

    source_record = _record_shape(authority["source_result"], name="source result")
    source_gate = champion.validate_champion_source(
        "v21b",
        source_record["path"],
        expected_sha256=source_record["sha256"],
    )
    calibration_record = _record_shape(
        authority["source_calibration"], name="source calibration"
    )
    calibration_path = validate_file_record(
        calibration_record, label="source calibration"
    )
    calibration_raw, calibration_sha, calibration_source = load_json_object(
        calibration_path,
        expected_sha256=calibration_record["sha256"],
        label="source graph calibration",
    )
    calibration, deployment = validate_source_calibration(
        calibration_raw,
    )
    fix3_execution_path = validate_file_record(
        calibration["execution_authority"],
        label="source FIX3 execution authority",
    )
    fix3_execution_raw, fix3_execution_sha, fix3_execution_source = load_json_object(
        fix3_execution_path,
        expected_sha256=calibration["execution_authority"]["sha256"],
        label="source FIX3 execution authority",
    )
    fix3_execution = fix3_calibration.validate_execution_authority(
        fix3_execution_raw
    )
    if (
        fix3_execution["fix2_result"] != calibration["supersedes_fix2_result"]
        or fix3_execution["implementation"] != file_record(
            Path(fix3_calibration.__file__).resolve()
        )
    ):
        raise ValueError("source FIX3 supersession chain differs")
    fix2_execution_raw, _, _ = load_json_object(
        fix3_execution["fix2_execution_authority"]["path"],
        expected_sha256=fix3_execution["fix2_execution_authority"]["sha256"],
        label="source FIX2 execution authority",
    )
    fix2_execution = fix2_calibration.validate_execution_authority(
        fix2_execution_raw
    )
    fix2_result_raw, _, _ = load_json_object(
        calibration["supersedes_fix2_result"]["path"],
        expected_sha256=calibration["supersedes_fix2_result"]["sha256"],
        label="superseded source FIX2 result",
    )
    if (
        fix2_result_raw.get("schema") != fix2_calibration.RESULT_SCHEMA
        or fix2_result_raw.get("execution_authority")
        != fix3_execution["fix2_execution_authority"]
        or fix2_result_raw.get("checkpoint") != fix2_execution["checkpoint"]
        or fix2_result_raw.get("source_access") != fix2_calibration.source_access()
        or fix2_result_raw.get("target_execution_performed") is not False
        or fix2_result_raw.get("content_authority_sha256")
        != canonical_json_sha256(
            {
                key: item
                for key, item in fix2_result_raw.items()
                if key != "content_authority_sha256"
            }
        )
    ):
        raise ValueError("source FIX3/FIX2 parent result binding differs")

    record_names = (
        "positive_o0_cache",
        "negative_o0_cache",
        "rank256_target_descriptor",
        "rank256_query_relevance",
        "region_feature_authority",
        "region_inference_authority",
        "renderer_geometry_checkpoint",
    )
    records: dict[str, dict[str, str]] = {}
    for name in record_names:
        record = _record_shape(authority[name], name=name)
        verified = validate_file_record(record, label=name)
        records[name] = {"path": str(verified), "sha256": record["sha256"]}
    output = _output_path(authority["output_cache"], name="output cache")
    report = _output_path(authority["output_report"], name="output report")
    if output == report:
        raise ValueError("O0 anchored residual outputs must differ")
    authority.update(records)
    authority.update(
        {
            "source_result": source_record,
            "source_calibration": {
                "path": str(calibration_source), "sha256": calibration_sha
            },
            "verified_source_gate": source_gate,
            "verified_calibration": calibration,
            "verified_calibration_execution": {
                "path": str(fix3_execution_source), "sha256": fix3_execution_sha
            },
            "graph_checkpoint": dict(fix2_execution["checkpoint"]),
            "deployment": deployment,
            "output_cache": output,
            "output_report": report,
            "verified_record": {"path": str(source), "sha256": digest},
        }
    )
    return authority


def run(args: argparse.Namespace) -> dict[str, Any]:
    execution = _load_and_validate_execution(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    output = Path(execution["output_cache"])
    report_path = Path(execution["output_report"])
    if output.exists() or output.is_symlink() or report_path.exists() or report_path.is_symlink():
        raise FileExistsError("O0 anchored residual outputs must be new")

    def load_record(name: str, description: str):
        record = execution[name]
        return load_torch_mapping(
            record["path"],
            expected_sha256=record["sha256"],
            map_location="cpu",
            label=description,
        )

    descriptor_raw, descriptor_sha, descriptor_path = load_record(
        "rank256_target_descriptor", "rank-256 target descriptor"
    )
    descriptor = champion.validate_target_descriptor(descriptor_raw)
    relevance_raw, relevance_sha, relevance_path = load_record(
        "rank256_query_relevance", "rank-256 query relevance"
    )
    relevance = champion.validate_query_relevance(relevance_raw)
    feature_raw, feature_sha, feature_path = load_record(
        "region_feature_authority", "target region features"
    )
    feature = validate_feature_authority(feature_raw)
    inference_raw, inference_sha, inference_path = load_record(
        "region_inference_authority", "target region inference"
    )
    inference = validate_inference_authority(inference_raw)
    _validate_rank256_binding(
        descriptor=descriptor,
        descriptor_record={"path": str(descriptor_path), "sha256": descriptor_sha},
        relevance=relevance,
        feature=feature,
        inference=inference,
        source_gate=execution["verified_source_gate"],
    )
    config = execution["deployment"]
    rule = inference["selected_rule"]
    if (
        not isinstance(rule, Mapping)
        or set(rule) != {"method", "maximum_regions", "threshold"}
        or int(rule["maximum_regions"]) != config.graph_safety_cap
        or float(rule["threshold"]) != config.raw_edge_probability_minimum
    ):
        raise ValueError("target graph rule differs from source calibration")
    if inference["checkpoint"] != execution["graph_checkpoint"]:
        raise ValueError("target graph checkpoint differs from source calibration")
    graph_checkpoint_raw, _, _ = load_torch_mapping(
        inference["checkpoint"]["path"],
        expected_sha256=inference["checkpoint"]["sha256"],
        map_location="cpu",
        label="source-promoted graph checkpoint",
    )
    graph_checkpoint = validate_checkpoint(graph_checkpoint_raw)

    accepted_record = feature["input_authority"]["accepted_v2"]
    accepted_path = validate_file_record(accepted_record, label="target AcceptedV2")
    accepted_raw, accepted_sha, accepted_source = load_torch_mapping(
        accepted_path,
        expected_sha256=accepted_record["sha256"],
        map_location="cpu",
        label="target AcceptedV2",
    )
    accepted = validate_target_accepted_v2_authority(accepted_raw)
    verified_accepted_record = {"path": str(accepted_source), "sha256": accepted_sha}
    v2.validate_renderer_geometry_binding(
        feature=feature,
        accepted=accepted,
        accepted_record=verified_accepted_record,
        renderer_geometry_checkpoint_sha256=execution[
            "renderer_geometry_checkpoint"
        ]["sha256"],
    )

    positive_raw, positive_sha, positive_path = load_record(
        "positive_o0_cache", "positive frozen O0 cache"
    )
    negative_raw, negative_sha, negative_path = load_record(
        "negative_o0_cache", "negative frozen O0 cache"
    )
    query_ids = tuple(str(item) for item in positive_raw.get("query_ids", ()))
    positive = v2.frozen.validate_ours_multiscale_query_score_cache(
        positive_raw,
        expected_xyz=torch.as_tensor(positive_raw.get("xyz")),
        expected_query_ids=query_ids,
        expected_renderer_geometry_checkpoint_sha256=execution[
            "renderer_geometry_checkpoint"
        ]["sha256"],
    )
    negative = v2.frozen.validate_ours_multiscale_query_score_cache(
        negative_raw,
        expected_xyz=positive_raw["xyz"],
        expected_query_ids=v2.frozen.NEGATIVE_PROMPTS,
        expected_renderer_geometry_checkpoint_sha256=execution[
            "renderer_geometry_checkpoint"
        ]["sha256"],
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
        if not bool(torch.equal(left, right) if torch.is_tensor(left) else left == right):
            raise ValueError(f"positive/negative frozen O0 cache {name} differs")
    if list(query_ids) != relevance["query_ids"]:
        raise ValueError("O0 and rank-256 query order differs")

    renderer_raw, renderer_sha, renderer_path = load_sha_bound_project_checkpoint_mapping(
        execution["renderer_geometry_checkpoint"]["path"],
        expected_sha256=execution["renderer_geometry_checkpoint"]["sha256"],
        map_location="cpu",
        label="renderer geometry checkpoint",
    )
    renderer_xyz = _renderer_checkpoint_xyz(renderer_raw)
    full_xyz = torch.as_tensor(positive_raw["xyz"]).float().cpu().contiguous()
    if renderer_sha != positive.renderer_geometry_checkpoint_sha256 or not torch.equal(
        renderer_xyz, full_xyz
    ):
        raise ValueError("renderer and frozen O0 geometry differ")

    graph_record = feature["input_authority"]["support_graph"]
    graph_path = validate_file_record(graph_record, label="target support graph")
    graph_raw, graph_sha, graph_source = load_torch_mapping(
        graph_path,
        expected_sha256=graph_record["sha256"],
        map_location="cpu",
        label="target support graph",
    )
    global_rows = torch.as_tensor(graph_raw["global_rows"]).long().cpu().contiguous()
    geometry_audit = v2.validate_support_geometry_binding(
        graph_xyz=graph_raw["xyz"],
        global_rows=global_rows,
        full_xyz=full_xyz,
        o0_valid=positive.valid,
    )
    v1.validate_scale_major_alignment(
        canonical_region_indices=accepted["canonical_region_indices"],
        scale_indices=accepted["scale_indices"],
        anchor_count=int(global_rows.numel()),
        o0_scale_radii_m=tuple(positive.scale_radii_m),
    )

    o0 = exact_o0_readout(
        positive_scores=positive.query_scores,
        negative_scores=negative.query_scores,
        xyz=full_xyz,
        valid=positive.valid,
        chunk_size=int(execution["knn_chunk_size"]),
    )
    evidence = build_region_evidence(
        o0_scores=o0.final_scores,
        region_rows=feature["region_rows"],
        core_mask=feature["token_mask"],
        primitive_valid=positive.valid,
        canonical_region_indices=feature["canonical_region_indices"],
        pair_indices=feature["pair_indices"],
        pair_probabilities=inference["pair_probabilities"],
        pair_features=feature["pair_features"],
        pair_feature_median=graph_checkpoint["normalization"]["median"],
        pair_feature_robust_scale=graph_checkpoint["normalization"]["robust_scale"],
        descriptor_reliability=descriptor["reliability_score"],
        descriptor_active=descriptor["active_update_mask"],
        descriptor_ood=descriptor["effective_ood_mask"],
        rank256_relevance=relevance["region_absolute_relevance"],
        config=config,
    )
    base_logits = torch.logit(o0.final_scores.clamp(O0_LOGIT_CLAMP, 1.0 - O0_LOGIT_CLAMP))
    fused = residual.o0_anchored_conformal_residual(
        o0_logits=base_logits.float().cpu().contiguous(),
        region_confidence_lower=evidence.lower,
        region_rows=feature["region_rows"],
        core_mask=feature["token_mask"],
        primitive_valid_mask=positive.valid,
        region_eligible_mask=evidence.eligible,
        canonical_region_indices=feature["canonical_region_indices"],
        query_gate=evidence.query_gate,
        config=config.residual_config(),
    )
    replayed = residual.o0_anchored_conformal_residual(
        o0_logits=base_logits.float().cpu().contiguous(),
        region_confidence_lower=evidence.lower,
        region_rows=feature["region_rows"],
        core_mask=feature["token_mask"],
        primitive_valid_mask=positive.valid,
        region_eligible_mask=evidence.eligible,
        canonical_region_indices=feature["canonical_region_indices"],
        query_gate=evidence.query_gate,
        config=config.residual_config(),
    )
    if (
        fused.selected_region_rows != replayed.selected_region_rows
        or fused.selected_canonical_region_indices
        != replayed.selected_canonical_region_indices
        or not torch.equal(fused.residual_logits, replayed.residual_logits)
        or not torch.equal(fused.fused_logits, replayed.fused_logits)
    ):
        raise RuntimeError("canonical residual replay stability failed")
    final_scores, changed = fuse_exact_o0_probabilities(o0.final_scores, fused)
    if (
        bool((final_scores < 0.0).any())
        or bool((final_scores > 1.0).any())
        or not torch.equal(final_scores[~changed], o0.final_scores[~changed])
        or not torch.equal(final_scores[:, ~evidence.query_gate], o0.final_scores[:, ~evidence.query_gate])
    ):
        raise RuntimeError("O0 anchored final-score invariant failed")

    tail_overlap = evidence.rank256_top_tail & evidence.candidate_region
    cache = {
        "schema": SCHEMA,
        "query_scores": final_scores.float().cpu().contiguous(),
        "valid": positive.valid.bool().cpu().contiguous(),
        "xyz": full_xyz,
        "metadata": {
            "query_names": list(query_ids),
            "score_semantics": "exact_O0_VALA_plus_source_fixed_nonnegative_direct_graph_residual",
            "canonical_capability": "exact_frozen_O0_canonical_negative_VALA_peak_scale",
            "o0_selection_threshold": O0_SELECTION_THRESHOLD,
            "source_calibration": execution["source_calibration"],
            "source_result": execution["source_result"],
            "residual_contract_sha256": residual.CONTRACT_SHA256,
            "target_descriptor": {"path": str(descriptor_path), "sha256": descriptor_sha},
            "query_relevance_diagnostic": {"path": str(relevance_path), "sha256": relevance_sha},
            "region_features": {"path": str(feature_path), "sha256": feature_sha},
            "region_inference": {"path": str(inference_path), "sha256": inference_sha},
            "positive_o0_cache": {"path": str(positive_path), "sha256": positive_sha},
            "negative_o0_cache": {"path": str(negative_path), "sha256": negative_sha},
            "renderer_geometry_checkpoint": {"path": str(renderer_path), "sha256": renderer_sha},
            "support_graph": {"path": str(graph_source), "sha256": graph_sha},
            "execution_authority": execution["verified_record"],
            "producer": file_record(IMPLEMENTATION),
        },
        "selection": {
            "selected_scale_indices": o0.selected_scale_indices,
            "anchor_region_counts": evidence.anchor_region.sum(dim=0).long(),
            "candidate_region_counts": evidence.candidate_region.sum(dim=0).long(),
            "query_gate": evidence.query_gate,
            "selected_region_rows": fused.selected_region_rows,
            "selected_canonical_region_indices": fused.selected_canonical_region_indices,
            "selected_lower_bounds": fused.selected_lower_bounds,
            "selected_gains": fused.selected_gains,
        },
    }
    written = write_torch_noclobber(output, cache)
    report = {
        "schema": SCHEMA,
        "status": "o0_anchored_graph_residual_cache_complete",
        "cache": file_record(written),
        "query_ids": list(query_ids),
        "selected_scale_ids": _selected_scale_names(
            list(positive.scale_ids), o0.selected_scale_indices
        ),
        "query_gate": evidence.query_gate.tolist(),
        "anchor_region_counts": evidence.anchor_region.sum(dim=0).tolist(),
        "candidate_region_counts": evidence.candidate_region.sum(dim=0).tolist(),
        "selected_region_counts": [len(item) for item in fused.selected_region_rows],
        "changed_primitive_counts": changed.sum(dim=0).tolist(),
        "rank256_top_tail_candidate_overlap_counts": tail_overlap.sum(dim=0).tolist(),
        "rank256_top_tail_role": "diagnostic_only_not_a_selection_input",
        "gate_diagnostics": {
            name: value.tolist() for name, value in evidence.diagnostics.items()
        },
        "geometry_audit": geometry_audit,
        "source_result": execution["source_result"],
        "source_calibration": execution["source_calibration"],
        "execution_authority": execution["verified_record"],
        "access_audit": execution["access_audit"],
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
