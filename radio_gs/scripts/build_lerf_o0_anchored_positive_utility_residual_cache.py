#!/usr/bin/env python3
"""Build a no-GT LERF cache with an honest positive-utility O0 residual.

The exact frozen O0 canonical-negative/VALA readout remains the canonical
primitive capability.  A source-promoted direct graph edge may add a bounded
non-negative residual only when the target query exposes the frozen O0 anchor
quorum and every reliability, pair-feature OOD, anchor-agreement, and replay
stability gate passes.  Selection is the positive-utility interface's fixed
strict-gain-above-zero, at-most-three-region policy.

This FIX4B entry point deliberately has no sequential-null, FWER, conformal,
target-quality, renderer-quality, or metric control.  It writes only a new
row-aligned score cache and an evidence report, both first-writer-only.
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

from radio_gs.interfaces import lerf_o0_anchored_positive_utility_residual as positive
from radio_gs.interfaces import surface_region_rank256_champion as champion
from radio_gs.interfaces.region_comembership_v2_formal import validate_checkpoint
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.querying.bounded_region_comembership_readout import (
    thresholded_adjacency,
)
from radio_gs.scripts import build_lerf_o0_anchored_graph_residual_cache as legacy
from radio_gs.scripts import build_lerf_region_comembership_external_cache_v1 as v1
from radio_gs.scripts import build_lerf_region_comembership_external_cache_v2 as v2
from radio_gs.scripts import audit_source_only_graph_expected_utility_fix4 as fix4_calibration
from radio_gs.scripts import calibrate_source_only_graph_confidence_v1 as fix2_calibration
from radio_gs.scripts import calibrate_source_only_graph_consumer_exact_fix3 as fix3_calibration
from radio_gs.scripts import finalize_source_only_graph_positive_utility_fix4b as fix4b_calibration
from radio_gs.scripts.infer_region_comembership_v2 import validate_inference_authority
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


SCHEMA = "radio_gs.lerf_o0_anchored_positive_utility_external_scores.v1"
EXECUTION_SCHEMA = "radio_gs.lerf_o0_anchored_positive_utility_execution.v1"
EXECUTION_STATUS = "authorized_source_fixed_positive_utility_target_score_cache_only"
IMPLEMENTATION = Path(__file__).resolve()
O0_SELECTION_THRESHOLD = legacy.O0_SELECTION_THRESHOLD
O0_LOGIT_CLAMP = legacy.O0_LOGIT_CLAMP
GRAPH_SAFETY_CAP = 8
TOP_TAIL_SIZE = 8
QUERY_TOP_TAIL_QUANTILE = 0.99
DEPENDENCIES = {
    "positive_utility_interface": Path(positive.__file__).resolve(),
    "legacy_exact_o0_helpers": Path(legacy.__file__).resolve(),
    "o0_frozen_readout": Path(v2.frozen.__file__).resolve(),
    "o0_target_binding": Path(v2.__file__).resolve(),
    "o0_scale_axis": Path(v1.__file__).resolve(),
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
    "source_calibration_fix4": Path(fix4_calibration.__file__).resolve(),
    "source_calibration_fix4b": Path(fix4b_calibration.__file__).resolve(),
}


@dataclass(frozen=True)
class PositiveUtilityDeployment:
    epsilon_logit: float
    novel_mass_reference: float
    minimum_reliability: float
    maximum_feature_ood_score: float
    minimum_anchor_agreement: float
    minimum_stability: float
    raw_edge_probability_minimum: float
    maximum_selected_regions: int
    anchor_quorum: int
    o0_supermajority_fraction: float
    o0_final_score_minimum: float
    feature_ood_raw_limit: float
    stability_required_fraction: float

    @property
    def calibrated_region_lower_minimum(self) -> float:
        probability = torch.tensor(self.raw_edge_probability_minimum)
        return float(torch.sigmoid(torch.logit(probability) - self.epsilon_logit))

    def residual_config(self) -> positive.SourceFixedPositiveUtilityConfig:
        return positive.SourceFixedPositiveUtilityConfig(
            epsilon_logit=self.epsilon_logit,
            novel_mass_reference=self.novel_mass_reference,
            minimum_reliability=self.minimum_reliability,
            maximum_feature_ood_score=self.maximum_feature_ood_score,
            minimum_anchor_agreement=self.minimum_anchor_agreement,
            minimum_stability=self.minimum_stability,
        )


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


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_deployment_config(value: object) -> PositiveUtilityDeployment:
    """Accept only the clean FIX4B deployment contract, without null fields."""

    required = {
        "interface",
        "edge_confidence_surrogate",
        "positive_utility_config",
        "selection",
        "query_gate",
        "anchor",
        "feature_OOD",
        "residual",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("FIX4B deployment fields differ")
    raw = dict(value)
    interface = raw["interface"]
    edge = raw["edge_confidence_surrogate"]
    utility = raw["positive_utility_config"]
    selection = raw["selection"]
    query_gate = raw["query_gate"]
    anchor = raw["anchor"]
    feature_ood = raw["feature_OOD"]
    residual_claim = raw["residual"]
    if (
        not isinstance(interface, Mapping)
        or set(interface) != {"schema", "contract_sha256"}
        or not isinstance(edge, Mapping)
        or set(edge)
        != {
            "raw_probability_minimum",
            "epsilon_logit",
            "lower_score_formula",
            "semantics",
        }
        or not isinstance(utility, Mapping)
        or set(utility)
        != {
            "epsilon_logit",
            "novel_mass_reference",
            "minimum_reliability",
            "maximum_feature_ood_score",
            "minimum_anchor_agreement",
            "minimum_stability",
        }
        or not isinstance(selection, Mapping)
        or set(selection)
        != {
            "minimum_positive_gain",
            "comparison",
            "maximum_selected_regions",
            "gain_formula",
            "tie_break",
        }
        or not isinstance(query_gate, Mapping)
        or set(query_gate) != {"inputs", "conjunction", "failed_gate"}
        or not isinstance(anchor, Mapping)
        or set(anchor) != {"O0_final_score_minimum", "valid_core_supermajority", "quorum"}
        or not isinstance(feature_ood, Mapping)
        or set(feature_ood) != {"raw_score_limit", "unit_score_maximum", "input"}
        or not isinstance(residual_claim, Mapping)
        or set(residual_claim)
        != {
            "per_primitive_logit_maximum",
            "sign",
            "aggregation",
            "canonical_capability",
        }
    ):
        raise ValueError("FIX4B deployment nested fields differ")
    config = PositiveUtilityDeployment(
        epsilon_logit=_finite(utility["epsilon_logit"], name="epsilon_logit"),
        novel_mass_reference=_finite(
            utility["novel_mass_reference"], name="novel_mass_reference"
        ),
        minimum_reliability=_finite(
            utility["minimum_reliability"], name="minimum_reliability"
        ),
        maximum_feature_ood_score=_finite(
            utility["maximum_feature_ood_score"],
            name="maximum_feature_ood_score",
        ),
        minimum_anchor_agreement=_finite(
            utility["minimum_anchor_agreement"],
            name="minimum_anchor_agreement",
        ),
        minimum_stability=_finite(
            utility["minimum_stability"], name="minimum_stability"
        ),
        raw_edge_probability_minimum=_finite(
            edge["raw_probability_minimum"],
            name="raw_edge_probability_minimum",
        ),
        maximum_selected_regions=int(selection["maximum_selected_regions"]),
        anchor_quorum=int(anchor["quorum"]),
        o0_supermajority_fraction=_finite(
            anchor["valid_core_supermajority"],
            name="o0_supermajority_fraction",
        ),
        o0_final_score_minimum=_finite(
            anchor["O0_final_score_minimum"], name="o0_final_score_minimum"
        ),
        feature_ood_raw_limit=_finite(
            feature_ood["raw_score_limit"], name="feature_ood_raw_limit"
        ),
        stability_required_fraction=_finite(
            utility["minimum_stability"], name="stability_required_fraction"
        ),
    )
    unit = (
        config.minimum_reliability,
        config.maximum_feature_ood_score,
        config.minimum_anchor_agreement,
        config.minimum_stability,
        config.raw_edge_probability_minimum,
        config.o0_supermajority_fraction,
        config.o0_final_score_minimum,
        config.stability_required_fraction,
    )
    if (
        any(not 0.0 <= item <= 1.0 for item in unit)
        or config.epsilon_logit < 0.0
        or config.novel_mass_reference <= 0.0
        or config.feature_ood_raw_limit <= 0.0
        or config.maximum_selected_regions != positive.MAXIMUM_REGIONS
        or config.anchor_quorum != fix2_calibration.ANCHOR_QUORUM
        or config.raw_edge_probability_minimum
        != fix2_calibration.RAW_EDGE_PROBABILITY_MINIMUM
        or config.novel_mass_reference != fix2_calibration.NOVEL_MASS_REFERENCE
        or config.o0_supermajority_fraction
        != fix2_calibration.O0_CORE_SUPERMAJORITY
        or config.o0_final_score_minimum != O0_SELECTION_THRESHOLD
        or interface
        != {"schema": positive.SCHEMA, "contract_sha256": positive.CONTRACT_SHA256}
        or float(edge["epsilon_logit"]) != config.epsilon_logit
        or edge["lower_score_formula"]
        != "sigmoid(logit(clamp(raw_edge_score,1e-7,1-1e-7))-epsilon_logit)"
        or edge["semantics"]
        != (
            "source_frozen_conservative_edge_confidence_surrogate_not_a_"
            "posterior_probability_or_conformal_or_FWER_guarantee"
        )
        or selection
        != {
            "minimum_positive_gain": 0.0,
            "comparison": "strictly_greater_than",
            "maximum_selected_regions": positive.MAXIMUM_REGIONS,
            "gain_formula": (
                "positive_part(2*lower_score-1)*unique_valid_marginal_primitives/256"
            ),
            "tie_break": "lower_canonical_region_index",
        }
        or query_gate
        != {
            "inputs": [
                "pair_feature_reliability",
                "pair_feature_OOD_score",
                "O0_anchor_agreement",
                "deterministic_replay_stability",
            ],
            "conjunction": True,
            "failed_gate": "bitwise_O0_only",
        }
        or feature_ood["input"]
        != "used_direct_edge_pair_features_exact_21_channel_order"
        or residual_claim
        != {
            "per_primitive_logit_maximum": config.epsilon_logit,
            "sign": "nonnegative_only",
            "aggregation": "pointwise_max",
            "canonical_capability": "exact_frozen_O0_primitive_logits",
        }
    ):
        raise ValueError("FIX4B positive-utility deployment contract differs")
    config.residual_config()
    return config


def validate_source_calibration(
    value: object,
) -> tuple[dict[str, Any], PositiveUtilityDeployment]:
    """Validate the promoted FIX4B result and reject old FIX3/FIX4 schemas."""

    required = {
        "schema",
        "schema_version",
        "status",
        "execution_authority",
        "parent_chain",
        "method_claim",
        "source_exact_consumer_audit",
        "promotion_gate",
        "deployment_config",
        "source_access",
        "benchmark_execution_authorized",
        "target_execution_performed",
        "content_authority_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("FIX4B source result fields differ")
    payload = dict(value)
    if (
        payload["schema"] != fix4b_calibration.RESULT_SCHEMA
        or payload["schema_version"] != 1
        or payload["status"]
        != "source_only_positive_utility_fix4b_promoted_target_unopened"
        or payload["method_claim"]
        != (
            "risk_bounded_positive_utility_soft_residual_using_a_source_frozen_"
            "conservative_edge_confidence_surrogate;not_posterior_not_"
            "conformal_not_FWER_and_not_hard_graph_completion"
        )
        or payload["source_access"] != fix4b_calibration.source_access()
        or payload["benchmark_execution_authorized"] is not False
        or payload["target_execution_performed"] is not False
        or payload["content_authority_sha256"]
        != canonical_json_sha256(
            {
                key: item
                for key, item in payload.items()
                if key != "content_authority_sha256"
            }
        )
    ):
        raise ValueError("FIX4B source result header differs")
    payload["execution_authority"] = legacy._record_shape(
        payload["execution_authority"], name="FIX4B execution authority"
    )
    parent_names = (
        "fix4_execution_authority",
        "fix4_result",
        "fix3_execution_authority",
        "fix3_result",
        "fix2_execution_authority",
        "fix2_result",
    )
    if not isinstance(payload["parent_chain"], Mapping) or set(
        payload["parent_chain"]
    ) != set(parent_names):
        raise ValueError("FIX4B parent chain fields differ")
    payload["parent_chain"] = {
        name: legacy._record_shape(
            payload["parent_chain"][name], name=f"FIX4B parent {name}"
        )
        for name in parent_names
    }
    gate = payload["promotion_gate"]
    expected_thresholds = {
        "minimum_every_validation_scene_Wilson95_lower": 0.95,
        "validation_pooled_marginal_weighted_signed_utility": (
            "strictly_greater_than_zero"
        ),
        "validation_true_pseudo_anchor_reach": "strictly_greater_than_fix3",
        "failure_action": "reject_fix4b_do_not_open_target",
    }
    expected_outcomes = {
        "both_validation_scene_Wilson95_lower_at_least_0.95": True,
        "validation_pooled_marginal_weighted_signed_utility_positive": True,
        "validation_true_pseudo_anchor_reach_strictly_exceeds_fix3": True,
        "passed": True,
    }
    if (
        not isinstance(gate, Mapping)
        or set(gate)
        != {
            "thresholds",
            "fix3_validation_true_pseudo_anchor_reach",
            "outcomes",
            "decision",
        }
        or gate["thresholds"] != expected_thresholds
        or gate["outcomes"] != expected_outcomes
        or gate["decision"] != "promote_source_only"
        or not math.isfinite(float(gate["fix3_validation_true_pseudo_anchor_reach"]))
    ):
        raise ValueError("FIX4B source promotion gate differs")
    audit = payload["source_exact_consumer_audit"]
    if (
        not isinstance(audit, Mapping)
        or set(audit) != {"per_scene", "by_split"}
        or not isinstance(audit["per_scene"], Mapping)
        or not isinstance(audit["by_split"], Mapping)
        or set(audit["by_split"]) != {"source_train", "source_validation"}
    ):
        raise ValueError("FIX4B source exact-consumer audit differs")
    validation = audit["by_split"]["source_validation"]
    signed = validation["marginal_weighted_signed_utility"]
    if (
        float(signed["marginal_weighted_signed_utility"]) <= 0.0
        or float(validation["true_pseudo_anchor_reach"])
        <= float(gate["fix3_validation_true_pseudo_anchor_reach"])
        or any(
            float(audit["per_scene"][scene]["selected_precision_Wilson95_lower"])
            < fix4b_calibration.MINIMUM_EVERY_VALIDATION_SCENE_WILSON_LOWER
            for scene in fix2_calibration.VALIDATION_SCENES
        )
    ):
        raise ValueError("FIX4B source promotion evidence failed")
    return payload, _validate_deployment_config(payload["deployment_config"])


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
    config: PositiveUtilityDeployment,
) -> RegionEvidence:
    """Construct direct-anchor evidence without any sequential-null gate."""

    scores = torch.as_tensor(o0_scores).detach().float().cpu().contiguous()
    rows = torch.as_tensor(region_rows).detach().long().cpu().contiguous()
    core = torch.as_tensor(core_mask).detach().bool().cpu().contiguous()
    valid = torch.as_tensor(primitive_valid).detach().bool().cpu().contiguous()
    canonical = torch.as_tensor(canonical_region_indices).detach().long().cpu().contiguous()
    reliability = torch.as_tensor(descriptor_reliability).detach().float().cpu().contiguous()
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
        raise ValueError("O0/region positive-utility evidence axes differ")
    safe = rows.clamp(min=0, max=scores.shape[0] - 1)
    usable = core & valid[safe]
    denominator = usable.sum(dim=1).clamp_min(1).float()
    positive_score = scores[safe]
    positive_mask = (
        (positive_score > float(config.o0_final_score_minimum))
        & usable[:, :, None]
    )
    positive_fraction = positive_mask.sum(dim=1).float() / denominator[:, None]
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
        or pair_feature_values.shape[0] != torch.as_tensor(pair_probabilities).numel()
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
    direct_support = torch.zeros((rows.shape[0], scores.shape[1]), dtype=torch.int64)
    best_probability = torch.zeros((rows.shape[0], scores.shape[1]), dtype=torch.float32)
    best_reliability = torch.zeros_like(best_probability)
    best_ood = torch.ones_like(best_probability)
    for region, neighbors in enumerate(adjacency):
        for neighbor, probability, edge_id in neighbors:
            if (
                float(edge_reliability[int(edge_id)]) < float(config.minimum_reliability)
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
    eligible = active & (~ood) & usable.any(dim=1)
    enough_anchors = anchor.sum(dim=0) >= int(config.anchor_quorum)
    candidate = (
        (direct_support >= 1)
        & (~anchor)
        & eligible[:, None]
        & enough_anchors[None, :]
    )
    lower = torch.zeros((rows.shape[0], scores.shape[1]), dtype=torch.float32)
    corrected = torch.sigmoid(
        torch.logit(best_probability.clamp(O0_LOGIT_CLAMP, 1.0 - O0_LOGIT_CLAMP))
        - float(config.epsilon_logit)
    )
    if bool(
        (
            corrected[candidate]
            < float(config.calibrated_region_lower_minimum) - 1e-6
        ).any()
    ):
        raise RuntimeError("source-corrected target edge lower score drifted")
    lower[candidate] = corrected[candidate]

    queries = int(scores.shape[1])
    diagnostic_reliability = torch.zeros(queries, dtype=torch.float32)
    diagnostic_ood = torch.ones(queries, dtype=torch.float32)
    anchor_agreement = torch.zeros(queries, dtype=torch.float32)
    stability = torch.zeros(queries, dtype=torch.float32)
    for query in range(queries):
        selected = candidate[:, query]
        if bool(selected.any()):
            diagnostic_reliability[query] = best_reliability[selected, query].amin()
            diagnostic_ood[query] = best_ood[selected, query].amax()
            anchor_agreement[query] = positive_fraction[anchor[:, query], query].amin()
            stability[query] = float(config.stability_required_fraction)
    query_gate = positive.source_fixed_query_gate(
        reliability=diagnostic_reliability,
        feature_ood_score=diagnostic_ood,
        anchor_agreement=anchor_agreement,
        stability=stability,
        config=config.residual_config(),
    ) & candidate.any(dim=0)
    top_tail = legacy._rank256_top_tail(
        rank,
        quantile=QUERY_TOP_TAIL_QUANTILE,
        maximum_rows=TOP_TAIL_SIZE,
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
                        (
                            best_ood[candidate[:, query], query]
                            > float(config.maximum_feature_ood_score)
                        ).float().mean()
                        if bool(candidate[:, query].any())
                        else torch.tensor(1.0)
                    )
                    for query in range(queries)
                ]
            ),
        },
        rank256_top_tail=top_tail,
    )


def fuse_exact_o0_probabilities(
    o0_scores: torch.Tensor,
    result: positive.O0AnchoredPositiveUtilityResidualResult,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update only positive-residual entries and preserve every other O0 bit."""

    base = torch.as_tensor(o0_scores).detach()
    delta = torch.as_tensor(result.residual_logits).detach()
    logits = torch.as_tensor(result.fused_logits).detach()
    if (
        base.dtype != torch.float32
        or base.device.type != "cpu"
        or delta.shape != base.shape
        or logits.shape != base.shape
        or not bool(torch.isfinite(base).all())
        or bool((base < 0.0).any())
        or bool((base > 1.0).any())
    ):
        raise ValueError("O0 positive-utility fusion inputs differ")
    changed = delta > 0.0
    fused = base.clone()
    fused[changed] = torch.sigmoid(logits[changed])
    if (
        not torch.equal(fused[~changed], base[~changed])
        or not torch.equal(fused[:, ~result.query_gate], base[:, ~result.query_gate])
    ):
        raise RuntimeError("positive-utility O0 probability fallback changed")
    return fused.contiguous(), changed.contiguous()


def _load_and_validate_execution(
    path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="positive-utility target execution authority",
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
        raise ValueError("positive-utility target execution header differs")
    if validate_file_record(authority["implementation"], label="implementation") != IMPLEMENTATION:
        raise ValueError("positive-utility target implementation differs")
    dependencies = authority["dependencies"]
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(DEPENDENCIES):
        raise ValueError("positive-utility target dependency fields differ")
    for name, dependency in DEPENDENCIES.items():
        if validate_file_record(dependencies[name], label=name) != dependency:
            raise ValueError(f"positive-utility target dependency differs: {name}")

    source_record = legacy._record_shape(authority["source_result"], name="source result")
    source_gate = champion.validate_champion_source(
        "v21b",
        source_record["path"],
        expected_sha256=source_record["sha256"],
    )
    calibration_record = legacy._record_shape(
        authority["source_calibration"], name="FIX4B source calibration"
    )
    calibration_path = validate_file_record(
        calibration_record, label="FIX4B source calibration"
    )
    calibration_raw, calibration_sha, calibration_source = load_json_object(
        calibration_path,
        expected_sha256=calibration_record["sha256"],
        label="FIX4B source calibration",
    )
    calibration, deployment = validate_source_calibration(calibration_raw)
    fix4b_execution_path = validate_file_record(
        calibration["execution_authority"], label="FIX4B execution authority"
    )
    fix4b_execution_raw, fix4b_execution_sha, fix4b_execution_source = load_json_object(
        fix4b_execution_path,
        expected_sha256=calibration["execution_authority"]["sha256"],
        label="FIX4B execution authority",
    )
    fix4b_execution = fix4b_calibration.validate_execution_authority(
        fix4b_execution_raw
    )
    if (
        fix4b_execution["implementation"]
        != file_record(Path(fix4b_calibration.__file__).resolve())
        or fix4b_execution["positive_utility_interface"]
        != file_record(Path(positive.__file__).resolve())
        or fix4b_execution["positive_utility_interface_contract_sha256"]
        != positive.CONTRACT_SHA256
        or any(
            fix4b_execution[name] != calibration["parent_chain"][name]
            for name in calibration["parent_chain"]
        )
    ):
        raise ValueError("FIX4B result/execution/parent binding differs")
    fix2_execution, _, _ = fix4b_calibration._validate_parent_chain(
        fix4b_execution
    )

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
        record = legacy._record_shape(authority[name], name=name)
        verified = validate_file_record(record, label=name)
        records[name] = {"path": str(verified), "sha256": record["sha256"]}
    output = legacy._output_path(authority["output_cache"], name="output cache")
    report = legacy._output_path(authority["output_report"], name="output report")
    if output == report:
        raise ValueError("positive-utility outputs must differ")
    authority.update(records)
    authority.update(
        {
            "source_result": source_record,
            "source_calibration": {
                "path": str(calibration_source),
                "sha256": calibration_sha,
            },
            "verified_source_gate": source_gate,
            "verified_calibration": calibration,
            "verified_calibration_execution": {
                "path": str(fix4b_execution_source),
                "sha256": fix4b_execution_sha,
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
    if (
        output.exists()
        or output.is_symlink()
        or report_path.exists()
        or report_path.is_symlink()
    ):
        raise FileExistsError("positive-utility outputs must be new")

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
    legacy._validate_rank256_binding(
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
        or int(rule["maximum_regions"]) != GRAPH_SAFETY_CAP
        or float(rule["threshold"]) != config.raw_edge_probability_minimum
    ):
        raise ValueError("target graph rule differs from FIX4B source contract")
    if inference["checkpoint"] != execution["graph_checkpoint"]:
        raise ValueError("target graph checkpoint differs from FIX4B parent chain")
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
    v2.validate_renderer_geometry_binding(
        feature=feature,
        accepted=accepted,
        accepted_record={"path": str(accepted_source), "sha256": accepted_sha},
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
    positive_cache = v2.frozen.validate_ours_multiscale_query_score_cache(
        positive_raw,
        expected_xyz=torch.as_tensor(positive_raw.get("xyz")),
        expected_query_ids=query_ids,
        expected_renderer_geometry_checkpoint_sha256=execution[
            "renderer_geometry_checkpoint"
        ]["sha256"],
    )
    negative_cache = v2.frozen.validate_ours_multiscale_query_score_cache(
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
        left, right = getattr(positive_cache, name), getattr(negative_cache, name)
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
    if (
        renderer_sha != positive_cache.renderer_geometry_checkpoint_sha256
        or not torch.equal(renderer_xyz, full_xyz)
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
        o0_valid=positive_cache.valid,
    )
    v1.validate_scale_major_alignment(
        canonical_region_indices=accepted["canonical_region_indices"],
        scale_indices=accepted["scale_indices"],
        anchor_count=int(global_rows.numel()),
        o0_scale_radii_m=tuple(positive_cache.scale_radii_m),
    )

    o0 = legacy.exact_o0_readout(
        positive_scores=positive_cache.query_scores,
        negative_scores=negative_cache.query_scores,
        xyz=full_xyz,
        valid=positive_cache.valid,
        chunk_size=int(execution["knn_chunk_size"]),
    )
    evidence = build_region_evidence(
        o0_scores=o0.final_scores,
        region_rows=feature["region_rows"],
        core_mask=feature["token_mask"],
        primitive_valid=positive_cache.valid,
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
    base_logits = torch.logit(
        o0.final_scores.clamp(O0_LOGIT_CLAMP, 1.0 - O0_LOGIT_CLAMP)
    )
    consumer_kwargs = {
        "o0_logits": base_logits.float().cpu().contiguous(),
        "region_confidence_lower": evidence.lower,
        "region_rows": feature["region_rows"],
        "core_mask": feature["token_mask"],
        "primitive_valid_mask": positive_cache.valid,
        "region_eligible_mask": evidence.eligible,
        "canonical_region_indices": feature["canonical_region_indices"],
        "query_gate": evidence.query_gate,
        "config": config.residual_config(),
    }
    fused = positive.o0_anchored_positive_utility_residual(**consumer_kwargs)
    replayed = positive.o0_anchored_positive_utility_residual(**consumer_kwargs)
    if (
        fused.selected_region_rows != replayed.selected_region_rows
        or fused.selected_canonical_region_indices
        != replayed.selected_canonical_region_indices
        or not torch.equal(fused.residual_logits, replayed.residual_logits)
        or not torch.equal(fused.fused_logits, replayed.fused_logits)
    ):
        raise RuntimeError("canonical positive-utility replay stability failed")
    final_scores, changed = fuse_exact_o0_probabilities(o0.final_scores, fused)

    primitive_count, query_count = changed.shape
    region_rows = torch.as_tensor(feature["region_rows"]).long().cpu()
    core_mask = torch.as_tensor(feature["token_mask"]).bool().cpu()
    safe_rows = region_rows.clamp(min=0, max=primitive_count - 1)
    valid_core = core_mask & positive_cache.valid[safe_rows]
    selected_union = torch.zeros_like(changed)
    for query, selected_rows in enumerate(fused.selected_region_rows):
        for row in selected_rows:
            selected_union[safe_rows[row, valid_core[row]], query] = True
    outside = ~selected_union
    outside_bitwise_o0 = torch.equal(
        final_scores[outside].view(torch.int32),
        o0.final_scores[outside].view(torch.int32),
    )
    failed_gate_bitwise_o0 = torch.equal(
        final_scores[:, ~evidence.query_gate].view(torch.int32),
        o0.final_scores[:, ~evidence.query_gate].view(torch.int32),
    )
    invalid_bitwise_o0 = torch.equal(
        final_scores[~positive_cache.valid].view(torch.int32),
        o0.final_scores[~positive_cache.valid].view(torch.int32),
    )
    if (
        bool((final_scores < 0.0).any())
        or bool((final_scores > 1.0).any())
        or bool((changed & ~selected_union).any())
        or not outside_bitwise_o0
        or not failed_gate_bitwise_o0
        or not invalid_bitwise_o0
    ):
        raise RuntimeError("positive-utility final-score invariant failed")

    tail_overlap = evidence.rank256_top_tail & evidence.candidate_region
    cache = {
        "schema": SCHEMA,
        "query_scores": final_scores.float().cpu().contiguous(),
        "valid": positive_cache.valid.bool().cpu().contiguous(),
        "xyz": full_xyz,
        "metadata": {
            "query_names": list(query_ids),
            "score_semantics": "exact_O0_VALA_plus_source_fixed_positive_utility_residual",
            "canonical_capability": "exact_frozen_O0_canonical_negative_VALA_peak_scale",
            "o0_selection_threshold": O0_SELECTION_THRESHOLD,
            "source_calibration": execution["source_calibration"],
            "source_result": execution["source_result"],
            "positive_utility_contract_sha256": positive.CONTRACT_SHA256,
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
            "selected_lower_scores": fused.selected_lower_scores,
            "selected_gains": fused.selected_gains,
            "selected_marginal_primitives": fused.selected_marginal_primitives,
        },
    }
    written = write_torch_noclobber(output, cache)
    report = {
        "schema": SCHEMA,
        "status": "o0_anchored_positive_utility_cache_complete",
        "cache": file_record(written),
        "query_ids": list(query_ids),
        "selected_scale_ids": legacy._selected_scale_names(
            list(positive_cache.scale_ids), o0.selected_scale_indices
        ),
        "query_gate": evidence.query_gate.tolist(),
        "anchor_region_counts": evidence.anchor_region.sum(dim=0).tolist(),
        "candidate_region_counts": evidence.candidate_region.sum(dim=0).tolist(),
        "selected_region_counts": [len(item) for item in fused.selected_region_rows],
        "selected_marginal_primitives": [list(item) for item in fused.selected_marginal_primitives],
        "changed_primitive_counts": changed.sum(dim=0).tolist(),
        "changed_primitive_total": int(changed.sum()),
        "rank256_top_tail_candidate_overlap_counts": tail_overlap.sum(dim=0).tolist(),
        "rank256_top_tail_role": "diagnostic_only_not_a_selection_input",
        "gate_diagnostics": {
            name: value.tolist() for name, value in evidence.diagnostics.items()
        },
        "bitwise_invariants": {
            "outside_selected_region_union_is_exact_O0": outside_bitwise_o0,
            "failed_query_gate_is_exact_O0": failed_gate_bitwise_o0,
            "invalid_primitive_is_exact_O0": invalid_bitwise_o0,
            "changed_outside_selected_region_union_count": int(
                (changed & ~selected_union).sum()
            ),
        },
        "geometry_audit": geometry_audit,
        "source_result": execution["source_result"],
        "source_calibration": execution["source_calibration"],
        "positive_utility_interface": file_record(Path(positive.__file__).resolve()),
        "positive_utility_contract_sha256": positive.CONTRACT_SHA256,
        "execution_authority": execution["verified_record"],
        "access_audit": execution["access_audit"],
    }
    if "null_activation" in report["gate_diagnostics"]:
        raise RuntimeError("FIX4B report exposed a forbidden null gate")
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
