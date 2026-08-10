#!/usr/bin/env python3
"""Run the promoted native-V3 relation with exact-anchor V2 fallback."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import region_comembership_native_v3_target as formal
from radio_gs.models.region_comembership_native_v3 import RegionCoMembershipNativeV3
from radio_gs.scripts import infer_region_comembership_v2 as parent_inference
from radio_gs.scripts import train_source_region_comembership_native_v3 as trainer
from radio_gs.scripts.materialize_region_comembership_features_native_v3 import (
    validate_feature_authority,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    validate_file_record,
    write_torch_noclobber,
)


SCHEMA = formal.INFERENCE_SCHEMA
SCHEMA_VERSION = formal.SCHEMA_VERSION
INFERENCE_BATCH_SIZE = 65536
CHANNEL_NAMES = (
    "canonical_region_indices",
    "pair_indices",
    "native_pair_active_mask",
    "legacy_v2_fallback_pair_mask",
    "pair_probabilities",
    "accepted_edge_mask",
)
IDENTITY_NAMES = (
    "schema",
    "schema_version",
    "scene_id",
    "domain",
    "producer",
    "target_execution_authority",
    "feature_authority",
    "checkpoint",
    "parent_v2_inference_authority",
    "selected_rule",
    "fallback_selected_rule",
    "fallback_contract",
    "source_access",
    "region_fingerprints_sha256",
    "canonical_axis_sha256",
    "pair_axis_sha256",
    "inference_axis_sha256",
    "tensor_authority_sha256",
)


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {name: payload[name] for name in IDENTITY_NAMES}


def _inference_axis_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            "region_fingerprints_sha256": payload["region_fingerprints_sha256"],
            "canonical_region_indices_sha256": payload["channel_sha256"]
            ["canonical_region_indices"],
            "pair_indices_sha256": payload["channel_sha256"]["pair_indices"],
        }
    )


def exact_anchor_fallback_fusion(
    *,
    native_probability: torch.Tensor,
    parent_probability: torch.Tensor,
    parent_accepted_edge_mask: torch.Tensor,
    native_pair_active_mask: torch.Tensor,
    native_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    native = torch.as_tensor(native_probability).detach().float().cpu()
    parent_p = torch.as_tensor(parent_probability).detach().float().cpu()
    parent_edge = torch.as_tensor(parent_accepted_edge_mask).detach().bool().cpu()
    active = torch.as_tensor(native_pair_active_mask).detach().bool().cpu()
    if (
        native.shape != parent_p.shape
        or parent_edge.shape != parent_p.shape
        or active.shape != parent_p.shape
        or parent_p.ndim != 1
        or not bool(torch.isfinite(native).all())
        or not bool(torch.isfinite(parent_p).all())
        or bool((native < 0).any())
        or bool((native > 1).any())
        or bool((parent_p < 0).any())
        or bool((parent_p > 1).any())
        or not 0.0 <= float(native_threshold) <= 1.0
    ):
        raise ValueError("native V3 exact-anchor fusion inputs differ")
    probability = parent_p.clone()
    accepted = parent_edge.clone()
    probability[active] = native[active]
    accepted[active] = native[active] >= float(native_threshold)
    fallback = ~active
    if (
        not torch.equal(probability[fallback], parent_p[fallback])
        or not torch.equal(accepted[fallback], parent_edge[fallback])
    ):
        raise RuntimeError("native V3 exact-anchor fallback changed")
    return probability.contiguous(), accepted.contiguous()


def query_free_graph_health(
    *,
    region_count: int,
    pair_indices: torch.Tensor,
    probability: torch.Tensor,
    accepted_edge_mask: torch.Tensor,
    native_pair_active_mask: torch.Tensor,
    parent_accepted_edge_mask: torch.Tensor,
) -> dict[str, Any]:
    regions = int(region_count)
    pairs = torch.as_tensor(pair_indices).detach().long().cpu()
    probability = torch.as_tensor(probability).detach().float().cpu()
    accepted = torch.as_tensor(accepted_edge_mask).detach().bool().cpu()
    active = torch.as_tensor(native_pair_active_mask).detach().bool().cpu()
    parent = torch.as_tensor(parent_accepted_edge_mask).detach().bool().cpu()
    if (
        regions <= 1
        or pairs.ndim != 2
        or pairs.shape[0] != 2
        or probability.shape != (pairs.shape[1],)
        or accepted.shape != probability.shape
        or active.shape != probability.shape
        or parent.shape != probability.shape
    ):
        raise ValueError("native V3 query-free health axes differ")
    degree = torch.zeros(regions, dtype=torch.int64)
    endpoints = pairs[:, accepted].reshape(-1)
    if endpoints.numel() > 0:
        degree.scatter_add_(0, endpoints, torch.ones_like(endpoints))
    parents = list(range(regions))
    sizes = [1] * regions

    def root(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    for left, right in pairs[:, accepted].t().tolist():
        a, b = root(int(left)), root(int(right))
        if a == b:
            continue
        if sizes[a] < sizes[b]:
            a, b = b, a
        parents[b] = a
        sizes[a] += sizes[b]
    component_sizes: dict[int, int] = {}
    for index in range(regions):
        component_sizes[root(index)] = component_sizes.get(root(index), 0) + 1
    quantiles = torch.quantile(
        probability.double(), torch.tensor([0.05, 0.5, 0.95], dtype=torch.double)
    )
    return {
        "canonical_regions": regions,
        "candidate_pairs": int(probability.numel()),
        "native_active_pairs": int(active.sum()),
        "legacy_v2_fallback_pairs": int((~active).sum()),
        "accepted_edges": int(accepted.sum()),
        "native_active_accepted_edges": int((accepted & active).sum()),
        "fallback_accepted_edges": int((accepted & ~active).sum()),
        "edge_additions_vs_parent_v2": int((accepted & ~parent).sum()),
        "edge_removals_vs_parent_v2": int((~accepted & parent).sum()),
        "isolated_regions": int((degree == 0).sum()),
        "connected_components_including_isolates": len(component_sizes),
        "largest_component_regions": max(component_sizes.values()),
        "largest_component_fraction": max(component_sizes.values()) / regions,
        "mean_accepted_degree": float(degree.double().mean()),
        "maximum_accepted_degree": int(degree.max()),
        "probability_mean": float(probability.double().mean()),
        "probability_p05": float(quantiles[0]),
        "probability_p50": float(quantiles[1]),
        "probability_p95": float(quantiles[2]),
        "fallback_probability_bitwise_equal": True,
        "fallback_edge_decision_bitwise_equal": True,
        "query_readout_executed": False,
        "target_metric_computed": False,
    }


def infer_native_probabilities(
    feature: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> torch.Tensor:
    normalization = checkpoint["normalization"]
    model = RegionCoMembershipNativeV3(
        normalization["median"], normalization["robust_scale"]
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    probability = torch.zeros(feature["pair_features"].shape[0], dtype=torch.float32)
    active = feature["native_pair_active_mask"]
    selected = torch.nonzero(active, as_tuple=False).flatten()
    with torch.inference_mode():
        for start in range(0, int(selected.numel()), INFERENCE_BATCH_SIZE):
            rows = selected[start : start + INFERENCE_BATCH_SIZE]
            probability[rows] = model.probability(feature["pair_features"][rows])
    return probability.contiguous()


def validate_inference_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("native V3 target inference authority must be a mapping")
    payload = dict(value)
    required = {
        *IDENTITY_NAMES,
        "content_authority_sha256",
        "region_fingerprints",
        *CHANNEL_NAMES,
        "channel_sha256",
        "query_free_health_topology_audit",
    }
    if (
        set(payload) != required
        or payload.get("schema") != SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("domain") != "target"
        or payload.get("fallback_contract") != formal.fallback_contract()
        or payload.get("source_access") != formal.access_audit(target_opened=True)
        or payload.get("content_authority_sha256")
        != canonical_json_sha256(_identity(payload))
    ):
        raise ValueError("native V3 target inference identity differs")
    validate_file_record(payload["producer"], label="native V3 inference producer")
    for name in (
        "target_execution_authority",
        "feature_authority",
        "checkpoint",
        "parent_v2_inference_authority",
    ):
        validate_file_record(payload[name], label=f"native V3 inference {name}")
    feature_raw, _, feature_path = load_torch_mapping(
        payload["feature_authority"]["path"],
        expected_sha256=payload["feature_authority"]["sha256"],
        map_location="cpu",
        label="native V3 inference feature authority",
    )
    feature = validate_feature_authority(feature_raw)
    checkpoint_raw, _, _ = load_torch_mapping(
        payload["checkpoint"]["path"],
        expected_sha256=payload["checkpoint"]["sha256"],
        map_location="cpu",
        label="native V3 inference checkpoint",
    )
    checkpoint = trainer.validate_checkpoint(checkpoint_raw)
    parent_raw, _, _ = load_torch_mapping(
        payload["parent_v2_inference_authority"]["path"],
        expected_sha256=payload["parent_v2_inference_authority"]["sha256"],
        map_location="cpu",
        label="native V3 inference parent V2 authority",
    )
    parent = parent_inference.validate_inference_authority(parent_raw)
    rule = payload.get("selected_rule")
    fallback_rule = payload.get("fallback_selected_rule")
    if (
        rule != checkpoint["selected_rule"]
        or fallback_rule != parent["selected_rule"]
        or payload["parent_v2_inference_authority"]
        != feature["input_authority"]["parent_v2_inference_authority"]
    ):
        raise ValueError("native V3 target inference rule/parent differs")
    canonical = torch.as_tensor(payload["canonical_region_indices"])
    pairs = torch.as_tensor(payload["pair_indices"])
    active = torch.as_tensor(payload["native_pair_active_mask"])
    fallback = torch.as_tensor(payload["legacy_v2_fallback_pair_mask"])
    probability = torch.as_tensor(payload["pair_probabilities"])
    accepted = torch.as_tensor(payload["accepted_edge_mask"])
    count = int(canonical.numel()) if canonical.ndim == 1 else -1
    if (
        count <= 1
        or canonical.dtype != torch.int64
        or pairs.dtype != torch.int64
        or pairs.ndim != 2
        or pairs.shape[0] != 2
        or active.dtype != torch.bool
        or active.shape != (pairs.shape[1],)
        or fallback.dtype != torch.bool
        or fallback.shape != active.shape
        or not torch.equal(fallback, ~active)
        or probability.dtype != torch.float32
        or probability.shape != active.shape
        or not bool(torch.isfinite(probability).all())
        or bool((probability < 0).any())
        or bool((probability > 1).any())
        or accepted.dtype != torch.bool
        or accepted.shape != active.shape
        or not torch.equal(probability[fallback], parent["pair_probabilities"][fallback])
        or not torch.equal(accepted[fallback], parent["accepted_edge_mask"][fallback])
        or not torch.equal(
            accepted[active],
            probability[active] >= float(rule["threshold"]),
        )
        or payload["region_fingerprints"] != feature["region_fingerprints"]
        or not torch.equal(canonical, feature["canonical_region_indices"])
        or not torch.equal(pairs, feature["pair_indices"])
        or not torch.equal(active, feature["native_pair_active_mask"])
    ):
        raise ValueError("native V3 target inference tensors differ")
    channels = payload.get("channel_sha256")
    if not isinstance(channels, Mapping) or set(channels) != set(CHANNEL_NAMES):
        raise ValueError("native V3 target inference channels differ")
    for name in CHANNEL_NAMES:
        if channels[name] != tensor_sha256(payload[name]):
            raise ValueError(f"native V3 target inference changed: {name}")
    health = query_free_graph_health(
        region_count=count,
        pair_indices=pairs,
        probability=probability,
        accepted_edge_mask=accepted,
        native_pair_active_mask=active,
        parent_accepted_edge_mask=parent["accepted_edge_mask"],
    )
    if (
        payload.get("query_free_health_topology_audit") != health
        or payload.get("region_fingerprints_sha256")
        != canonical_json_sha256(payload["region_fingerprints"])
        or payload.get("canonical_axis_sha256") != feature["canonical_axis_sha256"]
        or payload.get("pair_axis_sha256") != feature["pair_axis_sha256"]
        or payload.get("inference_axis_sha256") != _inference_axis_sha256(payload)
        or payload.get("tensor_authority_sha256")
        != canonical_json_sha256(payload["channel_sha256"])
    ):
        raise ValueError("native V3 target inference health/SHA axis differs")
    # Validate authority last here; run() validates it before opening feature data.
    formal.validate_target_execution_authority(
        payload["target_execution_authority"]["path"],
        expected_sha256=payload["target_execution_authority"]["sha256"],
        scene_id=str(payload["scene_id"]),
        expected_feature_output=feature_path,
    )
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"native V3 target inference exists: {output}")
    execution = formal.validate_target_execution_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        scene_id=str(args.scene_id),
        expected_feature_output=args.feature_authority,
        expected_inference_output=output,
    )
    feature_raw, feature_sha, feature_path = load_torch_mapping(
        args.feature_authority,
        expected_sha256=args.expected_feature_authority_sha256,
        map_location="cpu",
        label="native V3 target feature authority",
    )
    feature = validate_feature_authority(feature_raw)
    checkpoint_record = execution["verified_source_gate"]["checkpoint_record"]
    checkpoint_raw, checkpoint_sha, checkpoint_path = load_torch_mapping(
        checkpoint_record["path"],
        expected_sha256=checkpoint_record["sha256"],
        map_location="cpu",
        label="native V3 promoted checkpoint",
    )
    checkpoint = trainer.validate_checkpoint(checkpoint_raw)
    parent = execution["verified_parent_inference"]
    if (
        feature["target_execution_authority"] != execution["verified_record"]
        or feature["input_authority"] != execution["verified_target_inputs"]
        or checkpoint_sha != checkpoint_record["sha256"]
    ):
        raise ValueError("native V3 target inference authority chain differs")
    native_probability = infer_native_probabilities(feature, checkpoint)
    probability, accepted = exact_anchor_fallback_fusion(
        native_probability=native_probability,
        parent_probability=parent["pair_probabilities"],
        parent_accepted_edge_mask=parent["accepted_edge_mask"],
        native_pair_active_mask=feature["native_pair_active_mask"],
        native_threshold=float(checkpoint["selected_rule"]["threshold"]),
    )
    health = query_free_graph_health(
        region_count=int(feature["canonical_region_indices"].numel()),
        pair_indices=feature["pair_indices"],
        probability=probability,
        accepted_edge_mask=accepted,
        native_pair_active_mask=feature["native_pair_active_mask"],
        parent_accepted_edge_mask=parent["accepted_edge_mask"],
    )
    identity = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scene_id": str(args.scene_id),
        "domain": "target",
        "producer": file_record(Path(__file__).resolve()),
        "target_execution_authority": dict(execution["verified_record"]),
        "feature_authority": {"path": str(feature_path), "sha256": feature_sha},
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "parent_v2_inference_authority": dict(
            execution["verified_target_inputs"]["parent_v2_inference_authority"]
        ),
        "selected_rule": dict(checkpoint["selected_rule"]),
        "fallback_selected_rule": dict(parent["selected_rule"]),
        "fallback_contract": formal.fallback_contract(),
        "source_access": formal.access_audit(target_opened=True),
        "region_fingerprints_sha256": feature["region_fingerprints_sha256"],
        "canonical_axis_sha256": feature["canonical_axis_sha256"],
        "pair_axis_sha256": feature["pair_axis_sha256"],
        "inference_axis_sha256": "",
        "tensor_authority_sha256": "",
    }
    payload = {
        **identity,
        "content_authority_sha256": "",
        "region_fingerprints": list(feature["region_fingerprints"]),
        "canonical_region_indices": feature["canonical_region_indices"],
        "pair_indices": feature["pair_indices"],
        "native_pair_active_mask": feature["native_pair_active_mask"],
        "legacy_v2_fallback_pair_mask": feature[
            "legacy_v2_fallback_pair_mask"
        ],
        "pair_probabilities": probability,
        "accepted_edge_mask": accepted,
        "channel_sha256": {},
        "query_free_health_topology_audit": health,
    }
    payload["channel_sha256"] = {
        name: tensor_sha256(payload[name]) for name in CHANNEL_NAMES
    }
    payload["inference_axis_sha256"] = _inference_axis_sha256(payload)
    payload["tensor_authority_sha256"] = canonical_json_sha256(
        payload["channel_sha256"]
    )
    payload["content_authority_sha256"] = canonical_json_sha256(_identity(payload))
    validate_inference_authority(payload)
    written = write_torch_noclobber(output, payload)
    return {
        "status": "native_v3_query_free_target_inference_complete",
        "scene_id": payload["scene_id"],
        "output": file_record(written),
        "selected_rule": payload["selected_rule"],
        "fallback_selected_rule": payload["fallback_selected_rule"],
        "query_free_health_topology_audit": health,
        "query_executed": False,
        "target_metric_computed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    parser.add_argument("--feature-authority", required=True)
    parser.add_argument("--expected-feature-authority-sha256", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
