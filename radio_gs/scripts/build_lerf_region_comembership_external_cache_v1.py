#!/usr/bin/env python3
"""Build a gated LERF external cache from frozen RegionCoMembership inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.querying.region_comembership_readout import (
    connected_region_union_from_o0,
)
from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.scripts.infer_region_comembership_v1 import (
    SCHEMA as INFERENCE_SCHEMA,
    validate_feature_authority,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.lerf_region_comembership_external_scores.v1"


def validate_scale_major_alignment(
    *,
    canonical_region_indices: torch.Tensor,
    scale_indices: torch.Tensor,
    anchor_count: int,
    o0_scale_radii_m: tuple[float, ...],
) -> None:
    canonical = torch.as_tensor(canonical_region_indices).detach().long().cpu()
    scales = torch.as_tensor(scale_indices).detach().long().cpu()
    expected_radii = tuple(float(value) for value in SurfaceRegionContractV2().radii_m)
    if (
        int(anchor_count) <= 0
        or canonical.ndim != 1
        or scales.shape != canonical.shape
        or tuple(float(value) for value in o0_scale_radii_m) != expected_radii
        or bool((canonical < 0).any())
        or bool((canonical >= int(anchor_count) * len(expected_radii)).any())
        or not torch.equal(scales, canonical // int(anchor_count))
    ):
        raise ValueError("AcceptedV2 and O0 scale-major axis/radii differ")


def _validate_inference(
    payload: Mapping[str, Any], feature: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "scene_id",
        "domain",
        "producer",
        "feature_authority",
        "checkpoint",
        "probability_threshold",
        "source_access",
        "content_authority_sha256",
        "region_fingerprints",
        "canonical_region_indices",
        "pair_indices",
        "pair_probabilities",
        "accepted_edge_mask",
        "channel_sha256",
        "audit",
    }
    value = dict(payload)
    probability = torch.as_tensor(value.get("pair_probabilities"))
    selected = torch.as_tensor(value.get("accepted_edge_mask"))
    if (
        set(value) != required
        or value.get("schema") != INFERENCE_SCHEMA
        or value.get("schema_version") != 1
        or value.get("scene_id") != feature["scene_id"]
        or value.get("domain") != "target"
        or value.get("source_access") != feature["source_access"]
        or value.get("region_fingerprints") != feature["region_fingerprints"]
        or not torch.equal(
            value["canonical_region_indices"], feature["canonical_region_indices"]
        )
        or not torch.equal(value["pair_indices"], feature["pair_indices"])
        or probability.dtype != torch.float32
        or probability.shape != (feature["pair_indices"].shape[1],)
        or not bool(torch.isfinite(probability).all())
        or bool((probability < 0).any())
        or bool((probability > 1).any())
        or selected.dtype != torch.bool
        or selected.shape != probability.shape
        or not torch.equal(
            selected, probability >= float(value["probability_threshold"])
        )
    ):
        raise ValueError("RegionCoMembership target inference differs")
    validate_file_record(value["producer"], label="inference producer")
    validate_file_record(value["feature_authority"], label="feature authority")
    validate_file_record(value["checkpoint"], label="co-membership checkpoint")
    identity_keys = (
        "schema",
        "schema_version",
        "scene_id",
        "domain",
        "producer",
        "feature_authority",
        "checkpoint",
        "probability_threshold",
        "source_access",
    )
    if value["content_authority_sha256"] != canonical_json_sha256(
        {name: value[name] for name in identity_keys}
    ):
        raise ValueError("RegionCoMembership inference content identity changed")
    channel_names = (
        "canonical_region_indices",
        "pair_indices",
        "pair_probabilities",
        "accepted_edge_mask",
    )
    if set(value["channel_sha256"]) != set(channel_names):
        raise ValueError("RegionCoMembership inference channel mapping differs")
    for name in channel_names:
        if value["channel_sha256"].get(name) != tensor_sha256(value[name]):
            raise ValueError(f"RegionCoMembership inference channel changed: {name}")
    return value


def _scale_major_anchor_probability(
    remapped_probability: torch.Tensor, global_rows: torch.Tensor
) -> torch.Tensor:
    values = torch.as_tensor(remapped_probability).detach().float().cpu()
    rows = torch.as_tensor(global_rows).detach().long().cpu()
    if values.ndim != 3 or values.shape[1] != 3 or rows.ndim != 1:
        raise ValueError("O0 scale-major probability inputs differ")
    return values[rows].permute(1, 0, 2).reshape(-1, values.shape[2]).contiguous()


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_cache).expanduser().resolve()
    report_path = Path(args.output_report).expanduser().resolve()
    if (
        output.exists()
        or output.is_symlink()
        or report_path.exists()
        or report_path.is_symlink()
    ):
        raise FileExistsError("RegionCoMembership external cache outputs must be new")
    feature_raw, feature_sha, feature_path = load_torch_mapping(
        args.feature_authority,
        expected_sha256=args.expected_feature_authority_sha256,
        map_location="cpu",
        label="target RegionCoMembership feature authority",
    )
    feature = validate_feature_authority(feature_raw)
    if feature["domain"] != "target" or feature["target_execution_authority"] is None:
        raise ValueError(
            "external cache requires a gate-authorized target feature authority"
        )
    accepted_record = feature["input_authority"]["accepted_v2"]
    accepted_path = validate_file_record(
        accepted_record, label="feature AcceptedV2 authority"
    )
    accepted_raw, _, _ = load_torch_mapping(
        accepted_path,
        expected_sha256=accepted_record["sha256"],
        map_location="cpu",
        label="feature AcceptedV2 authority",
    )
    accepted = shard.validate_accepted_region_authority(accepted_raw)
    if (
        accepted["scene_id"] != feature["scene_id"]
        or not torch.equal(
            accepted["canonical_region_indices"],
            feature["canonical_region_indices"],
        )
        or not torch.equal(accepted["region_rows"], feature["region_rows"])
        or not torch.equal(accepted["token_mask"], feature["token_mask"])
        or accepted["region_fingerprints"] != feature["region_fingerprints"]
    ):
        raise ValueError("feature authority and AcceptedV2 canonical axis differ")
    inference_raw, inference_sha, inference_path = load_torch_mapping(
        args.inference_authority,
        expected_sha256=args.expected_inference_authority_sha256,
        map_location="cpu",
        label="target RegionCoMembership inference authority",
    )
    inference = _validate_inference(inference_raw, feature)
    if inference["feature_authority"] != {
        "path": str(feature_path),
        "sha256": feature_sha,
    }:
        raise ValueError("inference binds another feature authority")

    positive_payload, positive_sha, positive_path = load_torch_mapping(
        args.positive_cache,
        expected_sha256=args.expected_positive_cache_sha256,
        map_location="cpu",
        label="positive O0 cache",
    )
    negative_payload, negative_sha, negative_path = load_torch_mapping(
        args.negative_cache,
        expected_sha256=args.expected_negative_cache_sha256,
        map_location="cpu",
        label="canonical-negative O0 cache",
    )
    query_ids = tuple(str(value) for value in positive_payload["query_ids"])
    positive = frozen.validate_ours_multiscale_query_score_cache(
        positive_payload,
        expected_xyz=torch.as_tensor(positive_payload["xyz"]),
        expected_query_ids=query_ids,
        expected_renderer_geometry_checkpoint_sha256=(
            args.expected_renderer_geometry_checkpoint_sha256
        ),
    )
    negative = frozen.validate_ours_multiscale_query_score_cache(
        negative_payload,
        expected_xyz=positive_payload["xyz"],
        expected_query_ids=frozen.NEGATIVE_PROMPTS,
        expected_renderer_geometry_checkpoint_sha256=(
            args.expected_renderer_geometry_checkpoint_sha256
        ),
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
        if not bool(
            torch.equal(left, right) if torch.is_tensor(left) else left == right
        ):
            raise ValueError(f"positive/negative O0 cache {name} differs")
    probability = frozen.canonical_negative_relevancy_query_scores(
        positive.query_scores, negative.query_scores, logit_scale=10.0
    )
    count, scales, queries = probability.shape
    smoothed = frozen.vala_knn_smoothed_scores(
        probability.reshape(count, scales * queries),
        positive_payload["xyz"],
        k=10,
        chunk_size=args.knn_chunk_size,
        valid_mask=positive.valid,
    ).reshape(count, scales, queries)
    remapped = frozen.vala_minmax_remap_scores(
        smoothed.reshape(count, scales * queries), valid_mask=positive.valid
    ).reshape(count, scales, queries)

    graph_record = feature["input_authority"]["support_graph"]
    graph_path = validate_file_record(graph_record, label="feature support graph")
    graph, _, _ = load_torch_mapping(
        graph_path,
        expected_sha256=graph_record["sha256"],
        map_location="cpu",
        label="feature support graph",
    )
    global_rows = torch.as_tensor(graph["global_rows"]).long().cpu()
    full_xyz = torch.as_tensor(positive_payload["xyz"]).float().cpu()
    if not torch.equal(
        torch.as_tensor(graph["xyz"]).float(), full_xyz[global_rows]
    ) or not torch.equal(
        positive.valid,
        torch.zeros_like(positive.valid).index_fill_(0, global_rows, True),
    ):
        raise ValueError("O0 and AcceptedV2 support geometry differ")
    validate_scale_major_alignment(
        canonical_region_indices=accepted["canonical_region_indices"],
        scale_indices=accepted["scale_indices"],
        anchor_count=int(global_rows.numel()),
        o0_scale_radii_m=tuple(positive.scale_radii_m),
    )
    all_regions = _scale_major_anchor_probability(remapped, global_rows)
    canonical = feature["canonical_region_indices"].long()
    if int(canonical.max()) >= all_regions.shape[0]:
        raise ValueError("AcceptedV2 canonical indices exceed the O0 candidate axis")
    region_o0 = all_regions[canonical]
    readout = connected_region_union_from_o0(
        region_o0_scores=region_o0,
        pair_indices=feature["pair_indices"],
        pair_probabilities=inference["pair_probabilities"],
        probability_threshold=float(inference["probability_threshold"]),
        region_rows=feature["region_rows"],
        token_mask=feature["token_mask"],
        num_primitives=int(full_xyz.shape[0]),
    )
    cache = {
        "schema": SCHEMA,
        "query_scores": readout.primitive_membership,
        "valid": positive.valid.contiguous(),
        "xyz": full_xyz.contiguous(),
        "metadata": {
            "query_names": list(query_ids),
            "score_semantics": "binary_AcceptedV2_connected_region_union_membership",
            "candidate_axis": "AcceptedV2_canonical_4096",
            "seed_rule": "highest_frozen_O0_then_lowest_canonical_index",
            "pair_threshold": float(inference["probability_threshold"]),
            "feature_authority": {"path": str(feature_path), "sha256": feature_sha},
            "inference_authority": {
                "path": str(inference_path),
                "sha256": inference_sha,
            },
            "positive_cache": {"path": str(positive_path), "sha256": positive_sha},
            "negative_cache": {"path": str(negative_path), "sha256": negative_sha},
            "producer": file_record(Path(__file__).resolve()),
        },
        "selection": {
            "seed_region_indices": readout.seed_region_indices,
            "selected_region_masks": readout.selected_region_masks,
            "canonical_region_indices": canonical,
        },
    }
    written = write_torch_noclobber(output, cache)
    report = {
        "schema": SCHEMA,
        "status": "target_region_comembership_external_cache_complete",
        "cache": file_record(written),
        "query_ids": list(query_ids),
        "seed_region_indices": readout.seed_region_indices.tolist(),
        "selected_region_counts": readout.selected_region_masks.sum(dim=0).tolist(),
        "selected_primitive_counts": readout.primitive_membership.sum(dim=0)
        .int()
        .tolist(),
        "candidate_axis": "AcceptedV2_canonical_4096",
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_metric_computed": False,
    }
    write_frozen_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-authority", required=True)
    parser.add_argument("--expected-feature-authority-sha256", required=True)
    parser.add_argument("--inference-authority", required=True)
    parser.add_argument("--expected-inference-authority-sha256", required=True)
    parser.add_argument("--positive-cache", required=True)
    parser.add_argument("--expected-positive-cache-sha256", required=True)
    parser.add_argument("--negative-cache", required=True)
    parser.add_argument("--expected-negative-cache-sha256", required=True)
    parser.add_argument("--expected-renderer-geometry-checkpoint-sha256", required=True)
    parser.add_argument("--knn-chunk-size", type=int, default=65536)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--output-report", required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
