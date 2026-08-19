#!/usr/bin/env python3
"""Build a sealed LERF query-conditioned object-topology diagnostic cache.

This is deliberately *not* the query-independent P0 mask hierarchy.  The
current source cache contains official SAM3 masks prompted by the benchmark
query names.  It is legal source-only diagnostic evidence, but the output
metadata makes that limitation fail-visible.

No benchmark image, evaluation RGB, or benchmark mask is opened.  Query names
are an explicit query-time interface supplied by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.querying.identity_seeded_object_topology import (
    identity_seeded_object_topology_posterior,
    proposal_query_indices_from_names,
)
from radio_gs.models.sam3_proposal_registration import (
    fuse_scores_with_seeded_sam3_extent,
)
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    score_text_aligned_embeddings,
    vala_knn_minmax_scores,
)
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_identity_seeded_object_topology_scores.v1"
MEMBERSHIP_SCHEMAS = {
    "radio_gs.lerf_sam3_exact_mpr_memberships.v1": 1,
    "radio_gs.lerf_sam3_exact_mpr_memberships.v2": 2,
}


def _float32_rows_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _query_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names or len(names) != len(set(name.casefold() for name in names)):
        raise ValueError("query names must be non-empty and unique")
    return names


def _select_embedding_rows(payload: dict[str, Any], names: list[str]) -> torch.Tensor:
    cached_names = [str(value) for value in payload.get("queries", [])]
    embeddings = torch.as_tensor(payload.get("embeddings"), dtype=torch.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(cached_names):
        raise ValueError("text cache query/embedding rows differ")
    lookup = {name.casefold(): index for index, name in enumerate(cached_names)}
    missing = [name for name in names if name.casefold() not in lookup]
    if missing:
        raise ValueError(f"text cache misses query names: {missing}")
    return embeddings[[lookup[name.casefold()] for name in names]]


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"output already exists: {output}")
    query_names = _query_names(args.query_names)

    feature_path = Path(args.primitive_query_cache).expanduser().resolve()
    feature_payload = torch.load(feature_path, map_location="cpu")
    xyz = torch.as_tensor(feature_payload.get("xyz"), dtype=torch.float32)
    features = torch.as_tensor(
        feature_payload.get("summary_features", feature_payload.get("features")),
        dtype=torch.float32,
    )
    valid = torch.as_tensor(feature_payload.get("valid"), dtype=torch.bool)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("primitive cache xyz must have shape [N,3]")
    if features.ndim != 2 or features.shape[0] != xyz.shape[0]:
        raise ValueError("primitive features must be row-aligned with xyz")
    if valid.shape != (xyz.shape[0],) or not bool(valid.any()):
        raise ValueError("primitive valid mask differs")
    xyz_sha256 = _float32_rows_sha256(xyz)

    membership_path = Path(args.membership_cache).expanduser().resolve()
    membership_payload = torch.load(membership_path, map_location="cpu")
    metadata = dict(membership_payload.get("metadata", {}))
    if (
        membership_payload.get("schema") not in MEMBERSHIP_SCHEMAS
        or int(membership_payload.get("schema_version", -1))
        != MEMBERSHIP_SCHEMAS.get(str(membership_payload.get("schema", "")), -1)
        or str(membership_payload.get("scene", "")) != args.scene
        or int(membership_payload.get("num_rows", -1)) != int(xyz.shape[0])
        or str(metadata.get("xyz_sha256", "")) != xyz_sha256
    ):
        raise ValueError("membership cache scene or Gaussian rows differ")
    if any(
        bool(metadata.get(key, True))
        for key in ("benchmark_images_opened", "benchmark_masks_opened", "evaluation_rgb_opened")
    ):
        raise ValueError("membership cache did not preserve source-only access")
    proposal_names = membership_payload.get("proposal_query_names")
    if not isinstance(proposal_names, list):
        raise ValueError("membership cache lacks proposal query names")
    proposal_query_indices = proposal_query_indices_from_names(proposal_names, query_names)

    text_path = Path(args.text_embedding_cache).expanduser().resolve()
    canonical_path = Path(args.canonical_embedding_cache).expanduser().resolve()
    text_payload = torch.load(text_path, map_location="cpu")
    canonical_payload = torch.load(canonical_path, map_location="cpu")
    text = F.normalize(_select_embedding_rows(text_payload, query_names), dim=-1)
    canonical = F.normalize(
        torch.as_tensor(canonical_payload.get("embeddings"), dtype=torch.float32), dim=-1
    )
    if text.shape[1] != features.shape[1] or canonical.shape[1] != features.shape[1]:
        raise ValueError("text and primitive embedding dimensions differ")

    device = torch.device(args.device)
    score_chunks: list[torch.Tensor] = []
    step = max(1, int(args.chunk_size))
    with torch.inference_mode():
        text_device = text.to(device)
        canonical_device = canonical.to(device)
        for start in range(0, int(features.shape[0]), step):
            score_chunks.append(
                score_text_aligned_embeddings(
                    features[start : start + step].to(device),
                    text_device,
                    canonical_embeddings=canonical_device,
                    scoring="relevancy",
                    softmax_temperature=10.0,
                ).cpu()
            )
    raw_scores = torch.cat(score_chunks)
    raw_scores[~valid] = -1.0e4
    vala_scores = vala_knn_minmax_scores(
        raw_scores,
        xyz,
        k=10,
        chunk_size=int(args.knn_chunk_size),
        valid_mask=valid,
    )
    rows = torch.as_tensor(membership_payload.get("row_indices"), dtype=torch.long)
    proposals = torch.as_tensor(
        membership_payload.get("proposal_indices"), dtype=torch.long
    )
    membership_weights = torch.as_tensor(
        membership_payload.get("weights"), dtype=torch.float32
    )
    proposal_views = torch.as_tensor(
        membership_payload.get("proposal_view_indices"), dtype=torch.long
    )
    if args.posterior_mode == "legacy_seeded_residual":
        posterior, topology_stats = fuse_scores_with_seeded_sam3_extent(
            vala_scores,
            rows,
            proposals,
            membership_weights,
            proposal_views,
            proposal_query_indices=proposal_query_indices,
            alpha=float(args.legacy_alpha),
            proposal_mean_ratio=0.0,
            seed_support_ratio=float(args.seed_support_ratio),
            minimum_views=int(args.minimum_object_views),
            query_conditioned=True,
        )
        topology_stats["capability_track"] = (
            "probability_corrected_query_conditioned_legacy_residual_diagnostic"
        )
        topology_stats["query_independent_mask_hierarchy"] = False
    else:
        posterior, topology_stats = identity_seeded_object_topology_posterior(
            vala_scores,
            rows,
            proposals,
            membership_weights,
            proposal_views,
            proposal_query_indices,
            proposal_scores=(
                torch.as_tensor(
                    membership_payload.get("proposal_scores"), dtype=torch.float32
                )
                if membership_payload.get("proposal_scores") is not None
                else None
            ),
            seed_support_ratio=float(args.seed_support_ratio),
            identity_core_ratio=float(args.identity_core_ratio),
            minimum_object_views=int(args.minimum_object_views),
            minimum_row_views=int(args.minimum_row_views),
            extent_membership_floor=float(args.extent_membership_floor),
            sibling_exclusion_strength=float(args.sibling_exclusion_strength),
            unknown_policy=args.unknown_policy,
            membership_calibration=args.membership_calibration,
            use_proposal_quality=bool(args.use_proposal_quality),
        )
    posterior[~valid] = -1.0e4

    payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene": args.scene,
        "query_scores": posterior.cpu(),
        "valid": valid.cpu(),
        "xyz": xyz.cpu(),
        "metadata": {
            "query_names": query_names,
            "score_input": "method_v1_siglip2_relevancy_then_frozen_vala_knn10_minmax",
            "score_output": str(args.posterior_mode),
            "fixed_downstream_threshold": 0.6,
            "capability_track": "query_conditioned_source_sam_diagnostic_not_p0",
            "query_independent_mask_hierarchy": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "evaluation_rgb_opened": False,
            "primitive_query_cache": str(feature_path),
            "primitive_query_cache_sha256": sha256_file(feature_path),
            "membership_cache": str(membership_path),
            "membership_cache_sha256": sha256_file(membership_path),
            "text_embedding_cache": str(text_path),
            "text_embedding_cache_sha256": sha256_file(text_path),
            "canonical_embedding_cache": str(canonical_path),
            "canonical_embedding_cache_sha256": sha256_file(canonical_path),
            "xyz_sha256": xyz_sha256,
            "topology": topology_stats,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "scene": args.scene,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "capability_track": "query_conditioned_source_sam_diagnostic_not_p0",
        "query_independent_mask_hierarchy": False,
        "benchmark_masks_opened_for_construction": False,
        "evaluation_rgb_opened_for_construction": False,
        "topology": topology_stats,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--primitive-query-cache", required=True)
    parser.add_argument("--membership-cache", required=True)
    parser.add_argument("--text-embedding-cache", required=True)
    parser.add_argument("--canonical-embedding-cache", required=True)
    parser.add_argument("--query-names", required=True)
    parser.add_argument("--seed-support-ratio", type=float, default=0.80)
    parser.add_argument(
        "--posterior-mode",
        choices=("object_topology", "legacy_seeded_residual"),
        default="object_topology",
    )
    parser.add_argument("--legacy-alpha", type=float, default=0.25)
    parser.add_argument("--identity-core-ratio", type=float, default=0.80)
    parser.add_argument("--minimum-object-views", type=int, default=2)
    parser.add_argument("--minimum-row-views", type=int, default=1)
    parser.add_argument("--extent-membership-floor", type=float, default=0.50)
    parser.add_argument("--sibling-exclusion-strength", type=float, default=0.0)
    parser.add_argument(
        "--unknown-policy",
        choices=("preserve_text_prior", "negative_outside_topology"),
        default="preserve_text_prior",
    )
    parser.add_argument(
        "--membership-calibration",
        choices=("proposal_max", "pure_probability"),
        default="proposal_max",
    )
    parser.add_argument("--use-proposal-quality", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--knn-chunk-size", type=int, default=65536)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
