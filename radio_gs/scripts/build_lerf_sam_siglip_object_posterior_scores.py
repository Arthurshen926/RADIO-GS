#!/usr/bin/env python3
"""Build LERF typed text-object scores from official SAM3+SigLIP2 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.querying.sam_siglip_object_posterior import (
    sam_siglip_object_posterior,
)
from radio_gs.scripts.build_lerf_identity_seeded_object_topology_scores import (
    _query_names,
    _select_embedding_rows,
)
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    score_text_aligned_embeddings,
    vala_knn_minmax_scores,
)
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_sam_siglip_object_posterior_scores.v1"


def _xyz_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _score_embeddings(
    features: torch.Tensor,
    text: torch.Tensor,
    canonical: torch.Tensor,
    *,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    step = max(1, int(chunk_size))
    with torch.inference_mode():
        text_device = text.to(device)
        canonical_device = canonical.to(device)
        for start in range(0, int(features.shape[0]), step):
            chunks.append(
                score_text_aligned_embeddings(
                    F.normalize(features[start : start + step].float(), dim=-1).to(device),
                    text_device,
                    canonical_embeddings=canonical_device,
                    scoring="relevancy",
                    softmax_temperature=10.0,
                ).cpu()
            )
    return torch.cat(chunks) if chunks else torch.empty((0, text.shape[0]))


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"typed object score cache exists: {output}")
    query_names = _query_names(args.query_names)

    primitive_path = Path(args.primitive_query_cache).expanduser().resolve()
    primitive = torch.load(primitive_path, map_location="cpu", weights_only=False)
    primitive_metadata = dict(primitive.get("metadata", {}))
    xyz = torch.as_tensor(primitive.get("xyz")).float().cpu()
    features = torch.as_tensor(
        primitive.get("summary_features", primitive.get("features"))
    ).float().cpu()
    valid = torch.as_tensor(primitive.get("valid")).bool().cpu()
    xyz_sha = _xyz_sha256(xyz)
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or features.shape[0] != xyz.shape[0]
        or features.shape[1] != 1536
        or valid.shape != (xyz.shape[0],)
        or primitive_metadata.get("query_independent") is not True
        or any(
            bool(primitive_metadata.get(key, False))
            for key in (
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "benchmark_labels_opened",
                "text_queries_opened",
            )
        )
    ):
        raise ValueError("primitive Canonical Capability Feature cache differs")

    membership_path = Path(args.membership_cache).expanduser().resolve()
    membership = torch.load(membership_path, map_location="cpu", weights_only=False)
    membership_metadata = dict(membership.get("metadata", {}))
    num_proposals = int(membership.get("num_proposals", -1))
    if (
        membership.get("schema")
        != "radio_gs.lerf_multiscale_sam3_exact_mpr_memberships.v1"
        or int(membership.get("num_rows", -1)) != len(xyz)
        or membership_metadata.get("xyz_sha256") != xyz_sha
        or membership_metadata.get("query_independent_proposal_set") is not True
        or membership_metadata.get("query_independent_mask_hierarchy") is not True
        or membership_metadata.get("hierarchy_parent_edges_materialized") is not True
        or any(
            bool(membership_metadata.get(key, False))
            for key in (
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "evaluation_rgb_opened",
                "text_queries_opened",
            )
        )
    ):
        raise ValueError("query-independent multiscale exact-MPR membership differs")

    teacher_path = Path(args.proposal_teacher).expanduser().resolve()
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    teacher_metadata = dict(teacher.get("metadata", {}))
    descriptors = torch.as_tensor(teacher.get("descriptors")).float().cpu()
    context_descriptors = torch.as_tensor(teacher.get("context_descriptors")).float().cpu()
    if (
        teacher.get("schema")
        not in {
            "radio_gs.sam_mask_aligned_siglip2_spatial_teacher.v1",
            "radio_gs.multiscale_sam_mask_aligned_crop_summary_teacher.v1",
        }
        or descriptors.shape != (num_proposals, 1536)
        or context_descriptors.shape != descriptors.shape
        or not torch.equal(
            torch.as_tensor(teacher.get("proposal_view_indices")).long(),
            torch.as_tensor(membership.get("proposal_view_indices")).long(),
        )
        or teacher_metadata.get("teacher_space")
        not in {
            "official_siglip2_g_spatial_mask_aligned_pool",
            "official_siglip2_crop_summary",
        }
        or teacher_metadata.get("query_independent") is not True
        or teacher_metadata.get("source_only") is not True
        or teacher_metadata.get("official_sam3_decoder") is not True
        or any(
            bool(teacher_metadata.get(key, False))
            for key in (
                "benchmark_labels_opened",
                "benchmark_masks_opened",
                "benchmark_vocabulary_opened",
                "evaluation_rgb_opened",
                "text_queries_opened",
            )
        )
    ):
        raise ValueError("mask-aligned official SigLIP2 proposal teacher differs")

    text_path = Path(args.text_embedding_cache).expanduser().resolve()
    canonical_path = Path(args.canonical_embedding_cache).expanduser().resolve()
    text_payload = torch.load(text_path, map_location="cpu", weights_only=False)
    canonical_payload = torch.load(canonical_path, map_location="cpu", weights_only=False)
    text = F.normalize(_select_embedding_rows(text_payload, query_names), dim=-1)
    canonical = F.normalize(
        torch.as_tensor(canonical_payload.get("embeddings")).float(), dim=-1
    )
    if text.shape[1] != 1536 or canonical.shape[1] != 1536:
        raise ValueError("official SigLIP2 text embedding dimension differs")

    device = torch.device(args.device)
    raw_base = _score_embeddings(
        features,
        text,
        canonical,
        device=device,
        chunk_size=int(args.chunk_size),
    )
    raw_base[~valid] = -1.0e4
    base_scores = vala_knn_minmax_scores(
        raw_base,
        xyz,
        k=10,
        chunk_size=int(args.knn_chunk_size),
        valid_mask=valid,
    )
    base_scores[~valid] = -1.0e4
    identity_scores = raw_base.clone()
    identity_scores[~valid] = 0.0
    proposal_scores = _score_embeddings(
        descriptors,
        text,
        canonical,
        device=device,
        chunk_size=int(args.chunk_size),
    )
    context_scores = _score_embeddings(
        context_descriptors,
        text,
        canonical,
        device=device,
        chunk_size=int(args.chunk_size),
    )
    posterior, topology = sam_siglip_object_posterior(
        base_scores,
        torch.as_tensor(membership.get("row_indices")),
        torch.as_tensor(membership.get("proposal_indices")),
        torch.as_tensor(membership.get("weights")),
        torch.as_tensor(membership.get("proposal_view_indices")),
        torch.as_tensor(membership.get("proposal_parent_index")),
        proposal_scores,
        context_scores,
        proposal_quality=torch.as_tensor(membership.get("proposal_scores")),
        proposal_area_fraction=torch.as_tensor(
            membership.get("proposal_area_fraction")
        ),
        positive_core_ratio=float(args.positive_core_ratio),
        minimum_object_views=int(args.minimum_object_views),
        maximum_object_views=int(args.maximum_object_views),
        view_identity_margin=float(args.view_identity_margin),
        minimum_descriptor_score=float(args.minimum_descriptor_score),
        descriptor_gate=str(args.descriptor_gate),
        descriptor_listwise_margin=float(args.descriptor_listwise_margin),
        parent_identity_tolerance=float(args.parent_identity_tolerance),
        parent_field_peak_ratio=float(args.parent_field_peak_ratio),
        extent_membership_floor=float(args.extent_membership_floor),
        composition=str(args.composition),
        association_mode=str(args.association_mode),
        candidates_per_view=int(args.candidates_per_view),
        maximum_proposal_area_fraction=float(args.maximum_proposal_area_fraction),
        minimum_cross_view_jaccard=float(args.minimum_cross_view_jaccard),
        minimum_cross_view_overlap=float(args.minimum_cross_view_overlap),
        require_field_peak_anchor=not bool(args.allow_unanchored_component),
        latent_logit_temperature=float(args.latent_logit_temperature),
    )
    posterior[~valid] = -1.0e4
    payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene": str(args.scene),
        "query_scores": posterior.cpu(),
        # LERF evaluates two different random variables: where the named
        # object is (LocAcc) and which pixels/points belong to its full extent
        # (mIoU).  Keep the field identity posterior for localization instead
        # of asking the piecewise-flat SAM extent to carry an identity peak.
        "identity_query_scores": identity_scores.cpu(),
        "valid": valid,
        "xyz": xyz,
        "metadata": {
            "query_names": query_names,
            "query_family": "text_object_extent",
            "typed_posterior": "official_sam3_siglip2_identity_extent_factorization_v3",
            "localization_authority": "field_siglip2_relevancy_identity",
            "segmentation_authority": "sam_instance_extent_posterior",
            "separate_identity_localization": True,
            "score_input": "Universal_Field_v1_plus_query_free_source_region_hierarchy",
            "fixed_downstream_threshold": 0.6,
            "topology": topology,
            "primitive_query_cache": str(primitive_path),
            "primitive_query_cache_sha256": sha256_file(primitive_path),
            "membership_cache": str(membership_path),
            "membership_cache_sha256": sha256_file(membership_path),
            "proposal_teacher": str(teacher_path),
            "proposal_teacher_sha256": sha256_file(teacher_path),
            "text_embedding_cache": str(text_path),
            "text_embedding_cache_sha256": sha256_file(text_path),
            "canonical_embedding_cache": str(canonical_path),
            "canonical_embedding_cache_sha256": sha256_file(canonical_path),
            "xyz_sha256": xyz_sha,
            "persistent_second_semantic_field": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "evaluation_rgb_opened": False,
            "query_text_opened_at_readout": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "scene": str(args.scene),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "queries": len(query_names),
        "rows": len(xyz),
        "proposals": num_proposals,
        "topology": topology,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--primitive-query-cache", required=True)
    parser.add_argument("--membership-cache", required=True)
    parser.add_argument("--proposal-teacher", required=True)
    parser.add_argument("--text-embedding-cache", required=True)
    parser.add_argument("--canonical-embedding-cache", required=True)
    parser.add_argument("--query-names", required=True)
    parser.add_argument("--positive-core-ratio", type=float, default=0.80)
    parser.add_argument("--minimum-object-views", type=int, default=2)
    parser.add_argument("--maximum-object-views", type=int, default=12)
    parser.add_argument("--view-identity-margin", type=float, default=0.12)
    parser.add_argument("--minimum-descriptor-score", type=float, default=0.55)
    parser.add_argument(
        "--descriptor-gate",
        choices=("absolute", "query_listwise"),
        default="absolute",
    )
    parser.add_argument("--descriptor-listwise-margin", type=float, default=0.12)
    parser.add_argument("--parent-identity-tolerance", type=float, default=0.05)
    parser.add_argument("--parent-field-peak-ratio", type=float, default=0.75)
    parser.add_argument("--extent-membership-floor", type=float, default=0.50)
    parser.add_argument("--composition", choices=("maximum", "noisy_or"), default="maximum")
    parser.add_argument(
        "--association-mode",
        choices=("none", "weighted_jaccard_components", "latent_proposal_marginal"),
        default="weighted_jaccard_components",
    )
    parser.add_argument("--candidates-per-view", type=int, default=3)
    parser.add_argument("--maximum-proposal-area-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-cross-view-jaccard", type=float, default=0.02)
    parser.add_argument("--minimum-cross-view-overlap", type=float, default=0.15)
    parser.add_argument("--allow-unanchored-component", action="store_true")
    parser.add_argument("--latent-logit-temperature", type=float, default=8.0)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--knn-chunk-size", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    print(json.dumps(build(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
