#!/usr/bin/env python3
"""Lift per-view native LERF language responses through exact MPR.

This is the direct-language attribution ceiling: the frozen official SigLIP2
summary head and text inner products are applied to every legal source view
before exact front-to-back marginal aggregation.  Only query sufficient
statistics are lifted; no target RGB, annotation, threshold or metric is read.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.rendering.sparse_marginal_authority import load_sparse_exact_marginal_authority
from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
    accumulate_contribution_mean_channel_chunked,
    finalize_registered_mean_chunked,
)
from radio_gs.scripts.materialize_scannet_direct_language_score_cache import _load_backbone_frame
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_sha_bound_project_checkpoint_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


def _load_text_bank(path: str, digest: str, label: str) -> tuple[torch.Tensor, list[str], dict[str, str]]:
    payload, observed, source = load_sha_bound_project_checkpoint_mapping(
        path, expected_sha256=digest, map_location="cpu", label=label,
    )
    queries = payload.get("queries")
    embeddings = torch.as_tensor(payload.get("embeddings")).float()
    if (
        not isinstance(queries, list) or not queries
        or len(set(map(str, queries))) != len(queries)
        or embeddings.shape != (len(queries), 1536)
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError(f"{label} query axis differs")
    return F.normalize(embeddings, dim=1, eps=1e-8), list(map(str, queries)), {
        "path": str(source), "sha256": observed,
    }


def _load_score_fallback(path: str, digest: str, label: str) -> tuple[Mapping[str, object], dict[str, str]]:
    payload, observed, source = load_sha_bound_project_checkpoint_mapping(
        path, expected_sha256=digest, map_location="cpu", label=label,
    )
    return payload, {"path": str(source), "sha256": observed}


def materialize(args: argparse.Namespace) -> dict[str, object]:
    fallback_positive, fallback_positive_record = _load_score_fallback(
        args.fallback_positive_cache, args.expected_fallback_positive_cache_sha256,
        "LERF positive score fallback",
    )
    fallback_negative, fallback_negative_record = _load_score_fallback(
        args.fallback_negative_cache, args.expected_fallback_negative_cache_sha256,
        "LERF negative score fallback",
    )
    xyz = torch.as_tensor(fallback_positive.get("xyz")).detach().cpu().float().contiguous()
    negative_xyz = torch.as_tensor(fallback_negative.get("xyz")).detach().cpu().float().contiguous()
    positive_fallback_scores = torch.as_tensor(fallback_positive.get("query_scores")).float()
    negative_fallback_scores = torch.as_tensor(fallback_negative.get("query_scores")).float()
    fallback_valid = torch.as_tensor(fallback_positive.get("valid")).bool()
    negative_valid = torch.as_tensor(fallback_negative.get("valid")).bool()
    if (
        xyz.ndim != 2 or xyz.shape[1] != 3 or not torch.equal(xyz, negative_xyz)
        or positive_fallback_scores.ndim != 3 or negative_fallback_scores.ndim != 3
        or positive_fallback_scores.shape[:2] != (xyz.shape[0], 3)
        or negative_fallback_scores.shape[:2] != (xyz.shape[0], 3)
        or fallback_valid.shape != (xyz.shape[0],) or not torch.equal(fallback_valid, negative_valid)
        or args.fallback_scale_index not in (0, 1, 2)
    ):
        raise ValueError("LERF score fallback row or scale domain differs")

    manifest, manifest_sha, manifest_path = load_json_object(
        args.responsibility_authority,
        expected_sha256=args.expected_responsibility_authority_sha256,
        label="LERF exact marginal responsibility authority",
    )
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("LERF exact marginal metadata differs")
    if any(bool(metadata.get(key, False)) for key in (
        "benchmark_images_opened", "benchmark_masks_opened", "evaluation_rgb_opened",
    )):
        raise ValueError("LERF exact marginal authority opened forbidden target information")
    height, width = int(metadata.get("feature_height", -1)), int(metadata.get("feature_width", -1))
    assignments, _verified_sha, _verified_path = load_sparse_exact_marginal_authority(
        manifest_path, expected_metadata=metadata,
        expected_frame_indices=manifest["frame_indices"],
        num_gaussians=xyz.shape[0], num_pixels=height * width,
        expected_sha256=manifest_sha,
    )
    frame_indices = list(map(int, manifest["frame_indices"]))
    if height <= 0 or width <= 0 or len(assignments) != len(frame_indices):
        raise ValueError("LERF exact marginal view domain differs")

    positive, positive_queries, positive_record = _load_text_bank(
        args.positive_text_cache, args.expected_positive_text_cache_sha256,
        "LERF positive text bank",
    )
    negative, negative_queries, negative_record = _load_text_bank(
        args.negative_text_cache, args.expected_negative_text_cache_sha256,
        "LERF canonical-negative text bank",
    )
    if (
        list(map(str, fallback_positive.get("query_ids", []))) != positive_queries
        or list(map(str, fallback_negative.get("query_ids", []))) != negative_queries
        or positive_fallback_scores.shape[2] != len(positive_queries)
        or negative_fallback_scores.shape[2] != len(negative_queries)
    ):
        raise ValueError("LERF direct and fallback query axes differ")
    text = torch.cat((positive, negative), dim=0).to(args.device)
    channels = text.shape[0]

    head_path = Path(args.summary_head_weights).expanduser().resolve(strict=True)
    head_record = file_record(head_path)
    if head_record["sha256"] != args.expected_summary_head_sha256:
        raise ValueError("SigLIP2 summary-head SHA256 differs")
    head = SigLIP2SummaryHead.from_extracted_weights(str(head_path)).to(args.device).eval()
    head.requires_grad_(False)

    registered_sum = torch.zeros(xyz.shape[0], channels, dtype=torch.float32)
    registered_count = torch.zeros(xyz.shape[0], dtype=torch.float32)
    sum_staging = torch.empty_like(registered_sum)
    count_staging = torch.empty_like(registered_count)
    feature_dir = Path(args.feature_dir).expanduser().resolve(strict=True)
    feature_manifest_record = file_record(feature_dir / "frame_manifest.json")
    device = torch.device(args.device)
    with torch.inference_mode():
        for frame_index, assignment in tqdm(
            zip(frame_indices, assignments), total=len(frame_indices),
            desc=f"{args.scene} direct language exact MPR",
        ):
            raw = _load_backbone_frame(
                feature_dir, frame_index, height=height, width=width,
            ).to(device=device, dtype=torch.float32)
            tokens = raw.permute(1, 2, 0).reshape(1, height * width, 1280)
            descriptor = F.normalize(head(tokens).squeeze(0).float(), dim=-1, eps=1e-8)
            score = (descriptor @ text.T).T.reshape(channels, height, width)
            accumulate_contribution_mean_channel_chunked(
                score, assignment["gaussian_ids"], assignment["pixel_ids"],
                assignment["marginal_weights"], registered_sum, registered_count,
                channel_chunk_size=channels, cpu_sum_staging=sum_staging,
                cpu_count_staging=count_staging,
            )
            del raw, tokens, descriptor, score
    direct, observed = finalize_registered_mean_chunked(
        registered_sum, registered_count, row_chunk_size=args.row_chunk_size,
    )
    direct = direct.float()
    positive_count = positive.shape[0]
    positive_score = positive_fallback_scores[:, args.fallback_scale_index].clone()
    negative_score = negative_fallback_scores[:, args.fallback_scale_index].clone()
    positive_score[observed] = direct[observed, :positive_count]
    negative_score[observed] = direct[observed, positive_count:]
    valid = observed | fallback_valid
    positive_score[~valid] = 0
    negative_score[~valid] = 0

    output = Path(args.output).expanduser().resolve()
    payload = {
        "schema": "radio_gs.lerf_direct_language_score_cache.v1",
        "schema_version": 1,
        "scene": args.scene,
        "xyz": xyz,
        "valid": valid,
        "direct_observed": observed,
        "positive_query_scores": positive_score.contiguous(),
        "negative_query_scores": negative_score.contiguous(),
        "positive_query_ids": positive_queries,
        "negative_query_ids": negative_queries,
        "metadata": {
            "construction": (
                "per_view_frozen_siglip2_summary_head_then_text_response_then_"
                "exact_front_to_back_marginal_with_d512_totality_fallback"
            ),
            "query_independent": False,
            "evaluation_diagnostic_only": True,
            "postprocessing": "none",
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "evaluation_rgb_opened": False,
            "target_metrics_opened": False,
            "capability_before_mpr": True,
            "responsibility_authority": {"path": str(manifest_path), "sha256": manifest_sha},
            "totality_fallback": {
                "positive": fallback_positive_record,
                "negative": fallback_negative_record,
                "fixed_scale_index": args.fallback_scale_index,
            },
            "summary_head": head_record,
            "positive_text_bank": positive_record,
            "negative_text_bank": negative_record,
            "feature_frame_manifest": feature_manifest_record,
        },
    }
    write_torch_noclobber(output, payload)
    report = {
        "status": "complete",
        "scene": args.scene,
        "output": file_record(output),
        "rows": xyz.shape[0],
        "views": len(frame_indices),
        "positive_queries": positive_count,
        "negative_queries": negative.shape[0],
        "direct_observed_rows": int(observed.sum()),
        "direct_observed_fraction": float(observed.float().mean()),
        "valid_rows": int(valid.sum()),
        "valid_fraction": float(valid.float().mean()),
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--responsibility-authority", required=True)
    parser.add_argument("--expected-responsibility-authority-sha256", required=True)
    parser.add_argument("--summary-head-weights", required=True)
    parser.add_argument("--expected-summary-head-sha256", required=True)
    parser.add_argument("--positive-text-cache", required=True)
    parser.add_argument("--expected-positive-text-cache-sha256", required=True)
    parser.add_argument("--negative-text-cache", required=True)
    parser.add_argument("--expected-negative-text-cache-sha256", required=True)
    parser.add_argument("--fallback-positive-cache", required=True)
    parser.add_argument("--expected-fallback-positive-cache-sha256", required=True)
    parser.add_argument("--fallback-negative-cache", required=True)
    parser.add_argument("--expected-fallback-negative-cache-sha256", required=True)
    parser.add_argument("--fallback-scale-index", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--row-chunk-size", type=int, default=8192)
    print(json.dumps(materialize(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
