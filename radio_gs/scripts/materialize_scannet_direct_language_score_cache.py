#!/usr/bin/env python3
"""Materialize a ScanNet direct-language exact-MPR attribution cache.

For each source view this control applies the frozen RADIO SigLIP2 summary
head before lifting, computes the frozen split19/15/10 text logits, and exact-
MPR aggregates only those sufficient statistics.  Since inner products and
the MPR weighted mean are linear, this produces exactly the class logits of a
full 1536-D direct-language aggregation (up to ordinary FP32 roundoff) while
avoiding 1536/44 times redundant accumulation.  Unobserved Gaussian rows use
the frozen Method-v1 D512 score bank so the evaluated row domain is unchanged.

This is deliberately a query-dependent evaluation diagnostic, never a
persistent Universal Field artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.rendering.sparse_marginal_authority import (
    load_sparse_exact_marginal_authority,
)
from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
    accumulate_contribution_mean_channel_chunked,
    finalize_registered_mean_chunked,
)
from radio_gs.utils.checkpoint_io import load_trusted_checkpoint
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_sha_bound_project_checkpoint_mapping,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


SPLITS = ("19", "15", "10")


def _load_text_banks(base: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    source = Path(base).expanduser().resolve()
    banks: dict[str, torch.Tensor] = {}
    records: dict[str, object] = {}
    for split in SPLITS:
        path = source.with_name(f"{source.stem}_split{split}.pt")
        payload, digest, resolved = load_sha_bound_project_checkpoint_mapping(
            path,
            expected_sha256=sha256_file(path),
            map_location="cpu",
            label=f"ScanNet split{split} text bank",
        )
        values = F.normalize(torch.as_tensor(payload.get("embeddings")).float(), dim=-1)
        queries = payload.get("queries")
        if not isinstance(queries, list) or values.shape != (int(split), 1536):
            raise ValueError(f"ScanNet split{split} text bank differs")
        banks[split] = values
        records[split] = {
            "path": str(resolved),
            "sha256": digest,
            "queries": [str(item) for item in queries],
        }
    return banks, records


def _load_backbone_frame(
    feature_dir: Path,
    frame_index: int,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    stem = f"rgb_{int(frame_index)}"
    commit_path = feature_dir / ".extract_frame_commits" / f"{stem}.json"
    commit, _digest, _source = load_json_object(commit_path, label="feature-frame commit")
    records = commit.get("tensors")
    if not isinstance(records, list):
        raise ValueError("feature-frame commit tensor list differs")
    expected = next(
        (item for item in records if item.get("relative_path") == f"backbone/{stem}.pt"),
        None,
    )
    if not isinstance(expected, dict):
        raise ValueError("feature-frame commit lacks backbone tensor")
    path = feature_dir / str(expected["relative_path"])
    if sha256_file(path) != str(expected.get("sha256", "")):
        raise ValueError("feature-frame backbone SHA256 differs")
    value = torch.as_tensor(load_trusted_checkpoint(path, map_location="cpu"))
    if value.shape != (1280, int(height), int(width)) or value.dtype != torch.float16:
        raise ValueError("feature-frame backbone tensor shape or dtype differs")
    return value


def materialize(args: argparse.Namespace) -> dict[str, object]:
    fallback, fallback_sha, fallback_path = load_sha_bound_project_checkpoint_mapping(
        args.d512_query_cache,
        expected_sha256=args.expected_d512_query_cache_sha256,
        map_location="cpu",
        label="ScanNet D512 primitive query fallback",
    )
    xyz = torch.as_tensor(fallback.get("xyz")).detach().cpu().float().contiguous()
    fallback_features = F.normalize(
        torch.as_tensor(
            fallback.get("summary_features", fallback.get("features"))
        ).detach().cpu().float(),
        dim=-1,
        eps=1e-8,
    )
    if xyz.ndim != 2 or xyz.shape[1] != 3 or fallback_features.shape != (xyz.shape[0], 1536):
        raise ValueError("D512 fallback row domain differs")

    manifest, manifest_sha, manifest_path = load_json_object(
        args.responsibility_authority,
        expected_sha256=args.expected_responsibility_authority_sha256,
        label="current exact marginal responsibility authority",
    )
    height = int(manifest.get("metadata", {}).get("feature_height", -1))
    width = int(manifest.get("metadata", {}).get("feature_width", -1))
    assignments, _verified_sha, _verified_path = load_sparse_exact_marginal_authority(
        manifest_path,
        expected_metadata=manifest["metadata"],
        expected_frame_indices=manifest["frame_indices"],
        num_gaussians=int(xyz.shape[0]),
        num_pixels=height * width,
        expected_sha256=manifest_sha,
    )
    frame_indices = [int(item) for item in manifest["frame_indices"]]
    if len(assignments) != len(frame_indices) or height <= 0 or width <= 0:
        raise ValueError("exact marginal view domain differs")

    head_path = Path(args.summary_head_weights).expanduser().resolve(strict=True)
    head_record = file_record(head_path)
    if head_record["sha256"] != args.expected_summary_head_sha256:
        raise ValueError("summary-head SHA256 differs")
    head = SigLIP2SummaryHead.from_extracted_weights(str(head_path)).to(args.device).eval()
    head.requires_grad_(False)
    text_banks, text_records = _load_text_banks(args.text_embedding_cache)
    bank = torch.cat([text_banks[split] for split in SPLITS], dim=0).to(args.device)
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for split in SPLITS:
        offsets[split] = (cursor, cursor + int(split))
        cursor += int(split)

    registered_sum = torch.zeros(xyz.shape[0], cursor, dtype=torch.float32)
    registered_counts = torch.zeros(xyz.shape[0], dtype=torch.float32)
    cpu_sum_staging = torch.empty(xyz.shape[0], cursor, dtype=torch.float32)
    cpu_count_staging = torch.empty(xyz.shape[0], dtype=torch.float32)
    feature_dir = Path(args.feature_dir).expanduser().resolve(strict=True)
    feature_manifest_record = file_record(feature_dir / "frame_manifest.json")
    device = torch.device(args.device)
    with torch.inference_mode():
        for frame_index, assignment in tqdm(
            zip(frame_indices, assignments),
            total=len(frame_indices),
            desc="direct-language sufficient-statistic exact MPR",
        ):
            raw = _load_backbone_frame(
                feature_dir, frame_index, height=height, width=width
            ).to(device=device, dtype=torch.float32)
            tokens = raw.permute(1, 2, 0).reshape(1, height * width, 1280)
            descriptor = F.normalize(head(tokens).squeeze(0).float(), dim=-1, eps=1e-8)
            scores = (descriptor @ bank.T).T.reshape(cursor, height, width)
            accumulate_contribution_mean_channel_chunked(
                scores,
                assignment["gaussian_ids"],
                assignment["pixel_ids"],
                assignment["marginal_weights"],
                registered_sum,
                registered_counts,
                channel_chunk_size=cursor,
                cpu_sum_staging=cpu_sum_staging,
                cpu_count_staging=cpu_count_staging,
            )
            del raw, tokens, descriptor, scores
    direct_scores, observed = finalize_registered_mean_chunked(
        registered_sum, registered_counts, row_chunk_size=int(args.row_chunk_size)
    )
    direct_scores = direct_scores.float()
    score_banks: dict[str, torch.Tensor] = {}
    for split in SPLITS:
        start, stop = offsets[split]
        fallback_scores = fallback_features @ text_banks[split].T
        values = fallback_scores
        values[observed] = direct_scores[observed, start:stop]
        score_banks[split] = values.contiguous()

    output = Path(args.output).expanduser().resolve()
    payload = {
        "schema": "radio_gs.scannet_direct_language_score_cache.v1",
        "schema_version": 1,
        "xyz": xyz,
        "valid": torch.ones(xyz.shape[0], dtype=torch.bool),
        "direct_observed": observed,
        **{f"scores_split_{split}": score_banks[split] for split in SPLITS},
        "metadata": {
            "artifact_type": "radio_gs_scannet_direct_language_score_cache",
            "construction": (
                "per_view_frozen_summary_head_then_text_inner_product_then_"
                "exact_mpr_with_d512_totality_fallback"
            ),
            "sufficient_statistic_identity": (
                "inner_product_commutes_with_exact_weighted_mean; class_argmax_unchanged"
            ),
            "query_independent": False,
            "evaluation_diagnostic_only": True,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_labels_opened": True,
            "text_queries_opened": True,
            "postprocessing": "none",
            "responsibility_authority": {
                "path": str(manifest_path), "sha256": manifest_sha
            },
            "d512_totality_fallback": {
                "path": str(fallback_path), "sha256": fallback_sha
            },
            "summary_head": head_record,
            "text_banks": text_records,
            "feature_frame_manifest": feature_manifest_record,
        },
    }
    write_torch_noclobber(output, payload)
    report = {
        "status": "complete",
        "output": file_record(output),
        "rows": int(xyz.shape[0]),
        "direct_observed_rows": int(observed.sum()),
        "direct_observed_ratio": float(observed.float().mean()),
        "totality_fallback_rows": int((~observed).sum()),
        "views": len(frame_indices),
        "aggregated_channels": cursor,
        "avoided_full_descriptor_channels": 1536 - cursor,
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--responsibility-authority", required=True)
    parser.add_argument("--expected-responsibility-authority-sha256", required=True)
    parser.add_argument("--summary-head-weights", required=True)
    parser.add_argument("--expected-summary-head-sha256", required=True)
    parser.add_argument("--text-embedding-cache", required=True)
    parser.add_argument("--d512-query-cache", required=True)
    parser.add_argument("--expected-d512-query-cache-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--row-chunk-size", type=int, default=8192)
    print(json.dumps(materialize(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
