#!/usr/bin/env python3
"""Build field-only primitive text scores for identity-seed selection.

This cache is query-conditioned, but it never reads a region/proposal cache.
It can therefore choose an identity seed without circularly consuming the P0
extent that is subsequently audited around that seed.
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

from radio_gs.scripts.build_lerf_identity_seeded_object_topology_scores import (
    _query_names,
    _select_embedding_rows,
)
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    score_text_aligned_embeddings,
    vala_knn_minmax_scores,
)
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_field_only_primitive_identity_scores.v1"


def _xyz_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"field-only identity score output exists: {output}")
    query_names = _query_names(args.query_names)
    primitive_path = Path(args.primitive_query_cache).expanduser().resolve()
    primitive = torch.load(primitive_path, map_location="cpu", weights_only=False)
    primitive_metadata = dict(primitive.get("metadata", {}))
    xyz = torch.as_tensor(primitive.get("xyz")).float().cpu()
    features = torch.as_tensor(
        primitive.get("summary_features", primitive.get("features"))
    ).float().cpu()
    valid = torch.as_tensor(primitive.get("valid")).bool().cpu()
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or features.ndim != 2
        or features.shape[0] != len(xyz)
        or valid.shape != (len(xyz),)
        or not bool(valid.any())
        or primitive_metadata.get("query_independent") is not True
        or any(bool(primitive_metadata.get(key, False)) for key in (
            "benchmark_images_opened", "benchmark_masks_opened",
            "benchmark_labels_opened", "text_queries_opened",
        ))
    ):
        raise ValueError("primitive field cache lacks source-only canonical identity")

    text_path = Path(args.text_embedding_cache).expanduser().resolve()
    canonical_path = Path(args.canonical_embedding_cache).expanduser().resolve()
    text_payload = torch.load(text_path, map_location="cpu", weights_only=False)
    canonical_payload = torch.load(canonical_path, map_location="cpu", weights_only=False)
    text = F.normalize(_select_embedding_rows(text_payload, query_names), dim=-1)
    canonical = F.normalize(
        torch.as_tensor(canonical_payload.get("embeddings")).float().cpu(), dim=-1
    )
    if text.shape[1] != features.shape[1] or canonical.shape[1] != features.shape[1]:
        raise ValueError("text embedding and primitive field dimensions differ")

    device = torch.device(args.device)
    chunks: list[torch.Tensor] = []
    step = max(1, int(args.chunk_size))
    with torch.inference_mode():
        text_device = text.to(device)
        canonical_device = canonical.to(device)
        for start in range(0, len(features), step):
            chunks.append(score_text_aligned_embeddings(
                features[start : start + step].to(device),
                text_device,
                canonical_embeddings=canonical_device,
                scoring="relevancy",
                softmax_temperature=10.0,
            ).cpu())
    raw = torch.cat(chunks)
    raw[~valid] = -1.0e4
    scores = vala_knn_minmax_scores(
        raw, xyz, k=10, chunk_size=int(args.knn_chunk_size), valid_mask=valid
    )
    if scores.shape != (len(xyz), len(query_names)) or not bool(
        torch.isfinite(scores[valid]).all()
    ):
        raise ValueError("field-only primitive identity scores differ")
    scores[~valid] = -1.0e4
    payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene": str(args.scene),
        "query_scores": scores,
        "valid": valid,
        "xyz": xyz,
        "metadata": {
            "query_names": query_names,
            "score_role": "field_only_primitive_identity_seed",
            "score_formula": (
                "method_v1_official_siglip2_relevancy_then_frozen_vala_knn10_minmax"
            ),
            "region_membership_cache_opened": False,
            "proposal_cache_opened": False,
            "primitive_query_cache": str(primitive_path),
            "primitive_query_cache_sha256": sha256_file(primitive_path),
            "primitive_field_checkpoint": primitive_metadata.get("field_checkpoint"),
            "text_embedding_cache": str(text_path),
            "text_embedding_cache_sha256": sha256_file(text_path),
            "canonical_embedding_cache": str(canonical_path),
            "canonical_embedding_cache_sha256": sha256_file(canonical_path),
            "xyz_sha256": _xyz_sha256(xyz),
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
        "status": "complete_field_only_identity_seed_scores",
        "scene": str(args.scene),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "queries": len(query_names),
        "rows": len(xyz),
        "region_membership_cache_opened": False,
        "promotion": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--primitive-query-cache", required=True)
    parser.add_argument("--text-embedding-cache", required=True)
    parser.add_argument("--canonical-embedding-cache", required=True)
    parser.add_argument("--query-names", required=True)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--knn-chunk-size", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
