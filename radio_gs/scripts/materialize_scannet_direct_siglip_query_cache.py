#!/usr/bin/env python3
"""Materialize a total ScanNet direct-language teacher query cache.

The input teacher applies the frozen official SigLIP2 summary projection to
each source-view RADIO feature map *before* exact marginal-responsibility
aggregation.  Only observed Gaussian rows are defined by that teacher.  To
hold the evaluation domain fixed, unobserved rows inherit the frozen Method-v1
D512 query descriptor bit-for-bit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.five_benchmark_method_v1 import METHOD_ID
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_sha_bound_project_checkpoint_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


def fuse_observed_direct_rows(
    direct_features: torch.Tensor,
    observed: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Replace fallback rows only where the direct teacher is observed."""

    direct = torch.as_tensor(direct_features).detach().cpu().float()
    valid = torch.as_tensor(observed).detach().cpu().bool().reshape(-1)
    output = torch.as_tensor(fallback).detach().cpu().float().clone()
    if direct.ndim != 2 or direct.shape[1] != 1536:
        raise ValueError("direct SigLIP2 rows must have shape [N,1536]")
    if valid.shape != (direct.shape[0],) or output.shape != direct.shape:
        raise ValueError("direct, observed, and fallback row domains differ")
    if not bool(valid.any()):
        raise ValueError("direct teacher has no observed Gaussian row")
    if not bool(torch.isfinite(direct[valid]).all()) or not bool(torch.isfinite(output).all()):
        raise ValueError("direct or fallback descriptors contain NaN or infinity")
    output[valid] = F.normalize(direct[valid], dim=-1, eps=1e-8)
    return F.normalize(output, dim=-1, eps=1e-8).half().contiguous()


def materialize(args: argparse.Namespace) -> dict[str, object]:
    direct, direct_sha, direct_path = load_sha_bound_project_checkpoint_mapping(
        args.direct_siglip_mpr,
        expected_sha256=args.expected_direct_siglip_mpr_sha256,
        map_location="cpu",
        label="ScanNet direct SigLIP2 exact-MPR teacher",
    )
    fallback, fallback_sha, fallback_path = load_sha_bound_project_checkpoint_mapping(
        args.d512_query_cache,
        expected_sha256=args.expected_d512_query_cache_sha256,
        map_location="cpu",
        label="ScanNet D512 query-cache fallback",
    )
    xyz = torch.as_tensor(direct.get("xyz")).detach().cpu().float().contiguous()
    fallback_xyz = torch.as_tensor(fallback.get("xyz")).detach().cpu().float().contiguous()
    direct_features = torch.as_tensor(
        direct.get("summary_features", direct.get("features"))
    ).detach().cpu()
    observed = torch.as_tensor(direct.get("valid")).detach().cpu().bool().reshape(-1)
    fallback_features = torch.as_tensor(
        fallback.get("summary_features", fallback.get("features"))
    ).detach().cpu()
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not torch.equal(xyz, fallback_xyz):
        raise ValueError("direct teacher and D512 fallback xyz rows differ")
    if direct_features.shape != (xyz.shape[0], 1536) or observed.shape != (xyz.shape[0],):
        raise ValueError("direct teacher tensor domain differs")
    if fallback_features.shape != direct_features.shape:
        raise ValueError("D512 fallback descriptor domain differs")

    features = fuse_observed_direct_rows(direct_features, observed, fallback_features)
    output = Path(args.output).expanduser().resolve()
    payload = {
        "xyz": xyz,
        "summary_features": features,
        "features": features,
        "valid": torch.ones(xyz.shape[0], dtype=torch.bool),
        "direct_observed": observed,
        "metadata": {
            "schema_version": 1,
            "artifact_type": "radio_gs_method_v1_primitive_query_cache",
            "method_id": METHOD_ID,
            "feature_space": "official_siglip2_summary_descriptor_per_primitive",
            "construction": (
                "per_view_frozen_official_siglip2_projection_then_exact_mpr_"
                "with_d512_totality_fallback_then_l2"
            ),
            "direct_siglip_mpr": {"path": str(direct_path), "sha256": direct_sha},
            "d512_query_fallback": {"path": str(fallback_path), "sha256": fallback_sha},
            "query_independent": True,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_labels_opened": False,
            "text_queries_opened": False,
            "postprocessing": "none",
        },
    }
    write_torch_noclobber(output, payload)
    report = {
        "status": "complete",
        "cache": file_record(output),
        "rows": int(xyz.shape[0]),
        "direct_observed_rows": int(observed.sum()),
        "direct_observed_ratio": float(observed.float().mean()),
        "totality_fallback_rows": int((~observed).sum()),
        "direct_siglip_mpr": {"path": str(direct_path), "sha256": direct_sha},
        "d512_query_fallback": {"path": str(fallback_path), "sha256": fallback_sha},
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-siglip-mpr", required=True)
    parser.add_argument("--expected-direct-siglip-mpr-sha256", required=True)
    parser.add_argument("--d512-query-cache", required=True)
    parser.add_argument("--expected-d512-query-cache-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
