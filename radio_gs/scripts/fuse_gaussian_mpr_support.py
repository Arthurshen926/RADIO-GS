#!/usr/bin/env python3
"""Fill uncovered dominant MPR rows from a query-free support MPR cache.

The primary cache is authoritative wherever it has an observation.  The
support cache is used only for rows that are invalid in the primary cache, so
adjoint registration can increase primitive coverage without blurring the
dominant raster assignment on already supervised primitives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xyz_digest(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _tensor(payload: dict[str, Any], key: str) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"MPR cache lacks tensor {key!r}")
    return value


def _assert_query_free(payload: dict[str, Any], label: str) -> None:
    metadata = dict(payload.get("metadata", {}))
    contaminated = [
        key
        for key in (
            "benchmark_images_opened",
            "benchmark_masks_opened",
            "text_queries_opened",
        )
        if bool(metadata.get(key, False))
    ]
    if contaminated:
        raise ValueError(f"{label} MPR cache is benchmark-contaminated: {contaminated}")


def fuse_primary_with_support(
    primary: dict[str, Any],
    support: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a dominant-preserving union and a serialisable report."""

    _assert_query_free(primary, "primary")
    _assert_query_free(support, "support")
    primary_xyz = _tensor(primary, "xyz").float().cpu()
    support_xyz = _tensor(support, "xyz").float().cpu()
    if primary_xyz.shape != support_xyz.shape or _xyz_digest(primary_xyz) != _xyz_digest(
        support_xyz
    ):
        raise ValueError("primary and support MPR caches do not share row-aligned geometry")

    primary_features = _tensor(primary, "features").cpu()
    support_features = _tensor(support, "features").cpu()
    if primary_features.shape != support_features.shape:
        raise ValueError("primary and support feature tensors have different shapes")
    primary_valid = _tensor(primary, "valid").bool().cpu()
    support_valid = _tensor(support, "valid").bool().cpu()
    if primary_valid.shape != support_valid.shape or primary_valid.shape != (
        primary_xyz.shape[0],
    ):
        raise ValueError("MPR validity masks do not align with geometry")

    fallback = ~primary_valid & support_valid
    fused_valid = primary_valid | support_valid
    fused_features = primary_features.clone()
    fused_features[fallback] = support_features[fallback].to(fused_features.dtype)

    primary_counts = _tensor(primary, "view_counts").long().cpu()
    support_counts = _tensor(support, "view_counts").long().cpu()
    fused_counts = primary_counts.clone()
    fused_counts[fallback] = support_counts[fallback]

    primary_reliability = torch.as_tensor(primary.get("reliability")).cpu()
    support_reliability = torch.as_tensor(support.get("reliability")).cpu()
    if primary_reliability.shape != support_reliability.shape:
        raise ValueError("primary and support reliability tensors have different shapes")
    fused_reliability = primary_reliability.clone()
    fused_reliability[fallback] = support_reliability[fallback].to(
        fused_reliability.dtype
    )
    # The third fixed reliability channel identifies dominant registration.
    # This lets the compact fusion MLP treat fallback supervision as lower
    # confidence without changing the descriptor or validity contract.
    if fused_reliability.ndim == 2 and fused_reliability.shape[1] >= 3:
        fused_reliability[fused_valid, 2] = primary_valid[fused_valid].to(
            fused_reliability.dtype
        )

    primary_meta = dict(primary.get("metadata", {}))
    support_meta = dict(support.get("metadata", {}))
    metadata = {
        **primary_meta,
        "construction": "dominant_primary_with_query_free_support_completion",
        "aggregation_mode": "primary_then_support_completion",
        "primary_construction": primary_meta.get("construction", ""),
        "support_construction": support_meta.get("construction", ""),
        "primary_valid_count": int(primary_valid.sum()),
        "support_valid_count": int(support_valid.sum()),
        "fallback_valid_count": int(fallback.sum()),
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "query_names": [],
    }
    fused = {
        **primary,
        "features": fused_features,
        "valid": fused_valid,
        "view_counts": fused_counts,
        "reliability": fused_reliability,
        "metadata": metadata,
    }
    report = {
        "num_gaussians": int(primary_xyz.shape[0]),
        "primary_valid_count": int(primary_valid.sum()),
        "support_valid_count": int(support_valid.sum()),
        "fallback_valid_count": int(fallback.sum()),
        "fused_valid_count": int(fused_valid.sum()),
        "fused_valid_ratio": float(fused_valid.float().mean()),
        "primary_rows_preserved": bool(
            torch.equal(fused_features[primary_valid], primary_features[primary_valid])
        ),
    }
    return fused, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--support", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    primary_path = Path(args.primary)
    support_path = Path(args.support)
    primary = torch.load(primary_path, map_location="cpu")
    support = torch.load(support_path, map_location="cpu")
    if not isinstance(primary, dict) or not isinstance(support, dict):
        raise ValueError("MPR caches must be dictionaries")
    fused, report = fuse_primary_with_support(primary, support)
    fused["metadata"].update(
        {
            "primary_cache": str(primary_path.resolve()),
            "primary_cache_sha256": _sha256_file(primary_path),
            "support_cache": str(support_path.resolve()),
            "support_cache_sha256": _sha256_file(support_path),
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fused, output)
    report = {**report, "output": str(output), **fused["metadata"]}
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
