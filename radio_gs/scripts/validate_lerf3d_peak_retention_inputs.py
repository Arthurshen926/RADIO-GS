#!/usr/bin/env python3
"""Fail-closed preflight for one LERF3D Universal Field readout scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from radio_gs.utils.immutable_artifacts import sha256_file


METHOD_AUTHORITY_SHA256 = (
    "48e0d432f38ce9ad973e03a805e50a477c1e334ac78349f80eb6e6e02f6239c6"
)
SUMMARY_HEAD_SHA256 = (
    "41ccc47b2da9b1aed3ee1e80397dc721ec625e083054175c27698e8840b6263c"
)
TEXT_CACHE_SHA256 = (
    "2b8142252b541a585e248226d38efc30cde53cdef82d5b0ac371085102888e7b"
)
CANONICAL_CACHE_SHA256 = (
    "0699bcf9b4dbbbd74ccbee2164a06810f7137bc8b7f0a00238661b8e993a47d0"
)


def _resolve_and_hash(path: str | Path, expected: str, label: str) -> Path:
    source = Path(path).expanduser().resolve(strict=True)
    if sha256_file(source) != expected:
        raise ValueError(f"{label} SHA-256 differs")
    return source


def validate(args: argparse.Namespace) -> dict[str, Any]:
    cache = _resolve_and_hash(
        args.primitive_query_cache,
        args.expected_primitive_query_cache_sha256,
        "primitive query cache",
    )
    field = _resolve_and_hash(
        args.field, args.expected_field_sha256, "Universal Field"
    )
    renderer = _resolve_and_hash(
        args.renderer, args.expected_renderer_sha256, "renderer geometry"
    )
    config = _resolve_and_hash(args.config, args.expected_config_sha256, "config")
    method = _resolve_and_hash(
        args.method_authority, METHOD_AUTHORITY_SHA256, "Method-v1 authority"
    )
    summary = _resolve_and_hash(
        args.summary_head, SUMMARY_HEAD_SHA256, "official summary head"
    )
    text = _resolve_and_hash(args.text_cache, TEXT_CACHE_SHA256, "text cache")
    canonical = _resolve_and_hash(
        args.canonical_cache, CANONICAL_CACHE_SHA256, "canonical cache"
    )

    payload = torch.load(cache, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("primitive query cache must be a mapping")
    xyz = payload.get("xyz")
    features = payload.get("summary_features")
    valid = payload.get("valid")
    metadata = payload.get("metadata")
    if (
        not isinstance(xyz, torch.Tensor)
        or xyz.ndim != 2
        or tuple(xyz.shape[1:]) != (3,)
        or not isinstance(features, torch.Tensor)
        or features.shape != (int(xyz.shape[0]), 1536)
        or not isinstance(valid, torch.Tensor)
        or valid.dtype != torch.bool
        or valid.shape != (int(xyz.shape[0]),)
        or not isinstance(metadata, Mapping)
    ):
        raise ValueError("primitive query cache tensor contract differs")
    expected_literals = {
        "schema_version": 1,
        "artifact_type": "radio_gs_method_v1_primitive_query_cache",
        "method_id": "radio-gs-method-v1",
        "feature_space": "official_siglip2_summary_descriptor_per_primitive",
        "construction": (
            "canonical_feature_field_decode_then_frozen_official_summary_head_then_l2"
        ),
        "query_independent": True,
        "postprocessing": "none",
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "text_queries_opened": False,
    }
    if any(metadata.get(key) != value for key, value in expected_literals.items()):
        raise ValueError("primitive query cache method contract differs")

    def _record(name: str, expected_path: Path, expected_sha: str) -> None:
        value = metadata.get(name)
        if (
            not isinstance(value, Mapping)
            or Path(str(value.get("path", ""))).resolve() != expected_path
            or value.get("sha256") != expected_sha
        ):
            raise ValueError(f"primitive query cache {name} binding differs")

    _record("field_checkpoint", field, args.expected_field_sha256)
    _record("renderer_geometry_checkpoint", renderer, args.expected_renderer_sha256)
    _record("method_authority", method, METHOD_AUTHORITY_SHA256)
    _record("summary_head", summary, SUMMARY_HEAD_SHA256)
    return {
        "schema_version": 1,
        "artifact_type": "radio_gs_lerf3d_peak_retention_input_receipt",
        "scene": args.scene,
        "primitive_query_cache": {"path": str(cache), "sha256": args.expected_primitive_query_cache_sha256},
        "field": {"path": str(field), "sha256": args.expected_field_sha256},
        "renderer": {"path": str(renderer), "sha256": args.expected_renderer_sha256},
        "config": {"path": str(config), "sha256": args.expected_config_sha256},
        "method_authority": {"path": str(method), "sha256": METHOD_AUTHORITY_SHA256},
        "summary_head": {"path": str(summary), "sha256": SUMMARY_HEAD_SHA256},
        "text_cache": {"path": str(text), "sha256": TEXT_CACHE_SHA256},
        "canonical_cache": {"path": str(canonical), "sha256": CANONICAL_CACHE_SHA256},
        "num_gaussians": int(xyz.shape[0]),
        "valid_gaussians": int(valid.sum()),
        "benchmark_images_masks_labels_opened_by_preflight": False,
        "status": "pass",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--primitive-query-cache", required=True)
    parser.add_argument("--expected-primitive-query-cache-sha256", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--expected-field-sha256", required=True)
    parser.add_argument("--renderer", required=True)
    parser.add_argument("--expected-renderer-sha256", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--method-authority", required=True)
    parser.add_argument("--summary-head", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--canonical-cache", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(validate(build_parser().parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
