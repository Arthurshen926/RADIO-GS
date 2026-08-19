#!/usr/bin/env python3
"""Verify a LERF3D quarter-guard result used the sealed upstream inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from radio_gs.utils.immutable_artifacts import sha256_file


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _load(path: str | Path, label: str) -> tuple[dict[str, Any], Path]:
    source = Path(path).expanduser().resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, source


def validate(args: argparse.Namespace) -> dict[str, Any]:
    result, result_path = _load(args.result, "LERF3D result")
    receipt, receipt_path = _load(args.input_receipt, "input receipt")
    cache = Path(args.primitive_query_cache).resolve(strict=True)
    text = Path(args.text_cache).resolve(strict=True)
    canonical = Path(args.canonical_cache).resolve(strict=True)
    if (
        receipt.get("status") != "pass"
        or receipt.get("scene") != args.scene
        or Path(receipt.get("primitive_query_cache", {}).get("path", "")).resolve()
        != cache
    ):
        raise ValueError("input receipt differs")
    raw_args = result.get("args")
    scene = result.get("scene")
    if not isinstance(raw_args, Mapping) or not isinstance(scene, Mapping):
        raise ValueError("result structure differs")
    registration = scene.get("registration")
    if (
        raw_args.get("protocol_preset") != "vala_repo_3d"
        or raw_args.get("vala_post_mask_refinement")
        != args.expected_post_mask_refinement
        or Path(str(raw_args.get("external_query_feature_cache", ""))).resolve()
        != cache
        or Path(str(raw_args.get("text_embedding_cache", ""))).resolve() != text
        or Path(str(raw_args.get("canonical_embedding_cache", ""))).resolve()
        != canonical
        or not isinstance(registration, Mapping)
        or registration.get("source") != "external_query_feature_cache"
        or Path(str(registration.get("path", ""))).resolve() != cache
        or int(registration.get("registered_gaussians", 0))
        != int(registration.get("total_gaussians", -1))
    ):
        raise ValueError("result upstream or readout binding differs")
    sam_report: dict[str, Any] = {"enabled": False}
    if args.sam_membership_cache:
        sam_cache = Path(args.sam_membership_cache).resolve(strict=True)
        if (
            not args.expected_sam_membership_cache_sha256
            or sha256_file(sam_cache) != args.expected_sam_membership_cache_sha256
        ):
            raise ValueError("SAM membership cache SHA-256 differs")
        extent = scene.get("sam3_seed_extent")
        if (
            Path(str(raw_args.get("sam3_exact_mpr_membership_cache", ""))).resolve()
            != sam_cache
            or float(raw_args.get("sam3_seed_extent_alpha", -1.0)) != args.sam_seed_alpha
            or float(raw_args.get("sam3_seed_extent_proposal_mean_ratio", -1.0))
            != args.sam_proposal_mean_ratio
            or float(raw_args.get("sam3_seed_extent_seed_support_ratio", -1.0))
            != args.sam_seed_support_ratio
            or int(raw_args.get("sam3_seed_extent_minimum_views", -1))
            != args.sam_minimum_views
            or _as_bool(raw_args.get("sam3_seed_extent_query_conditioned", False))
            != bool(args.sam_query_conditioned)
            or not isinstance(extent, Mapping)
            or Path(str(extent.get("cache", ""))).resolve() != sam_cache
            or extent.get("cache_sha256") != args.expected_sam_membership_cache_sha256
            or extent.get("membership_lifting")
            != "exact_front_to_back_marginal_target_weight"
        ):
            raise ValueError("seed-conditioned exact-MPR SAM extent binding differs")
        sam_report = {
            "enabled": bool(extent.get("enabled", False)),
            "cache": str(sam_cache),
            "cache_sha256": args.expected_sam_membership_cache_sha256,
            "alpha": args.sam_seed_alpha,
            "proposal_mean_ratio": args.sam_proposal_mean_ratio,
            "seed_support_ratio": args.sam_seed_support_ratio,
            "minimum_views": args.sam_minimum_views,
            "query_conditioned": bool(args.sam_query_conditioned),
            "queries_with_extent": int(extent.get("num_queries_with_extent", 0)),
            "membership_lifting": extent.get("membership_lifting"),
        }
    return {
        "status": "pass",
        "scene": args.scene,
        "result": str(result_path),
        "input_receipt": str(receipt_path),
        "registration_source": registration["source"],
        "registered_gaussians": int(registration["registered_gaussians"]),
        "primitive_query_cache": str(cache),
        "protocol_preset": "vala_repo_3d",
        "extent": args.expected_post_mask_refinement,
        "sam_seed_extent": sam_report,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--input-receipt", required=True)
    parser.add_argument("--primitive-query-cache", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--canonical-cache", required=True)
    parser.add_argument("--sam-membership-cache", default="")
    parser.add_argument("--expected-sam-membership-cache-sha256", default="")
    parser.add_argument("--sam-seed-alpha", type=float, default=0.0)
    parser.add_argument("--sam-proposal-mean-ratio", type=float, default=0.5)
    parser.add_argument("--sam-seed-support-ratio", type=float, default=0.8)
    parser.add_argument("--sam-minimum-views", type=int, default=2)
    parser.add_argument("--sam-query-conditioned", action="store_true")
    parser.add_argument(
        "--expected-post-mask-refinement",
        default="peak_component_retention_guard",
    )
    print(json.dumps(validate(parser.parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
