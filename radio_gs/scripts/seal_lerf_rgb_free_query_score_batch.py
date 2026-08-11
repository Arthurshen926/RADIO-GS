#!/usr/bin/env python3
"""Seal an RGB-free LERF derivative/query-score batch before benchmark access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected object JSON: {path}")
    return value


def _require_false(metadata: dict[str, Any], key: str, path: Path) -> None:
    if metadata.get(key) is not False:
        raise ValueError(f"{path}: {key} must be explicitly false")


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_query_independent(path: Path, field_sha256: str) -> dict[str, Any]:
    artifact = _artifact(path)
    sidecar_path = path.with_suffix(path.suffix + ".json")
    sidecar = _load_json(sidecar_path)
    contract = sidecar.get("metadata", sidecar)
    if not isinstance(contract, dict):
        raise ValueError(f"{sidecar_path}: invalid nested metadata")
    _require_false(contract, "benchmark_images_opened", sidecar_path)
    _require_false(contract, "benchmark_masks_opened", sidecar_path)
    _require_false(contract, "text_queries_opened", sidecar_path)
    query_independent = contract.get("query_independent") is True
    if path.name.startswith("support_graph_"):
        query_independent = query_independent or all(
            isinstance(contract.get(key), dict)
            and contract[key].get("query_independent") is True
            for key in ("feature_hash", "capability_affinity", "surface_relation")
        )
    if path.name.startswith("descriptor_"):
        query_independent = query_independent or (
            contract.get("query_set_invariant") is True
            and contract.get("text_queries_opened") is False
            and contract.get("construction")
            == "canonical_radio_surface_region_readout_then_official_summary_head"
        )
    if not query_independent:
        raise ValueError(f"{sidecar_path}: query_independent must be true")
    embedded_field = contract.get("field_checkpoint_sha256")
    if embedded_field is None and isinstance(
        contract.get("capability_metadata"), dict
    ):
        embedded_field = contract["capability_metadata"].get(
            "field_checkpoint_sha256"
        )
    if embedded_field != field_sha256:
        raise ValueError(
            f"{sidecar_path}: field SHA mismatch {embedded_field!r} != {field_sha256!r}"
        )
    artifact["metadata"] = _artifact(sidecar_path)
    return artifact


def _validate_score(
    path: Path,
    *,
    field_sha256: str,
    expected_queries: int,
) -> dict[str, Any]:
    artifact = _artifact(path)
    sidecar_path = path.with_suffix(path.suffix + ".json")
    sidecar = _load_json(sidecar_path)
    if sidecar.get("query_score_dtype") != "torch.float32":
        raise ValueError(f"{sidecar_path}: score dtype is not torch.float32")
    expected_shape = [168791, 3, expected_queries]
    if sidecar.get("query_score_shape_n3q") != expected_shape:
        raise ValueError(f"{sidecar_path}: unexpected score shape")
    authority = sidecar.get("shared_renderer_authority", {})
    calibration = authority.get("calibration_constraints", {})
    for key in (
        "benchmark_annotations_opened",
        "benchmark_images_opened",
        "benchmark_masks_opened",
        "benchmark_metrics_opened",
        "peak_normalization_applied",
        "scale_reduction_applied",
        "softmax_applied",
        "temperature_applied",
        "threshold_applied",
    ):
        _require_false(calibration, key, sidecar_path)
    geometry = authority.get("geometry_axis", {})
    if geometry.get("field_checkpoint_sha256") != field_sha256:
        raise ValueError(f"{sidecar_path}: geometry field SHA mismatch")
    payload = torch.load(path, map_location="cpu")
    scores = payload.get("query_scores") if isinstance(payload, dict) else None
    if not isinstance(scores, torch.Tensor):
        raise ValueError(f"{path}: missing tensor query_scores")
    if scores.dtype != torch.float32 or list(scores.shape) != expected_shape:
        raise ValueError(f"{path}: tensor dtype/shape contract failed")
    if payload.get("field_checkpoint_sha256") != field_sha256:
        raise ValueError(f"{path}: payload field SHA mismatch")
    artifact["metadata"] = _artifact(sidecar_path)
    artifact["tensor_shape"] = expected_shape
    artifact["tensor_dtype"] = "torch.float32"
    artifact["query_count"] = expected_queries
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--expected-field-checkpoint-sha256", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to clobber seal: {output}")
    field = Path(args.field_checkpoint).resolve()
    prereg = Path(args.preregistration).resolve()
    if _sha256(field) != args.expected_field_checkpoint_sha256:
        raise ValueError("field checkpoint SHA mismatch")
    if _sha256(prereg) != args.expected_preregistration_sha256:
        raise ValueError("preregistration SHA mismatch")

    names = {
        "factorized_primitive_state": "factorized_primitive_state_v3.pt",
        "official_dino_sam3_views": "official_dino_sam3_views_v3.pt",
        "support_graph": "support_graph_v3_v2_isomorphic.pt",
        "surface_region_descriptor": "descriptor_v3_v2_isomorphic.pt",
    }
    derivatives = {
        key: _validate_query_independent(
            root / name, args.expected_field_checkpoint_sha256
        )
        for key, name in names.items()
    }
    query_sidecar = root / "descriptor_v3_v2_isomorphic_query.pt"
    derivatives["surface_region_query_sidecar"] = _artifact(query_sidecar)

    scores = {
        "positive_fp32": _validate_score(
            root / "positive_fp32.pt",
            field_sha256=args.expected_field_checkpoint_sha256,
            expected_queries=21,
        ),
        "negative_fp32": _validate_score(
            root / "negative_fp32.pt",
            field_sha256=args.expected_field_checkpoint_sha256,
            expected_queries=4,
        ),
    }
    report = {
        "schema": "radio_gs.lerf_rgb_free_query_score_batch_seal.v1",
        "status": "all_query_free_derivatives_and_fp32_scores_sealed_before_benchmark_metrics",
        "scene_id": "figurines",
        "root": str(root),
        "preregistration": _artifact(prereg),
        "field_checkpoint": _artifact(field),
        "derivatives": derivatives,
        "query_scores": scores,
        "access_audit": {
            "benchmark_images_opened": False,
            "benchmark_annotations_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_metrics_opened": False,
            "target_rgb_or_sam_used": False,
            "all_artifacts_local_ssd": str(root).startswith(
                "/root/RADIO-GS/local_ssd_results/"
            ),
        },
        "next_action": "stop; benchmark metric remains unauthorized",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
