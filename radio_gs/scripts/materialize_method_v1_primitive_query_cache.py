#!/usr/bin/env python3
"""Decode one Method-v1 field into query-independent primitive descriptors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.field import load_factorized_canonical_field_checkpoint
from radio_gs.five_benchmark_method_v1 import (
    METHOD_ID,
    validate_complete_field_payload,
    validate_method_authority,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts.materialize_lerf_multiscale_query_score_cache import (
    _renderer_checkpoint_xyz,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_sha_bound_project_checkpoint_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


DEFAULT_AUTHORITY = (
    Path(__file__).resolve().parents[2]
    / "paper/artifacts/five_benchmark_method_v1_authority_20260815.json"
)


def _xyz_sha256(value: torch.Tensor) -> str:
    rows = torch.as_tensor(value).detach().cpu().float().contiguous()
    digest = hashlib.sha256()
    for start in range(0, int(rows.shape[0]), 65_536):
        digest.update(rows[start : start + 65_536].numpy().tobytes(order="C"))
    return digest.hexdigest()


@torch.inference_mode()
def decode_method_v1_primitive_query_rows(
    field,
    summary_head: torch.nn.Module,
    *,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    """Apply the frozen official summary head independently to field rows."""

    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    field = field.to(device).eval()
    summary_head = summary_head.to(device).eval()
    try:
        head_dtype = next(summary_head.parameters()).dtype
    except StopIteration:
        head_dtype = torch.float32
    chunks: list[torch.Tensor] = []
    for start in range(0, int(field.num_gaussians), int(chunk_size)):
        stop = min(start + int(chunk_size), int(field.num_gaussians))
        indices = torch.arange(start, stop, device=device, dtype=torch.long)
        raw_radio = field.radio_features(indices).float()
        projected = summary_head(raw_radio.unsqueeze(0).to(head_dtype)).squeeze(0)
        chunks.append(F.normalize(projected.float(), dim=-1, eps=1e-8).half().cpu())
    return torch.cat(chunks, dim=0).contiguous()


def materialize(args: argparse.Namespace) -> dict[str, object]:
    authority, authority_sha256, authority_path = load_json_object(
        args.authority,
        label="five-benchmark Method-v1 authority",
    )
    validate_method_authority(authority)
    field, payload, _signature = load_factorized_canonical_field_checkpoint(
        args.field,
        map_location="cpu",
        expected_sha256=args.expected_field_sha256,
    )
    validate_complete_field_payload(payload)
    field_path = Path(args.field).expanduser().resolve(strict=True)
    field_record = file_record(field_path)

    geometry_payload, geometry_sha256, geometry_path = (
        load_sha_bound_project_checkpoint_mapping(
            args.geometry_checkpoint,
            expected_sha256=args.expected_geometry_sha256,
            map_location="cpu",
            label="Method-v1 renderer geometry checkpoint",
        )
    )
    xyz = _renderer_checkpoint_xyz(geometry_payload)
    expected_geometry = payload.get("geometry_fingerprint", {})
    if (
        int(xyz.shape[0]) != int(field.num_gaussians)
        or _xyz_sha256(xyz) != expected_geometry.get("xyz_sha256")
    ):
        raise ValueError("Method-v1 field and renderer geometry rows differ")

    summary_path = Path(args.summary_head_weights).expanduser().resolve(strict=True)
    summary_record = file_record(summary_path)
    if (
        args.expected_summary_head_sha256
        and summary_record["sha256"] != args.expected_summary_head_sha256
    ):
        raise ValueError("official summary-head SHA256 differs")
    head = SigLIP2SummaryHead.from_extracted_weights(str(summary_path))
    features = decode_method_v1_primitive_query_rows(
        field,
        head,
        device=torch.device(args.device),
        chunk_size=int(args.chunk_size),
    )
    if features.shape != (int(field.num_gaussians), 1536):
        raise RuntimeError("Method-v1 primitive query descriptor shape differs")
    valid = torch.ones(int(field.num_gaussians), dtype=torch.bool)
    output = Path(args.output).expanduser().resolve()
    cache = {
        "xyz": xyz,
        "summary_features": features,
        "features": features,
        "valid": valid,
        "metadata": {
            "schema_version": 1,
            "artifact_type": "radio_gs_method_v1_primitive_query_cache",
            "method_id": METHOD_ID,
            "feature_space": "official_siglip2_summary_descriptor_per_primitive",
            "construction": (
                "canonical_feature_field_decode_then_frozen_official_summary_head_then_l2"
            ),
            "field_checkpoint": field_record,
            "renderer_geometry_checkpoint": {
                "path": str(geometry_path),
                "sha256": geometry_sha256,
            },
            "summary_head": summary_record,
            "method_authority": {
                "path": str(authority_path),
                "sha256": authority_sha256,
            },
            "query_independent": True,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_labels_opened": False,
            "text_queries_opened": False,
            "postprocessing": "none",
        },
    }
    write_torch_noclobber(output, cache)
    output_record = file_record(output)
    report = {
        "status": "complete",
        "cache": output_record,
        "rows": int(features.shape[0]),
        "dimension": int(features.shape[1]),
        "valid_rows": int(valid.sum()),
        "field": field_record,
        "geometry": {"path": str(geometry_path), "sha256": geometry_sha256},
        "summary_head": summary_record,
        "method_authority_sha256": authority_sha256,
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", required=True)
    parser.add_argument("--expected-field-sha256", required=True)
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--expected-geometry-sha256", required=True)
    parser.add_argument("--summary-head-weights", required=True)
    parser.add_argument("--expected-summary-head-sha256", required=True)
    parser.add_argument("--authority", default=str(DEFAULT_AUTHORITY))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=8192)
    args = parser.parse_args()
    print(json.dumps(materialize(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
