#!/usr/bin/env python3
"""Materialize explicitly opt-in FP32 LERF primitive/text cosine responses.

This is a versioned sibling of the immutable FP16 materializer.  Keeping it
in a separate source file preserves every legacy default, schema, contract,
and source digest.  The method and query interface are unchanged; only the
score storage precision is promoted from FP16 to FP32.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from radio_gs.evaluation.openclip_readout import cosine_logits
from radio_gs.scripts import materialize_lerf_multiscale_query_score_cache as fp16
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_sha_bound_project_checkpoint_mapping,
    load_torch_mapping,
    sha256_file,
    stable_descriptor_load,
    write_frozen_json,
    write_torch_noclobber,
)


SCHEMA_VERSION = 3
CACHE_VERSION = 4
DIRECT3D_CONTRACT = "radio_gs.ours_lerf_direct3d_multiscale_query_scores_fp32.v4"
SHARED_AUTHORITY_CONTRACT = "radio_gs.lerf_multiscale_query_score_fp32_authority.v4"
SCORE_DTYPE = "torch.float32"


def _compile_query_scores_fp32(
    descriptors: torch.Tensor,
    global_rows: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    total_rows: int,
    chunk_size: int,
) -> torch.Tensor:
    """Use the legacy cosine implementation without the final FP16 cast."""

    if isinstance(chunk_size, bool) or int(chunk_size) <= 0:
        raise ValueError("chunk_size must be a positive integer")
    scores = torch.zeros(
        total_rows,
        fp16.SCALE_COUNT,
        int(text_embeddings.shape[0]),
        dtype=torch.float32,
    )
    for scale_index in range(fp16.SCALE_COUNT):
        for start in range(0, int(global_rows.numel()), int(chunk_size)):
            stop = min(int(global_rows.numel()), start + int(chunk_size))
            scores[global_rows[start:stop], scale_index] = cosine_logits(
                descriptors[start:stop, scale_index].float(),
                text_embeddings.float(),
            )
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("materialized FP32 query scores contain NaN or infinity")
    return scores.contiguous()


def _file_record(path: Path, digest: str) -> dict[str, str]:
    return {"path": str(path), "sha256": digest}


def materialize_fp32(
    *,
    descriptor_cache: str | Path,
    descriptor_cache_sha256: str,
    text_query_cache: str | Path,
    text_query_cache_sha256: str,
    field_checkpoint: str | Path,
    field_checkpoint_sha256: str,
    readout_checkpoint: str | Path,
    readout_checkpoint_sha256: str,
    renderer_geometry_checkpoint: str | Path,
    renderer_geometry_checkpoint_sha256: str,
    output: str | Path,
    chunk_size: int = 4096,
    allow_missing_text_canonicalization_metadata: bool = False,
) -> dict[str, Any]:
    """Build one immutable, strictly versioned FP32 Direct3D cache."""

    descriptor_expected = fp16._require_sha256(
        descriptor_cache_sha256, label="descriptor_cache_sha256"
    )
    text_expected = fp16._require_sha256(
        text_query_cache_sha256, label="text_query_cache_sha256"
    )
    field_expected = fp16._require_sha256(
        field_checkpoint_sha256, label="field_checkpoint_sha256"
    )
    readout_expected = fp16._require_sha256(
        readout_checkpoint_sha256, label="readout_checkpoint_sha256"
    )
    renderer_expected = fp16._require_sha256(
        renderer_geometry_checkpoint_sha256,
        label="renderer_geometry_checkpoint_sha256",
    )
    output_path = fp16._canonical_output(output)
    report_path = output_path.with_suffix(output_path.suffix + ".json")
    fp16._preflight_outputs(output_path, report_path)

    descriptor_payload, descriptor_digest, descriptor_path = load_torch_mapping(
        descriptor_cache,
        expected_sha256=descriptor_expected,
        map_location="cpu",
        label="SurfaceRegion multiscale descriptor cache",
    )
    text_payload, text_digest, text_path = load_torch_mapping(
        text_query_cache,
        expected_sha256=text_expected,
        map_location="cpu",
        label="frozen SigLIP2 query cache",
    )
    _, field_digest, field_path = stable_descriptor_load(
        field_checkpoint,
        lambda handle: None,
        expected_sha256=field_expected,
        label="canonical field checkpoint",
    )
    readout_payload, readout_digest, readout_path = load_torch_mapping(
        readout_checkpoint,
        expected_sha256=readout_expected,
        map_location="cpu",
        label="surface-region readout checkpoint",
    )
    renderer_payload, renderer_digest, renderer_path = (
        load_sha_bound_project_checkpoint_mapping(
            renderer_geometry_checkpoint,
            expected_sha256=renderer_expected,
            map_location="cpu",
            label="renderer geometry checkpoint",
        )
    )
    implementation_path = Path(__file__).resolve(strict=True)
    implementation_digest = sha256_file(implementation_path)
    legacy_implementation_path = Path(fp16.__file__).resolve(strict=True)
    legacy_implementation_digest = sha256_file(legacy_implementation_path)

    descriptor = fp16._validate_descriptor_cache(
        descriptor_payload,
        field_checkpoint_path=field_path,
        field_checkpoint_sha256=field_digest,
        readout_checkpoint_path=readout_path,
        readout_checkpoint_sha256=readout_digest,
        readout_native_scales=fp16._readout_native_scales(readout_payload),
    )
    renderer_xyz = fp16._renderer_checkpoint_xyz(renderer_payload)
    if renderer_xyz.shape != descriptor["xyz"].shape or not torch.equal(
        renderer_xyz, descriptor["xyz"]
    ):
        raise ValueError(
            "renderer geometry xyz/count/row-order differs from descriptor cache"
        )
    text = fp16._validate_text_query_cache(
        text_payload,
        allow_missing_canonicalization_metadata=(
            allow_missing_text_canonicalization_metadata
        ),
    )
    query_scores = _compile_query_scores_fp32(
        descriptor["descriptors"],
        descriptor["global_rows"],
        text["embeddings"],
        total_rows=int(descriptor["xyz"].shape[0]),
        chunk_size=int(chunk_size),
    )
    query_scores_sha = fp16._tensor_sha256(query_scores)
    query_order_sha = canonical_json_sha256(list(text["query_ids"]))
    scale_ids = tuple(
        fp16._scale_id(radius) for radius in descriptor["native_scales"]
    )
    scale_records = [
        {"id": scale_id, "value": radius, "unit": "meter"}
        for scale_id, radius in zip(scale_ids, descriptor["native_scales"])
    ]
    sources = {
        "descriptor_cache": _file_record(descriptor_path, descriptor_digest),
        "text_query_cache": _file_record(text_path, text_digest),
        "field_checkpoint": _file_record(field_path, field_digest),
        "readout_checkpoint": _file_record(readout_path, readout_digest),
        "renderer_geometry_checkpoint": _file_record(renderer_path, renderer_digest),
        "materializer_source": _file_record(
            implementation_path, implementation_digest
        ),
        "legacy_fp16_materializer_source": _file_record(
            legacy_implementation_path, legacy_implementation_digest
        ),
    }
    authority: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": fp16.ARTIFACT_TYPE,
        "contract": SHARED_AUTHORITY_CONTRACT,
        "score_semantics": "raw_independent_normalized_cosine",
        "score_formula": fp16.SCORING_FORMULA,
        "score_implementation": fp16.SCORING_IMPLEMENTATION,
        "score_dtype": SCORE_DTYPE,
        "precision_contract": {
            "normalization_dtype": SCORE_DTYPE,
            "matmul_dtype": SCORE_DTYPE,
            "storage_dtype": SCORE_DTYPE,
            "post_matmul_quantization": False,
            "legacy_fp16_default_changed": False,
        },
        "scale_axis": scale_records,
        "query_axis": {
            "ids": list(text["query_ids"]),
            "order_sha256": query_order_sha,
            "embedding_tensor_sha256": text["embedding_tensor_sha256"],
            "text_encoder": "official_siglip2_g",
            "model_name": fp16.SIGLIP2_MODEL_NAME,
            "text_canonicalization": fp16.SIGLIP2_TEXT_CANONICALIZATION,
            "text_canonicalization_metadata_present": text[
                "canonicalization_metadata_present"
            ],
            "text_canonicalization_authority": (
                "source_cache_metadata"
                if text["canonicalization_metadata_present"]
                else "explicit_legacy_frozen_cache_allowance"
            ),
            "prompt_templates": ["{query}"],
        },
        "geometry_axis": {
            **descriptor["geometry_fingerprint"],
            "valid_sha256": fp16._tensor_sha256(descriptor["valid"]),
            "field_checkpoint_sha256": field_digest,
            "readout_checkpoint_sha256": readout_digest,
            "renderer_geometry_checkpoint_sha256": renderer_digest,
            "renderer_xyz_sha256": fp16._direct_xyz_sha256(renderer_xyz),
        },
        "descriptor_axis": {
            "dimension": fp16.DESCRIPTOR_DIMENSION,
            "row_storage": descriptor["row_storage"],
            "valid_rows": int(descriptor["valid"].sum()),
            "features_by_scale_sha256": descriptor["descriptor_tensor_sha256"],
            "global_rows_sha256": fp16._tensor_sha256(
                descriptor["global_rows"]
            ),
            "readout_checkpoint_sha256": descriptor[
                "readout_checkpoint_sha256"
            ],
            "official_radio_checkpoint_sha256": descriptor[
                "official_radio_checkpoint_sha256"
            ],
        },
        "query_scores_sha256": query_scores_sha,
        "source_artifacts": sources,
        "consumer_contracts": {
            "direct3d": {
                "contract": DIRECT3D_CONTRACT,
                "tensor_layout": "[primitive_row,scale,query]",
                "scale_selection": "downstream_frozen_VALA_readout_only",
            },
            "lerf2d_scalar_map_renderer": {
                "score_semantics": fp16.SCORE_SEMANTICS_2D,
                "tensor_layout_before_render": "[primitive_row,scale,query]",
                "scale_ids": list(scale_ids),
                "query_text_axis": list(text["query_ids"]),
                "camera_and_occurrence_query_ids": (
                    "must_be_bound_by_renderer_without_reencoding_text"
                ),
            },
        },
        "calibration_constraints": {
            "softmax_applied": False,
            "temperature_applied": False,
            "peak_normalization_applied": False,
            "threshold_applied": False,
            "scale_reduction_applied": False,
            "benchmark_images_opened": False,
            "benchmark_annotations_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_metrics_opened": False,
        },
    }
    payload = {
        "version": CACHE_VERSION,
        "contract": DIRECT3D_CONTRACT,
        "query_scores": query_scores,
        "query_ids": list(text["query_ids"]),
        "scale_ids": list(scale_ids),
        "scale_radii_m": list(descriptor["native_scales"]),
        "xyz": descriptor["xyz"],
        "valid": descriptor["valid"],
        "geometry_fingerprint": descriptor["geometry_fingerprint"],
        "field_checkpoint_sha256": field_digest,
        "readout_checkpoint_sha256": readout_digest,
        "renderer_geometry_checkpoint_sha256": renderer_digest,
        "authority": authority,
    }
    write_torch_noclobber(output_path, payload)
    output_digest = sha256_file(output_path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": fp16.ARTIFACT_TYPE,
        "status": "complete_explicit_fp32_query_score_materialization",
        "query_score_cache": _file_record(output_path, output_digest),
        "query_score_shape_n3q": list(query_scores.shape),
        "query_score_dtype": SCORE_DTYPE,
        "valid_primitives": int(descriptor["valid"].sum()),
        "total_primitives": int(descriptor["valid"].numel()),
        "queries": int(query_scores.shape[2]),
        "execution": {
            "device": "cpu",
            "chunk_size": int(chunk_size),
            "chunk_size_changes_method": False,
            "explicit_fp32_opt_in": True,
            "legacy_fp16_default_changed": False,
            "allow_missing_text_canonicalization_metadata": bool(
                allow_missing_text_canonicalization_metadata
            ),
        },
        "shared_renderer_authority": authority,
    }
    write_frozen_json(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-cache", required=True)
    parser.add_argument("--descriptor-cache-sha256", required=True)
    parser.add_argument("--text-query-cache", required=True)
    parser.add_argument("--text-query-cache-sha256", required=True)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--field-checkpoint-sha256", required=True)
    parser.add_argument("--readout-checkpoint", required=True)
    parser.add_argument("--readout-checkpoint-sha256", required=True)
    parser.add_argument("--renderer-geometry-checkpoint", required=True)
    parser.add_argument("--renderer-geometry-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument(
        "--allow-missing-text-canonicalization-metadata", action="store_true"
    )
    args = parser.parse_args(argv)
    report = materialize_fp32(
        descriptor_cache=args.descriptor_cache,
        descriptor_cache_sha256=args.descriptor_cache_sha256,
        text_query_cache=args.text_query_cache,
        text_query_cache_sha256=args.text_query_cache_sha256,
        field_checkpoint=args.field_checkpoint,
        field_checkpoint_sha256=args.field_checkpoint_sha256,
        readout_checkpoint=args.readout_checkpoint,
        readout_checkpoint_sha256=args.readout_checkpoint_sha256,
        renderer_geometry_checkpoint=args.renderer_geometry_checkpoint,
        renderer_geometry_checkpoint_sha256=(
            args.renderer_geometry_checkpoint_sha256
        ),
        output=args.output,
        chunk_size=args.chunk_size,
        allow_missing_text_canonicalization_metadata=(
            args.allow_missing_text_canonicalization_metadata
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
