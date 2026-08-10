#!/usr/bin/env python3
"""Materialize genuine MPR-to-frozen-text raw cosine scores for LERF.

This premetric sibling reads the already sealed official crop-summary MPR,
normalizes ``MPR.features`` and the frozen positive/canonical-negative text
embeddings in FP32, computes their independent cosine responses, and copies
each single MPR response into the three native template scale slots.  It
performs no negative routing, kNN, min-max, thresholding, scale selection,
rendering, or target evaluation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen
from radio_gs.scripts import materialize_lerf_valid_domain_knn_candidate as template_loader
from radio_gs.training.tensor_cache_io import validate_mpr_cache_payload
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.lerf_genuine_mpr_raw_query_scores.v1"
CACHE_VERSION = 4
CACHE_CONTRACT = "radio_gs.ours_lerf_direct3d_multiscale_query_scores_fp32.v4"
AUTHORITY_CONTRACT = "radio_gs.lerf_multiscale_query_score_fp32_authority.v4"
SCORE_FORMULA = "l2_normalize(descriptor) @ l2_normalize(text_embedding).T"
SCALE_COUNT = 3
FEATURE_DIMENSION = 1536


def access_audit() -> dict[str, bool]:
    return {
        "genuine_mpr_opened": True,
        "raw_positive_geometry_template_opened": True,
        "raw_canonical_negative_geometry_template_opened": True,
        "frozen_positive_text_bank_opened": True,
        "frozen_canonical_negative_text_bank_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_opened": False,
        "target_metrics_computed": False,
        "negative_probability_routing_applied": False,
        "knn_or_minmax_applied": False,
        "gpu_used": False,
        "result_dependent_parameters": False,
    }


def select_text_axis(
    bank: Mapping[str, Any], query_ids: Sequence[str]
) -> tuple[torch.Tensor, dict[str, Any]]:
    queries = bank.get("queries")
    embeddings = bank.get("embeddings")
    expected = tuple(str(value) for value in query_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("query_ids must be non-empty and unique")
    if (
        bank.get("prompt_templates") != ["{query}"]
        or bank.get("text_encoder") != "siglip2"
        or not isinstance(bank.get("model_name"), str)
        or not bank.get("model_name")
        or not isinstance(queries, list)
        or len(set(str(value) for value in queries)) != len(queries)
        or not torch.is_tensor(embeddings)
        or embeddings.dtype != torch.float32
        or embeddings.ndim != 2
        or embeddings.shape != (len(queries), FEATURE_DIMENSION)
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError("frozen text embedding bank contract differs")
    lookup = {str(query): index for index, query in enumerate(queries)}
    missing = [query for query in expected if query not in lookup]
    if missing:
        raise ValueError(f"frozen text bank lacks exact queries: {missing}")
    selected = embeddings[[lookup[query] for query in expected]].detach().float().cpu().contiguous()
    if bool((selected.norm(dim=1) <= 1e-8).any()):
        raise ValueError("selected frozen text embeddings must be nonzero")
    metadata = {
        "ids": list(expected),
        "order_sha256": canonical_json_sha256(list(expected)),
        "embedding_tensor_sha256": frozen.tensor_sha256_typed(selected),
        "text_encoder": str(bank["text_encoder"]),
        "model_name": str(bank["model_name"]),
        "text_canonicalization": "frozen_protocol_exact_raw_embedding_bank",
        "text_canonicalization_metadata_present": True,
        "text_canonicalization_authority": "source_cache_metadata",
        "prompt_templates": ["{query}"],
    }
    return selected, metadata


def compute_raw_scores(
    features: torch.Tensor,
    valid: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    values = torch.as_tensor(features).detach().cpu()
    mask = torch.as_tensor(valid).detach().bool().cpu().reshape(-1)
    text = torch.as_tensor(text_embeddings).detach().float().cpu().contiguous()
    if isinstance(chunk_size, bool) or int(chunk_size) <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if (
        values.ndim != 2
        or values.shape[1] != FEATURE_DIMENSION
        or values.dtype not in {torch.float16, torch.float32}
        or mask.shape != (values.shape[0],)
        or not bool(mask.any())
        or text.ndim != 2
        or text.shape[1] != FEATURE_DIMENSION
        or text.shape[0] == 0
        or not bool(torch.isfinite(values).all())
        or not bool(torch.isfinite(text).all())
    ):
        raise ValueError("MPR feature, validity, or text axes differ")
    if bool(values[~mask].ne(0).any()):
        raise ValueError("invalid MPR feature rows must be exact zero")
    normalized_text = F.normalize(text, dim=1, eps=1e-8)
    scores = torch.zeros((values.shape[0], text.shape[0]), dtype=torch.float32)
    rows = torch.where(mask)[0]
    step = int(chunk_size)
    for start in range(0, int(rows.numel()), step):
        selected_rows = rows[start : start + step]
        descriptors = values[selected_rows].float()
        if bool((descriptors.norm(dim=1) <= 1e-8).any()):
            raise ValueError("valid MPR feature rows must be nonzero")
        scores[selected_rows] = F.normalize(descriptors, dim=1, eps=1e-8) @ normalized_text.T
    query_scores = scores[:, None, :].expand(-1, SCALE_COUNT, -1).contiguous()
    if query_scores.dtype != torch.float32 or not bool(torch.isfinite(query_scores).all()):
        raise ValueError("MPR raw scores must be finite contiguous FP32")
    if not torch.equal(query_scores[:, 0], query_scores[:, 1]) or not torch.equal(
        query_scores[:, 1], query_scores[:, 2]
    ):
        raise AssertionError("MPR scale replication is not bitwise exact")
    if bool(query_scores[~mask].ne(0).any()):
        raise AssertionError("invalid MPR score rows must be exact zero")
    return query_scores


def _validate_geometry(
    mpr: Mapping[str, Any],
    template_raw: Mapping[str, Any],
) -> None:
    mpr_xyz = torch.as_tensor(mpr["xyz"]).detach().cpu().contiguous()
    template_xyz = torch.as_tensor(template_raw["xyz"]).detach().cpu().contiguous()
    if mpr_xyz.dtype != torch.float32 or template_xyz.dtype != torch.float32:
        raise ValueError("genuine MPR and raw template xyz must both be FP32")
    if mpr_xyz.shape != template_xyz.shape or not torch.equal(mpr_xyz, template_xyz):
        raise ValueError("genuine MPR geometry differs from raw template bitwise")
    mpr_geometry = mpr.get("geometry_fingerprint")
    template_geometry = template_raw.get("geometry_fingerprint")
    if not isinstance(mpr_geometry, Mapping) or not isinstance(template_geometry, Mapping):
        raise ValueError("MPR/template geometry fingerprints are missing")
    if (
        int(mpr_geometry.get("num_gaussians", -1))
        != int(template_geometry.get("num_gaussians", -2))
        or str(mpr_geometry.get("xyz_sha256", ""))
        != str(template_geometry.get("xyz_sha256", "different"))
    ):
        raise ValueError("genuine MPR geometry fingerprint differs from raw template")


def _canonical_new_outputs(args: argparse.Namespace) -> dict[str, Path]:
    raw = {
        "positive_cache": args.output_positive_cache,
        "positive_report": args.output_positive_report,
        "negative_cache": args.output_negative_cache,
        "negative_report": args.output_negative_report,
    }
    outputs = {
        role: Path(value).expanduser().resolve() for role, value in raw.items()
    }
    if any(str(outputs[role]) != raw[role] for role in outputs):
        raise ValueError("output paths must be canonical absolute")
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("all four MPR raw output paths must be distinct")
    if any(path.exists() or path.is_symlink() for path in outputs.values()):
        raise FileExistsError("all four MPR raw outputs must be new")
    return outputs


def _build_payload(
    *,
    mpr: Mapping[str, Any],
    template_raw: Mapping[str, Any],
    template: frozen.OursMultiscaleQueryScoreCache,
    query_scores: torch.Tensor,
    query_axis: Mapping[str, Any],
    mpr_record: Mapping[str, str],
    mpr_sidecar_record: Mapping[str, str],
    text_bank_record: Mapping[str, str],
    materializer_record: Mapping[str, str],
    template_positive_record: Mapping[str, str],
    template_negative_record: Mapping[str, str],
    score_role: str,
    features_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if score_role not in {"positive_benchmark_queries", "canonical_negative_queries"}:
        raise ValueError("MPR raw score role differs")
    template_authority = template_raw["authority"]
    template_sources = template_authority["source_artifacts"]
    required_template_sources = (
        "field_checkpoint",
        "readout_checkpoint",
        "renderer_geometry_checkpoint",
        "legacy_fp16_materializer_source",
    )
    if any(
        not isinstance(template_sources.get(role), Mapping)
        for role in required_template_sources
    ):
        raise ValueError("raw template source authority is incomplete")
    sources = {
        "descriptor_cache": dict(mpr_record),
        "text_query_cache": dict(text_bank_record),
        "field_checkpoint": dict(template_sources["field_checkpoint"]),
        "readout_checkpoint": dict(template_sources["readout_checkpoint"]),
        "renderer_geometry_checkpoint": dict(
            template_sources["renderer_geometry_checkpoint"]
        ),
        "materializer_source": dict(materializer_record),
        "legacy_fp16_materializer_source": dict(
            template_sources["legacy_fp16_materializer_source"]
        ),
        "genuine_mpr_sidecar": dict(mpr_sidecar_record),
        "raw_positive_geometry_template": dict(template_positive_record),
        "raw_negative_geometry_template": dict(template_negative_record),
    }
    valid = torch.as_tensor(mpr["valid"]).detach().bool().cpu().contiguous()
    mpr_metadata = mpr.get("metadata")
    if not isinstance(mpr_metadata, Mapping):
        raise ValueError("genuine MPR metadata is missing")
    selected_frames = mpr_metadata.get("selected_frame_indices")
    if not isinstance(selected_frames, list):
        raise ValueError("genuine MPR selected-frame metadata is missing")
    geometry = dict(template_authority["geometry_axis"])
    geometry["valid_sha256"] = frozen.tensor_sha256_typed(valid)
    global_rows = torch.where(valid)[0]
    scale_axis = [
        {"id": scale_id, "value": radius, "unit": "meter"}
        for scale_id, radius in zip(template.scale_ids, template.scale_radii_m)
    ]
    authority = {
        "schema_version": 3,
        "artifact_type": "radio_gs_lerf_multiscale_primitive_query_score_cache",
        "contract": AUTHORITY_CONTRACT,
        "score_semantics": "raw_independent_normalized_cosine",
        "score_formula": SCORE_FORMULA,
        "score_implementation": str(Path(__file__).resolve()),
        "score_dtype": "torch.float32",
        "score_role": score_role,
        "precision_contract": {
            "normalization_dtype": "torch.float32",
            "matmul_dtype": "torch.float32",
            "storage_dtype": "torch.float32",
            "post_matmul_quantization": False,
            "legacy_fp16_default_changed": False,
        },
        "scale_axis": scale_axis,
        "query_axis": dict(query_axis),
        "geometry_axis": geometry,
        "descriptor_axis": {
            "dimension": FEATURE_DIMENSION,
            "row_storage": "dense_zero_invalid_rows",
            "valid_rows": int(valid.sum()),
            "features_sha256": features_sha256,
            "global_rows_sha256": frozen.tensor_sha256_typed(global_rows),
            "source": "sealed_genuine_official_crop_summary_mpr_features",
            "feature_normalization": "per_valid_row_l2_fp32",
            "scale_semantics": (
                "single_MPR_descriptor_replicated_bitwise_to_three_template_slots"
            ),
            "template_binding_semantics": (
                "accepted_O2_raw_cache_supplies_only_row_geometry_native_scale_axis_"
                "and_required_legacy_checkpoint_container_fields"
            ),
            "template_representation_inherited": False,
            "mpr_lifting_contract": {
                "construction": str(mpr_metadata.get("construction", "")),
                "aggregation_mode": str(
                    mpr_metadata.get("aggregation_mode", "")
                ),
                "registration_weight_mode": str(
                    mpr_metadata.get("registration_weight_mode", "")
                ),
                "raster_view_fusion": str(
                    mpr_metadata.get("raster_view_fusion", "")
                ),
                "num_declared_views": int(
                    mpr_metadata.get("num_declared_views", -1)
                ),
                "selected_frame_indices_sha256": canonical_json_sha256(
                    selected_frames
                ),
                "query_independent": bool(
                    mpr_metadata.get(
                        "query_independent",
                        mpr_metadata.get("observation_lifting_contract", {}).get(
                            "query_independent", False
                        )
                        if isinstance(
                            mpr_metadata.get("observation_lifting_contract"),
                            Mapping,
                        )
                        else False,
                    )
                ),
            },
            "capacity_match_to_O2_template": False,
            "capacity_mismatch_reason": (
                "genuine_MPR_uses_its_sealed_historical_lifting_operator_while_O2_"
                "template_is_used_only_for_geometry_scale_and_container_compatibility"
            ),
        },
        "query_scores_sha256": frozen.tensor_sha256_typed(query_scores),
        "source_artifacts": sources,
        "consumer_contracts": {
            "direct3d": {
                "contract": CACHE_CONTRACT,
                "tensor_layout": "[primitive_row,scale,query]",
                "scale_selection": "downstream_only_not_run_in_this_premetric_stage",
            }
        },
        "calibration_constraints": {
            "softmax_applied": False,
            "temperature_applied": False,
            "negative_probability_applied": False,
            "knn_applied": False,
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
        "contract": CACHE_CONTRACT,
        "query_scores": query_scores,
        "query_ids": list(template.query_ids),
        "scale_ids": list(template.scale_ids),
        "scale_radii_m": list(template.scale_radii_m),
        "xyz": torch.as_tensor(mpr["xyz"])
        .detach()
        .float()
        .cpu()
        .contiguous(),
        "valid": valid,
        "geometry_fingerprint": dict(mpr["geometry_fingerprint"]),
        "field_checkpoint_sha256": template.field_checkpoint_sha256,
        "readout_checkpoint_sha256": template.readout_checkpoint_sha256,
        "renderer_geometry_checkpoint_sha256": (
            template.renderer_geometry_checkpoint_sha256
        ),
        "authority": authority,
    }
    return payload, authority


def _build_report(
    *,
    scene_id: str,
    score_role: str,
    payload: Mapping[str, Any],
    template: frozen.OursMultiscaleQueryScoreCache,
    output_record: Mapping[str, str],
) -> dict[str, Any]:
    query_scores = torch.as_tensor(payload["query_scores"])
    valid = torch.as_tensor(payload["valid"]).bool()
    valid_scores = query_scores[valid][:, 0]
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete_source_only_premetric_genuine_mpr_raw_query_scores",
        "scene_id": scene_id,
        "score_role": score_role,
        "output_cache": dict(output_record),
        "authority": payload["authority"],
        "source_diagnostics": {
            "total_primitives": int(query_scores.shape[0]),
            "mpr_valid_primitives": int(valid.sum()),
            "template_valid_primitives": int(template.valid.sum()),
            "valid_intersection": int((valid & template.valid).sum()),
            "valid_mpr_only": int((valid & ~template.valid).sum()),
            "valid_template_only": int((~valid & template.valid).sum()),
            "scale_slots_bitwise_identical": bool(
                torch.equal(query_scores[:, 0], query_scores[:, 1])
                and torch.equal(query_scores[:, 1], query_scores[:, 2])
            ),
            "invalid_rows_exact_zero": bool(query_scores[~valid].eq(0).all()),
            "valid_score_min": float(valid_scores.min()),
            "valid_score_mean": float(valid_scores.mean()),
            "valid_score_max": float(valid_scores.max()),
            "finite": bool(torch.isfinite(query_scores).all()),
        },
        "access_audit": access_audit(),
        "metric_execution_authorized": False,
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    outputs = _canonical_new_outputs(args)

    mpr_payload, mpr_sha256, mpr_source = load_torch_mapping(
        args.mpr_cache,
        expected_sha256=args.mpr_cache_sha256,
        map_location="cpu",
        label="sealed genuine official crop-summary MPR",
    )
    mpr = validate_mpr_cache_payload(
        mpr_payload,
        expected_feature_space="semantic_descriptor",
        require_reliability=True,
        require_formal_safety=False,
    )
    sidecar, sidecar_sha256, sidecar_source = load_json_object(
        args.mpr_sidecar,
        expected_sha256=args.mpr_sidecar_sha256,
        label="genuine MPR sidecar",
    )
    if (
        int(sidecar.get("num_gaussians", -1)) != int(mpr["xyz"].shape[0])
        or int(sidecar.get("valid_count", -1)) != int(mpr["valid"].sum())
        or sidecar.get("metadata") != mpr.get("metadata")
    ):
        raise ValueError("genuine MPR sidecar differs from tensor payload")

    (
        template_raw,
        template_negative_raw,
        template,
        template_negative,
        template_positive_record,
        template_negative_record,
    ) = template_loader._load_pair(
        args.raw_template_positive,
        args.raw_template_positive_sha256,
        args.raw_template_negative,
        args.raw_template_negative_sha256,
    )
    _validate_geometry(mpr, template_raw)
    _validate_geometry(mpr, template_negative_raw)
    if tuple(template.scale_ids) != tuple(
        str(float(value)) for value in template.scale_radii_m
    ):
        raise ValueError("raw template native scale identifiers differ")

    positive_bank, positive_bank_sha256, positive_bank_source = load_torch_mapping(
        args.frozen_positive_text_bank,
        expected_sha256=args.frozen_positive_text_bank_sha256,
        map_location="cpu",
        label="frozen all-query SigLIP2 bank",
    )
    negative_bank, negative_bank_sha256, negative_bank_source = load_torch_mapping(
        args.frozen_negative_text_bank,
        expected_sha256=args.frozen_negative_text_bank_sha256,
        map_location="cpu",
        label="frozen canonical-negative SigLIP2 bank",
    )
    positive_embeddings, positive_query_axis = select_text_axis(
        positive_bank, template.query_ids
    )
    negative_embeddings, negative_query_axis = select_text_axis(
        negative_bank, template_negative.query_ids
    )
    if tuple(template_negative.query_ids) != tuple(frozen.NEGATIVE_PROMPTS):
        raise ValueError("canonical-negative query axis differs")
    positive_scores = compute_raw_scores(
        mpr["features"],
        mpr["valid"],
        positive_embeddings,
        chunk_size=args.chunk_size,
    )
    negative_scores = compute_raw_scores(
        mpr["features"],
        mpr["valid"],
        negative_embeddings,
        chunk_size=args.chunk_size,
    )
    mpr_record = {"path": str(mpr_source), "sha256": mpr_sha256}
    sidecar_record = {"path": str(sidecar_source), "sha256": sidecar_sha256}
    positive_bank_record = {
        "path": str(positive_bank_source),
        "sha256": positive_bank_sha256,
    }
    negative_bank_record = {
        "path": str(negative_bank_source),
        "sha256": negative_bank_sha256,
    }
    materializer_record = file_record(Path(__file__).resolve())
    features_sha = frozen.tensor_sha256_typed(
        torch.as_tensor(mpr["features"]).detach().cpu().contiguous()
    )
    common = {
        "mpr": mpr,
        "mpr_record": mpr_record,
        "mpr_sidecar_record": sidecar_record,
        "materializer_record": materializer_record,
        "template_positive_record": template_positive_record,
        "template_negative_record": template_negative_record,
        "features_sha256": features_sha,
    }
    positive_payload, _positive_authority = _build_payload(
        **common,
        template_raw=template_raw,
        template=template,
        query_scores=positive_scores,
        query_axis=positive_query_axis,
        text_bank_record=positive_bank_record,
        score_role="positive_benchmark_queries",
    )
    negative_payload, _negative_authority = _build_payload(
        **common,
        template_raw=template_negative_raw,
        template=template_negative,
        query_scores=negative_scores,
        query_axis=negative_query_axis,
        text_bank_record=negative_bank_record,
        score_role="canonical_negative_queries",
    )

    # Validate the completed pair in memory before publishing either cache.
    for payload, cache in (
        (positive_payload, template),
        (negative_payload, template_negative),
    ):
        frozen.validate_ours_multiscale_query_score_cache(
            payload,
            expected_xyz=payload["xyz"],
            expected_query_ids=cache.query_ids,
            expected_renderer_geometry_checkpoint_sha256=(
                cache.renderer_geometry_checkpoint_sha256
            ),
        )
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    write_torch_noclobber(outputs["positive_cache"], positive_payload)
    write_torch_noclobber(outputs["negative_cache"], negative_payload)
    positive_report = _build_report(
        scene_id=args.scene_id,
        score_role="positive_benchmark_queries",
        payload=positive_payload,
        template=template,
        output_record=file_record(outputs["positive_cache"]),
    )
    negative_report = _build_report(
        scene_id=args.scene_id,
        score_role="canonical_negative_queries",
        payload=negative_payload,
        template=template_negative,
        output_record=file_record(outputs["negative_cache"]),
    )
    write_frozen_json(outputs["positive_report"], positive_report)
    write_frozen_json(outputs["negative_report"], negative_report)
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete_source_only_premetric_genuine_mpr_raw_query_score_pair",
        "scene_id": args.scene_id,
        "positive": {
            **positive_report,
            "output_report": file_record(outputs["positive_report"]),
        },
        "canonical_negative": {
            **negative_report,
            "output_report": file_record(outputs["negative_report"]),
        },
        "access_audit": access_audit(),
        "metric_execution_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--mpr-cache", required=True)
    parser.add_argument("--mpr-cache-sha256", required=True)
    parser.add_argument("--mpr-sidecar", required=True)
    parser.add_argument("--mpr-sidecar-sha256", required=True)
    parser.add_argument("--raw-template-positive", required=True)
    parser.add_argument("--raw-template-positive-sha256", required=True)
    parser.add_argument("--raw-template-negative", required=True)
    parser.add_argument("--raw-template-negative-sha256", required=True)
    parser.add_argument("--frozen-positive-text-bank", required=True)
    parser.add_argument("--frozen-positive-text-bank-sha256", required=True)
    parser.add_argument("--frozen-negative-text-bank", required=True)
    parser.add_argument("--frozen-negative-text-bank-sha256", required=True)
    parser.add_argument("--output-positive-cache", required=True)
    parser.add_argument("--output-positive-report", required=True)
    parser.add_argument("--output-negative-cache", required=True)
    parser.add_argument("--output-negative-report", required=True)
    parser.add_argument("--chunk-size", type=int, default=4096)
    return parser


def main() -> None:
    result = materialize(build_parser().parse_args())
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "AUTHORITY_CONTRACT",
    "CACHE_CONTRACT",
    "CACHE_VERSION",
    "FEATURE_DIMENSION",
    "SCHEMA",
    "access_audit",
    "compute_raw_scores",
    "materialize",
    "select_text_axis",
]
