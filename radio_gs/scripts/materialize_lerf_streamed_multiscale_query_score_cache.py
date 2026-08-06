#!/usr/bin/env python3
"""Wrap cold-streamed three-scale LERF scores in the frozen v2 authority.

This is an execution-equivalent alternative to materializing an
``[M,3,1536]`` descriptor cache and then taking query cosines.  The upstream
builder preserves each descriptor scale, quantizes it to fp16, evaluates the
same independent normalized cosine on CPU, and stores only ``[N,3,Q]`` fp16
scores.  This wrapper verifies every source binding before emitting the exact
Direct3D cache contract consumed by the frozen evaluator.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from radio_gs.scripts.materialize_lerf_multiscale_query_score_cache import (
    ARTIFACT_TYPE,
    DESCRIPTOR_DIMENSION,
    DIRECT3D_CONTRACT,
    SCALE_COUNT,
    SCHEMA_VERSION,
    SCORE_SEMANTICS_2D,
    SCORING_FORMULA,
    SCORING_IMPLEMENTATION,
    SHARED_AUTHORITY_CONTRACT,
    SIGLIP2_MODEL_NAME,
    SIGLIP2_TEXT_CANONICALIZATION,
    _canonical_output,
    _direct_xyz_sha256,
    _file_record,
    _geometry_fingerprint,
    _preflight_outputs,
    _readout_native_scales,
    _renderer_checkpoint_xyz,
    _require_sha256,
    _scale_id,
    _tensor_sha256,
    _validate_text_query_cache,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_sha_bound_project_checkpoint_mapping,
    load_torch_mapping,
    sha256_file,
    stable_descriptor_load,
    write_frozen_json,
    write_torch_noclobber,
)


STREAMED_FEATURE_SPACE = "primitive_text_query_scores_multiscale_unreduced"
STREAMED_CONSTRUCTION = (
    "cold_streaming_surface_region_readout_then_independent_cosine"
)
STREAMED_SCALE_AGGREGATION = "none_frozen_downstream_only"
STREAMED_COMPLETION_REASON = (
    "frozen_direct3d_requires_raw_unreduced_scale_scores"
)
PROBABILITY_STREAMED_SCHEMA_VERSION = 4
PROBABILITY_FEATURE_SPACE = (
    "primitive_canonical_negative_probability_multiscale_unreduced"
)
PROBABILITY_SCORE_SEMANTICS = "canonical_negative_bernoulli_probability"
PROBABILITY_DIRECT3D_VERSION = 3
PROBABILITY_DIRECT3D_CONTRACT = (
    "radio_gs.ours_lerf_direct3d_multiscale_query_probabilities.v3"
)
PROBABILITY_AUTHORITY_CONTRACT = (
    "radio_gs.lerf_multiscale_query_probability_authority.v3"
)
PROBABILITY_CONSTRUCTIONS = {
    "surface_residual_codebook_slotwise_head_then_query_router",
    "surface_residual_codebook_exact_frozen_v2_slot0",
}
PROBABILITY_SCORING = {
    "canonical_negative_bernoulli_query_router_v1",
    "canonical_negative_bernoulli_frozen_v2_slot0",
}


def _resolve_declared_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    try:
        return Path(value).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} path cannot be resolved") from exc


def _require_source_binding(
    metadata: Mapping[str, Any],
    *,
    role: str,
    source: Path,
    digest: str,
) -> None:
    declared = _resolve_declared_path(metadata.get(role), label=role)
    if declared != source:
        raise ValueError(f"streamed score cache {role} path differs")
    declared_digest = _require_sha256(
        metadata.get(f"{role}_sha256"), label=f"{role}_sha256"
    )
    if declared_digest != digest:
        raise ValueError(f"streamed score cache {role} SHA256 differs")


def _validated_provenance_artifact(
    provenance: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, str]:
    path = _resolve_declared_path(provenance.get(role), label=role)
    digest = _require_sha256(
        provenance.get(f"{role}_sha256"), label=f"{role}_sha256"
    )
    if sha256_file(path) != digest:
        raise ValueError(f"streamed score cache {role} SHA256 differs")
    return {"path": str(path), "sha256": digest}


def _validate_streamed_scores(
    payload: Mapping[str, Any],
    *,
    text_query_cache_path: Path,
    text_query_cache_sha256: str,
    field_checkpoint_path: Path,
    field_checkpoint_sha256: str,
    readout_checkpoint_path: Path,
    readout_checkpoint_sha256: str,
    readout_native_scales: tuple[float, ...],
) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("streamed score cache metadata must be a mapping")
    probability_mode = (
        metadata.get("score_semantics") == PROBABILITY_SCORE_SEMANTICS
    )
    required = {
        "schema_version": (
            PROBABILITY_STREAMED_SCHEMA_VERSION if probability_mode else 3
        ),
        "feature_space": (
            PROBABILITY_FEATURE_SPACE if probability_mode else STREAMED_FEATURE_SPACE
        ),
        "scale_aggregation": STREAMED_SCALE_AGGREGATION,
        "scale_count": SCALE_COUNT,
        "semantic_cache_materialized": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": True,
    }
    if probability_mode:
        if metadata.get("construction") not in PROBABILITY_CONSTRUCTIONS:
            raise ValueError("streamed probability construction differs")
        if metadata.get("scoring") not in PROBABILITY_SCORING:
            raise ValueError("streamed probability scoring route differs")
        if metadata.get("value_range") != [0.0, 1.0]:
            raise ValueError("streamed probability value range differs")
        if float(metadata.get("logit_scale", -1.0)) != 10.0:
            raise ValueError("streamed probability logit scale differs")
        if metadata.get("generic_negative_queries") != [
            "object",
            "things",
            "stuff",
            "texture",
        ]:
            raise ValueError("streamed probability generic negatives differ")
        probability_route = str(metadata.get("probability_route", ""))
        if probability_route not in {
            "query_router_v1",
            "exact_frozen_v2_slot0_control",
        }:
            raise ValueError("streamed probability route differs")
    else:
        required.update(
            {
                "construction": STREAMED_CONSTRUCTION,
                "scoring": "raw_independent_normalized_cosine",
            }
        )
        probability_route = ""
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"streamed score cache metadata.{key} differs: "
                f"{metadata.get(key)!r} vs {expected!r}"
            )
    completion = metadata.get("completion")
    if completion != {"applied": False, "reason": STREAMED_COMPLETION_REASON}:
        raise ValueError("streamed score cache must not apply completion")

    raw_radii = metadata.get("scale_radii_m")
    if not isinstance(raw_radii, (list, tuple)) or len(raw_radii) != SCALE_COUNT:
        raise ValueError("streamed score cache must bind exactly three radii")
    try:
        radii = tuple(float(value) for value in raw_radii)
    except (TypeError, ValueError) as exc:
        raise ValueError("streamed score cache radii are malformed") from exc
    if (
        any(not math.isfinite(value) or value <= 0.0 for value in radii)
        or len(set(radii)) != SCALE_COUNT
        or any(left >= right for left, right in zip(radii, radii[1:]))
    ):
        raise ValueError("streamed score cache radii must be positive and ordered")
    if radii != readout_native_scales:
        raise ValueError("streamed score scales differ from readout native scales")

    query_names = metadata.get("query_names")
    if not isinstance(query_names, (list, tuple)) or not all(
        isinstance(value, str) and value for value in query_names
    ):
        raise ValueError("streamed score query_names must be a non-empty string list")
    query_ids = tuple(query_names)
    if not query_ids or len(set(query_ids)) != len(query_ids):
        raise ValueError("streamed score query IDs must be non-empty and unique")
    if (
        _resolve_declared_path(
            metadata.get("text_embedding_cache"), label="text_embedding_cache"
        )
        != text_query_cache_path
    ):
        raise ValueError("streamed score text cache path differs")
    declared_text_sha = _require_sha256(
        metadata.get("text_embedding_cache_sha256"),
        label="text_embedding_cache_sha256",
    )
    if declared_text_sha != text_query_cache_sha256:
        raise ValueError("streamed score text cache SHA256 differs")

    implementation = metadata.get("streaming_implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("streamed score cache lacks implementation binding")
    implementation_path = _resolve_declared_path(
        implementation.get("path"), label="streaming_implementation"
    )
    implementation_sha = _require_sha256(
        implementation.get("sha256"), label="streaming_implementation.sha256"
    )
    if sha256_file(implementation_path) != implementation_sha:
        raise ValueError("streamed score implementation SHA256 differs")

    provenance = metadata.get("semantic_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("streamed score cache lacks semantic provenance")
    provenance_required = {
        "source": (
            "canonical_radio_surface_region_residual_codebook_query_router"
            if probability_mode
            else "canonical_radio_surface_region_readout"
        ),
        "query_set_invariant": not probability_mode,
        "official_summary_head": True,
        "custom_text_projection": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": probability_mode,
        "region_radii_m": list(radii),
    }
    for key, expected in provenance_required.items():
        if provenance.get(key) != expected:
            raise ValueError(f"streamed semantic provenance {key} differs")
    _require_source_binding(
        provenance,
        role="field_checkpoint",
        source=field_checkpoint_path,
        digest=field_checkpoint_sha256,
    )
    _require_source_binding(
        provenance,
        role="readout_checkpoint",
        source=readout_checkpoint_path,
        digest=readout_checkpoint_sha256,
    )
    radio_sha = _require_sha256(
        provenance.get("official_radio_checkpoint_sha256"),
        label="official_radio_checkpoint_sha256",
    )
    probability_sources: dict[str, dict[str, str]] = {}
    if probability_mode:
        if (
            provenance.get("representation_query_set_invariant") is not True
            or provenance.get("query_router_query_dependent") is not True
            or provenance.get("exact_frozen_v2_slot0_control") is not True
            or provenance.get("query_router_score_contract")
            != "canonical_negative_bernoulli_query_first"
            or float(provenance.get("query_router_logit_scale", -1.0)) != 10.0
            or provenance.get("slot_projection_contract")
            != "four_independent_official_head_calls_Bx1x1280"
        ):
            raise ValueError("streamed probability provenance differs")
        for role in (
            "residual_codebook_checkpoint",
            "query_router_checkpoint",
            "generic_negative_text_cache",
        ):
            probability_sources[role] = _validated_provenance_artifact(
                provenance, role=role
            )
        if provenance.get("generic_negative_queries") != [
            "object",
            "things",
            "stuff",
            "texture",
        ]:
            raise ValueError("streamed probability provenance negatives differ")

    xyz = payload.get("xyz")
    valid = payload.get("valid")
    scores = payload.get("features")
    if not isinstance(xyz, torch.Tensor) or not xyz.is_floating_point():
        raise ValueError("streamed score xyz must be floating")
    xyz = xyz.detach().cpu().float().contiguous()
    geometry = _geometry_fingerprint(xyz)
    count = int(xyz.shape[0])
    if (
        not isinstance(valid, torch.Tensor)
        or valid.dtype != torch.bool
        or tuple(valid.shape) != (count,)
    ):
        raise ValueError("streamed score valid mask is not row aligned")
    valid = valid.detach().cpu().contiguous()
    if not bool(valid.any()):
        raise ValueError("streamed score cache must keep a valid primitive")
    if (
        not isinstance(scores, torch.Tensor)
        or scores.dtype != torch.float16
        or tuple(scores.shape) != (count, SCALE_COUNT, len(query_ids))
    ):
        raise ValueError(
            "streamed scores must be fp16 [primitive,3,query] in frozen order"
        )
    scores = scores.detach().cpu().contiguous()
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("streamed scores contain NaN or infinity")
    if probability_mode:
        if bool((scores.float() < 0.0).any()) or bool(
            (scores.float() > 1.0).any()
        ):
            raise ValueError("Bernoulli probability scores must lie in [0,1]")
    elif bool((scores.float().abs() > 1.0005).any()):
        raise ValueError("normalized cosine scores exceed their valid range")
    if bool(scores[~valid].any()):
        raise ValueError("invalid primitive rows must have zero streamed scores")
    if provenance.get("field_geometry_xyz_sha256") != geometry["xyz_sha256"]:
        raise ValueError("streamed score field geometry fingerprint differs")

    return {
        "xyz": xyz,
        "valid": valid,
        "query_scores": scores,
        "query_ids": query_ids,
        "native_scales": radii,
        "geometry_fingerprint": geometry,
        "official_radio_checkpoint_sha256": radio_sha,
        "streaming_implementation_path": implementation_path,
        "streaming_implementation_sha256": implementation_sha,
        "score_semantics": (
            PROBABILITY_SCORE_SEMANTICS
            if probability_mode
            else "raw_independent_normalized_cosine"
        ),
        "probability_route": probability_route,
        "probability_sources": probability_sources,
    }


def materialize_streamed(
    *,
    streamed_score_cache: str | Path,
    streamed_score_cache_sha256: str,
    text_query_cache: str | Path,
    text_query_cache_sha256: str,
    field_checkpoint: str | Path,
    field_checkpoint_sha256: str,
    readout_checkpoint: str | Path,
    readout_checkpoint_sha256: str,
    renderer_geometry_checkpoint: str | Path,
    renderer_geometry_checkpoint_sha256: str,
    output: str | Path,
) -> dict[str, Any]:
    """Validate a streamed derivative and emit one immutable v2 cache."""

    streamed_expected = _require_sha256(
        streamed_score_cache_sha256, label="streamed_score_cache_sha256"
    )
    text_expected = _require_sha256(
        text_query_cache_sha256, label="text_query_cache_sha256"
    )
    field_expected = _require_sha256(
        field_checkpoint_sha256, label="field_checkpoint_sha256"
    )
    readout_expected = _require_sha256(
        readout_checkpoint_sha256, label="readout_checkpoint_sha256"
    )
    renderer_expected = _require_sha256(
        renderer_geometry_checkpoint_sha256,
        label="renderer_geometry_checkpoint_sha256",
    )
    output_path = _canonical_output(output)
    report_path = output_path.with_suffix(output_path.suffix + ".json")
    _preflight_outputs(output_path, report_path)

    streamed_payload, streamed_digest, streamed_path = load_torch_mapping(
        streamed_score_cache,
        expected_sha256=streamed_expected,
        map_location="cpu",
        label="streamed multiscale query score cache",
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

    streamed = _validate_streamed_scores(
        streamed_payload,
        text_query_cache_path=text_path,
        text_query_cache_sha256=text_digest,
        field_checkpoint_path=field_path,
        field_checkpoint_sha256=field_digest,
        readout_checkpoint_path=readout_path,
        readout_checkpoint_sha256=readout_digest,
        readout_native_scales=_readout_native_scales(readout_payload),
    )
    text = _validate_text_query_cache(text_payload)
    if streamed["query_ids"] != text["query_ids"]:
        raise ValueError("streamed score query order differs from frozen text cache")
    renderer_xyz = _renderer_checkpoint_xyz(renderer_payload)
    if renderer_xyz.shape != streamed["xyz"].shape or not torch.equal(
        renderer_xyz, streamed["xyz"]
    ):
        raise ValueError("renderer geometry xyz/count/row-order differs from scores")

    query_scores = streamed["query_scores"]
    query_scores_sha = _tensor_sha256(query_scores)
    query_order_sha = canonical_json_sha256(list(streamed["query_ids"]))
    scale_ids = tuple(_scale_id(value) for value in streamed["native_scales"])
    scale_records = [
        {"id": scale_id, "value": radius, "unit": "meter"}
        for scale_id, radius in zip(scale_ids, streamed["native_scales"])
    ]
    sources = {
        "streamed_query_score_cache": _file_record(
            streamed_path, streamed_digest
        ),
        "text_query_cache": _file_record(text_path, text_digest),
        "field_checkpoint": _file_record(field_path, field_digest),
        "readout_checkpoint": _file_record(readout_path, readout_digest),
        "renderer_geometry_checkpoint": _file_record(
            renderer_path, renderer_digest
        ),
        "streaming_source": _file_record(
            streamed["streaming_implementation_path"],
            streamed["streaming_implementation_sha256"],
        ),
        "materializer_source": _file_record(
            implementation_path, implementation_digest
        ),
        **streamed["probability_sources"],
    }
    probability_mode = streamed["score_semantics"] == PROBABILITY_SCORE_SEMANTICS
    direct_contract = (
        PROBABILITY_DIRECT3D_CONTRACT if probability_mode else DIRECT3D_CONTRACT
    )
    authority_contract = (
        PROBABILITY_AUTHORITY_CONTRACT
        if probability_mode
        else SHARED_AUTHORITY_CONTRACT
    )
    schema_version = (
        PROBABILITY_DIRECT3D_VERSION if probability_mode else SCHEMA_VERSION
    )
    if probability_mode:
        score_formula = (
            "slotwise sigmoid(10*(positive_cosine-max_generic_negative_cosine)); "
            "frozen query-conditioned residual attention"
            if streamed["probability_route"] == "query_router_v1"
            else "sigmoid(10*(slot0_positive_cosine-max_slot0_generic_negative_cosine))"
        )
        score_implementation = (
            "radio_gs.interfaces.surface_region_query_router."
            "SurfaceRegionQueryRouterV1"
        )
    else:
        score_formula = SCORING_FORMULA
        score_implementation = SCORING_IMPLEMENTATION
    authority = {
        "schema_version": schema_version,
        "artifact_type": ARTIFACT_TYPE,
        "contract": authority_contract,
        "score_semantics": streamed["score_semantics"],
        "score_formula": score_formula,
        "score_implementation": score_implementation,
        "score_dtype": "torch.float16",
        **(
            {
                "probability_route": streamed["probability_route"],
                "value_range": [0.0, 1.0],
                "logit_scale": 10.0,
                "generic_negative_queries": [
                    "object",
                    "things",
                    "stuff",
                    "texture",
                ],
            }
            if probability_mode
            else {}
        ),
        "scale_axis": scale_records,
        "query_axis": {
            "ids": list(streamed["query_ids"]),
            "order_sha256": query_order_sha,
            "embedding_tensor_sha256": text["embedding_tensor_sha256"],
            "text_encoder": "official_siglip2_g",
            "model_name": SIGLIP2_MODEL_NAME,
            "text_canonicalization": SIGLIP2_TEXT_CANONICALIZATION,
            "prompt_templates": ["{query}"],
        },
        "geometry_axis": {
            **streamed["geometry_fingerprint"],
            "valid_sha256": _tensor_sha256(streamed["valid"]),
            "field_checkpoint_sha256": field_digest,
            "readout_checkpoint_sha256": readout_digest,
            "renderer_geometry_checkpoint_sha256": renderer_digest,
            "renderer_xyz_sha256": _direct_xyz_sha256(renderer_xyz),
        },
        "descriptor_axis": {
            "dimension": DESCRIPTOR_DIMENSION,
            "materialized": False,
            "execution_representation": "streamed_scalar_scores_only",
            "valid_rows": int(streamed["valid"].sum()),
            "streamed_query_score_cache_sha256": streamed_digest,
            "readout_checkpoint_sha256": readout_digest,
            "official_radio_checkpoint_sha256": streamed[
                "official_radio_checkpoint_sha256"
            ],
        },
        "query_scores_sha256": query_scores_sha,
        "source_artifacts": sources,
        "consumer_contracts": {
            "direct3d": {
                "contract": direct_contract,
                "tensor_layout": "[primitive_row,scale,query]",
                "scale_selection": "downstream_frozen_VALA_readout_only",
            },
            "lerf2d_scalar_map_renderer": {
                "score_semantics": (
                    PROBABILITY_SCORE_SEMANTICS
                    if probability_mode
                    else SCORE_SEMANTICS_2D
                ),
                "tensor_layout_before_render": "[primitive_row,scale,query]",
                "scale_ids": list(scale_ids),
                "query_text_axis": list(streamed["query_ids"]),
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
        "version": schema_version,
        "contract": direct_contract,
        "query_scores": query_scores,
        "query_ids": list(streamed["query_ids"]),
        "scale_ids": list(scale_ids),
        "scale_radii_m": list(streamed["native_scales"]),
        "xyz": streamed["xyz"],
        "valid": streamed["valid"],
        "geometry_fingerprint": streamed["geometry_fingerprint"],
        "field_checkpoint_sha256": field_digest,
        "readout_checkpoint_sha256": readout_digest,
        "renderer_geometry_checkpoint_sha256": renderer_digest,
        "authority": authority,
    }
    write_torch_noclobber(output_path, payload)
    output_digest = sha256_file(output_path)
    report = {
        "schema_version": schema_version,
        "artifact_type": ARTIFACT_TYPE,
        "status": "complete_calibration_free_streamed_query_score_materialization",
        "query_score_cache": _file_record(output_path, output_digest),
        "query_score_shape_n3q": list(query_scores.shape),
        "valid_primitives": int(streamed["valid"].sum()),
        "total_primitives": int(streamed["valid"].numel()),
        "queries": int(query_scores.shape[2]),
        "execution": {
            "device": "cpu",
            "descriptor_cache_materialized": False,
            "changes_method": False,
        },
        "score_semantics": streamed["score_semantics"],
        "probability_route": streamed["probability_route"],
        "shared_renderer_authority": authority,
    }
    write_frozen_json(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streamed-score-cache", required=True)
    parser.add_argument("--streamed-score-cache-sha256", required=True)
    parser.add_argument("--text-query-cache", required=True)
    parser.add_argument("--text-query-cache-sha256", required=True)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--field-checkpoint-sha256", required=True)
    parser.add_argument("--readout-checkpoint", required=True)
    parser.add_argument("--readout-checkpoint-sha256", required=True)
    parser.add_argument("--renderer-geometry-checkpoint", required=True)
    parser.add_argument("--renderer-geometry-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = materialize_streamed(
        streamed_score_cache=args.streamed_score_cache,
        streamed_score_cache_sha256=args.streamed_score_cache_sha256,
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
    )
    print(report["query_score_cache"]["path"])


if __name__ == "__main__":
    main()
