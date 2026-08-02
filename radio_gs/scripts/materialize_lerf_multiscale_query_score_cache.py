#!/usr/bin/env python3
"""Materialize frozen three-scale LERF primitive/text cosine responses.

The adapter is deliberately calibration-free.  It only combines an immutable,
query-independent SurfaceRegion descriptor cache with an immutable official
SigLIP2 query cache.  It never opens benchmark images, annotations, masks, or
metrics, and it does not expose a temperature, threshold, peak normalization,
or scale-reduction option.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from radio_gs.evaluation.openclip_readout import cosine_logits
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_sha_bound_project_checkpoint_mapping,
    load_torch_mapping,
    sha256_file,
    stable_descriptor_load,
    write_frozen_json,
    write_torch_noclobber,
)


SCHEMA_VERSION = 2
ARTIFACT_TYPE = "radio_gs_lerf_multiscale_primitive_query_score_cache"
DIRECT3D_CONTRACT = "radio_gs.ours_lerf_direct3d_multiscale_query_scores.v2"
SHARED_AUTHORITY_CONTRACT = "radio_gs.lerf_multiscale_query_score_authority.v2"
SCALE_COUNT = 3
DESCRIPTOR_DIMENSION = 1536
SIGLIP2_MODEL_NAME = "google/siglip2-giant-opt-patch16-384"
SIGLIP2_TEXT_CANONICALIZATION = "official_c_radio_siglip2_g"
SCORING_FORMULA = "l2_normalize(descriptor) @ l2_normalize(text_embedding).T"
SCORING_IMPLEMENTATION = "radio_gs.evaluation.openclip_readout.cosine_logits"
SCORE_SEMANTICS_2D = "raw_query_relevance_pre_occam_activation"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _canonical_output(path: str | Path) -> Path:
    raw = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    raw.parent.mkdir(parents=True, exist_ok=True)
    return raw.parent.resolve(strict=True) / raw.name


def _preflight_outputs(output: Path, report: Path) -> None:
    for path in (output, report):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"immutable output already exists: {path}")


def _direct_xyz_sha256(xyz: torch.Tensor) -> str:
    """Match ``eval_lerf_direct_3d_selection.tensor_sha256_float32``."""

    values = torch.as_tensor(xyz).detach().cpu().float().contiguous()
    digest = hashlib.sha256()
    for start in range(0, int(values.shape[0]), 65_536):
        digest.update(values[start : start + 65_536].numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    """Hash dtype, shape, and tensor bytes without a whole-tensor copy."""

    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.layout != torch.strided:
        raise ValueError("authority tensor hashing requires a strided tensor")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    if tensor.ndim == 0:
        digest.update(tensor.contiguous().numpy().tobytes(order="C"))
    else:
        for start in range(0, int(tensor.shape[0]), 4096):
            digest.update(
                tensor[start : start + 4096]
                .contiguous()
                .numpy()
                .tobytes(order="C")
            )
    return digest.hexdigest()


def _geometry_fingerprint(xyz: torch.Tensor) -> dict[str, Any]:
    values = torch.as_tensor(xyz).detach().cpu().float()
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"descriptor cache xyz must be [N,3], got {tuple(values.shape)}")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("descriptor cache xyz contains NaN or infinity")
    if values.shape[0] == 0:
        minimum = maximum = mean = [0.0, 0.0, 0.0]
    else:
        minimum = [float(item) for item in values.min(dim=0).values.tolist()]
        maximum = [float(item) for item in values.max(dim=0).values.tolist()]
        mean = [float(item) for item in values.mean(dim=0).tolist()]
    return {
        "num_gaussians": int(values.shape[0]),
        "xyz_sha256": _direct_xyz_sha256(values),
        "xyz_min": minimum,
        "xyz_max": maximum,
        "xyz_mean": mean,
    }


def _scale_id(radius: float) -> str:
    """Return one stable human-readable ID without imposing a scale value."""

    return str(float(radius))


def _require_native_scales(metadata: Mapping[str, Any]) -> tuple[float, ...]:
    """Bind exactly three descriptor/readout-native scales in source order."""

    raw = metadata.get("region_radii_m")
    if not isinstance(raw, (list, tuple)) or len(raw) != SCALE_COUNT:
        raise ValueError(
            "descriptor cache metadata.region_radii_m must bind exactly "
            f"{SCALE_COUNT} native scales"
        )
    try:
        observed = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("descriptor cache region_radii_m is malformed") from exc
    if any(not math.isfinite(value) or value <= 0.0 for value in observed):
        raise ValueError("descriptor cache native scales must be finite and positive")
    if len(set(observed)) != SCALE_COUNT or any(
        left >= right for left, right in zip(observed, observed[1:])
    ):
        raise ValueError(
            "descriptor cache native scales must be unique and strictly increasing "
            "in their original readout order"
        )
    region_contract = metadata.get("region_contract")
    if not isinstance(region_contract, Mapping):
        raise ValueError("descriptor cache region_contract must be a mapping")
    contract_radii = region_contract.get("radii_m")
    if not isinstance(contract_radii, (list, tuple)):
        raise ValueError("descriptor cache region_contract.radii_m is malformed")
    try:
        contract_values = tuple(float(value) for value in contract_radii)
    except (TypeError, ValueError) as exc:
        raise ValueError("descriptor cache region_contract.radii_m is malformed") from exc
    if contract_values != observed:
        raise ValueError("descriptor cache region contract scale/order mismatch")
    return observed


def _require_metadata_file_binding(
    metadata: Mapping[str, Any],
    *,
    role: str,
    source: Path,
    digest: str,
) -> None:
    declared_digest = _require_sha256(
        metadata.get(f"{role}_sha256"),
        label=f"descriptor cache {role}_sha256",
    )
    if declared_digest != digest:
        raise ValueError(f"descriptor cache {role} SHA256 differs")
    raw_path = metadata.get(role)
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"descriptor cache {role} path is missing")
    try:
        declared_path = Path(raw_path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"descriptor cache {role} path cannot be resolved") from exc
    if declared_path != source:
        raise ValueError(f"descriptor cache {role} path differs")


def _readout_native_scales(payload: Mapping[str, Any]) -> tuple[float, ...]:
    if payload.get("schema_version") != 3:
        raise ValueError("surface-region readout checkpoint schema differs")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("surface-region readout checkpoint lacks provenance")
    region_contract = provenance.get("region_contract")
    if not isinstance(region_contract, Mapping):
        raise ValueError("surface-region readout checkpoint lacks region contract")
    raw = region_contract.get("radii_m")
    if not isinstance(raw, (list, tuple)) or len(raw) != SCALE_COUNT:
        raise ValueError("surface-region readout must bind exactly three native scales")
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("surface-region readout native scales are malformed") from exc
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("surface-region readout native scales must be finite and positive")
    if len(set(values)) != SCALE_COUNT or any(
        left >= right for left, right in zip(values, values[1:])
    ):
        raise ValueError(
            "surface-region readout native scales must be unique and strictly "
            "increasing in source order"
        )
    return values


def _renderer_checkpoint_xyz(payload: Mapping[str, Any]) -> torch.Tensor:
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("renderer geometry checkpoint lacks model_state_dict")
    xyz = state.get("_xyz")
    if not isinstance(xyz, torch.Tensor) or not xyz.is_floating_point():
        raise ValueError("renderer geometry checkpoint lacks floating _xyz")
    values = xyz.detach().cpu().float().contiguous()
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"renderer geometry _xyz must be [N,3], got {tuple(values.shape)}")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("renderer geometry _xyz contains NaN or infinity")
    return values


def _validate_descriptor_cache(
    payload: Mapping[str, Any],
    *,
    field_checkpoint_path: Path,
    field_checkpoint_sha256: str,
    readout_checkpoint_path: Path,
    readout_checkpoint_sha256: str,
    readout_native_scales: tuple[float, ...],
) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("descriptor cache metadata must be a mapping")
    required_metadata = {
        "feature_space": "official_siglip2_summary_descriptor_multiscale",
        "source": "canonical_radio_surface_region_readout",
        "query_set_invariant": True,
        "official_summary_head": True,
        "custom_text_projection": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    for field, expected in required_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"descriptor cache metadata.{field} differs: "
                f"expected {expected!r}, got {metadata.get(field)!r}"
            )
    native_scales = _require_native_scales(metadata)
    if native_scales != readout_native_scales:
        raise ValueError("descriptor cache native scale/order differs from readout checkpoint")
    _require_metadata_file_binding(
        metadata,
        role="field_checkpoint",
        source=field_checkpoint_path,
        digest=field_checkpoint_sha256,
    )
    _require_metadata_file_binding(
        metadata,
        role="readout_checkpoint",
        source=readout_checkpoint_path,
        digest=readout_checkpoint_sha256,
    )
    readout_sha = _require_sha256(
        metadata.get("readout_checkpoint_sha256"),
        label="descriptor cache readout_checkpoint_sha256",
    )
    radio_sha = _require_sha256(
        metadata.get("official_radio_checkpoint_sha256"),
        label="descriptor cache official_radio_checkpoint_sha256",
    )

    xyz = payload.get("xyz")
    valid = payload.get("valid")
    descriptors = payload.get("features_by_scale")
    if not isinstance(xyz, torch.Tensor) or not xyz.is_floating_point():
        raise ValueError("descriptor cache xyz must be a floating tensor")
    xyz = xyz.detach().cpu().float()
    geometry = _geometry_fingerprint(xyz)
    count = int(xyz.shape[0])
    if (
        not isinstance(valid, torch.Tensor)
        or valid.dtype != torch.bool
        or tuple(valid.shape) != (count,)
    ):
        raise ValueError(f"descriptor cache valid must be row-aligned bool [{count}]")
    valid = valid.detach().cpu()
    if not bool(valid.any()):
        raise ValueError("descriptor cache valid must keep at least one primitive")
    if not isinstance(descriptors, torch.Tensor) or not descriptors.is_floating_point():
        raise ValueError("descriptor cache features_by_scale must be a floating tensor")
    descriptors = descriptors.detach().cpu()
    if descriptors.ndim != 3 or tuple(descriptors.shape[1:]) != (
        SCALE_COUNT,
        DESCRIPTOR_DIMENSION,
    ):
        raise ValueError(
            "descriptor cache features_by_scale must be "
            f"[M,3,{DESCRIPTOR_DIMENSION}]"
        )

    if "global_rows" in payload:
        global_rows = payload["global_rows"]
        if (
            not isinstance(global_rows, torch.Tensor)
            or global_rows.dtype != torch.int64
            or global_rows.ndim != 1
        ):
            raise ValueError("descriptor cache global_rows must be an int64 vector")
        global_rows = global_rows.detach().cpu()
        if descriptors.shape[0] != global_rows.numel():
            raise ValueError("sparse descriptors do not align with global_rows")
        if not torch.equal(global_rows, torch.where(valid)[0]):
            raise ValueError("descriptor cache global_rows must exactly equal where(valid)")
        row_storage = "sparse_valid_rows"
    else:
        if descriptors.shape[0] != count:
            raise ValueError("dense descriptors do not align with xyz/valid")
        global_rows = torch.where(valid)[0]
        descriptors = descriptors[global_rows]
        row_storage = "dense_rows_selected_by_valid"

    if not bool(torch.isfinite(descriptors).all()):
        raise ValueError("descriptor cache features_by_scale contains NaN or infinity")
    norms = descriptors.float().norm(dim=-1)
    if bool((norms <= 1e-8).any()):
        raise ValueError("valid multiscale descriptor rows must be nonzero")

    source_geometry = payload.get("geometry_fingerprint")
    if source_geometry is not None:
        if not isinstance(source_geometry, Mapping):
            raise ValueError("descriptor cache geometry_fingerprint must be a mapping")
        if int(source_geometry.get("num_gaussians", -1)) != count:
            raise ValueError("descriptor cache geometry_fingerprint count differs")
        if source_geometry.get("xyz_sha256") != geometry["xyz_sha256"]:
            raise ValueError("descriptor cache geometry_fingerprint xyz SHA256 differs")
    if metadata.get("field_geometry_xyz_sha256") != geometry["xyz_sha256"]:
        raise ValueError("descriptor cache field geometry xyz SHA256 differs")

    return {
        "xyz": xyz.contiguous(),
        "valid": valid.contiguous(),
        "descriptors": descriptors.contiguous(),
        "global_rows": global_rows.contiguous(),
        "geometry_fingerprint": geometry,
        "row_storage": row_storage,
        "readout_checkpoint_sha256": readout_sha,
        "official_radio_checkpoint_sha256": radio_sha,
        "descriptor_tensor_sha256": _tensor_sha256(descriptors),
        "native_scales": native_scales,
        "field_checkpoint_sha256": field_checkpoint_sha256,
        "readout_checkpoint_sha256": readout_checkpoint_sha256,
    }


def _validate_text_query_cache(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_queries = payload.get("queries")
    if not isinstance(raw_queries, (list, tuple)) or not raw_queries:
        raise ValueError("frozen text query cache queries must be a non-empty list")
    if not all(isinstance(value, str) and value for value in raw_queries):
        raise ValueError("frozen text query cache contains an invalid query ID")
    query_ids = tuple(raw_queries)
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("frozen text query cache query IDs must be unique")
    declared_ids = payload.get("query_ids", list(query_ids))
    if not isinstance(declared_ids, (list, tuple)) or tuple(declared_ids) != query_ids:
        raise ValueError("frozen text query cache query_ids/order differs from queries")
    if payload.get("prompt_templates") != ["{query}"]:
        raise ValueError("frozen text query cache must use the exact {query} template")
    if payload.get("text_encoder") != "siglip2":
        raise ValueError("frozen text query cache must use the official SigLIP2 encoder")
    if payload.get("model_name") != SIGLIP2_MODEL_NAME:
        raise ValueError("frozen text query cache model_name differs")
    if payload.get("text_canonicalization") != SIGLIP2_TEXT_CANONICALIZATION:
        raise ValueError("frozen text query cache canonicalization differs")
    embeddings = payload.get("embeddings")
    expected_shape = (len(query_ids), DESCRIPTOR_DIMENSION)
    if (
        not isinstance(embeddings, torch.Tensor)
        or not embeddings.is_floating_point()
        or tuple(embeddings.shape) != expected_shape
    ):
        raise ValueError(f"frozen text embeddings must be {expected_shape}")
    embeddings = embeddings.detach().cpu().contiguous()
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("frozen text embeddings contain NaN or infinity")
    if bool((embeddings.float().norm(dim=-1) <= 1e-8).any()):
        raise ValueError("frozen text embeddings must be nonzero")
    return {
        "query_ids": query_ids,
        "embeddings": embeddings,
        "embedding_tensor_sha256": _tensor_sha256(embeddings),
    }


def _compile_query_scores(
    descriptors: torch.Tensor,
    global_rows: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    total_rows: int,
    chunk_size: int,
) -> torch.Tensor:
    if isinstance(chunk_size, bool) or int(chunk_size) <= 0:
        raise ValueError("chunk_size must be a positive integer")
    query_scores = torch.zeros(
        total_rows,
        SCALE_COUNT,
        int(text_embeddings.shape[0]),
        dtype=torch.float16,
    )
    for scale_index in range(SCALE_COUNT):
        for start in range(0, int(global_rows.numel()), int(chunk_size)):
            stop = min(int(global_rows.numel()), start + int(chunk_size))
            # This is the repository's existing calibration-free canonical
            # text interface.  It performs independent normalized cosine and
            # has no temperature, threshold, or cross-query normalization.
            scores = cosine_logits(
                descriptors[start:stop, scale_index],
                text_embeddings,
            )
            query_scores[global_rows[start:stop], scale_index] = scores.half()
    if not bool(torch.isfinite(query_scores).all()):
        raise FloatingPointError("materialized query scores contain NaN or infinity")
    return query_scores.contiguous()


def _file_record(path: Path, digest: str) -> dict[str, str]:
    return {"path": str(path), "sha256": digest}


def materialize(
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
) -> dict[str, Any]:
    """Build one immutable Direct3D cache and its shared renderer receipt."""

    descriptor_expected = _require_sha256(
        descriptor_cache_sha256, label="descriptor_cache_sha256"
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

    descriptor = _validate_descriptor_cache(
        descriptor_payload,
        field_checkpoint_path=field_path,
        field_checkpoint_sha256=field_digest,
        readout_checkpoint_path=readout_path,
        readout_checkpoint_sha256=readout_digest,
        readout_native_scales=_readout_native_scales(readout_payload),
    )
    renderer_xyz = _renderer_checkpoint_xyz(renderer_payload)
    if renderer_xyz.shape != descriptor["xyz"].shape or not torch.equal(
        renderer_xyz, descriptor["xyz"]
    ):
        raise ValueError(
            "renderer geometry xyz/count/row-order differs from descriptor cache"
        )
    text = _validate_text_query_cache(text_payload)
    query_scores = _compile_query_scores(
        descriptor["descriptors"],
        descriptor["global_rows"],
        text["embeddings"],
        total_rows=int(descriptor["xyz"].shape[0]),
        chunk_size=int(chunk_size),
    )
    query_scores_sha = _tensor_sha256(query_scores)
    query_order_sha = canonical_json_sha256(list(text["query_ids"]))
    scale_ids = tuple(_scale_id(radius) for radius in descriptor["native_scales"])
    scale_records = [
        {"id": scale_id, "value": radius, "unit": "meter"}
        for scale_id, radius in zip(scale_ids, descriptor["native_scales"])
    ]
    sources = {
        "descriptor_cache": _file_record(descriptor_path, descriptor_digest),
        "text_query_cache": _file_record(text_path, text_digest),
        "field_checkpoint": _file_record(field_path, field_digest),
        "readout_checkpoint": _file_record(readout_path, readout_digest),
        "renderer_geometry_checkpoint": _file_record(
            renderer_path, renderer_digest
        ),
        "materializer_source": _file_record(
            implementation_path, implementation_digest
        ),
    }
    shared_authority = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "contract": SHARED_AUTHORITY_CONTRACT,
        "score_semantics": "raw_independent_normalized_cosine",
        "score_formula": SCORING_FORMULA,
        "score_implementation": SCORING_IMPLEMENTATION,
        "score_dtype": "torch.float16",
        "scale_axis": scale_records,
        "query_axis": {
            "ids": list(text["query_ids"]),
            "order_sha256": query_order_sha,
            "embedding_tensor_sha256": text["embedding_tensor_sha256"],
            "text_encoder": "official_siglip2_g",
            "model_name": SIGLIP2_MODEL_NAME,
            "text_canonicalization": SIGLIP2_TEXT_CANONICALIZATION,
            "prompt_templates": ["{query}"],
        },
        "geometry_axis": {
            **descriptor["geometry_fingerprint"],
            "valid_sha256": _tensor_sha256(descriptor["valid"]),
            "field_checkpoint_sha256": field_digest,
            "readout_checkpoint_sha256": readout_digest,
            "renderer_geometry_checkpoint_sha256": renderer_digest,
            "renderer_xyz_sha256": _direct_xyz_sha256(renderer_xyz),
        },
        "descriptor_axis": {
            "dimension": DESCRIPTOR_DIMENSION,
            "row_storage": descriptor["row_storage"],
            "valid_rows": int(descriptor["valid"].sum()),
            "features_by_scale_sha256": descriptor["descriptor_tensor_sha256"],
            "global_rows_sha256": _tensor_sha256(descriptor["global_rows"]),
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
                "score_semantics": SCORE_SEMANTICS_2D,
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
        "version": 2,
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
        "authority": shared_authority,
    }
    write_torch_noclobber(output_path, payload)
    output_digest = sha256_file(output_path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "complete_calibration_free_query_score_materialization",
        "query_score_cache": _file_record(output_path, output_digest),
        "query_score_shape_n3q": list(query_scores.shape),
        "valid_primitives": int(descriptor["valid"].sum()),
        "total_primitives": int(descriptor["valid"].numel()),
        "queries": int(query_scores.shape[2]),
        "execution": {
            "device": "cpu",
            "chunk_size": int(chunk_size),
            "chunk_size_changes_method": False,
        },
        "shared_renderer_authority": shared_authority,
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
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="CPU execution chunk only; it does not alter any query score",
    )
    args = parser.parse_args(argv)
    report = materialize(
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
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
