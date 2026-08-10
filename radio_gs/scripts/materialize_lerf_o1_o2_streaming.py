#!/usr/bin/env python3
"""Materialize scene-general, source-only LERF O1/O2 raw score caches.

The source-view axis is retained only transiently.  The durable query-free
representation is one FP16 equal-view teacher mean per accepted multiscale
row.  O1 and O2 raw positive/canonical-negative score caches are emitted in
the exact frozen O0 geometry/query envelope.  This entrypoint has no target
metric or annotation reader.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F

from radio_gs.config import load_config
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
    _load_bundle_feature_maps,
    _validated_feature_bundle,
)
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    accumulate_raster_contribution_features,
    tensor_sha256_typed,
)
from radio_gs.scripts.materialize_lerf_teacher_view_oracle_matrix import (
    geodesic_project,
)
from radio_gs.scripts.materialize_lerf_teacher_view_siglip_authority import (
    _canonicalize_view_axis,
    _load_responsibility_view,
    _update_top_views,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


AUTHORITY_SCHEMA = "radio_gs.lerf_o1_o2_streaming_execution.v1"
MEAN_SCHEMA = "radio_gs.lerf_source_teacher_mean_siglip.v1"
RESULT_SCHEMA = "radio_gs.lerf_o1_o2_streaming_result.v1"
SCHEMA_VERSION = 1
RAW_CACHE_CONTRACT = "radio_gs.ours_lerf_direct3d_multiscale_query_scores_fp32.v4"
RAW_AUTHORITY_CONTRACT = "radio_gs.lerf_multiscale_query_score_fp32_authority.v4"
NEGATIVE_QUERIES = ("object", "things", "stuff", "texture")
SCALE_IDS = ("0.25", "0.45", "0.7")
SCALE_RADII_M = (0.25, 0.45, 0.7)
SOURCE_VIEW_COUNT = 120
TOP_VIEW_COUNT = 4
O1_MAXIMUM_ANGLE_RADIANS = 0.15
PREFLIGHT_BATCH_CANDIDATES = (128, 64)
PACING_SECONDS_PER_PROJECTION_BATCH = 0.05
PROGRESS_VIEW_MILESTONES = (30, 60, 90, 120)


def method_contract() -> dict[str, Any]:
    return {
        "schema": AUTHORITY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source_view_count": SOURCE_VIEW_COUNT,
        "teacher_retention": "top4_exact_marginal_mass_desc_then_frame_id_asc",
        "teacher_aggregation": "equal_view_normalized_mean",
        "durable_teacher_dtype": "torch.float16",
        "top4_descriptors_durable": False,
        "O1": {
            "operation": "closed_form_unit_sphere_geodesic_projection",
            "maximum_angle_radians": O1_MAXIMUM_ANGLE_RADIANS,
            "per_scale": True,
        },
        "O2": {
            "operation": "normalized_equal_view_teacher_mean",
            "repeated_scale_slots": 3,
        },
        "scale_ids": list(SCALE_IDS),
        "scale_radii_m": list(SCALE_RADII_M),
        "canonical_negative_queries": list(NEGATIVE_QUERIES),
        "score_dtype": "torch.float32",
        "score_semantics": "raw_independent_normalized_cosine",
        "per_scene_or_per_query_hyperparameters": False,
        "O3_materialized": False,
        "O4_materialized": False,
        "metric_execution_authorized": False,
    }


METHOD_CONTRACT_SHA256 = canonical_json_sha256(method_contract())


def access_audit() -> dict[str, bool]:
    return {
        "source_feature_bundle_opened": True,
        "source_responsibility_opened": True,
        "exact_query_axis_opened": True,
        "exact_o0_pair_opened": True,
        "target_images_opened": False,
        "target_ground_truth_opened": False,
        "target_masks_opened": False,
        "target_metrics_opened": False,
        "target_quality_readout_executed": False,
    }


def _record(path: str, digest: str, *, label: str) -> dict[str, str]:
    result = {"path": str(Path(path).expanduser().resolve()), "sha256": str(digest)}
    validate_file_record(result, label=label)
    return result


def _record_value(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    result = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    validate_file_record(result, label=label)
    return result


def _new(path: str | Path, *, label: str) -> Path:
    raw = str(path)
    result = Path(raw).expanduser().resolve()
    if raw != str(result):
        raise ValueError(f"{label} must be canonical absolute")
    if result.exists() or result.is_symlink():
        raise FileExistsError(f"{label} already exists: {result}")
    return result


def _validate_text_bank(value: Mapping[str, Any], *, expected_queries: list[str]) -> torch.Tensor:
    queries = value.get("queries")
    embeddings = value.get("embeddings")
    if (
        list(queries) if isinstance(queries, list) else None
    ) != expected_queries or not torch.is_tensor(embeddings):
        raise ValueError("O1/O2 text bank axis differs")
    embeddings = embeddings.detach().float().cpu().contiguous()
    if (
        embeddings.shape != (len(expected_queries), 1536)
        or not bool(torch.isfinite(embeddings).all())
        or bool((torch.linalg.vector_norm(embeddings, dim=-1) <= 0).any())
    ):
        raise ValueError("O1/O2 text embeddings differ")
    return F.normalize(embeddings, dim=-1)


def _validate_base_descriptor_general(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], torch.Tensor]:
    """Validate the shared v5 multiscale core across legacy optional fields."""

    raw, _, _ = load_torch_mapping(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="scene-general AcceptedV2 multiscale descriptor",
    )
    core = {
        "xyz", "features", "summary_features", "global_rows",
        "features_by_scale", "valid", "metadata",
    }
    optional = {"primary_valid", "semantic_confidence"}
    if not core <= set(raw) or not set(raw) <= core | optional:
        raise ValueError("scene-general AcceptedV2 multiscale fields differ")
    payload = dict(raw)
    xyz = payload["xyz"].detach().float().cpu()
    valid = payload["valid"].detach().bool().cpu()
    rows = payload["global_rows"].detach().long().cpu()
    descriptors = payload["features_by_scale"]
    metadata = payload.get("metadata")
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or valid.shape != (xyz.shape[0],)
        or rows.shape != (int(valid.sum()),)
        or not torch.equal(rows, torch.where(valid)[0])
        or not torch.is_tensor(descriptors)
        or descriptors.shape != (rows.numel(), 3, 1536)
        or not descriptors.is_floating_point()
        or not bool(torch.isfinite(descriptors).all())
        or not isinstance(metadata, Mapping)
        or metadata.get("schema_version") != 5
        or metadata.get("query_set_invariant") is not True
        or metadata.get("benchmark_images_opened") is not False
        or metadata.get("benchmark_masks_opened") is not False
        or metadata.get("text_queries_opened") is not False
        or tuple(metadata.get("region_radii_m", [])) != SCALE_RADII_M
    ):
        raise ValueError("scene-general AcceptedV2 multiscale descriptor differs")
    if "primary_valid" in payload:
        primary = payload["primary_valid"]
        if not torch.is_tensor(primary) or primary.shape != valid.shape or primary.dtype != torch.bool:
            raise ValueError("scene-general AcceptedV2 primary-valid axis differs")
    if "semantic_confidence" in payload:
        confidence = payload["semantic_confidence"]
        if (
            not torch.is_tensor(confidence)
            or confidence.shape != valid.shape
            or not confidence.is_floating_point()
            or not bool(torch.isfinite(confidence).all())
        ):
            raise ValueError("scene-general AcceptedV2 semantic-confidence axis differs")
    payload["xyz"] = xyz
    payload["valid"] = valid
    payload["global_rows"] = rows
    return payload, rows


def _validate_responsibility_payload(
    payload: Mapping[str, Any],
    *,
    descriptor_xyz_sha256: str,
    feature_frame_ids: set[int],
) -> dict[str, Any]:
    required = {
        "formula_contract", "formula_sha256", "frame_indices", "metadata",
        "num_gaussians", "num_pixels", "schema", "schema_version",
        "total_hits", "views",
    }
    metadata = payload.get("metadata")
    formula = payload.get("formula_contract")
    frames = payload.get("frame_indices")
    views = payload.get("views")
    if (
        set(payload) != required
        or payload.get("schema")
        != "radio_gs.sparse_exact_marginal_responsibility_authority.v1"
        or payload.get("schema_version") != 1
        or not isinstance(metadata, Mapping)
        or metadata.get("assignment_mode")
        != "exact_front_to_back_sparse_marginal"
        or metadata.get("registration_weight_mode")
        != "exact_front_to_back_marginal_responsibility"
        or metadata.get("query_independent") is not True
        or metadata.get("benchmark_images_opened") is not False
        or metadata.get("benchmark_masks_opened") is not False
        or metadata.get("xyz_sha256") != descriptor_xyz_sha256
        or not isinstance(metadata.get("excluded_frame_ids"), list)
        or not metadata.get("excluded_frame_ids")
        or not isinstance(formula, Mapping)
        or formula.get("query_independent") is not True
        or formula.get("feature_independent") is not True
        or payload.get("formula_sha256") != canonical_json_sha256(formula)
        or not isinstance(frames, list)
        or not isinstance(views, list)
        or len(frames) != SOURCE_VIEW_COUNT
        or len(views) != SOURCE_VIEW_COUNT
        or len(set(frames)) != SOURCE_VIEW_COUNT
        or set(frames) & set(metadata.get("excluded_frame_ids"))
        or not set(frames) <= feature_frame_ids
        or frames != [int(record.get("frame_index", -1)) for record in views]
        or int(payload.get("total_hits", -1))
        != sum(int(record.get("num_hits", -1)) for record in views)
    ):
        raise ValueError("scene-general exact responsibility authority differs")
    previous: tuple[int, int] | None = None
    for position, record in enumerate(views):
        if not isinstance(record, Mapping) or set(record) != {
            "frame_index", "num_hits", "relative_path", "sha256", "view_index"
        }:
            raise ValueError("scene-general responsibility view record differs")
        relative = Path(str(record["relative_path"]))
        key = (int(record["frame_index"]), int(record["view_index"]))
        if (
            int(record["view_index"]) != position
            or int(record["num_hits"]) < 0
            or relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or (previous is not None and key <= previous)
        ):
            raise ValueError("scene-general responsibility view order differs")
        previous = key
    return dict(payload)


def _validate_o0_pair(
    positive: Mapping[str, Any],
    negative: Mapping[str, Any],
    *,
    base: Mapping[str, Any],
    positive_queries: list[str],
    renderer_sha256: str,
) -> None:
    pos = positive.get("query_scores")
    neg = negative.get("query_scores")
    xyz = base["xyz"].detach().float().cpu()
    valid = base["valid"].detach().bool().cpu()
    if (
        positive.get("version") != 4
        or negative.get("version") != 4
        or positive.get("contract") != RAW_CACHE_CONTRACT
        or negative.get("contract") != RAW_CACHE_CONTRACT
        or not torch.is_tensor(pos)
        or not torch.is_tensor(neg)
        or pos.shape != (xyz.shape[0], 3, len(positive_queries))
        or neg.shape != (xyz.shape[0], 3, len(NEGATIVE_QUERIES))
        or pos.dtype != torch.float32
        or neg.dtype != torch.float32
        or not bool(torch.isfinite(pos).all())
        or not bool(torch.isfinite(neg).all())
        or list(positive.get("query_ids", [])) != positive_queries
        or tuple(negative.get("query_ids", [])) != NEGATIVE_QUERIES
        or tuple(positive.get("scale_ids", [])) != SCALE_IDS
        or tuple(negative.get("scale_ids", [])) != SCALE_IDS
        or tuple(positive.get("scale_radii_m", [])) != SCALE_RADII_M
        or tuple(negative.get("scale_radii_m", [])) != SCALE_RADII_M
        or positive.get("geometry_fingerprint") != negative.get("geometry_fingerprint")
        or positive.get("renderer_geometry_checkpoint_sha256") != renderer_sha256
        or negative.get("renderer_geometry_checkpoint_sha256") != renderer_sha256
        or positive.get("field_checkpoint_sha256")
        != base["metadata"].get("field_checkpoint_sha256")
        or negative.get("field_checkpoint_sha256")
        != base["metadata"].get("field_checkpoint_sha256")
        or not torch.equal(positive.get("xyz"), xyz)
        or not torch.equal(negative.get("xyz"), xyz)
        or not torch.equal(positive.get("valid"), valid)
        or not torch.equal(negative.get("valid"), valid)
    ):
        raise ValueError("scene-general exact O0/base/query lineage differs")


def prepare_inputs(authority_path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    raw, digest, authority_source = load_json_object(
        authority_path,
        expected_sha256=expected_sha256,
        label="O1/O2 streaming execution authority",
    )
    required = {
        "schema", "schema_version", "status", "scene_id", "implementation",
        "method_contract", "method_contract_sha256", "feature_output_bundle_sha256",
        "inputs", "outputs",
        "execution", "query_free_materialization_authorized",
        "metric_execution_authorized", "access_audit",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("O1/O2 execution authority fields differ")
    authority = dict(raw)
    implementation = _record_value(authority["implementation"], label="O1/O2 implementation")
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != SCHEMA_VERSION
        or authority.get("status") != "authorized_source_only_premetric_o1_o2_streaming"
        or authority.get("method_contract") != method_contract()
        or authority.get("method_contract_sha256") != METHOD_CONTRACT_SHA256
        or implementation != file_record(Path(__file__).resolve())
        or authority.get("query_free_materialization_authorized") is not True
        or authority.get("metric_execution_authorized") is not False
        or authority.get("access_audit") != access_audit()
        or authority.get("execution") != {
            "physical_gpu": 0,
            "cuda_visible_devices": "0",
            "program_device": "cuda:0",
            "projection_batch_candidates": list(PREFLIGHT_BATCH_CANDIDATES),
            "pacing_seconds_per_projection_batch": PACING_SECONDS_PER_PROJECTION_BATCH,
            "thermal_poll_seconds": 300,
            "soft_pause_temperature_c": 0,
            "maximum_temperature_c": 88,
        }
    ):
        raise ValueError("O1/O2 execution authority header differs")
    inputs_raw = authority.get("inputs")
    expected_inputs = {
        "base_descriptor", "responsibility_authority", "feature_manifest",
        "scene_config", "renderer_geometry_checkpoint", "official_radio_checkpoint",
        "positive_text", "negative_text", "o0_positive", "o0_negative",
        "frozen_metric_config",
    }
    if not isinstance(inputs_raw, Mapping) or set(inputs_raw) != expected_inputs:
        raise ValueError("O1/O2 execution inputs differ")
    records = {
        name: _record_value(inputs_raw[name], label=f"O1/O2 {name}")
        for name in sorted(expected_inputs)
    }
    for name, record in records.items():
        validate_file_record(record, label=f"O1/O2 {name}")
    base_path = Path(records["base_descriptor"]["path"])
    base, rows = _validate_base_descriptor_general(
        base_path, records["base_descriptor"]["sha256"]
    )
    feature_path = Path(records["feature_manifest"]["path"])
    if feature_path.name != "frame_manifest.json":
        raise ValueError("O1/O2 feature manifest name differs")
    feature_manifest, validation, tensor_records = _validated_feature_bundle(
        feature_path.parent,
        expected_output_bundle_sha256=str(authority["feature_output_bundle_sha256"]),
    )
    if validation["manifest_sha256"] != records["feature_manifest"]["sha256"]:
        raise ValueError("O1/O2 feature manifest SHA differs")
    feature_frames = {int(record["frame_idx"]) for record in feature_manifest["frames"]}
    responsibility_raw, _, responsibility_path = load_json_object(
        records["responsibility_authority"]["path"],
        expected_sha256=records["responsibility_authority"]["sha256"],
        label="O1/O2 exact responsibility authority",
    )
    responsibility = _validate_responsibility_payload(
        responsibility_raw,
        descriptor_xyz_sha256=str(base["metadata"]["field_geometry_xyz_sha256"]),
        feature_frame_ids=feature_frames,
    )
    metadata = responsibility["metadata"]
    responsibility_root = Path(responsibility_path).parent
    for view in responsibility["views"]:
        sidecar = (responsibility_root / str(view["relative_path"])).resolve()
        if responsibility_root not in sidecar.parents or not sidecar.is_file():
            raise ValueError("O1/O2 responsibility sidecar is missing or unsafe")
    if (
        records["scene_config"]["path"] != str(Path(metadata["config"]).resolve())
        or records["renderer_geometry_checkpoint"]["path"]
        != str(Path(metadata["checkpoint"]).resolve())
        or records["renderer_geometry_checkpoint"]["sha256"]
        != metadata["geometry_checkpoint_sha256"]
        or int(responsibility["num_gaussians"]) != int(base["xyz"].shape[0])
        or int(responsibility["num_pixels"])
        != int(metadata["feature_height"]) * int(metadata["feature_width"])
    ):
        raise ValueError("O1/O2 scene config/geometry alignment differs")
    for name in (
        "scene_config", "renderer_geometry_checkpoint", "official_radio_checkpoint",
        "frozen_metric_config",
    ):
        if sha256_file(records[name]["path"]) != records[name]["sha256"]:
            raise ValueError(f"O1/O2 {name} SHA differs")
    scene_id = str(authority.get("scene_id", ""))
    source_config = load_config(records["scene_config"]["path"])
    frozen_config = load_config(records["frozen_metric_config"]["path"])
    if (
        not scene_id
        or Path(str(getattr(source_config, "feature_dir", ""))).expanduser().resolve()
        != feature_path.parent
        or Path(str(getattr(frozen_config, "scene_root", ""))).name != scene_id
        or base["metadata"].get("official_radio_checkpoint_sha256")
        != records["official_radio_checkpoint"]["sha256"]
    ):
        raise ValueError("O1/O2 scene/frozen-config/RADIO lineage differs")
    positive_text_raw, _, _ = load_torch_mapping(
        records["positive_text"]["path"],
        expected_sha256=records["positive_text"]["sha256"],
        map_location="cpu",
        label="O1/O2 positive text",
    )
    negative_text_raw, _, _ = load_torch_mapping(
        records["negative_text"]["path"],
        expected_sha256=records["negative_text"]["sha256"],
        map_location="cpu",
        label="O1/O2 negative text",
    )
    positive_queries = list(positive_text_raw.get("queries", []))
    positive_embeddings = _validate_text_bank(
        positive_text_raw, expected_queries=positive_queries
    )
    negative_embeddings = _validate_text_bank(
        negative_text_raw, expected_queries=list(NEGATIVE_QUERIES)
    )
    o0_positive_raw, _, _ = load_torch_mapping(
        records["o0_positive"]["path"],
        expected_sha256=records["o0_positive"]["sha256"],
        map_location="cpu",
        label="O1/O2 O0 positive",
    )
    o0_negative_raw, _, _ = load_torch_mapping(
        records["o0_negative"]["path"],
        expected_sha256=records["o0_negative"]["sha256"],
        map_location="cpu",
        label="O1/O2 O0 negative",
    )
    _validate_o0_pair(
        o0_positive_raw,
        o0_negative_raw,
        base=base,
        positive_queries=positive_queries,
        renderer_sha256=records["renderer_geometry_checkpoint"]["sha256"],
    )
    outputs = authority.get("outputs")
    expected_outputs = {
        "teacher_mean", "o1_positive", "o1_negative", "o2_positive",
        "o2_negative", "result",
    }
    if not isinstance(outputs, Mapping) or set(outputs) != expected_outputs:
        raise ValueError("O1/O2 execution outputs differ")
    resolved_outputs: dict[str, str] = {}
    for name, path in outputs.items():
        raw_path = str(path)
        resolved = str(Path(raw_path).expanduser().resolve())
        if raw_path != resolved:
            raise ValueError(f"O1/O2 {name} output must be canonical absolute")
        resolved_outputs[name] = resolved
    return {
        "authority": authority,
        "authority_record": {"path": str(authority_source), "sha256": digest},
        "records": records,
        "base": base,
        "rows": rows,
        "responsibility": responsibility,
        "responsibility_path": responsibility_path,
        "feature_manifest": feature_manifest,
        "feature_validation": validation,
        "tensor_records": tensor_records,
        "positive_queries": positive_queries,
        "positive_embeddings": positive_embeddings,
        "negative_embeddings": negative_embeddings,
        "o0_positive": dict(o0_positive_raw),
        "o0_negative": dict(o0_negative_raw),
        "outputs": resolved_outputs,
    }


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    authority_output = _new(args.authority_output, label="O1/O2 authority")
    output_root = Path(args.output_dir).expanduser().resolve()
    if str(output_root) != str(args.output_dir):
        raise ValueError("O1/O2 output directory must be canonical absolute")
    names = {
        "teacher_mean": f"{args.scene_id}_teacher_mean_fp16.pt",
        "o1_positive": f"{args.scene_id}_o1_positive.pt",
        "o1_negative": f"{args.scene_id}_o1_negative.pt",
        "o2_positive": f"{args.scene_id}_o2_positive.pt",
        "o2_negative": f"{args.scene_id}_o2_negative.pt",
        "result": f"{args.scene_id}_o1_o2_streaming_result.json",
    }
    outputs = {
        name: str(_new(output_root / filename, label=f"O1/O2 {name}"))
        for name, filename in names.items()
    }
    inputs = {
        "base_descriptor": _record(args.base_descriptor, args.base_descriptor_sha256, label="base descriptor"),
        "responsibility_authority": _record(args.responsibility_authority, args.responsibility_authority_sha256, label="responsibility authority"),
        "feature_manifest": _record(args.feature_manifest, args.feature_manifest_sha256, label="feature manifest"),
        "scene_config": _record(args.scene_config, args.scene_config_sha256, label="scene config"),
        "renderer_geometry_checkpoint": _record(args.renderer_geometry_checkpoint, args.renderer_geometry_checkpoint_sha256, label="renderer geometry checkpoint"),
        "official_radio_checkpoint": _record(args.official_radio_checkpoint, args.official_radio_checkpoint_sha256, label="official RADIO checkpoint"),
        "positive_text": _record(args.positive_text, args.positive_text_sha256, label="positive text"),
        "negative_text": _record(args.negative_text, args.negative_text_sha256, label="negative text"),
        "o0_positive": _record(args.o0_positive, args.o0_positive_sha256, label="O0 positive"),
        "o0_negative": _record(args.o0_negative, args.o0_negative_sha256, label="O0 negative"),
        "frozen_metric_config": _record(args.frozen_metric_config, args.frozen_metric_config_sha256, label="frozen metric config"),
    }
    contract = method_contract()
    authority = {
        "schema": AUTHORITY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "authorized_source_only_premetric_o1_o2_streaming",
        "scene_id": str(args.scene_id),
        "implementation": file_record(Path(__file__).resolve()),
        "method_contract": contract,
        "method_contract_sha256": METHOD_CONTRACT_SHA256,
        "feature_output_bundle_sha256": str(args.feature_output_bundle_sha256),
        "inputs": inputs,
        "outputs": outputs,
        "execution": {
            "physical_gpu": 0,
            "cuda_visible_devices": "0",
            "program_device": "cuda:0",
            "projection_batch_candidates": list(PREFLIGHT_BATCH_CANDIDATES),
            "pacing_seconds_per_projection_batch": PACING_SECONDS_PER_PROJECTION_BATCH,
            "thermal_poll_seconds": 300,
            "soft_pause_temperature_c": 0,
            "maximum_temperature_c": 88,
        },
        "query_free_materialization_authorized": True,
        "metric_execution_authorized": False,
        "access_audit": access_audit(),
    }
    # Validate the candidate authority through the same full validator by writing
    # first-writer-only, then reopening it recursively.
    write_frozen_json(authority_output, authority)
    record = file_record(authority_output)
    prepare_inputs(record["path"], expected_sha256=record["sha256"])
    return {"status": "authorized", "authority": record, "outputs": outputs}


def _project_view(
    *,
    prepared: Mapping[str, Any],
    record: Mapping[str, Any],
    global_to_accepted: torch.Tensor,
    head: SigLIP2SummaryHead,
    device: torch.device,
    projection_batch: int,
    pace: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    metadata = prepared["responsibility"]["metadata"]
    feature_map = _load_bundle_feature_maps(
        feature_dir=Path(prepared["records"]["feature_manifest"]["path"]).parent,
        selected_frame_indices=[int(record["frame_index"])],
        subdir="backbone",
        expected_dim=1280,
        feature_size=(int(metadata["feature_height"]), int(metadata["feature_width"])),
        tensor_records=prepared["tensor_records"],
        normalize=False,
        output_dtype=torch.float32,
    )
    gaussian, pixels, marginal = _load_responsibility_view(
        prepared["responsibility"], prepared["responsibility_path"], record
    )
    local = global_to_accepted[gaussian]
    keep = local >= 0
    frame_sum, frame_mass = accumulate_raster_contribution_features(
        feature_map.to(device),
        local[keep].to(device),
        pixels[keep].to(device),
        marginal[keep].to(device),
        n_gaussians=int(prepared["rows"].numel()),
    )
    active = torch.where(frame_mass > 0)[0]
    parameter = next(head.parameters())
    parts: list[torch.Tensor] = []
    for start in range(0, int(active.numel()), int(projection_batch)):
        selected = active[start : start + int(projection_batch)]
        raw = frame_sum[selected].float() / frame_mass[selected, None].clamp_min(1e-8)
        projected = F.normalize(
            head(raw[:, None].to(dtype=parameter.dtype))[:, 0].float(), dim=-1
        )
        parts.append(projected.half().cpu())
        if pace:
            torch.cuda.synchronize(device)
            time.sleep(PACING_SECONDS_PER_PROJECTION_BATCH)
    descriptors = (
        torch.cat(parts, dim=0)
        if parts
        else torch.empty(0, 1536, dtype=torch.float16)
    )
    masses = frame_mass[active].float().cpu()
    active_cpu = active.cpu()
    del feature_map, gaussian, pixels, marginal, local, keep, frame_sum, frame_mass
    torch.cuda.empty_cache()
    return active_cpu, descriptors, masses


def select_projection_batch(
    candidates: tuple[int, ...], runner: Callable[[int], int]
) -> tuple[int, int]:
    """Return first passing batch and peak bytes; only CUDA OOM permits fallback."""

    last_error: BaseException | None = None
    for candidate in candidates:
        try:
            return int(candidate), int(runner(int(candidate)))
        except torch.cuda.OutOfMemoryError as error:
            last_error = error
            torch.cuda.empty_cache()
    raise RuntimeError("all O1/O2 preflight batch candidates exhausted") from last_error


def _typed_stream_hasher(shape: tuple[int, ...], dtype: torch.dtype) -> hashlib._Hash:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(dtype), "shape": list(shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    return digest


def _score_descriptors(
    *,
    base: torch.Tensor,
    teacher_mean: torch.Tensor,
    teacher_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized_base = F.normalize(base.float(), dim=-1)
    normalized_mean = F.normalize(teacher_mean.float(), dim=-1)
    o1 = normalized_base.clone()
    for scale in range(3):
        projected = geodesic_project(
            normalized_base[:, scale], normalized_mean, O1_MAXIMUM_ANGLE_RADIANS
        )
        o1[:, scale] = torch.where(
            teacher_valid[:, None], projected, normalized_base[:, scale]
        )
    return o1, normalized_mean


def _raw_cache(
    template: Mapping[str, Any],
    scores: torch.Tensor,
    *,
    oracle: str,
    representation: Mapping[str, str],
    text_cache: Mapping[str, str],
    descriptor_sha256: str,
) -> dict[str, Any]:
    payload = {key: value for key, value in template.items() if key != "authority"}
    payload["query_scores"] = scores.contiguous().float()
    authority = copy.deepcopy(template["authority"])
    authority["contract"] = RAW_AUTHORITY_CONTRACT
    authority["score_semantics"] = "raw_independent_normalized_cosine"
    authority["score_formula"] = "l2_normalize(descriptor) @ l2_normalize(text_embedding).T"
    authority["score_implementation"] = str(Path(__file__).resolve())
    authority["score_dtype"] = "torch.float32"
    authority["query_scores_sha256"] = tensor_sha256_typed(payload["query_scores"])
    authority["descriptor_axis"]["features_by_scale_sha256"] = descriptor_sha256
    authority["descriptor_axis"]["oracle"] = oracle
    authority["descriptor_axis"]["execution_representation"] = "source_teacher_mean_streaming_v1"
    authority["source_artifacts"]["descriptor_cache"] = dict(representation)
    authority["source_artifacts"]["text_query_cache"] = dict(text_cache)
    authority["source_artifacts"]["materializer_source"] = file_record(Path(__file__).resolve())
    authority["calibration_constraints"]["benchmark_metrics_opened"] = False
    payload["authority"] = authority
    return payload


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    prepared = prepare_inputs(
        args.execution_authority,
        expected_sha256=args.execution_authority_sha256,
    )
    outputs = prepared["outputs"]
    for name, path in outputs.items():
        _new(path, label=f"O1/O2 {name} output")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("O1/O2 streaming requires CUDA_VISIBLE_DEVICES=0")
    config = load_config(prepared["records"]["scene_config"]["path"])
    expected_feature_dir = Path(prepared["records"]["feature_manifest"]["path"]).parent
    if Path(str(getattr(config, "feature_dir", ""))).expanduser().resolve() != expected_feature_dir:
        raise ValueError("O1/O2 scene config feature directory differs")
    rows = prepared["rows"]
    n_rows = int(rows.numel())
    global_to_accepted = torch.full(
        (int(prepared["base"]["xyz"].shape[0]),), -1, dtype=torch.long
    )
    global_to_accepted[rows] = torch.arange(n_rows)
    head = SigLIP2SummaryHead.from_radio_checkpoint(
        prepared["records"]["official_radio_checkpoint"]["path"],
        expected_sha256=prepared["records"]["official_radio_checkpoint"]["sha256"],
    ).to(device).eval().requires_grad_(False)
    densest = max(prepared["responsibility"]["views"], key=lambda row: int(row["num_hits"]))

    def preflight(candidate: int) -> int:
        torch.cuda.reset_peak_memory_stats(device)
        active, descriptors, mass = _project_view(
            prepared=prepared,
            record=densest,
            global_to_accepted=global_to_accepted,
            head=head,
            device=device,
            projection_batch=candidate,
            pace=False,
        )
        peak = int(torch.cuda.max_memory_allocated(device))
        del active, descriptors, mass
        torch.cuda.empty_cache()
        return peak

    projection_batch, preflight_peak = select_projection_batch(
        PREFLIGHT_BATCH_CANDIDATES, preflight
    )
    print(json.dumps({
        "event": "preflight_pass",
        "selected_projection_batch": projection_batch,
        "peak_cuda_bytes": preflight_peak,
        "densest_frame_id": int(densest["frame_index"]),
    }), flush=True)

    top_descriptors = torch.zeros(n_rows, TOP_VIEW_COUNT, 1536, dtype=torch.float16)
    top_mass = torch.zeros(n_rows, TOP_VIEW_COUNT, dtype=torch.float32)
    top_frame_ids = torch.full((n_rows, TOP_VIEW_COUNT), -1, dtype=torch.int32)
    view_started = time.monotonic()
    for position, record in enumerate(prepared["responsibility"]["views"], start=1):
        active, descriptors, mass = _project_view(
            prepared=prepared,
            record=record,
            global_to_accepted=global_to_accepted,
            head=head,
            device=device,
            projection_batch=projection_batch,
            pace=True,
        )
        _update_top_views(
            top_descriptors=top_descriptors,
            top_mass=top_mass,
            top_frame_ids=top_frame_ids,
            rows=active,
            descriptors=descriptors,
            mass=mass,
            frame_id=int(record["frame_index"]),
        )
        if position in PROGRESS_VIEW_MILESTONES:
            elapsed = time.monotonic() - view_started
            eta = elapsed / position * (SOURCE_VIEW_COUNT - position)
            print(json.dumps({
                "event": "source_view_progress",
                "views_complete": position,
                "views_total": SOURCE_VIEW_COUNT,
                "fraction": position / SOURCE_VIEW_COUNT,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
            }), flush=True)
        del active, descriptors, mass
    top_descriptors, top_mass, top_frame_ids = _canonicalize_view_axis(
        top_descriptors, top_mass, top_frame_ids
    )
    teacher_mask = top_frame_ids >= 0
    observed_counts = teacher_mask.sum(dim=1)
    teacher_mean_float = F.normalize(
        (top_descriptors.float() * teacher_mask[:, :, None]).sum(dim=1), dim=-1
    )
    teacher_valid = observed_counts > 0
    teacher_mean_float[~teacher_valid] = 0
    teacher_mean_half = teacher_mean_float.half().contiguous()
    del top_descriptors, top_mass, top_frame_ids, teacher_mask, teacher_mean_float
    mean_payload = {
        "schema": MEAN_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scene_id": prepared["authority"]["scene_id"],
        "global_rows": rows.clone(),
        "teacher_mean": teacher_mean_half,
        "teacher_valid": teacher_valid.contiguous(),
        "retained_view_count": observed_counts.to(torch.uint8).contiguous(),
        "producer": file_record(Path(__file__).resolve()),
        "execution_authority": dict(prepared["authority_record"]),
        "input_authority": {
            "base_descriptor": dict(prepared["records"]["base_descriptor"]),
            "responsibility_authority": dict(prepared["records"]["responsibility_authority"]),
            "feature_manifest": dict(prepared["records"]["feature_manifest"]),
            "official_radio_checkpoint": dict(prepared["records"]["official_radio_checkpoint"]),
        },
        "method_contract_sha256": prepared["authority"]["method_contract_sha256"],
        "teacher_mean_sha256": tensor_sha256_typed(teacher_mean_half),
        "access_audit": access_audit(),
    }
    write_torch_noclobber(outputs["teacher_mean"], mean_payload)
    mean_record = file_record(outputs["teacher_mean"])

    base_features = prepared["base"]["features_by_scale"]
    positive_scores_o1 = prepared["o0_positive"]["query_scores"].clone()
    negative_scores_o1 = prepared["o0_negative"]["query_scores"].clone()
    positive_scores_o2 = positive_scores_o1.clone()
    negative_scores_o2 = negative_scores_o1.clone()
    positive_embeddings = prepared["positive_embeddings"].to(device)
    negative_embeddings = prepared["negative_embeddings"].to(device)
    o1_hasher = _typed_stream_hasher((n_rows, 3, 1536), torch.float32)
    o2_hasher = _typed_stream_hasher((n_rows, 1536), torch.float32)
    score_batch = 256
    for start in range(0, n_rows, score_batch):
        stop = min(n_rows, start + score_batch)
        global_rows = rows[start:stop]
        o1, mean = _score_descriptors(
            base=base_features[start:stop],
            teacher_mean=teacher_mean_half[start:stop],
            teacher_valid=teacher_valid[start:stop],
        )
        o1_hasher.update(o1.contiguous().numpy().tobytes(order="C"))
        o2_hasher.update(mean.contiguous().numpy().tobytes(order="C"))
        active = teacher_valid[start:stop]
        if bool(active.any()):
            selected_rows = global_rows[active]
            o1_active = o1[active].to(device)
            mean_active = mean[active].to(device)
            positive_scores_o1[selected_rows] = torch.einsum(
                "bsd,qd->bsq", o1_active, positive_embeddings
            ).cpu()
            negative_scores_o1[selected_rows] = torch.einsum(
                "bsd,qd->bsq", o1_active, negative_embeddings
            ).cpu()
            o2_positive = torch.einsum(
                "bd,qd->bq", mean_active, positive_embeddings
            )[:, None].expand(-1, 3, -1)
            o2_negative = torch.einsum(
                "bd,qd->bq", mean_active, negative_embeddings
            )[:, None].expand(-1, 3, -1)
            positive_scores_o2[selected_rows] = o2_positive.cpu()
            negative_scores_o2[selected_rows] = o2_negative.cpu()
    o1_descriptor_sha = o1_hasher.hexdigest()
    o2_descriptor_sha = o2_hasher.hexdigest()
    cache_specs = (
        ("o1_positive", prepared["o0_positive"], positive_scores_o1, "O1", prepared["records"]["positive_text"], o1_descriptor_sha),
        ("o1_negative", prepared["o0_negative"], negative_scores_o1, "O1", prepared["records"]["negative_text"], o1_descriptor_sha),
        ("o2_positive", prepared["o0_positive"], positive_scores_o2, "O2", prepared["records"]["positive_text"], o2_descriptor_sha),
        ("o2_negative", prepared["o0_negative"], negative_scores_o2, "O2", prepared["records"]["negative_text"], o2_descriptor_sha),
    )
    output_records: dict[str, dict[str, str]] = {"teacher_mean": mean_record}
    for name, template, scores, oracle, text_record, descriptor_sha in cache_specs:
        payload = _raw_cache(
            template,
            scores,
            oracle=oracle,
            representation=mean_record,
            text_cache=text_record,
            descriptor_sha256=descriptor_sha,
        )
        write_torch_noclobber(outputs[name], payload)
        output_records[name] = file_record(outputs[name])
    result = {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_source_only_premetric_o1_o2_streaming",
        "scene_id": prepared["authority"]["scene_id"],
        "execution_authority": dict(prepared["authority_record"]),
        "outputs": output_records,
        "accepted_rows": n_rows,
        "rows_with_teacher": int(teacher_valid.sum()),
        "rows_without_teacher_bitwise_o0_fallback": int((~teacher_valid).sum()),
        "selected_projection_batch": projection_batch,
        "preflight_peak_cuda_bytes": preflight_peak,
        "elapsed_seconds": time.monotonic() - started,
        "method_contract_sha256": prepared["authority"]["method_contract_sha256"],
        "access_audit": access_audit(),
        "metric_execution_authorized": False,
        "metric_executed": False,
        "next_gate": "independent_exact_o0_control_replay_before_any_metric_authority",
    }
    write_frozen_json(outputs["result"], result)
    return {**result, "result": file_record(outputs["result"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-authority")
    build.add_argument("--scene-id", required=True)
    for name in (
        "base_descriptor", "responsibility_authority", "feature_manifest",
        "scene_config", "renderer_geometry_checkpoint", "official_radio_checkpoint",
        "positive_text", "negative_text", "o0_positive", "o0_negative",
        "frozen_metric_config",
    ):
        option = name.replace("_", "-")
        build.add_argument(f"--{option}", required=True)
        build.add_argument(f"--{option}-sha256", required=True)
    build.add_argument("--feature-output-bundle-sha256", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--authority-output", required=True)
    build.set_defaults(handler=build_authority)
    run = commands.add_parser("materialize")
    run.add_argument("--execution-authority", required=True)
    run.add_argument("--execution-authority-sha256", required=True)
    run.set_defaults(handler=materialize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.handler(args), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "AUTHORITY_SCHEMA",
    "MEAN_SCHEMA",
    "METHOD_CONTRACT_SHA256",
    "O1_MAXIMUM_ANGLE_RADIANS",
    "access_audit",
    "method_contract",
    "select_projection_batch",
]
