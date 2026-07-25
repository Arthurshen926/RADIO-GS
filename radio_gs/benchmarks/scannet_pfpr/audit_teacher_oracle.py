#!/usr/bin/env python3
"""Evaluator-only teacher-versus-field diagnosis for ScanNet-PFPR.

This utility is deliberately *not* a PFPR submission path.  It opens the
private anchor/frame record solely to answer a methodological question that a
formal score cannot answer by itself: is a failure caused by the pose-free
crop descriptor, or by the canonical DINO field/readout?  It never writes a
method prediction vector and marks every result as evaluator-only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from radio_gs.benchmarks.agile3d_scannet40.evaluate_canonical_field import (
    _load_scene_geometry,
    _sha256,
)
from radio_gs.benchmarks.scannet_pfir.protocol import load_matrix
from radio_gs.interfaces import OfficialRadioRuntime, load_canonical_capability_bank
from radio_gs.querying.query_compilers import continuous_gaussian_readout

from .protocol import (
    DEPTH_ALIGNED_QUERY_RASTER_V2,
    ProtocolConfig,
    SUPPORTED_BENCHMARK_VERSIONS,
    aggregate_query_metrics,
    evaluate_ranked_locations,
    fixed_radius_nms,
    protocol_config_from_record,
)
from .score_dino_center import (
    center_spatial_descriptor,
    sample_spatial_descriptor_at_pixels,
)


def _image_tensor(
    path: Path,
    device: torch.device,
    *,
    target_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if target_size is not None and rgb.size != tuple(map(int, target_size)):
            rgb = rgb.resize(tuple(map(int, target_size)), Image.Resampling.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _crop_descriptors(
    runtime: OfficialRadioRuntime,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    descriptors: list[torch.Tensor] = []
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(records), int(batch_size)):
        images = [_image_tensor(Path(str(item["crop_rgb_path"])), device)[0] for item in records[start : start + int(batch_size)]]
        batch = torch.stack(images)
        _summary, spatial = runtime.encode_adaptor_images(
            batch, "dino_v3_7b", feature_fmt="NCHW"
        )
        descriptors.append(
            torch.from_numpy(center_spatial_descriptor(spatial)).to(device)
        )
    return F.normalize(torch.cat(descriptors, dim=0), dim=-1, eps=1e-8)


def _color_pixel_from_private_depth(
    scene_dir: Path,
    record: Mapping[str, Any],
) -> tuple[float, float]:
    frame_id = str(record["source_frame_id"])
    uv = np.asarray(record["source_depth_pixel_uv"], dtype=np.int64).reshape(-1)
    if uv.shape != (2,):
        raise ValueError("private PFPR depth pixel must be [u,v]")
    depth = np.asarray(Image.open(scene_dir / "depth" / f"{frame_id}.png"), dtype=np.float32) / 1000.0
    u, v = (int(value) for value in uv)
    if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
        raise ValueError("private PFPR depth pixel lies outside its source frame")
    value = float(depth[v, u])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("private PFPR source depth is invalid")
    depth_intrinsic = load_matrix(scene_dir / "intrinsics_depth.txt")
    color_intrinsic = load_matrix(scene_dir / "intrinsics_color.txt")
    x = (float(u) - depth_intrinsic[0, 2]) * value / depth_intrinsic[0, 0]
    y = (float(v) - depth_intrinsic[1, 2]) * value / depth_intrinsic[1, 1]
    return (
        float(color_intrinsic[0, 0] * x / value + color_intrinsic[0, 2]),
        float(color_intrinsic[1, 1] * y / value + color_intrinsic[1, 2]),
    )


def _teacher_anchor_descriptors(
    runtime: OfficialRadioRuntime,
    scene_dir: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    config: ProtocolConfig,
) -> torch.Tensor:
    """Read the 2-D teacher at a private anchor; diagnostic only."""

    cached: dict[str, tuple[torch.Tensor, int, int]] = {}
    descriptors: list[torch.Tensor] = []
    for record in records:
        frame_id = str(record["source_frame_id"])
        depth_uv = np.asarray(record["source_depth_pixel_uv"], dtype=np.int64).reshape(-1)
        if depth_uv.shape != (2,):
            raise ValueError("private PFPR depth pixel must be [u,v]")
        target_size = None
        if config.query_raster_contract == DEPTH_ALIGNED_QUERY_RASTER_V2:
            with Image.open(scene_dir / "depth" / f"{frame_id}.png") as depth_image:
                depth = np.asarray(depth_image, dtype=np.uint16)
            target_size = (int(depth.shape[1]), int(depth.shape[0]))
        if frame_id not in cached:
            image_path = scene_dir / "color" / f"{frame_id}.jpg"
            _summary, spatial = runtime.encode_adaptor_images(
                _image_tensor(image_path, device, target_size=target_size),
                "dino_v3_7b",
                feature_fmt="NCHW",
            )
            if target_size is None:
                with Image.open(image_path) as image:
                    width, height = image.size
            else:
                width, height = target_size
            cached[frame_id] = (spatial.detach(), int(width), int(height))
        spatial, width, height = cached[frame_id]
        pixel = (
            depth_uv.astype(np.float32)
            if config.query_raster_contract == DEPTH_ALIGNED_QUERY_RASTER_V2
            else np.asarray(_color_pixel_from_private_depth(scene_dir, record), dtype=np.float32)
        )
        descriptor = sample_spatial_descriptor_at_pixels(
            spatial,
            pixel,
            image_width=width,
            image_height=height,
        )
        descriptors.append(descriptor[0])
    return F.normalize(torch.stack(descriptors), dim=-1, eps=1e-8)


def _candidate_indices(
    gaussian_xyz: torch.Tensor,
    candidate_xyz: np.ndarray,
    *,
    count: int,
) -> torch.Tensor:
    source = torch.as_tensor(gaussian_xyz).float().cpu().numpy()
    _distance, rows = cKDTree(source).query(candidate_xyz, k=min(int(count), len(source)))
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim == 1:
        rows = rows[:, None]
    return torch.from_numpy(np.ascontiguousarray(rows))


def _direct_dino_mpr_field(
    scene_dir: Path,
    *,
    bank_xyz: torch.Tensor,
    bank_valid: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Load the training-only DINO MPR target for a causal diagnostic.

    This is intentionally unavailable to formal PFPR scoring.  It tells us
    whether a gap already exists in multi-view DINO registration, or is
    introduced while compressing that registration into the canonical field.
    """

    path = scene_dir / "dino_v3_mpr.pt"
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("direct DINO MPR diagnostic cache must be a mapping")
    required = {"xyz", "features", "valid", "metadata"}
    if not required.issubset(payload):
        raise ValueError("direct DINO MPR diagnostic cache is incomplete")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("direct DINO MPR diagnostic metadata is malformed")
    if str(metadata.get("feature_space", "")) != "dino_v3":
        raise ValueError("direct DINO MPR diagnostic cache has the wrong space")
    if any(
        metadata.get(key) is True
        for key in ("benchmark_masks_opened", "text_queries_opened", "query_opened")
    ):
        raise ValueError("direct DINO MPR diagnostic cache is benchmark-contaminated")
    xyz = torch.as_tensor(payload["xyz"]).float().cpu()
    valid = torch.as_tensor(payload["valid"]).bool().cpu()
    expected_xyz = torch.as_tensor(bank_xyz).float().cpu()
    expected_valid = torch.as_tensor(bank_valid).bool().cpu()
    if xyz.shape != expected_xyz.shape or not torch.allclose(
        xyz, expected_xyz, atol=1e-6, rtol=0.0
    ):
        raise ValueError("direct DINO MPR diagnostic geometry does not align")
    if valid.shape != expected_valid.shape or not torch.equal(valid, expected_valid):
        raise ValueError("direct DINO MPR diagnostic validity does not align")
    features = torch.as_tensor(payload["features"])
    if features.ndim != 2 or features.shape[0] != len(valid):
        raise ValueError("direct DINO MPR diagnostic features are malformed")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("direct DINO MPR diagnostic features contain NaN or infinity")
    return F.normalize(features[valid].to(device).float(), dim=-1, eps=1e-8)


def _cell_precision(covariance: torch.Tensor, *, voxel_size_m: float) -> torch.Tensor:
    if voxel_size_m <= 0:
        raise ValueError("candidate voxel size must be positive")
    identity = torch.eye(3, dtype=covariance.dtype, device=covariance.device)
    return torch.linalg.pinv(
        covariance + float(voxel_size_m) ** 2 / 12.0 * identity
    )


def continuous_feature_readout(
    gaussian_xyz: torch.Tensor,
    covariance: torch.Tensor,
    features: torch.Tensor,
    points: torch.Tensor,
    *,
    gaussian_precision: torch.Tensor,
    opacity: torch.Tensor,
    candidate_k: int,
) -> torch.Tensor:
    """Diagnostic continuous feature readout for a known private point only."""

    xyz = torch.as_tensor(gaussian_xyz).float()
    field = F.normalize(torch.as_tensor(features, device=xyz.device).float(), dim=-1, eps=1e-8)
    queries = torch.as_tensor(points, device=xyz.device).float()
    if queries.ndim == 1:
        queries = queries[None]
    indices = torch.cdist(queries, xyz).topk(
        min(int(candidate_k), len(xyz)), dim=1, largest=False
    ).indices
    delta = xyz[indices] - queries[:, None]
    precision = torch.as_tensor(gaussian_precision, device=xyz.device).float()[indices]
    mahalanobis = torch.einsum("pki,pkij,pkj->pk", delta, precision, delta)
    weights = torch.exp(-0.5 * mahalanobis) * torch.as_tensor(
        opacity, device=xyz.device
    ).float()[indices]
    vectors = (weights[..., None] * field[indices]).sum(dim=1)
    return F.normalize(vectors, dim=-1, eps=1e-8)


def _score_descriptor_matrix(
    field: torch.Tensor,
    descriptors: torch.Tensor,
    gaussian_xyz: torch.Tensor,
    covariance: torch.Tensor,
    opacity: torch.Tensor,
    candidate_xyz: np.ndarray,
    *,
    precision: torch.Tensor,
    candidate_indices: torch.Tensor,
    support: torch.Tensor,
    chunk_size: int = 65536,
) -> list[np.ndarray]:
    """Use exactly the PFPR scalar-to-continuous candidate readout contract."""

    values = F.normalize(field.float(), dim=-1, eps=1e-8)
    queries = F.normalize(descriptors.float(), dim=-1, eps=1e-8)
    point_tensor = torch.from_numpy(np.asarray(candidate_xyz, dtype=np.float32)).to(values.device)
    valid = support >= 1e-6
    output: list[np.ndarray] = []
    for descriptor in queries:
        primitive_score = torch.empty(len(values), device=values.device)
        for start in range(0, len(values), int(chunk_size)):
            stop = min(start + int(chunk_size), len(values))
            primitive_score[start:stop] = values[start:stop] @ descriptor
        point_score, _ = continuous_gaussian_readout(
            gaussian_xyz,
            covariance,
            primitive_score,
            point_tensor,
            gaussian_precision=precision,
            opacity=opacity,
            candidate_k=int(candidate_indices.shape[1]),
            candidate_indices=candidate_indices,
        )
        output.append(point_score.masked_fill(~valid, -1e30).cpu().numpy().astype(np.float32))
    return output


def _rank_scores(
    candidate_xyz: np.ndarray,
    scores: Sequence[np.ndarray],
    records: Sequence[Mapping[str, Any]],
    *,
    config: ProtocolConfig,
) -> list[dict[str, Any]]:
    if len(scores) != len(records):
        raise ValueError("PFPR diagnostic scores/records disagree")
    rows: list[dict[str, Any]] = []
    maximum = max(config.retrieval_ks)
    for values, record in zip(scores, records):
        selected = fixed_radius_nms(
            candidate_xyz,
            values,
            radius_m=float(config.nms_radius_m),
            maximum=maximum,
        )
        metrics = evaluate_ranked_locations(
            candidate_xyz[selected],
            np.asarray(record["anchor_world_xyz"], dtype=np.float32),
            config=config,
        )
        rows.append(
            {
                "query_id": str(record["query_id"]),
                "scene_id": str(record["scene_id"]),
                **metrics,
            }
        )
    return rows


def _aggregate_by_scene(rows: Sequence[Mapping[str, Any]], config: ProtocolConfig) -> dict[str, Any]:
    micro = aggregate_query_metrics(rows, config=config)
    by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scene[str(row["scene_id"])].append(row)
    per_scene = {
        name: aggregate_query_metrics(values, config=config)
        for name, values in sorted(by_scene.items())
    }
    macro = {
        name: float(np.mean([values[name] for values in per_scene.values()]))
        for name in micro
    }
    return {"metrics_query_micro": micro, "metrics_scene_macro": macro, "per_scene": per_scene}


def _load_records(
    benchmark_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, np.ndarray], ProtocolConfig, str]:
    method = json.loads((benchmark_dir / "manifest.method.json").read_text(encoding="utf-8"))
    public = json.loads((benchmark_dir / "manifest.public.json").read_text(encoding="utf-8"))
    evaluator = json.loads((benchmark_dir / "manifest.evaluator.json").read_text(encoding="utf-8"))
    benchmark_version = str(method.get("benchmark_version", ""))
    if (
        benchmark_version not in SUPPORTED_BENCHMARK_VERSIONS
        or any(payload.get("benchmark_version") != benchmark_version for payload in (public, evaluator))
    ):
        raise ValueError("not a frozen supported ScanNet-PFPR benchmark")
    config = protocol_config_from_record(
        benchmark_version, evaluator.get("protocol_config", {})
    )
    method_by_id = {str(item["query_id"]): dict(item) for item in method.get("queries", [])}
    candidates: dict[str, np.ndarray] = {}
    for item in public.get("scene_domains", []):
        scene = str(item["scene_id"])
        path = Path(str(item["candidate_xyz_path"]))
        values = np.load(path, allow_pickle=False)
        if scene in candidates or values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("PFPR public candidate domain is malformed")
        candidates[scene] = np.asarray(values, dtype=np.float32)
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for private in evaluator.get("queries", []):
        query_id = str(private["query_id"])
        method_record = method_by_id.get(query_id)
        if method_record is None:
            raise ValueError("private PFPR query is absent from the method manifest")
        if set(method_record.get("available_method_inputs", ())) != {"scene_id", "crop_rgb"}:
            raise ValueError("method manifest no longer has the frozen PFPR inputs")
        scene = str(private["scene_id"])
        if scene != str(method_record["scene_id"]) or scene not in candidates:
            raise ValueError("PFPR query/domain alignment is invalid")
        records[scene].append({**method_record, **dict(private)})
    return dict(records), candidates, config, benchmark_version


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run a private diagnostic without creating a valid submission artifact."""

    benchmark_dir = Path(args.benchmark_dir)
    field_root = Path(args.field_root)
    frames_root = Path(args.frames_root)
    geometry_root = Path(args.geometry_cache_root)
    output = Path(args.output)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records_by_scene, candidates_by_scene, config, benchmark_version = _load_records(
        benchmark_dir
    )
    requested = set(str(args.scene_names).replace(",", " ").split())
    if requested:
        unknown = requested - set(records_by_scene)
        if unknown:
            raise ValueError(f"unknown PFPR diagnostic scenes: {sorted(unknown)}")
        records_by_scene = {name: records_by_scene[name] for name in sorted(requested)}
    runtime = OfficialRadioRuntime.load(
        radio_repo=args.radio_repo,
        version=args.radio_version,
        adaptor_names=("dino_v3_7b",),
        device=device,
    )
    all_rows: dict[str, list[dict[str, Any]]] = {
        "crop_center_dino_to_field": [],
        "source_anchor_teacher_dino_to_field": [],
        "canonical_anchor_dino_self_retrieval": [],
        "crop_center_dino_to_direct_mpr_diagnostic": [],
        "source_anchor_teacher_dino_to_direct_mpr_diagnostic": [],
    }
    descriptor_cosines: list[dict[str, Any]] = []
    scene_reports: list[dict[str, Any]] = []
    try:
        for scene_id in sorted(records_by_scene):
            records = records_by_scene[scene_id]
            scene_dir = field_root / "canonical_fields" / scene_id
            field_path = scene_dir / "canonical_mpr_v2.pt"
            capability_path = scene_dir / "official_dino_sam3_views.pt"
            if not field_path.is_file() or not capability_path.is_file():
                raise FileNotFoundError(f"PFPR canonical field is incomplete: {scene_dir}")
            field_hash = _sha256(field_path)
            bank = load_canonical_capability_bank(
                capability_path, expected_field_checkpoint_sha256=field_hash
            )
            xyz, covariance, native_precision, opacity, cache_reused = _load_scene_geometry(
                scene_dir,
                bank_xyz=bank.xyz,
                valid_rows=bank.global_rows,
                expected_field_sha256=field_hash,
                cache_path=geometry_root / f"{scene_id}.pt",
                device=device,
            )
            try:
                field = F.normalize(
                    bank.appearance[bank.global_rows].to(device).float(), dim=-1, eps=1e-8
                )
                direct_mpr = _direct_dino_mpr_field(
                    scene_dir,
                    bank_xyz=bank.xyz,
                    bank_valid=bank.valid,
                    device=device,
                )
                if direct_mpr.shape != field.shape:
                    raise ValueError("direct DINO MPR diagnostic feature shape differs from field")
                candidate_xyz = candidates_by_scene[scene_id]
                candidate_indices = _candidate_indices(
                    xyz, candidate_xyz, count=int(args.candidate_k)
                ).to(device)
                cell_precision = _cell_precision(
                    covariance, voxel_size_m=float(config.candidate_voxel_size_m)
                )
                point_tensor = torch.from_numpy(candidate_xyz).to(device)
                _unused, support = continuous_gaussian_readout(
                    xyz,
                    covariance,
                    torch.ones(len(xyz), device=device),
                    point_tensor,
                    gaussian_precision=cell_precision,
                    opacity=opacity,
                    candidate_k=int(args.candidate_k),
                    candidate_indices=candidate_indices,
                )
                crop = _crop_descriptors(
                    runtime, records, device=device, batch_size=int(args.crop_batch_size)
                )
                teacher = _teacher_anchor_descriptors(
                    runtime,
                    frames_root / scene_id,
                    records,
                    device=device,
                    config=config,
                )
                private_anchors = torch.as_tensor(
                    np.asarray([item["anchor_world_xyz"] for item in records], dtype=np.float32),
                    device=device,
                )
                self_oracle = continuous_feature_readout(
                    xyz,
                    covariance,
                    field,
                    private_anchors,
                    gaussian_precision=native_precision,
                    opacity=opacity,
                    candidate_k=int(args.candidate_k),
                )
                variants = {
                    "crop_center_dino_to_field": crop,
                    "source_anchor_teacher_dino_to_field": teacher,
                    "canonical_anchor_dino_self_retrieval": self_oracle,
                }
                for name, descriptors in variants.items():
                    scores = _score_descriptor_matrix(
                        field,
                        descriptors,
                        xyz,
                        covariance,
                        opacity,
                        candidate_xyz,
                        precision=cell_precision,
                        candidate_indices=candidate_indices,
                        support=support,
                    )
                    all_rows[name].extend(_rank_scores(candidate_xyz, scores, records, config=config))
                for name, descriptors in {
                    "crop_center_dino_to_direct_mpr_diagnostic": crop,
                    "source_anchor_teacher_dino_to_direct_mpr_diagnostic": teacher,
                }.items():
                    scores = _score_descriptor_matrix(
                        direct_mpr,
                        descriptors,
                        xyz,
                        covariance,
                        opacity,
                        candidate_xyz,
                        precision=cell_precision,
                        candidate_indices=candidate_indices,
                        support=support,
                    )
                    all_rows[name].extend(_rank_scores(candidate_xyz, scores, records, config=config))
                similarity = (crop * teacher).sum(dim=-1).detach().cpu().numpy()
                descriptor_cosines.extend(
                    {
                        "query_id": str(record["query_id"]),
                        "scene_id": scene_id,
                        "crop_to_source_anchor_teacher_dino_cosine": float(value),
                    }
                    for record, value in zip(records, similarity)
                )
                scene_reports.append(
                    {
                        "scene_id": scene_id,
                        "queries": len(records),
                        "candidate_points": len(candidate_xyz),
                        "continuous_support_fraction": float((support >= 1e-6).float().mean()),
                        "geometry_cache_reused": bool(cache_reused),
                        "direct_dino_mpr_opened_for_diagnostic_only": True,
                    }
                )
            finally:
                del bank
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        del runtime
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "diagnostic": "ScanNet-PFPR teacher-versus-canonical-field oracle",
        "formal_prediction_eligible": False,
        "reason": (
            "opens evaluator-private held-out source frame, pixel, and anchor only to "
            "separate crop-descriptor, multi-view registration, and canonical-field "
            "error; direct DINO MPR is training-only and it writes no submission scores"
        ),
        "protocol": {
            "benchmark_version": benchmark_version,
            "query_raster_contract": config.query_raster_contract,
            "private_anchor_opened": True,
            "private_source_frame_and_depth_pixel_opened": True,
            "instance_identity_used": False,
            "query_pose_used": False,
            "candidate_domain": "frozen_public_5cm_mesh_geometry",
            "candidate_readout": "continuous_opacity_weighted_gaussian_convolved_with_5cm_voxel_cell",
            "field_hash_checked": True,
            "direct_dino_mpr_training_cache_opened_for_diagnostic_only": True,
            "direct_dino_mpr_is_available_to_method": False,
        },
        "scene_reports": scene_reports,
        "descriptor_alignment": {
            "mean_crop_to_source_anchor_teacher_dino_cosine": float(
                np.mean([row["crop_to_source_anchor_teacher_dino_cosine"] for row in descriptor_cosines])
            ),
            "median_crop_to_source_anchor_teacher_dino_cosine": float(
                np.median([row["crop_to_source_anchor_teacher_dino_cosine"] for row in descriptor_cosines])
            ),
            "rows": descriptor_cosines,
        },
        "variants": {
            name: {**_aggregate_by_scene(rows, config), "rows": rows}
            for name, rows in all_rows.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--field-root", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--geometry-cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene-names", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--radio-repo", default="/root/RADIO")
    parser.add_argument("--radio-version", default="c-radio_v4-h")
    parser.add_argument("--crop-batch-size", type=int, default=4)
    parser.add_argument("--candidate-k", type=int, default=64)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
