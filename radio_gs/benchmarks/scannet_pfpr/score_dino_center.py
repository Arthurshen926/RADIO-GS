#!/usr/bin/env python3
"""Score PFPR patches with the official DINO spatial adaptor and canonical field.

This is a geometry-only point-retrieval baseline: it opens the method manifest
and public candidate geometry, never the evaluator's private 3-D anchors.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from radio_gs.benchmarks.agile3d_scannet40.evaluate_canonical_field import (
    FULL_OBSERVATION_MIN_MEANINGFUL_SUPPORT,
    _field_source_metadata,
    _load_scene_geometry,
    _sha256,
    observation_source_from_render_contract,
    validate_capability_teacher_fidelity,
    validate_continuous_support_threshold,
    validate_full_observation_mpr_contract,
)
from radio_gs.interfaces import (
    GlobalCropContextAdapter,
    GlobalCropSpatialAdapter,
    OfficialRadioRuntime,
    crop_spatial_adapter_sha256,
    load_canonical_capability_bank,
)
from radio_gs.querying.query_compilers import continuous_gaussian_readout
from radio_gs.querying.score_calibration import fit_scene_space_calibration

from .protocol import (
    PFPR_V2_BENCHMARK_VERSION,
    SUPPORTED_BENCHMARK_VERSIONS,
    fixed_radius_nms,
    protocol_config_from_record,
    validate_field_query_exclusion_commitment,
)


def center_spatial_descriptor(spatial: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return an L2-normalized center 3x3 official DINO descriptor per patch."""

    values = torch.as_tensor(spatial).float()
    if values.ndim != 4 or values.shape[1] <= 0 or min(values.shape[-2:]) <= 0:
        raise ValueError("official DINO spatial map must be [B,D,H,W]")
    height, width = values.shape[-2:]
    center_y, center_x = height // 2, width // 2
    y0, y1 = max(0, center_y - 1), min(height, center_y + 2)
    x0, x1 = max(0, center_x - 1), min(width, center_x + 2)
    descriptor = values[:, :, y0:y1, x0:x1].mean(dim=(-2, -1))
    return F.normalize(descriptor, dim=-1, eps=1e-8).cpu().numpy().astype(np.float32)


def center_token_descriptor(spatial: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return the normalized central official-DINO token for a patch.

    PFPR anchors are defined by the patch center.  This is therefore a
    resolution-preserving, query-only ablation of the default center-3x3
    average, not a benchmark-specific crop or target adjustment.
    """

    values = torch.as_tensor(spatial).float()
    if values.ndim != 4 or values.shape[1] <= 0 or min(values.shape[-2:]) <= 0:
        raise ValueError("official DINO spatial map must be [B,D,H,W]")
    center_y, center_x = values.shape[-2] // 2, values.shape[-1] // 2
    descriptor = values[:, :, center_y, center_x]
    return F.normalize(descriptor, dim=-1, eps=1e-8).cpu().numpy().astype(np.float32)


def sample_spatial_descriptor_at_pixels(
    spatial: torch.Tensor,
    pixel_xy: torch.Tensor | np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Bilinearly read NCHW DINO descriptors at original-image pixel centers.

    The official runtime may resize an RGB image to a supported shape before
    returning its spatial grid.  This conversion is image geometry only and
    is shared by generic training/oracle utilities; it never uses a 3-D target
    or a prediction.
    """

    values = torch.as_tensor(spatial).float()
    pixels = torch.as_tensor(pixel_xy, device=values.device).float()
    if values.ndim != 4 or values.shape[0] != 1 or values.shape[1] <= 0:
        raise ValueError("spatial descriptor map must be [1,D,H,W]")
    if pixels.ndim == 1:
        pixels = pixels[None]
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixel_xy must be [Q,2]")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if bool((pixels[:, 0] < 0).any()) or bool((pixels[:, 0] >= image_width).any()):
        raise ValueError("pixel x lies outside the original image")
    if bool((pixels[:, 1] < 0).any()) or bool((pixels[:, 1] >= image_height).any()):
        raise ValueError("pixel y lies outside the original image")
    grid = torch.stack(
        [
            2.0 * (pixels[:, 0] + 0.5) / float(image_width) - 1.0,
            2.0 * (pixels[:, 1] + 0.5) / float(image_height) - 1.0,
        ],
        dim=-1,
    ).reshape(1, 1, -1, 2)
    sampled = F.grid_sample(
        values,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0, :, 0].transpose(0, 1)
    return F.normalize(sampled, dim=-1, eps=1e-8)


def _load_method_queries(
    benchmark_dir: Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, np.ndarray],
    str,
    dict[str, str],
]:
    method = json.loads((benchmark_dir / "manifest.method.json").read_text(encoding="utf-8"))
    public = json.loads((benchmark_dir / "manifest.public.json").read_text(encoding="utf-8"))
    version = str(method.get("benchmark_version", ""))
    if version not in SUPPORTED_BENCHMARK_VERSIONS or public.get("benchmark_version") != version:
        raise ValueError("not a supported ScanNet-PFPR release")
    protocol_config_from_record(version, public.get("protocol_config", {}))
    candidates: dict[str, np.ndarray] = {}
    exclusion_commitments: dict[str, str] = {}
    for item in public.get("scene_domains", []):
        scene = str(item.get("scene_id", ""))
        path = Path(str(item.get("candidate_xyz_path", "")))
        if not scene or scene in candidates or not path.is_file():
            raise ValueError("public PFPR candidate-domain record is invalid")
        xyz = np.load(path, allow_pickle=False)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
            raise ValueError("public PFPR candidate domain is invalid")
        candidates[scene] = np.asarray(xyz, dtype=np.float32)
        digest = str(item.get("excluded_query_source_frame_ids_sha256", ""))
        if version == PFPR_V2_BENCHMARK_VERSION and not digest:
            raise ValueError("PFPR v2 public scene domain lacks held-out-frame commitment")
        exclusion_commitments[scene] = digest
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in method.get("queries", []):
        if set(query.get("available_method_inputs", ())) != {"scene_id", "crop_rgb"}:
            raise ValueError("PFPR method manifest exposes unsupported query inputs")
        scene = str(query.get("scene_id", ""))
        image = Path(str(query.get("crop_rgb_path", "")))
        if scene not in candidates or not image.is_file():
            raise ValueError("PFPR method query/candidate geometry is invalid")
        by_scene[scene].append(dict(query))
    if not by_scene:
        raise ValueError("PFPR method manifest has no queries")
    if version == PFPR_V2_BENCHMARK_VERSION and set(exclusion_commitments) != set(by_scene):
        raise ValueError("PFPR v2 public exclusion commitments do not cover method scenes")
    return dict(by_scene), candidates, version, exclusion_commitments


def _image_batch(records: Sequence[dict[str, Any]], device: torch.device) -> torch.Tensor:
    arrays = [
        np.asarray(Image.open(Path(record["crop_rgb_path"])).convert("RGB"), dtype=np.float32)
        / 255.0
        for record in records
    ]
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("PFPR v1 requires fixed-size RGB patch tensors")
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(device)


def _encode_scene_queries(
    records: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    radio_repo: str,
    radio_version: str,
    batch_size: int,
    query_pooling: str = "center3x3",
    crop_spatial_adapter: GlobalCropSpatialAdapter | None = None,
    crop_context_adapter: GlobalCropContextAdapter | None = None,
    preserve_raw_context_pair: bool = False,
) -> np.ndarray:
    query_pooling = str(query_pooling)
    if query_pooling not in {"center3x3", "center", "center_late_fusion"}:
        raise ValueError(
            "PFPR query pooling must be center3x3, center, or center_late_fusion"
        )
    if crop_spatial_adapter is not None and crop_context_adapter is not None:
        raise ValueError("PFPR accepts at most one frozen crop query adapter")
    if crop_context_adapter is not None and query_pooling != "center3x3":
        raise ValueError("the crop-context adapter requires center3x3 query pooling")
    if preserve_raw_context_pair and crop_context_adapter is None:
        raise ValueError("raw/context pairing requires a frozen crop-context adapter")
    runtime = OfficialRadioRuntime.load(
        radio_repo=radio_repo,
        version=radio_version,
        adaptor_names=("dino_v3_7b",),
        device=device,
    )
    descriptors: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(records), int(batch_size)):
                batch = _image_batch(records[start : start + int(batch_size)], device)
                _summary, spatial = runtime.encode_adaptor_images(
                    batch, "dino_v3_7b", feature_fmt="NCHW"
                )
                center3x3 = torch.from_numpy(center_spatial_descriptor(spatial)).to(device)
                center = torch.from_numpy(center_token_descriptor(spatial)).to(device)
                descriptor = (
                    torch.stack((center3x3, center), dim=1)
                    if query_pooling == "center_late_fusion"
                    else (center3x3 if query_pooling == "center3x3" else center)
                )
                if crop_context_adapter is not None:
                    raw_descriptor = descriptor
                    context = F.normalize(
                        torch.as_tensor(spatial, device=device).float().mean(dim=(-2, -1)),
                        dim=-1,
                        eps=1e-8,
                    )
                    descriptor = crop_context_adapter(descriptor, context)
                    if preserve_raw_context_pair:
                        descriptor = torch.stack((raw_descriptor, descriptor), dim=1)
                descriptors.append(descriptor.cpu().numpy().astype(np.float32))
    finally:
        del runtime
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = np.concatenate(descriptors, axis=0)
    if crop_spatial_adapter is not None:
        with torch.inference_mode():
            result = (
                crop_spatial_adapter(
                    torch.from_numpy(result).to(device)
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    return result


def _fuse_query_prototype_scores(
    scores: torch.Tensor, *, temperature: float = 0.1
) -> torch.Tensor:
    """Fuse query-visible center prototypes without choosing one from metrics."""

    values = torch.as_tensor(scores).float()
    if values.ndim != 2 or values.shape[0] <= 0:
        raise ValueError("prototype scores must be a non-empty [M,P] matrix")
    if temperature <= 0:
        raise ValueError("late-fusion temperature must be positive")
    return float(temperature) * (
        torch.logsumexp(values / float(temperature), dim=0)
        - np.log(float(values.shape[0]))
    )


def _vector_candidate_similarity(
    gaussian_xyz: torch.Tensor,
    covariance: torch.Tensor,
    field: torch.Tensor,
    query: torch.Tensor,
    points: torch.Tensor,
    *,
    precision: torch.Tensor,
    opacity: torch.Tensor,
    candidate_indices: torch.Tensor,
    coherence_sqrt: bool,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Interpolate a feature vector, normalize it, then compare to query."""

    output: list[torch.Tensor] = []
    for start in range(0, len(points), int(chunk_size)):
        stop = min(start + int(chunk_size), len(points))
        indices = candidate_indices[start:stop]
        delta = gaussian_xyz[indices] - points[start:stop, None]
        mahalanobis = torch.einsum(
            "pki,pkij,pkj->pk", delta, precision[indices], delta
        )
        weights = torch.exp(-0.5 * mahalanobis).clamp_min(0.0) * opacity[indices]
        support = weights.sum(dim=1).clamp_min(1e-12)
        vector = torch.einsum("pk,pkd->pd", weights, field[indices]) / support[:, None]
        coherence = vector.norm(dim=-1).clamp(0.0, 1.0)
        score = F.normalize(vector, dim=-1, eps=1e-8) @ query
        if coherence_sqrt:
            score = score * coherence.sqrt()
        output.append(score)
    return torch.cat(output)


def _interleaved_coarse_to_fine_scores(
    candidate_xyz: np.ndarray,
    fine_scores: torch.Tensor | np.ndarray,
    coarse_scores: torch.Tensor | np.ndarray,
    valid: torch.Tensor | np.ndarray,
    *,
    region_radius_m: float,
    maximum_regions: int,
    rank_one_scores: torch.Tensor | np.ndarray | None = None,
    rank_one_valid: torch.Tensor | np.ndarray | None = None,
    refine_rank_one_with_fine: bool = False,
) -> np.ndarray:
    """Interleave precise global proposals with context-proposed regions."""

    xyz = np.asarray(candidate_xyz, dtype=np.float32)
    fine = np.asarray(torch.as_tensor(fine_scores).detach().cpu(), dtype=np.float32)
    coarse = np.asarray(
        torch.as_tensor(coarse_scores).detach().cpu(), dtype=np.float32
    )
    valid_array = np.asarray(
        torch.as_tensor(valid).detach().cpu(), dtype=np.bool_
    )
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("candidate xyz must be [N,3]")
    if fine.shape != (len(xyz),) or coarse.shape != fine.shape:
        raise ValueError("coarse/fine scores must align with candidate xyz")
    if valid_array.shape != fine.shape:
        raise ValueError("valid mask must align with candidate xyz")
    if not np.isfinite(region_radius_m) or float(region_radius_m) <= 0.0:
        raise ValueError("coarse-to-fine region radius must be positive")
    if int(maximum_regions) <= 0:
        raise ValueError("maximum coarse-to-fine regions must be positive")
    rank_one = (
        fine
        if rank_one_scores is None
        else np.asarray(
            torch.as_tensor(rank_one_scores).detach().cpu(), dtype=np.float32
        )
    )
    rank_one_valid_array = (
        valid_array
        if rank_one_valid is None
        else np.asarray(
            torch.as_tensor(rank_one_valid).detach().cpu(), dtype=np.bool_
        )
    )
    if rank_one.shape != fine.shape or rank_one_valid_array.shape != fine.shape:
        raise ValueError("rank-one scores and validity must align with candidates")

    masked_fine = np.where(valid_array, fine, -1e30).astype(np.float32)
    masked_coarse = np.where(valid_array, coarse, -1e30).astype(np.float32)
    masked_rank_one = np.where(
        rank_one_valid_array, rank_one, -1e30
    ).astype(np.float32)
    fine_centers = fixed_radius_nms(
        xyz,
        masked_fine,
        radius_m=float(region_radius_m),
        maximum=int(maximum_regions),
    )
    coarse_centers = fixed_radius_nms(
        xyz,
        masked_coarse,
        radius_m=float(region_radius_m),
        maximum=int(maximum_regions),
    )
    if not len(fine_centers):
        raise ValueError("coarse-to-fine readout has no valid fine proposal")
    rank_one_centers = fixed_radius_nms(
        xyz,
        masked_rank_one,
        radius_m=float(region_radius_m),
        maximum=1,
    )
    if not len(rank_one_centers):
        raise ValueError("coarse-to-fine readout has no authoritative rank-one proposal")

    tree = cKDTree(xyz)
    rank_one_index = int(rank_one_centers[0])
    if bool(refine_rank_one_with_fine):
        local = np.asarray(
            tree.query_ball_point(
                xyz[rank_one_index], r=float(region_radius_m)
            ),
            dtype=np.int64,
        )
        local = local[valid_array[local]]
        if len(local):
            rank_one_index = int(local[np.argmax(masked_fine[local])])
    refined_coarse: list[int] = []
    for center in coarse_centers:
        local = np.asarray(
            tree.query_ball_point(xyz[int(center)], r=float(region_radius_m)),
            dtype=np.int64,
        )
        if not len(local):
            continue
        local = local[valid_array[local]]
        if len(local):
            refined_coarse.append(int(local[np.argmax(masked_fine[local])]))

    # The precise stream owns rank one. Remaining global and context-localized
    # proposals are interleaved with no learned or benchmark-tuned score weight.
    order: list[int] = [rank_one_index]
    fine_offset = 1
    if rank_one_scores is not None:
        # A support-completed fine stream can disagree with the authoritative
        # primary stream. Keep its strongest proposal instead of assuming it
        # duplicates rank one.
        order.append(int(fine_centers[0]))
        fine_offset = 1
    for rank in range(int(maximum_regions)):
        if rank < len(refined_coarse):
            order.append(int(refined_coarse[rank]))
        if rank + fine_offset < len(fine_centers):
            order.append(int(fine_centers[rank + fine_offset]))

    ranked = np.full(len(xyz), -1e30, dtype=np.float32)
    next_score = float(len(order) + 1)
    for index in order:
        if ranked[index] <= -1e20:
            ranked[index] = next_score
            next_score -= 1.0
    return ranked


def _candidate_indices(gaussian_xyz: torch.Tensor, candidate_xyz: np.ndarray, count: int) -> torch.Tensor:
    source = torch.as_tensor(gaussian_xyz).float().cpu().numpy()
    _distance, rows = cKDTree(source).query(candidate_xyz, k=min(int(count), len(source)))
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim == 1:
        rows = rows[:, None]
    return torch.from_numpy(np.ascontiguousarray(rows))


def _scene_artifact(scene_dir: Path, filename: str, *, role: str) -> Path:
    """Resolve an alternative field ablation artifact inside one scene only."""

    name = str(filename).strip()
    if not name or Path(name).name != name:
        raise ValueError(f"{role} must be a basename inside the scene field directory")
    return scene_dir / name


def _cell_convolved_precision(covariance: torch.Tensor, voxel_size_m: float) -> torch.Tensor:
    if voxel_size_m <= 0:
        raise ValueError("PFPR candidate voxel size must be positive")
    variance = float(voxel_size_m) ** 2 / 12.0
    identity = torch.eye(3, dtype=covariance.dtype, device=covariance.device)
    return torch.linalg.pinv(covariance + variance * identity)


def calibrate_query_and_field(
    field: torch.Tensor,
    query: torch.Tensor,
    *,
    method: str = "none",
    sample_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one query-independent scene-space transform to DINO similarity.

    The calibration is fit from the already frozen canonical primitive field
    only.  A query is transformed with exactly the same field statistics, so
    no evaluator anchor, target, pose, depth, label, mask, or metric can enter
    this compiler.  ``none`` preserves the historical normalized-cosine path
    bit-for-bit.
    """

    field_values = F.normalize(torch.as_tensor(field).float(), dim=-1, eps=1e-8)
    query_values = F.normalize(torch.as_tensor(query).float(), dim=-1, eps=1e-8)
    if field_values.ndim != 2 or query_values.ndim != 2:
        raise ValueError("PFPR DINO field/query values must be matrices")
    if field_values.shape[1] != query_values.shape[1]:
        raise ValueError("PFPR DINO field and query dimensions must match")
    if method == "none":
        return field_values, query_values
    if method != "diagonal_robust":
        raise ValueError("PFPR feature calibration must be none or diagonal_robust")
    calibration = fit_scene_space_calibration(
        field_values,
        method=method,
        sample_size=int(sample_size),
        background_centroids=0,
    )
    return calibration.transform(field_values), calibration.transform(query_values)


def validate_pfpr_observation_contract(
    expected_contract: str,
    source_contract: Mapping[str, object],
) -> None:
    """Fail closed on a named PFPR field with mismatched source provenance.

    Field construction already validates this contract, but the scorer repeats
    the inexpensive check so an edited render YAML cannot relabel a sparse or
    query-leaking field after reconstruction.  This opens only render-contract
    provenance, never the private PFPR anchor or source-frame records.
    """

    expected = str(expected_contract).strip()
    if not expected:
        return
    declared = str(source_contract.get("declared_source_contract", ""))
    if declared != expected:
        raise ValueError(
            f"PFPR expected observation contract {expected!r}, found {declared!r}"
        )
    expected_versions = {
        "dense_pfpr_queryheldout_v1": "scannet-pfpr-query-heldout-field-v1",
        "scannet_full_observation_pfpr_queryheldout_v1": (
            "scannet_full_observation_pfpr_queryheldout_v1"
        ),
    }
    required_version = expected_versions.get(expected)
    if required_version is None:
        return
    if not str(source_contract.get("field_source_contract_sha256", "")):
        raise ValueError(f"{expected} requires an auditable source-contract digest")
    found_version = str(source_contract.get("field_source_contract_version", ""))
    if found_version != required_version:
        raise ValueError(
            f"{expected} requires matching source-contract version "
            f"{required_version!r}, found {found_version!r}"
        )


def _score_scene(
    scene_id: str,
    records: Sequence[dict[str, Any]],
    candidate_xyz: np.ndarray,
    *,
    benchmark_version: str,
    expected_query_source_exclusion_digest: str,
    field_root: Path,
    geometry_cache_root: Path,
    descriptors: np.ndarray,
    device: torch.device,
    candidate_k: int,
    voxel_size_m: float,
    expected_observation_contract: str,
    require_support_gate: bool,
    minimum_support_fraction: float,
    readout_support_threshold: float,
    output_dir: Path,
    field_checkpoint_name: str,
    capability_cache_name: str,
    feature_calibration: str,
    calibration_sample_size: int,
    candidate_readout: str = "scalar",
    require_official_extracted_capability_teachers: bool = False,
) -> dict[str, Any]:
    scene_dir = field_root / "canonical_fields" / scene_id
    field_path = _scene_artifact(
        scene_dir, field_checkpoint_name, role="field checkpoint name"
    )
    capability_path = _scene_artifact(
        scene_dir, capability_cache_name, role="capability cache name"
    )
    if not field_path.is_file() or not capability_path.is_file():
        raise FileNotFoundError(f"PFPR canonical field is incomplete: {scene_dir}")
    field_hash = _sha256(field_path)
    source_metadata = _field_source_metadata(scene_dir)
    source_contract = observation_source_from_render_contract(
        Path(str(source_metadata.get("config", "")))
    )
    validate_pfpr_observation_contract(
        str(expected_observation_contract), source_contract
    )
    validate_field_query_exclusion_commitment(
        str(benchmark_version),
        str(expected_query_source_exclusion_digest),
        str(source_contract.get("field_source_excluded_query_frame_ids_sha256", "")),
    )
    # A full `.sens` render source is not enough on its own: reject a field
    # whose geometry was rebuilt densely but whose canonical DINO evidence was
    # still lifted through a historical 120-view temporal MPR cache.  This is
    # the same label-free provenance gate used by direct AGILE3D prediction.
    validate_full_observation_mpr_contract(
        (
            "scannet_full_observation_pilot"
            if str(expected_observation_contract)
            == "scannet_full_observation_pfpr_queryheldout_v1"
            else ""
        ),
        source_metadata,
        expected_source_contract_sha256=str(
            source_contract["field_source_contract_sha256"]
        ),
        expected_source_contract_version=str(
            source_contract["field_source_contract_version"]
        ),
    )
    declared_contract = str(source_contract["declared_source_contract"])
    bank = load_canonical_capability_bank(
        capability_path, expected_field_checkpoint_sha256=field_hash
    )
    teacher_fidelity = validate_capability_teacher_fidelity(
        bank.metadata,
        require_official_extracted=bool(
            require_official_extracted_capability_teachers
        ),
    )
    gaussian_xyz, covariance, _precision, opacity, _reused = _load_scene_geometry(
        scene_dir,
        bank_xyz=bank.xyz,
        valid_rows=bank.global_rows,
        expected_field_sha256=field_hash,
        cache_path=geometry_cache_root / f"{scene_id}.pt",
        device=device,
    )
    try:
        field = bank.appearance[bank.global_rows].to(device).float()
        primary_local_mask = None
        if candidate_readout == "context_coarse_fine_primary_anchor":
            field_payload = torch.load(field_path, map_location="cpu")
            completion = dict(field_payload.get("support_completion", {}))
            base_mpr_path = str(completion.get("base_mpr_cache", ""))
            if not base_mpr_path:
                raise ValueError(
                    "primary-anchor readout requires a support-completed field"
                )
            base_mpr = torch.load(base_mpr_path, map_location="cpu")
            primary_valid = torch.as_tensor(base_mpr["valid"]).bool().cpu()
            primary_local_mask = primary_valid[bank.global_rows].to(device)
        query = torch.from_numpy(np.asarray(descriptors, dtype=np.float32)).to(device)
        query_shape = query.shape
        if query.ndim == 3:
            query = query.reshape(-1, query.shape[-1])
        field, query = calibrate_query_and_field(
            field,
            query,
            method=str(feature_calibration),
            sample_size=int(calibration_sample_size),
        )
        if len(query_shape) == 3:
            query = query.reshape(query_shape)
        if field.shape[1] != query.shape[-1]:
            raise ValueError("official DINO patch descriptor does not match canonical DINO field")
        precision = _cell_convolved_precision(covariance, voxel_size_m)
        indices = _candidate_indices(gaussian_xyz, candidate_xyz, candidate_k).to(device)
        point_tensor = torch.from_numpy(candidate_xyz).to(device)
        support_probe = torch.ones(len(gaussian_xyz), device=device)
        _unused, support = continuous_gaussian_readout(
            gaussian_xyz,
            covariance,
            support_probe,
            point_tensor,
            gaussian_precision=precision,
            opacity=opacity,
            candidate_k=int(candidate_k),
            candidate_indices=indices,
        )
        valid = support >= float(readout_support_threshold)
        support_fraction = float(valid.float().mean())
        if require_support_gate and support_fraction < float(minimum_support_fraction):
            raise ValueError(
                f"{scene_id} fails PFPR continuous support gate: "
                f"{support_fraction:.6f} < {float(minimum_support_fraction):.6f}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        for local_index, record in enumerate(records):
            prototypes = query[local_index]
            if prototypes.ndim == 1:
                prototypes = prototypes[None]
            if candidate_readout in {
                "context_coarse_fine_equal",
                "context_coarse_fine_interleave",
                "context_coarse_fine_primary_anchor",
            }:
                if len(prototypes) != 2:
                    raise ValueError(
                        "context coarse/fine readout requires [raw, adapted] prototypes"
                    )
                prototype_readouts = ("scalar", "vector_normalized")
            else:
                prototype_readouts = (candidate_readout,) * len(prototypes)
            prototype_scores: list[torch.Tensor] = []
            primary_anchor_scores = None
            primary_anchor_valid = None
            for prototype, prototype_readout in zip(
                prototypes, prototype_readouts
            ):
                if prototype_readout == "scalar":
                    primitive_scores = torch.empty(len(field), device=device)
                    for start in range(0, len(field), 65536):
                        stop = min(start + 65536, len(field))
                        primitive_scores[start:stop] = field[start:stop] @ prototype
                    point_scores, _point_support = continuous_gaussian_readout(
                        gaussian_xyz,
                        covariance,
                        primitive_scores,
                        point_tensor,
                        gaussian_precision=precision,
                        opacity=opacity,
                        candidate_k=int(candidate_k),
                        candidate_indices=indices,
                    )
                    if (
                        candidate_readout
                        == "context_coarse_fine_primary_anchor"
                        and primary_anchor_scores is None
                    ):
                        assert primary_local_mask is not None
                        primary_anchor_scores, primary_support = (
                            continuous_gaussian_readout(
                                gaussian_xyz,
                                covariance,
                                primitive_scores,
                                point_tensor,
                                gaussian_precision=precision,
                                opacity=opacity * primary_local_mask.float(),
                                candidate_k=int(candidate_k),
                                candidate_indices=indices,
                            )
                        )
                        primary_anchor_valid = (
                            primary_support >= float(readout_support_threshold)
                        )
                else:
                    point_scores = _vector_candidate_similarity(
                        gaussian_xyz,
                        covariance,
                        field,
                        prototype,
                        point_tensor,
                        precision=precision,
                        opacity=opacity,
                        candidate_indices=indices,
                        coherence_sqrt=(
                            prototype_readout
                            == "vector_normalized_coherence_sqrt"
                        ),
                    )
                prototype_scores.append(point_scores)
            if candidate_readout in {
                "context_coarse_fine_interleave",
                "context_coarse_fine_primary_anchor",
            }:
                values = _interleaved_coarse_to_fine_scores(
                    candidate_xyz,
                    prototype_scores[0],
                    prototype_scores[1],
                    valid,
                    region_radius_m=2.0 * float(voxel_size_m),
                    maximum_regions=int(candidate_k),
                    rank_one_scores=primary_anchor_scores,
                    rank_one_valid=primary_anchor_valid,
                    refine_rank_one_with_fine=(
                        candidate_readout
                        == "context_coarse_fine_primary_anchor"
                    ),
                )
            else:
                point_scores = (
                    prototype_scores[0]
                    if len(prototype_scores) == 1
                    else (
                        torch.stack(prototype_scores).mean(dim=0)
                        if candidate_readout == "context_coarse_fine_equal"
                        else _fuse_query_prototype_scores(
                            torch.stack(prototype_scores)
                        )
                    )
                )
                values = (
                    point_scores.masked_fill(~valid, -1e30)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            np.save(output_dir / f"{record['query_id']}.npy", values)
        return {
            "scene_id": scene_id,
            "queries": len(records),
            "candidate_points": len(candidate_xyz),
            "continuous_support_fraction": support_fraction,
            "field_checkpoint_sha256": field_hash,
            "source_observation_root": str(source_contract["source_observation_root"]),
            "declared_source_contract": declared_contract,
            "public_query_source_exclusion_digest": str(
                expected_query_source_exclusion_digest
            ),
            "field_source_excluded_query_frame_ids_sha256": str(
                source_contract.get("field_source_excluded_query_frame_ids_sha256", "")
            ),
            "mpr_observation_contract": str(
                dict(source_metadata.get("observation_lifting_contract", {})).get(
                    "name", ""
                )
            ),
            "mpr_declared_views": int(source_metadata.get("num_declared_views", 0)),
            "mpr_full_observation_coverage_order_applied": bool(
                source_metadata.get("full_observation_coverage_order_applied", False)
            ),
            "mpr_full_observation_source_view_count": int(
                source_metadata.get("full_observation_source_view_count", 0)
            ),
            "support_gate_required": bool(require_support_gate),
            "minimum_support_fraction": float(minimum_support_fraction),
            "readout_support_threshold": float(readout_support_threshold),
            "continuous_support_quantiles": {
                name: float(value)
                for name, value in zip(
                    ("p00", "p01", "p05", "p10", "p50", "p90", "p100"),
                    torch.quantile(
                        support.float(),
                        torch.tensor(
                            [0.0, 0.01, 0.05, 0.10, 0.50, 0.90, 1.0],
                            device=support.device,
                        ),
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                )
            },
            "feature_calibration": str(feature_calibration),
            "calibration_sample_size": int(calibration_sample_size),
            "candidate_readout": str(candidate_readout),
            **teacher_fidelity,
        }
    finally:
        del bank
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_dir = Path(args.benchmark_dir)
    field_root = Path(args.field_root)
    geometry_cache_root = Path(args.geometry_cache_root)
    output_dir = Path(args.prediction_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not 0.0 <= float(args.minimum_support_fraction) <= 1.0:
        raise ValueError("minimum_support_fraction must be in [0,1]")
    validate_continuous_support_threshold(
        str(args.expected_observation_contract),
        float(args.readout_support_threshold),
    )
    if str(args.feature_calibration) not in {"none", "diagonal_robust"}:
        raise ValueError("PFPR feature calibration must be none or diagonal_robust")
    if str(args.candidate_readout) not in {
        "scalar",
        "vector_normalized",
        "vector_normalized_coherence_sqrt",
        "context_coarse_fine_equal",
        "context_coarse_fine_interleave",
        "context_coarse_fine_primary_anchor",
    }:
        raise ValueError("unknown PFPR candidate readout")
    if str(args.candidate_readout) in {
        "context_coarse_fine_equal",
        "context_coarse_fine_interleave",
        "context_coarse_fine_primary_anchor",
    }:
        if not str(args.crop_context_adapter_checkpoint).strip():
            raise ValueError(
                "context coarse/fine readout requires a crop-context adapter"
            )
        if str(args.query_pooling) != "center3x3":
            raise ValueError(
                "context coarse/fine readout requires center3x3 query pooling"
            )
    if int(args.calibration_sample_size) <= 0:
        raise ValueError("PFPR calibration sample size must be positive")
    (
        by_scene,
        candidates,
        benchmark_version,
        exclusion_commitments,
    ) = _load_method_queries(benchmark_dir)
    crop_spatial_adapter = None
    crop_spatial_adapter_manifest = None
    crop_spatial_adapter_path = ""
    crop_context_adapter = None
    crop_context_adapter_manifest = None
    crop_context_adapter_path = ""
    if (
        str(args.crop_spatial_adapter_checkpoint).strip()
        and str(args.crop_context_adapter_checkpoint).strip()
    ):
        raise ValueError("choose either crop-spatial or crop-context adapter, not both")
    if str(args.crop_spatial_adapter_checkpoint).strip():
        if str(args.query_pooling) != "center3x3":
            raise ValueError(
                "the frozen crop-spatial adapter was trained for center3x3 "
                "queries and cannot be applied to center-token ablations"
            )
        crop_spatial_adapter_path = str(Path(args.crop_spatial_adapter_checkpoint).resolve())
        crop_spatial_adapter, crop_spatial_adapter_manifest = (
            GlobalCropSpatialAdapter.from_checkpoint(crop_spatial_adapter_path)
        )
        crop_spatial_adapter = crop_spatial_adapter.to(device).eval()
    if str(args.crop_context_adapter_checkpoint).strip():
        if str(args.query_pooling) != "center3x3":
            raise ValueError(
                "the frozen crop-context adapter was trained for center3x3 "
                "queries and cannot be applied to center-token ablations"
            )
        crop_context_adapter_path = str(
            Path(args.crop_context_adapter_checkpoint).resolve()
        )
        crop_context_adapter, crop_context_adapter_manifest = (
            GlobalCropContextAdapter.from_checkpoint(crop_context_adapter_path)
        )
        crop_context_adapter = crop_context_adapter.to(device).eval()
    if str(args.scene_names).strip():
        requested = set(str(args.scene_names).replace(",", " ").split())
        unknown = requested - set(by_scene)
        if unknown:
            raise ValueError(f"unknown PFPR scenes: {sorted(unknown)}")
        by_scene = {scene: by_scene[scene] for scene in sorted(requested)}
    reports: list[dict[str, Any]] = []
    for scene in sorted(by_scene):
        descriptors = _encode_scene_queries(
            by_scene[scene],
            device=device,
            radio_repo=args.radio_repo,
            radio_version=args.radio_version,
            batch_size=int(args.batch_size),
            query_pooling=str(args.query_pooling),
            crop_spatial_adapter=crop_spatial_adapter,
            crop_context_adapter=crop_context_adapter,
            preserve_raw_context_pair=(
                str(args.candidate_readout)
                in {
                    "context_coarse_fine_equal",
                    "context_coarse_fine_interleave",
                    "context_coarse_fine_primary_anchor",
                }
            ),
        )
        reports.append(
            _score_scene(
                scene,
                by_scene[scene],
                candidates[scene],
                benchmark_version=benchmark_version,
                expected_query_source_exclusion_digest=exclusion_commitments[scene],
                field_root=field_root,
                geometry_cache_root=geometry_cache_root,
                descriptors=descriptors,
                device=device,
                candidate_k=int(args.candidate_k),
                voxel_size_m=float(args.candidate_voxel_size_m),
                expected_observation_contract=str(args.expected_observation_contract),
                require_support_gate=bool(args.require_support_gate),
                minimum_support_fraction=float(args.minimum_support_fraction),
                readout_support_threshold=float(args.readout_support_threshold),
                output_dir=output_dir,
                field_checkpoint_name=str(args.field_checkpoint_name),
                capability_cache_name=str(args.capability_cache_name),
                feature_calibration=str(args.feature_calibration),
                calibration_sample_size=int(args.calibration_sample_size),
                candidate_readout=str(args.candidate_readout),
                require_official_extracted_capability_teachers=bool(
                    args.require_official_extracted_capability_teachers
                ),
            )
        )
    if crop_context_adapter is not None:
        method_name = (
            "official_dino_v3_7b_center_3x3_global_crop_context_adapter"
            "_to_canonical_dino_field"
        )
    elif crop_spatial_adapter is not None:
        method_name = (
            "official_dino_v3_7b_center_3x3_global_crop_spatial_adapter"
            "_to_canonical_dino_field"
        )
    else:
        method_name = (
            f"official_dino_v3_7b_{args.query_pooling}_to_canonical_dino_field"
        )
    if str(args.feature_calibration) != "none":
        method_name = f"{method_name}_with_{args.feature_calibration}_scene_calibration"
    if str(args.candidate_readout) != "scalar":
        method_name = f"{method_name}_with_{args.candidate_readout}_readout"

    report = {
        "benchmark": benchmark_version,
        "method": method_name,
        "protocol": {
            "method_manifest_only": True,
            "evaluator_anchor_opened": False,
            "query_descriptor": (
                "official_dino_v3_7b_raw_center_3x3_and_frozen_context_adapted_center_3x3_pair"
                if str(args.candidate_readout)
                in {
                    "context_coarse_fine_equal",
                    "context_coarse_fine_interleave",
                    "context_coarse_fine_primary_anchor",
                }
                else (
                "official_dino_v3_7b_center_3x3_plus_spatial_global_mean_then_frozen_global_crop_context_adapter"
                if crop_context_adapter is not None
                else (
                "official_dino_v3_7b_center_3x3_then_frozen_global_crop_spatial_adapter"
                if crop_spatial_adapter is not None
                else (
                    "official_dino_v3_7b_center3x3_and_center_token_fixed_late_fusion"
                    if str(args.query_pooling) == "center_late_fusion"
                    else (
                        "official_dino_v3_7b_center_3x3_l2_mean"
                        if str(args.query_pooling) == "center3x3"
                        else "official_dino_v3_7b_center_token_l2_normalized"
                    )
                )
                )
                )
            ),
            "crop_spatial_adapter": (
                {
                    "checkpoint": crop_spatial_adapter_path,
                    "checkpoint_sha256": crop_spatial_adapter_sha256(crop_spatial_adapter_path),
                    "manifest": crop_spatial_adapter_manifest.__dict__,
                    "query_independent": True,
                    "evaluator_anchor_opened": False,
                }
                if crop_spatial_adapter is not None and crop_spatial_adapter_manifest is not None
                else None
            ),
            "crop_context_adapter": (
                {
                    "checkpoint": crop_context_adapter_path,
                    "checkpoint_sha256": crop_spatial_adapter_sha256(
                        crop_context_adapter_path
                    ),
                    "manifest": crop_context_adapter_manifest.__dict__,
                    "query_independent": True,
                    "evaluator_anchor_opened": False,
                }
                if crop_context_adapter is not None
                and crop_context_adapter_manifest is not None
                else None
            ),
            "primitive_score": (
                "scene_diagonal_robust_normalized_cosine"
                if str(args.feature_calibration) == "diagonal_robust"
                else "cosine"
            ),
            "feature_calibration": str(args.feature_calibration),
            "calibration_sample_size": int(args.calibration_sample_size),
            "candidate_readout": str(args.candidate_readout),
            "candidate_readout_geometry": (
                "continuous_opacity_weighted_gaussian_convolved_with_5cm_voxel_cell"
            ),
            "query_prototype_fusion": (
                {
                    "method": "equal_score_mean",
                    "weights": [0.5, 0.5],
                    "prototypes": ["raw_center3x3", "context_adapted_center3x3"],
                    "readouts": ["scalar", "vector_normalized"],
                }
                if str(args.candidate_readout) == "context_coarse_fine_equal"
                else (
                    {
                        "method": "fine_top1_then_interleaved_context_region_proposals",
                        "region_radius": "2x_candidate_voxel_size",
                        "prototypes": ["raw_center3x3", "context_adapted_center3x3"],
                        "readouts": ["scalar", "vector_normalized"],
                        "region_refinement": "raw_scalar_argmax",
                        "uses_query_pose_depth_or_gt": False,
                    }
                    if str(args.candidate_readout)
                    == "context_coarse_fine_interleave"
                    else (
                    {
                        "method": "logmeanexp",
                        "temperature": 0.1,
                        "prototype_count": 2,
                    }
                    if str(args.query_pooling) == "center_late_fusion"
                    else None
                    )
                )
            ),
            "candidate_k": int(args.candidate_k),
            "candidate_voxel_size_m": float(args.candidate_voxel_size_m),
            "field_checkpoint_name": str(args.field_checkpoint_name),
            "capability_cache_name": str(args.capability_cache_name),
            "expected_observation_contract": str(args.expected_observation_contract),
            "support_gate_required": bool(args.require_support_gate),
            "minimum_support_fraction": float(args.minimum_support_fraction),
            "readout_support_threshold": float(args.readout_support_threshold),
            "requires_official_extracted_capability_teachers": bool(
                args.require_official_extracted_capability_teachers
            ),
        },
        "scene_reports": reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prediction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--field-root", required=True)
    parser.add_argument("--geometry-cache-root", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--field-checkpoint-name", default="canonical_mpr_v2.pt")
    parser.add_argument("--capability-cache-name", default="official_dino_sam3_views.pt")
    parser.add_argument(
        "--require-official-extracted-capability-teachers",
        action="store_true",
        help=(
            "require native official C-RADIO DINO/SAM maps in both MPR and "
            "render-fidelity provenance before scoring"
        ),
    )
    parser.add_argument("--scene-names", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--radio-repo", default="/root/RADIO")
    parser.add_argument("--radio-version", default="c-radio_v4-h")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--query-pooling",
        choices=("center3x3", "center", "center_late_fusion"),
        default="center3x3",
        help=(
            "frozen PFPR patch-center descriptor; center and center_late_fusion "
            "are explicit query-only ablations"
        ),
    )
    parser.add_argument(
        "--candidate-readout",
        choices=(
            "scalar",
            "vector_normalized",
            "vector_normalized_coherence_sqrt",
            "context_coarse_fine_equal",
            "context_coarse_fine_interleave",
            "context_coarse_fine_primary_anchor",
        ),
        default="scalar",
        help="explicit candidate feature-readout ablation; scalar is the frozen default",
    )
    parser.add_argument("--candidate-k", type=int, default=64)
    parser.add_argument("--candidate-voxel-size-m", type=float, default=0.05)
    parser.add_argument(
        "--feature-calibration",
        choices=("none", "diagonal_robust"),
        default="none",
        help="query-independent canonical-DINO scene normalization",
    )
    parser.add_argument("--calibration-sample-size", type=int, default=8192)
    parser.add_argument(
        "--expected-observation-contract",
        default="",
        help="fail closed unless every render contract declares this source contract",
    )
    parser.add_argument("--require-support-gate", action="store_true")
    parser.add_argument("--minimum-support-fraction", type=float, default=0.95)
    parser.add_argument(
        "--readout-support-threshold",
        type=float,
        default=1e-6,
        help=(
            "minimum continuous Gaussian support for public point scoring; "
            f"full-observation fields require >= {FULL_OBSERVATION_MIN_MEANINGFUL_SUPPORT:g}"
        ),
    )
    parser.add_argument(
        "--crop-spatial-adapter-checkpoint",
        default="",
        help="optional frozen global crop-to-full-image DINO bridge; never scene-trained",
    )
    parser.add_argument(
        "--crop-context-adapter-checkpoint",
        default="",
        help=(
            "optional frozen global centre-plus-context bridge trained only on "
            "scene-disjoint RGB crop/full-image pairs"
        ),
    )
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
