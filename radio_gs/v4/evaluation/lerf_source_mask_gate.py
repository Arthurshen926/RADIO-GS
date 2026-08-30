"""Source-only SAM3 geometry gate for a calibrated LERF surface carrier."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.data.lerf_dataset import _read_cameras_binary
from radio_gs.models.sam3_multiscale_hierarchy import unpack_masks
from radio_gs.v4.carrier import Camera, GaussianCarrier, SurfaceVoxelCarrier
from radio_gs.v4.contracts.geometry_receipt import GeometryReceipt, HashedInput, sha256_file
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_images
from radio_gs.v4.registration.surface_projection import projection_entropy


def _index(image_id: str) -> int:
    values = re.findall(r"\d+", image_id)
    if not values:
        raise ValueError(f"SAM3 image id has no frame index: {image_id}")
    return int(values[-1])


def _mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        raise RuntimeError("LERF source-mask metric has no finite observations")
    return float(np.mean(finite))


def _load_sam_records(manifest_paths: list[Path]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("contract") != "official-sam3-query-free-multiscale-hierarchy-manifest-v1":
            raise ValueError("SAM3 manifest contract differs")
        contract = manifest.get("generation_contract", {})
        if contract.get("official_decoder") is not True or contract.get("query_free") is not True:
            raise ValueError("SAM3 source masks are not official query-free decoder outputs")
        for record in manifest["images"]:
            frame_index = _index(str(record["image_id"]))
            if frame_index in records:
                raise ValueError(f"duplicate SAM3 source frame {frame_index}")
            output = Path(record["output"]).resolve(strict=True)
            if sha256_file(output) != record["output_sha256"]:
                raise ValueError(f"SAM3 cache digest differs for frame {frame_index}")
            records[frame_index] = {**record, "output": str(output)}
    return records


def _masks(path: Path, height: int, width: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    source_height, source_width = map(int, payload["mask_shape"])
    masks = torch.from_numpy(unpack_masks(payload["packed_masks"], width=source_width)).float()
    if masks.shape[1:] != (source_height, source_width):
        raise ValueError("SAM3 mask raster differs from its cache shape")
    return F.interpolate(masks[:, None], size=(height, width), mode="nearest")[:, 0]


def _lift_and_render(carrier, camera: Camera, masks: torch.Tensor, device: torch.device):
    projection = carrier.project(camera)
    element_ids = projection.element_ids.to(device)
    pixel_ids = projection.pixel_ids.to(device)
    weights = projection.weights.to(device)
    flat_masks = masks.reshape(masks.shape[0], -1).T.to(device)
    posterior_sum = torch.zeros(carrier.num_elements, masks.shape[0], device=device)
    posterior_sum.index_add_(0, element_ids, flat_masks[pixel_ids] * weights[:, None])
    element_weight = torch.zeros(carrier.num_elements, device=device)
    element_weight.scatter_add_(0, element_ids, weights)
    posterior = posterior_sum / element_weight.clamp_min(1e-12)[:, None]
    rendered = torch.zeros(projection.num_pixels, masks.shape[0], device=device)
    rendered.index_add_(0, pixel_ids, posterior[element_ids] * weights[:, None])
    if projection.normalization == "weighted_mean":
        pixel_weight = torch.zeros(projection.num_pixels, device=device)
        pixel_weight.scatter_add_(0, pixel_ids, weights)
        rendered /= pixel_weight.clamp_min(1e-12)[:, None]
    return posterior, rendered.reshape(camera.height, camera.width, -1), projection


def _mutually_exclusive_purity_fast(
    posterior: torch.Tensor,
    masks: torch.Tensor,
    pair_chunk_size: int = 8,
) -> tuple[list[float], int]:
    """Vectorized equivalent of the source-mask disjoint-proposal purity metric.

    Pair chunks bound temporary GPU memory for Gaussian carriers while preserving
    the exact per-pair averaging used by the reference implementation.
    """

    binary = (masks > 0.5).reshape(masks.shape[0], -1).float()
    overlap = binary @ binary.T
    first, second = torch.where(torch.triu(overlap == 0, diagonal=1))
    pair_count = int(first.numel())
    values: list[float] = []
    for start in range(0, pair_count, pair_chunk_size):
        stop = min(start + pair_chunk_size, pair_count)
        first_chunk = first[start:stop].to(posterior.device)
        second_chunk = second[start:stop].to(posterior.device)
        first_evidence = posterior[:, first_chunk]
        second_evidence = posterior[:, second_chunk]
        denominator = first_evidence + second_evidence
        observed = denominator > 0
        purity = torch.maximum(first_evidence, second_evidence) / denominator.clamp_min(1e-12)
        observed_count = observed.sum(0)
        pair_mean = (purity * observed).sum(0) / observed_count.clamp_min(1)
        values.extend(
            float(value)
            for value, count in zip(pair_mean.cpu(), observed_count.cpu())
            if int(count) > 0
        )
    return values, pair_count


def _evaluate(
    carrier,
    cameras: list[Camera],
    masks_by_view: list[torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    roundtrip, leakage, purity, coverage, entropy = [], [], [], [], []
    per_view = []
    exclusive_pairs = 0
    for camera, masks in zip(cameras, masks_by_view):
        posterior, rendered, projection = _lift_and_render(carrier, camera, masks, device)
        target = masks.permute(1, 2, 0).to(device)
        prediction = rendered.float().clamp(0, 1)
        intersection = (prediction * target).sum((0, 1))
        union = prediction.sum((0, 1)) + target.sum((0, 1)) - intersection
        view_roundtrip = intersection / union.clamp_min(1e-12)
        view_leakage = (prediction * (target < 0.5)).sum((0, 1)) / prediction.sum((0, 1)).clamp_min(1e-12)
        roundtrip.extend(map(float, view_roundtrip.cpu()))
        leakage.extend(map(float, view_leakage.cpu()))
        values, pair_count = _mutually_exclusive_purity_fast(posterior, masks)
        purity.extend(values)
        exclusive_pairs += pair_count
        support = projection.pixel_weight_sum() > 0
        binary_target = masks > 0.5
        view_coverage = (binary_target & support[None]).sum((1, 2)) / binary_target.sum((1, 2)).clamp_min(1)
        coverage.extend(map(float, view_coverage))
        view_entropy = projection_entropy(projection)
        entropy.append(view_entropy)
        per_view.append({
            "camera_key": camera.key,
            "proposal_count": int(masks.shape[0]),
            "same_view_mask_roundtrip_soft_iou": float(view_roundtrip.mean()),
            "same_view_boundary_leakage": float(view_leakage.mean()),
            "mutually_exclusive_element_purity": _mean(values) if values else None,
            "source_mask_surface_coverage": float(view_coverage.mean()),
            "registration_entropy": view_entropy,
        })
        del posterior
    return {
        "same_view_mask_roundtrip_soft_iou": _mean(roundtrip),
        "same_view_boundary_leakage": _mean(leakage),
        "mutually_exclusive_element_purity": _mean(purity),
        "mutually_exclusive_proposal_pair_count": exclusive_pairs,
        "source_mask_surface_coverage": _mean(coverage),
        "registration_entropy": {key: _mean([record[key] for record in entropy]) for key in entropy[0]},
        "proposal_count": int(sum(masks.shape[0] for masks in masks_by_view)),
        "per_view_diagnostics": per_view,
        "cross_view_metric": "omitted: untracked best-proposal matching is non-gating and quadratic in proposal count",
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(int(getattr(args, "cpu_threads", 4)))
    scene_root = Path(args.scene_root).resolve(strict=True)
    source_authority_path = Path(args.source_rgb_authority).resolve(strict=True)
    source_authority = json.loads(source_authority_path.read_text())
    if source_authority.get("contract") != "sam3-query-free-source-rgb-authority-v1":
        raise ValueError("source RGB authority contract differs")
    sam_manifests = [Path(value).resolve(strict=True) for value in args.sam_manifest]
    sam_records = _load_sam_records(sam_manifests)
    expected_frames = {_index(str(record["image_id"])) for record in source_authority["images"]}
    if set(sam_records) != expected_frames:
        raise ValueError("SAM3 cache frames differ from the sealed source RGB authority")

    exact_path = Path(args.exact_mpr_authority).resolve(strict=True)
    exact = json.loads(exact_path.read_text())
    exact_views = {int(record["frame_index"]): record for record in exact["views"]}
    missing_projection = sorted(expected_frames - set(exact_views))
    if missing_projection:
        raise KeyError(f"source frames lack exact-MPR projections: {missing_projection}")
    projection_paths = {
        str(index): exact_path.parent / exact_views[index]["relative_path"] for index in expected_frames
    }

    sparse = scene_root / "sparse" / "0"
    cameras_raw = _read_cameras_binary(sparse / "cameras.bin")
    colmap_views = _read_images(sparse / "images.bin", cameras_raw)
    height, width = int(exact["metadata"]["feature_height"]), int(exact["metadata"]["feature_width"])
    ordered_frames = sorted(expected_frames)
    cameras, masks_by_view = [], []
    for frame_index in ordered_frames:
        view = colmap_views[frame_index]
        intrinsic = view.intrinsic.clone()
        intrinsic[0] *= width / view.width
        intrinsic[1] *= height / view.height
        cameras.append(Camera(str(frame_index), intrinsic, view.camera_to_world, height, width))
        masks_by_view.append(_masks(Path(sam_records[frame_index]["output"]), height, width))

    surface_path = Path(args.surface_carrier).resolve(strict=True)
    surface = torch.load(surface_path, map_location="cpu")
    carriers = {
        "gaussian_exact_renderer": GaussianCarrier(int(exact["num_gaussians"]), projection_paths),
        "moge3_calibrated_sparse_surface": SurfaceVoxelCarrier(
            surface["centres"],
            float(surface["voxel_size_colmap"]),
            normals=surface["normals"],
            confidence=surface["confidence"],
            maximum_splat_radius=args.maximum_splat_radius,
            surface_band_voxels=args.surface_band_voxels,
            maximum_contributors_per_pixel=args.maximum_contributors_per_pixel,
        ),
    }
    results = {}
    for name, carrier in carriers.items():
        print(f"evaluating LERF source-only carrier: {name}", flush=True)
        results[name] = _evaluate(carrier, cameras, masks_by_view, torch.device(args.compute_device))
    baseline = results["gaussian_exact_renderer"]
    candidate = results["moge3_calibrated_sparse_surface"]
    comparison = {
        "roundtrip_delta": candidate["same_view_mask_roundtrip_soft_iou"] - baseline["same_view_mask_roundtrip_soft_iou"],
        "same_view_leakage_reduction": baseline["same_view_boundary_leakage"] - candidate["same_view_boundary_leakage"],
        "mutually_exclusive_purity_delta": candidate["mutually_exclusive_element_purity"] - baseline["mutually_exclusive_element_purity"],
        "coverage_delta": candidate["source_mask_surface_coverage"] - baseline["source_mask_surface_coverage"],
        "effective_contributors_reduction": baseline["registration_entropy"]["effective_contributors"] - candidate["registration_entropy"]["effective_contributors"],
    }
    primary = {
        "roundtrip": comparison["roundtrip_delta"] > args.minimum_delta,
        "leakage": comparison["same_view_leakage_reduction"] > args.minimum_delta,
        "purity": comparison["mutually_exclusive_purity_delta"] >= 0,
    }
    receipt = GeometryReceipt(
        carrier="lerf_gaussian_vs_calibrated_moge3_sparse_surface",
        coordinate_convention="colmap_world_opencv_camera_feature_raster",
        inputs=(
            HashedInput.seal("source_rgb_authority", source_authority_path),
            HashedInput.seal("exact_mpr_authority", exact_path),
            HashedInput.seal("surface_carrier", surface_path),
            HashedInput.seal("colmap_cameras", sparse / "cameras.bin"),
            HashedInput.seal("colmap_images", sparse / "images.bin"),
            *tuple(HashedInput.seal(f"sam3_manifest_{index}", path) for index, path in enumerate(sam_manifests)),
        ),
        source_rgb_opened=True,
        target_rgb_opened=False,
        benchmark_images_opened=False,
        benchmark_masks_opened=False,
        benchmark_labels_opened=False,
        model_family="MoGe-3",
        model_checkpoint_sha256=str(surface.get("model_checkpoint_sha256", args.model_checkpoint_sha256)),
        metadata={
            "source_only": True,
            "scene_label": args.scene_label,
            "source_view_count": len(cameras),
            "maximum_splat_radius": args.maximum_splat_radius,
            "surface_band_voxels": args.surface_band_voxels,
            "maximum_contributors_per_pixel": args.maximum_contributors_per_pixel,
        },
    )
    report = {
        "schema": "radio_gs.surface_object_memory_v4.lerf_source_mask_geometry_gate.v1",
        "scene_label": args.scene_label,
        "source_view_count": len(cameras),
        "raster_shape": [height, width],
        "projection_configuration": {
            "maximum_splat_radius": args.maximum_splat_radius,
            "surface_band_voxels": args.surface_band_voxels,
            "maximum_contributors_per_pixel": args.maximum_contributors_per_pixel,
        },
        "results": results,
        "comparison_to_gaussian": comparison,
        "primary_directions": primary,
        "passes_scene_gate": all(primary.values()),
        "coverage_is_reported_not_compensatory": True,
        "geometry_receipt": receipt.to_dict(),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--scene-label", required=True)
    parser.add_argument("--source-rgb-authority", required=True)
    parser.add_argument("--sam-manifest", action="append", required=True)
    parser.add_argument("--exact-mpr-authority", required=True)
    parser.add_argument("--surface-carrier", required=True)
    parser.add_argument("--model-checkpoint-sha256", required=True)
    parser.add_argument("--maximum-splat-radius", type=int, default=3)
    parser.add_argument("--surface-band-voxels", type=float, default=1.5)
    parser.add_argument("--maximum-contributors-per-pixel", type=int, default=8)
    parser.add_argument("--minimum-delta", type=float, default=0.0)
    parser.add_argument("--compute-device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"passes_scene_gate": report["passes_scene_gate"], "comparison": report["comparison_to_gaussian"]}, indent=2))


if __name__ == "__main__":
    main()
