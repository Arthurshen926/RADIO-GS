"""Source-only SAM mask transport ladder without benchmark labels or queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.v4.carrier import GaussianCarrier, MeshCarrier, SurfaceCarrier, SurfaceVoxelCarrier
from radio_gs.v4.contracts.geometry_receipt import GeometryReceipt, HashedInput, sha256_file
from radio_gs.v4.evaluation.geometry_ladder import _load_cameras, _load_mesh
from radio_gs.v4.registration.surface_projection import element_purity, projection_entropy


def _load_masks(record: dict[str, Any], height: int, width: int) -> torch.Tensor:
    payload = torch.load(Path(record["mask_cache"]).resolve(strict=True), map_location="cpu")
    shape = tuple(map(int, payload["mask_shape"]))
    packed = torch.as_tensor(payload["packed_masks"]).cpu().numpy()
    values = np.unpackbits(packed, axis=-1, bitorder="little")[..., : shape[1]].astype(bool)
    masks = torch.from_numpy(values)
    if masks.shape[1:] != shape:
        raise ValueError("packed source mask shape differs from receipt")
    return F.interpolate(masks[:, None].float(), size=(height, width), mode="nearest")[:, 0]


def _lift_mask_matrix(carrier: SurfaceCarrier, camera, masks: torch.Tensor) -> torch.Tensor:
    projection = carrier.project(camera)
    numerator = torch.zeros(carrier.num_elements, masks.shape[0])
    numerator.index_add_(
        0,
        projection.element_ids,
        masks.reshape(masks.shape[0], -1)[:, projection.pixel_ids].T * projection.weights[:, None],
    )
    denominator = torch.zeros(carrier.num_elements)
    denominator.scatter_add_(0, projection.element_ids, projection.weights)
    return numerator / denominator.clamp_min(1e-12)[:, None]


def _soft_iou(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.float().clamp(0, 1)
    target = target.float()
    intersection = (prediction * target).sum()
    union = prediction.sum() + target.sum() - intersection
    return float(intersection / union.clamp_min(1e-12))


def _leakage(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.float().clamp(0, 1)
    return float(prediction[target < 0.5].sum() / prediction.sum().clamp_min(1e-12))


def _mutually_exclusive_purity(
    posterior: torch.Tensor, masks: torch.Tensor
) -> tuple[list[float], int]:
    """Purity only for pixel-disjoint proposals, never parent/part overlaps."""

    values: list[float] = []
    pair_count = 0
    binary = masks > 0.5
    for first in range(binary.shape[0]):
        for second in range(first + 1, binary.shape[0]):
            if bool((binary[first] & binary[second]).any()):
                continue
            pair_count += 1
            evidence = posterior[:, [first, second]]
            denominator = evidence.sum(-1)
            observed = denominator > 0
            if bool(observed.any()):
                values.append(float((evidence.max(-1).values[observed] / denominator[observed]).mean()))
    return values, pair_count


def _evaluate(
    carrier: SurfaceCarrier,
    cameras,
    masks_by_view: list[torch.Tensor],
) -> dict[str, Any]:
    posteriors_by_view = [
        _lift_mask_matrix(carrier, camera, masks)
        for camera, masks in zip(cameras, masks_by_view)
    ]
    counts = [value.shape[1] for value in posteriors_by_view]
    offsets = np.cumsum([0, *counts])
    all_posteriors = torch.cat(posteriors_by_view, dim=1)
    rendered = [carrier.render_posterior(all_posteriors, camera) for camera in cameras]
    same_iou, same_leakage, cross_best_iou, cross_best_leakage = [], [], [], []
    for source_index, source_masks in enumerate(masks_by_view):
        for local_index, source_mask in enumerate(source_masks):
            proposal_index = int(offsets[source_index] + local_index)
            same_prediction = rendered[source_index][..., proposal_index]
            same_iou.append(_soft_iou(same_prediction, source_mask))
            same_leakage.append(_leakage(same_prediction, source_mask))
            for target_index, target_masks in enumerate(masks_by_view):
                if target_index == source_index or target_masks.shape[0] == 0:
                    continue
                prediction = rendered[target_index][..., proposal_index]
                overlaps = torch.tensor([_soft_iou(prediction, target) for target in target_masks])
                selected = int(overlaps.argmax())
                cross_best_iou.append(float(overlaps[selected]))
                cross_best_leakage.append(_leakage(prediction, target_masks[selected]))
    hierarchy_inclusive_purity = [
        element_purity(posterior, posterior.sum(-1) > 0)
        for posterior in posteriors_by_view
    ]
    exclusive_purity_values: list[float] = []
    exclusive_pair_count = 0
    for posterior, masks in zip(posteriors_by_view, masks_by_view):
        values, count = _mutually_exclusive_purity(posterior, masks)
        exclusive_purity_values.extend(values)
        exclusive_pair_count += count
    entropy = [projection_entropy(carrier.project(camera)) for camera in cameras]
    mask_coverage = []
    for camera, masks in zip(cameras, masks_by_view):
        support = carrier.project(camera).pixel_weight_sum() > 0
        for mask in masks:
            mask_coverage.append(float((support & (mask > 0.5)).sum() / (mask > 0.5).sum().clamp_min(1)))
    def mean(values: list[float]) -> float:
        finite = [value for value in values if np.isfinite(value)]
        if not finite:
            raise RuntimeError("source-mask metric has no finite observations")
        return float(np.mean(finite))
    return {
        "same_view_mask_roundtrip_soft_iou": mean(same_iou),
        "same_view_boundary_leakage": mean(same_leakage),
        "cross_view_best_target_proposal_soft_iou": mean(cross_best_iou),
        "cross_view_best_target_proposal_leakage": mean(cross_best_leakage),
        "mutually_exclusive_element_purity": mean(exclusive_purity_values),
        "mutually_exclusive_proposal_pair_count": exclusive_pair_count,
        "hierarchy_inclusive_element_purity_diagnostic": mean(hierarchy_inclusive_purity),
        "source_mask_surface_coverage": mean(mask_coverage),
        "registration_entropy": {
            key: mean([record[key] for record in entropy]) for key in entropy[0]
        },
        "proposal_count": int(sum(counts)),
        "cross_view_pair_count": len(cross_best_iou),
        "cross_view_metric_boundary": (
            "best available target proposal; proposals are not tracked, so this is a "
            "source-only transport diagnostic rather than object-correspondence ground truth"
        ),
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    transforms = Path(args.transforms).resolve(strict=True)
    mesh_path = Path(args.mesh).resolve(strict=True)
    authority_path = Path(args.source_authority).resolve(strict=True)
    authority = torch.load(authority_path, map_location="cpu")
    metadata = authority.get("metadata", {})
    forbidden = ("benchmark_images_opened", "benchmark_masks_opened", "evaluation_rgb_opened", "text_queries_opened")
    if any(metadata.get(key) is not False for key in forbidden):
        raise ValueError("source authority is not sealed from benchmark/query inputs")
    records = list(metadata["source_records"])
    height, width = int(metadata["feature_height"]), int(metadata["feature_width"])
    cameras = _load_cameras(transforms, records, height, width)
    masks_by_view = [_load_masks(record, height, width) for record in records]
    if any(masks.shape[0] != int(record["num_proposals"]) for masks, record in zip(masks_by_view, records)):
        raise ValueError("source proposal count differs from mask receipt")
    vertices, triangles = _load_mesh(mesh_path)
    mesh_surface = MeshCarrier(vertices, triangles)
    carriers: dict[str, SurfaceCarrier] = {
        "gaussian_exact_renderer": GaussianCarrier(
            int(authority["num_rows"]),
            {str(int(record["frame_id"])): record["responsibility_view"] for record in records},
        ),
        "mesh_surface": mesh_surface,
        "mesh_derived_sparse_surface": SurfaceVoxelCarrier.from_points(
            vertices,
            args.voxel_size,
            normals=mesh_surface.normals,
            maximum_splat_radius=args.maximum_splat_radius,
            surface_band_voxels=args.surface_band_voxels,
            maximum_contributors_per_pixel=args.maximum_contributors_per_pixel,
        ),
    }
    results = {}
    for name, carrier in carriers.items():
        print(f"evaluating source-only carrier: {name}", flush=True)
        results[name] = _evaluate(carrier, cameras, masks_by_view)
    baseline = results["gaussian_exact_renderer"]
    comparison = {}
    for name in ("mesh_surface", "mesh_derived_sparse_surface"):
        candidate = results[name]
        comparison[name] = {
            "roundtrip_delta": candidate["same_view_mask_roundtrip_soft_iou"] - baseline["same_view_mask_roundtrip_soft_iou"],
            "transfer_delta": candidate["cross_view_best_target_proposal_soft_iou"] - baseline["cross_view_best_target_proposal_soft_iou"],
            "same_view_leakage_reduction": baseline["same_view_boundary_leakage"] - candidate["same_view_boundary_leakage"],
            "cross_view_leakage_reduction": baseline["cross_view_best_target_proposal_leakage"] - candidate["cross_view_best_target_proposal_leakage"],
            "mutually_exclusive_purity_delta": candidate["mutually_exclusive_element_purity"] - baseline["mutually_exclusive_element_purity"],
            "coverage_delta": candidate["source_mask_surface_coverage"] - baseline["source_mask_surface_coverage"],
            "effective_contributors_reduction": baseline["registration_entropy"]["effective_contributors"] - candidate["registration_entropy"]["effective_contributors"],
        }
    inputs = [
        HashedInput.seal("camera_transforms", transforms),
        HashedInput.seal("source_authority", authority_path),
        HashedInput.seal("mesh_geometry", mesh_path),
        *[
            HashedInput.seal(f"source_mask_{index}", record["mask_cache"])
            for index, record in enumerate(records)
        ],
        *[
            HashedInput.seal(f"source_projection_{index}", record["responsibility_view"])
            for index, record in enumerate(records)
        ],
    ]
    receipt = GeometryReceipt(
        carrier="multi_carrier_source_mask_ladder",
        coordinate_convention="nerf_opengl_to_opencv_camera_to_world_feature_raster",
        inputs=tuple(inputs),
        source_rgb_opened=False,
        target_rgb_opened=False,
        benchmark_images_opened=False,
        benchmark_masks_opened=False,
        benchmark_labels_opened=False,
        metadata={
            "source_only": True,
            "voxel_size": args.voxel_size,
            "maximum_splat_radius": args.maximum_splat_radius,
            "surface_band_voxels": args.surface_band_voxels,
            "maximum_contributors_per_pixel": args.maximum_contributors_per_pixel,
        },
    )
    report = {
        "schema": "radio_gs.surface_object_memory_v4.source_mask_geometry_ladder.v1",
        "stage": "geometry_registration_only",
        "scene_label": args.scene_label,
        "source_view_count": len(cameras),
        "raster_shape": [height, width],
        "results": results,
        "comparison_to_gaussian": comparison,
        "geometry_receipt": receipt.to_dict(),
        "downstream_gates": {
            "object_codebook_started": False,
            "query_encoder_started": False,
            "compression_started": False,
        },
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--source-authority", required=True)
    parser.add_argument("--scene-label", required=True)
    parser.add_argument("--voxel-size", type=float, required=True)
    parser.add_argument("--maximum-splat-radius", type=int, default=3)
    parser.add_argument("--surface-band-voxels", type=float, default=1.5)
    parser.add_argument("--maximum-contributors-per-pixel", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (
        args.voxel_size <= 0
        or args.maximum_splat_radius < 0
        or args.surface_band_voxels < 0
        or args.maximum_contributors_per_pixel <= 0
    ):
        parser.error("voxel size must be positive and splat radius non-negative")
    report = run(args)
    print(json.dumps(report["comparison_to_gaussian"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
