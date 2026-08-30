"""Fit a query-free proposal metric from source geometry consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.v4.carrier import MeshCarrier, SurfaceVoxelCarrier
from radio_gs.v4.contracts.geometry_receipt import GeometryReceipt, HashedInput, sha256_file
from radio_gs.v4.evaluation.geometry_ladder import _load_cameras, _load_mesh
from radio_gs.v4.evaluation.lerf_source_mask_gate import _load_sam_records, _masks
from radio_gs.v4.evaluation.real_sam_token_association import _lift_masks


def _descriptor_records(paths: list[Path]) -> dict[int, Path]:
    records = {}
    for path in paths:
        for row in json.loads(path.read_text())["records"]:
            records[int(row["frame_id"])] = Path(row["output"]).resolve(strict=True)
    return records


def _pair_cosine(first: torch.Tensor, second: torch.Tensor, log_weight: torch.Tensor) -> torch.Tensor:
    weight = log_weight.exp().clamp(0.05, 20.0)
    first = F.normalize(first * weight, dim=-1, eps=1e-8)
    second = F.normalize(second * weight, dim=-1, eps=1e-8)
    return (first * second).sum(-1)


def _metrics(positive: torch.Tensor, negative: torch.Tensor) -> dict:
    values = torch.cat([positive, negative])
    labels = torch.cat([torch.ones_like(positive), torch.zeros_like(negative)])
    order = values.argsort()
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(len(order), dtype=torch.float32, device=order.device)
    positive_ranks = ranks[labels > 0.5]
    auc = (
        (positive_ranks.sum() - len(positive) * (len(positive) - 1) / 2)
        / max(len(positive) * len(negative), 1)
    )
    return {
        "positive_count": int(len(positive)),
        "negative_count": int(len(negative)),
        "positive_cosine_mean": float(positive.mean()),
        "negative_cosine_mean": float(negative.mean()),
        "pair_auc": float(auc),
    }


@torch.no_grad()
def _collect_pairs(
    descriptors: list[torch.Tensor],
    positive: list[torch.Tensor],
    roots: list[torch.Tensor],
    view_indices: list[int],
    *,
    positive_containment: float,
    positive_iou: float,
    negative_iou: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_first, positive_second, negative_first, negative_second = [], [], [], []
    for left_pos, left_view in enumerate(view_indices):
        left_ids = torch.where(roots[left_view])[0]
        left_evidence = positive[left_view][left_ids]
        left_mass = left_evidence.sum(-1)
        # Same-view mutually exclusive roots provide factual negatives.
        intersection = left_evidence @ left_evidence.T
        union = left_mass[:, None] + left_mass[None] - intersection
        iou = intersection / union.clamp_min(1e-8)
        rows, cols = torch.where(torch.triu(iou <= negative_iou, diagonal=1))
        if len(rows):
            negative_first.append(descriptors[left_view][left_ids[rows]])
            negative_second.append(descriptors[left_view][left_ids[cols]])
        for right_view in view_indices[left_pos + 1 :]:
            right_ids = torch.where(roots[right_view])[0]
            right_evidence = positive[right_view][right_ids]
            right_mass = right_evidence.sum(-1)
            intersection = left_evidence @ right_evidence.T
            union = left_mass[:, None] + right_mass[None] - intersection
            iou = intersection / union.clamp_min(1e-8)
            containment = intersection / torch.minimum(left_mass[:, None], right_mass[None]).clamp_min(1e-8)
            rows, cols = torch.where((iou >= positive_iou) & (containment >= positive_containment))
            if len(rows):
                positive_first.append(descriptors[left_view][left_ids[rows]])
                positive_second.append(descriptors[right_view][right_ids[cols]])
    if not positive_first or not negative_first:
        raise RuntimeError("source proposal pairs are insufficient for metric fitting")
    return (
        torch.cat(positive_first),
        torch.cat(positive_second),
        torch.cat(negative_first),
        torch.cat(negative_second),
    )


def run(args: argparse.Namespace) -> dict:
    transforms = Path(args.transforms).resolve(strict=True)
    mesh_path = Path(args.mesh).resolve(strict=True)
    selection_path = Path(args.selection_authority).resolve(strict=True)
    descriptor_manifests = [Path(value).resolve(strict=True) for value in args.descriptor_manifest]
    sam_manifests = [Path(value).resolve(strict=True) for value in args.sam_manifest]
    descriptor_records = _descriptor_records(descriptor_manifests)
    sam_records = _load_sam_records(sam_manifests)
    selection = json.loads(selection_path.read_text())
    frame_ids = list(map(int, selection["selections"][str(args.maximum_view_count)]["selected_frame_ids"]))[: args.view_count]
    cameras = _load_cameras(
        transforms, [{"frame_id": value} for value in frame_ids], args.feature_height, args.feature_width
    )
    vertices, triangles = _load_mesh(mesh_path)
    mesh = MeshCarrier(vertices, triangles)
    surface = SurfaceVoxelCarrier.from_points(
        vertices,
        args.voxel_size,
        normals=mesh.normals,
        maximum_splat_radius=args.maximum_splat_radius,
        surface_band_voxels=args.surface_band_voxels,
        maximum_contributors_per_pixel=args.maximum_contributors_per_pixel,
    )
    descriptors, positives, roots = [], [], []
    for frame_id, camera in zip(frame_ids, cameras):
        mask_path = Path(sam_records[frame_id]["output"]).resolve(strict=True)
        masks = _masks(mask_path, args.feature_height, args.feature_width)
        lifted, _ = _lift_masks(surface, camera, masks, torch.device("cpu"))
        mask_payload = torch.load(mask_path, map_location="cpu")
        descriptor_payload = torch.load(descriptor_records[frame_id], map_location="cpu")
        descriptor = torch.as_tensor(descriptor_payload["descriptor"]).float()
        if descriptor.shape[0] != masks.shape[0]:
            raise ValueError("descriptor and proposal counts differ")
        descriptors.append(descriptor)
        positives.append(lifted)
        roots.append(torch.as_tensor(mask_payload["parent_index"]) < 0)

    train_views = list(range(args.training_view_count))
    validation_views = list(range(args.training_view_count, args.view_count))
    train_pairs = _collect_pairs(
        descriptors, positives, roots, train_views,
        positive_containment=args.positive_containment,
        positive_iou=args.positive_iou,
        negative_iou=args.negative_iou,
    )
    validation_pairs = _collect_pairs(
        descriptors, positives, roots, validation_views,
        positive_containment=args.positive_containment,
        positive_iou=args.positive_iou,
        negative_iou=args.negative_iou,
    )
    device = torch.device(args.device)
    train_pairs = tuple(value.to(device) for value in train_pairs)
    validation_pairs = tuple(value.to(device) for value in validation_pairs)
    log_weight = torch.nn.Parameter(torch.zeros(train_pairs[0].shape[1], device=device))
    optimizer = torch.optim.Adam([log_weight], lr=args.learning_rate)
    for _ in range(args.step_count):
        positive_score = _pair_cosine(train_pairs[0], train_pairs[1], log_weight)
        negative_score = _pair_cosine(train_pairs[2], train_pairs[3], log_weight)
        loss = (1 - positive_score).mean() + F.relu(negative_score - args.negative_margin).mean()
        loss = loss + args.identity_regularization * log_weight.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_before = _metrics(
            _pair_cosine(train_pairs[0], train_pairs[1], torch.zeros_like(log_weight)),
            _pair_cosine(train_pairs[2], train_pairs[3], torch.zeros_like(log_weight)),
        )
        train_after = _metrics(
            _pair_cosine(train_pairs[0], train_pairs[1], log_weight),
            _pair_cosine(train_pairs[2], train_pairs[3], log_weight),
        )
        validation_before = _metrics(
            _pair_cosine(validation_pairs[0], validation_pairs[1], torch.zeros_like(log_weight)),
            _pair_cosine(validation_pairs[2], validation_pairs[3], torch.zeros_like(log_weight)),
        )
        validation_after = _metrics(
            _pair_cosine(validation_pairs[0], validation_pairs[1], log_weight),
            _pair_cosine(validation_pairs[2], validation_pairs[3], log_weight),
        )
    checkpoint = Path(args.output_checkpoint).resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "radio_gs.surface_object_memory_v4.source_association_metric.v1",
        "channel_weight": log_weight.detach().cpu().exp(),
        "query_or_label_used": False,
        "training_frame_ids": frame_ids[: args.training_view_count],
        "validation_frame_ids": frame_ids[args.training_view_count :],
    }, checkpoint)
    receipt = GeometryReceipt(
        carrier="frozen_sparse_surface_source_pair_metric_fit",
        coordinate_convention="surface_overlap_pseudo_pairs",
        inputs=(
            HashedInput.seal("label_free_view_selection", selection_path),
            HashedInput.seal("camera_transforms", transforms),
            HashedInput.seal("mesh_geometry", mesh_path),
            *tuple(HashedInput.seal(f"sam_manifest_{i}", path) for i, path in enumerate(sam_manifests)),
            *tuple(HashedInput.seal(f"descriptor_manifest_{i}", path) for i, path in enumerate(descriptor_manifests)),
        ),
        source_rgb_opened=False,
        target_rgb_opened=False,
        benchmark_images_opened=False,
        benchmark_masks_opened=False,
        benchmark_labels_opened=False,
        metadata={"query_independent": True, "oracle_labels_used": False, "validation_views_held_out": True},
    )
    report = {
        "schema": "radio_gs.surface_object_memory_v4.source_association_metric_fit.v1",
        "training": {"before": train_before, "after": train_after},
        "heldout_source_validation": {"before": validation_before, "after": validation_after},
        "policy": {
            "positive_containment": args.positive_containment,
            "positive_iou": args.positive_iou,
            "negative_iou": args.negative_iou,
            "training_view_count": args.training_view_count,
            "validation_view_count": args.view_count - args.training_view_count,
        },
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "geometry_receipt": receipt.to_dict(),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-authority", required=True)
    parser.add_argument("--sam-manifest", action="append", required=True)
    parser.add_argument("--descriptor-manifest", action="append", required=True)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--maximum-view-count", type=int, default=64)
    parser.add_argument("--view-count", type=int, default=16)
    parser.add_argument("--training-view-count", type=int, default=12)
    parser.add_argument("--feature-height", type=int, default=60)
    parser.add_argument("--feature-width", type=int, default=81)
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--maximum-splat-radius", type=int, default=1)
    parser.add_argument("--surface-band-voxels", type=float, default=1.5)
    parser.add_argument("--maximum-contributors-per-pixel", type=int, default=8)
    parser.add_argument("--positive-containment", type=float, default=0.50)
    parser.add_argument("--positive-iou", type=float, default=0.15)
    parser.add_argument("--negative-iou", type=float, default=0.01)
    parser.add_argument("--negative-margin", type=float, default=0.25)
    parser.add_argument("--identity-regularization", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--step-count", type=int, default=500)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 < args.training_view_count < args.view_count:
        raise ValueError("training views must leave a non-empty held-out source cohort")
    report = run(args)
    print(json.dumps({"training": report["training"], "heldout": report["heldout_source_validation"]}, indent=2))


if __name__ == "__main__":
    main()
