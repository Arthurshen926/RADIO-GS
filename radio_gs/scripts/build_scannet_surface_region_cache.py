#!/usr/bin/env python3
"""Build query-free 3-D surface-region/official-summary pairs from ScanNet.

Only ``color``, ``depth``, ``pose`` and the two intrinsic files are opened.
Semantic/instance labels are deliberately forbidden from the input contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor

from radio_gs.interfaces.frozen_radio_views import OfficialCropSummaryRuntime
from radio_gs.interfaces.surface_region_summary import surface_region_geometry


FORBIDDEN_EVAL_SCENES = {"scene0062_00", "scene0140_00", "scene0200_00"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scene_names(split_file: Path, root: Path) -> list[str]:
    names = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    names = [name for name in names if name not in FORBIDDEN_EVAL_SCENES]
    return [name for name in names if (root / name).is_dir()]


def _frame_paths(scene_dir: Path) -> list[tuple[Path, Path, Path]]:
    triples = []
    for color in sorted((scene_dir / "color").glob("*.jpg")):
        depth = scene_dir / "depth" / f"{color.stem}.png"
        pose = scene_dir / "pose" / f"{color.stem}.txt"
        if depth.is_file() and pose.is_file():
            matrix = np.loadtxt(pose)
            if matrix.shape == (4, 4) and np.isfinite(matrix).all():
                triples.append((color, depth, pose))
    return triples


def _select_evenly(values: list, count: int) -> list:
    if len(values) <= int(count):
        return values
    indices = np.linspace(0, len(values) - 1, int(count)).round().astype(int)
    return [values[int(index)] for index in indices]


def _load_color(path: Path) -> torch.Tensor:
    return pil_to_tensor(Image.open(path).convert("RGB")).float().div_(255.0)


def _load_depth(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.asarray(Image.open(path), dtype=np.float32).copy()).div_(1000.0)


def _lift_observation(
    depth: torch.Tensor,
    depth_intrinsic: torch.Tensor,
    color_intrinsic: torch.Tensor,
    camera_to_world: torch.Tensor,
    spatial: torch.Tensor,
    *,
    stride: int,
    color_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lift one depth map and bilinearly sample the matching RADIO map."""

    height, width = depth.shape
    ys = torch.arange(0, height, int(stride), dtype=torch.long)
    xs = torch.arange(0, width, int(stride), dtype=torch.long)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    z = depth[yy, xx]
    valid = torch.isfinite(z) & (z > 0.20) & (z < 8.0)
    x = (xx.float() - depth_intrinsic[0, 2]) * z / depth_intrinsic[0, 0]
    y = (yy.float() - depth_intrinsic[1, 2]) * z / depth_intrinsic[1, 1]
    camera = torch.stack([x, y, z, torch.ones_like(z)], dim=-1)[valid]
    world = (camera @ camera_to_world.T)[:, :3]

    # Color and depth share the exported pose but have different intrinsics.
    u = color_intrinsic[0, 0] * x / z.clamp_min(1e-6) + color_intrinsic[0, 2]
    v = color_intrinsic[1, 1] * y / z.clamp_min(1e-6) + color_intrinsic[1, 2]
    color_width, color_height = (float(value) for value in color_size)
    grid = torch.stack(
        [2.0 * u / color_width - 1.0, 2.0 * v / color_height - 1.0], dim=-1
    )[valid]
    features = F.grid_sample(
        spatial[None].float(), grid[None, None], mode="bilinear",
        padding_mode="border", align_corners=False,
    )[0, :, 0].T
    footprint = (z[valid] / depth_intrinsic[0, 0] * float(stride)).clamp_min(1e-3)
    return world, features, footprint


def _voxel_fuse(
    xyz: torch.Tensor, features: torch.Tensor, footprint: torch.Tensor, voxel_size: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    keys = torch.floor(xyz / float(voxel_size)).to(torch.int64)
    unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
    count = torch.bincount(inverse, minlength=unique.shape[0]).float()
    fused_xyz = torch.zeros(unique.shape[0], 3).index_add_(0, inverse, xyz) / count[:, None]
    fused_features = torch.zeros(unique.shape[0], features.shape[1]).index_add_(
        0, inverse, features
    ) / count[:, None]
    fused_footprint = torch.zeros(unique.shape[0]).index_add_(0, inverse, footprint) / count
    return fused_xyz, fused_features, fused_footprint, count


def _surface_radius_graph(xyz: torch.Tensor, voxel_size: float) -> csr_matrix:
    points = xyz.numpy()
    pairs = cKDTree(points).query_pairs(r=float(voxel_size) * 1.8, output_type="ndarray")
    if pairs.size == 0:
        raise RuntimeError("surface radius graph contains no edges")
    distances = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    values = np.concatenate([distances, distances]).astype(np.float32)
    return csr_matrix((values, (rows, cols)), shape=(len(points), len(points)))


def _region_indices(
    graph: csr_matrix, seed: int, radius: float, *, min_tokens: int, max_tokens: int,
    rng: random.Random,
) -> np.ndarray | None:
    distance = dijkstra(graph, directed=False, indices=int(seed), limit=float(radius))
    rows = np.flatnonzero(np.isfinite(distance))
    if rows.size < int(min_tokens):
        return None
    if rows.size > int(max_tokens):
        rows = np.asarray(rng.sample(rows.tolist(), int(max_tokens)), dtype=np.int64)
    return rows


def _project_region_box(
    xyz: torch.Tensor, depth: torch.Tensor, depth_intrinsic: torch.Tensor,
    color_intrinsic: torch.Tensor, camera_to_world: torch.Tensor,
    color_size: tuple[int, int], *, min_visible: int,
) -> list[int] | None:
    world_to_camera = torch.linalg.inv(camera_to_world)
    camera = torch.cat([xyz, torch.ones(len(xyz), 1)], dim=1) @ world_to_camera.T
    z = camera[:, 2]
    valid = z > 0.15
    if not bool(valid.any()):
        return None
    camera = camera[valid]
    z = z[valid]
    ud = depth_intrinsic[0, 0] * camera[:, 0] / z + depth_intrinsic[0, 2]
    vd = depth_intrinsic[1, 1] * camera[:, 1] / z + depth_intrinsic[1, 2]
    ix, iy = ud.round().long(), vd.round().long()
    inside = (ix >= 0) & (iy >= 0) & (ix < depth.shape[1]) & (iy < depth.shape[0])
    visible = torch.zeros_like(inside)
    if bool(inside.any()):
        observed = depth[iy[inside], ix[inside]]
        visible[inside] = (observed > 0) & ((observed - z[inside]).abs() < 0.10)
    if int(visible.sum()) < int(min_visible):
        return None
    camera, z = camera[visible], z[visible]
    u = color_intrinsic[0, 0] * camera[:, 0] / z + color_intrinsic[0, 2]
    v = color_intrinsic[1, 1] * camera[:, 1] / z + color_intrinsic[1, 2]
    width, height = color_size
    x0, x1 = float(u.min()), float(u.max())
    y0, y1 = float(v.min()), float(v.max())
    pad = 0.12 * max(x1 - x0, y1 - y0, 16.0)
    left, right = max(0, int(math.floor(x0 - pad))), min(width, int(math.ceil(x1 + pad)))
    top, bottom = max(0, int(math.floor(y0 - pad))), min(height, int(math.ceil(y1 + pad)))
    if right - left < 24 or bottom - top < 24:
        return None
    return [top, left, bottom, right]


def _teacher_medoid(tokens: torch.Tensor, descriptors: torch.Tensor | None = None) -> int:
    """Select the view nearest the multiview centre in official semantic space."""
    values = tokens if descriptors is None else descriptors
    normalized = F.normalize(values.float(), dim=-1, eps=1e-8)
    return int((normalized @ normalized.T).sum(dim=1).argmax())


@torch.no_grad()
def build(args: argparse.Namespace) -> dict:
    root, split_file = Path(args.dataset_root), Path(args.split_file)
    split_role = str(args.split_role)
    if split_role not in {"train", "validation"}:
        raise ValueError("split-role must be train or validation")
    scenes = _scene_names(split_file, root)
    scenes = scenes[int(args.shard_index)::int(args.shard_count)][:int(args.max_scenes)]
    if not scenes:
        raise RuntimeError("no ScanNet scenes selected")
    if FORBIDDEN_EVAL_SCENES.intersection(scenes):
        raise RuntimeError("paper evaluation scenes leaked into bridge data")
    runtime = OfficialCropSummaryRuntime.load(
        checkpoint_path=args.radio_checkpoint, radio_repo=args.radio_repo,
        version=args.radio_version, device=args.device,
    )
    device = torch.device(args.device)
    rng = random.Random(int(args.seed) + int(args.shard_index) * 100003)
    records, feature_rows, geometry_rows, masks, reliability_rows = [], [], [], [], []
    teacher_tokens, teacher_descriptors, teacher_masks = [], [], []
    failures = {}
    radii = tuple(float(v) for v in str(args.region_radii).replace(",", " ").split())
    for scene_name in scenes:
        scene_dir = root / scene_name
        try:
            frames = _select_evenly(_frame_paths(scene_dir), int(args.frames_per_scene))
            if len(frames) < 2:
                raise RuntimeError("fewer than two valid RGB-D-pose frames")
            kd = torch.from_numpy(np.loadtxt(scene_dir / "intrinsics_depth.txt")).float()
            kc = torch.from_numpy(np.loadtxt(scene_dir / "intrinsics_color.txt")).float()
            lifted_xyz, lifted_features, lifted_footprint = [], [], []
            frame_data = []
            for color_path, depth_path, pose_path in frames:
                color, depth = _load_color(color_path), _load_depth(depth_path)
                pose = torch.from_numpy(np.loadtxt(pose_path)).float()
                input_image = F.interpolate(
                    color[None], (int(args.radio_resolution), int(args.radio_resolution)),
                    mode="bilinear", align_corners=False,
                ).to(device)
                spatial = runtime.encode_training_pair(input_image)[0][0].cpu()
                xyz, feat, footprint = _lift_observation(
                    depth, kd, kc, pose, spatial, stride=int(args.depth_stride),
                    color_size=(int(color.shape[2]), int(color.shape[1])),
                )
                lifted_xyz.append(xyz); lifted_features.append(feat); lifted_footprint.append(footprint)
                frame_data.append((color_path, color, depth, pose))
            xyz, features, footprint, counts = _voxel_fuse(
                torch.cat(lifted_xyz), torch.cat(lifted_features),
                torch.cat(lifted_footprint), float(args.voxel_size),
            )
            graph = _surface_radius_graph(xyz, float(args.voxel_size))
            candidates = list(range(len(xyz))); rng.shuffle(candidates)
            scene_regions = 0
            for seed in candidates:
                if scene_regions >= int(args.regions_per_scene):
                    break
                radius = radii[scene_regions % len(radii)]
                indices = _region_indices(
                    graph, seed, radius, min_tokens=int(args.min_tokens),
                    max_tokens=int(args.max_tokens), rng=rng,
                )
                if indices is None:
                    continue
                idx = torch.from_numpy(indices)
                crops, views = [], []
                for color_path, color, depth, pose in frame_data:
                    box = _project_region_box(
                        xyz[idx], depth, kd, kc, pose,
                        (int(color.shape[2]), int(color.shape[1])),
                        min_visible=int(args.min_visible_tokens),
                    )
                    if box is None:
                        continue
                    top, left, bottom, right = box
                    crops.append(F.interpolate(
                        color[:, top:bottom, left:right][None],
                        (int(args.radio_resolution), int(args.radio_resolution)),
                        mode="bilinear", align_corners=False,
                    )[0])
                    views.append({"frame": color_path.name, "crop_box_tlbr": box})
                    if len(crops) >= int(args.teacher_views):
                        break
                if len(crops) < 2:
                    continue
                _, tokens, descriptors = runtime.encode_training_pair(torch.stack(crops).to(device))
                view_count = len(crops)
                padded_features = torch.zeros(int(args.max_tokens), 1280, dtype=torch.float16)
                padded_geometry = torch.zeros(int(args.max_tokens), 12, dtype=torch.float16)
                padded_mask = torch.zeros(int(args.max_tokens), dtype=torch.bool)
                padded_reliability = torch.zeros(int(args.max_tokens), 1, dtype=torch.float16)
                n = len(idx)
                rel = (1.0 - torch.exp(-counts[idx, None] / 2.0)).clamp(0, 1)
                scale = footprint[idx, None].expand(-1, 3).clamp(
                    min=float(args.voxel_size) * 0.5, max=float(args.voxel_size) * 2.0
                )
                geom = surface_region_geometry(
                    xyz[idx], scale, torch.ones(n, 1), rel, float(radius)
                )
                padded_features[:n] = features[idx].half()
                padded_geometry[:n] = geom.half()
                padded_mask[:n] = True; padded_reliability[:n] = rel.half()
                max_views = int(args.teacher_views)
                padded_tokens = torch.zeros(max_views, 1280, dtype=torch.float16)
                padded_descriptors = torch.zeros(max_views, descriptors.shape[1], dtype=torch.float16)
                padded_teacher_mask = torch.zeros(max_views, dtype=torch.bool)
                padded_tokens[:view_count] = tokens.half().cpu()
                padded_descriptors[:view_count] = descriptors.half().cpu()
                padded_teacher_mask[:view_count] = True
                feature_rows.append(padded_features); geometry_rows.append(padded_geometry)
                masks.append(padded_mask); reliability_rows.append(padded_reliability)
                teacher_tokens.append(padded_tokens); teacher_descriptors.append(padded_descriptors)
                teacher_masks.append(padded_teacher_mask)
                records.append({
                    "scene": scene_name, "seed": int(seed), "physical_radius_m": radius,
                    "tokens": n, "teacher_views": views,
                    "teacher_medoid": _teacher_medoid(tokens, descriptors),
                })
                scene_regions += 1
            if scene_regions == 0:
                raise RuntimeError("no multi-view surface region survived visibility checks")
        except Exception as error:
            failures[scene_name] = f"{type(error).__name__}: {error}"
    if not records:
        raise RuntimeError(f"all scenes failed: {failures}")
    metadata = {
        "schema_version": 2, "training_scope": "global_cross_scene_3d_surface",
        "dataset_id": "ScanNet_frames_25k_query_free", "split_role": split_role,
        "split_file": str(split_file.resolve()), "split_file_sha256": _sha256(split_file),
        "uses_benchmark_test_vocabulary": False, "uses_benchmark_scenes": False,
        "annotations_opened": False, "labels_opened": False, "instances_opened": False,
        "text_opened": False, "region_construction": "depth_pose_surface_radius_graph",
        "radio_version": runtime.version,
        "radio_checkpoint_sha256": runtime.radio_checkpoint_sha256,
        "scene_names": sorted({record["scene"] for record in records}),
        "region_records": records, "failed_scenes": failures,
        "forbidden_eval_scenes": sorted(FORBIDDEN_EVAL_SCENES),
    }
    payload = {
        "radio_features": torch.stack(feature_rows),
        "geometry": torch.stack(geometry_rows), "token_mask": torch.stack(masks),
        "reliability": torch.stack(reliability_rows),
        "official_summary_tokens": torch.stack(teacher_tokens),
        "official_crop_summaries": torch.stack(teacher_descriptors),
        "teacher_mask": torch.stack(teacher_masks), "metadata": metadata,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = {
        "output": str(output.resolve()), "regions": len(records),
        "scenes": len(metadata["scene_names"]), "failed_scenes": failures,
        "split_role": split_role, "split_file_sha256": metadata["split_file_sha256"],
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split-role", choices=("train", "validation"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-scenes", type=int, default=16)
    parser.add_argument("--frames-per-scene", type=int, default=8)
    parser.add_argument("--regions-per-scene", type=int, default=12)
    parser.add_argument("--region-radii", default="0.25,0.45,0.70")
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--depth-stride", type=int, default=8)
    parser.add_argument("--min-tokens", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--min-visible-tokens", type=int, default=12)
    parser.add_argument("--teacher-views", type=int, default=3)
    parser.add_argument("--radio-resolution", type=int, default=384)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--radio-repo", default="/root/RADIO")
    parser.add_argument("--radio-version", default="c-radio_v4-h")
    parser.add_argument("--radio-checkpoint", default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
