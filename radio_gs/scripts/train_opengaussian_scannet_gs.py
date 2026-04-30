#!/usr/bin/env python3
"""Train RGB 3DGS geometry for prepared OpenGaussian ScanNet scenes.

Expected input layout is produced by ``prepare_opengaussian_scannet_scene.py``:

    dataset/scannet_og/{scene}/
        color/
        transforms_train.json
        transforms_test.json
        transforms.json
        points3d.ply

The final geometry is written as a standard 3DGS PLY:

    output/3dgs_models/scannet_og/{scene}/{tag}/point_cloud/iteration_{iters}/point_cloud.ply
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from radio_gs.data.benchmark_paths import extract_feature_frame_index
from radio_gs.data.lerf_dataset import _parse_transforms_json
from radio_gs.scripts.train_colmap_gs import compute_scene_scale, save_ply
from radio_gs.scripts.train_rgb_gs import SimpleGaussianModel, l1_loss, ssim_loss


C0 = 0.28209479177387814


def _resolve_image_path(scene_root: Path, file_path: str) -> Path:
    raw = Path(file_path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(scene_root / raw)
    if raw.suffix:
        candidates.append(scene_root / "color" / raw.name)
    else:
        for suffix in (".jpg", ".png", ".jpeg"):
            candidates.append(scene_root / f"{file_path}{suffix}")
            candidates.append(scene_root / "color" / f"{raw.name}{suffix}")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image for transforms frame not found: {file_path}")


def _load_points3d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"OpenGaussian points3d.ply not found: {path}")

    from plyfile import PlyData

    plydata = PlyData.read(str(path))
    vertex = plydata.elements[0]
    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )
    names = set(vertex.data.dtype.names or ())
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.stack(
            [
                np.asarray(vertex["red"], dtype=np.float32),
                np.asarray(vertex["green"], dtype=np.float32),
                np.asarray(vertex["blue"], dtype=np.float32),
            ],
            axis=1,
        ) / 255.0
    else:
        rgb = np.full_like(xyz, 0.5, dtype=np.float32)
    return xyz, rgb


def _estimate_log_scales(points: torch.Tensor, scene_scale: float, max_sample: int = 1024) -> torch.Tensor:
    n_points = points.shape[0]
    sample_count = min(n_points, max_sample)
    sample = points.detach().cpu()[torch.randperm(n_points)[:sample_count]]
    chunk = 1024
    nn_dists = []
    for start in range(0, sample_count, chunk):
        end = min(start + chunk, sample_count)
        dists = torch.cdist(sample[start:end], sample)
        rows = torch.arange(end - start)
        dists[rows, start + rows] = float("inf")
        nn_dists.append(dists.min(dim=1).values)
    median_nn = torch.cat(nn_dists).median().item()
    init_scale = max(median_nn * 0.5, scene_scale * 1e-4)
    print(f"  Median NN distance: {median_nn:.4f}, init scale: {init_scale:.4f}", flush=True)
    return torch.full((n_points, 3), math.log(init_scale), dtype=torch.float32)


def _init_gaussians_from_points(
    points_ply: Path,
    max_points: int | None = None,
    scale_sample_points: int = 1024,
) -> dict[str, torch.Tensor]:
    xyz, rgb = _load_points3d(points_ply)
    if max_points is not None and xyz.shape[0] > max_points:
        rng = np.random.default_rng(42)
        keep = rng.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[keep]
        rgb = rgb[keep]

    points = torch.from_numpy(xyz.astype(np.float32))
    colors = torch.from_numpy(rgb.astype(np.float32))
    scene_scale = compute_scene_scale(xyz)
    print(
        f"Initialized {points.shape[0]:,} Gaussians from {points_ply} "
        f"(scene scale={scene_scale:.2f})",
        flush=True,
    )

    return {
        "means": points,
        "sh_dc": (colors - 0.5) / C0,
        "log_scales": _estimate_log_scales(points, scene_scale, max_sample=scale_sample_points),
        "quats": torch.nn.functional.pad(torch.ones(points.shape[0], 1), (0, 3)),
        "logit_opacity": torch.full((points.shape[0], 1), math.log(0.1 / 0.9), dtype=torch.float32),
    }


def _load_scene(scene_root: Path, split: str, frame_stride: int, max_frames: int | None):
    transforms_path = scene_root / f"transforms_{split}.json"
    if not transforms_path.exists():
        transforms_path = scene_root / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"No transforms file found for split {split} in {scene_root}")

    parsed = _parse_transforms_json(str(transforms_path))
    w = int(parsed.get("w", 0))
    h = int(parsed.get("h", 0))
    fx = float(parsed.get("fl_x", 0.0))
    fy = float(parsed.get("fl_y", fx))
    cx = float(parsed.get("cx", w / 2.0 - 0.5))
    cy = float(parsed.get("cy", h / 2.0 - 0.5))
    if not all([w, h, fx, fy]):
        raise ValueError(f"Transforms file lacks camera intrinsics: {transforms_path}")

    frame_records = []
    for dense_idx, (c2w, file_path) in enumerate(zip(parsed["c2w_list"], parsed["file_paths"])):
        try:
            frame_id = extract_feature_frame_index(Path(file_path))
        except ValueError:
            frame_id = dense_idx
        frame_records.append((frame_id, c2w, file_path))
    frame_records.sort(key=lambda item: item[0])
    frame_records = frame_records[:: max(1, frame_stride)]
    if max_frames is not None:
        frame_records = frame_records[:max_frames]

    image_paths = []
    w2cs = []
    frame_ids = []
    for frame_id, c2w, file_path in frame_records:
        image_path = _resolve_image_path(scene_root, file_path)
        image_paths.append(image_path)
        w2cs.append(torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)))
        frame_ids.append(int(frame_id))

    if not image_paths:
        raise RuntimeError(f"No RGB frames loaded from {transforms_path}")
    return image_paths, w2cs, frame_ids, fx, fy, cx, cy, w, h


def _load_rgb_tensor(path: Path, width: int, height: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)


def train(args) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    scene_root = Path(args.scene_root).resolve()
    scene = args.scene or scene_root.name
    image_paths, w2cs_cpu, frame_ids, fx, fy, cx, cy, width, height = _load_scene(
        scene_root,
        split=args.split,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    k = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    w2cs = [w2c.to(device) for w2c in w2cs_cpu]

    init_data = _init_gaussians_from_points(
        scene_root / "points3d.ply",
        max_points=args.max_points,
        scale_sample_points=args.scale_sample_points,
    )
    for key, value in init_data.items():
        init_data[key] = value.to(device)

    model = SimpleGaussianModel(init_data, sh_degree=args.sh_degree).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": [model.means], "lr": args.lr_means},
            {"params": [model.sh_coeffs], "lr": args.lr_sh},
            {"params": [model.log_scales], "lr": args.lr_scale},
            {"params": [model.quats], "lr": args.lr_quat},
            {"params": [model.logit_opacity], "lr": args.lr_opacity},
        ]
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.01 ** (1.0 / max(args.iters, 1)))

    out_dir = Path(args.output_dir) / scene / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = out_dir / "point_cloud" / f"iteration_{args.iters}"
    ply_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "scene": scene,
        "scene_root": str(scene_root),
        "split": args.split,
        "frame_ids": frame_ids,
        "image_paths": [str(path) for path in image_paths],
        "image_height": height,
        "image_width": width,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "iters": args.iters,
        "tag": args.tag,
    }
    (out_dir / "train_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    bg = torch.zeros(3, device=device)
    best_psnr = 0.0
    pbar = tqdm(range(args.iters), desc=f"Training OpenGaussian ScanNet GS {scene}")
    for step in pbar:
        idx = random.randint(0, len(image_paths) - 1)
        gt_img = _load_rgb_tensor(image_paths[idx], width, height).to(device, non_blocking=True)
        pred_rgb = model.render(w2cs[idx], k, width, height, bg)["rgb"].clamp(0.0, 1.0)
        loss = (1.0 - args.lambda_ssim) * l1_loss(pred_rgb, gt_img) + args.lambda_ssim * ssim_loss(pred_rgb, gt_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step == 0 or (step + 1) % args.log_every == 0:
            with torch.no_grad():
                mse = ((pred_rgb - gt_img) ** 2).mean()
                psnr = -10.0 * math.log10(mse.item() + 1e-10)
            pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.1f}")

        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            eval_ids = list(range(0, len(image_paths), max(1, len(image_paths) // 10)))
            psnr_sum = 0.0
            with torch.no_grad():
                for eval_idx in eval_ids:
                    rgb = model.render(w2cs[eval_idx], k, width, height, bg)["rgb"].clamp(0.0, 1.0)
                    gt_eval = _load_rgb_tensor(image_paths[eval_idx], width, height).to(
                        device,
                        non_blocking=True,
                    )
                    mse = ((rgb - gt_eval) ** 2).mean()
                    psnr_sum += -10.0 * math.log10(mse.item() + 1e-10)
            avg_psnr = psnr_sum / len(eval_ids)
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save(model.state_dict(), str(out_dir / "best.pth"))

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            mid_dir = out_dir / "point_cloud" / f"iteration_{step + 1}"
            mid_dir.mkdir(parents=True, exist_ok=True)
            _save_model_ply(model, mid_dir / "point_cloud.ply")

    torch.save(model.state_dict(), str(out_dir / "final.pth"))
    _save_model_ply(model, ply_dir / "point_cloud.ply")
    print(f"Training complete. Best PSNR: {best_psnr:.2f} dB")
    print(f"Output saved to {ply_dir / 'point_cloud.ply'}")


def _save_model_ply(model: SimpleGaussianModel, path: Path) -> None:
    export_state = {
        "means": model.means,
        "scales": model.log_scales,
        "quats": model.quats,
        "opacities": model.logit_opacity,
        "sh0": model.sh_coeffs[:, :1, :],
        "shN": model.sh_coeffs[:, 1:, :],
    }
    save_ply(str(path), export_state, model.sh_degree)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--output_dir", default="output/3dgs_models/scannet_og")
    parser.add_argument("--tag", default="og_rgb_3dgs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--iters", type=int, default=30000)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--max_points", type=int, default=None)
    parser.add_argument(
        "--scale_sample_points",
        type=int,
        default=1024,
        help="Number of points sampled for initial NN scale estimation",
    )
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--lambda_ssim", type=float, default=0.2)
    parser.add_argument("--lr_means", type=float, default=1.6e-4)
    parser.add_argument("--lr_sh", type=float, default=2.5e-3)
    parser.add_argument("--lr_scale", type=float, default=5e-3)
    parser.add_argument("--lr_quat", type=float, default=1e-3)
    parser.add_argument("--lr_opacity", type=float, default=5e-2)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--save_every", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
