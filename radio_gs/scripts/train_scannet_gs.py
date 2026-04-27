#!/usr/bin/env python3
"""Train a minimal RGB 3DGS model on a prepared ScanNet RGB-D scene."""

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
from radio_gs.data.scannet_dataset import _load_scannet_pose
from radio_gs.scripts.train_colmap_gs import save_ply
from radio_gs.scripts.train_rgb_gs import (
    SimpleGaussianModel,
    init_gaussians_from_depth,
    l1_loss,
    render_all_frames,
    ssim_loss,
)


def load_intrinsics(scene_root: Path) -> tuple[float, float, float, float]:
    intrinsic_path = scene_root / "intrinsic" / "intrinsic_depth.txt"
    if not intrinsic_path.exists():
        raise FileNotFoundError(f"Missing ScanNet intrinsic file: {intrinsic_path}")
    intrinsic = np.loadtxt(str(intrinsic_path), dtype=np.float32).reshape(4, 4)
    return float(intrinsic[0, 0]), float(intrinsic[1, 1]), float(intrinsic[0, 2]), float(intrinsic[1, 2])


def list_scannet_frame_ids(scene_root: Path) -> list[int]:
    pose_dir = scene_root / "pose"
    frame_ids = []
    for pose_path in pose_dir.glob("*.txt"):
        try:
            frame_ids.append(extract_feature_frame_index(pose_path))
        except ValueError:
            continue
    return sorted(set(frame_ids))


def load_scannet_data(scene_root: Path, frame_stride: int, max_frames: int | None):
    color_dir = scene_root / "color"
    depth_dir = scene_root / "depth"
    pose_dir = scene_root / "pose"

    frame_ids_all = list_scannet_frame_ids(scene_root)
    if not frame_ids_all:
        raise RuntimeError(f"No ScanNet pose frames found under {pose_dir}")

    selected = frame_ids_all[:: max(1, frame_stride)]
    if max_frames is not None:
        selected = selected[:max_frames]

    images = []
    depths = []
    c2ws = []
    kept_frame_ids = []

    for frame_idx in selected:
        color_path = color_dir / f"{frame_idx}.jpg"
        if not color_path.exists():
            color_path = color_dir / f"{frame_idx}.png"
        depth_path = depth_dir / f"{frame_idx}.png"
        pose_path = pose_dir / f"{frame_idx}.txt"
        if not color_path.exists() or not depth_path.exists() or not pose_path.exists():
            continue

        c2w = _load_scannet_pose(str(pose_path))
        if c2w is None:
            continue

        color = Image.open(color_path).convert("RGB")
        depth = np.array(Image.open(depth_path), dtype=np.uint16)
        if color.size != (depth.shape[1], depth.shape[0]):
            color = color.resize((depth.shape[1], depth.shape[0]), Image.Resampling.BILINEAR)

        images.append(torch.from_numpy(np.array(color, dtype=np.float32) / 255.0))
        depths.append(torch.from_numpy(depth.astype(np.float32) / 1000.0))
        c2ws.append(torch.from_numpy(c2w.astype(np.float32)))
        kept_frame_ids.append(frame_idx)

    if not images:
        raise RuntimeError(f"No valid ScanNet RGB-D frames found in {scene_root}")

    return images, depths, c2ws, kept_frame_ids


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    scene_root = Path(args.scene_root).resolve()
    scene = args.scene or scene_root.name
    fx, fy, cx, cy = load_intrinsics(scene_root)

    images, depths, c2ws, frame_ids = load_scannet_data(
        scene_root,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    n_frames = len(images)
    H, W = depths[0].shape
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device)
    w2cs = [torch.inverse(c2w.to(device)) for c2w in c2ws]

    init_data = init_gaussians_from_depth(
        images,
        depths,
        c2ws,
        fx,
        fy,
        cx,
        cy,
        n_init_frames=args.init_frames,
        stride=args.init_stride,
        max_points=args.max_points,
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
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.01 ** (1.0 / args.iters))

    out_dir = Path(args.output_dir) / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = out_dir / "point_cloud" / f"iteration_{args.iters}"
    ply_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "scene_root": str(scene_root),
        "scene": scene,
        "frame_ids": frame_ids,
        "image_height": int(H),
        "image_width": int(W),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "iters": args.iters,
    }
    (out_dir / "train_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    bg = torch.zeros(3, device=device)
    best_psnr = 0.0
    pbar = tqdm(range(args.iters), desc="Training ScanNet RGB GS")
    for step in pbar:
        idx = random.randint(0, n_frames - 1)
        gt_img = images[idx].to(device, non_blocking=True)
        result = model.render(w2cs[idx], K, W, H, bg)
        pred_rgb = result["rgb"].clamp(0, 1)

        loss = 0.8 * l1_loss(pred_rgb, gt_img) + 0.2 * ssim_loss(pred_rgb, gt_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 500 == 0 or step == 0:
            with torch.no_grad():
                mse = ((pred_rgb - gt_img) ** 2).mean()
                psnr = -10 * math.log10(mse.item() + 1e-10)
            pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.1f}")

        if (step + 1) % 2000 == 0:
            eval_ids = list(range(0, n_frames, max(1, n_frames // 10)))
            psnr_sum = 0.0
            with torch.no_grad():
                for eval_idx in eval_ids:
                    res = model.render(w2cs[eval_idx], K, W, H, bg)
                    gt_eval = images[eval_idx].to(device, non_blocking=True)
                    mse = ((res["rgb"].clamp(0, 1) - gt_eval) ** 2).mean()
                    psnr_sum += -10 * math.log10(mse.item() + 1e-10)
            avg_psnr = psnr_sum / len(eval_ids)
            print(f"\n  [Iter {step + 1}] Eval PSNR: {avg_psnr:.2f} dB ({len(eval_ids)} frames)")
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save(model.state_dict(), str(out_dir / "best.pth"))
                print(f"  -> New best: {best_psnr:.2f} dB")

    torch.save(model.state_dict(), str(out_dir / "final.pth"))
    export_state = {
        "means": model.means,
        "scales": model.log_scales,
        "quats": model.quats,
        "opacities": model.logit_opacity,
        "sh0": model.sh_coeffs[:, :1, :],
        "shN": model.sh_coeffs[:, 1:, :],
    }
    save_ply(str(ply_dir / "point_cloud.ply"), export_state, model.sh_degree)
    print(f"\nTraining complete. Best PSNR: {best_psnr:.2f} dB")
    print(f"Model saved to {out_dir}")

    if args.render_all:
        render_all_frames(model, w2cs, K, W, H, bg, out_dir, n_frames)


def main():
    parser = argparse.ArgumentParser(description="Train 3DGS RGB model on a prepared ScanNet scene")
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--output_dir", default="output/3dgs_models/scannet")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=30000)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--init_frames", type=int, default=50)
    parser.add_argument("--init_stride", type=int, default=8)
    parser.add_argument("--max_points", type=int, default=200000)
    parser.add_argument("--lr_means", type=float, default=1.6e-4)
    parser.add_argument("--lr_sh", type=float, default=2.5e-3)
    parser.add_argument("--lr_scale", type=float, default=5e-3)
    parser.add_argument("--lr_quat", type=float, default=1e-3)
    parser.add_argument("--lr_opacity", type=float, default=5e-2)
    parser.add_argument("--render_all", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
