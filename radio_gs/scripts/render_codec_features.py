"""Render decoded 1280d features from a trained model for a chosen split.

This script loads a trained RADIO-GS model and renders decoded features
for the requested train / val views.  The output is a directory of ``rgb_N.pt`` files
in the same format as the ground-truth RADIO features, so that
``pretrain_oracle_head.py`` (or any other consumer) can use them directly
to train a **domain-matched** depth head.

Motivation
----------
The "oracle" depth head is trained on GT RADIO features, but during GS
training it receives *rendered* features which have a distribution shift
(codec quantisation, splatting noise, etc.).  Training the depth head on
rendered features closes this gap.

Usage
-----
    python radio_gs/scripts/render_codec_features.py \
        --config radio_gs/configs/replica_hybrid_v14_room_0_nofdh_240ep.yaml \
        --checkpoint output/radio_gs/room0_hybrid_v14_nofdh_240ep/checkpoints/best.pth \
        --output_dir output/radio_gs/rendered_features/room0_nofdh_240ep_train \
        --split train \
        --gpu 5

The output directory will contain:
    backbone/rgb_0.pt, backbone/rgb_1.pt, ...   (decoded 1280d features)
    depth/depth_0.png, depth/depth_1.png, ...    (symlinked GT depth)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from radio_gs.config import load_config
from radio_gs.data.benchmark_paths import (
    list_feature_paths,
    load_w2c_from_pose_dir,
    load_w2c_from_pose_file,
    resolve_dataset_type,
    resolve_scene_root,
    resolve_split_data_dir,
    resolve_split_feature_dir,
    resolve_split_frame_ids,
    resolve_split_pose_source,
)
from radio_gs.geometry_utils import resolve_use_2dgs
from radio_gs.scripts.eval_rendered import load_model_and_render


@torch.no_grad()
def render_all_training_features(
    model,
    codec,
    renderer,
    sharpener,
    refiner,
    config,
    is_hybrid,
    device: torch.device,
    split: str = "train",
) -> list[tuple[int, torch.Tensor]]:
    """Render decoded 1280d features for the requested split.

    Returns a list of ``(flat_index, decoded_tensor)`` pairs where
    ``decoded_tensor`` has shape ``[1280, fH, fW]``.
    """
    from radio_gs.scripts.eval_rendered import (
        _build_refiner_guide,
        _hybrid_decode,
        _load_rgb_guide,
        _render_rgb_guide,
    )
    from radio_gs.rendering.feature_renderer import FeatureFieldRenderer

    dataset_type = resolve_dataset_type(config)
    scene_root = resolve_scene_root(config)
    train_split = getattr(config, "train_split", "Sequence_1")
    val_split = getattr(config, "val_split", "Sequence_2")

    feature_size = (
        getattr(config, "feature_height", 30),
        getattr(config, "feature_width", 40),
    )
    rgb_guide_enabled = getattr(config, "refiner_rgb_guide", False)
    depth_guide_enabled = getattr(config, "refiner_depth_guide", False)
    self_guided = getattr(config, "self_guided", False)
    use_rendered_rgb = self_guided

    # Optional RGB renderer for self-guided refiner
    rgb_renderer = None
    if use_rendered_rgb and rgb_guide_enabled:
        use_2dgs = resolve_use_2dgs(config)
        rgb_renderer = FeatureFieldRenderer(
            image_height=getattr(config, "image_height", 480),
            image_width=getattr(config, "image_width", 640),
            fx=getattr(config, "fx", 320.0),
            fy=getattr(config, "fy", 320.0),
            cx=getattr(config, "cx", 319.5),
            cy=getattr(config, "cy", 239.5),
            use_2dgs=use_2dgs,
        ).to(device)

    mixed_split = getattr(config, "mixed_split", False) and dataset_type == "replica"

    frames: list[tuple[int, np.ndarray, str | None]] = []

    if mixed_split:
        poses_s1 = (
            np.loadtxt(str(scene_root / train_split / "traj_w_c.txt"))
            .reshape(-1, 4, 4)
            .astype(np.float32)
        )
        poses_s2 = (
            np.loadtxt(str(scene_root / val_split / "traj_w_c.txt"))
            .reshape(-1, 4, 4)
            .astype(np.float32)
        )
        w2c_s1 = np.linalg.inv(poses_s1)
        w2c_s2 = np.linalg.inv(poses_s2)
        n_s1, n_s2 = len(poses_s1), len(poses_s2)
        total = n_s1 + n_s2
        mixed_ratio = getattr(config, "mixed_train_ratio", 0.8)
        mixed_seed = getattr(config, "mixed_seed", 42)
        gen = torch.Generator().manual_seed(mixed_seed)
        perm = torch.randperm(total, generator=gen).tolist()
        train_size = int(mixed_ratio * total)
        train_indices = sorted(perm[:train_size])
        val_indices = sorted(perm[train_size:])
        selected_indices = train_indices if split == "train" else val_indices

        for ci in selected_indices:
            if ci < n_s1:
                seq, fidx = train_split, ci
                w2c_mat = w2c_s1[fidx]
                rgb_dir = str(scene_root / seq / "rgb") if rgb_guide_enabled else None
            else:
                seq, fidx = val_split, ci - n_s1
                w2c_mat = w2c_s2[fidx]
                rgb_dir = str(scene_root / seq / "rgb") if rgb_guide_enabled else None
            frames.append((ci, w2c_mat, rgb_dir))
    else:
        split_key = "train" if split == "train" else "val"
        feat_dir = resolve_split_feature_dir(config, split_key)
        frame_ids = resolve_split_frame_ids(config, split_key)
        feat_paths = list_feature_paths(feat_dir, frame_ids=frame_ids)
        frame_ids_all = [int(p.stem.split("_")[1]) for p in feat_paths]

        pose_file, pose_dir = resolve_split_pose_source(config, split_key)
        if pose_dir:
            w2c_all = load_w2c_from_pose_dir(pose_dir, frame_ids_all)
        elif pose_file:
            w2c_all = load_w2c_from_pose_file(pose_file, frame_ids_all)
        else:
            raise RuntimeError(f"No pose source found for {split_key} split")

        train_rgb_dir = None
        if rgb_guide_enabled and not use_rendered_rgb:
            rdir = resolve_split_data_dir(config, split_key, "rgb")
            train_rgb_dir = str(rdir) if rdir else None

        for j, fid in enumerate(frame_ids_all):
            frames.append((fid, w2c_all[j], train_rgb_dir))

    # Render
    results: list[tuple[int, torch.Tensor]] = []
    print(f"  Rendering {len(frames)} {split} frames ...")

    for flat_idx, w2c_mat, rgb_dir in tqdm(frames, desc="Rendering+Decoding"):
        pose = torch.from_numpy(w2c_mat[np.newaxis]).to(device)

        if self_guided:
            result = renderer.render_features_and_rgb(model, pose)
            self_rgb = result["rgb"]
        else:
            result = renderer.render_features_batch(model, pose)
            self_rgb = None

        rendered = sharpener(result["feature_map"])

        if refiner is not None:
            guide = None
            if self_rgb is not None:
                guide = self_rgb
            elif rgb_renderer is not None:
                guide = _render_rgb_guide(model, rgb_renderer, pose[0], feature_size)
            elif rgb_dir is not None:
                frame_idx_for_rgb = flat_idx
                if mixed_split:
                    n_s1_local = len(w2c_s1)
                    frame_idx_for_rgb = flat_idx if flat_idx < n_s1_local else flat_idx - n_s1_local
                guide = _load_rgb_guide(rgb_dir, frame_idx_for_rgb, feature_size, dataset_type=dataset_type)
                if guide is not None:
                    guide = guide.to(device)

            if depth_guide_enabled or getattr(config, "refiner_alpha_guide", False) or getattr(config, "refiner_boundary_guide", False):
                guide = _build_refiner_guide(result, config, rgb_guide=guide)

            rendered = refiner(rendered, guide=guide)

        if is_hybrid:
            rendered = _hybrid_decode(model, rendered, result, pose, renderer.K)

        decoded = codec.decoder(rendered).squeeze(0).cpu()  # [1280, fH, fW]
        results.append((flat_idx, decoded))

    return results


def link_depth_maps(
    config,
    frame_indices: list[int],
    output_depth_dir: Path,
):
    """Create symlinks (or copies) of GT depth maps for the rendered frames."""
    dataset_type = resolve_dataset_type(config)
    scene_root = resolve_scene_root(config)
    train_split = getattr(config, "train_split", "Sequence_1")
    val_split = getattr(config, "val_split", "Sequence_2")
    mixed_split = getattr(config, "mixed_split", False) and dataset_type == "replica"

    output_depth_dir.mkdir(parents=True, exist_ok=True)

    if mixed_split:
        n_s1 = len(
            np.loadtxt(str(scene_root / train_split / "traj_w_c.txt")).reshape(-1, 4, 4)
        )
        for flat_idx in frame_indices:
            if flat_idx < n_s1:
                seq, fidx = train_split, flat_idx
            else:
                seq, fidx = val_split, flat_idx - n_s1
            src = scene_root / seq / "depth" / f"depth_{fidx}.png"
            dst = output_depth_dir / f"depth_{flat_idx}.png"
            if src.exists() and not dst.exists():
                os.symlink(src, dst)
    else:
        depth_dir = resolve_split_data_dir(config, "train", "depth")
        if depth_dir is None:
            print("WARNING: No depth directory found, skipping depth linking")
            return
        for fid in frame_indices:
            src = depth_dir / f"depth_{fid}.png"
            dst = output_depth_dir / f"depth_{fid}.png"
            if src.exists() and not dst.exists():
                os.symlink(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Render decoded 1280d features from a trained model"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Model config YAML"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint .pth"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory (will contain backbone/ and depth/ subdirs)",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Which evaluation split to render",
    )
    parser.add_argument(
        "--link_depth",
        action="store_true",
        default=True,
        help="Symlink GT depth maps alongside features (default: True)",
    )
    parser.add_argument(
        "--no_link_depth",
        action="store_false",
        dest="link_depth",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda")

    out_root = Path(args.output_dir)
    feat_dir = out_root / "backbone"
    depth_dir = out_root / "depth"
    feat_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Render Codec Features ===")
    print(f"  Config:     {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output:     {args.output_dir}")
    print(f"  GPU:        {args.gpu}")
    print(f"  Split:      {args.split}")

    print(f"\nLoading model ...")
    model, codec, renderer, sharpener, refiner, config, is_hybrid = (
        load_model_and_render(args.config, args.checkpoint)
    )

    print(f"\nRendering features ...")
    results = render_all_training_features(
        model, codec, renderer, sharpener, refiner, config, is_hybrid, device, split=args.split
    )

    print(f"\nSaving {len(results)} feature maps to {feat_dir} ...")
    frame_indices = []
    for flat_idx, decoded in tqdm(results, desc="Saving"):
        torch.save(decoded, feat_dir / f"rgb_{flat_idx}.pt")
        frame_indices.append(flat_idx)

    if args.link_depth:
        print(f"\nLinking depth maps to {depth_dir} ...")
        link_depth_maps(config, frame_indices, depth_dir)

    print(f"\n=== Done ===")
    print(f"  Features: {feat_dir} ({len(results)} files)")
    if args.link_depth:
        print(f"  Depth:    {depth_dir}")
    print(f"\nTo train domain-matched depth head:")
    print(f"  python radio_gs/scripts/pretrain_oracle_head.py \\")
    print(f"    --feature_dir {feat_dir} \\")
    print(f"    --depth_dir {depth_dir} \\")
    print(f"    --output_path output/radio_gs/oracle_heads/room_0_dm_depth_head.pth \\")
    print(f"    --epochs 500 --gpu {args.gpu}")


if __name__ == "__main__":
    main()
