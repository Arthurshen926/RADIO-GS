"""Visualize intermediate RADIO-GS feature-flow stages.

This script renders one or more views and saves both grouped comparisons and a
single composite figure that traces how features evolve through the pipeline.

Outputs per selected view:
    grouped/latent_flow/frame_<label>.png
    grouped/hybrid_flow/frame_<label>.png
    grouped/radio_flow/frame_<label>.png
    grouped/geometry_guides/frame_<label>.png
    composite/frame_<label>.png

And summary grids across all selected views:
    grouped/*/grid.png
    composite/grid.png

Optional raw tensors can also be saved for further analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from radio_gs.config import load_config
from radio_gs.data.benchmark_paths import (
    extract_feature_frame_index,
    list_feature_paths,
    load_w2c_from_pose_dir,
    load_w2c_from_pose_file,
    resolve_dataset_type,
    resolve_rgb_path,
    resolve_scene_root,
    resolve_split_data_dir,
    resolve_split_feature_dir,
    resolve_split_frame_ids,
    resolve_split_pose_source,
)
from radio_gs.geometry_utils import resolve_use_2dgs
from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
from radio_gs.models.featsharp_3d import FeatSharp3D
from radio_gs.models.hcd_codec import HCDCodec
from radio_gs.models.screen_refiner import (
    ScreenSpaceRefiner,
    build_refiner_guide,
    compute_refiner_extra_channels,
)
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_renderer(config, image_height=None, image_width=None):
    height = image_height or getattr(config, "feature_height", 30)
    width = image_width or getattr(config, "feature_width", 40)
    return FeatureFieldRenderer(
        image_height=height,
        image_width=width,
        fx=getattr(config, "fx", 320.0) * width / getattr(config, "image_width", 640),
        fy=getattr(config, "fy", 320.0) * height / getattr(config, "image_height", 480),
        cx=getattr(config, "cx", 319.5) * width / getattr(config, "image_width", 640),
        cy=getattr(config, "cy", 239.5) * height / getattr(config, "image_height", 480),
        max_channels_per_chunk=getattr(config, "max_channels_per_chunk", 32),
        use_2dgs=resolve_use_2dgs(config),
    ).to(device)


def load_pipeline(config_path, checkpoint_path):
    config = load_config(config_path)

    architecture = getattr(config, "architecture", "explicit")
    is_hybrid = architecture == "hybrid"
    if is_hybrid:
        from radio_gs.models.hybrid_gaussian import HybridFeatureGaussian

        latent_dim = getattr(config, "hybrid_latent_dim", 16)
        model = HybridFeatureGaussian(
            latent_dim=latent_dim,
            hash_output_dim=getattr(config, "hash_output_dim", 48),
            fine_dim=getattr(config, "fine_dim", 64),
            coarse_dim=getattr(config, "coarse_dim", 64),
            output_dim=getattr(config, "hybrid_output_dim", 128),
            num_levels=getattr(config, "hash_levels", 16),
            features_per_level=getattr(config, "hash_features_per_level", 2),
            log2_hashmap_size=getattr(config, "hash_log2_size", 19),
            base_resolution=getattr(config, "hash_base_resolution", 16),
            max_resolution=getattr(config, "hash_max_resolution", 2048),
            decoupled_heads=getattr(config, "hybrid_decoupled_heads", False),
            use_semantic_adaptor=getattr(config, "hybrid_semantic_adaptor", False),
            semantic_adaptor_mode=getattr(config, "hybrid_semantic_adaptor_mode", "confidence"),
            semantic_adaptor_hidden_dim=getattr(config, "hybrid_semantic_adaptor_hidden_dim", 64),
            semantic_adaptor_use_geometry_guidance=getattr(
                config, "hybrid_semantic_adaptor_use_geometry_guidance", True
            ),
            semantic_adaptor_use_depth_guidance=getattr(
                config, "hybrid_semantic_adaptor_use_depth_guidance", False
            ),
            semantic_adaptor_residual=getattr(
                config, "hybrid_semantic_adaptor_residual", True
            ),
        )
    else:
        latent_dim = getattr(config, "latent_dim", 64)
        model = ExplicitFeatureGaussian(latent_dim=latent_dim)

    ply_path = getattr(config, "ply_path", "")
    if ply_path:
        model.load_from_ply(ply_path)
    model = model.to(device).eval()

    codec = HCDCodec(
        input_dim=getattr(config, "radio_feature_dim", 1280),
        bottleneck_dim=getattr(config, "bottleneck_dim", 64),
        dual_stream=getattr(config, "dual_stream", True),
        symmetric_decoder=getattr(config, "symmetric_decoder", False),
    ).to(device).eval()

    renderer = build_renderer(config)

    sharpener = FeatSharp3D(
        mode=getattr(config, "featsharp_mode", "analytical"),
        feature_dim=latent_dim,
        strength=getattr(config, "featsharp_strength", 0.3),
    ).to(device).eval()

    refiner = None
    if getattr(config, "use_refiner", False):
        extra_ch = compute_refiner_extra_channels(
            rgb_guide=getattr(config, "refiner_rgb_guide", False),
            depth_guide=getattr(config, "refiner_depth_guide", False),
            depth_grad=getattr(config, "refiner_depth_grad", False),
            alpha_guide=getattr(config, "refiner_alpha_guide", False),
            boundary_guide=getattr(config, "refiner_boundary_guide", False),
        )
        refiner = ScreenSpaceRefiner(
            latent_dim=latent_dim,
            hidden_dim=getattr(config, "refiner_hidden_dim", 128),
            num_blocks=getattr(config, "refiner_num_blocks", 4),
            dropout=getattr(config, "refiner_dropout", 0.1),
            extra_channels=extra_ch,
            norm_type=getattr(config, "refiner_norm_type", "gn"),
        ).to(device).eval()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    codec.load_state_dict(checkpoint["codec_state_dict"], strict=False)
    if "sharpener_state_dict" in checkpoint:
        sharpener.load_state_dict(checkpoint["sharpener_state_dict"], strict=False)
    if refiner is not None and "refiner_state_dict" in checkpoint:
        refiner.load_state_dict(checkpoint["refiner_state_dict"], strict=False)

    return model, codec, renderer, sharpener, refiner, config, is_hybrid


def ensure_feature_map(feat: torch.Tensor, target_hw: Optional[tuple[int, int]] = None) -> torch.Tensor:
    feat = feat.float().cpu()
    if feat.dim() == 4:
        feat = feat.squeeze(0)
    if feat.dim() == 2:
        if target_hw is None:
            raise ValueError("target_hw required when feature tensor is flattened")
        height, width = target_hw
        feat = feat.reshape(height, width, feat.shape[1]).permute(2, 0, 1)
    if feat.dim() != 3:
        raise ValueError(f"Expected feature map [C,H,W], got shape {tuple(feat.shape)}")
    if target_hw is not None and feat.shape[-2:] != target_hw:
        feat = F.interpolate(
            feat.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=False
        ).squeeze(0)
    return feat.contiguous()


def subsample_positions(n_total: int, n_views: int, sample_indices: Optional[List[int]] = None) -> List[int]:
    if n_total <= 0:
        return []
    if sample_indices:
        picked = [idx for idx in sample_indices if 0 <= idx < n_total]
        if not picked:
            raise ValueError(f"No valid sample indices in range [0, {n_total})")
        return picked
    step = max(1, n_total // max(1, n_views))
    return list(range(0, n_total, step))[:n_views]


def resolve_visualization_samples(config, split: str, n_views: int, sample_indices: Optional[List[int]] = None):
    dataset_type = resolve_dataset_type(config)
    scene_root = resolve_scene_root(config)
    train_split = getattr(config, "train_split", "Sequence_1")
    val_split = getattr(config, "val_split", "Sequence_2")
    mixed_split = getattr(config, "mixed_split", False) and dataset_type == "replica"

    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split}")

    if mixed_split:
        poses_s1 = np.loadtxt(str(scene_root / train_split / "traj_w_c.txt")).reshape(-1, 4, 4).astype(np.float32)
        poses_s2 = np.loadtxt(str(scene_root / val_split / "traj_w_c.txt")).reshape(-1, 4, 4).astype(np.float32)
        w2c_s1 = np.linalg.inv(poses_s1)
        w2c_s2 = np.linalg.inv(poses_s2)
        n_s1 = len(w2c_s1)
        n_s2 = len(w2c_s2)
        total = n_s1 + n_s2
        train_size = int(getattr(config, "mixed_train_ratio", 0.8) * total)
        generator = torch.Generator().manual_seed(getattr(config, "mixed_seed", 42))
        perm = torch.randperm(total, generator=generator).tolist()
        train_sel = sorted(perm[:train_size])
        val_sel = sorted(perm[train_size:])
        selected = train_sel if split == "train" else val_sel
        picked = subsample_positions(len(selected), n_views, sample_indices)

        train_feat_dir = resolve_split_feature_dir(config, "train")
        val_feat_dir = resolve_split_feature_dir(config, "val")
        train_feat_map = {
            extract_feature_frame_index(path): path for path in list_feature_paths(train_feat_dir)
        }
        val_feat_map = {
            extract_feature_frame_index(path): path for path in list_feature_paths(val_feat_dir)
        }

        samples = []
        for picked_idx in picked:
            combined_idx = selected[picked_idx]
            if combined_idx < n_s1:
                seq_name = train_split
                frame_id = combined_idx
                pose_w2c = w2c_s1[frame_id]
                gt_feat_path = train_feat_map[frame_id]
            else:
                seq_name = val_split
                frame_id = combined_idx - n_s1
                pose_w2c = w2c_s2[frame_id]
                gt_feat_path = val_feat_map[frame_id]
            rgb_dir = scene_root / seq_name / "rgb"
            rgb_path = resolve_rgb_path(rgb_dir, frame_id, dataset_type)
            samples.append(
                {
                    "label": f"{seq_name}_f{frame_id:04d}",
                    "frame_id": frame_id,
                    "pose_w2c": pose_w2c,
                    "gt_feat_path": gt_feat_path,
                    "rgb_path": str(rgb_path) if rgb_path is not None else None,
                }
            )
        return samples

    train_frame_ids_cfg = resolve_split_frame_ids(config, "train")
    val_frame_ids_cfg = resolve_split_frame_ids(config, "val")
    use_generic_partition = (
        dataset_type != "replica"
        and train_frame_ids_cfg is None
        and val_frame_ids_cfg is None
    )

    if use_generic_partition:
        feat_dir = resolve_split_feature_dir(config, "train")
        feat_paths_all = list_feature_paths(feat_dir)
        frame_ids_all = [extract_feature_frame_index(path) for path in feat_paths_all]
        pose_file, pose_dir = resolve_split_pose_source(config, "train")
        if pose_dir:
            w2c_all = load_w2c_from_pose_dir(pose_dir, frame_ids_all)
        elif pose_file:
            w2c_all = load_w2c_from_pose_file(pose_file, frame_ids_all)
        else:
            raise ValueError("No pose source configured for generic dataset split")
        generator = torch.Generator().manual_seed(getattr(config, "mixed_seed", 42))
        perm = torch.randperm(len(frame_ids_all), generator=generator).tolist()
        train_cut = int(getattr(config, "mixed_train_ratio", 0.8) * len(frame_ids_all))
        selected_positions = sorted(perm[:train_cut] if split == "train" else perm[train_cut:])
        picked = subsample_positions(len(selected_positions), n_views, sample_indices)
        rgb_dir = resolve_split_data_dir(config, "train", "rgb")

        samples = []
        for picked_idx in picked:
            source_idx = selected_positions[picked_idx]
            frame_id = frame_ids_all[source_idx]
            rgb_path = resolve_rgb_path(rgb_dir, frame_id, dataset_type) if rgb_dir is not None else None
            samples.append(
                {
                    "label": f"{split}_f{frame_id:04d}",
                    "frame_id": frame_id,
                    "pose_w2c": w2c_all[source_idx],
                    "gt_feat_path": feat_paths_all[source_idx],
                    "rgb_path": str(rgb_path) if rgb_path is not None else None,
                }
            )
        return samples

    feat_dir = resolve_split_feature_dir(config, split)
    feat_paths_all = list_feature_paths(feat_dir, frame_ids=resolve_split_frame_ids(config, split))
    frame_ids_all = [extract_feature_frame_index(path) for path in feat_paths_all]
    pose_file, pose_dir = resolve_split_pose_source(config, split)
    if pose_dir:
        w2c_all = load_w2c_from_pose_dir(pose_dir, frame_ids_all)
    elif pose_file:
        w2c_all = load_w2c_from_pose_file(pose_file, frame_ids_all)
    else:
        raise ValueError(f"No pose source configured for split={split}")
    picked = subsample_positions(len(frame_ids_all), n_views, sample_indices)
    rgb_dir = resolve_split_data_dir(config, split, "rgb")

    samples = []
    for picked_idx in picked:
        frame_id = frame_ids_all[picked_idx]
        rgb_path = resolve_rgb_path(rgb_dir, frame_id, dataset_type) if rgb_dir is not None else None
        samples.append(
            {
                "label": f"{split}_f{frame_id:04d}",
                "frame_id": frame_id,
                "pose_w2c": w2c_all[picked_idx],
                "gt_feat_path": feat_paths_all[picked_idx],
                "rgb_path": str(rgb_path) if rgb_path is not None else None,
            }
        )
    return samples


def load_rgb_image(rgb_path: Optional[str]) -> Optional[np.ndarray]:
    if not rgb_path:
        return None
    image = cv2.imread(str(rgb_path))
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_positions(model, position_map: torch.Tensor) -> torch.Tensor:
    xyz = model.get_xyz()
    margin = 0.1
    lo = xyz.min(dim=0).values - margin
    hi = xyz.max(dim=0).values + margin
    extent = (hi - lo).clamp(min=1e-6)
    return ((position_map - lo.view(1, 3, 1, 1)) / extent.view(1, 3, 1, 1)).clamp(0, 1)


def tensor_to_rgb_preview(x: Optional[torch.Tensor]) -> np.ndarray:
    if x is None:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    tensor = x.detach().float().cpu()
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 3:
        raise ValueError(f"Expected tensor preview input [C,H,W], got {tuple(tensor.shape)}")
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.shape[0] == 2:
        third = tensor.mean(dim=0, keepdim=True)
        tensor = torch.cat([tensor, third], dim=0)
    elif tensor.shape[0] > 3:
        tensor = tensor[:3]
    lo = tensor.amin(dim=(1, 2), keepdim=True)
    hi = tensor.amax(dim=(1, 2), keepdim=True)
    tensor = (tensor - lo) / (hi - lo + 1e-6)
    return (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def shared_pca_colorize(features_list: List[torch.Tensor], n_components: int = 3) -> List[np.ndarray]:
    if not features_list:
        return []
    channels = {int(feat.shape[0]) for feat in features_list}
    if len(channels) != 1:
        raise ValueError("shared_pca_colorize requires all feature maps to have the same channel count")
    all_flat = []
    shapes = []
    for feat in features_list:
        feat = ensure_feature_map(feat)
        channels, height, width = feat.shape
        shapes.append((height, width))
        all_flat.append(feat.reshape(channels, -1).T.cpu().numpy())
    stacked = np.concatenate(all_flat, axis=0)
    mean = stacked.mean(axis=0)
    centered = stacked - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:n_components]
    images = []
    offset = 0
    for height, width in shapes:
        count = height * width
        proj = centered[offset : offset + count] @ basis.T
        offset += count
        for comp in range(n_components):
            vmin = proj[:, comp].min()
            vmax = proj[:, comp].max()
            if vmax - vmin > 1e-6:
                proj[:, comp] = (proj[:, comp] - vmin) / (vmax - vmin)
            else:
                proj[:, comp] = 0.5
        images.append((proj.reshape(height, width, n_components) * 255.0).astype(np.uint8))
    return images


def individual_pca_colorize(feat: torch.Tensor) -> np.ndarray:
    return shared_pca_colorize([ensure_feature_map(feat)])[0]


def colorize_single_map(x: Optional[torch.Tensor], colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    if x is None:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    tensor = x.detach().float().cpu()
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    array = tensor.numpy().astype(np.float32)
    if array.size == 0:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    valid = np.isfinite(array)
    if not valid.any():
        return np.zeros((*array.shape, 3), dtype=np.uint8)
    lo = array[valid].min()
    hi = array[valid].max()
    array = (array - lo) / (hi - lo + 1e-6)
    colored = cv2.applyColorMap((np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8), colormap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def position_map_to_rgb(position_map: torch.Tensor) -> np.ndarray:
    tensor = ensure_feature_map(position_map)
    if tensor.shape[0] < 3:
        tensor = F.pad(tensor, (0, 0, 0, 0, 0, 3 - tensor.shape[0]))
    tensor = tensor[:3]
    lo = tensor.amin(dim=(1, 2), keepdim=True)
    hi = tensor.amax(dim=(1, 2), keepdim=True)
    tensor = (tensor - lo) / (hi - lo + 1e-6)
    return (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def cosine_heatmap(decoded: torch.Tensor, gt: torch.Tensor) -> np.ndarray:
    decoded = ensure_feature_map(decoded)
    gt = ensure_feature_map(gt, target_hw=decoded.shape[-2:])
    decoded_norm = F.normalize(decoded, p=2, dim=0)
    gt_norm = F.normalize(gt, p=2, dim=0)
    cos = F.cosine_similarity(
        decoded_norm.reshape(decoded.shape[0], -1),
        gt_norm.reshape(gt.shape[0], -1),
        dim=0,
    ).reshape(decoded.shape[-2], decoded.shape[-1]).cpu().numpy()
    cos = np.clip(cos, 0.0, 1.0)
    colored = cv2.applyColorMap((cos * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def upscale(img: np.ndarray, scale: int) -> np.ndarray:
    return cv2.resize(
        img,
        (img.shape[1] * scale, img.shape[0] * scale),
        interpolation=cv2.INTER_LINEAR,
    )


def ensure_same_height(images: List[np.ndarray]) -> List[np.ndarray]:
    target_h = max(img.shape[0] for img in images)
    out = []
    for img in images:
        if img.shape[0] == target_h:
            out.append(img)
            continue
        scale = target_h / img.shape[0]
        target_w = max(1, int(round(img.shape[1] * scale)))
        out.append(cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR))
    return out


def ensure_same_width(images: List[np.ndarray]) -> List[np.ndarray]:
    target_w = max(img.shape[1] for img in images)
    out = []
    for img in images:
        if img.shape[1] == target_w:
            out.append(img)
            continue
        scale = target_w / img.shape[1]
        target_h = max(1, int(round(img.shape[0] * scale)))
        out.append(cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR))
    return out


def add_text(img: np.ndarray, text: str, pos=(5, 20), font_scale=0.5) -> np.ndarray:
    result = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(result, (x - 2, y - th - 4), (x + tw + 2, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(result, text, pos, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return result


def hconcat_with_border(images: List[np.ndarray], border: int = 3) -> np.ndarray:
    images = ensure_same_height(images)
    parts = []
    for idx, img in enumerate(images):
        if idx > 0:
            parts.append(np.full((img.shape[0], border, 3), 255, dtype=np.uint8))
        parts.append(img)
    return np.concatenate(parts, axis=1)


def vconcat_with_border(images: List[np.ndarray], border: int = 3) -> np.ndarray:
    images = ensure_same_width(images)
    parts = []
    for idx, img in enumerate(images):
        if idx > 0:
            parts.append(np.full((border, img.shape[1], 3), 255, dtype=np.uint8))
        parts.append(img)
    return np.concatenate(parts, axis=0)


def make_header(texts: List[str], cell_width: int, height: int = 30, border: int = 3) -> np.ndarray:
    cells = []
    for idx, text in enumerate(texts):
        cell = np.zeros((height, cell_width, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
        x = max(0, (cell_width - tw) // 2)
        y = (height + th) // 2
        cv2.putText(cell, text, (x, y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if idx > 0:
            cells.append(np.zeros((height, border, 3), dtype=np.uint8))
        cells.append(cell)
    return np.concatenate(cells, axis=1)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def build_gt_rgb_image(sample: Dict[str, Any]) -> Optional[np.ndarray]:
    return load_rgb_image(sample.get("rgb_path"))


@torch.no_grad()
def render_feature_flow(model, codec, renderer, sharpener, refiner, config, sample, is_hybrid):
    pose = torch.from_numpy(sample["pose_w2c"][np.newaxis]).to(device)
    frame_id = int(sample["frame_id"])
    feature_hw = (getattr(config, "feature_height", 30), getattr(config, "feature_width", 40))
    dataset_type = resolve_dataset_type(config)
    rgb_guide_enabled = getattr(config, "refiner_rgb_guide", False)
    self_guided = getattr(config, "self_guided", False)

    if self_guided and rgb_guide_enabled:
        render_result = renderer.render_features_and_rgb(model, pose)
        rgb_guide = render_result["rgb"]
    else:
        render_result = renderer.render_features_batch(model, pose)
        rgb_guide = None
        if rgb_guide_enabled:
            rgb_path = sample.get("rgb_path")
            if rgb_path:
                guide_img = load_rgb_image(rgb_path)
                if guide_img is not None:
                    guide_img = cv2.resize(guide_img, (feature_hw[1], feature_hw[0]), interpolation=cv2.INTER_LINEAR)
                    rgb_guide = (
                        torch.from_numpy(guide_img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                    )

    latent_rendered = render_result["feature_map"].float()
    depth_map = render_result.get("depth_map")
    alpha_map = render_result.get("alpha_map")

    latent_sharpened = sharpener(latent_rendered)

    guide = None
    if refiner is not None:
        guide = build_refiner_guide(
            render_result,
            rgb_guide=rgb_guide,
            use_depth_guide=getattr(config, "refiner_depth_guide", False),
            use_depth_grad=getattr(config, "refiner_depth_grad", False),
            depth_grad_scale=getattr(config, "refiner_depth_grad_scale", 10.0),
            use_alpha_guide=getattr(config, "refiner_alpha_guide", False),
            use_boundary_guide=getattr(config, "refiner_boundary_guide", False),
        )
        latent_refined = refiner(latent_sharpened, guide=guide)
    else:
        latent_refined = latent_sharpened

    position_map = None
    fine_feat = None
    hash_feat = None
    coarse_feat = None
    geometry_feat = None
    semantic_feat = None
    fused_feat = None
    geometry_gate = None
    semantic_gate = None
    decoded_input = latent_refined

    if is_hybrid:
        from radio_gs.models.hybrid_gaussian import unproject_depth_to_positions

        if depth_map is None:
            raise RuntimeError("Hybrid visualization requires depth_map from renderer")
        position_map = unproject_depth_to_positions(
            depth_map.float(),
            pose.float(),
            renderer.K.float(),
            depth_map.shape[1],
            depth_map.shape[2],
        )
        position_map = normalize_positions(model, position_map)
        fine_feat = model.fine_decoder(latent_refined.float())
        hash_feat = model.hash_field.forward_screen_space(position_map)
        coarse_feat = model.coarse_decoder(hash_feat)
        if getattr(model, "decoupled_heads", False):
            aux = model.fusion_head(
                fine_feat,
                coarse_feat,
                return_aux=True,
                depth_map=depth_map.float(),
            )
            fused_feat = aux["fused"]
            geometry_feat = aux.get("geometry")
            semantic_feat = aux.get("semantic")
            geometry_gate = aux.get("geometry_gate")
            semantic_gate = aux.get("semantic_gate")
        else:
            fused_feat = model.fusion_head(fine_feat, coarse_feat)
        decoded_input = fused_feat

    decoded_1280 = codec.decoder(decoded_input.float())
    gt_1280 = ensure_feature_map(
        torch.load(sample["gt_feat_path"], map_location="cpu"),
        target_hw=decoded_1280.shape[-2:],
    )

    return {
        "label": sample["label"],
        "frame_id": frame_id,
        "rgb_native": build_gt_rgb_image(sample),
        "latent_rendered": ensure_feature_map(latent_rendered),
        "latent_sharpened": ensure_feature_map(latent_sharpened),
        "latent_refined": ensure_feature_map(latent_refined),
        "depth_map": depth_map.detach().float().cpu() if depth_map is not None else None,
        "alpha_map": alpha_map.detach().float().cpu() if alpha_map is not None else None,
        "refiner_guide": guide.detach().float().cpu() if guide is not None else None,
        "position_map": ensure_feature_map(position_map) if position_map is not None else None,
        "fine_feat": ensure_feature_map(fine_feat) if fine_feat is not None else None,
        "hash_feat": ensure_feature_map(hash_feat) if hash_feat is not None else None,
        "coarse_feat": ensure_feature_map(coarse_feat) if coarse_feat is not None else None,
        "geometry_feat": ensure_feature_map(geometry_feat) if geometry_feat is not None else None,
        "semantic_feat": ensure_feature_map(semantic_feat) if semantic_feat is not None else None,
        "fused_feat": ensure_feature_map(fused_feat) if fused_feat is not None else None,
        "geometry_gate": geometry_gate.detach().float().cpu() if geometry_gate is not None else None,
        "semantic_gate": semantic_gate.detach().float().cpu() if semantic_gate is not None else None,
        "decoded_1280": ensure_feature_map(decoded_1280),
        "gt_1280": gt_1280,
    }


def assign_pca_images(flows: List[Dict[str, Any]]) -> None:
    latent_keys = ["latent_rendered", "latent_sharpened", "latent_refined"]
    latent_feats = [flow[key] for flow in flows for key in latent_keys if flow.get(key) is not None]
    latent_images = shared_pca_colorize(latent_feats) if latent_feats else []
    offset = 0
    for flow in flows:
        flow["pca_latent"] = {}
        for key in latent_keys:
            if flow.get(key) is not None:
                flow["pca_latent"][key] = latent_images[offset]
                offset += 1

    if flows and all(flow.get("fine_feat") is not None for flow in flows) and all(
        flow["fine_feat"].shape[0] == flows[0]["fine_feat"].shape[0] for flow in flows
    ) and all(flow.get("coarse_feat") is not None for flow in flows) and all(
        flow["coarse_feat"].shape[0] == flows[0]["fine_feat"].shape[0] for flow in flows
    ):
        fine_coarse_feats = [flow["fine_feat"] for flow in flows] + [flow["coarse_feat"] for flow in flows]
        fine_coarse_images = shared_pca_colorize(fine_coarse_feats)
        n = len(flows)
        for idx, flow in enumerate(flows):
            flow["pca_fine"] = fine_coarse_images[idx]
            flow["pca_coarse"] = fine_coarse_images[n + idx]
    else:
        for flow in flows:
            if flow.get("fine_feat") is not None:
                flow["pca_fine"] = individual_pca_colorize(flow["fine_feat"])
            if flow.get("coarse_feat") is not None:
                flow["pca_coarse"] = individual_pca_colorize(flow["coarse_feat"])

    output_keys = []
    for key in ["geometry_feat", "semantic_feat", "fused_feat"]:
        if any(flow.get(key) is not None for flow in flows):
            output_keys.append(key)
    if output_keys:
        grouped_by_channels: Dict[int, List[tuple[int, str, torch.Tensor]]] = {}
        for flow_idx, flow in enumerate(flows):
            for key in output_keys:
                feat = flow.get(key)
                if feat is None:
                    continue
                grouped_by_channels.setdefault(int(feat.shape[0]), []).append((flow_idx, key, feat))
        for entries in grouped_by_channels.values():
            images = shared_pca_colorize([feat for _, _, feat in entries])
            for image, (flow_idx, key, _) in zip(images, entries):
                flows[flow_idx].setdefault("pca_output", {})[key] = image

    radio_feats = [flow["decoded_1280"] for flow in flows] + [flow["gt_1280"] for flow in flows]
    radio_images = shared_pca_colorize(radio_feats)
    n = len(flows)
    for idx, flow in enumerate(flows):
        flow["pca_decoded"] = radio_images[idx]
        flow["pca_gt"] = radio_images[n + idx]


def build_group_images(flow: Dict[str, Any], scale: int) -> Dict[str, np.ndarray]:
    target_hw = flow["decoded_1280"].shape[-2:]
    display_hw = (target_hw[0] * scale, target_hw[1] * scale)
    rgb_native = flow.get("rgb_native")
    if rgb_native is None:
        rgb_panel_img = np.zeros((display_hw[0], display_hw[1], 3), dtype=np.uint8)
    else:
        rgb_panel_img = cv2.resize(
            rgb_native,
            (display_hw[1], display_hw[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    rgb_panel = add_text(rgb_panel_img, "Input RGB")

    latent_panels = [
        rgb_panel,
        add_text(upscale(flow["pca_latent"]["latent_rendered"], scale), "Rendered Latent"),
        add_text(upscale(flow["pca_latent"]["latent_sharpened"], scale), "FeatSharp Latent"),
        add_text(upscale(flow["pca_latent"]["latent_refined"], scale), "Refined Latent"),
    ]
    latent_flow = hconcat_with_border(latent_panels, border=3)

    hybrid_panels = []
    if flow.get("position_map") is not None:
        hybrid_panels.append(add_text(upscale(position_map_to_rgb(flow["position_map"]), scale), "Position Map"))
    if flow.get("fine_feat") is not None:
        hybrid_panels.append(add_text(upscale(flow["pca_fine"], scale), "Fine Branch"))
    if flow.get("coarse_feat") is not None:
        hybrid_panels.append(add_text(upscale(flow["pca_coarse"], scale), "Coarse Branch"))
    if flow.get("geometry_feat") is not None and "pca_output" in flow and "geometry_feat" in flow["pca_output"]:
        hybrid_panels.append(add_text(upscale(flow["pca_output"]["geometry_feat"], scale), "Geometry Branch"))
    if flow.get("semantic_feat") is not None and "pca_output" in flow and "semantic_feat" in flow["pca_output"]:
        hybrid_panels.append(add_text(upscale(flow["pca_output"]["semantic_feat"], scale), "Semantic Branch"))
    if flow.get("fused_feat") is not None:
        fused_img = flow.get("pca_output", {}).get("fused_feat", individual_pca_colorize(flow["fused_feat"]))
        hybrid_panels.append(add_text(upscale(fused_img, scale), "Fused Hybrid"))
    hybrid_flow = hconcat_with_border(hybrid_panels, border=3) if hybrid_panels else np.zeros((32, 32, 3), dtype=np.uint8)

    radio_panels = [
        add_text(upscale(flow["pca_decoded"], scale), "Decoded 1280d"),
        add_text(upscale(flow["pca_gt"], scale), "GT RADIO 1280d"),
        add_text(upscale(cosine_heatmap(flow["decoded_1280"], flow["gt_1280"]), scale), "Decoded vs GT Cosine"),
    ]
    radio_flow = hconcat_with_border(radio_panels, border=3)

    geometry_panels = [
        add_text(upscale(colorize_single_map(flow.get("depth_map"), cv2.COLORMAP_INFERNO), scale), "Render Depth"),
        add_text(upscale(colorize_single_map(flow.get("alpha_map"), cv2.COLORMAP_VIRIDIS), scale), "Alpha Map"),
        add_text(upscale(tensor_to_rgb_preview(flow.get("refiner_guide")), scale), "Refiner Guide"),
    ]
    if flow.get("geometry_gate") is not None:
        geometry_panels.append(
            add_text(upscale(colorize_single_map(flow.get("geometry_gate"), cv2.COLORMAP_MAGMA), scale), "Geometry Gate")
        )
    if flow.get("semantic_gate") is not None:
        geometry_panels.append(
            add_text(upscale(colorize_single_map(flow.get("semantic_gate"), cv2.COLORMAP_MAGMA), scale), "Semantic Gate")
        )
    geometry_guides = hconcat_with_border(geometry_panels, border=3)

    composite = vconcat_with_border(
        [latent_flow, hybrid_flow, radio_flow, geometry_guides],
        border=4,
    )
    composite = add_text(composite, f"{flow['label']}", pos=(8, 24), font_scale=0.7)

    return {
        "latent_flow": latent_flow,
        "hybrid_flow": hybrid_flow,
        "radio_flow": radio_flow,
        "geometry_guides": geometry_guides,
        "composite": composite,
    }


def build_group_grid(images: List[np.ndarray], labels: List[str], border: int = 3) -> np.ndarray:
    if not images:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    rows = []
    for label, image in zip(labels, images):
        rows.append(add_text(image, label, pos=(8, 24), font_scale=0.7))
    return vconcat_with_border(rows, border=border)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RADIO-GS intermediate feature flow")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--n_views", type=int, default=6)
    parser.add_argument(
        "--sample_indices",
        nargs="+",
        type=int,
        help="Optional indices within the resolved split subset to visualize",
    )
    parser.add_argument("--scale", type=int, default=16)
    parser.add_argument("--save_tensors", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    grouped_root = out_root / "grouped"
    composite_root = out_root / "composite"
    tensor_root = out_root / "tensors"
    for subdir in [
        grouped_root / "latent_flow",
        grouped_root / "hybrid_flow",
        grouped_root / "radio_flow",
        grouped_root / "geometry_guides",
        composite_root,
    ]:
        subdir.mkdir(parents=True, exist_ok=True)
    if args.save_tensors:
        tensor_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading pipeline from {args.checkpoint}...")
    model, codec, renderer, sharpener, refiner, config, is_hybrid = load_pipeline(
        args.config, args.checkpoint
    )

    samples = resolve_visualization_samples(
        config,
        split=args.split,
        n_views=args.n_views,
        sample_indices=args.sample_indices,
    )
    if not samples:
        raise RuntimeError("No samples resolved for visualization")

    print(f"Rendering feature flow for {len(samples)} {args.split} views...")
    flows = []
    for sample in tqdm(samples, desc="Feature flow"):
        flow = render_feature_flow(
            model,
            codec,
            renderer,
            sharpener,
            refiner,
            config,
            sample,
            is_hybrid,
        )
        flows.append(flow)
        if args.save_tensors:
            tensor_payload = {
                key: value
                for key, value in flow.items()
                if isinstance(value, torch.Tensor)
            }
            torch.save(tensor_payload, tensor_root / f"frame_{flow['label']}.pt")

    assign_pca_images(flows)

    grouped_collections: Dict[str, List[np.ndarray]] = {
        "latent_flow": [],
        "hybrid_flow": [],
        "radio_flow": [],
        "geometry_guides": [],
        "composite": [],
    }
    labels = []

    for flow in flows:
        labels.append(flow["label"])
        group_images = build_group_images(flow, scale=args.scale)
        for key, image in group_images.items():
            grouped_collections[key].append(image)
            if key == "composite":
                save_rgb(composite_root / f"frame_{flow['label']}.png", image)
            else:
                save_rgb(grouped_root / key / f"frame_{flow['label']}.png", image)

    for key, images in grouped_collections.items():
        grid = build_group_grid(images, labels, border=4)
        if key == "composite":
            save_rgb(composite_root / "grid.png", grid)
        else:
            save_rgb(grouped_root / key / "grid.png", grid)

    manifest = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "output_dir": str(out_root.resolve()),
        "split": args.split,
        "n_views": len(flows),
        "scale": args.scale,
        "save_tensors": args.save_tensors,
        "is_hybrid": is_hybrid,
        "samples": [
            {
                "label": flow["label"],
                "frame_id": int(flow["frame_id"]),
            }
            for flow in flows
        ],
        "stages": {
            "latent_flow": ["input_rgb", "rendered_latent", "featsharp_latent", "refined_latent"],
            "hybrid_flow": [
                "position_map",
                "fine_branch",
                "coarse_branch",
                "geometry_branch",
                "semantic_branch",
                "fused_hybrid",
            ],
            "radio_flow": ["decoded_1280", "gt_radio_1280", "decoded_gt_cosine"],
            "geometry_guides": ["render_depth", "alpha_map", "refiner_guide", "geometry_gate", "semantic_gate"],
        },
    }
    with open(out_root / "feature_flow_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("Saved feature-flow visualizations:")
    print(f"  grouped:   {grouped_root}")
    print(f"  composite: {composite_root}")
    if args.save_tensors:
        print(f"  tensors:   {tensor_root}")


if __name__ == "__main__":
    main()
