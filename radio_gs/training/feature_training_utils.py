"""Data and geometry utilities for RADIO-GS feature-field training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from radio_gs.data.benchmark_paths import (
    extract_feature_frame_index,
    list_feature_paths,
    load_w2c_from_pose_dir,
    load_w2c_from_pose_file,
    resolve_dataset_type,
    resolve_depth_path,
    resolve_rgb_path,
    resolve_semantics_path,
)
from radio_gs.training.tensor_cache_io import load_training_tensor_cache


class FoundationFeatureMapProjector(nn.Module):
    """Apply a frozen token projector to decoded `[B,C,H,W]` feature maps."""

    def __init__(self, projector: nn.Module) -> None:
        super().__init__()
        self.projector = projector

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                f"FoundationFeatureMapProjector expects [B,C,H,W], got {tuple(features.shape)}"
            )
        tokens = features.flatten(2).transpose(1, 2)
        return self.projector(tokens)

class FoundationMaskLogitProjector(nn.Module):
    """Small trainable probe for official mask-logit distillation caches."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_masks: int = 32,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if output_masks <= 0:
            raise ValueError("output_masks must be positive")
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_masks, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                f"FoundationMaskLogitProjector expects [B,C,H,W], got {tuple(features.shape)}"
            )
        return self.net(features)

def resolve_foundation_cache_path(root: str | Path, frame_id: int | str) -> Optional[Path]:
    """Resolve an optional per-frame official foundation cache file."""
    raw_root = str(root).strip()
    if not raw_root:
        return None
    root_path = Path(raw_root).expanduser()
    if root_path.is_file():
        return root_path
    if not root_path.exists():
        return None
    frame_raw = str(frame_id)
    candidates = [
        f"{frame_raw}.pt",
        f"frame_{frame_raw}.pt",
        f"rgb_{frame_raw}.pt",
    ]
    try:
        frame_int = int(frame_id)
    except (TypeError, ValueError):
        frame_int = None
    if frame_int is not None:
        candidates.extend(
            [
                f"{frame_int:06d}.pt",
                f"frame_{frame_int:06d}.pt",
                f"rgb_{frame_int:06d}.pt",
            ]
        )
    for name in candidates:
        path = root_path / name
        if path.exists():
            return path
    return None

def parse_direct_point_text_splits(raw: str | None, default_split: str) -> list[str]:
    """Parse a comma/space separated ScanNet text split list."""
    if raw is None or str(raw).strip() == "":
        values = [str(default_split)]
    else:
        values = str(raw).replace(",", " ").split()
    splits: list[str] = []
    for split in values:
        split = str(split).strip()
        if split and split not in splits:
            splits.append(split)
    return splits

def parse_radio_adaptor_names(raw: str | None) -> list[str]:
    """Parse comma/space separated RADIO adaptor names."""
    if raw is None or str(raw).strip() == "":
        return []
    names: list[str] = []
    for name in str(raw).replace(",", " ").split():
        name = name.strip()
        if name and name not in names:
            names.append(name)
    return names

def merge_radio_adaptor_names(*groups: list[str]) -> list[str]:
    """Merge adaptor-name groups while preserving first-seen order."""
    merged: list[str] = []
    for group in groups:
        for name in group:
            if name and name not in merged:
                merged.append(name)
    return merged

def read_ply_xyz(path: str | Path) -> torch.Tensor:
    """Read vertex xyz coordinates from a PLY file."""
    xyz, _ = read_ply_xyz_labels(path)
    return xyz

def read_ply_xyz_labels(path: str | Path) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Read vertex xyz coordinates and optional raw labels from a PLY file."""
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    missing = [name for name in ("x", "y", "z") if name not in vertex.data.dtype.names]
    if missing:
        raise ValueError(f"PLY is missing vertex coordinate fields {missing}: {path}")
    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )
    labels = None
    if "label" in vertex.data.dtype.names:
        labels_np = np.asarray(vertex["label"], dtype=np.int64)
        labels = torch.from_numpy(labels_np.copy()).long()
    return torch.from_numpy(xyz.copy()).float(), labels

def resolve_scannet_label_ply(scene_root: str | Path, scene: str | None = None) -> Path:
    """Resolve the OpenGaussian/ScanNet label mesh PLY used for point eval."""
    root = Path(scene_root)
    scene_name = scene or root.name
    preferred = root / f"{scene_name}_vh_clean_2.labels.ply"
    if preferred.exists():
        return preferred
    matches = sorted(root.glob("*.labels.ply"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No *.labels.ply file found in {root}")

def sample_multiview_radio_targets(
    points_xyz: torch.Tensor,
    gt_features: torch.Tensor,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
    *,
    depth_map: Optional[torch.Tensor] = None,
    alpha_map: Optional[torch.Tensor] = None,
    depth_tolerance: float = 0.08,
    relative_depth_tolerance: float = 0.02,
    alpha_threshold: float = 0.0,
    normalize_sampled_features: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample multi-view RADIO targets for world-space points.

    Args:
        points_xyz: ``[N, 3]`` world-space points.
        gt_features: ``[B, C, H, W]`` feature maps.
        pose_w2c: ``[B, 4, 4]`` world-to-camera matrices.
        K: feature-resolution camera intrinsics.
        depth_map: optional ``[B, H, W]`` depth visibility map.
        alpha_map: optional ``[B, H, W]`` opacity visibility map.
        normalize_sampled_features: L2-normalize each valid per-view sampled
            feature before multi-view averaging.

    Returns:
        target features ``[N, C]``, valid mask ``[N]``, and valid view counts
        ``[N]``.
    """
    if points_xyz.dim() != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected points_xyz [N,3], got {tuple(points_xyz.shape)}")
    if gt_features.dim() != 4:
        raise ValueError(f"Expected gt_features [B,C,H,W], got {tuple(gt_features.shape)}")
    if pose_w2c.dim() != 3 or pose_w2c.shape[-2:] != (4, 4):
        raise ValueError(f"Expected pose_w2c [B,4,4], got {tuple(pose_w2c.shape)}")

    n_points = points_xyz.shape[0]
    batch_size, channels, height, width = gt_features.shape
    if n_points == 0:
        return (
            torch.empty(0, channels, device=gt_features.device, dtype=gt_features.dtype),
            torch.empty(0, device=gt_features.device, dtype=torch.bool),
            torch.empty(0, device=gt_features.device, dtype=torch.long),
        )
    if pose_w2c.shape[0] != batch_size:
        raise ValueError(
            f"pose_w2c batch ({pose_w2c.shape[0]}) does not match features ({batch_size})"
        )

    device = gt_features.device
    points = points_xyz.to(device=device, dtype=torch.float32)
    poses = pose_w2c.to(device=device, dtype=torch.float32)
    intrinsics = K.to(device=device, dtype=torch.float32)

    ones = torch.ones(n_points, 1, device=device, dtype=torch.float32)
    points_h = torch.cat([points, ones], dim=1)
    cam = torch.einsum("bij,nj->bni", poses, points_h)
    z = cam[..., 2]
    z_safe = z.clamp_min(1e-6)
    u = intrinsics[0, 0] * (cam[..., 0] / z_safe) + intrinsics[0, 2]
    v = intrinsics[1, 1] * (cam[..., 1] / z_safe) + intrinsics[1, 2]

    valid = (
        (z > 1e-6)
        & (u >= 0.0)
        & (u <= float(width - 1))
        & (v >= 0.0)
        & (v <= float(height - 1))
    )

    if width > 1:
        grid_x = 2.0 * u / float(width - 1) - 1.0
    else:
        grid_x = torch.zeros_like(u)
    if height > 1:
        grid_y = 2.0 * v / float(height - 1) - 1.0
    else:
        grid_y = torch.zeros_like(v)
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(batch_size, n_points, 1, 2)

    if depth_map is not None:
        depth = depth_map.to(device=device, dtype=torch.float32)
        if depth.dim() == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        if depth.shape != (batch_size, height, width):
            depth = F.interpolate(
                depth[:, None],
                size=(height, width),
                mode="bilinear",
                align_corners=True,
            )[:, 0]
        sampled_depth = F.grid_sample(
            depth[:, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0, :, 0]
        tolerance = torch.maximum(
            torch.full_like(z, float(depth_tolerance)),
            z.abs() * float(relative_depth_tolerance),
        )
        valid = valid & (sampled_depth > 0.0) & ((sampled_depth - z).abs() <= tolerance)

    if alpha_map is not None and alpha_threshold > 0:
        alpha = alpha_map.to(device=device, dtype=torch.float32)
        if alpha.dim() == 4 and alpha.shape[1] == 1:
            alpha = alpha[:, 0]
        if alpha.shape != (batch_size, height, width):
            alpha = F.interpolate(
                alpha[:, None],
                size=(height, width),
                mode="bilinear",
                align_corners=True,
            )[:, 0]
        sampled_alpha = F.grid_sample(
            alpha[:, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0, :, 0]
        valid = valid & (sampled_alpha >= float(alpha_threshold))

    sampled = F.grid_sample(
        gt_features.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, :, :, 0].permute(2, 0, 1)
    if normalize_sampled_features:
        sampled = F.normalize(sampled.float(), dim=-1)
    valid_points_views = valid.T
    view_counts = valid_points_views.sum(dim=1)
    weights = valid_points_views.to(sampled.dtype).unsqueeze(-1)
    targets = (sampled * weights).sum(dim=1)
    denom = view_counts.clamp_min(1).to(sampled.dtype).unsqueeze(-1)
    targets = targets / denom
    return targets.to(dtype=gt_features.dtype), view_counts > 0, view_counts

def select_visible_gaussian_indices(
    points_xyz: torch.Tensor,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    sample_count: int,
    depth_map: Optional[torch.Tensor] = None,
    alpha_map: Optional[torch.Tensor] = None,
    depth_tolerance: float = 0.08,
    relative_depth_tolerance: float = 0.02,
    alpha_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select point indices visible in the current training views.

    The returned indices are relative to ``points_xyz``.  If fewer than
    ``sample_count`` points are visible, all visible points are returned.
    """
    if points_xyz.dim() != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected points_xyz [N,3], got {tuple(points_xyz.shape)}")
    if pose_w2c.dim() != 3 or pose_w2c.shape[-2:] != (4, 4):
        raise ValueError(f"Expected pose_w2c [B,4,4], got {tuple(pose_w2c.shape)}")
    if sample_count <= 0 or points_xyz.shape[0] == 0:
        empty = torch.empty(0, device=points_xyz.device, dtype=torch.long)
        visible = torch.zeros(points_xyz.shape[0], device=points_xyz.device, dtype=torch.bool)
        return empty, visible

    device = points_xyz.device
    points = points_xyz.to(device=device, dtype=torch.float32)
    poses = pose_w2c.to(device=device, dtype=torch.float32)
    intrinsics = K.to(device=device, dtype=torch.float32)
    batch_size = poses.shape[0]
    n_points = points.shape[0]
    height = int(image_height)
    width = int(image_width)

    ones = torch.ones(n_points, 1, device=device, dtype=torch.float32)
    points_h = torch.cat([points, ones], dim=1)
    cam = torch.einsum("bij,nj->bni", poses, points_h)
    z = cam[..., 2]
    z_safe = z.clamp_min(1e-6)
    u = intrinsics[0, 0] * (cam[..., 0] / z_safe) + intrinsics[0, 2]
    v = intrinsics[1, 1] * (cam[..., 1] / z_safe) + intrinsics[1, 2]

    visible = (
        (z > 1e-6)
        & (u >= 0.0)
        & (u <= float(width - 1))
        & (v >= 0.0)
        & (v <= float(height - 1))
    )

    if width > 1:
        grid_x = 2.0 * u / float(width - 1) - 1.0
    else:
        grid_x = torch.zeros_like(u)
    if height > 1:
        grid_y = 2.0 * v / float(height - 1) - 1.0
    else:
        grid_y = torch.zeros_like(v)
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(batch_size, n_points, 1, 2)

    if depth_map is not None:
        depth = depth_map.to(device=device, dtype=torch.float32)
        if depth.dim() == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        if depth.shape != (batch_size, height, width):
            depth = F.interpolate(
                depth[:, None],
                size=(height, width),
                mode="bilinear",
                align_corners=True,
            )[:, 0]
        sampled_depth = F.grid_sample(
            depth[:, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0, :, 0]
        tolerance = torch.maximum(
            torch.full_like(z, float(depth_tolerance)),
            z.abs() * float(relative_depth_tolerance),
        )
        visible = visible & (sampled_depth > 0.0) & ((sampled_depth - z).abs() <= tolerance)

    if alpha_map is not None and alpha_threshold > 0:
        alpha = alpha_map.to(device=device, dtype=torch.float32)
        if alpha.dim() == 4 and alpha.shape[1] == 1:
            alpha = alpha[:, 0]
        if alpha.shape != (batch_size, height, width):
            alpha = F.interpolate(
                alpha[:, None],
                size=(height, width),
                mode="bilinear",
                align_corners=True,
            )[:, 0]
        sampled_alpha = F.grid_sample(
            alpha[:, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0, :, 0]
        visible = visible & (sampled_alpha >= float(alpha_threshold))

    point_visible = visible.any(dim=0)
    visible_indices = torch.nonzero(point_visible, as_tuple=False).flatten()
    if visible_indices.numel() > sample_count:
        order = torch.randperm(visible_indices.numel(), device=device)[:sample_count]
        visible_indices = visible_indices[order]
    return visible_indices, point_visible

class SimpleRadioDataset(Dataset):
    """Loads pre-extracted RADIO features + poses for distillation training."""

    def __init__(
        self,
        feature_dir: str,
        pose_file: Optional[str] = None,
        pose_dir: Optional[str] = None,
        depth_dir: Optional[str] = None,
        semantics_dir: Optional[str] = None,
        rgb_dir: Optional[str] = None,
        feature_size: Optional[tuple] = None,
        split: str = "train",
        dataset_type: str = "replica",
        frame_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        self.feature_dir = Path(feature_dir)
        self.pose_file = Path(pose_file) if pose_file else None
        self.pose_dir = Path(pose_dir) if pose_dir else None
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.semantics_dir = Path(semantics_dir) if semantics_dir else None
        self.rgb_dir = Path(rgb_dir) if rgb_dir else None
        self.feature_size = feature_size  # (H, W) for downsampling RGB
        self.split = split
        self.dataset_type = resolve_dataset_type(dataset_type)
        self.frame_filter = {int(fid) for fid in frame_ids} if frame_ids is not None else None

        # --- discover feature files (backbone/rgb_{idx}.pt) ---------------
        self.feature_paths = list_feature_paths(self.feature_dir, frame_ids=frame_ids)
        assert len(self.feature_paths) > 0, (
            f"No feature files found in {self.feature_dir}"
        )
        self.frame_indices = [extract_feature_frame_index(path) for path in self.feature_paths]

        # --- load poses (traj_w_c.txt: one 4x4 c2w per line) --------------
        self.poses_w2c = self._load_poses()
        assert len(self.poses_w2c) == len(self.feature_paths), (
            f"Pose count ({len(self.poses_w2c)}) does not match features "
            f"({len(self.feature_paths)})"
        )

    # ------------------------------------------------------------------
    def _load_poses(self) -> np.ndarray:
        """Load poses from a flat traj file or a per-frame pose directory."""
        if self.pose_dir is not None:
            return load_w2c_from_pose_dir(self.pose_dir, self.frame_indices)
        if self.pose_file is None:
            raise ValueError("Either pose_file or pose_dir must be provided")
        return load_w2c_from_pose_file(self.pose_file, self.frame_indices)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.feature_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        radio_feat = load_training_tensor_cache(
            self.feature_paths[idx],
            map_location="cpu",
            purpose="RADIO feature cache",
        )  # [C, Hp, Wp]
        if radio_feat.dim() == 4:
            radio_feat = radio_feat.squeeze(0)

        # Upsample features if target resolution exceeds native resolution
        if self.feature_size is not None:
            tgt_h, tgt_w = self.feature_size
            _, cur_h, cur_w = radio_feat.shape
            if tgt_h > cur_h or tgt_w > cur_w:
                radio_feat = F.interpolate(
                    radio_feat.float().unsqueeze(0),
                    size=(tgt_h, tgt_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).half()

        frame_idx = self.frame_indices[idx]
        pose_w2c = torch.from_numpy(self.poses_w2c[idx])  # [4, 4]

        depth: Optional[torch.Tensor] = None
        depth_path = resolve_depth_path(self.depth_dir, frame_idx, self.dataset_type)
        if depth_path is not None and depth_path.exists():
            import cv2

            d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if d is not None:
                depth = torch.from_numpy(d.astype(np.float32) / 1000.0).clone()

        semantics: Optional[torch.Tensor] = None
        sem_path = resolve_semantics_path(self.semantics_dir, frame_idx, self.dataset_type)
        if sem_path is not None and sem_path.exists():
            from PIL import Image

            with Image.open(sem_path) as sem_img:
                sem = np.array(sem_img, dtype=np.int64)
            semantics = torch.from_numpy(sem).clone()

        # --- optional RGB guide (downsampled to feature resolution) --------
        rgb_guide: Optional[torch.Tensor] = None
        if self.rgb_dir is not None:
            import cv2

            rgb_path = resolve_rgb_path(self.rgb_dir, frame_idx, self.dataset_type)
            if rgb_path is not None and rgb_path.exists():
                img = cv2.imread(str(rgb_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if self.feature_size is not None:
                        img = cv2.resize(img, (self.feature_size[1], self.feature_size[0]))
                    rgb_guide = torch.from_numpy(img.copy()).float().permute(2, 0, 1) / 255.0

        out: Dict[str, torch.Tensor] = {
            "radio_features": radio_feat,
            "pose_w2c": pose_w2c,
            "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
        }
        if depth is not None:
            out["depth"] = depth
        if semantics is not None:
            out["semantics"] = semantics
        if rgb_guide is not None:
            out["rgb_guide"] = rgb_guide
        return out
