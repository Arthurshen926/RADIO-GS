"""
Hybrid DCFF-style Feature Gaussian (Architecture B)
====================================================
Combines per-Gaussian compact latent codes (rendered via alpha-blending) with a
multi-resolution spatial hash grid (queried by 3D position) to reconstruct
high-dimensional RADIO features through screen-space decoding.

Components:
    SpatialHashField   – pure-PyTorch multi-resolution hash encoding + MLP
    FineDecoder        – 1×1 Conv decoder for per-Gaussian latent maps
    CoarseDecoder      – 1×1 Conv decoder for hash-grid feature maps
    FusionHead         – fuses fine + coarse streams into output features
    HybridFeatureGaussian – full model with frozen geometry + learnable latent/hash/decoders
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement


# ---------------------------------------------------------------------------
# Multi-resolution spatial hash encoding (pure PyTorch, no CUDA extensions)
# ---------------------------------------------------------------------------

class SpatialHashField(nn.Module):
    """Multi-resolution hash grid encoding followed by a small MLP.

    Each resolution level maintains a learnable embedding table indexed by a
    spatial hash of voxel-corner coordinates.  Trilinear interpolation within
    each cell produces a per-level feature; all levels are concatenated and
    decoded by an MLP.
    """

    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 48,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        max_resolution: int = 2048,
        hidden_dim: int = 64,
        num_mlp_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2 ** log2_hashmap_size
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution

        # Per-level growth factor: geometric spacing from base to max resolution
        if num_levels > 1:
            self.growth_factor = math.exp(
                math.log(max_resolution / base_resolution) / (num_levels - 1)
            )
        else:
            self.growth_factor = 1.0

        # Pre-compute integer resolution for each level
        resolutions: List[int] = []
        for l in range(num_levels):
            res = int(math.floor(base_resolution * (self.growth_factor ** l)))
            resolutions.append(max(res, 1))
        self.register_buffer(
            "_resolutions", torch.tensor(resolutions, dtype=torch.long)
        )

        # Learnable hash tables — one per level
        self.hash_tables = nn.ParameterList([
            nn.Parameter(torch.empty(self.hashmap_size, features_per_level))
            for _ in range(num_levels)
        ])
        for table in self.hash_tables:
            nn.init.normal_(table, mean=0.0, std=0.01)

        # Large primes for the spatial hash function
        self.register_buffer(
            "_primes",
            torch.tensor([1, 2654435761, 805459861], dtype=torch.long),
        )

        # MLP: hash_features → output_dim
        encoding_dim = num_levels * features_per_level
        layers: list[nn.Module] = []
        in_dim = encoding_dim
        for i in range(num_mlp_layers):
            out = output_dim if i == num_mlp_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out))
            if i < num_mlp_layers - 1:
                layers.append(nn.ReLU(inplace=True))
            in_dim = out
        self.mlp = nn.Sequential(*layers)

    # -- hash helpers -------------------------------------------------------

    def _hash_coords(self, int_coords: torch.Tensor) -> torch.Tensor:
        """Spatial hash of integer grid coordinates.

        Args:
            int_coords: [N, 8, 3]  (8 corners of a voxel, integer coords)
        Returns:
            indices: [N, 8] indices into the hash table
        """
        # XOR-multiply with primes then mod table size
        hashed = int_coords[..., 0] * self._primes[0]
        hashed = hashed ^ (int_coords[..., 1] * self._primes[1])
        hashed = hashed ^ (int_coords[..., 2] * self._primes[2])
        return hashed % self.hashmap_size

    def _encode_level(
        self, positions: torch.Tensor, level: int
    ) -> torch.Tensor:
        """Trilinearly interpolated hash-grid lookup for one level.

        Args:
            positions: [N, 3] in [0, 1] normalised scene coordinates
        Returns:
            features: [N, features_per_level]
        """
        res = self._resolutions[level].item()
        # Continuous voxel coordinates
        pos_scaled = positions * res  # [N, 3]
        # Floor integer coordinates (clamp for safety)
        pos_floor = torch.floor(pos_scaled).long()  # [N, 3]
        # Fractional part for trilinear weights
        frac = pos_scaled - pos_floor.float()  # [N, 3]

        # 8 corner offsets of the unit cube
        offsets = positions.new_tensor(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
             [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
            dtype=torch.long,
        )  # [8, 3]

        corners = pos_floor.unsqueeze(1) + offsets.unsqueeze(0)  # [N, 8, 3]
        indices = self._hash_coords(corners)  # [N, 8]

        # Gather embeddings from the hash table
        table = self.hash_tables[level]  # [T, F]
        corner_feats = table[indices.clamp(0, self.hashmap_size - 1)]  # [N, 8, F]

        # Trilinear interpolation weights
        wx = frac[:, 0:1]  # [N, 1]
        wy = frac[:, 1:2]
        wz = frac[:, 2:3]

        # Interpolate along z
        c00 = corner_feats[:, 0] * (1 - wz) + corner_feats[:, 1] * wz
        c01 = corner_feats[:, 2] * (1 - wz) + corner_feats[:, 3] * wz
        c10 = corner_feats[:, 4] * (1 - wz) + corner_feats[:, 5] * wz
        c11 = corner_feats[:, 6] * (1 - wz) + corner_feats[:, 7] * wz
        # Interpolate along y
        c0 = c00 * (1 - wy) + c01 * wy
        c1 = c10 * (1 - wy) + c11 * wy
        # Interpolate along x
        feat = c0 * (1 - wx) + c1 * wx  # [N, F]
        return feat

    # -- public API ---------------------------------------------------------

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Encode 3-D positions through the multi-resolution hash grid + MLP.

        Args:
            positions: [N, 3] normalised to [0, 1]
        Returns:
            features: [N, output_dim]
        """
        level_feats = [self._encode_level(positions, l) for l in range(self.num_levels)]
        encoded = torch.cat(level_feats, dim=-1)  # [N, num_levels * features_per_level]
        return self.mlp(encoded)

    def forward_screen_space(self, position_map: torch.Tensor) -> torch.Tensor:
        """Hash-grid query for a dense position map.

        Args:
            position_map: [B, 3, H, W] world-space 3-D positions
        Returns:
            features: [B, output_dim, H, W]
        """
        B, C, H, W = position_map.shape
        assert C == 3, f"Expected 3-channel position map, got {C}"
        # Reshape to (B*H*W, 3)
        pos_flat = position_map.permute(0, 2, 3, 1).reshape(-1, 3)
        feat_flat = self.forward(pos_flat)  # [B*H*W, output_dim]
        return feat_flat.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Screen-space decoders (1×1 Conv)
# ---------------------------------------------------------------------------

class FineDecoder(nn.Module):
    """Decode per-Gaussian rendered latent map to fine features."""

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 64, fine_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, fine_dim, 1),
        )

    def forward(self, latent_map: torch.Tensor) -> torch.Tensor:
        """[B, latent_dim, H, W] → [B, fine_dim, H, W]"""
        return self.net(latent_map)


class CoarseDecoder(nn.Module):
    """Decode hash-grid features to coarse features."""

    def __init__(self, hash_output_dim: int = 48, hidden_dim: int = 64, coarse_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hash_output_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, coarse_dim, 1),
        )

    def forward(self, hash_feat: torch.Tensor) -> torch.Tensor:
        """[B, hash_output_dim, H, W] → [B, coarse_dim, H, W]"""
        return self.net(hash_feat)


class FusionHead(nn.Module):
    """Fuse fine + coarse feature streams with adaptive gating."""

    def __init__(self, fine_dim: int = 64, coarse_dim: int = 64, hidden_dim: int = 128, output_dim: int = 128):
        super().__init__()
        in_dim = fine_dim + coarse_dim
        # Adaptive gate: learns per-pixel weighting of fine vs coarse
        self.gate = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 2, 1),
            nn.Softmax(dim=1),
        )
        # Main fusion pathway (deeper than V1)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, 1),
        )

    def forward(self, fine_feat: torch.Tensor, coarse_feat: torch.Tensor) -> torch.Tensor:
        """[B, fine_dim, H, W] + [B, coarse_dim, H, W] → [B, output_dim, H, W]"""
        concat = torch.cat([fine_feat, coarse_feat], dim=1)
        weights = self.gate(concat)  # [B, 2, H, W]
        # Scale streams by learned weights (additive residual to avoid vanishing)
        fine_weighted = fine_feat * (weights[:, 0:1] + 1.0)
        coarse_weighted = coarse_feat * (weights[:, 1:2] + 1.0)
        fused_input = torch.cat([fine_weighted, coarse_weighted], dim=1)
        return self.fuse(fused_input)


class DecoupledFusionHead(nn.Module):
    """Explicit geometry/semantic heads before final feature fusion."""

    class SemanticFilterAdaptor(nn.Module):
        """LESV-style semantic reliability filtering before final fusion."""

        def __init__(
            self,
            feat_dim: int,
            hidden_dim: int = 64,
            mode: str = "confidence",
            use_geometry_guidance: bool = True,
            use_depth_guidance: bool = False,
            residual: bool = True,
        ):
            super().__init__()
            if mode not in {"confidence", "refinement"}:
                raise ValueError(f"Unsupported semantic adaptor mode: {mode}")
            self.mode = mode
            self.use_geometry_guidance = use_geometry_guidance
            self.use_depth_guidance = use_depth_guidance
            self.residual = residual

            extra_ch = 0
            if use_geometry_guidance:
                extra_ch += 1
            if use_depth_guidance:
                extra_ch += 1

            self.confidence_net = nn.Sequential(
                nn.Conv2d(feat_dim + extra_ch, hidden_dim, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, 1, 1),
            )
            nn.init.normal_(self.confidence_net[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.confidence_net[-1].bias)

            if self.mode == "refinement":
                self.refinement_net = nn.Sequential(
                    nn.Conv2d(feat_dim + 1, hidden_dim, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, feat_dim, 1),
                )
                nn.init.normal_(self.refinement_net[-1].weight, mean=0.0, std=1e-3)
                nn.init.zeros_(self.refinement_net[-1].bias)
            else:
                self.refinement_net = None

        @staticmethod
        def _normalize_map(x: torch.Tensor) -> torch.Tensor:
            dims = tuple(range(2, x.dim()))
            min_v = x.amin(dim=dims, keepdim=True)
            max_v = x.amax(dim=dims, keepdim=True)
            return (x - min_v) / (max_v - min_v + 1e-6)

        def forward(
            self,
            semantic_feat: torch.Tensor,
            geometry_feat: Optional[torch.Tensor] = None,
            depth_map: Optional[torch.Tensor] = None,
        ) -> dict[str, torch.Tensor]:
            feat_dtype = semantic_feat.dtype
            feat_float = semantic_feat.float()
            conf_inputs = [feat_float]

            if self.use_geometry_guidance and geometry_feat is not None:
                geom_norm = geometry_feat.float().norm(dim=1, keepdim=True)
                conf_inputs.append(self._normalize_map(geom_norm))

            if self.use_depth_guidance and depth_map is not None:
                depth = depth_map.float()
                if depth.dim() == 3:
                    depth = depth.unsqueeze(1)
                if depth.shape[-2:] != semantic_feat.shape[-2:]:
                    depth = F.interpolate(
                        depth,
                        size=semantic_feat.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                conf_inputs.append(self._normalize_map(depth))

            confidence = 2.0 * torch.sigmoid(
                self.confidence_net(torch.cat(conf_inputs, dim=1))
            )
            semantic_filtered = semantic_feat * confidence.to(dtype=feat_dtype)

            result = {
                "semantic_filtered": semantic_filtered,
                "semantic_confidence": confidence,
            }
            if self.refinement_net is not None:
                refinement = self.refinement_net(
                    torch.cat([semantic_filtered.float(), confidence], dim=1)
                ).to(dtype=feat_dtype)
                semantic_filtered = (
                    semantic_filtered + refinement if self.residual else refinement
                )
                result["semantic_filtered"] = semantic_filtered
                result["semantic_refinement"] = refinement
            return result

    def __init__(
        self,
        fine_dim: int = 64,
        coarse_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 128,
        use_semantic_adaptor: bool = False,
        semantic_adaptor_mode: str = "confidence",
        semantic_adaptor_hidden_dim: int = 64,
        semantic_adaptor_use_geometry_guidance: bool = True,
        semantic_adaptor_use_depth_guidance: bool = False,
        semantic_adaptor_residual: bool = True,
    ):
        super().__init__()
        in_dim = fine_dim + coarse_dim
        gate_hidden = max(hidden_dim // 2, 32)
        self.geometry_gate = nn.Sequential(
            nn.Conv2d(in_dim, gate_hidden, 1),
            nn.GELU(),
            nn.Conv2d(gate_hidden, 1, 1),
            nn.Sigmoid(),
        )
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(in_dim, gate_hidden, 1),
            nn.GELU(),
            nn.Conv2d(gate_hidden, 1, 1),
            nn.Sigmoid(),
        )
        self.geometry_head = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, 1),
        )
        self.semantic_head = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, 1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(output_dim * 2, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, 1),
        )
        self.semantic_adaptor = (
            self.SemanticFilterAdaptor(
                feat_dim=output_dim,
                hidden_dim=semantic_adaptor_hidden_dim,
                mode=semantic_adaptor_mode,
                use_geometry_guidance=semantic_adaptor_use_geometry_guidance,
                use_depth_guidance=semantic_adaptor_use_depth_guidance,
                residual=semantic_adaptor_residual,
            )
            if use_semantic_adaptor
            else None
        )

    def forward(
        self,
        fine_feat: torch.Tensor,
        coarse_feat: torch.Tensor,
        return_aux: bool = False,
        depth_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        concat = torch.cat([fine_feat, coarse_feat], dim=1)
        geom_gate = self.geometry_gate(concat)
        sem_gate = self.semantic_gate(concat)

        geometry_input = torch.cat([fine_feat, coarse_feat * geom_gate], dim=1)
        semantic_input = torch.cat([coarse_feat, fine_feat * sem_gate], dim=1)
        geometry_feat = self.geometry_head(geometry_input)
        semantic_feat = self.semantic_head(semantic_input)
        adaptor_aux: dict[str, torch.Tensor] = {}
        if self.semantic_adaptor is not None:
            adaptor_result = self.semantic_adaptor(
                semantic_feat,
                geometry_feat=geometry_feat,
                depth_map=depth_map,
            )
            semantic_feat = adaptor_result["semantic_filtered"]
            adaptor_aux = {
                key: value
                for key, value in adaptor_result.items()
                if key != "semantic_filtered"
            }
        fused = self.fuse(torch.cat([geometry_feat, semantic_feat], dim=1))

        if return_aux:
            result = {
                "fused": fused,
                "geometry": geometry_feat,
                "semantic": semantic_feat,
                "geometry_gate": geom_gate,
                "semantic_gate": sem_gate,
            }
            result.update(adaptor_aux)
            return result
        return fused


# ---------------------------------------------------------------------------
# Utility: depth un-projection
# ---------------------------------------------------------------------------

def unproject_depth_to_positions(
    depth_map: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Convert rendered depth + camera parameters to world-space 3-D positions.

    Args:
        depth_map: [B, H, W] rendered depth in camera space
        viewmat:   [B, 4, 4] world-to-camera rigid transform
        K:         [3, 3] camera intrinsics (shared across batch)
        height:    image height  (must match depth_map.shape[1])
        width:     image width   (must match depth_map.shape[2])

    Returns:
        positions: [B, 3, H, W] world-space 3-D coordinates
    """
    device = depth_map.device
    B = depth_map.shape[0]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Pixel grid (shared across batch)
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )  # [H, W] each

    # Back-project to camera-space 3-D
    z = depth_map  # [B, H, W]
    x = (u.unsqueeze(0) - cx) / fx * z  # [B, H, W]
    y = (v.unsqueeze(0) - cy) / fy * z  # [B, H, W]

    pts_cam = torch.stack([x, y, z, torch.ones_like(z)], dim=1)  # [B, 4, H, W]

    # Camera-to-world: invert viewmat (rigid → transpose rotation, negate translation)
    R = viewmat[:, :3, :3]  # [B, 3, 3]
    t = viewmat[:, :3, 3:]  # [B, 3, 1]
    R_inv = R.transpose(1, 2)
    t_inv = -R_inv @ t  # [B, 3, 1]
    cam2world = torch.zeros(B, 4, 4, device=device, dtype=viewmat.dtype)
    cam2world[:, :3, :3] = R_inv
    cam2world[:, :3, 3:] = t_inv
    cam2world[:, 3, 3] = 1.0

    # Transform to world coordinates
    pts_flat = pts_cam.reshape(B, 4, -1)  # [B, 4, H*W]
    world_pts = (cam2world @ pts_flat)[:, :3]  # [B, 3, H*W]
    return world_pts.reshape(B, 3, height, width)


# ---------------------------------------------------------------------------
# Hybrid Feature Gaussian Model
# ---------------------------------------------------------------------------

class HybridFeatureGaussian(nn.Module):
    """Architecture B: Hybrid per-Gaussian latent + spatial hash grid.

    Frozen 3DGS geometry is loaded from a PLY file.  Two learnable pathways
    produce screen-space features that are fused into the final output:

        fine path:   rendered per-Gaussian latent  →  FineDecoder
        coarse path: 3-D position hash grid query  →  CoarseDecoder

    The two streams are concatenated and decoded by a FusionHead.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hash_output_dim: int = 48,
        fine_dim: int = 64,
        coarse_dim: int = 64,
        output_dim: int = 128,
        # SpatialHashField kwargs
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        max_resolution: int = 2048,
        hash_hidden_dim: int = 64,
        hash_mlp_layers: int = 2,
        # Decoder hidden dims
        fine_hidden_dim: int = 64,
        coarse_hidden_dim: int = 64,
        fusion_hidden_dim: int = 128,
        decoupled_heads: bool = False,
        use_semantic_adaptor: bool = False,
        semantic_adaptor_mode: str = "confidence",
        semantic_adaptor_hidden_dim: int = 64,
        semantic_adaptor_use_geometry_guidance: bool = True,
        semantic_adaptor_use_depth_guidance: bool = False,
        semantic_adaptor_residual: bool = True,
    ):
        super().__init__()
        self._latent_dim = latent_dim
        self._output_dim = output_dim
        self.decoupled_heads = decoupled_heads
        if use_semantic_adaptor and not decoupled_heads:
            raise ValueError(
                "Semantic adaptor requires hybrid_decoupled_heads=true"
            )

        # --- frozen geometry (populated by load_from_ply) ---
        self.register_buffer("_xyz", torch.empty(0))
        self.register_buffer("_rotation", torch.empty(0))
        self.register_buffer("_scaling", torch.empty(0))
        self.register_buffer("_opacity", torch.empty(0))
        self.register_buffer("_features_dc", torch.empty(0))
        self.register_buffer("_features_rest", torch.empty(0))
        self._sh_degree = 0

        # Activation helpers (match GaussianFeatureModel conventions)
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = F.normalize

        # --- learnable per-Gaussian latent codes ---
        self._latent = nn.Parameter(torch.empty(0))

        # --- spatial hash field ---
        self.hash_field = SpatialHashField(
            input_dim=3,
            output_dim=hash_output_dim,
            num_levels=num_levels,
            features_per_level=features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            hidden_dim=hash_hidden_dim,
            num_mlp_layers=hash_mlp_layers,
        )

        # --- screen-space decoders ---
        self.fine_decoder = FineDecoder(latent_dim, fine_hidden_dim, fine_dim)
        self.coarse_decoder = CoarseDecoder(hash_output_dim, coarse_hidden_dim, coarse_dim)
        if decoupled_heads:
            self.fusion_head = DecoupledFusionHead(
                fine_dim,
                coarse_dim,
                fusion_hidden_dim,
                output_dim,
                use_semantic_adaptor=use_semantic_adaptor,
                semantic_adaptor_mode=semantic_adaptor_mode,
                semantic_adaptor_hidden_dim=semantic_adaptor_hidden_dim,
                semantic_adaptor_use_geometry_guidance=semantic_adaptor_use_geometry_guidance,
                semantic_adaptor_use_depth_guidance=semantic_adaptor_use_depth_guidance,
                semantic_adaptor_residual=semantic_adaptor_residual,
            )
        else:
            self.fusion_head = FusionHead(fine_dim, coarse_dim, fusion_hidden_dim, output_dim)

    # -- accessors (match ExplicitFeatureGaussian API) ---------------------

    @property
    def num_gaussians(self) -> int:
        return self._xyz.shape[0]

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    def get_rotation(self) -> torch.Tensor:
        return self.rotation_activation(self._rotation, dim=-1)

    def get_scaling(self) -> torch.Tensor:
        return self.scaling_activation(self._scaling)

    def get_opacity(self) -> torch.Tensor:
        return self.opacity_activation(self._opacity)

    # -- latent accessors ---------------------------------------------------

    def get_latent(self) -> torch.Tensor:
        """Return per-Gaussian latent codes [N, latent_dim]."""
        return self._latent

    def get_features(self) -> torch.Tensor:
        """Return features used for rasterization (= latent codes)."""
        return self._latent

    def get_sh_colors(self) -> torch.Tensor:
        """Return SH coefficients [N, K, 3] for RGB rendering."""
        if self._features_rest.numel() > 0:
            return torch.cat([self._features_dc, self._features_rest], dim=1)
        return self._features_dc  # [N, 1, 3]

    # -- PLY I/O ------------------------------------------------------------

    def load_from_ply(self, ply_path: str) -> None:
        """Load pre-trained 3DGS geometry from PLY and freeze it.

        Initialises per-Gaussian latent codes with small random values.
        """
        print(f"[HybridFeatureGaussian] Loading PLY: {ply_path}")
        plydata = PlyData.read(ply_path)
        vertex = plydata.elements[0]
        N = vertex.count

        xyz = np.stack(
            [np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])],
            axis=1,
        )
        opacity = np.asarray(vertex["opacity"])[..., np.newaxis]
        features_dc = np.zeros((N, 1, 3))
        for i in range(3):
            features_dc[:, 0, i] = np.asarray(vertex[f"f_dc_{i}"])

        scale_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        scales = np.stack([np.asarray(vertex[n]) for n in scale_names], axis=1)

        rot_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith("rot")],
            key=lambda x: int(x.split("_")[-1]),
        )
        rots = np.stack([np.asarray(vertex[n]) for n in rot_names], axis=1)

        # Re-register as buffers so .to(device) moves them properly
        self.register_buffer("_xyz", torch.tensor(xyz, dtype=torch.float32))
        self.register_buffer("_rotation", torch.tensor(rots, dtype=torch.float32))
        self.register_buffer("_scaling", torch.tensor(scales, dtype=torch.float32))
        self.register_buffer("_opacity", torch.tensor(opacity, dtype=torch.float32))
        self.register_buffer("_features_dc", torch.tensor(features_dc, dtype=torch.float32))

        # Load SH rest coefficients for RGB rendering
        rest_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith("f_rest_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        if rest_names:
            features_rest = np.stack(
                [np.asarray(vertex[n]) for n in rest_names], axis=1
            )  # [N, K*3]
            num_sh_coeffs = features_rest.shape[1] // 3
            features_rest = features_rest.reshape(N, num_sh_coeffs, 3)
            self.register_buffer(
                "_features_rest", torch.tensor(features_rest, dtype=torch.float32)
            )
            # Infer SH degree from number of coefficients
            import math
            total_coeffs = num_sh_coeffs + 1  # +1 for DC
            self._sh_degree = int(math.sqrt(total_coeffs)) - 1
        else:
            self.register_buffer("_features_rest", torch.empty(0))
            self._sh_degree = 0

        # Initialise learnable latent codes
        latent = torch.randn(N, self._latent_dim) * 0.01
        self._latent = nn.Parameter(latent)

        print(f"  Gaussians: {N}, latent_dim: {self._latent_dim}")
        print(f"  Trainable latent params: {N * self._latent_dim:,}")

    # -- screen-space decoding ----------------------------------------------

    def decode_screen_space(
        self,
        latent_map: torch.Tensor,
        position_map: torch.Tensor,
        view_dirs: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        depth_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Decode rendered latent + 3-D position maps into output features.

        Args:
            latent_map:   [B, latent_dim, H, W] alpha-blended per-Gaussian latent
            position_map: [B, 3, H, W] world-space positions (from depth un-projection)
            view_dirs:    [B, 3, H, W] optional view directions (reserved for future use)

        Returns:
            features: [B, output_dim, H, W]
        """
        # Fine pathway: decode the rendered latent
        fine_feat = self.fine_decoder(latent_map)  # [B, fine_dim, H, W]

        # Coarse pathway: query hash field at 3-D positions
        hash_feat = self.hash_field.forward_screen_space(position_map)  # [B, hash_out, H, W]
        coarse_feat = self.coarse_decoder(hash_feat)  # [B, coarse_dim, H, W]

        # Fusion
        if self.decoupled_heads:
            return self.fusion_head(
                fine_feat,
                coarse_feat,
                return_aux=return_aux,
                depth_map=depth_map,
            )
        return self.fusion_head(fine_feat, coarse_feat)  # [B, output_dim, H, W]

    # -- direct 3-D point querying ------------------------------------------

    def scene_bounds(self, margin: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return padded world-space bounds inferred from loaded Gaussians."""
        xyz = self.get_xyz()
        if xyz.numel() == 0:
            raise RuntimeError("HybridFeatureGaussian geometry is empty; call load_from_ply first")
        lo = xyz.min(dim=0).values - margin
        hi = xyz.max(dim=0).values + margin
        return lo, hi

    def normalize_world_positions(
        self,
        positions: torch.Tensor,
        *,
        margin: float = 0.1,
        clamp: bool = True,
    ) -> torch.Tensor:
        """Normalize world-space positions to the hash field's ``[0, 1]`` cube.

        Supports both point tensors ``[N, 3]`` and dense maps ``[B, 3, H, W]``.
        The bounds match the training-time normalization used in
        ``train_feature_field.py``.
        """
        lo, hi = self.scene_bounds(margin=margin)
        lo = lo.to(device=positions.device, dtype=positions.dtype)
        hi = hi.to(device=positions.device, dtype=positions.dtype)
        extent = (hi - lo).clamp(min=1e-6)
        if positions.dim() == 2:
            normed = (positions - lo.view(1, 3)) / extent.view(1, 3)
        elif positions.dim() == 4:
            normed = (positions - lo.view(1, 3, 1, 1)) / extent.view(1, 3, 1, 1)
        else:
            raise ValueError(
                f"Expected positions shaped [N,3] or [B,3,H,W], got {tuple(positions.shape)}"
            )
        return normed.clamp(0.0, 1.0) if clamp else normed

    def _decode_point_features(
        self,
        latent_points: torch.Tensor,
        normalized_points: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Decode point-aligned latent/position tensors.

        Args:
            latent_points: ``[N, latent_dim]``.
            normalized_points: ``[N, 3]`` in ``[0, 1]``.
        """
        if latent_points.dim() != 2 or latent_points.shape[1] != self._latent_dim:
            raise ValueError(
                f"Expected latent_points [N,{self._latent_dim}], got {tuple(latent_points.shape)}"
            )
        if normalized_points.dim() != 2 or normalized_points.shape[1] != 3:
            raise ValueError(f"Expected normalized_points [N,3], got {tuple(normalized_points.shape)}")

        n_points = latent_points.shape[0]
        latent_map = latent_points.reshape(n_points, self._latent_dim, 1, 1)
        position_map = normalized_points.reshape(n_points, 3, 1, 1)
        decoded = self.decode_screen_space(
            latent_map.float(),
            position_map.float(),
            return_aux=return_aux,
        )
        if not return_aux:
            return decoded[:, :, 0, 0].contiguous()

        if torch.is_tensor(decoded):
            return {"features": decoded[:, :, 0, 0].contiguous()}

        assert isinstance(decoded, dict)
        out: dict[str, torch.Tensor] = {}
        for key, value in decoded.items():
            if (
                torch.is_tensor(value)
                and value.dim() == 4
                and value.shape[0] == n_points
                and value.shape[-2:] == (1, 1)
            ):
                out[key] = value[:, :, 0, 0].contiguous()
            else:
                out[key] = value
        if "fused" in out:
            out["features"] = out["fused"]
        return out

    def _knn_gaussians(
        self,
        points_xyz: torch.Tensor,
        k: int,
        *,
        point_chunk_size: int = 2048,
        gaussian_chunk_size: int = 16384,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Memory-bounded Euclidean KNN over loaded Gaussian centers."""
        xyz = self.get_xyz().to(device=points_xyz.device, dtype=points_xyz.dtype)
        n_gaussians = xyz.shape[0]
        if n_gaussians == 0:
            raise RuntimeError("HybridFeatureGaussian geometry is empty; call load_from_ply first")
        k = min(max(int(k), 1), n_gaussians)

        all_dists: list[torch.Tensor] = []
        all_indices: list[torch.Tensor] = []
        for p0 in range(0, points_xyz.shape[0], point_chunk_size):
            pts = points_xyz[p0 : p0 + point_chunk_size]
            best_d = torch.full(
                (pts.shape[0], k),
                float("inf"),
                device=points_xyz.device,
                dtype=points_xyz.dtype,
            )
            best_i = torch.full(
                (pts.shape[0], k),
                -1,
                device=points_xyz.device,
                dtype=torch.long,
            )
            for g0 in range(0, n_gaussians, gaussian_chunk_size):
                g1 = min(g0 + gaussian_chunk_size, n_gaussians)
                dist = torch.cdist(pts.float(), xyz[g0:g1].float()).to(points_xyz.dtype)
                local_k = min(k, g1 - g0)
                d_chunk, i_chunk = torch.topk(dist, k=local_k, largest=False, dim=1)
                i_chunk = i_chunk + g0
                cand_d = torch.cat([best_d, d_chunk], dim=1)
                cand_i = torch.cat([best_i, i_chunk], dim=1)
                best_d, order = torch.topk(cand_d, k=k, largest=False, dim=1)
                best_i = cand_i.gather(1, order)
                del dist
            all_dists.append(best_d)
            all_indices.append(best_i)
        return torch.cat(all_dists, dim=0), torch.cat(all_indices, dim=0)

    @staticmethod
    def _quaternion_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
        """Convert unit quaternions in ``w, x, y, z`` order to rotation matrices."""
        q = F.normalize(quat.float(), p=2, dim=-1)
        w, x, y, z = q.unbind(dim=-1)
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z
        return torch.stack(
            [
                ww + xx - yy - zz,
                2.0 * (xy - wz),
                2.0 * (xz + wy),
                2.0 * (xy + wz),
                ww - xx + yy - zz,
                2.0 * (yz - wx),
                2.0 * (xz - wy),
                2.0 * (yz + wx),
                ww - xx - yy + zz,
            ],
            dim=-1,
        ).reshape(*q.shape[:-1], 3, 3)

    def query_compact_points(
        self,
        points_xyz: torch.Tensor,
        k: int = 8,
        candidate_k: Optional[int] = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Query RADIO-GS compact features directly at world-space 3-D points.

        Neighbouring Gaussian latents are aggregated with an opacity-weighted,
        scale-aware Gaussian kernel, then fused with the hash-field coarse branch
        using the same decoders as screen-space rendering.

        Args:
            points_xyz: ``[N, 3]`` world-space point coordinates.
            k: Number of neighbouring Gaussians to aggregate.
            candidate_k: Optional Euclidean candidate count before pruning by
                opacity-weighted anisotropic Gaussian density.  Values larger
                than ``k`` let broad/high-density Gaussians compete even when
                they are outside the closest Euclidean ``k``.
            return_aux: Include KNN indices/weights and decoder auxiliaries.

        Returns:
            ``[N, output_dim]`` compact bottleneck features, or an aux dict.
        """
        if points_xyz.dim() != 2 or points_xyz.shape[1] != 3:
            raise ValueError(f"Expected points_xyz [N,3], got {tuple(points_xyz.shape)}")
        if self._latent.numel() == 0:
            raise RuntimeError("Per-Gaussian latent codes are empty; load a checkpoint first")
        if points_xyz.shape[0] == 0:
            empty_feat = torch.empty(
                0,
                self._output_dim,
                device=self._latent.device,
                dtype=self._latent.dtype,
            )
            if return_aux:
                return {
                    "features": empty_feat,
                    "gaussian_indices": torch.empty(
                        0,
                        0,
                        device=self._latent.device,
                        dtype=torch.long,
                    ),
                    "weights": torch.empty(
                        0,
                        0,
                        device=self._latent.device,
                        dtype=torch.float32,
                    ),
                    "normalized_points": torch.empty(
                        0,
                        3,
                        device=self._latent.device,
                        dtype=torch.float32,
                    ),
                    "latent": torch.empty(
                        0,
                        self._latent_dim,
                        device=self._latent.device,
                        dtype=self._latent.dtype,
                    ),
                    "euclidean_dist": torch.empty(
                        0,
                        0,
                        device=self._latent.device,
                        dtype=torch.float32,
                    ),
                    "mahalanobis_dist2": torch.empty(
                        0,
                        0,
                        device=self._latent.device,
                        dtype=torch.float32,
                    ),
                    "density": torch.empty(
                        0,
                        0,
                        device=self._latent.device,
                        dtype=torch.float32,
                    ),
                }
            return empty_feat

        device = self._latent.device
        dtype = self._latent.dtype if self._latent.is_floating_point() else torch.float32
        points = points_xyz.to(device=device, dtype=torch.float32)
        k = max(int(k), 1)
        candidate_k = k if candidate_k is None else max(int(candidate_k), k)
        _, candidate_idx = self._knn_gaussians(points, k=candidate_k)

        xyz = self.get_xyz().to(device=device, dtype=torch.float32)
        scales = self.get_scaling().to(device=device, dtype=torch.float32).clamp_min(1e-6)
        rotations = self.get_rotation().to(device=device, dtype=torch.float32)
        opacity = self.get_opacity().to(device=device, dtype=torch.float32).squeeze(-1)

        neigh_xyz = xyz[candidate_idx]              # [N, candidate_k, 3]
        neigh_scales = scales[candidate_idx]        # [N, candidate_k, 3]
        neigh_rot = rotations[candidate_idx]        # [N, candidate_k, 4]
        delta = points.unsqueeze(1) - neigh_xyz
        euclidean_dist = torch.linalg.norm(delta, dim=-1)
        rot_mats = self._quaternion_to_rotation_matrix(neigh_rot)
        local_delta = torch.einsum("nki,nkij->nkj", delta.float(), rot_mats)
        mahalanobis_dist2 = ((local_delta / neigh_scales) ** 2).sum(dim=-1)
        density = torch.exp(-0.5 * mahalanobis_dist2)
        raw_weights = density * opacity[candidate_idx].clamp_min(1e-6)
        if candidate_idx.shape[1] > k:
            selected_k = min(k, candidate_idx.shape[1])
            raw_weights, order = torch.topk(
                raw_weights,
                k=selected_k,
                largest=True,
                dim=1,
            )
            knn_idx = candidate_idx.gather(1, order)
            gather_vec = order.unsqueeze(-1).expand(-1, -1, 3)
            neigh_xyz = neigh_xyz.gather(1, gather_vec)
            neigh_scales = neigh_scales.gather(1, gather_vec)
            neigh_rot = neigh_rot.gather(1, order.unsqueeze(-1).expand(-1, -1, 4))
            euclidean_dist = euclidean_dist.gather(1, order)
            mahalanobis_dist2 = mahalanobis_dist2.gather(1, order)
            density = density.gather(1, order)
        else:
            knn_idx = candidate_idx

        weights = raw_weights
        weight_sum = weights.sum(dim=1, keepdim=True)
        uniform = torch.full_like(weights, 1.0 / weights.shape[1])
        weights = torch.where(weight_sum > 1e-12, weights / weight_sum.clamp_min(1e-12), uniform)

        latent = self.get_latent().to(device=device, dtype=torch.float32)
        latent_points = (latent[knn_idx] * weights.unsqueeze(-1)).sum(dim=1)
        normalized_points = self.normalize_world_positions(points)
        decoded = self._decode_point_features(
            latent_points.to(dtype=dtype),
            normalized_points,
            return_aux=return_aux,
        )

        if not return_aux:
            return decoded
        assert isinstance(decoded, dict)
        decoded.update(
            {
                "gaussian_indices": knn_idx,
                "weights": weights,
                "normalized_points": normalized_points,
                "latent": latent_points,
                "euclidean_dist": euclidean_dist,
                "mahalanobis_dist2": mahalanobis_dist2,
                "density": density,
            }
        )
        return decoded

    def query_gaussian_points(
        self,
        gaussian_indices: torch.Tensor,
        points_xyz: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Decode compact features for selected Gaussian rows.

        By default the hash/coarse branch is evaluated at the selected Gaussian
        centers.  ``points_xyz`` can override those coordinates while keeping
        the per-Gaussian latent indexed by row; this is useful for row-aligned
        ScanNet label vertices whose original point coordinates differ from the
        optimized Gaussian centers.
        """
        if self._latent.numel() == 0:
            raise RuntimeError("Per-Gaussian latent codes are empty; load a checkpoint first")
        indices = torch.as_tensor(gaussian_indices, device=self._latent.device, dtype=torch.long).reshape(-1)
        if indices.numel() == 0:
            empty_feat = torch.empty(
                0,
                self._output_dim,
                device=self._latent.device,
                dtype=self._latent.dtype,
            )
            if return_aux:
                return {"features": empty_feat, "gaussian_indices": indices}
            return empty_feat
        if int(indices.min()) < 0 or int(indices.max()) >= self.num_gaussians:
            raise IndexError(
                f"Gaussian index out of range for {self.num_gaussians} Gaussians"
            )

        latent_points = self.get_latent()[indices]
        centers = self.get_xyz().to(device=self._latent.device, dtype=torch.float32)[indices]
        if points_xyz is None:
            points = centers
        else:
            points = torch.as_tensor(
                points_xyz,
                device=self._latent.device,
                dtype=torch.float32,
            )
            if points.shape != centers.shape:
                raise ValueError(
                    "points_xyz must match gaussian_indices as [N,3], got "
                    f"{tuple(points.shape)} for {tuple(indices.shape)} indices"
                )
        normalized_points = self.normalize_world_positions(points)
        decoded = self._decode_point_features(
            latent_points,
            normalized_points,
            return_aux=return_aux,
        )
        if not return_aux:
            return decoded
        assert isinstance(decoded, dict)
        decoded.update(
            {
                "gaussian_indices": indices,
                "weights": torch.ones(indices.shape[0], 1, device=indices.device),
                "query_points": points,
                "gaussian_centers": centers,
                "normalized_points": normalized_points,
                "latent": latent_points,
                "euclidean_dist": torch.linalg.norm(points - centers, dim=-1, keepdim=True),
                "mahalanobis_dist2": torch.zeros(
                    indices.shape[0],
                    1,
                    device=indices.device,
                    dtype=torch.float32,
                ),
                "density": torch.ones(
                    indices.shape[0],
                    1,
                    device=indices.device,
                    dtype=torch.float32,
                ),
            }
        )
        return decoded

    # -- trainable parameters -----------------------------------------------

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Return all learnable parameters (latent + hash field + decoders)."""
        params: List[torch.nn.Parameter] = [self._latent]
        params.extend(self.hash_field.parameters())
        params.extend(self.fine_decoder.parameters())
        params.extend(self.coarse_decoder.parameters())
        params.extend(self.fusion_head.parameters())
        return params

    # -- checkpoint I/O -----------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save all learnable state (latent codes + hash field + decoders)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "latent": self._latent.data,
            "hash_field": self.hash_field.state_dict(),
            "fine_decoder": self.fine_decoder.state_dict(),
            "coarse_decoder": self.coarse_decoder.state_dict(),
            "fusion_head": self.fusion_head.state_dict(),
            "config": {
                "latent_dim": self._latent_dim,
                "output_dim": self._output_dim,
            },
        }
        torch.save(state, path)
        print(f"[HybridFeatureGaussian] Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load learnable state from a checkpoint."""
        state = torch.load(path, map_location="cpu")
        device = self._xyz.device if self._xyz.numel() > 0 else "cpu"

        self._latent = nn.Parameter(state["latent"].to(device))
        self.hash_field.load_state_dict(state["hash_field"])
        self.fine_decoder.load_state_dict(state["fine_decoder"])
        self.coarse_decoder.load_state_dict(state["coarse_decoder"])
        self.fusion_head.load_state_dict(state["fusion_head"])
        print(f"[HybridFeatureGaussian] Checkpoint loaded: {path}")
