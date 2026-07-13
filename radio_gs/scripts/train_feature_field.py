"""Train RADIO-GS feature field via distillation.

Supports Architecture A (Explicit per-Gaussian) and Architecture B (Hybrid
DCFF-style), with optional HCD codec compression and FeatSharp-3D integration.

Usage:
    python radio_gs/scripts/train_feature_field.py \
        --config radio_gs/configs/replica_explicit.yaml \
        [--resume path/to/checkpoint.pth] \
        [--warmstart path/to/weights.pth]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from radio_gs.artifact_paths import (
    DEFAULT_SIGLIP2_PROJECTION_WEIGHTS,
    DEFAULT_SIGLIP2_TEXT_EMBEDDINGS,
    resolve_siglip_projection_path,
    resolve_siglip_text_embeddings_path,
)
from radio_gs.config import RadioGSConfig, load_config
from radio_gs.scripts.audit_vpr_cache_alignment import audit_vpr_cache_payload_alignment
from radio_gs.geometry_utils import resolve_use_2dgs
from radio_gs.data.benchmark_paths import (
    extract_feature_frame_index,
    list_feature_paths,
    load_w2c_from_pose_dir,
    load_w2c_from_pose_file,
    resolve_dataset_type,
    resolve_depth_path,
    resolve_rgb_path,
    resolve_scene_root,
    resolve_semantics_path,
    resolve_split_data_dir,
    resolve_split_feature_dir,
    resolve_split_frame_ids,
    resolve_split_pose_source,
)
from radio_gs.heads.grounding_head import QueryGroundingAuxLoss
from radio_gs.heads.depth_head import DepthHead, DepthLoss
from radio_gs.heads.segmentation_head import SegmentationHead, SegmentationLoss, compute_miou
from radio_gs.losses.distillation_loss import (
    BoundaryAwareFeatureLoss,
    DepthGuidedFeatureLoss,
    DistillationLoss,
    GeometricEdgeAlignmentLoss,
    GradientWeightedLoss,
    MultiViewConsistencyLoss,
    TotalVariationLoss,
)
from radio_gs.losses.radio_adaptor_loss import (
    compute_radio_adaptor_alignment_loss,
    compute_radio_adaptor_cross_view_loss,
    compute_radio_adaptor_cross_view_mask_propagation_loss,
    compute_radio_adaptor_cross_view_propagation_loss,
    compute_radio_adaptor_local_affinity_loss,
    compute_radio_adaptor_mask_logit_loss,
    compute_radio_adaptor_peak_background_loss,
    compute_radio_adaptor_region_loss,
    compute_radio_adaptor_relation_loss,
    compute_radio_adaptor_token_contrast_loss,
)
from radio_gs.losses.text_heatmap_distill_loss import (
    compute_text_heatmap_distill_loss,
)
from radio_gs.losses.samclip_mask_loss import (
    SamClipMaskEntry,
    compute_samclip_mask_losses,
    load_samclip_mask_manifest,
)
from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
from radio_gs.models.featsharp_3d import FeatSharp3D
from radio_gs.models.feature_quality import (
    cosine_feature_quality_target,
    visibility_target_from_alpha,
)
from radio_gs.models.foundation_cache import (
    FoundationCache,
    compute_foundation_cache_supervision_loss,
    load_foundation_cache,
)
from radio_gs.models.hcd_codec import build_feature_codec
from radio_gs.models.hybrid_gaussian import HybridFeatureGaussian
from radio_gs.models.point_summary_adapter import (
    CompactToSummaryAdapter,
    append_point_summary_context,
    point_summary_context_dim,
)
from radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint
from radio_gs.models.siglip_projection import SigLIP2FeatureProjection, SigLIP2SummaryHead
from radio_gs.models.screen_refiner import (
    ScreenSpaceRefiner,
    build_refiner_guide,
    compute_refiner_extra_channels,
)
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer
from radio_gs.replica_constants import GROUNDING_QUERIES
from radio_gs.scannet_constants import (
    NYU40_ID_TO_NAME,
    OPENGAUSSIAN_NYU40_CLASS_SPLITS,
)
from radio_gs.utils.checkpoint_io import load_trusted_checkpoint
from radio_gs.training.artifact_mixin import TrainingArtifactMixin
from radio_gs.training.feature_supervision_mixin import FeatureSupervisionMixin
from radio_gs.training.feature_training_utils import (
    FoundationFeatureMapProjector,
    FoundationMaskLogitProjector,
    SimpleRadioDataset,
    merge_radio_adaptor_names,
    parse_direct_point_text_splits,
    parse_radio_adaptor_names,
    read_ply_xyz,
    read_ply_xyz_labels,
    resolve_foundation_cache_path,
    resolve_scannet_label_ply,
    sample_multiview_radio_targets,
    select_visible_gaussian_indices,
)


def set_quality_visibility_heads_only_trainable(
    model: nn.Module,
    *,
    extra_modules: Optional[List[nn.Module]] = None,
) -> int:
    """Freeze a feature field and leave only reliability heads trainable."""
    for param in model.parameters():
        param.requires_grad = False
    for module in extra_modules or []:
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = False

    trainable = 0
    fusion_head = getattr(model, "fusion_head", None)
    for head_name in ("quality_head", "visibility_head"):
        head = getattr(fusion_head, head_name, None) if fusion_head is not None else None
        if head is None:
            continue
        for param in head.parameters():
            param.requires_grad = True
            trainable += int(param.numel())

    if trainable <= 0:
        raise ValueError(
            "quality_visibility_heads_only requires hybrid_quality_head and/or "
            "hybrid_visibility_head to be enabled"
        )
    return trainable


from radio_gs.training.tensor_cache_io import load_training_tensor_cache

try:
    from torch.utils.tensorboard import SummaryWriter

    _HAS_TB = True
except ImportError:
    _HAS_TB = False


def audit_direct_point_teacher_cache_alignment_for_training(
    payload: Dict[str, Any],
    model_xyz: torch.Tensor,
    *,
    model_scales: Optional[torch.Tensor] = None,
    model_rotations: Optional[torch.Tensor] = None,
    model_opacities: Optional[torch.Tensor] = None,
    direct_point_source: str,
    direct_point_query_mode: str,
    cache_path: str = "",
    fail_max_l2: float = 1e-5,
) -> Dict[str, Any]:
    """Fail fast when a row-aligned Gaussian teacher cache is not geometry-aligned."""
    if direct_point_source != "gaussian" or direct_point_query_mode != "gaussian_index":
        return {
            "cache_path": str(cache_path),
            "status": "skipped",
            "passed": True,
            "message": (
                "row-alignment audit is only required for "
                "direct_point_source=gaussian and direct_point_query_mode=gaussian_index"
            ),
        }
    report = audit_vpr_cache_payload_alignment(
        payload,
        model_xyz.detach().cpu(),
        model_scales=model_scales.detach().cpu() if model_scales is not None else None,
        model_rotations=model_rotations.detach().cpu() if model_rotations is not None else None,
        model_opacities=model_opacities.detach().cpu() if model_opacities is not None else None,
        fail_max_l2=fail_max_l2,
        cache_path=str(cache_path),
    )
    if not report.get("passed", False):
        raise RuntimeError(
            "direct_point_teacher_cache row-alignment audit failed: "
            f"{report.get('message', '')}; "
            f"cache={cache_path}; "
            f"max_l2={report.get('max_l2', 'n/a')}; "
            f"fail_max_l2={report.get('fail_max_l2', fail_max_l2)}"
        )
    return report


























# ===================================================================
# Dataset
# ===================================================================



# ===================================================================
# Trainer
# ===================================================================

class RadioGSTrainer(FeatureSupervisionMixin, TrainingArtifactMixin):
    """Training loop for RADIO-GS feature field distillation."""

    def __init__(self, config: RadioGSConfig) -> None:
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_time_unix = time.time()
        self.run_start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.run_status = "initializing"
        self.failure_info: Optional[Dict[str, str]] = None

        # Training mode: "latent" trains in 64d space with frozen decoder,
        # "decoded" (default/legacy) trains through decoder in 1280d space
        self.train_mode = getattr(config, "train_mode", "decoded")

        # Reproducibility
        self._set_seed(getattr(config, "seed", 42))

        # Output directories
        self.output_dir = Path(getattr(config, "output_dir", "output/radio_gs"))
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.vis_dir = self.output_dir / "visualizations"
        self.report_dir = self.output_dir / "reports"
        for d in (self.ckpt_dir, self.log_dir, self.vis_dir, self.report_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Components
        self.model = self.build_model(config).to(self.device)
        self.codec = self._build_codec(config).to(self.device)
        self.renderer = FeatureFieldRenderer(
            image_height=getattr(config, "feature_height", 30),
            image_width=getattr(config, "feature_width", 40),
            fx=getattr(config, "fx", 320.0) * getattr(config, "feature_width", 40) / getattr(config, "image_width", 640),
            fy=getattr(config, "fy", 320.0) * getattr(config, "feature_height", 30) / getattr(config, "image_height", 480),
            cx=getattr(config, "cx", 319.5) * getattr(config, "feature_width", 40) / getattr(config, "image_width", 640),
            cy=getattr(config, "cy", 239.5) * getattr(config, "feature_height", 30) / getattr(config, "image_height", 480),
            max_channels_per_chunk=getattr(config, "max_channels_per_chunk", 32),
            use_2dgs=resolve_use_2dgs(config),
        ).to(self.device)
        self.sharpener = FeatSharp3D(
            mode=getattr(config, "featsharp_mode", "analytical"),
            feature_dim=self._resolve_latent_dim(config),
            strength=getattr(config, "featsharp_strength", 0.5),
        ).to(self.device)

        # Optional screen-space refiner (corrects alpha-blending artifacts)
        self.use_refiner = getattr(config, "use_refiner", False)
        self.refiner_rgb_guide = getattr(config, "refiner_rgb_guide", False)
        self.refiner_depth_guide = getattr(config, "refiner_depth_guide", False)
        self.refiner_alpha_guide = getattr(config, "refiner_alpha_guide", False)
        self.refiner_boundary_guide = getattr(config, "refiner_boundary_guide", False)
        self.self_guided = getattr(config, "self_guided", False)
        self.train_sh = getattr(config, "train_sh", False)
        self.rgb_loss_weight = getattr(config, "rgb_loss_weight", 0.0)
        self._is_hybrid = getattr(config, "architecture", "explicit") == "hybrid"
        self.hybrid_decoupled_heads = getattr(config, "hybrid_decoupled_heads", False)
        self.hybrid_semantic_adaptor_reg_weight = getattr(
            config, "hybrid_semantic_adaptor_reg_weight", 0.0
        )
        self.quality_loss_weight = float(getattr(config, "quality_loss_weight", 0.0))
        self.visibility_loss_weight = float(getattr(config, "visibility_loss_weight", 0.0))
        self.visibility_target_binary = bool(
            getattr(config, "visibility_target_binary", False)
        )
        self.visibility_alpha_threshold = float(
            getattr(config, "visibility_alpha_threshold", 0.02)
        )

        # Enable SH training if requested
        if self.train_sh and hasattr(self.model, "enable_sh_training"):
            self.model.enable_sh_training()
            self._log("Joint RGB training: SH coefficients unfrozen")

        if self.use_refiner:
            extra_ch = compute_refiner_extra_channels(
                rgb_guide=self.refiner_rgb_guide,
                depth_guide=self.refiner_depth_guide,
                depth_grad=getattr(config, "refiner_depth_grad", False),
                alpha_guide=self.refiner_alpha_guide,
                boundary_guide=self.refiner_boundary_guide,
            )
            self.refiner = ScreenSpaceRefiner(
                latent_dim=self._resolve_latent_dim(config),
                hidden_dim=getattr(config, "refiner_hidden_dim", 128),
                num_blocks=getattr(config, "refiner_num_blocks", 4),
                dropout=getattr(config, "refiner_dropout", 0.1),
                extra_channels=extra_ch,
                norm_type=getattr(config, "refiner_norm_type", "gn"),
            ).to(self.device)
        else:
            self.refiner = None

        # In latent mode, freeze codec entirely
        if self.train_mode == "latent":
            for p in self.codec.parameters():
                p.requires_grad = False
            self._log("Latent mode: codec frozen, training in 64d space")

        # Losses
        self.distill_loss_fn = DistillationLoss(
            l2_weight=getattr(config, "l2_weight", 1.0),
            cosine_weight=getattr(config, "cosine_weight", 0.5),
            channel_std_weight=getattr(config, "channel_std_weight", 0.0),
        )
        self.mv_loss_fn = MultiViewConsistencyLoss()
        self.tv_loss_fn = TotalVariationLoss()
        self.gradient_loss_weight = getattr(config, "gradient_loss_weight", 0.0)
        self.gradient_loss_fn: Optional[GradientWeightedLoss] = None
        if self.gradient_loss_weight > 0:
            self.gradient_loss_fn = GradientWeightedLoss(
                base_weight=1.0, edge_multiplier=3.0,
            ).to(self.device)
        self.depth_loss_weight = getattr(config, "depth_loss_weight", 0.0)
        self.geom_depth_loss_weight = getattr(config, "geom_depth_loss_weight", 0.0)
        self.depth_guided_feat_weight = getattr(config, "depth_guided_feature_weight", 0.0)
        self.depth_guided_feat_loss: Optional[DepthGuidedFeatureLoss] = None
        if self.depth_guided_feat_weight > 0:
            self.depth_guided_feat_loss = DepthGuidedFeatureLoss().to(self.device)
        self.geometric_edge_loss_weight = getattr(config, "geometric_edge_loss_weight", 0.0)
        self.geometric_edge_loss_fn: Optional[GeometricEdgeAlignmentLoss] = None
        if self.geometric_edge_loss_weight > 0:
            self.geometric_edge_loss_fn = GeometricEdgeAlignmentLoss().to(self.device)
        self.boundary_aware_loss_weight = getattr(config, "boundary_aware_loss_weight", 0.0)
        self.boundary_aware_loss_fn: Optional[BoundaryAwareFeatureLoss] = None
        if self.boundary_aware_loss_weight > 0:
            self.boundary_aware_loss_fn = BoundaryAwareFeatureLoss(
                sharpness_weight=getattr(config, "boundary_aware_sharpness_weight", 1.0),
                smoothness_weight=getattr(config, "boundary_aware_smoothness_weight", 1.0),
                edge_threshold=getattr(config, "boundary_aware_edge_threshold", 0.1),
            ).to(self.device)
        self.hybrid_semantic_aux_weight = getattr(config, "hybrid_semantic_aux_weight", 0.0)
        self.direct_point_loss_weight = getattr(config, "direct_point_loss_weight", 0.0)
        self.direct_point_sample_count = max(0, int(getattr(config, "direct_point_sample_count", 2048)))
        self.direct_point_sample_strategy = str(
            getattr(config, "direct_point_sample_strategy", "uniform")
        )
        self.direct_point_query_mode = getattr(config, "direct_point_query_mode", "gaussian_index")
        self.direct_point_gaussian_position_mode = str(
            getattr(config, "direct_point_gaussian_position_mode", "label_point")
        )
        self.direct_point_source = getattr(config, "direct_point_source", "gaussian")
        self.direct_point_teacher_cache = str(
            getattr(config, "direct_point_teacher_cache", "") or ""
        )
        self.direct_point_teacher_cache_feature_key = str(
            getattr(config, "direct_point_teacher_cache_feature_key", "") or ""
        )
        self.direct_point_teacher_cache_feature_space = str(
            getattr(config, "direct_point_teacher_cache_feature_space", "") or ""
        )
        self.direct_point_teacher_feature_space = "radio"
        self.direct_point_feature_key = getattr(config, "direct_point_feature_key", "features")
        self.direct_point_k = max(1, int(getattr(config, "direct_point_k", 8)))
        self.direct_point_candidate_k = max(
            0, int(getattr(config, "direct_point_candidate_k", 0) or 0)
        )
        self.direct_point_depth_tolerance = float(getattr(config, "direct_point_depth_tolerance", 0.08))
        self.direct_point_relative_depth_tolerance = float(
            getattr(config, "direct_point_relative_depth_tolerance", 0.02)
        )
        self.direct_point_alpha_threshold = float(getattr(config, "direct_point_alpha_threshold", 0.02))
        self.direct_point_summary_alignment_weight = float(
            getattr(config, "direct_point_summary_alignment_weight", 0.0)
        )
        self.direct_point_relation_weight = float(
            getattr(config, "direct_point_relation_weight", 0.0)
        )
        self.direct_point_relation_max_points = max(
            2, int(getattr(config, "direct_point_relation_max_points", 256))
        )
        self.direct_point_summary_adapter_weight = float(
            getattr(config, "direct_point_summary_adapter_weight", 0.0)
        )
        self.direct_point_text_loss_weight = float(
            getattr(config, "direct_point_text_loss_weight", 0.0)
        )
        self.direct_point_adapter_text_loss_weight = float(
            getattr(config, "direct_point_adapter_text_loss_weight", 0.0)
        )
        self.direct_point_adapter_text_distill_weight = float(
            getattr(config, "direct_point_adapter_text_distill_weight", 0.0)
        )
        self.direct_point_text_pseudo_ce_weight = float(
            getattr(config, "direct_point_text_pseudo_ce_weight", 0.0)
        )
        self.direct_point_text_pseudo_ce_confidence_threshold = float(
            getattr(
                config,
                "direct_point_text_pseudo_ce_confidence_threshold",
                0.0,
            )
        )
        self.direct_point_text_pseudo_ce_logit_scale = float(
            getattr(config, "direct_point_text_pseudo_ce_logit_scale", 1.0)
        )
        self.direct_point_text_pseudo_ce_center_logits = bool(
            getattr(config, "direct_point_text_pseudo_ce_center_logits", False)
        )
        self.direct_point_text_pseudo_ce_splits = str(
            getattr(config, "direct_point_text_pseudo_ce_splits", "") or ""
        )
        self.direct_point_adapter_text_pseudo_ce_weight = float(
            getattr(config, "direct_point_adapter_text_pseudo_ce_weight", 0.0)
        )
        self.direct_point_adapter_text_pseudo_ce_confidence_threshold = float(
            getattr(
                config,
                "direct_point_adapter_text_pseudo_ce_confidence_threshold",
                0.0,
            )
        )
        self.direct_point_adapter_text_pseudo_ce_logit_scale = float(
            getattr(config, "direct_point_adapter_text_pseudo_ce_logit_scale", 1.0)
        )
        self.direct_point_adapter_text_pseudo_ce_center_logits = bool(
            getattr(config, "direct_point_adapter_text_pseudo_ce_center_logits", False)
        )
        self.direct_point_adapter_text_pseudo_ce_splits = str(
            getattr(config, "direct_point_adapter_text_pseudo_ce_splits", "") or ""
        )
        self.direct_point_adapter_decoder_anchor_weight = float(
            getattr(config, "direct_point_adapter_decoder_anchor_weight", 0.0)
        )
        self.direct_point_text_temperature = max(
            1e-6, float(getattr(config, "direct_point_text_temperature", 0.07))
        )
        self.direct_point_text_ce_weighting = str(
            getattr(config, "direct_point_text_ce_weighting", "none")
        )
        self.direct_point_text_ce_min_weight = float(
            getattr(config, "direct_point_text_ce_min_weight", 0.5)
        )
        self.direct_point_text_ce_max_weight = float(
            getattr(config, "direct_point_text_ce_max_weight", 3.0)
        )
        self.direct_point_text_distill_weight = float(
            getattr(config, "direct_point_text_distill_weight", 0.0)
        )
        self.direct_point_text_distill_temperature = max(
            1e-6,
            float(getattr(config, "direct_point_text_distill_temperature", 1.0)),
        )
        self.direct_point_text_distill_confidence_threshold = float(
            getattr(config, "direct_point_text_distill_confidence_threshold", 0.0)
        )
        self.direct_point_view_count_weighting = str(
            getattr(config, "direct_point_view_count_weighting", "none") or "none"
        )
        self.direct_point_view_count_min_weight = float(
            getattr(config, "direct_point_view_count_min_weight", 0.0)
        )
        self.direct_point_view_count_percentile_low = float(
            getattr(config, "direct_point_view_count_percentile_low", 5.0)
        )
        self.direct_point_view_count_percentile_high = float(
            getattr(config, "direct_point_view_count_percentile_high", 95.0)
        )
        self.point_summary_adapter_context_features = str(
            getattr(config, "point_summary_adapter_context_features", "") or ""
        )
        self.point_summary_adapter_context_dim = point_summary_context_dim(
            self.point_summary_adapter_context_features
        )
        self.point_summary_adapter_view_count_max: Optional[torch.Tensor] = None
        self.direct_point_text_contrast_weight = float(
            getattr(config, "direct_point_text_contrast_weight", 0.0)
        )
        self.direct_point_text_contrast_temperature = max(
            1e-6,
            float(getattr(config, "direct_point_text_contrast_temperature", 0.1)),
        )
        self.direct_point_text_contrast_confidence_threshold = float(
            getattr(config, "direct_point_text_contrast_confidence_threshold", 0.0)
        )
        self.direct_point_text_contrast_pair_weighting = str(
            getattr(config, "direct_point_text_contrast_pair_weighting", "none") or "none"
        )
        self.direct_point_text_contrast_max_points = max(
            0,
            int(getattr(config, "direct_point_text_contrast_max_points", 4096) or 0),
        )
        self.direct_point_text_contrast_center_logits = bool(
            getattr(config, "direct_point_text_contrast_center_logits", False)
        )
        self.direct_point_query_logit_distill_weight = float(
            getattr(config, "direct_point_query_logit_distill_weight", 0.0)
        )
        self.direct_point_query_logit_distill_embeddings_path = str(
            getattr(config, "direct_point_query_logit_distill_embeddings", "") or ""
        )
        self.direct_point_query_logit_distill_temperature = float(
            getattr(config, "direct_point_query_logit_distill_temperature", 1.0)
        )
        self.direct_point_query_logit_distill_confidence_threshold = float(
            getattr(config, "direct_point_query_logit_distill_confidence_threshold", 0.0)
        )
        self.direct_point_query_logit_distill_embeddings: Optional[torch.Tensor] = None
        self.direct_point_query_support_distill_weight = float(
            getattr(config, "direct_point_query_support_distill_weight", 0.0)
        )
        self.direct_point_query_support_distill_embeddings_path = str(
            getattr(config, "direct_point_query_support_distill_embeddings", "") or ""
        )
        self.direct_point_query_support_distill_temperature = float(
            getattr(config, "direct_point_query_support_distill_temperature", 0.25)
        )
        self.direct_point_query_support_distill_confidence_threshold = float(
            getattr(config, "direct_point_query_support_distill_confidence_threshold", 0.0)
        )
        self.direct_point_query_support_distill_logit_norm = str(
            getattr(config, "direct_point_query_support_distill_logit_norm", "none") or "none"
        ).lower()
        self.direct_point_query_support_distill_embeddings: Optional[torch.Tensor] = None
        self.direct_point_render_consistency_weight = float(
            getattr(config, "direct_point_render_consistency_weight", 0.0)
        )
        self.direct_point_render_consistency_mode = str(
            getattr(config, "direct_point_render_consistency_mode", "cosine") or "cosine"
        )
        self.direct_point_cached_visible_fraction = float(
            getattr(config, "direct_point_cached_visible_fraction", 0.0)
        )
        self.direct_point_cached_visible_candidate_multiplier = max(
            1,
            int(
                getattr(
                    config,
                    "direct_point_cached_visible_candidate_multiplier",
                    1,
                )
                or 1
            ),
        )
        self.direct_point_cached_visible_balance = bool(
            getattr(config, "direct_point_cached_visible_balance", False)
        )
        self.direct_point_proposal_consistency_weight = float(
            getattr(config, "direct_point_proposal_consistency_weight", 0.0)
        )
        self.direct_point_proposal_contrast_weight = float(
            getattr(config, "direct_point_proposal_contrast_weight", 0.0)
        )
        self.direct_point_proposal_contrast_temperature = float(
            getattr(config, "direct_point_proposal_contrast_temperature", 0.07)
        )
        self.direct_point_proposal_voxel_size = float(
            getattr(config, "direct_point_proposal_voxel_size", 0.05)
        )
        self.direct_point_proposal_min_count = max(
            1, int(getattr(config, "direct_point_proposal_min_count", 2) or 1)
        )
        self.direct_point_proposal_space = str(
            getattr(config, "direct_point_proposal_space", "auto") or "auto"
        )
        self.direct_point_text_split = str(
            getattr(config, "direct_point_text_split", "19")
        )
        self.direct_point_text_pseudo_ce_split_list = parse_direct_point_text_splits(
            self.direct_point_text_pseudo_ce_splits,
            self.direct_point_text_split,
        )
        self.direct_point_adapter_text_pseudo_ce_split_list = parse_direct_point_text_splits(
            self.direct_point_adapter_text_pseudo_ce_splits,
            self.direct_point_text_split,
        )
        self.direct_point_text_split_ids: list[int] = []
        self.direct_point_text_embeddings: Optional[torch.Tensor] = None
        self.direct_point_text_pseudo_ce_banks: list[
            tuple[str, list[int], torch.Tensor]
        ] = []
        self.direct_point_adapter_text_pseudo_ce_banks: list[
            tuple[str, list[int], torch.Tensor]
        ] = []
        if self.direct_point_query_mode not in {"gaussian_index", "knn"}:
            raise ValueError(
                "direct_point_query_mode must be one of: gaussian_index, knn"
            )
        if self.direct_point_gaussian_position_mode not in {"gaussian_center", "label_point"}:
            raise ValueError(
                "direct_point_gaussian_position_mode must be one of: "
                "gaussian_center, label_point"
            )
        if self.direct_point_sample_strategy not in {
            "uniform",
            "class_balanced",
            "teacher_balanced",
        }:
            raise ValueError(
                "direct_point_sample_strategy must be one of: "
                "uniform, class_balanced, teacher_balanced"
            )
        if self.direct_point_source not in {"gaussian", "label_ply", "points3d"}:
            raise ValueError(
                "direct_point_source must be one of: gaussian, label_ply, points3d"
            )
        if self.direct_point_feature_key not in {"features", "fused", "semantic", "geometry"}:
            raise ValueError(
                "direct_point_feature_key must be one of: features, fused, semantic, geometry"
            )
        configured_text_splits = [
            self.direct_point_text_split,
            *self.direct_point_text_pseudo_ce_split_list,
            *self.direct_point_adapter_text_pseudo_ce_split_list,
        ]
        invalid_text_splits = sorted(
            {
                split
                for split in configured_text_splits
                if split not in OPENGAUSSIAN_NYU40_CLASS_SPLITS
            }
        )
        if invalid_text_splits:
            raise ValueError(
                "direct point text splits must be one of "
                f"{sorted(OPENGAUSSIAN_NYU40_CLASS_SPLITS)}, got {invalid_text_splits}"
            )
        if self.direct_point_text_ce_weighting not in {
            "none",
            "inverse_batch",
            "inverse_pool",
            "sqrt_inverse_pool_capped",
        }:
            raise ValueError(
                "direct_point_text_ce_weighting must be one of: "
                "none, inverse_batch, inverse_pool, sqrt_inverse_pool_capped"
            )
        if self.direct_point_view_count_weighting not in {"none", "log", "clipped_log"}:
            raise ValueError(
                "direct_point_view_count_weighting must be one of: none, log, clipped_log"
            )
        if self.direct_point_text_contrast_pair_weighting not in {"none", "visibility"}:
            raise ValueError(
                "direct_point_text_contrast_pair_weighting must be one of: none, visibility"
            )
        if self.direct_point_render_consistency_mode not in {"cosine", "mse"}:
            raise ValueError(
                "direct_point_render_consistency_mode must be one of: cosine, mse"
            )
        if self.direct_point_render_consistency_weight < 0:
            raise ValueError("direct_point_render_consistency_weight must be non-negative")
        if self.direct_point_query_support_distill_weight < 0:
            raise ValueError("direct_point_query_support_distill_weight must be non-negative")
        if self.direct_point_query_support_distill_temperature <= 0:
            raise ValueError("direct_point_query_support_distill_temperature must be positive")
        if self.direct_point_query_support_distill_logit_norm not in {"none", "center", "zscore"}:
            raise ValueError(
                "direct_point_query_support_distill_logit_norm must be one of: none, center, zscore"
            )
        if self.direct_point_proposal_consistency_weight < 0:
            raise ValueError("direct_point_proposal_consistency_weight must be non-negative")
        if self.direct_point_proposal_contrast_weight < 0:
            raise ValueError("direct_point_proposal_contrast_weight must be non-negative")
        if self.direct_point_proposal_contrast_temperature <= 0:
            raise ValueError("direct_point_proposal_contrast_temperature must be positive")
        if self.direct_point_proposal_voxel_size <= 0:
            raise ValueError("direct_point_proposal_voxel_size must be positive")
        if self.direct_point_proposal_space not in {"auto", "adapter", "decoder"}:
            raise ValueError("direct_point_proposal_space must be one of: auto, adapter, decoder")
        if not 0.0 <= self.direct_point_cached_visible_fraction <= 1.0:
            raise ValueError("direct_point_cached_visible_fraction must be between 0 and 1")
        if self.direct_point_cached_visible_candidate_multiplier < 1:
            raise ValueError("direct_point_cached_visible_candidate_multiplier must be >= 1")
        if self.direct_point_view_count_min_weight < 0:
            raise ValueError("direct_point_view_count_min_weight must be non-negative")
        if self.direct_point_view_count_percentile_high < self.direct_point_view_count_percentile_low:
            raise ValueError(
                "direct_point_view_count_percentile_high must be >= "
                "direct_point_view_count_percentile_low"
            )
        if self.direct_point_text_ce_min_weight < 0:
            raise ValueError("direct_point_text_ce_min_weight must be non-negative")
        if self.direct_point_text_ce_max_weight < self.direct_point_text_ce_min_weight:
            raise ValueError(
                "direct_point_text_ce_max_weight must be >= direct_point_text_ce_min_weight"
            )
        if self.direct_point_source == "points3d" and self.direct_point_query_mode == "gaussian_index":
            raise ValueError(
                "direct_point_query_mode=gaussian_index is only valid with "
                "direct_point_source=gaussian or row-aligned label_ply; use "
                "direct_point_query_mode=knn for unaligned point-cloud supervision"
            )
        if self.direct_point_loss_weight > 0 and not self._is_hybrid:
            self._log("direct_point_loss_weight is only supported for hybrid models; disabling")
            self.direct_point_loss_weight = 0.0
        self.direct_point_pool: Optional[torch.Tensor] = None
        self.direct_point_pool_labels: Optional[torch.Tensor] = None
        self.direct_point_teacher_features: Optional[torch.Tensor] = None
        self.direct_point_teacher_valid: Optional[torch.Tensor] = None
        self.direct_point_teacher_view_counts: Optional[torch.Tensor] = None
        self.direct_point_teacher_pseudo_label_cache: Optional[torch.Tensor] = None
        self.direct_point_teacher_cache_alignment_report: Optional[Dict[str, Any]] = None
        self.point_summary_adapter_metadata: Dict[str, Any] = {}
        self.point_summary_adapter_epoch: Optional[int] = None
        self.point_summary_adapter_best_metric: Optional[float] = None
        if self.direct_point_loss_weight > 0 and self._is_hybrid:
            self.direct_point_pool = self._load_direct_point_pool(config)
            if self.direct_point_teacher_cache:
                self._load_direct_point_teacher_cache(config)
        self.point_summary_adapter: Optional[CompactToSummaryAdapter] = None
        if (
            self.direct_point_summary_adapter_weight > 0
            or self.direct_point_adapter_text_loss_weight > 0
            or self.direct_point_adapter_text_distill_weight > 0
            or self.direct_point_adapter_text_pseudo_ce_weight > 0
            or self.direct_point_adapter_decoder_anchor_weight > 0
            or self.direct_point_query_support_distill_weight > 0
        ):
            if not self._is_hybrid:
                self._log("direct point summary adapter losses require hybrid models; disabling")
                self.direct_point_summary_adapter_weight = 0.0
                self.direct_point_adapter_text_loss_weight = 0.0
                self.direct_point_adapter_text_distill_weight = 0.0
                self.direct_point_adapter_text_pseudo_ce_weight = 0.0
                self.direct_point_adapter_decoder_anchor_weight = 0.0
                self.direct_point_query_support_distill_weight = 0.0
            else:
                self.point_summary_adapter = CompactToSummaryAdapter(
                    input_dim=getattr(
                        config,
                        "bottleneck_dim",
                        getattr(config, "hybrid_output_dim", 128),
                    )
                    + self.point_summary_adapter_context_dim,
                    output_dim=1536,
                    hidden_dim=getattr(config, "point_summary_adapter_hidden_dim", 512),
                    num_layers=getattr(config, "point_summary_adapter_num_layers", 2),
                    dropout=getattr(config, "point_summary_adapter_dropout", 0.0),
                ).to(self.device)
        self.depth_alpha_threshold = getattr(config, "depth_alpha_threshold", 0.05)
        self.depth_head: Optional[DepthHead] = None
        self.depth_supervision_loss: Optional[DepthLoss] = None
        self.geom_depth_supervision_loss: Optional[DepthLoss] = None
        # Frozen depth head supervision (core innovation)
        self.frozen_depth_head_weight = getattr(config, "frozen_depth_head_weight", 0.0)
        self.frozen_depth_head_weight_target = self.frozen_depth_head_weight  # for curriculum
        self.frozen_depth_warmup_epochs = getattr(config, "frozen_depth_warmup_epochs", 0)
        self.frozen_depth_teacher = getattr(config, "frozen_depth_teacher", "geom_depth")
        self.frozen_depth_head: Optional[DepthHead] = None
        self.frozen_depth_loss_fn: Optional[DepthLoss] = None
        self.frozen_depth_gradient_weight = getattr(config, "frozen_depth_gradient_weight", 0.0)
        if self.frozen_depth_head_weight > 0:
            frozen_path = getattr(config, "frozen_depth_head_path", "")
            if not frozen_path or not Path(frozen_path).exists():
                raise FileNotFoundError(
                    f"frozen_depth_head_path required when frozen_depth_head_weight > 0, "
                    f"got: '{frozen_path}'"
                )
            self._log(f"Loading frozen depth head from {frozen_path}")
            ckpt = load_trusted_checkpoint(frozen_path, map_location=self.device)
            head_cfg = ckpt.get("config", {})
            self.frozen_depth_head = DepthHead(
                feature_dim=head_cfg.get("feature_dim", getattr(config, "radio_feature_dim", 1280)),
                hidden_dim=head_cfg.get("hidden_dim", getattr(config, "frozen_depth_head_hidden_dim", 256)),
                num_layers=head_cfg.get("num_layers", getattr(config, "frozen_depth_head_num_layers", 3)),
                head_type=head_cfg.get("head_type", getattr(config, "frozen_depth_head_type", "mlp")),
            ).to(self.device)
            state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            self.frozen_depth_head.load_state_dict(state)
            # Freeze all parameters — gradients flow through features only
            for p in self.frozen_depth_head.parameters():
                p.requires_grad = False
            self.frozen_depth_head.eval()
            self.frozen_depth_loss_fn = DepthLoss(
                loss_type=getattr(config, "frozen_depth_loss_type", "scale_invariant"),
                weight=1.0,
            )
            self._log(f"Frozen depth head loaded ({sum(p.numel() for p in self.frozen_depth_head.parameters()) / 1e6:.3f}M params, all frozen)")

        self.frozen_seg_head_weight = getattr(config, "frozen_seg_head_weight", 0.0)
        self.frozen_seg_loss_type = getattr(config, "frozen_seg_loss_type", "kl")
        self.frozen_seg_temperature = float(getattr(config, "frozen_seg_temperature", 1.0))
        self.frozen_seg_head: Optional[SegmentationHead] = None
        if self.frozen_seg_head_weight > 0:
            frozen_seg_path = getattr(config, "frozen_seg_head_path", "")
            if not frozen_seg_path or not Path(frozen_seg_path).exists():
                raise FileNotFoundError(
                    f"frozen_seg_head_path required when frozen_seg_head_weight > 0, "
                    f"got: '{frozen_seg_path}'"
                )
            self._log(f"Loading frozen segmentation head from {frozen_seg_path}")
            ckpt = load_trusted_checkpoint(frozen_seg_path, map_location=self.device)
            head_cfg = ckpt.get("config", {})
            self.frozen_seg_head = SegmentationHead(
                feature_dim=head_cfg.get("feature_dim", getattr(config, "radio_feature_dim", 1280)),
                num_classes=head_cfg.get(
                    "num_classes",
                    getattr(config, "frozen_seg_num_classes", getattr(config, "seg_num_classes", 40)),
                ),
                hidden_dim=head_cfg.get("hidden_dim", getattr(config, "frozen_seg_head_hidden_dim", 256)),
                num_layers=head_cfg.get("num_layers", getattr(config, "frozen_seg_head_num_layers", 3)),
                head_type=head_cfg.get("head_type", getattr(config, "frozen_seg_head_type", "mlp")),
            ).to(self.device)
            state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            self.frozen_seg_head.load_state_dict(state)
            for p in self.frozen_seg_head.parameters():
                p.requires_grad = False
            self.frozen_seg_head.eval()
            self._log(
                "Frozen segmentation head loaded "
                f"({sum(p.numel() for p in self.frozen_seg_head.parameters()) / 1e6:.3f}M params, all frozen)"
            )

        self.seg_loss_weight = getattr(config, "seg_loss_weight", 0.0)
        self.seg_head: Optional[SegmentationHead] = None
        self.seg_loss_fn: Optional[SegmentationLoss] = None
        self.siglip_alignment_weight = getattr(config, "siglip_alignment_weight", 0.0)
        self.grounding_query_loss_weight = getattr(
            config, "grounding_query_loss_weight", 0.0
        )
        self.grounding_query_temperature = getattr(
            config, "grounding_query_temperature", 1.0
        )
        self.grounding_query_loss_fn: Optional[QueryGroundingAuxLoss] = None
        self.grounding_query_names: List[str] = []
        self.grounding_query_class_ids: List[int] = []
        self.grounding_text_embeddings: Optional[torch.Tensor] = None
        self.siglip_projection: Optional[SigLIP2FeatureProjection] = None
        self.siglip_summary_head: Optional[SigLIP2SummaryHead] = None
        self.siglip_summary_alignment_weight = getattr(
            config, "siglip_summary_alignment_weight", 0.0
        )
        self.text_heatmap_distill_weight = float(
            getattr(config, "text_heatmap_distill_weight", 0.0)
        )
        self.text_heatmap_distill_embeddings_path = str(
            getattr(config, "text_heatmap_distill_embeddings", "") or ""
        )
        self.text_heatmap_distill_downsample = max(
            1, int(getattr(config, "text_heatmap_distill_downsample", 2))
        )
        self.text_heatmap_distill_temperature = float(
            getattr(config, "text_heatmap_distill_temperature", 20.0)
        )
        self.text_heatmap_distill_mode = str(
            getattr(config, "text_heatmap_distill_mode", "query") or "query"
        )
        self.text_heatmap_distill_embeddings: Optional[torch.Tensor] = None
        self.radio_adaptor_alignment_names = parse_radio_adaptor_names(
            getattr(config, "radio_adaptor_alignment_names", "")
        )
        self.radio_adaptor_alignment_weight = float(
            getattr(config, "radio_adaptor_alignment_weight", 0.0)
        )
        self.radio_adaptor_alignment_kind = str(
            getattr(config, "radio_adaptor_alignment_kind", "feature_projection")
        )
        self.radio_adaptor_relation_names = parse_radio_adaptor_names(
            getattr(config, "radio_adaptor_relation_names", "")
        )
        self.radio_adaptor_relation_weight = float(
            getattr(config, "radio_adaptor_relation_weight", 0.0)
        )
        self.radio_adaptor_relation_downsample = max(
            1, int(getattr(config, "radio_adaptor_relation_downsample", 1))
        )
        self.radio_adaptor_relation_max_tokens = int(
            getattr(config, "radio_adaptor_relation_max_tokens", 512)
        )
        self.radio_adaptor_relation_temperature = float(
            getattr(config, "radio_adaptor_relation_temperature", 1.0)
        )
        self.radio_adaptor_local_affinity_names = parse_radio_adaptor_names(
            getattr(config, "radio_adaptor_local_affinity_names", "")
        )
        self.radio_adaptor_local_affinity_weight = float(
            getattr(config, "radio_adaptor_local_affinity_weight", 0.0)
        )
        self.radio_adaptor_local_affinity_downsample = max(
            1, int(getattr(config, "radio_adaptor_local_affinity_downsample", 1))
        )
        self.radio_adaptor_local_affinity_radius = max(
            1, int(getattr(config, "radio_adaptor_local_affinity_radius", 1))
        )
        self.radio_adaptor_token_contrast_names = parse_radio_adaptor_names(
            getattr(config, "radio_adaptor_token_contrast_names", "")
        )
        self.radio_adaptor_token_contrast_weight = float(
            getattr(config, "radio_adaptor_token_contrast_weight", 0.0)
        )
        self.radio_adaptor_token_contrast_downsample = max(
            1, int(getattr(config, "radio_adaptor_token_contrast_downsample", 1))
        )
        self.radio_adaptor_token_contrast_max_tokens = int(
            getattr(config, "radio_adaptor_token_contrast_max_tokens", 512)
        )
        self.radio_adaptor_token_contrast_temperature = float(
            getattr(config, "radio_adaptor_token_contrast_temperature", 0.07)
        )
        self.radio_adaptor_peak_background_names = parse_radio_adaptor_names(
            getattr(config, "radio_adaptor_peak_background_names", "")
        )
        self.radio_adaptor_peak_background_weight = float(
            getattr(config, "radio_adaptor_peak_background_weight", 0.0)
        )
        self.radio_adaptor_peak_background_downsample = max(
            1, int(getattr(config, "radio_adaptor_peak_background_downsample", 1))
        )
        self.radio_adaptor_peak_background_max_tokens = int(
            getattr(config, "radio_adaptor_peak_background_max_tokens", 512)
        )
        self.radio_adaptor_peak_background_num_anchors = int(
            getattr(config, "radio_adaptor_peak_background_num_anchors", 16)
        )
        self.radio_adaptor_peak_background_temperature = float(
            getattr(config, "radio_adaptor_peak_background_temperature", 0.2)
        )
        self.radio_adaptor_peak_background_anchor_strategy = str(
            getattr(
                config,
                "radio_adaptor_peak_background_anchor_strategy",
                "linspace",
            )
            or "linspace"
        )
        if self.radio_adaptor_peak_background_anchor_strategy not in {
            "linspace",
            "distinctive",
        }:
            raise ValueError(
                "radio_adaptor_peak_background_anchor_strategy must be "
                "'linspace' or 'distinctive'"
            )
        self.radio_adaptor_region_names = parse_radio_adaptor_names(
            getattr(config, "radio_adaptor_region_names", "")
        )
        self.radio_adaptor_region_weight = float(
            getattr(config, "radio_adaptor_region_weight", 0.0)
        )
        self.radio_adaptor_region_downsample = max(
            1, int(getattr(config, "radio_adaptor_region_downsample", 1))
        )
        self.radio_adaptor_region_max_tokens = int(
            getattr(config, "radio_adaptor_region_max_tokens", 512)
        )
        self.radio_adaptor_region_num_anchors = int(
            getattr(config, "radio_adaptor_region_num_anchors", 16)
        )
        self.radio_adaptor_region_temperature = float(
            getattr(config, "radio_adaptor_region_temperature", 0.07)
        )
        self.radio_adaptor_mask_logit_names = parse_radio_adaptor_names(
            getattr(config, "radio_adaptor_mask_logit_names", "")
        )
        self.radio_adaptor_mask_logit_weight = float(
            getattr(config, "radio_adaptor_mask_logit_weight", 0.0)
        )
        self.radio_adaptor_mask_logit_downsample = max(
            1, int(getattr(config, "radio_adaptor_mask_logit_downsample", 1))
        )
        self.radio_adaptor_mask_logit_max_tokens = int(
            getattr(config, "radio_adaptor_mask_logit_max_tokens", 512)
        )
        self.radio_adaptor_mask_logit_num_anchors = int(
            getattr(config, "radio_adaptor_mask_logit_num_anchors", 16)
        )
        self.radio_adaptor_mask_logit_temperature = float(
            getattr(config, "radio_adaptor_mask_logit_temperature", 0.07)
        )
        self.radio_adaptor_cross_view_names = parse_radio_adaptor_names(
            getattr(config, "radio_adaptor_cross_view_names", "")
        )
        self.radio_adaptor_cross_view_weight = float(
            getattr(config, "radio_adaptor_cross_view_weight", 0.0)
        )
        self.radio_adaptor_cross_view_downsample = max(
            1, int(getattr(config, "radio_adaptor_cross_view_downsample", 2))
        )
        self.radio_adaptor_cross_view_max_tokens = int(
            getattr(config, "radio_adaptor_cross_view_max_tokens", 256)
        )
        self.radio_adaptor_cross_view_temperature = float(
            getattr(config, "radio_adaptor_cross_view_temperature", 1.0)
        )
        self.radio_adaptor_cross_view_objective = str(
            getattr(config, "radio_adaptor_cross_view_objective", "mse")
        )
        if self.radio_adaptor_cross_view_objective not in {"mse", "transport_cycle"}:
            raise ValueError(
                "radio_adaptor_cross_view_objective must be 'mse' or "
                f"'transport_cycle', got {self.radio_adaptor_cross_view_objective!r}"
            )
        self.radio_adaptor_cross_view_propagation_names = parse_radio_adaptor_names(
            getattr(config, "radio_adaptor_cross_view_propagation_names", "")
        )
        self.radio_adaptor_cross_view_propagation_weight = float(
            getattr(config, "radio_adaptor_cross_view_propagation_weight", 0.0)
        )
        self.radio_adaptor_cross_view_propagation_downsample = max(
            1,
            int(getattr(config, "radio_adaptor_cross_view_propagation_downsample", 2)),
        )
        self.radio_adaptor_cross_view_propagation_max_tokens = int(
            getattr(config, "radio_adaptor_cross_view_propagation_max_tokens", 256)
        )
        self.radio_adaptor_cross_view_propagation_num_anchors = int(
            getattr(config, "radio_adaptor_cross_view_propagation_num_anchors", 16)
        )
        self.radio_adaptor_cross_view_propagation_temperature = float(
            getattr(config, "radio_adaptor_cross_view_propagation_temperature", 0.2)
        )
        self.radio_adaptor_cross_view_propagation_anchor_strategy = str(
            getattr(
                config,
                "radio_adaptor_cross_view_propagation_anchor_strategy",
                "linspace",
            )
            or "linspace"
        )
        if self.radio_adaptor_cross_view_propagation_anchor_strategy not in {
            "linspace",
            "distinctive",
        }:
            raise ValueError(
                "radio_adaptor_cross_view_propagation_anchor_strategy must be "
                "'linspace' or 'distinctive'"
            )
        self.radio_adaptor_cross_view_mask_propagation_names = parse_radio_adaptor_names(
            getattr(config, "radio_adaptor_cross_view_mask_propagation_names", "")
        )
        self.radio_adaptor_cross_view_mask_propagation_weight = float(
            getattr(config, "radio_adaptor_cross_view_mask_propagation_weight", 0.0)
        )
        self.radio_adaptor_cross_view_mask_propagation_downsample = max(
            1,
            int(getattr(config, "radio_adaptor_cross_view_mask_propagation_downsample", 2)),
        )
        self.radio_adaptor_cross_view_mask_propagation_max_tokens = int(
            getattr(config, "radio_adaptor_cross_view_mask_propagation_max_tokens", 256)
        )
        self.radio_adaptor_cross_view_mask_propagation_num_anchors = int(
            getattr(config, "radio_adaptor_cross_view_mask_propagation_num_anchors", 16)
        )
        self.radio_adaptor_cross_view_mask_propagation_temperature = float(
            getattr(config, "radio_adaptor_cross_view_mask_propagation_temperature", 0.2)
        )
        self.radio_adaptor_cross_view_mask_propagation_anchor_strategy = str(
            getattr(
                config,
                "radio_adaptor_cross_view_mask_propagation_anchor_strategy",
                "linspace",
            )
            or "linspace"
        )
        if self.radio_adaptor_cross_view_mask_propagation_anchor_strategy not in {
            "linspace",
            "distinctive",
        }:
            raise ValueError(
                "radio_adaptor_cross_view_mask_propagation_anchor_strategy must be "
                "'linspace' or 'distinctive'"
            )
        self.foundation_cache_root = str(getattr(config, "foundation_cache_root", "") or "")
        self.foundation_cache_weight = float(getattr(config, "foundation_cache_weight", 0.0))
        self.foundation_cache_heads = parse_radio_adaptor_names(
            getattr(config, "foundation_cache_heads", "")
        )
        self.foundation_cache_mask_logit_weight = float(
            getattr(config, "foundation_cache_mask_logit_weight", 0.0)
        )
        self.foundation_cache_mask_boundary_weight = float(
            getattr(config, "foundation_cache_mask_boundary_weight", 0.0)
        )
        self.foundation_cache_token_weight = float(
            getattr(config, "foundation_cache_token_weight", 0.0)
        )
        self.foundation_cache_region_consistency_weight = float(
            getattr(config, "foundation_cache_region_consistency_weight", 0.0)
        )
        self.foundation_cache_region_separation_weight = float(
            getattr(config, "foundation_cache_region_separation_weight", 0.0)
        )
        self.foundation_cache_feature_boundary_weight = float(
            getattr(config, "foundation_cache_feature_boundary_weight", 0.0)
        )
        self.foundation_cache_region_score_threshold = float(
            getattr(config, "foundation_cache_region_score_threshold", 0.0)
        )
        self.foundation_cache_region_max_masks = int(
            getattr(config, "foundation_cache_region_max_masks", 16)
        )
        self.foundation_cache_region_separation_margin = float(
            getattr(config, "foundation_cache_region_separation_margin", 0.25)
        )
        self.foundation_cache_require_official = bool(
            getattr(config, "foundation_cache_require_official", False)
        )
        self.foundation_cache_mask_projector_hidden_dim = int(
            getattr(config, "foundation_cache_mask_projector_hidden_dim", 256)
        )
        self.foundation_cache_mask_projector_masks = int(
            getattr(config, "foundation_cache_mask_projector_masks", 32)
        )
        self.foundation_cache_projectors = nn.ModuleDict()
        self.radio_adaptor_alignment_adaptors = nn.ModuleDict()
        if self.depth_loss_weight > 0 or self.geom_depth_loss_weight > 0:
            self.depth_head = DepthHead(
                feature_dim=getattr(config, "radio_feature_dim", 1280),
                hidden_dim=getattr(config, "depth_head_hidden_dim", 256),
                num_layers=getattr(config, "depth_head_num_layers", 3),
                head_type=getattr(config, "depth_head_type", "mlp"),
            ).to(self.device)
            self.depth_supervision_loss = DepthLoss(
                loss_type=getattr(config, "depth_supervision_loss_type", "scale_invariant"),
                weight=1.0,
            )
            self.geom_depth_supervision_loss = DepthLoss(
                loss_type=getattr(
                    config,
                    "geom_depth_supervision_loss_type",
                    getattr(config, "depth_supervision_loss_type", "scale_invariant"),
                ),
                weight=1.0,
            )
        if self.seg_loss_weight > 0:
            self.seg_head = SegmentationHead(
                feature_dim=getattr(config, "radio_feature_dim", 1280),
                num_classes=getattr(config, "seg_num_classes", 40),
                hidden_dim=getattr(config, "seg_head_hidden_dim", 256),
                num_layers=getattr(config, "seg_head_num_layers", 2),
                head_type=getattr(config, "seg_head_type", "mlp"),
            ).to(self.device)
            self.seg_loss_fn = SegmentationLoss(
                loss_type=getattr(config, "seg_loss_type", "ce"),
                ignore_index=getattr(config, "seg_ignore_index", 255),
            )
        foundation_uses_siglip = (
            self.foundation_cache_weight > 0
            and self.foundation_cache_token_weight > 0
            and any(name in {"siglip2", "siglip2-g"} for name in self.foundation_cache_heads)
        )
        if self.siglip_alignment_weight > 0 or foundation_uses_siglip:
            proj_path = resolve_siglip_projection_path(
                getattr(
                    config,
                    "siglip_projection_weights",
                    DEFAULT_SIGLIP2_PROJECTION_WEIGHTS,
                )
            )
            if not proj_path.exists():
                raise FileNotFoundError(
                    f"SigLIP2 projection weights not found: {proj_path}"
                )
            self.siglip_projection = SigLIP2FeatureProjection().to(self.device)
            self.siglip_projection.load_state_dict(
                load_training_tensor_cache(
                    proj_path,
                    map_location="cpu",
                    purpose="SigLIP2 projection weights",
                )
            )
            self.siglip_projection.eval()
            for param in self.siglip_projection.parameters():
                param.requires_grad = False
        enabled_radio_adaptors = merge_radio_adaptor_names(
            self.radio_adaptor_alignment_names
            if self.radio_adaptor_alignment_weight > 0
            else [],
            self.radio_adaptor_relation_names
            if self.radio_adaptor_relation_weight > 0
            else [],
            self.radio_adaptor_local_affinity_names
            if self.radio_adaptor_local_affinity_weight > 0
            else [],
            self.radio_adaptor_token_contrast_names
            if self.radio_adaptor_token_contrast_weight > 0
            else [],
            self.radio_adaptor_peak_background_names
            if self.radio_adaptor_peak_background_weight > 0
            else [],
            self.radio_adaptor_region_names
            if self.radio_adaptor_region_weight > 0
            else [],
            self.radio_adaptor_mask_logit_names
            if self.radio_adaptor_mask_logit_weight > 0
            else [],
            self.radio_adaptor_cross_view_names
            if self.radio_adaptor_cross_view_weight > 0
            else [],
            self.radio_adaptor_cross_view_propagation_names
            if self.radio_adaptor_cross_view_propagation_weight > 0
            else [],
            self.radio_adaptor_cross_view_mask_propagation_names
            if self.radio_adaptor_cross_view_mask_propagation_weight > 0
            else [],
            [
                name
                for name in self.foundation_cache_heads
                if (
                    self.foundation_cache_weight > 0
                    and self.foundation_cache_token_weight > 0
                    and name not in {"siglip2", "siglip2-g"}
                )
            ],
        )
        if enabled_radio_adaptors:
            radio_ckpt_path = Path(
                getattr(
                    config,
                    "radio_adaptor_alignment_checkpoint",
                    "/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
                )
            ).expanduser()
            if not radio_ckpt_path.exists():
                raise FileNotFoundError(
                    f"RADIO adaptor checkpoint not found: {radio_ckpt_path}"
                )
            for adaptor_name in enabled_radio_adaptors:
                adaptor = load_radio_adaptor_from_checkpoint(
                    radio_ckpt_path,
                    adaptor_name,
                    kind=self.radio_adaptor_alignment_kind,
                ).to(self.device)
                adaptor.eval()
                for param in adaptor.parameters():
                    param.requires_grad = False
                self.radio_adaptor_alignment_adaptors[adaptor_name] = adaptor
            self._log(
                "Loaded RADIO adaptor alignment heads: "
                f"{enabled_radio_adaptors} "
                f"kind={self.radio_adaptor_alignment_kind} "
                f"alignment_weight={self.radio_adaptor_alignment_weight:g} "
                f"relation_weight={self.radio_adaptor_relation_weight:g} "
                f"local_affinity_weight={self.radio_adaptor_local_affinity_weight:g} "
                f"token_contrast_weight={self.radio_adaptor_token_contrast_weight:g} "
                f"peak_background_weight={self.radio_adaptor_peak_background_weight:g} "
                f"peak_background_anchor_strategy="
                f"{self.radio_adaptor_peak_background_anchor_strategy} "
                f"region_weight={self.radio_adaptor_region_weight:g} "
                f"mask_logit_weight={self.radio_adaptor_mask_logit_weight:g} "
                f"cross_view_weight={self.radio_adaptor_cross_view_weight:g} "
                f"cross_view_objective={self.radio_adaptor_cross_view_objective} "
                f"cross_view_propagation_weight="
                f"{self.radio_adaptor_cross_view_propagation_weight:g} "
                f"cross_view_mask_propagation_weight="
                f"{self.radio_adaptor_cross_view_mask_propagation_weight:g}"
            )
        if self.foundation_cache_weight > 0:
            foundation_uses_mask_logits = (
                self.foundation_cache_mask_logit_weight > 0
                or self.foundation_cache_mask_boundary_weight > 0
            )
            for head_name in self.foundation_cache_heads:
                if foundation_uses_mask_logits and head_name == "sam3":
                    self.foundation_cache_projectors[head_name] = FoundationMaskLogitProjector(
                        input_dim=getattr(config, "radio_feature_dim", 1280),
                        hidden_dim=self.foundation_cache_mask_projector_hidden_dim,
                        output_masks=self.foundation_cache_mask_projector_masks,
                    ).to(self.device)
                    continue
                if head_name in {"siglip2", "siglip2-g"}:
                    if self.siglip_projection is not None:
                        projector = FoundationFeatureMapProjector(self.siglip_projection)
                        self.foundation_cache_projectors["siglip2"] = projector
                        self.foundation_cache_projectors["siglip2-g"] = projector
                    continue
                if head_name in self.radio_adaptor_alignment_adaptors:
                    self.foundation_cache_projectors[head_name] = FoundationFeatureMapProjector(
                        self.radio_adaptor_alignment_adaptors[head_name]
                    )
            self._log(
                "Foundation cache supervision configured: "
                f"root={self.foundation_cache_root or '<none>'} "
                f"heads={self.foundation_cache_heads or 'all-cache-heads'} "
                f"projectors={list(self.foundation_cache_projectors.keys())} "
                f"weight={self.foundation_cache_weight:g} "
                f"token_weight={self.foundation_cache_token_weight:g} "
                f"mask_logit_weight={self.foundation_cache_mask_logit_weight:g} "
                f"mask_boundary_weight={self.foundation_cache_mask_boundary_weight:g} "
                f"region_consistency_weight={self.foundation_cache_region_consistency_weight:g} "
                f"region_separation_weight={self.foundation_cache_region_separation_weight:g} "
                f"feature_boundary_weight={self.foundation_cache_feature_boundary_weight:g} "
                f"region_score_threshold={self.foundation_cache_region_score_threshold:g} "
                f"region_max_masks={self.foundation_cache_region_max_masks} "
                f"region_separation_margin={self.foundation_cache_region_separation_margin:g} "
                f"mask_projector_hidden={self.foundation_cache_mask_projector_hidden_dim} "
                f"mask_projector_masks={self.foundation_cache_mask_projector_masks} "
                f"require_official={self.foundation_cache_require_official}"
            )
        if (
            self.siglip_summary_alignment_weight > 0
            or self.text_heatmap_distill_weight > 0
            or self.grounding_query_loss_weight > 0
            or self.direct_point_summary_alignment_weight > 0
            or self.direct_point_relation_weight > 0
            or self.direct_point_summary_adapter_weight > 0
            or self.direct_point_text_loss_weight > 0
            or self.direct_point_adapter_text_loss_weight > 0
            or self.direct_point_text_distill_weight > 0
            or self.direct_point_adapter_text_distill_weight > 0
            or self.direct_point_text_pseudo_ce_weight > 0
            or self.direct_point_adapter_text_pseudo_ce_weight > 0
            or self.direct_point_adapter_decoder_anchor_weight > 0
        ):
            summary_path = Path(
                getattr(
                    config,
                    "siglip_summary_head_weights",
                    "checkpoints/siglip2_summary_head.pth",
                )
            )
            if not summary_path.exists():
                raise FileNotFoundError(
                    f"SigLIP2 summary head weights not found: {summary_path}"
                )
            self.siglip_summary_head = SigLIP2SummaryHead().to(self.device)
            self.siglip_summary_head.load_state_dict(
                load_training_tensor_cache(
                    summary_path,
                    map_location="cpu",
                    purpose="SigLIP2 summary head weights",
                )
            )
            self.siglip_summary_head.eval()
            for param in self.siglip_summary_head.parameters():
                param.requires_grad = False
            self._log(
                f"SigLIP2 summary head loaded for text-space alignment "
                f"(image_weight={self.siglip_summary_alignment_weight}, "
                f"point_weight={self.direct_point_summary_alignment_weight})"
            )

        if self.text_heatmap_distill_weight > 0:
            if self.siglip_summary_head is None:
                raise RuntimeError(
                    "text_heatmap_distill requires SigLIP2SummaryHead; "
                    "set siglip_summary_head_weights"
                )
            self.text_heatmap_distill_embeddings = (
                self._load_text_heatmap_distill_embeddings(config)
            )
            self._log(
                "Loaded text heatmap distillation bank: "
                f"{self.text_heatmap_distill_embeddings.shape[0]} queries "
                f"T={self.text_heatmap_distill_temperature:g} "
                f"mode={self.text_heatmap_distill_mode} "
                f"weight={self.text_heatmap_distill_weight:g}"
            )

        if self.direct_point_query_logit_distill_weight > 0:
            if self.siglip_summary_head is None:
                raise RuntimeError(
                    "direct point query-logit distillation requires SigLIP2SummaryHead; "
                    "set siglip_summary_head_weights"
                )
            self.direct_point_query_logit_distill_embeddings = (
                self._load_direct_point_query_logit_distill_embeddings(config)
            )
            self._log(
                "Loaded direct point query-logit distillation bank: "
                f"{self.direct_point_query_logit_distill_embeddings.shape[0]} queries "
                f"T={self.direct_point_query_logit_distill_temperature:g} "
                f"weight={self.direct_point_query_logit_distill_weight:g}"
            )
        if self.direct_point_query_support_distill_weight > 0:
            if self.siglip_summary_head is None:
                raise RuntimeError(
                    "direct point query-support distillation requires SigLIP2SummaryHead; "
                    "set siglip_summary_head_weights"
                )
            support_path = (
                self.direct_point_query_support_distill_embeddings_path
                or self.direct_point_query_logit_distill_embeddings_path
            )
            self.direct_point_query_support_distill_embeddings = (
                self._load_direct_point_query_logit_distill_embeddings(
                    config,
                    raw_path=support_path,
                    purpose="direct point query-support distillation embeddings",
                )
            )
            self._log(
                "Loaded direct point query-support distillation bank: "
                f"{self.direct_point_query_support_distill_embeddings.shape[0]} queries "
                f"T={self.direct_point_query_support_distill_temperature:g} "
                f"weight={self.direct_point_query_support_distill_weight:g}"
            )

        direct_point_text_bank_needed = (
            self.direct_point_text_loss_weight > 0
            or self.direct_point_adapter_text_loss_weight > 0
            or self.direct_point_text_distill_weight > 0
            or self.direct_point_adapter_text_distill_weight > 0
            or self.direct_point_text_pseudo_ce_weight > 0
            or self.direct_point_adapter_text_pseudo_ce_weight > 0
        )
        if direct_point_text_bank_needed:
            if self.siglip_summary_head is None:
                raise RuntimeError(
                    "direct point text losses require SigLIP2SummaryHead; "
                    "set siglip_summary_head_weights"
                )
            (
                self.direct_point_text_split_ids,
                self.direct_point_text_embeddings,
            ) = self._load_direct_point_text_embeddings(
                config,
                split=self.direct_point_text_split,
            )
            if self.direct_point_text_pseudo_ce_weight > 0:
                self.direct_point_text_pseudo_ce_banks = (
                    self._load_direct_point_text_embedding_banks(
                        config,
                        self.direct_point_text_pseudo_ce_split_list,
                    )
                )
            if self.direct_point_adapter_text_pseudo_ce_weight > 0:
                self.direct_point_adapter_text_pseudo_ce_banks = (
                    self._load_direct_point_text_embedding_banks(
                        config,
                        self.direct_point_adapter_text_pseudo_ce_split_list,
                    )
                )
            if (
                self.direct_point_pool_labels is None
                and (
                    self.direct_point_text_loss_weight > 0
                    or self.direct_point_adapter_text_loss_weight > 0
                )
            ):
                self._log(
                    "direct point text losses require label PLY labels; disabling "
                    "direct_point_text_loss_weight and direct_point_adapter_text_loss_weight"
                )
                self.direct_point_text_loss_weight = 0.0
                self.direct_point_adapter_text_loss_weight = 0.0
            names = [
                NYU40_ID_TO_NAME.get(class_id, f"class_{class_id}")
                for class_id in self.direct_point_text_split_ids
            ]
            self._log(
                "Loaded direct point text bank: "
                f"split={self.direct_point_text_split} "
                f"classes={len(names)} "
                f"ce_temperature={self.direct_point_text_temperature:g} "
                f"distill_weight={self.direct_point_text_distill_weight:g} "
                f"adapter_distill_weight={self.direct_point_adapter_text_distill_weight:g} "
                f"pseudo_ce_weight={self.direct_point_text_pseudo_ce_weight:g} "
                f"pseudo_ce_splits={self.direct_point_text_pseudo_ce_split_list} "
                f"adapter_pseudo_ce_weight={self.direct_point_adapter_text_pseudo_ce_weight:g} "
                f"adapter_pseudo_ce_splits={self.direct_point_adapter_text_pseudo_ce_split_list}"
            )

        if self.grounding_query_loss_weight > 0:
            if resolve_dataset_type(getattr(config, "dataset_type", "replica")) != "replica":
                self._log(
                    "grounding_query_loss_weight is currently implemented for "
                    "Replica only; disabling grounding query aux loss"
                )
                self.grounding_query_loss_weight = 0.0
            else:
                self.grounding_query_loss_fn = QueryGroundingAuxLoss(
                    feature_dim=1536,
                    temperature=self.grounding_query_temperature,
                ).to(self.device)
                (
                    self.grounding_query_names,
                    self.grounding_query_class_ids,
                    self.grounding_text_embeddings,
                ) = self._load_grounding_text_embeddings(config)
                self._log(
                    "Loaded grounding query aux bank: "
                    f"{len(self.grounding_query_names)} queries "
                    f"from {getattr(config, 'grounding_text_embeddings', '')}"
                )

        self.quality_visibility_heads_only = bool(
            getattr(config, "quality_visibility_heads_only", False)
        )
        if self.quality_visibility_heads_only:
            extra_modules: List[nn.Module] = [self.codec, self.sharpener]
            for module in (
                getattr(self, "refiner", None),
                getattr(self, "depth_head", None),
                getattr(self, "seg_head", None),
                getattr(self, "point_summary_adapter", None),
                getattr(self, "foundation_cache_projectors", None),
            ):
                if module is not None:
                    extra_modules.append(module)
            trainable_reliability_params = set_quality_visibility_heads_only_trainable(
                self.model,
                extra_modules=extra_modules,
            )
            self._log(
                "Quality/visibility heads-only mode: froze feature field and "
                f"left {trainable_reliability_params:,} reliability-head params trainable"
            )

        # Feature norm regularization weight
        self.feat_norm_weight = getattr(config, "feat_norm_weight", 0.0)

        # SAM-CLIP mask-level supervision for LangSplat cache ablations.
        self.samclip_mask_loss_weight = float(
            getattr(config, "samclip_mask_loss_weight", 0.0)
        )
        self.samclip_contrastive_loss_weight = float(
            getattr(config, "samclip_contrastive_loss_weight", 0.0)
        )
        self.samclip_background_loss_weight = float(
            getattr(config, "samclip_background_loss_weight", 0.0)
        )
        self.samclip_contrastive_temperature = max(
            1e-6, float(getattr(config, "samclip_contrastive_temperature", 0.07))
        )
        self.samclip_mask_min_pixels = max(
            1, int(getattr(config, "samclip_mask_min_pixels", 16))
        )
        self.samclip_mask_max_regions = max(
            0, int(getattr(config, "samclip_mask_max_regions", 64))
        )
        self.samclip_mask_cache_size = max(
            0, int(getattr(config, "samclip_mask_cache_size", 8))
        )
        self.samclip_mask_entries: Dict[int, SamClipMaskEntry] = {}
        self.samclip_mask_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        if self.samclip_mask_loss_weight < 0:
            raise ValueError("samclip_mask_loss_weight must be non-negative")
        if self.samclip_contrastive_loss_weight < 0:
            raise ValueError("samclip_contrastive_loss_weight must be non-negative")
        if self.samclip_background_loss_weight < 0:
            raise ValueError("samclip_background_loss_weight must be non-negative")
        if (
            self.samclip_mask_loss_weight > 0
            or self.samclip_contrastive_loss_weight > 0
            or self.samclip_background_loss_weight > 0
        ):
            samclip_root = str(
                getattr(config, "samclip_language_feature_dir", "")
                or getattr(config, "feature_dir", "")
                or ""
            )
            if not samclip_root:
                raise ValueError(
                    "samclip_language_feature_dir or feature_dir is required when "
                    "SAM-CLIP mask losses are enabled"
                )
            self.samclip_mask_entries = load_samclip_mask_manifest(samclip_root)
            self._log(
                "SAM-CLIP mask supervision enabled: "
                f"frames={len(self.samclip_mask_entries)} "
                f"prototype_weight={self.samclip_mask_loss_weight:g} "
                f"contrastive_weight={self.samclip_contrastive_loss_weight:g} "
                f"background_weight={self.samclip_background_loss_weight:g} "
                f"temperature={self.samclip_contrastive_temperature:g} "
                f"min_pixels={self.samclip_mask_min_pixels} "
                f"max_regions={self.samclip_mask_max_regions}"
            )

        # Optimizer with separate LR groups
        self.optimizer = self._build_optimizer(config)
        self.scheduler = self._build_scheduler(config)
        self.scaler = GradScaler()

        # Datasets + loaders
        self.train_dataset, self.val_dataset = self.build_dataset(config)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=getattr(config, "batch_size", 4),
            shuffle=bool(getattr(config, "train_shuffle", True)),
            num_workers=getattr(config, "num_workers", 4),
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_batch,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=getattr(config, "num_workers", 4),
            pin_memory=True,
            collate_fn=self._collate_batch,
        )

        # Logging
        self.writer: Optional[SummaryWriter] = None
        if _HAS_TB:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Tracking
        self.start_epoch = 1
        self.global_step = 0
        self.best_cosine = -1.0
        self.best_metric_name = getattr(config, "best_metric", "cosine")
        self.best_metric_mode = getattr(config, "best_metric_mode", "auto")
        self.best_selection_score = float("-inf")
        self.best_selection_value: Optional[float] = None
        self.best_epoch = 0
        self.last_train_metrics: Dict[str, float] = {}
        self.last_val_metrics: Dict[str, float] = {}
        self.resolved_config_path = (
            str(Path(getattr(config, "config_path", "")).resolve())
            if getattr(config, "config_path", None)
            else None
        )
        self.metrics_history_path = self.report_dir / "metrics_history.jsonl"
        self.resolved_config_path_json = self.report_dir / "resolved_config.json"

        self._write_run_manifest()

        self._log(f"Model params: {self._count_params(self.model):.2f}M")
        self._log(f"Codec params: {self._count_params(self.codec):.2f}M")
        self._log(f"Sharpener mode: {self.sharpener.mode}")
        if self.use_refiner and self.refiner is not None:
            self._log(f"Refiner params: {self._count_params(self.refiner):.2f}M")
        if self.depth_head is not None:
            self._log(f"Depth aux head params: {self._count_params(self.depth_head):.2f}M")
        if self.frozen_depth_head is not None:
            self._log(f"Frozen depth head params: {self._count_params(self.frozen_depth_head):.2f}M (frozen)")
            self._log(f"Frozen depth teacher: {self.frozen_depth_teacher}")
        if self.frozen_seg_head is not None:
            self._log(f"Frozen seg head params: {self._count_params(self.frozen_seg_head):.2f}M (frozen)")
        if self.seg_head is not None:
            self._log(f"Seg aux head params: {self._count_params(self.seg_head):.2f}M")
        if self.point_summary_adapter is not None:
            self._log(
                f"Point summary adapter params: "
                f"{self._count_params(self.point_summary_adapter):.2f}M"
            )
            if self.point_summary_adapter_context_features:
                self._log(
                    "Point summary adapter context: "
                    f"{self.point_summary_adapter_context_features}"
                )
        self._log(
            f"Best checkpoint metric: {self.best_metric_name} "
            f"(mode={self.best_metric_mode})"
        )

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_latent_dim(config: RadioGSConfig) -> int:
        arch = getattr(config, "architecture", "explicit")
        if arch == "hybrid":
            return getattr(config, "hybrid_latent_dim", 16)
        return getattr(config, "latent_dim", 64)

    def build_model(self, config: RadioGSConfig) -> nn.Module:
        arch = getattr(config, "architecture", "explicit")
        if arch == "explicit":
            model = ExplicitFeatureGaussian(
                latent_dim=getattr(config, "latent_dim", 64),
                train_sh=getattr(config, "train_sh", False),
            )
        elif arch == "hybrid":
            model = HybridFeatureGaussian(
                latent_dim=getattr(config, "hybrid_latent_dim", 16),
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
                semantic_adaptor_mode=getattr(
                    config, "hybrid_semantic_adaptor_mode", "confidence"
                ),
                semantic_adaptor_hidden_dim=getattr(
                    config, "hybrid_semantic_adaptor_hidden_dim", 64
                ),
                semantic_adaptor_use_geometry_guidance=getattr(
                    config, "hybrid_semantic_adaptor_use_geometry_guidance", True
                ),
                semantic_adaptor_use_depth_guidance=getattr(
                    config, "hybrid_semantic_adaptor_use_depth_guidance", False
                ),
                semantic_adaptor_residual=getattr(
                    config, "hybrid_semantic_adaptor_residual", True
                ),
                use_quality_head=getattr(config, "hybrid_quality_head", False),
                use_visibility_head=getattr(config, "hybrid_visibility_head", False),
            )
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        ply_path = getattr(config, "ply_path", None)
        if ply_path:
            self._log(f"Loading geometry from {ply_path}")
            model.load_from_ply(ply_path)

        return model

    @staticmethod
    def _build_codec(config: RadioGSConfig) -> nn.Module:
        return build_feature_codec(
            input_dim=getattr(config, "radio_feature_dim", 1280),
            bottleneck_dim=getattr(config, "bottleneck_dim", 64),
            codec_type=getattr(config, "codec_type", "hcd"),
            dual_stream=getattr(config, "dual_stream", True),
            symmetric_decoder=getattr(config, "symmetric_decoder", False),
            hidden_normalization=getattr(
                config, "codec_hidden_normalization", "legacy_group"
            ),
            final_normalization=getattr(
                config, "codec_final_normalization", "legacy_group"
            ),
        )

    def _build_optimizer(self, config: RadioGSConfig) -> optim.Optimizer:
        arch = getattr(config, "architecture", "explicit")
        # Feature embeddings (always trainable)
        feature_params = [self.model._feature if arch == "explicit" else self.model._latent]
        param_groups = [
            {
                "params": feature_params,
                "lr": getattr(config, "lr_features", 1e-3),
                "name": "features",
            },
        ]
        # Hybrid architecture: hash grid + screen-space decoders
        if arch == "hybrid":
            param_groups.append({
                "params": list(self.model.hash_field.parameters()),
                "lr": getattr(config, "lr_hash", 1e-3),
                "name": "hash_field",
            })
            hybrid_decoder_params = (
                list(self.model.fine_decoder.parameters())
                + list(self.model.coarse_decoder.parameters())
                + list(self.model.fusion_head.parameters())
            )
            param_groups.append({
                "params": hybrid_decoder_params,
                "lr": getattr(config, "lr_decoder", 1e-4),
                "name": "hybrid_decoders",
            })
        # SH params (separate group for joint RGB training)
        if self.train_sh and hasattr(self.model, "_sh_dc_param"):
            sh_params = [self.model._sh_dc_param]
            if hasattr(self.model, "_sh_rest_param") and self.model._sh_rest_param is not None:
                sh_params.append(self.model._sh_rest_param)
            param_groups.append(
                {
                    "params": sh_params,
                    "lr": getattr(config, "lr_sh", 5e-4),
                    "name": "sh_colors",
                }
            )
        # Only add decoder to optimizer if not in latent mode (decoder is frozen)
        if self.train_mode != "latent":
            param_groups.append(
                {
                    "params": self.codec.decoder.parameters(),
                    "lr": getattr(config, "lr_decoder", 1e-4),
                    "name": "decoder",
                }
            )
        if self.sharpener.mode not in ("analytical", "none"):
            param_groups.append(
                {
                    "params": self.sharpener.parameters(),
                    "lr": getattr(config, "lr_heads", 1e-4),
                    "name": "sharpener",
                }
            )
        if self.use_refiner and self.refiner is not None:
            param_groups.append(
                {
                    "params": self.refiner.parameters(),
                    "lr": getattr(config, "lr_refiner", 5e-4),
                    "name": "refiner",
                }
            )
        if self.depth_head is not None:
            param_groups.append(
                {
                    "params": self.depth_head.parameters(),
                    "lr": getattr(config, "lr_heads", 1e-4),
                    "name": "depth_head",
                }
            )
        if self.seg_head is not None:
            param_groups.append(
                {
                    "params": self.seg_head.parameters(),
                    "lr": getattr(config, "lr_heads", 1e-4),
                    "name": "seg_head",
                }
            )
        if self.point_summary_adapter is not None:
            param_groups.append(
                {
                    "params": self.point_summary_adapter.parameters(),
                    "lr": getattr(config, "lr_point_summary_adapter", 1e-4),
                    "name": "point_summary_adapter",
                }
            )
        foundation_params = [
            param
            for param in self.foundation_cache_projectors.parameters()
            if param.requires_grad
        ]
        if foundation_params:
            param_groups.append(
                {
                    "params": foundation_params,
                    "lr": getattr(config, "lr_heads", 1e-4),
                    "name": "foundation_cache_projectors",
                }
            )
        return optim.AdamW(
            param_groups,
            weight_decay=getattr(config, "weight_decay", 1e-5),
            betas=(0.9, 0.999),
        )

    def _build_scheduler(
        self, config: RadioGSConfig
    ) -> optim.lr_scheduler._LRScheduler:
        warmup_epochs = getattr(config, "warmup_epochs", 5)
        total_epochs = getattr(config, "epochs", 100)

        cosine = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_epochs - warmup_epochs, 1),
            eta_min=1e-6,
        )
        if warmup_epochs > 0:
            warmup = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                total_iters=warmup_epochs,
            )
            return optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        return cosine

    @staticmethod
    def _metric_prefers_min(metric_name: str) -> bool:
        return metric_name in {
            "mse",
            "depth_gt",
            "depth_geom",
            "frozen_depth",
            "siglip_align",
            "summary_align",
            "radio_adaptors",
            "radio_relations",
            "radio_local_affinity",
            "radio_regions",
            "radio_cross_view_propagation",
            "ground_query",
            "seg_aux",
            "frozen_seg",
        }

    def _resolve_best_metric(self, metrics: Dict[str, float]) -> Tuple[str, float, float]:
        metric_name = self.best_metric_name
        if metric_name == "proxy_depth":
            components: list[float] = []
            if "frozen_depth" in metrics:
                components.append(float(metrics["frozen_depth"]))
            if "depth_geom" in metrics:
                components.append(0.5 * float(metrics["depth_geom"]))
            if "depth_gt" in metrics:
                components.append(0.25 * float(metrics["depth_gt"]))
            if "mse" in metrics:
                components.append(0.05 * float(metrics["mse"]))
            if not components:
                raise KeyError(
                    "best_metric=proxy_depth requested, but no depth proxy metrics are available"
                )
            value = float(sum(components))
            return metric_name, value, -value

        if metric_name not in metrics:
            raise KeyError(
                f"best_metric='{metric_name}' not found in validation metrics: "
                f"{sorted(metrics.keys())}"
            )

        value = float(metrics[metric_name])
        mode = self.best_metric_mode
        if mode == "auto":
            maximize = not self._metric_prefers_min(metric_name)
        elif mode == "max":
            maximize = True
        elif mode == "min":
            maximize = False
        else:
            raise ValueError(f"Unknown best_metric_mode '{mode}'")
        score = value if maximize else -value
        return metric_name, value, score

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def _load_direct_point_pool(self, config: RadioGSConfig) -> Optional[torch.Tensor]:
        """Load optional non-Gaussian 3-D points for direct point distillation."""
        if self.direct_point_source == "gaussian":
            return None

        scene_root = resolve_scene_root(config)
        scene = getattr(config, "scene", Path(scene_root).name)
        explicit_path = getattr(config, "direct_point_ply_path", "") or ""
        if explicit_path:
            ply_path = Path(str(explicit_path).format(scene=scene))
        elif self.direct_point_source == "label_ply":
            ply_path = resolve_scannet_label_ply(scene_root, scene)
        else:
            ply_path = Path(scene_root) / "points3d.ply"

        if not ply_path.exists():
            raise FileNotFoundError(
                f"direct_point_source={self.direct_point_source} requested, "
                f"but point PLY does not exist: {ply_path}"
            )
        points, labels = read_ply_xyz_labels(ply_path)
        if points.numel() == 0:
            raise ValueError(f"Direct point PLY has no vertices: {ply_path}")
        points = points.to(device=self.device, dtype=torch.float32).contiguous()
        if labels is not None:
            self.direct_point_pool_labels = labels.to(
                device=self.device, dtype=torch.long
            ).contiguous()
        self._log(
            f"Direct point supervision source: {self.direct_point_source} "
            f"({points.shape[0]} points from {ply_path}, "
            f"labels={'yes' if self.direct_point_pool_labels is not None else 'no'})"
        )
        return points

    def _load_direct_point_teacher_cache(self, config: RadioGSConfig) -> None:
        """Load optional pre-aggregated multiview RADIO teacher point features."""
        cache_raw = str(getattr(config, "direct_point_teacher_cache", "") or "")
        if not cache_raw:
            return
        scene_root = resolve_scene_root(config)
        scene = getattr(config, "scene", Path(scene_root).name)
        cache_path = Path(cache_raw.format(scene=scene))
        if not cache_path.is_absolute():
            cache_path = Path.cwd() / cache_path
        if not cache_path.exists():
            raise FileNotFoundError(f"direct_point_teacher_cache not found: {cache_path}")

        payload = load_training_tensor_cache(
            cache_path,
            map_location="cpu",
            purpose="direct point teacher cache",
        )
        if not isinstance(payload, dict):
            raise ValueError(f"Teacher cache must be a dict payload: {cache_path}")
        requested_key = str(
            getattr(config, "direct_point_teacher_cache_feature_key", "")
            or self.direct_point_teacher_cache_feature_key
            or ""
        )
        if requested_key:
            if requested_key not in payload:
                raise KeyError(
                    f"Teacher cache missing requested feature key '{requested_key}': "
                    f"{cache_path}"
                )
            feature_key = requested_key
        elif "features" in payload:
            feature_key = "features"
        elif "summary_features" in payload:
            feature_key = "summary_features"
        else:
            raise KeyError(
                "Teacher cache missing feature tensor: expected 'features' or "
                f"'summary_features' in {cache_path}"
            )

        requested_space = str(
            getattr(config, "direct_point_teacher_cache_feature_space", "")
            or self.direct_point_teacher_cache_feature_space
            or ""
        )
        metadata = payload.get("metadata") or {}
        payload_space = str(payload.get("feature_space") or metadata.get("feature_space") or "")
        if requested_space:
            feature_space = requested_space
        elif payload_space:
            feature_space = payload_space
        elif feature_key == "summary_features":
            feature_space = "siglip_summary"
        else:
            feature_space = "radio"
        aliases = {
            "": "radio",
            "radio": "radio",
            "teacher": "radio",
            "teacher_1280": "radio",
            "siglip": "siglip_summary",
            "siglip2": "siglip_summary",
            "siglip_summary": "siglip_summary",
            "summary": "siglip_summary",
        }
        feature_space = aliases.get(feature_space.lower(), feature_space.lower())
        if feature_space not in {"radio", "siglip_summary"}:
            raise ValueError(
                "direct_point_teacher_cache_feature_space must be 'radio' or "
                f"'siglip_summary', got {feature_space!r}"
            )
        if feature_space == "siglip_summary":
            summary_losses_enabled = any(
                float(value) > 0.0
                for value in (
                    self.direct_point_summary_alignment_weight,
                    self.direct_point_summary_adapter_weight,
                    self.direct_point_text_distill_weight,
                    self.direct_point_text_pseudo_ce_weight,
                    self.direct_point_text_contrast_weight,
                    self.direct_point_adapter_text_distill_weight,
                    self.direct_point_adapter_text_pseudo_ce_weight,
                    self.direct_point_adapter_text_loss_weight,
                    self.direct_point_adapter_decoder_anchor_weight,
                )
            )
            if not summary_losses_enabled:
                raise ValueError(
                    "direct_point_teacher_cache is in SigLIP summary space but no "
                    "direct-point summary/text/adapter loss is enabled; this would "
                    "make direct point supervision a no-op."
                )

        features = torch.as_tensor(payload[feature_key], dtype=torch.float32)
        if features.dim() != 2:
            raise ValueError(
                f"Teacher cache {feature_key} must be [N,C], got {tuple(features.shape)}"
            )
        num_points = int(features.shape[0])
        if bool(getattr(config, "direct_point_teacher_cache_require_xyz_alignment", True)):
            get_scaling = getattr(self.model, "get_scaling", None)
            get_rotation = getattr(self.model, "get_rotation", None)
            get_opacity = getattr(self.model, "get_opacity", None)
            report = audit_direct_point_teacher_cache_alignment_for_training(
                payload,
                self.model.get_xyz().detach(),
                model_scales=get_scaling().detach() if callable(get_scaling) else None,
                model_rotations=get_rotation().detach() if callable(get_rotation) else None,
                model_opacities=get_opacity().detach() if callable(get_opacity) else None,
                direct_point_source=self.direct_point_source,
                direct_point_query_mode=self.direct_point_query_mode,
                cache_path=str(cache_path),
                fail_max_l2=float(
                    getattr(config, "direct_point_teacher_cache_fail_max_l2", 1e-5)
                ),
            )
            self.direct_point_teacher_cache_alignment_report = report
            if report.get("status") != "skipped":
                self._log(
                    "Direct point teacher cache row alignment: "
                    f"status={report.get('status')} "
                    f"max_l2={report.get('max_l2', 0.0):.3e} "
                    f"mean_l2={report.get('mean_l2', 0.0):.3e} "
                    f"hash_match={report.get('xyz_sha256_match')}"
                )
        xyz_payload = payload.get("xyz")
        if xyz_payload is not None:
            xyz = torch.as_tensor(xyz_payload, dtype=torch.float32)
            if xyz.shape != (num_points, 3):
                raise ValueError(
                    "Teacher cache xyz must match features as [N,3], got "
                    f"{tuple(xyz.shape)} for {tuple(features.shape)}"
                )
            xyz = xyz.to(device=self.device, dtype=torch.float32).contiguous()
            if self.direct_point_pool is None:
                self.direct_point_pool = xyz
            elif int(self.direct_point_pool.shape[0]) != num_points:
                raise ValueError(
                    "Teacher cache point count does not match direct point pool: "
                    f"{num_points} vs {int(self.direct_point_pool.shape[0])}"
                )

        labels_payload = payload.get("labels")
        if labels_payload is not None and self.direct_point_pool_labels is None:
            labels = torch.as_tensor(labels_payload, dtype=torch.long)
            if labels.shape[0] != num_points:
                raise ValueError(
                    f"Teacher cache labels length {labels.shape[0]} does not match {num_points}"
                )
            self.direct_point_pool_labels = labels.to(self.device).contiguous()

        valid_payload = payload.get("valid")
        if valid_payload is None:
            valid = torch.ones(num_points, dtype=torch.bool)
        else:
            valid = torch.as_tensor(valid_payload, dtype=torch.bool)
            if valid.shape[0] != num_points:
                raise ValueError(
                    f"Teacher cache valid length {valid.shape[0]} does not match {num_points}"
                )
        counts_payload = payload.get("view_counts")
        if counts_payload is None:
            view_counts = valid.float()
        else:
            view_counts = torch.as_tensor(counts_payload, dtype=torch.float32)
            if view_counts.shape[0] != num_points:
                raise ValueError(
                    "Teacher cache view_counts length "
                    f"{view_counts.shape[0]} does not match {num_points}"
                )

        self.direct_point_teacher_features = features.to(self.device).contiguous()
        self.direct_point_teacher_cache_feature_key = feature_key
        self.direct_point_teacher_feature_space = feature_space
        self.direct_point_teacher_valid = valid.to(self.device).contiguous()
        self.direct_point_teacher_view_counts = view_counts.to(self.device).contiguous()
        positive_counts = self.direct_point_teacher_view_counts[
            self.direct_point_teacher_view_counts > 0
        ]
        self.point_summary_adapter_view_count_max = (
            positive_counts.max().detach()
            if positive_counts.numel() > 0
            else torch.tensor(1.0, device=self.device)
        )
        direct_readout_mode = (
            "gaussian"
            if self.direct_point_query_mode == "gaussian_index"
            else str(self.direct_point_query_mode)
        )
        valid_mask_mode = "teacher_cache"
        context_features = str(
            getattr(self, "point_summary_adapter_context_features", "") or ""
        )
        self.point_summary_adapter_metadata.update(
            {
                "teacher_cache": str(cache_path),
                "teacher_cache_feature_key": feature_key,
                "teacher_feature_space": feature_space,
                "teacher_cache_num_points": int(num_points),
                "teacher_cache_feature_dim": int(features.shape[1]),
                "teacher_cache_valid_count": int(valid.sum().item()),
                "direct_point_source": str(self.direct_point_source),
                "direct_point_query_mode": str(self.direct_point_query_mode),
                "direct_point_gaussian_position_mode": str(
                    self.direct_point_gaussian_position_mode
                ),
                "compact_feature_key": str(self.direct_point_feature_key),
                "point_summary_adapter_context_features": context_features,
                "point_summary_adapter_view_count_max": float(
                    self.point_summary_adapter_view_count_max.detach().cpu()
                ),
                "direct_head_contract": {
                    "compact_feature_key": str(self.direct_point_feature_key),
                    "direct_readout_mode": direct_readout_mode,
                    "point_summary_adapter_blend_alpha": 1.0,
                    "point_summary_adapter_valid_mask_mode": valid_mask_mode,
                    "point_summary_adapter_context_features": context_features,
                    "teacher_feature_space": feature_space,
                    "teacher_cache_feature_key": feature_key,
                    "teacher_cache": str(cache_path),
                    "direct_point_source": str(self.direct_point_source),
                    "direct_point_query_mode": str(self.direct_point_query_mode),
                    "direct_point_gaussian_position_mode": str(
                        self.direct_point_gaussian_position_mode
                    ),
                },
            }
        )
        self._log(
            "Loaded direct point teacher cache: "
            f"{cache_path} ({num_points} points, dim={features.shape[1]}, "
            f"feature_key={feature_key}, feature_space={feature_space}, "
            f"valid={int(valid.sum())}/{num_points})"
        )

    def build_dataset(
        self, config: RadioGSConfig
    ) -> Tuple[Dataset, Dataset]:
        dataset_type = resolve_dataset_type(config)
        scene_root = resolve_scene_root(config)
        train_split = getattr(config, "train_split", "Sequence_1")
        val_split = getattr(config, "val_split", "Sequence_2")
        mixed_split = getattr(config, "mixed_split", False)

        # RGB guide setup
        feature_size = (
            getattr(config, "feature_height", 30),
            getattr(config, "feature_width", 40),
        )
        rgb_dir_train = str(resolve_split_data_dir(config, "train", "rgb")) if self.refiner_rgb_guide and resolve_split_data_dir(config, "train", "rgb") is not None else None
        rgb_dir_val = str(resolve_split_data_dir(config, "val", "rgb")) if self.refiner_rgb_guide and resolve_split_data_dir(config, "val", "rgb") is not None else None

        if dataset_type != "replica":
            train_feature_dir = resolve_split_feature_dir(config, "train")
            val_feature_dir = resolve_split_feature_dir(config, "val")
            train_pose_file, train_pose_dir = resolve_split_pose_source(config, "train")
            val_pose_file, val_pose_dir = resolve_split_pose_source(config, "val")
            train_depth_dir = resolve_split_data_dir(config, "train", "depth")
            val_depth_dir = resolve_split_data_dir(config, "val", "depth")
            train_semantics_dir = resolve_split_data_dir(config, "train", "semantics")
            val_semantics_dir = resolve_split_data_dir(config, "val", "semantics")
            train_frame_ids = resolve_split_frame_ids(config, "train")
            val_frame_ids = resolve_split_frame_ids(config, "val")

            if train_frame_ids is not None and val_frame_ids is not None:
                train_ds = SimpleRadioDataset(
                    feature_dir=str(train_feature_dir),
                    pose_file=train_pose_file,
                    pose_dir=train_pose_dir,
                    depth_dir=str(train_depth_dir) if train_depth_dir else None,
                    semantics_dir=str(train_semantics_dir) if train_semantics_dir else None,
                    rgb_dir=rgb_dir_train,
                    feature_size=feature_size,
                    split="train",
                    dataset_type=dataset_type,
                    frame_ids=train_frame_ids,
                )
                val_ds = SimpleRadioDataset(
                    feature_dir=str(val_feature_dir),
                    pose_file=val_pose_file,
                    pose_dir=val_pose_dir,
                    depth_dir=str(val_depth_dir) if val_depth_dir else None,
                    semantics_dir=str(val_semantics_dir) if val_semantics_dir else None,
                    rgb_dir=rgb_dir_val,
                    feature_size=feature_size,
                    split="val",
                    dataset_type=dataset_type,
                    frame_ids=val_frame_ids,
                )
                self._log(
                    f"{dataset_type} split lists: Train {len(train_ds)} frames | Val {len(val_ds)} frames"
                )
                return train_ds, val_ds

            full_ds = SimpleRadioDataset(
                feature_dir=str(train_feature_dir),
                pose_file=train_pose_file,
                pose_dir=train_pose_dir,
                depth_dir=str(train_depth_dir) if train_depth_dir else None,
                semantics_dir=str(train_semantics_dir) if train_semantics_dir else None,
                rgb_dir=rgb_dir_train,
                feature_size=feature_size,
                split="train",
                dataset_type=dataset_type,
            )
            train_ratio = getattr(config, "mixed_train_ratio", 0.8)
            train_size = int(train_ratio * len(full_ds))
            val_size = len(full_ds) - train_size
            seed = getattr(config, "mixed_seed", 42)
            gen = torch.Generator().manual_seed(seed)
            train_ds, val_ds = torch.utils.data.random_split(
                full_ds, [train_size, val_size], generator=gen
            )
            self._log(
                f"{dataset_type} random split: {len(full_ds)} total → Train: {train_size} | "
                f"Val: {val_size} (ratio={train_ratio}, seed={seed})"
            )
            return train_ds, val_ds

        if mixed_split:
            # Merge both sequences and random 80/20 split
            ds_seq1 = SimpleRadioDataset(
                feature_dir=str(resolve_split_feature_dir(config, "train")),
                pose_file=resolve_split_pose_source(config, "train")[0],
                pose_dir=resolve_split_pose_source(config, "train")[1],
                depth_dir=str(resolve_split_data_dir(config, "train", "depth")) if resolve_split_data_dir(config, "train", "depth") else None,
                semantics_dir=str(resolve_split_data_dir(config, "train", "semantics")) if resolve_split_data_dir(config, "train", "semantics") else None,
                rgb_dir=rgb_dir_train,
                feature_size=feature_size,
                split="train",
                dataset_type=dataset_type,
            )
            ds_seq2 = SimpleRadioDataset(
                feature_dir=str(resolve_split_feature_dir(config, "val")),
                pose_file=resolve_split_pose_source(config, "val")[0],
                pose_dir=resolve_split_pose_source(config, "val")[1],
                depth_dir=str(resolve_split_data_dir(config, "val", "depth")) if resolve_split_data_dir(config, "val", "depth") else None,
                semantics_dir=str(resolve_split_data_dir(config, "val", "semantics")) if resolve_split_data_dir(config, "val", "semantics") else None,
                rgb_dir=rgb_dir_val,
                feature_size=feature_size,
                split="train",
                dataset_type=dataset_type,
            )
            combined = ConcatDataset([ds_seq1, ds_seq2])
            total = len(combined)
            train_ratio = getattr(config, "mixed_train_ratio", 0.8)
            train_size = int(train_ratio * total)
            val_size = total - train_size
            seed = getattr(config, "mixed_seed", 42)
            gen = torch.Generator().manual_seed(seed)
            train_ds, val_ds = torch.utils.data.random_split(
                combined, [train_size, val_size], generator=gen
            )
            self._log(
                f"Mixed split: {total} total → Train: {train_size} | Val: {val_size} "
                f"(ratio={train_ratio}, seed={seed})"
            )
            return train_ds, val_ds

        train_ds = SimpleRadioDataset(
            feature_dir=str(resolve_split_feature_dir(config, "train")),
            pose_file=resolve_split_pose_source(config, "train")[0],
            pose_dir=resolve_split_pose_source(config, "train")[1],
            depth_dir=str(resolve_split_data_dir(config, "train", "depth")) if resolve_split_data_dir(config, "train", "depth") else None,
            semantics_dir=str(resolve_split_data_dir(config, "train", "semantics")) if resolve_split_data_dir(config, "train", "semantics") else None,
            rgb_dir=rgb_dir_train,
            feature_size=feature_size,
            split="train",
            dataset_type=dataset_type,
        )
        val_ds = SimpleRadioDataset(
            feature_dir=str(resolve_split_feature_dir(config, "val")),
            pose_file=resolve_split_pose_source(config, "val")[0],
            pose_dir=resolve_split_pose_source(config, "val")[1],
            depth_dir=str(resolve_split_data_dir(config, "val", "depth")) if resolve_split_data_dir(config, "val", "depth") else None,
            semantics_dir=str(resolve_split_data_dir(config, "val", "semantics")) if resolve_split_data_dir(config, "val", "semantics") else None,
            rgb_dir=rgb_dir_val,
            feature_size=feature_size,
            split="val",
            dataset_type=dataset_type,
        )
        self._log(f"Train: {len(train_ds)} frames  |  Val: {len(val_ds)} frames")
        return train_ds, val_ds

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        if self.train_mode != "latent":
            self.codec.train()
        self.sharpener.train()
        if self.use_refiner and self.refiner is not None:
            self.refiner.train()
        if self.depth_head is not None:
            self.depth_head.train()
        if self.seg_head is not None:
            self.seg_head.train()

        loss_accum = {
            "total": 0.0,
            "distill": 0.0,
            "compact": 0.0,
            "tv": 0.0,
            "gradient": 0.0,
            "depth_feat": 0.0,
            "geom_edge": 0.0,
            "boundary": 0.0,
            "sem_aux": 0.0,
            "sem_adaptor_reg": 0.0,
            "quality": 0.0,
            "visibility": 0.0,
            "rgb": 0.0,
            "depth_gt": 0.0,
            "depth_geom": 0.0,
            "frozen_depth": 0.0,
            "seg_aux": 0.0,
            "frozen_seg": 0.0,
            "direct_point": 0.0,
            "direct_point_valid": 0.0,
            "direct_point_feature_distill": 0.0,
            "direct_point_summary": 0.0,
            "direct_point_relation": 0.0,
            "direct_point_text": 0.0,
            "direct_point_text_valid": 0.0,
            "direct_point_text_acc": 0.0,
            "direct_point_adapter_text": 0.0,
            "direct_point_adapter_text_valid": 0.0,
            "direct_point_adapter_text_acc": 0.0,
            "direct_point_text_distill": 0.0,
            "direct_point_text_distill_valid": 0.0,
            "direct_point_text_distill_teacher_conf": 0.0,
            "direct_point_text_distill_agreement": 0.0,
            "direct_point_text_pseudo_ce": 0.0,
            "direct_point_text_pseudo_ce_valid": 0.0,
            "direct_point_text_pseudo_ce_teacher_conf": 0.0,
            "direct_point_text_pseudo_ce_agreement": 0.0,
            "direct_point_text_contrast": 0.0,
            "direct_point_text_contrast_valid": 0.0,
            "direct_point_text_contrast_teacher_conf": 0.0,
            "direct_point_text_contrast_agreement": 0.0,
            "direct_point_render_consistency": 0.0,
            "direct_point_render_consistency_valid": 0.0,
            "direct_point_adapter_text_distill": 0.0,
            "direct_point_adapter_text_distill_valid": 0.0,
            "direct_point_adapter_text_distill_teacher_conf": 0.0,
            "direct_point_adapter_text_distill_agreement": 0.0,
            "direct_point_adapter_text_pseudo_ce": 0.0,
            "direct_point_adapter_text_pseudo_ce_valid": 0.0,
            "direct_point_adapter_text_pseudo_ce_teacher_conf": 0.0,
            "direct_point_adapter_text_pseudo_ce_agreement": 0.0,
            "direct_point_adapter_decoder_anchor": 0.0,
            "direct_point_query_support_distill": 0.0,
            "direct_point_query_support_distill_valid": 0.0,
            "direct_point_query_support_distill_teacher_conf": 0.0,
            "direct_point_query_support_distill_top1": 0.0,
            "direct_point_proposal_consistency": 0.0,
            "direct_point_proposal_consistency_valid": 0.0,
            "direct_point_proposal_contrast": 0.0,
            "direct_point_proposal_contrast_valid": 0.0,
            "direct_point_proposal_contrast_num_proposals": 0.0,
            "direct_point_view_weight_mean": 0.0,
            "direct_point_view_weight_max": 0.0,
            "siglip_align": 0.0,
            "summary_align": 0.0,
            "text_heatmaps": 0.0,
            "radio_adaptors": 0.0,
            "radio_relations": 0.0,
            "radio_local_affinity": 0.0,
            "radio_token_contrast": 0.0,
            "radio_regions": 0.0,
            "radio_mask_logits": 0.0,
            "radio_cross_views": 0.0,
            "radio_cross_view_propagation": 0.0,
            "radio_cross_view_mask_propagation": 0.0,
            "foundation_cache": 0.0,
            "samclip_mask": 0.0,
            "samclip_mask_proto": 0.0,
            "samclip_mask_contrast": 0.0,
            "samclip_mask_background": 0.0,
            "samclip_mask_regions": 0.0,
            "ground_query": 0.0,
            "ground_query_acc": 0.0,
            "ground_query_valid": 0.0,
        }
        cos_accum = 0.0
        n_batches = 0
        log_every = getattr(self.cfg, "log_every", 100)

        pbar = tqdm(
            self.train_loader,
            desc=f"Train E{epoch:03d}",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in pbar:
            gt_features = batch["radio_features"].to(self.device)   # [B, C, Hp, Wp]
            pose_w2c = batch["pose_w2c"].to(self.device)         # [B, 4, 4]
            if self._is_hybrid:
                gt_features = gt_features.float()
                pose_w2c = pose_w2c.float()

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=not self._is_hybrid):
                # Render compact features (and optionally RGB) from 3DGS
                rendered_rgb = None
                l_rgb = torch.tensor(0.0, device=self.device)
                hybrid_aux = None

                if self.self_guided and self.train_sh:
                    # Joint rendering with SH training: features + RGB, backprop RGB loss
                    result = self.renderer.render_features_and_rgb(
                        self.model, pose_w2c
                    )
                    result = self._canonicalize_render_result(
                        result,
                        batch_size=pose_w2c.shape[0],
                        spatial_size=result["feature_map"].shape[-2:],
                    )
                    rendered_compact = result["feature_map"]
                    rendered_rgb = result["rgb"]

                    gt_rgb = batch.get("rgb_guide")
                    if gt_rgb is not None:
                        gt_rgb = gt_rgb.to(self.device)
                        l_rgb = F.l1_loss(rendered_rgb.float(), gt_rgb.float())
                elif self.self_guided:
                    # Self-guided with frozen SH: render RGB as guide (no gradient)
                    result = self.renderer.render_features_and_rgb(
                        self.model, pose_w2c
                    )
                    result = self._canonicalize_render_result(
                        result,
                        batch_size=pose_w2c.shape[0],
                        spatial_size=result["feature_map"].shape[-2:],
                    )
                    rendered_compact = result["feature_map"]
                    rendered_rgb = result["rgb"].detach()
                else:
                    result = self.renderer.render_features_batch(
                        self.model, pose_w2c
                    )
                    result = self._canonicalize_render_result(
                        result,
                        batch_size=pose_w2c.shape[0],
                        spatial_size=result["feature_map"].shape[-2:],
                    )
                    rendered_compact = result["feature_map"]

                # Sharpen rendered features
                rendered_compact = self.sharpener(rendered_compact)

                # Apply screen-space refiner if enabled
                if self.use_refiner and self.refiner is not None:
                    guide = self._build_guide(batch, result, rendered_rgb)
                    rendered_compact = self.refiner(rendered_compact, guide=guide)

                # Hybrid architecture: decode via hash grid + fusion
                if self._is_hybrid:
                    from radio_gs.models.hybrid_gaussian import unproject_depth_to_positions
                    Bf, _, Hf, Wf = rendered_compact.shape
                    depth_map = self._canonicalize_spatial_map(
                        result.get("depth_map"),
                        batch_size=Bf,
                        spatial_size=(Hf, Wf),
                    )
                    position_map = unproject_depth_to_positions(
                        depth_map, pose_w2c.float(), self.renderer.K.float(),
                        depth_map.shape[1], depth_map.shape[2],
                    )
                    position_map = self._normalize_positions(position_map)
                    decode_result = self.model.decode_screen_space(
                        rendered_compact.float(),
                        position_map,
                        return_aux=self.hybrid_decoupled_heads,
                        depth_map=depth_map,
                    )
                    if self.hybrid_decoupled_heads:
                        hybrid_aux = decode_result
                        rendered_compact = decode_result["fused"]
                    else:
                        rendered_compact = decode_result

                if self.train_mode == "latent":
                    # LATENT MODE: gt_features are already 64d (pre-encoded)
                    gt_compact = gt_features
                    if gt_compact.shape[-2:] != rendered_compact.shape[-2:]:
                        gt_compact = F.interpolate(
                            gt_compact,
                            size=rendered_compact.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    # Primary loss: cosine + L2 in latent space
                    l_cos = 1.0 - F.cosine_similarity(
                        rendered_compact.float().flatten(2),
                        gt_compact.float().flatten(2),
                        dim=1,
                    ).mean()
                    l_l2 = F.mse_loss(rendered_compact.float(), gt_compact.float())
                    l2_w = getattr(self.cfg, "l2_weight", 1.0)
                    cos_w = getattr(self.cfg, "cosine_weight", 0.5)
                    l_distill = l2_w * l_l2 + cos_w * l_cos

                    l_compact = torch.tensor(0.0, device=self.device)
                    decoded_for_depth = self.codec.decoder(rendered_compact)

                    # Feature norm regularization
                    l_feat_norm = torch.tensor(0.0, device=self.device)
                    if self.feat_norm_weight > 0:
                        feat_norms = rendered_compact.float().norm(dim=1).mean()
                        gt_norms = gt_compact.float().norm(dim=1).mean()
                        l_feat_norm = (feat_norms - gt_norms).abs()

                else:
                    # DECODED MODE (legacy V1/V2): compare in 1280d space
                    gt_radio = gt_features
                    with torch.no_grad():
                        gt_compact = self.codec.encoder(gt_radio)

                    decoded = self.codec.decoder(rendered_compact)

                    if decoded.shape[-2:] != gt_radio.shape[-2:]:
                        gt_radio_rs = F.interpolate(
                            gt_radio,
                            size=decoded.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    else:
                        gt_radio_rs = gt_radio

                    distill_dict = self.distill_loss_fn(decoded, gt_radio_rs)
                    l_distill = distill_dict["total"]

                    if gt_compact.shape[-2:] != rendered_compact.shape[-2:]:
                        gt_compact_rs = F.interpolate(
                            gt_compact,
                            size=rendered_compact.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    else:
                        gt_compact_rs = gt_compact
                    l_compact = F.mse_loss(rendered_compact, gt_compact_rs)
                    l_feat_norm = torch.tensor(0.0, device=self.device)
                    decoded_for_depth = decoded

                depth_losses = self._compute_depth_aux_losses(
                    batch=batch,
                    render_result=result,
                    decoded=decoded_for_depth,
                )
                frozen_depth_losses = self._compute_frozen_depth_loss(
                    render_result=result,
                    decoded=decoded_for_depth,
                    teacher_features=gt_radio_rs if self.train_mode != "latent" else None,
                )
                seg_losses = self._compute_seg_aux_losses(
                    batch=batch,
                    decoded=decoded_for_depth if self.train_mode != "latent" else None,
                )
                frozen_seg_losses = self._compute_frozen_seg_loss(
                    decoded=decoded_for_depth if self.train_mode != "latent" else None,
                    teacher_features=gt_radio_rs if self.train_mode != "latent" else None,
                )
                l_siglip = self._compute_siglip_alignment_loss(
                    decoded=decoded_for_depth if self.train_mode != "latent" else None,
                    target=gt_radio_rs if self.train_mode != "latent" else None,
                )
                l_summary = self._compute_summary_alignment_loss(
                    decoded=decoded_for_depth if self.train_mode != "latent" else None,
                    target=gt_radio_rs if self.train_mode != "latent" else None,
                )
                l_text_heatmaps = self._compute_text_heatmap_distill_loss(
                    decoded=decoded_for_depth if self.train_mode != "latent" else None,
                    target=gt_radio_rs if self.train_mode != "latent" else None,
                )
                l_radio_adaptors = self._compute_radio_adaptor_alignment_loss(
                    decoded=decoded_for_depth if self.train_mode != "latent" else None,
                    target=gt_radio_rs if self.train_mode != "latent" else None,
                )
                l_radio_relations = self._compute_radio_adaptor_relation_loss(
                    decoded=decoded_for_depth if self.train_mode != "latent" else None,
                    target=gt_radio_rs if self.train_mode != "latent" else None,
                )
                l_radio_local_affinity = (
                    self._compute_radio_adaptor_local_affinity_loss(
                        decoded=decoded_for_depth
                        if self.train_mode != "latent"
                        else None,
                        target=gt_radio_rs if self.train_mode != "latent" else None,
                    )
                )
                l_radio_token_contrast = self._compute_radio_adaptor_token_contrast_loss(
                    decoded=decoded_for_depth if self.train_mode != "latent" else None,
                    target=gt_radio_rs if self.train_mode != "latent" else None,
                )
                l_radio_peak_background = self._compute_radio_adaptor_peak_background_loss(
                    decoded=decoded_for_depth if self.train_mode != "latent" else None,
                    target=gt_radio_rs if self.train_mode != "latent" else None,
                )
                l_radio_regions = self._compute_radio_adaptor_region_loss(
                    decoded=decoded_for_depth if self.train_mode != "latent" else None,
                    target=gt_radio_rs if self.train_mode != "latent" else None,
                )
                l_radio_mask_logits = self._compute_radio_adaptor_mask_logit_loss(
                    decoded=decoded_for_depth if self.train_mode != "latent" else None,
                    target=gt_radio_rs if self.train_mode != "latent" else None,
                )
                l_radio_cross_views = self._compute_radio_adaptor_cross_view_loss(
                    decoded=decoded_for_depth if self.train_mode != "latent" else None,
                    target=gt_radio_rs if self.train_mode != "latent" else None,
                )
                l_radio_cross_view_propagation = (
                    self._compute_radio_adaptor_cross_view_propagation_loss(
                        decoded=decoded_for_depth if self.train_mode != "latent" else None,
                        target=gt_radio_rs if self.train_mode != "latent" else None,
                    )
                )
                l_radio_cross_view_mask_propagation = (
                    self._compute_radio_adaptor_cross_view_mask_propagation_loss(
                        decoded=decoded_for_depth if self.train_mode != "latent" else None,
                        target=gt_radio_rs if self.train_mode != "latent" else None,
                    )
                )
                foundation_cache_stats = self._compute_foundation_cache_loss(
                    batch=batch,
                    decoded=decoded_for_depth if self.train_mode != "latent" else None,
                )
                l_foundation_cache = foundation_cache_stats["loss"]
                samclip_mask_stats = self._compute_samclip_mask_loss(
                    batch=batch,
                    decoded=decoded_for_depth,
                )
                l_samclip_mask = samclip_mask_stats["loss"]

                l_tv = self.tv_loss_fn(rendered_compact)

                # Gradient-weighted loss for sharper boundaries
                l_gradient = torch.tensor(0.0, device=self.device)
                if self.gradient_loss_fn is not None and decoded_for_depth is not None:
                    gt_for_grad = gt_radio_rs if self.train_mode != "latent" else gt_compact
                    pred_for_grad = decoded_for_depth if self.train_mode != "latent" else rendered_compact
                    l_gradient = self.gradient_loss_fn(pred_for_grad, gt_for_grad)

                # Depth-guided feature smoothness loss
                l_depth_feat = torch.tensor(0.0, device=self.device)
                geom_depth = result.get("depth_map")
                alpha_for_edges = result.get("alpha_map")
                if self.depth_guided_feat_loss is not None and geom_depth is not None:
                    feat_for_smooth = (
                        hybrid_aux["geometry"]
                        if hybrid_aux is not None and "geometry" in hybrid_aux
                        else rendered_compact
                    )
                    gd = self._canonicalize_spatial_map(
                        geom_depth,
                        batch_size=feat_for_smooth.shape[0],
                        spatial_size=feat_for_smooth.shape[-2:],
                        add_channel_dim=True,
                    )
                    l_depth_feat = self.depth_guided_feat_loss(feat_for_smooth, gd)

                # Boundary-aware feature loss
                l_boundary = torch.tensor(0.0, device=self.device)
                if self.boundary_aware_loss_fn is not None and geom_depth is not None:
                    pred_feat = decoded_for_depth if self.train_mode != "latent" else rendered_compact
                    gt_feat = gt_radio_rs if self.train_mode != "latent" else gt_compact
                    gd_ba = self._canonicalize_spatial_map(
                        geom_depth,
                        batch_size=pred_feat.shape[0],
                        spatial_size=pred_feat.shape[-2:],
                        add_channel_dim=True,
                    )
                    alpha_ba = self._canonicalize_spatial_map(
                        alpha_for_edges,
                        batch_size=pred_feat.shape[0],
                        spatial_size=pred_feat.shape[-2:],
                        add_channel_dim=True,
                    )
                    l_boundary = self.boundary_aware_loss_fn(pred_feat, gt_feat, gd_ba, alpha_ba)

                l_geom_edge = torch.tensor(0.0, device=self.device)
                l_semantic_aux = torch.tensor(0.0, device=self.device)
                l_semantic_adaptor_reg = torch.tensor(0.0, device=self.device)
                l_quality = torch.tensor(0.0, device=self.device)
                l_visibility = torch.tensor(0.0, device=self.device)
                l_direct_point = torch.tensor(0.0, device=self.device)
                direct_point_valid = torch.tensor(0.0, device=self.device)
                direct_point_feature_distill = torch.tensor(0.0, device=self.device)
                direct_point_summary = torch.tensor(0.0, device=self.device)
                direct_point_relation = torch.tensor(0.0, device=self.device)
                direct_point_text = torch.tensor(0.0, device=self.device)
                direct_point_text_valid = torch.tensor(0.0, device=self.device)
                direct_point_text_acc = torch.tensor(0.0, device=self.device)
                direct_point_adapter_text = torch.tensor(0.0, device=self.device)
                direct_point_adapter_text_valid = torch.tensor(0.0, device=self.device)
                direct_point_adapter_text_acc = torch.tensor(0.0, device=self.device)
                direct_point_text_distill = torch.tensor(0.0, device=self.device)
                direct_point_text_distill_valid = torch.tensor(0.0, device=self.device)
                direct_point_text_distill_teacher_conf = torch.tensor(0.0, device=self.device)
                direct_point_text_distill_agreement = torch.tensor(0.0, device=self.device)
                direct_point_text_pseudo_ce = torch.tensor(0.0, device=self.device)
                direct_point_text_pseudo_ce_valid = torch.tensor(0.0, device=self.device)
                direct_point_text_pseudo_ce_teacher_conf = torch.tensor(0.0, device=self.device)
                direct_point_text_pseudo_ce_agreement = torch.tensor(0.0, device=self.device)
                direct_point_text_contrast = torch.tensor(0.0, device=self.device)
                direct_point_text_contrast_valid = torch.tensor(0.0, device=self.device)
                direct_point_text_contrast_teacher_conf = torch.tensor(0.0, device=self.device)
                direct_point_text_contrast_agreement = torch.tensor(0.0, device=self.device)
                direct_point_render_consistency = torch.tensor(0.0, device=self.device)
                direct_point_render_consistency_valid = torch.tensor(0.0, device=self.device)
                direct_point_adapter_text_distill = torch.tensor(0.0, device=self.device)
                direct_point_adapter_text_distill_valid = torch.tensor(0.0, device=self.device)
                direct_point_adapter_text_distill_teacher_conf = torch.tensor(0.0, device=self.device)
                direct_point_adapter_text_distill_agreement = torch.tensor(0.0, device=self.device)
                direct_point_adapter_text_pseudo_ce = torch.tensor(0.0, device=self.device)
                direct_point_adapter_text_pseudo_ce_valid = torch.tensor(0.0, device=self.device)
                direct_point_adapter_text_pseudo_ce_teacher_conf = torch.tensor(0.0, device=self.device)
                direct_point_adapter_text_pseudo_ce_agreement = torch.tensor(0.0, device=self.device)
                direct_point_adapter_decoder_anchor = torch.tensor(0.0, device=self.device)
                direct_point_query_support_distill = torch.tensor(0.0, device=self.device)
                direct_point_query_support_distill_valid = torch.tensor(0.0, device=self.device)
                direct_point_query_support_distill_teacher_conf = torch.tensor(0.0, device=self.device)
                direct_point_query_support_distill_top1 = torch.tensor(0.0, device=self.device)
                direct_point_proposal_consistency = torch.tensor(0.0, device=self.device)
                direct_point_proposal_consistency_valid = torch.tensor(0.0, device=self.device)
                direct_point_proposal_contrast = torch.tensor(0.0, device=self.device)
                direct_point_proposal_contrast_valid = torch.tensor(0.0, device=self.device)
                direct_point_proposal_contrast_num_proposals = torch.tensor(0.0, device=self.device)
                direct_point_view_weight_mean = torch.tensor(0.0, device=self.device)
                direct_point_view_weight_max = torch.tensor(0.0, device=self.device)
                l_ground_query = torch.tensor(0.0, device=self.device)
                ground_query_acc = torch.tensor(0.0, device=self.device)
                ground_query_valid = torch.tensor(0.0, device=self.device)
                sem_decoded = None
                if hybrid_aux is not None and self.train_mode != "latent":
                    sem_decoded = self.codec.decoder(hybrid_aux["semantic"])
                    if sem_decoded.shape[-2:] != gt_radio_rs.shape[-2:]:
                        sem_target = F.interpolate(
                            gt_radio_rs,
                            size=sem_decoded.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    else:
                        sem_target = gt_radio_rs
                    if self.hybrid_semantic_aux_weight > 0:
                        l_semantic_aux = self.distill_loss_fn(
                            sem_decoded, sem_target
                        )["total"]
                    if self.grounding_query_loss_weight > 0:
                        ground_query_stats = self._compute_grounding_query_loss(
                            batch=batch,
                            decoded=sem_decoded,
                        )
                        l_ground_query = ground_query_stats["loss"]
                        ground_query_acc = ground_query_stats["accuracy"]
                        ground_query_valid = ground_query_stats["valid_ratio"]
                if hybrid_aux is not None and geom_depth is not None:
                    if self.geometric_edge_loss_fn is not None:
                        gd = self._canonicalize_spatial_map(
                            geom_depth,
                            batch_size=hybrid_aux["geometry"].shape[0],
                            spatial_size=hybrid_aux["geometry"].shape[-2:],
                            add_channel_dim=True,
                        )
                        alpha_map = self._canonicalize_spatial_map(
                            alpha_for_edges,
                            batch_size=hybrid_aux["geometry"].shape[0],
                            spatial_size=hybrid_aux["geometry"].shape[-2:],
                            add_channel_dim=True,
                        )
                        l_geom_edge = self.geometric_edge_loss_fn(
                            hybrid_aux["geometry"], gd, alpha_map,
                        )
                if (
                    hybrid_aux is not None
                    and "semantic_confidence" in hybrid_aux
                    and self.hybrid_semantic_adaptor_reg_weight > 0
                ):
                    l_semantic_adaptor_reg = (
                        hybrid_aux["semantic_confidence"].float() - 1.0
                    ).pow(2).mean()
                if (
                    hybrid_aux is not None
                    and self.train_mode != "latent"
                    and self.quality_loss_weight > 0
                    and "quality_logit" in hybrid_aux
                    and decoded_for_depth is not None
                ):
                    quality_target = cosine_feature_quality_target(
                        decoded_for_depth.detach(),
                        gt_radio_rs.detach(),
                    )
                    if quality_target.shape[-2:] != hybrid_aux["quality_logit"].shape[-2:]:
                        quality_target = F.interpolate(
                            quality_target,
                            size=hybrid_aux["quality_logit"].shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    l_quality = F.binary_cross_entropy_with_logits(
                        hybrid_aux["quality_logit"].float(),
                        quality_target.float(),
                    )
                if (
                    hybrid_aux is not None
                    and self.visibility_loss_weight > 0
                    and "visibility_logit" in hybrid_aux
                    and alpha_for_edges is not None
                ):
                    visibility_target = visibility_target_from_alpha(
                        alpha_for_edges,
                        hybrid_aux["visibility_logit"],
                        threshold=self.visibility_alpha_threshold,
                        binary=self.visibility_target_binary,
                    )
                    l_visibility = F.binary_cross_entropy_with_logits(
                        hybrid_aux["visibility_logit"].float(),
                        visibility_target.float(),
                    )
                if self.direct_point_loss_weight > 0 and self.train_mode != "latent":
                    direct_point_stats = self._compute_direct_point_loss(
                        batch=batch,
                        render_result=result,
                        target_features=gt_radio_rs,
                        rendered_compact=rendered_compact,
                    )
                    l_direct_point = direct_point_stats["loss"]
                    direct_point_valid = direct_point_stats["valid_ratio"]
                    direct_point_feature_distill = direct_point_stats.get(
                        "feature_distill", direct_point_feature_distill
                    )
                    direct_point_summary = direct_point_stats.get(
                        "summary", direct_point_summary
                    )
                    direct_point_relation = direct_point_stats.get(
                        "relation", direct_point_relation
                    )
                    direct_point_text = direct_point_stats.get("text", direct_point_text)
                    direct_point_text_valid = direct_point_stats.get(
                        "text_valid_ratio", direct_point_text_valid
                    )
                    direct_point_text_acc = direct_point_stats.get(
                        "text_acc", direct_point_text_acc
                    )
                    direct_point_adapter_text = direct_point_stats.get(
                        "adapter_text", direct_point_adapter_text
                    )
                    direct_point_adapter_text_valid = direct_point_stats.get(
                        "adapter_text_valid_ratio", direct_point_adapter_text_valid
                    )
                    direct_point_adapter_text_acc = direct_point_stats.get(
                        "adapter_text_acc", direct_point_adapter_text_acc
                    )
                    direct_point_text_distill = direct_point_stats.get(
                        "text_distill", direct_point_text_distill
                    )
                    direct_point_text_distill_valid = direct_point_stats.get(
                        "text_distill_valid_ratio", direct_point_text_distill_valid
                    )
                    direct_point_text_distill_teacher_conf = direct_point_stats.get(
                        "text_distill_teacher_conf",
                        direct_point_text_distill_teacher_conf,
                    )
                    direct_point_text_distill_agreement = direct_point_stats.get(
                        "text_distill_agreement",
                        direct_point_text_distill_agreement,
                    )
                    direct_point_text_pseudo_ce = direct_point_stats.get(
                        "text_pseudo_ce",
                        direct_point_text_pseudo_ce,
                    )
                    direct_point_text_pseudo_ce_valid = direct_point_stats.get(
                        "text_pseudo_ce_valid_ratio",
                        direct_point_text_pseudo_ce_valid,
                    )
                    direct_point_text_pseudo_ce_teacher_conf = direct_point_stats.get(
                        "text_pseudo_ce_teacher_conf",
                        direct_point_text_pseudo_ce_teacher_conf,
                    )
                    direct_point_text_pseudo_ce_agreement = direct_point_stats.get(
                        "text_pseudo_ce_agreement",
                        direct_point_text_pseudo_ce_agreement,
                    )
                    direct_point_text_contrast = direct_point_stats.get(
                        "text_contrast",
                        direct_point_text_contrast,
                    )
                    direct_point_text_contrast_valid = direct_point_stats.get(
                        "text_contrast_valid_ratio",
                        direct_point_text_contrast_valid,
                    )
                    direct_point_text_contrast_teacher_conf = direct_point_stats.get(
                        "text_contrast_teacher_conf",
                        direct_point_text_contrast_teacher_conf,
                    )
                    direct_point_text_contrast_agreement = direct_point_stats.get(
                        "text_contrast_agreement",
                        direct_point_text_contrast_agreement,
                    )
                    direct_point_render_consistency = direct_point_stats.get(
                        "render_consistency",
                        direct_point_render_consistency,
                    )
                    direct_point_render_consistency_valid = direct_point_stats.get(
                        "render_consistency_valid_ratio",
                        direct_point_render_consistency_valid,
                    )
                    direct_point_adapter_text_distill = direct_point_stats.get(
                        "adapter_text_distill",
                        direct_point_adapter_text_distill,
                    )
                    direct_point_adapter_text_distill_valid = direct_point_stats.get(
                        "adapter_text_distill_valid_ratio",
                        direct_point_adapter_text_distill_valid,
                    )
                    direct_point_adapter_text_distill_teacher_conf = direct_point_stats.get(
                        "adapter_text_distill_teacher_conf",
                        direct_point_adapter_text_distill_teacher_conf,
                    )
                    direct_point_adapter_text_distill_agreement = direct_point_stats.get(
                        "adapter_text_distill_agreement",
                        direct_point_adapter_text_distill_agreement,
                    )
                    direct_point_adapter_text_pseudo_ce = direct_point_stats.get(
                        "adapter_text_pseudo_ce",
                        direct_point_adapter_text_pseudo_ce,
                    )
                    direct_point_adapter_text_pseudo_ce_valid = direct_point_stats.get(
                        "adapter_text_pseudo_ce_valid_ratio",
                        direct_point_adapter_text_pseudo_ce_valid,
                    )
                    direct_point_adapter_text_pseudo_ce_teacher_conf = direct_point_stats.get(
                        "adapter_text_pseudo_ce_teacher_conf",
                        direct_point_adapter_text_pseudo_ce_teacher_conf,
                    )
                    direct_point_adapter_text_pseudo_ce_agreement = direct_point_stats.get(
                        "adapter_text_pseudo_ce_agreement",
                        direct_point_adapter_text_pseudo_ce_agreement,
                    )
                    direct_point_adapter_decoder_anchor = direct_point_stats.get(
                        "adapter_decoder_anchor",
                        direct_point_adapter_decoder_anchor,
                    )
                    direct_point_query_support_distill = direct_point_stats.get(
                        "query_support_distill",
                        direct_point_query_support_distill,
                    )
                    direct_point_query_support_distill_valid = direct_point_stats.get(
                        "query_support_distill_valid_ratio",
                        direct_point_query_support_distill_valid,
                    )
                    direct_point_query_support_distill_teacher_conf = direct_point_stats.get(
                        "query_support_distill_teacher_conf",
                        direct_point_query_support_distill_teacher_conf,
                    )
                    direct_point_query_support_distill_top1 = direct_point_stats.get(
                        "query_support_distill_top1_agreement",
                        direct_point_query_support_distill_top1,
                    )
                    direct_point_proposal_consistency = direct_point_stats.get(
                        "proposal_consistency",
                        direct_point_proposal_consistency,
                    )
                    direct_point_proposal_consistency_valid = direct_point_stats.get(
                        "proposal_consistency_valid_ratio",
                        direct_point_proposal_consistency_valid,
                    )
                    direct_point_proposal_contrast = direct_point_stats.get(
                        "proposal_contrast",
                        direct_point_proposal_contrast,
                    )
                    direct_point_proposal_contrast_valid = direct_point_stats.get(
                        "proposal_contrast_valid_ratio",
                        direct_point_proposal_contrast_valid,
                    )
                    direct_point_proposal_contrast_num_proposals = direct_point_stats.get(
                        "proposal_contrast_num_proposals",
                        direct_point_proposal_contrast_num_proposals,
                    )
                    direct_point_view_weight_mean = direct_point_stats.get(
                        "view_weight_mean",
                        direct_point_view_weight_mean,
                    )
                    direct_point_view_weight_max = direct_point_stats.get(
                        "view_weight_max",
                        direct_point_view_weight_max,
                    )

                adaptor_w = getattr(self.cfg, "adaptor_weight", 0.1)
                tv_w = getattr(self.cfg, "tv_weight", 0.01)
                loss = l_distill + adaptor_w * l_compact + tv_w * l_tv
                if self.gradient_loss_weight > 0:
                    loss = loss + self.gradient_loss_weight * l_gradient
                if self.depth_guided_feat_weight > 0:
                    loss = loss + self.depth_guided_feat_weight * l_depth_feat
                if self.geometric_edge_loss_weight > 0:
                    loss = loss + self.geometric_edge_loss_weight * l_geom_edge
                if self.boundary_aware_loss_weight > 0:
                    loss = loss + self.boundary_aware_loss_weight * l_boundary
                if self.hybrid_semantic_aux_weight > 0:
                    loss = loss + self.hybrid_semantic_aux_weight * l_semantic_aux
                if self.hybrid_semantic_adaptor_reg_weight > 0:
                    loss = loss + self.hybrid_semantic_adaptor_reg_weight * l_semantic_adaptor_reg
                if self.quality_loss_weight > 0:
                    loss = loss + self.quality_loss_weight * l_quality
                if self.visibility_loss_weight > 0:
                    loss = loss + self.visibility_loss_weight * l_visibility
                if self.direct_point_loss_weight > 0:
                    loss = loss + self.direct_point_loss_weight * l_direct_point
                if self.grounding_query_loss_weight > 0:
                    loss = loss + self.grounding_query_loss_weight * l_ground_query
                if self.feat_norm_weight > 0:
                    loss = loss + self.feat_norm_weight * l_feat_norm
                if self.rgb_loss_weight > 0:
                    loss = loss + self.rgb_loss_weight * l_rgb
                loss = (
                    loss
                    + depth_losses["total"]
                    + frozen_depth_losses["total"]
                    + seg_losses["total"]
                    + frozen_seg_losses["total"]
                    + l_siglip
                    + l_summary
                    + l_text_heatmaps
                    + l_radio_adaptors
                    + l_radio_relations
                    + l_radio_local_affinity
                    + l_radio_token_contrast
                    + l_radio_peak_background
                    + l_radio_regions
                    + l_radio_mask_logits
                    + l_radio_cross_views
                    + l_radio_cross_view_propagation
                    + l_radio_cross_view_mask_propagation
                    + l_foundation_cache
                    + l_samclip_mask
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_clip = getattr(self.cfg, "grad_clip", 10.0)
            nn.utils.clip_grad_norm_(
                self._all_trainable_params(), max_norm=grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Compute monitoring cosine in appropriate space
            with torch.no_grad():
                if self.train_mode == "latent":
                    cos_sim = F.cosine_similarity(
                        rendered_compact.detach().float().flatten(2),
                        gt_compact.detach().float().flatten(2),
                        dim=1,
                    ).mean()
                else:
                    cos_sim = F.cosine_similarity(
                        decoded.detach().float().flatten(2),
                        gt_radio_rs.detach().float().flatten(2),
                        dim=1,
                    ).mean()

            loss_accum["total"] += loss.item()
            loss_accum["distill"] += l_distill.item()
            loss_accum["compact"] += l_compact.item()
            loss_accum["tv"] += l_tv.item()
            loss_accum["gradient"] += l_gradient.item()
            loss_accum["depth_feat"] += l_depth_feat.item()
            loss_accum["geom_edge"] += l_geom_edge.item()
            loss_accum["boundary"] += l_boundary.item()
            loss_accum["sem_aux"] += l_semantic_aux.item()
            loss_accum["sem_adaptor_reg"] += l_semantic_adaptor_reg.item()
            loss_accum["quality"] += l_quality.item()
            loss_accum["visibility"] += l_visibility.item()
            loss_accum["rgb"] += l_rgb.item()
            loss_accum["depth_gt"] += depth_losses["depth_gt"].item()
            loss_accum["depth_geom"] += depth_losses["depth_geom"].item()
            loss_accum["frozen_depth"] += frozen_depth_losses["total"].item()
            loss_accum["seg_aux"] += seg_losses["total"].item()
            loss_accum["frozen_seg"] += frozen_seg_losses["total"].item()
            loss_accum["direct_point"] += l_direct_point.item()
            loss_accum["direct_point_valid"] += direct_point_valid.item()
            loss_accum["direct_point_feature_distill"] += (
                direct_point_feature_distill.item()
            )
            loss_accum["direct_point_summary"] += direct_point_summary.item()
            loss_accum["direct_point_relation"] += direct_point_relation.item()
            loss_accum["direct_point_text"] += direct_point_text.item()
            loss_accum["direct_point_text_valid"] += direct_point_text_valid.item()
            loss_accum["direct_point_text_acc"] += direct_point_text_acc.item()
            loss_accum["direct_point_adapter_text"] += direct_point_adapter_text.item()
            loss_accum["direct_point_adapter_text_valid"] += direct_point_adapter_text_valid.item()
            loss_accum["direct_point_adapter_text_acc"] += direct_point_adapter_text_acc.item()
            loss_accum["direct_point_text_distill"] += direct_point_text_distill.item()
            loss_accum["direct_point_text_distill_valid"] += direct_point_text_distill_valid.item()
            loss_accum["direct_point_text_distill_teacher_conf"] += (
                direct_point_text_distill_teacher_conf.item()
            )
            loss_accum["direct_point_text_distill_agreement"] += (
                direct_point_text_distill_agreement.item()
            )
            loss_accum["direct_point_text_pseudo_ce"] += (
                direct_point_text_pseudo_ce.item()
            )
            loss_accum["direct_point_text_pseudo_ce_valid"] += (
                direct_point_text_pseudo_ce_valid.item()
            )
            loss_accum["direct_point_text_pseudo_ce_teacher_conf"] += (
                direct_point_text_pseudo_ce_teacher_conf.item()
            )
            loss_accum["direct_point_text_pseudo_ce_agreement"] += (
                direct_point_text_pseudo_ce_agreement.item()
            )
            loss_accum["direct_point_text_contrast"] += direct_point_text_contrast.item()
            loss_accum["direct_point_text_contrast_valid"] += (
                direct_point_text_contrast_valid.item()
            )
            loss_accum["direct_point_text_contrast_teacher_conf"] += (
                direct_point_text_contrast_teacher_conf.item()
            )
            loss_accum["direct_point_text_contrast_agreement"] += (
                direct_point_text_contrast_agreement.item()
            )
            loss_accum["direct_point_render_consistency"] += (
                direct_point_render_consistency.item()
            )
            loss_accum["direct_point_render_consistency_valid"] += (
                direct_point_render_consistency_valid.item()
            )
            loss_accum["direct_point_adapter_text_distill"] += direct_point_adapter_text_distill.item()
            loss_accum["direct_point_adapter_text_distill_valid"] += (
                direct_point_adapter_text_distill_valid.item()
            )
            loss_accum["direct_point_adapter_text_distill_teacher_conf"] += (
                direct_point_adapter_text_distill_teacher_conf.item()
            )
            loss_accum["direct_point_adapter_text_distill_agreement"] += (
                direct_point_adapter_text_distill_agreement.item()
            )
            loss_accum["direct_point_adapter_text_pseudo_ce"] += (
                direct_point_adapter_text_pseudo_ce.item()
            )
            loss_accum["direct_point_adapter_text_pseudo_ce_valid"] += (
                direct_point_adapter_text_pseudo_ce_valid.item()
            )
            loss_accum["direct_point_adapter_text_pseudo_ce_teacher_conf"] += (
                direct_point_adapter_text_pseudo_ce_teacher_conf.item()
            )
            loss_accum["direct_point_adapter_text_pseudo_ce_agreement"] += (
                direct_point_adapter_text_pseudo_ce_agreement.item()
            )
            loss_accum["direct_point_adapter_decoder_anchor"] += (
                direct_point_adapter_decoder_anchor.item()
            )
            loss_accum["direct_point_query_support_distill"] += (
                direct_point_query_support_distill.item()
            )
            loss_accum["direct_point_query_support_distill_valid"] += (
                direct_point_query_support_distill_valid.item()
            )
            loss_accum["direct_point_query_support_distill_teacher_conf"] += (
                direct_point_query_support_distill_teacher_conf.item()
            )
            loss_accum["direct_point_query_support_distill_top1"] += (
                direct_point_query_support_distill_top1.item()
            )
            loss_accum["direct_point_proposal_consistency"] += (
                direct_point_proposal_consistency.item()
            )
            loss_accum["direct_point_proposal_consistency_valid"] += (
                direct_point_proposal_consistency_valid.item()
            )
            loss_accum["direct_point_proposal_contrast"] += (
                direct_point_proposal_contrast.item()
            )
            loss_accum["direct_point_proposal_contrast_valid"] += (
                direct_point_proposal_contrast_valid.item()
            )
            loss_accum["direct_point_proposal_contrast_num_proposals"] += (
                direct_point_proposal_contrast_num_proposals.item()
            )
            loss_accum["direct_point_view_weight_mean"] += direct_point_view_weight_mean.item()
            loss_accum["direct_point_view_weight_max"] += direct_point_view_weight_max.item()
            loss_accum["siglip_align"] += l_siglip.item()
            loss_accum["summary_align"] += l_summary.item()
            loss_accum["text_heatmaps"] += l_text_heatmaps.item()
            loss_accum["radio_adaptors"] += l_radio_adaptors.item()
            loss_accum["radio_relations"] += l_radio_relations.item()
            loss_accum["radio_local_affinity"] += l_radio_local_affinity.item()
            loss_accum["radio_token_contrast"] += l_radio_token_contrast.item()
            loss_accum["radio_regions"] += l_radio_regions.item()
            loss_accum["radio_mask_logits"] += l_radio_mask_logits.item()
            loss_accum["radio_cross_views"] += l_radio_cross_views.item()
            loss_accum["radio_cross_view_propagation"] += (
                l_radio_cross_view_propagation.item()
            )
            loss_accum["radio_cross_view_mask_propagation"] += (
                l_radio_cross_view_mask_propagation.item()
            )
            loss_accum["foundation_cache"] += l_foundation_cache.item()
            loss_accum["samclip_mask"] += l_samclip_mask.item()
            loss_accum["samclip_mask_proto"] += samclip_mask_stats["prototype"].item()
            loss_accum["samclip_mask_contrast"] += samclip_mask_stats["contrastive"].item()
            loss_accum["samclip_mask_background"] += samclip_mask_stats["background"].item()
            loss_accum["samclip_mask_regions"] += samclip_mask_stats["valid_regions"].item()
            loss_accum["ground_query"] += l_ground_query.item()
            loss_accum["ground_query_acc"] += ground_query_acc.item()
            loss_accum["ground_query_valid"] += ground_query_valid.item()
            cos_accum += cos_sim.item()
            n_batches += 1
            self.global_step += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}", cos=f"{cos_sim.item():.4f}"
            )

            # Periodic logging
            if self.global_step % log_every == 0 and self.writer is not None:
                self.writer.add_scalar(
                    "train/loss", loss.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/distill", l_distill.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/compact", l_compact.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/tv", l_tv.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/gradient", l_gradient.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/geom_edge", l_geom_edge.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/boundary", l_boundary.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/sem_aux", l_semantic_aux.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/sem_adaptor_reg",
                    l_semantic_adaptor_reg.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/quality", l_quality.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/visibility", l_visibility.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/cosine", cos_sim.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/depth_gt", depth_losses["depth_gt"].item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/depth_geom", depth_losses["depth_geom"].item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/frozen_depth", frozen_depth_losses["total"].item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/seg_aux", seg_losses["total"].item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/frozen_seg", frozen_seg_losses["total"].item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/text_heatmaps", l_text_heatmaps.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/radio_adaptors", l_radio_adaptors.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/radio_relations", l_radio_relations.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/radio_local_affinity",
                    l_radio_local_affinity.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/radio_token_contrast",
                    l_radio_token_contrast.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/radio_regions", l_radio_regions.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/radio_mask_logits",
                    l_radio_mask_logits.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/radio_cross_views", l_radio_cross_views.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/radio_cross_view_propagation",
                    l_radio_cross_view_propagation.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/radio_cross_view_mask_propagation",
                    l_radio_cross_view_mask_propagation.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/foundation_cache",
                    l_foundation_cache.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/samclip_mask", l_samclip_mask.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/samclip_mask_proto",
                    samclip_mask_stats["prototype"].item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/samclip_mask_contrast",
                    samclip_mask_stats["contrastive"].item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/samclip_mask_background",
                    samclip_mask_stats["background"].item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/samclip_mask_regions",
                    samclip_mask_stats["valid_regions"].item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point", l_direct_point.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/direct_point_valid", direct_point_valid.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/direct_point_text", direct_point_text.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/direct_point_text_valid",
                    direct_point_text_valid.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_text_acc",
                    direct_point_text_acc.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_adapter_text",
                    direct_point_adapter_text.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_adapter_text_valid",
                    direct_point_adapter_text_valid.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_adapter_text_acc",
                    direct_point_adapter_text_acc.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_text_distill",
                    direct_point_text_distill.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_text_distill_valid",
                    direct_point_text_distill_valid.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_text_distill_teacher_conf",
                    direct_point_text_distill_teacher_conf.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_text_distill_agreement",
                    direct_point_text_distill_agreement.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_text_pseudo_ce",
                    direct_point_text_pseudo_ce.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_text_pseudo_ce_valid",
                    direct_point_text_pseudo_ce_valid.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_text_pseudo_ce_teacher_conf",
                    direct_point_text_pseudo_ce_teacher_conf.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_text_pseudo_ce_agreement",
                    direct_point_text_pseudo_ce_agreement.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_adapter_text_distill",
                    direct_point_adapter_text_distill.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_render_consistency",
                    direct_point_render_consistency.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_render_consistency_valid",
                    direct_point_render_consistency_valid.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_adapter_text_distill_valid",
                    direct_point_adapter_text_distill_valid.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_adapter_text_distill_teacher_conf",
                    direct_point_adapter_text_distill_teacher_conf.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_adapter_text_distill_agreement",
                    direct_point_adapter_text_distill_agreement.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_adapter_text_pseudo_ce",
                    direct_point_adapter_text_pseudo_ce.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_adapter_text_pseudo_ce_valid",
                    direct_point_adapter_text_pseudo_ce_valid.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_adapter_text_pseudo_ce_teacher_conf",
                    direct_point_adapter_text_pseudo_ce_teacher_conf.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_adapter_text_pseudo_ce_agreement",
                    direct_point_adapter_text_pseudo_ce_agreement.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_adapter_decoder_anchor",
                    direct_point_adapter_decoder_anchor.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_query_support_distill",
                    direct_point_query_support_distill.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_query_support_distill_top1",
                    direct_point_query_support_distill_top1.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_proposal_consistency",
                    direct_point_proposal_consistency.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_proposal_consistency_valid",
                    direct_point_proposal_consistency_valid.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_proposal_contrast",
                    direct_point_proposal_contrast.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_proposal_contrast_valid",
                    direct_point_proposal_contrast_valid.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/direct_point_proposal_contrast_num_proposals",
                    direct_point_proposal_contrast_num_proposals.item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train/siglip_align", l_siglip.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/summary_align", l_summary.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/ground_query", l_ground_query.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/ground_query_acc", ground_query_acc.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/ground_query_valid",
                    ground_query_valid.item(),
                    self.global_step,
                )
                if hybrid_aux is not None and "semantic_confidence" in hybrid_aux:
                    conf = hybrid_aux["semantic_confidence"].float()
                    self.writer.add_scalar(
                        "train/sem_conf_mean", conf.mean().item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "train/sem_conf_std", conf.std().item(), self.global_step
                    )
                lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/lr", lr, self.global_step)

        # Epoch averages
        if n_batches == 0:
            return {}
        metrics = {k: v / n_batches for k, v in loss_accum.items()}
        metrics["cosine"] = cos_accum / n_batches
        lr = self.optimizer.param_groups[0]["lr"]
        self._log(
            f"[Train E{epoch:03d}] loss={metrics['total']:.4f} "
            f"cosine={metrics['cosine']:.4f} lr={lr:.2e}"
        )
        return metrics

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        self.codec.eval()
        self.sharpener.eval()
        if self.use_refiner and self.refiner is not None:
            self.refiner.eval()
        if self.depth_head is not None:
            self.depth_head.eval()
        if self.seg_head is not None:
            self.seg_head.eval()

        cos_latent_accum = 0.0
        cos_decoded_accum = 0.0
        mse_accum = 0.0
        depth_gt_accum = 0.0
        depth_geom_accum = 0.0
        frozen_depth_accum = 0.0
        seg_aux_accum = 0.0
        seg_aux_miou_accum = 0.0
        frozen_seg_accum = 0.0
        siglip_align_accum = 0.0
        summary_align_accum = 0.0
        text_heatmap_accum = 0.0
        radio_adaptor_align_accum = 0.0
        radio_adaptor_relation_accum = 0.0
        radio_adaptor_local_affinity_accum = 0.0
        radio_adaptor_token_contrast_accum = 0.0
        radio_adaptor_peak_background_accum = 0.0
        radio_adaptor_region_accum = 0.0
        radio_adaptor_mask_logit_accum = 0.0
        radio_adaptor_cross_view_accum = 0.0
        radio_adaptor_cross_view_propagation_accum = 0.0
        radio_adaptor_cross_view_mask_propagation_accum = 0.0
        ground_query_accum = 0.0
        ground_query_acc_metric = 0.0
        ground_query_valid_accum = 0.0
        n = 0

        for batch in tqdm(
            self.val_loader, desc=f"Val   E{epoch:03d}", leave=False, dynamic_ncols=True
        ):
            gt_features = batch["radio_features"].to(self.device)
            pose_w2c = batch["pose_w2c"].to(self.device)

            rendered_rgb = None
            if self.self_guided:
                val_result = self.renderer.render_features_and_rgb(self.model, pose_w2c)
                val_result = self._canonicalize_render_result(
                    val_result,
                    batch_size=pose_w2c.shape[0],
                    spatial_size=val_result["feature_map"].shape[-2:],
                )
                rendered_rgb = val_result["rgb"]
            else:
                val_result = self.renderer.render_features_batch(self.model, pose_w2c)
                val_result = self._canonicalize_render_result(
                    val_result,
                    batch_size=pose_w2c.shape[0],
                    spatial_size=val_result["feature_map"].shape[-2:],
                )
            rendered_compact = val_result["feature_map"]
            rendered_compact = self.sharpener(rendered_compact)
            if self.use_refiner and self.refiner is not None:
                guide = self._build_guide(batch, val_result, rendered_rgb=rendered_rgb)
                rendered_compact = self.refiner(rendered_compact, guide=guide)

            # Hybrid decode: latent + hash grid → fused output
            hybrid_aux = None
            if self._is_hybrid:
                from radio_gs.models.hybrid_gaussian import unproject_depth_to_positions
                depth_map = self._canonicalize_spatial_map(
                    val_result.get("depth_map"),
                    batch_size=rendered_compact.shape[0],
                    spatial_size=rendered_compact.shape[-2:],
                )
                position_map = unproject_depth_to_positions(
                    depth_map, pose_w2c.float(), self.renderer.K.float(),
                    depth_map.shape[1], depth_map.shape[2],
                )
                position_map = self._normalize_positions(position_map)
                decode_result = self.model.decode_screen_space(
                    rendered_compact.float(),
                    position_map,
                    return_aux=self.hybrid_decoupled_heads,
                    depth_map=depth_map,
                )
                if self.hybrid_decoupled_heads:
                    hybrid_aux = decode_result
                    rendered_compact = decode_result["fused"]
                else:
                    rendered_compact = decode_result

            if self.train_mode == "latent":
                # gt_features are 64d
                gt_compact = gt_features
                if gt_compact.shape[-2:] != rendered_compact.shape[-2:]:
                    gt_compact = F.interpolate(
                        gt_compact, size=rendered_compact.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )
                cos_latent = F.cosine_similarity(
                    rendered_compact.float().flatten(2),
                    gt_compact.float().flatten(2),
                    dim=1,
                ).mean()
                cos_latent_accum += cos_latent.item()

                # Also decode and compare to 1280d GT for monitoring
                decoded = self.codec.decoder(rendered_compact)
                # Load 1280d GT for this frame
                gt_1280_path = self._get_1280d_val_path(batch)
                if gt_1280_path is not None:
                    gt_1280 = (
                        load_training_tensor_cache(
                            gt_1280_path,
                            map_location="cpu",
                            purpose="validation 1280-D feature cache",
                        )
                        .float()
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    if gt_1280.shape[-2:] != decoded.shape[-2:]:
                        gt_1280 = F.interpolate(
                            gt_1280, size=decoded.shape[-2:],
                            mode="bilinear", align_corners=False,
                        )
                    cos_dec = F.cosine_similarity(
                        decoded.float().flatten(2),
                        gt_1280.float().flatten(2),
                        dim=1,
                    ).mean()
                    mse = F.mse_loss(decoded.float(), gt_1280.float())
                    cos_decoded_accum += cos_dec.item()
                    mse_accum += mse.item()
                else:
                    cos_decoded_accum += cos_latent.item()
                    mse_accum += F.mse_loss(rendered_compact.float(), gt_compact.float()).item()
                decoded_for_depth = decoded
            else:
                # Decoded mode: gt_features are 1280d
                gt_radio = gt_features
                decoded = self.codec.decoder(rendered_compact)
                if decoded.shape[-2:] != gt_radio.shape[-2:]:
                    gt_radio = F.interpolate(
                        gt_radio, size=decoded.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )
                cos_dec = F.cosine_similarity(
                    decoded.float().flatten(2),
                    gt_radio.float().flatten(2),
                    dim=1,
                ).mean()
                mse = F.mse_loss(decoded.float(), gt_radio.float())
                cos_decoded_accum += cos_dec.item()
                cos_latent_accum += cos_dec.item()
                mse_accum += mse.item()
                decoded_for_depth = decoded

            depth_losses = self._compute_depth_aux_losses(
                batch=batch,
                render_result=val_result,
                decoded=decoded_for_depth,
            )
            frozen_depth_losses = self._compute_frozen_depth_loss(
                render_result=val_result,
                decoded=decoded_for_depth,
                teacher_features=gt_radio if self.train_mode != "latent" else None,
            )
            seg_losses = self._compute_seg_aux_losses(
                batch=batch,
                decoded=decoded_for_depth if self.train_mode != "latent" else None,
            )
            frozen_seg_losses = self._compute_frozen_seg_loss(
                decoded=decoded_for_depth if self.train_mode != "latent" else None,
                teacher_features=gt_radio if self.train_mode != "latent" else None,
            )
            depth_gt_accum += depth_losses["depth_gt"].item()
            depth_geom_accum += depth_losses["depth_geom"].item()
            frozen_depth_accum += frozen_depth_losses["total"].item()
            seg_aux_accum += seg_losses["total"].item()
            seg_aux_miou_accum += seg_losses["miou"]
            frozen_seg_accum += frozen_seg_losses["total"].item()
            siglip_align_accum += self._compute_siglip_alignment_loss(
                decoded=decoded_for_depth if self.train_mode != "latent" else None,
                target=gt_radio if self.train_mode != "latent" else None,
            ).item()
            summary_align_accum += self._compute_summary_alignment_loss(
                decoded=decoded_for_depth if self.train_mode != "latent" else None,
                target=gt_radio if self.train_mode != "latent" else None,
            ).item()
            radio_adaptor_align_accum += self._compute_radio_adaptor_alignment_loss(
                decoded=decoded_for_depth if self.train_mode != "latent" else None,
                target=gt_radio if self.train_mode != "latent" else None,
            ).item()
            text_heatmap_accum += self._compute_text_heatmap_distill_loss(
                decoded=decoded_for_depth if self.train_mode != "latent" else None,
                target=gt_radio if self.train_mode != "latent" else None,
            ).item()
            radio_adaptor_relation_accum += self._compute_radio_adaptor_relation_loss(
                decoded=decoded_for_depth if self.train_mode != "latent" else None,
                target=gt_radio if self.train_mode != "latent" else None,
            ).item()
            radio_adaptor_local_affinity_accum += (
                self._compute_radio_adaptor_local_affinity_loss(
                    decoded=decoded_for_depth if self.train_mode != "latent" else None,
                    target=gt_radio if self.train_mode != "latent" else None,
                ).item()
            )
            radio_adaptor_token_contrast_accum += self._compute_radio_adaptor_token_contrast_loss(
                decoded=decoded_for_depth if self.train_mode != "latent" else None,
                target=gt_radio if self.train_mode != "latent" else None,
            ).item()
            radio_adaptor_peak_background_accum += self._compute_radio_adaptor_peak_background_loss(
                decoded=decoded_for_depth if self.train_mode != "latent" else None,
                target=gt_radio if self.train_mode != "latent" else None,
            ).item()
            radio_adaptor_region_accum += self._compute_radio_adaptor_region_loss(
                decoded=decoded_for_depth if self.train_mode != "latent" else None,
                target=gt_radio if self.train_mode != "latent" else None,
            ).item()
            radio_adaptor_mask_logit_accum += self._compute_radio_adaptor_mask_logit_loss(
                decoded=decoded_for_depth if self.train_mode != "latent" else None,
                target=gt_radio if self.train_mode != "latent" else None,
            ).item()
            radio_adaptor_cross_view_accum += self._compute_radio_adaptor_cross_view_loss(
                decoded=decoded_for_depth if self.train_mode != "latent" else None,
                target=gt_radio if self.train_mode != "latent" else None,
            ).item()
            radio_adaptor_cross_view_propagation_accum += (
                self._compute_radio_adaptor_cross_view_propagation_loss(
                    decoded=decoded_for_depth if self.train_mode != "latent" else None,
                    target=gt_radio if self.train_mode != "latent" else None,
                ).item()
            )
            radio_adaptor_cross_view_mask_propagation_accum += (
                self._compute_radio_adaptor_cross_view_mask_propagation_loss(
                    decoded=decoded_for_depth if self.train_mode != "latent" else None,
                    target=gt_radio if self.train_mode != "latent" else None,
                ).item()
            )
            semantic_decoded = None
            if (
                hybrid_aux is not None
                and "semantic" in hybrid_aux
                and self.train_mode != "latent"
            ):
                semantic_decoded = self.codec.decoder(hybrid_aux["semantic"])
            ground_query_stats = self._compute_grounding_query_loss(
                batch=batch,
                decoded=semantic_decoded if semantic_decoded is not None else decoded_for_depth,
            )
            ground_query_accum += ground_query_stats["loss"].item()
            ground_query_acc_metric += ground_query_stats["accuracy"].item()
            ground_query_valid_accum += ground_query_stats["valid_ratio"].item()

            n += 1

        if n == 0:
            return {}

        avg_cos_latent = cos_latent_accum / n
        avg_cos_decoded = cos_decoded_accum / n
        avg_mse = mse_accum / n
        psnr = -10.0 * np.log10(avg_mse + 1e-8)

        # Primary metric for best model selection: latent cosine in latent mode
        primary_cos = avg_cos_latent if self.train_mode == "latent" else avg_cos_decoded

        metrics = {
            "cosine": primary_cos,
            "cosine_latent": avg_cos_latent,
            "cosine_decoded": avg_cos_decoded,
            "mse": avg_mse,
            "psnr": psnr,
            "depth_gt": depth_gt_accum / n,
            "depth_geom": depth_geom_accum / n,
            "frozen_depth": frozen_depth_accum / n,
            "seg_aux": seg_aux_accum / n,
            "seg_aux_miou": seg_aux_miou_accum / n,
            "frozen_seg": frozen_seg_accum / n,
            "siglip_align": siglip_align_accum / n,
            "summary_align": summary_align_accum / n,
            "text_heatmaps": text_heatmap_accum / n,
            "radio_adaptors": radio_adaptor_align_accum / n,
            "radio_relations": radio_adaptor_relation_accum / n,
            "radio_local_affinity": radio_adaptor_local_affinity_accum / n,
            "radio_token_contrast": radio_adaptor_token_contrast_accum / n,
            "radio_peak_background": radio_adaptor_peak_background_accum / n,
            "radio_regions": radio_adaptor_region_accum / n,
            "radio_mask_logits": radio_adaptor_mask_logit_accum / n,
            "radio_cross_views": radio_adaptor_cross_view_accum / n,
            "radio_cross_view_propagation": radio_adaptor_cross_view_propagation_accum / n,
            "radio_cross_view_mask_propagation": (
                radio_adaptor_cross_view_mask_propagation_accum / n
            ),
            "ground_query": ground_query_accum / n,
            "ground_query_acc": ground_query_acc_metric / n,
            "ground_query_valid": ground_query_valid_accum / n,
        }

        if self.writer is not None:
            self.writer.add_scalar("val/cosine_latent", avg_cos_latent, epoch)
            self.writer.add_scalar("val/cosine_decoded", avg_cos_decoded, epoch)
            self.writer.add_scalar("val/psnr", psnr, epoch)
            self.writer.add_scalar("val/depth_gt", metrics["depth_gt"], epoch)
            self.writer.add_scalar("val/depth_geom", metrics["depth_geom"], epoch)
            self.writer.add_scalar("val/frozen_depth", metrics["frozen_depth"], epoch)
            self.writer.add_scalar("val/seg_aux", metrics["seg_aux"], epoch)
            self.writer.add_scalar("val/seg_aux_miou", metrics["seg_aux_miou"], epoch)
            self.writer.add_scalar("val/frozen_seg", metrics["frozen_seg"], epoch)
            self.writer.add_scalar("val/siglip_align", metrics["siglip_align"], epoch)
            self.writer.add_scalar("val/summary_align", metrics["summary_align"], epoch)
            self.writer.add_scalar("val/text_heatmaps", metrics["text_heatmaps"], epoch)
            self.writer.add_scalar("val/radio_adaptors", metrics["radio_adaptors"], epoch)
            self.writer.add_scalar("val/radio_relations", metrics["radio_relations"], epoch)
            self.writer.add_scalar(
                "val/radio_local_affinity",
                metrics["radio_local_affinity"],
                epoch,
            )
            self.writer.add_scalar(
                "val/radio_token_contrast",
                metrics["radio_token_contrast"],
                epoch,
            )
            self.writer.add_scalar("val/radio_regions", metrics["radio_regions"], epoch)
            self.writer.add_scalar("val/radio_mask_logits", metrics["radio_mask_logits"], epoch)
            self.writer.add_scalar("val/radio_cross_views", metrics["radio_cross_views"], epoch)
            self.writer.add_scalar(
                "val/radio_cross_view_propagation",
                metrics["radio_cross_view_propagation"],
                epoch,
            )
            self.writer.add_scalar(
                "val/radio_cross_view_mask_propagation",
                metrics["radio_cross_view_mask_propagation"],
                epoch,
            )
            self.writer.add_scalar("val/ground_query", metrics["ground_query"], epoch)
            self.writer.add_scalar("val/ground_query_acc", metrics["ground_query_acc"], epoch)
            self.writer.add_scalar("val/ground_query_valid", metrics["ground_query_valid"], epoch)

        self._save_vis(epoch)

        self._log(
            f"[Val E{epoch:03d}] cos_latent={avg_cos_latent:.4f} "
            f"cos_decoded={avg_cos_decoded:.4f} psnr={psnr:.2f}"
        )
        return metrics

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        total_epochs = getattr(self.cfg, "epochs", 100)
        eval_every = getattr(self.cfg, "eval_every", 5)
        save_every = getattr(self.cfg, "save_every", 10)
        self.run_status = "running"

        self._log(
            f"Starting training: epochs {self.start_epoch}→{total_epochs}, "
            f"eval_every={eval_every}, save_every={save_every}"
        )

        for epoch in range(self.start_epoch, total_epochs + 1):
            # Curriculum FDH: ramp weight from 0 → target over warmup epochs
            if self.frozen_depth_warmup_epochs > 0 and self.frozen_depth_head_weight_target > 0:
                ramp = min(1.0, epoch / self.frozen_depth_warmup_epochs)
                self.frozen_depth_head_weight = ramp * self.frozen_depth_head_weight_target
            train_metrics = self.train_epoch(epoch)
            self.last_train_metrics = train_metrics

            if epoch % eval_every == 0 or epoch == total_epochs:
                val_metrics = self.validate(epoch)
                self.last_val_metrics = val_metrics
                metric_name, metric_value, metric_score = self._resolve_best_metric(
                    val_metrics
                )
                self.best_cosine = max(self.best_cosine, val_metrics.get("cosine", -1.0))
                is_best = metric_score > self.best_selection_score
                if is_best:
                    self.best_selection_score = metric_score
                    self.best_selection_value = metric_value
                    self.best_epoch = epoch
                    self._log(
                        f"  ★ New best! {metric_name}={metric_value:.4f} "
                        f"cosine={val_metrics.get('cosine', 0):.4f} "
                        f"psnr={val_metrics.get('psnr', 0):.2f}"
                    )
                self.save_checkpoint(epoch, val_metrics, is_best=is_best)
                self._append_metrics_history(epoch, train_metrics, val_metrics)
            elif epoch % save_every == 0:
                self.save_checkpoint(epoch, train_metrics)
                self._append_metrics_history(epoch, train_metrics, None)
            else:
                self._append_metrics_history(epoch, train_metrics, None)

            self.scheduler.step()
            self._write_experiment_report(epoch)

        self._log("Training complete.")
        self.run_status = "completed"
        self._write_experiment_report(total_epochs, final=True)
        if self.writer is not None:
            self.writer.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _all_trainable_params(self):
        """Gather all trainable parameters for gradient clipping."""
        params = list(self.model.trainable_parameters())
        if self.train_mode != "latent":
            params += list(self.codec.decoder.parameters())
        if self.sharpener.mode not in ("analytical", "none"):
            params += list(self.sharpener.parameters())
        if self.use_refiner and self.refiner is not None:
            params += list(self.refiner.parameters())
        if self.depth_head is not None:
            params += list(self.depth_head.parameters())
        if self.seg_head is not None:
            params += list(self.seg_head.parameters())
        params += [
            param
            for param in self.foundation_cache_projectors.parameters()
            if param.requires_grad
        ]
        return params

    @staticmethod
    def _collate_batch(batch):
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            return torch.stack([item.clone() for item in batch], dim=0)
        if isinstance(elem, dict):
            return {
                key: RadioGSTrainer._collate_batch([item[key] for item in batch])
                for key in elem
            }
        if isinstance(elem, (int, float)):
            return torch.tensor(batch)
        return batch

    def _load_samclip_mask_frame(
        self, frame_idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Load cached CLIP prototypes and the selected SAM segment level."""
        frame_idx = int(frame_idx)
        if frame_idx in self.samclip_mask_cache:
            return self.samclip_mask_cache[frame_idx]
        entry = self.samclip_mask_entries.get(frame_idx)
        if entry is None:
            return None

        feature_np = np.asarray(np.load(entry.feature_path), dtype=np.float32)
        segment_np = np.load(entry.segments_path, mmap_mode="r")
        level = int(getattr(self.cfg, "samclip_feature_level", 0))
        if segment_np.ndim == 3:
            if level < 0 or level >= int(segment_np.shape[0]):
                raise ValueError(
                    f"SAM-CLIP level {level} is outside segment map levels "
                    f"[0,{int(segment_np.shape[0])}) for frame {frame_idx}"
                )
            selected_segment_np = np.asarray(segment_np[level], dtype=np.int64)
        elif segment_np.ndim == 2:
            selected_segment_np = np.asarray(segment_np, dtype=np.int64)
        else:
            raise ValueError(
                f"Expected SAM segment map [L,H,W] or [H,W], got "
                f"{tuple(segment_np.shape)} for frame {frame_idx}"
            )

        loaded = (
            torch.from_numpy(feature_np.copy()),
            torch.from_numpy(selected_segment_np.copy()),
        )
        if self.samclip_mask_cache_size > 0:
            if len(self.samclip_mask_cache) >= self.samclip_mask_cache_size:
                oldest = next(iter(self.samclip_mask_cache))
                self.samclip_mask_cache.pop(oldest, None)
            self.samclip_mask_cache[frame_idx] = loaded
        return loaded

    def _compute_samclip_mask_loss(
        self,
        *,
        batch: Dict[str, torch.Tensor],
        decoded: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        zero_ref = decoded if decoded is not None else torch.zeros((), device=self.device)
        zero = zero_ref.sum() * 0.0
        stats = {
            "loss": zero,
            "prototype": zero,
            "contrastive": zero,
            "background": zero,
            "valid_regions": torch.zeros((), device=self.device),
        }
        if (
            decoded is None
            or (
                self.samclip_mask_loss_weight <= 0
                and self.samclip_contrastive_loss_weight <= 0
                and self.samclip_background_loss_weight <= 0
            )
        ):
            return stats

        frame_indices = batch.get("frame_idx")
        if frame_indices is None:
            return stats
        if frame_indices.dim() == 0:
            frame_indices = frame_indices.view(1)

        proto_terms: List[torch.Tensor] = []
        contrast_terms: List[torch.Tensor] = []
        background_terms: List[torch.Tensor] = []
        region_terms: List[torch.Tensor] = []
        for batch_idx, frame_idx_tensor in enumerate(frame_indices):
            loaded = self._load_samclip_mask_frame(int(frame_idx_tensor.item()))
            if loaded is None:
                continue
            prototypes_cpu, segments_cpu = loaded
            losses = compute_samclip_mask_losses(
                decoded[batch_idx],
                prototypes_cpu.to(self.device, non_blocking=True),
                segments_cpu.to(self.device, non_blocking=True),
                min_pixels=self.samclip_mask_min_pixels,
                max_regions=self.samclip_mask_max_regions,
                contrastive_temperature=self.samclip_contrastive_temperature,
            )
            proto_terms.append(losses["prototype_loss"])
            contrast_terms.append(losses["contrastive_loss"])
            background_terms.append(losses["background_loss"])
            region_terms.append(losses["valid_regions"])

        if not proto_terms:
            return stats

        prototype = torch.stack(proto_terms).mean()
        contrastive = torch.stack(contrast_terms).mean()
        background = torch.stack(background_terms).mean()
        valid_regions = torch.stack(region_terms).float().mean()
        total = (
            self.samclip_mask_loss_weight * prototype
            + self.samclip_contrastive_loss_weight * contrastive
            + self.samclip_background_loss_weight * background
        )
        return {
            "loss": total,
            "prototype": prototype,
            "contrastive": contrastive,
            "background": background,
            "valid_regions": valid_regions,
        }

    def _canonicalize_spatial_map(
        self,
        x: Optional[torch.Tensor],
        *,
        batch_size: int,
        spatial_size: Tuple[int, int],
        add_channel_dim: bool = False,
        fill_value: float = 0.0,
    ) -> Optional[torch.Tensor]:
        """Coerce malformed map tensors to [B,H,W] or [B,1,H,W]."""
        if x is None:
            return None

        target_numel = batch_size * spatial_size[0] * spatial_size[1]
        x = x.to(self.device).float().contiguous()

        if x.dim() == 4:
            if x.shape[1] == 1:
                x = x[:, 0]
            elif x.shape[-1] == 1:
                x = x[..., 0]
        elif x.dim() == 0:
            x = x.view(1, 1, 1).expand(batch_size, *spatial_size)
        elif x.dim() == 1:
            if x.numel() == target_numel:
                x = x.view(batch_size, *spatial_size)
            elif x.numel() == spatial_size[0] * spatial_size[1]:
                x = x.view(1, *spatial_size).expand(batch_size, -1, -1)
            else:
                x = x.new_full((batch_size, *spatial_size), fill_value)
        elif x.dim() == 2:
            if x.shape == spatial_size:
                x = x.unsqueeze(0).expand(batch_size, -1, -1)
            elif x.shape[0] == batch_size and x.shape[1] == spatial_size[0] * spatial_size[1]:
                x = x.view(batch_size, *spatial_size)
            elif x.numel() == target_numel:
                x = x.reshape(batch_size, *spatial_size)
            else:
                x = x.new_full((batch_size, *spatial_size), fill_value)
        elif x.dim() != 3:
            x = x.new_full((batch_size, *spatial_size), fill_value)

        if x.dim() == 3 and x.shape[0] != batch_size:
            if x.shape[0] == 1:
                x = x.expand(batch_size, -1, -1)
            elif x.numel() == target_numel:
                x = x.reshape(batch_size, *spatial_size)
            else:
                x = x.new_full((batch_size, *spatial_size), fill_value)

        if x.dim() == 3 and x.shape[-2:] != spatial_size:
            x = F.interpolate(
                x.unsqueeze(1),
                size=spatial_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        if x.dim() != 3:
            x = x.new_full((batch_size, *spatial_size), fill_value)

        if add_channel_dim:
            return x.unsqueeze(1)
        return x

    def _canonicalize_render_result(
        self,
        render_result: Dict[str, torch.Tensor],
        *,
        batch_size: int,
        spatial_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """Normalize renderer depth/alpha maps to a consistent batch-first format."""
        render_result = dict(render_result)
        for key in ("depth_map", "alpha_map", "geom_depth", "geom_alpha"):
            if key in render_result:
                render_result[key] = self._canonicalize_spatial_map(
                    render_result[key],
                    batch_size=batch_size,
                    spatial_size=spatial_size,
                )
        return render_result

    @staticmethod
    def _resize_map(
        x: torch.Tensor,
        size: Tuple[int, int],
        is_mask: bool = False,
    ) -> torch.Tensor:
        """Resize a dense [B,H,W] or [B,1,H,W] map to the target spatial size."""
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3 and x.shape[0] != 1:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.shape[-2:] == size:
            return x.float()
        if is_mask:
            return F.interpolate(x.float(), size=size, mode="nearest")
        if x.shape[-2] >= size[0] and x.shape[-1] >= size[1]:
            return F.interpolate(x.float(), size=size, mode="area")
        return F.interpolate(x.float(), size=size, mode="bilinear", align_corners=False)

    def _compute_depth_aux_losses(
        self,
        batch: Dict[str, torch.Tensor],
        render_result: Dict[str, torch.Tensor],
        decoded: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero = decoded.sum() * 0.0
        losses = {
            "total": zero,
            "depth_gt": zero,
            "depth_geom": zero,
        }
        if self.depth_head is None:
            return losses

        gt_depth = batch.get("depth")
        if gt_depth is None:
            return losses

        pred_depth = self.depth_head(decoded.float())
        target_size = pred_depth.shape[-2:]
        gt_depth = self._resize_map(gt_depth.to(self.device).float(), target_size)
        alpha = self._resize_map(
            render_result["alpha_map"].to(self.device).float(),
            target_size,
        )
        valid_mask = (gt_depth > 0) & (alpha > self.depth_alpha_threshold)

        if self.depth_loss_weight > 0 and valid_mask.any():
            assert self.depth_supervision_loss is not None
            losses["depth_gt"] = self.depth_supervision_loss(pred_depth, gt_depth, valid_mask)

        if self.geom_depth_loss_weight > 0:
            geom_key = "geom_depth" if "geom_depth" in render_result else "depth_map"
            geom_raw = render_result[geom_key].to(self.device).float()
            if getattr(self.cfg, "geom_depth_detach", True):
                geom_raw = geom_raw.detach()
            geom_depth = self._resize_map(geom_raw, target_size)
            geom_mask = (geom_depth > 0) & (alpha > self.depth_alpha_threshold)
            geom_mask = geom_mask & valid_mask
            if geom_mask.any():
                assert self.geom_depth_supervision_loss is not None
                losses["depth_geom"] = self.geom_depth_supervision_loss(
                    pred_depth, geom_depth, geom_mask
                )

        losses["total"] = (
            self.depth_loss_weight * losses["depth_gt"]
            + self.geom_depth_loss_weight * losses["depth_geom"]
        )
        return losses

    def _compute_frozen_depth_loss(
        self,
        render_result: Dict[str, torch.Tensor],
        decoded: torch.Tensor,
        teacher_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute frozen depth head regularization loss.

        Mechanism:
        - geom_depth mode: decoded features → frozen head → predicted depth
          vs geometric depth from 3DGS.
        - gt_features mode: decoded features → frozen head vs frozen head on GT RADIO features.

        Gradients flow to decoded features only.
        """
        zero = decoded.sum() * 0.0
        losses = {"frozen_depth": zero, "frozen_depth_grad": zero, "total": zero}

        if self.frozen_depth_head is None or self.frozen_depth_head_weight <= 0:
            return losses

        # Predict depth from decoded features via frozen head
        # NOTE: no torch.no_grad() — we need gradients through features
        frozen_pred = self.frozen_depth_head(decoded.float())  # [B, 1, H, W]
        target_size = frozen_pred.shape[-2:]

        if self.frozen_depth_teacher == "gt_features":
            if teacher_features is None:
                return losses
            teacher_input = teacher_features.to(self.device).float()
            if teacher_input.shape[-2:] != target_size:
                teacher_input = F.interpolate(
                    teacher_input,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            with torch.no_grad():
                teacher_depth = self.frozen_depth_head(teacher_input)
            valid_mask = torch.isfinite(teacher_depth) & torch.isfinite(frozen_pred)
            valid_mask = valid_mask & (teacher_depth > 0)
            if valid_mask.any():
                assert self.frozen_depth_loss_fn is not None
                losses["frozen_depth"] = self.frozen_depth_loss_fn(
                    frozen_pred,
                    teacher_depth,
                    valid_mask,
                )
                if self.frozen_depth_gradient_weight > 0:
                    losses["frozen_depth_grad"] = self._depth_gradient_loss(
                        frozen_pred,
                        teacher_depth,
                        valid_mask,
                    )
            losses["total"] = (
                self.frozen_depth_head_weight * losses["frozen_depth"]
                + self.frozen_depth_gradient_weight * losses["frozen_depth_grad"]
            )
            return losses

        # Default teacher: geometric depth from 3DGS (high quality, detached)
        geom_key = "geom_depth" if "geom_depth" in render_result else "depth_map"
        geom_raw = render_result[geom_key].to(self.device).float().detach()
        alpha = render_result.get("alpha_map")

        geom_depth = self._resize_map(geom_raw, target_size)
        geom_mask = geom_depth > 0.01

        if alpha is not None:
            alpha_rs = self._resize_map(
                alpha.to(self.device).float().detach(), target_size
            )
            geom_mask = geom_mask & (alpha_rs > self.depth_alpha_threshold)

        if not geom_mask.any():
            return losses

        # Per-image scale-shift alignment (detached — gradients only through features)
        aligned_pred = self._align_depth_scale_shift(
            frozen_pred, geom_depth, geom_mask
        )

        # Primary loss: scale-invariant comparison
        assert self.frozen_depth_loss_fn is not None
        losses["frozen_depth"] = self.frozen_depth_loss_fn(
            aligned_pred, geom_depth, geom_mask
        )

        # Optional gradient matching loss (edge alignment)
        if self.frozen_depth_gradient_weight > 0:
            losses["frozen_depth_grad"] = self._depth_gradient_loss(
                aligned_pred, geom_depth, geom_mask
            )

        losses["total"] = (
            self.frozen_depth_head_weight * losses["frozen_depth"]
            + self.frozen_depth_gradient_weight * losses["frozen_depth_grad"]
        )
        return losses

    def _compute_frozen_seg_loss(
        self,
        decoded: Optional[torch.Tensor],
        teacher_features: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        zero = (
            decoded.sum() * 0.0
            if decoded is not None
            else torch.tensor(0.0, device=self.device)
        )
        losses = {"frozen_seg": zero, "total": zero}
        if (
            self.frozen_seg_head is None
            or self.frozen_seg_head_weight <= 0
            or decoded is None
            or teacher_features is None
        ):
            return losses

        pred_logits = self.frozen_seg_head(decoded.float())
        teacher_input = teacher_features.to(self.device).float()
        if teacher_input.shape[-2:] != pred_logits.shape[-2:]:
            teacher_input = F.interpolate(
                teacher_input,
                size=pred_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        with torch.no_grad():
            teacher_logits = self.frozen_seg_head(teacher_input)

        temp = max(self.frozen_seg_temperature, 1e-6)
        if self.frozen_seg_loss_type == "mse":
            seg_loss = F.mse_loss(pred_logits, teacher_logits)
        else:
            log_probs = F.log_softmax(pred_logits / temp, dim=1)
            teacher_probs = F.softmax(teacher_logits / temp, dim=1)
            seg_loss = F.kl_div(log_probs, teacher_probs, reduction="batchmean") * (temp ** 2)

        losses["frozen_seg"] = seg_loss
        losses["total"] = self.frozen_seg_head_weight * seg_loss
        return losses

    @staticmethod
    def _align_depth_scale_shift(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-image least-squares scale-shift alignment.

        Solves: target ≈ scale * pred + shift  (on masked pixels)
        Returns aligned pred. Scale/shift are detached so gradients
        flow only through the original pred values.
        """
        B = pred.shape[0]
        aligned = pred.clone()
        for b in range(B):
            m = mask[b].bool() if mask.dim() == 4 else mask.bool()
            if m.dim() > 2:
                m = m.squeeze(0)
            p_vals = pred[b].squeeze()[m.squeeze()].detach()
            t_vals = target[b].squeeze()[m.squeeze()].detach()
            if p_vals.numel() < 10:
                continue
            # Least-squares: [scale, shift] = (A^T A)^-1 A^T t
            A = torch.stack([p_vals, torch.ones_like(p_vals)], dim=1)
            try:
                params = torch.linalg.lstsq(A, t_vals.unsqueeze(1)).solution.squeeze()
                scale, shift = params[0].detach(), params[1].detach()
                # Clamp scale to avoid degenerate solutions
                scale = scale.clamp(min=0.1, max=10.0)
                aligned[b] = pred[b] * scale + shift
            except Exception:
                pass  # Fall back to unaligned if lstsq fails
        return aligned

    @staticmethod
    def _depth_gradient_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Loss on spatial depth gradients for edge alignment."""
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        mask_dx = mask[:, :, :, 1:] & mask[:, :, :, :-1] if mask.dim() == 4 else None
        mask_dy = mask[:, :, 1:, :] & mask[:, :, :-1, :] if mask.dim() == 4 else None

        loss_dx = F.l1_loss(pred_dx[mask_dx], tgt_dx[mask_dx]) if mask_dx is not None and mask_dx.any() else pred_dx.sum() * 0.0
        loss_dy = F.l1_loss(pred_dy[mask_dy], tgt_dy[mask_dy]) if mask_dy is not None and mask_dy.any() else pred_dy.sum() * 0.0

        return (loss_dx + loss_dy) * 0.5

    def _project_siglip_features(self, features: torch.Tensor) -> torch.Tensor:
        assert self.siglip_projection is not None
        B, C, H, W = features.shape
        feat_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, C).float()
        projected = self.siglip_projection(feat_flat)
        projected = projected.permute(0, 2, 1).reshape(B, -1, H, W)
        return F.normalize(projected, dim=1)






    def _compute_seg_aux_losses(
        self,
        batch: Dict[str, torch.Tensor],
        decoded: Optional[torch.Tensor],
    ) -> Dict[str, float | torch.Tensor]:
        zero = torch.tensor(0.0, device=self.device)
        losses: Dict[str, float | torch.Tensor] = {
            "total": zero,
            "miou": 0.0,
        }
        if self.seg_head is None or self.seg_loss_fn is None or decoded is None:
            return losses
        gt_sem = batch.get("semantics")
        if gt_sem is None:
            return losses
        gt_sem = gt_sem.to(self.device).long()
        if gt_sem.shape[-2:] != decoded.shape[-2:]:
            gt_sem = self._resize_map(
                gt_sem.unsqueeze(1).float(), decoded.shape[-2:], is_mask=True
            ).squeeze(1).long()
        seg_logits = self.seg_head(decoded.float())
        seg_loss = self.seg_loss_fn(seg_logits, gt_sem)
        losses["total"] = self.seg_loss_weight * seg_loss
        with torch.no_grad():
            pred = seg_logits.argmax(dim=1)
            losses["miou"] = compute_miou(
                pred,
                gt_sem,
                num_classes=getattr(self.cfg, "seg_num_classes", 40),
                ignore_index=getattr(self.cfg, "seg_ignore_index", 255),
            )
        return losses

    def _normalize_positions(self, position_map: torch.Tensor) -> torch.Tensor:
        """Normalize world-space positions to [0, 1] using scene bounds from Gaussians."""
        if not hasattr(self, "_scene_bounds"):
            xyz = self.model.get_xyz()
            margin = 0.1
            self._scene_bounds = (
                xyz.min(dim=0).values - margin,
                xyz.max(dim=0).values + margin,
            )
        lo, hi = self._scene_bounds
        extent = (hi - lo).clamp(min=1e-6)
        # position_map: [B, 3, H, W]
        lo_v = lo.view(1, 3, 1, 1)
        extent_v = extent.view(1, 3, 1, 1)
        return ((position_map - lo_v) / extent_v).clamp(0.0, 1.0)

    def _build_guide(
        self,
        batch: dict,
        render_result: dict,
        rendered_rgb: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Build the guide signal for the screen-space refiner.

        Supports RGB guide (GT or self-rendered), depth guide, or both.
        Returns None if no guide is configured.
        """
        rgb_part = None
        if self.refiner_rgb_guide:
            if self.self_guided and rendered_rgb is not None:
                rgb_part = rendered_rgb.detach()
            else:
                rgb = batch.get("rgb_guide")
                if rgb is not None:
                    rgb_part = rgb.to(self.device)
                else:
                    B, _, H, W = render_result["feature_map"].shape
                    rgb_part = torch.zeros(B, 3, H, W, device=self.device)

        B, _, H, W = render_result["feature_map"].shape
        render_result = self._canonicalize_render_result(
            render_result,
            batch_size=B,
            spatial_size=(H, W),
        )

        return build_refiner_guide(
            render_result,
            rgb_guide=rgb_part,
            use_depth_guide=self.refiner_depth_guide,
            use_depth_grad=getattr(self.cfg, "refiner_depth_grad", False),
            depth_grad_scale=getattr(self.cfg, "refiner_depth_grad_scale", 10.0),
            use_alpha_guide=self.refiner_alpha_guide,
            use_boundary_guide=self.refiner_boundary_guide,
        )

    def _get_1280d_val_path(self, batch) -> Optional[str]:
        """In latent mode, try to locate the original 1280d feature for monitoring."""
        try:
            idx = batch["frame_idx"].item()
            val_1280_dir = getattr(self.cfg, "val_1280d_dir", None)
            if val_1280_dir is None:
                # Derive from feature_dir: replace 64d with 1280d
                feat_dir = getattr(self.cfg, "feature_dir", "")
                val_split = getattr(self.cfg, "val_split", "Sequence_2")
                train_split = getattr(self.cfg, "train_split", "Sequence_1")
                val_1280_dir = feat_dir.replace("64d", "1280d").replace(train_split, val_split)
            p = Path(val_1280_dir) / "backbone" / f"rgb_{idx}.pt"
            if not p.exists():
                p = Path(val_1280_dir) / f"rgb_{idx}.pt"
            return str(p) if p.exists() else None
        except Exception:
            return None

    @staticmethod
    def _count_params(module: nn.Module) -> float:
        return sum(p.numel() for p in module.parameters()) / 1e6

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_file = self.log_dir / "training.log"
        with open(log_file, "a") as f:
            f.write(line + "\n")

    @staticmethod
    def _feature_to_pca_rgb(feat: torch.Tensor) -> torch.Tensor:
        flat = feat[0].float().flatten(1)
        mean = flat.mean(dim=1, keepdim=True)
        centered = flat - mean
        U, _S, _V = torch.pca_lowrank(centered.T, q=3)
        rgb = U.T.reshape(3, *feat.shape[-2:])
        return (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    @staticmethod
    def _write_ppm(path: Path, image: torch.Tensor) -> None:
        image = image.detach().clamp(0.0, 1.0).cpu()
        if image.dim() == 3 and image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        if image.dim() != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected [3,H,W] image, got {tuple(image.shape)}")
        h, w = image.shape[1], image.shape[2]
        array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        with open(path, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
            f.write(array.tobytes())

    @torch.no_grad()
    def _render_validation_sample(
        self,
        sample: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gt = sample["radio_features"].unsqueeze(0).to(self.device)
        pose = sample["pose_w2c"].unsqueeze(0).to(self.device)

        rendered_rgb = None
        if self.self_guided:
            result = self.renderer.render_features_and_rgb(self.model, pose)
            result = self._canonicalize_render_result(
                result,
                batch_size=pose.shape[0],
                spatial_size=result["feature_map"].shape[-2:],
            )
            rendered_rgb = result["rgb"]
        else:
            result = self.renderer.render_features_batch(self.model, pose)
            result = self._canonicalize_render_result(
                result,
                batch_size=pose.shape[0],
                spatial_size=result["feature_map"].shape[-2:],
            )

        rendered_compact = self.sharpener(result["feature_map"])
        if self.use_refiner and self.refiner is not None:
            guide = self._build_guide(sample, result, rendered_rgb=rendered_rgb)
            rendered_compact = self.refiner(rendered_compact, guide=guide)

        if self._is_hybrid:
            from radio_gs.models.hybrid_gaussian import unproject_depth_to_positions

            depth_map = self._canonicalize_spatial_map(
                result.get("depth_map"),
                batch_size=rendered_compact.shape[0],
                spatial_size=rendered_compact.shape[-2:],
            )
            position_map = unproject_depth_to_positions(
                depth_map,
                pose.float(),
                self.renderer.K.float(),
                depth_map.shape[1],
                depth_map.shape[2],
            )
            position_map = self._normalize_positions(position_map)
            decode_result = self.model.decode_screen_space(
                rendered_compact.float(),
                position_map,
                return_aux=self.hybrid_decoupled_heads,
                depth_map=depth_map,
            )
            rendered_compact = (
                decode_result["fused"]
                if self.hybrid_decoupled_heads
                else decode_result
            )

        decoded = self.codec.decoder(rendered_compact)
        if decoded.shape[-2:] != gt.shape[-2:]:
            gt = F.interpolate(
                gt, size=decoded.shape[-2:], mode="bilinear", align_corners=False
            )
        return gt, decoded

    @torch.no_grad()
    def _save_vis(self, epoch: int) -> None:
        """Save PCA visualisation for the first validation sample."""
        try:
            sample = self.val_dataset[0]
            gt, decoded = self._render_validation_sample(sample)

            for tag, feat in [("gt", gt), ("decoded", decoded)]:
                rgb = self._feature_to_pca_rgb(feat)
                if self.writer is not None:
                    self.writer.add_image(f"val/{tag}", rgb, epoch)
                self._write_ppm(self.vis_dir / f"epoch_{epoch:03d}_{tag}_pca.ppm", rgb)
        except Exception as exc:
            self._log(f"Visualization save failed at epoch {epoch}: {exc}")


# ===================================================================
# Entry point
# ===================================================================

def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _load_training_lock(lock_path: Path) -> dict:
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _acquire_training_lock(config: RadioGSConfig, config_path: str) -> Path | None:
    if os.environ.get("RADIO_GS_DISABLE_TRAIN_LOCK", "").strip() == "1":
        return None

    output_dir = Path(getattr(config, "output_dir", "output/radio_gs"))
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    lock_path = report_dir / "training.lock"
    payload = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": config_path,
    }
    encoded = json.dumps(payload, indent=2).encode("utf-8")

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "wb") as f:
                f.write(encoded)
            return lock_path
        except FileExistsError:
            existing = _load_training_lock(lock_path)
            existing_pid = existing.get("pid")
            if isinstance(existing_pid, int) and _pid_is_running(existing_pid):
                print(
                    f"Training lock is active for {output_dir}: "
                    f"pid={existing_pid}, config={existing.get('config_path', 'unknown')}",
                    file=sys.stderr,
                )
                raise SystemExit(75)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def _release_training_lock(lock_path: Path | None) -> None:
    if lock_path is None:
        return
    existing = _load_training_lock(lock_path)
    if existing.get("pid") != os.getpid():
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RADIO-GS feature field via distillation."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--warmstart", default=None, help="Load model weights only"
    )
    parser.add_argument(
        "--pretrained_codec", default=None,
        help="Path to pretrained HCD codec checkpoint (from train_codec.py)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setattr(config, "config_path", args.config)
    trainer: Optional[RadioGSTrainer] = None
    training_lock = _acquire_training_lock(config, args.config)

    try:
        trainer = RadioGSTrainer(config)

        # Load pretrained codec first (before resume/warmstart which may override)
        if args.pretrained_codec:
            ckpt = load_trusted_checkpoint(
                args.pretrained_codec, map_location=trainer.device
            )
            trainer.codec.load_state_dict(ckpt["codec_state_dict"])
            trainer._log(f"Loaded pretrained codec from {args.pretrained_codec}")

        if args.resume:
            trainer.load_checkpoint(args.resume, resume=True)
        elif args.warmstart:
            trainer.load_checkpoint(args.warmstart, resume=False)
        else:
            # Check config for resume_from / warmstart_from
            resume_from = getattr(config, "resume_from", None) or None
            warmstart_from = getattr(config, "warmstart_from", None) or None
            if resume_from:
                trainer.load_checkpoint(resume_from, resume=False)
                trainer._log(f"Warmstart from config: {resume_from}")
            elif warmstart_from:
                trainer.load_checkpoint(warmstart_from, resume=False)

        trainer.train()
    except Exception as exc:
        if trainer is not None:
            trainer.run_status = "failed"
            trainer._write_failure_report(exc)
            trainer._write_experiment_report(max(trainer.start_epoch, 1), final=False)
            trainer._log(f"Training failed: {type(exc).__name__}: {exc}")
        else:
            bootstrap_output_dir = Path(getattr(config, "output_dir", "output/radio_gs"))
            bootstrap_report_dir = bootstrap_output_dir / "reports"
            bootstrap_report_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "exp_name": getattr(config, "exp_name", bootstrap_output_dir.name),
                "status": "failed_before_trainer_init",
                "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config_path": args.config,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            with open(bootstrap_report_dir / "failure.json", "w", encoding="utf-8") as f:
                json.dump(failure, f, indent=2)
        raise
    finally:
        if trainer is not None and trainer.writer is not None:
            trainer.writer.flush()
        _release_training_lock(training_lock)


if __name__ == "__main__":
    main()
