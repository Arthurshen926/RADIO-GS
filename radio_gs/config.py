"""
RADIO-GS configuration system.

YAML-based config with dataclass schema. No Hydra — matches existing ICLPose pattern.

Usage:
    config = load_config("radio_gs/configs/my_exp.yaml")
    config = override_from_args(config, args)
    save_config(config, "output/radio_gs/config_snapshot.yaml")
"""

import argparse
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

import yaml

from radio_gs.artifact_paths import (
    DEFAULT_SIGLIP2_PROJECTION_WEIGHTS,
    DEFAULT_SIGLIP2_TEXT_EMBEDDINGS,
)


@dataclass
class RadioGSConfig:
    """Configuration for RADIO-GS training and evaluation."""

    # Experiment
    exp_name: str = "radio_gs_default"
    output_dir: str = "output/radio_gs"
    seed: int = 42

    # Scene
    scene: str = "room_0"
    ply_path: str = ""
    dataset_type: str = "replica"  # "replica", "scannet", or "lerf"
    scene_root: str = ""  # optional absolute/relative override for scene root

    # RADIO
    radio_version: str = "c-radio_v4-h"
    radio_repo: str = os.environ.get("RADIO_REPO", "")
    radio_feature_dim: int = 1280

    # Architecture
    architecture: str = "explicit"  # "explicit" or "hybrid"
    latent_dim: int = 64  # for explicit
    hybrid_latent_dim: int = 16  # for hybrid
    hash_levels: int = 16
    hash_features_per_level: int = 2
    hash_log2_size: int = 19
    hash_base_resolution: int = 16
    hash_max_resolution: int = 2048
    hash_output_dim: int = 48
    fine_dim: int = 64
    coarse_dim: int = 64
    hybrid_output_dim: int = 128
    hybrid_decoupled_heads: bool = False
    hybrid_semantic_adaptor: bool = False
    hybrid_semantic_adaptor_mode: str = "confidence"  # "confidence" or "refinement"
    hybrid_semantic_adaptor_hidden_dim: int = 64
    hybrid_semantic_adaptor_use_geometry_guidance: bool = True
    hybrid_semantic_adaptor_use_depth_guidance: bool = False
    hybrid_semantic_adaptor_residual: bool = True
    hybrid_semantic_adaptor_reg_weight: float = 0.0
    hybrid_quality_head: bool = False
    hybrid_visibility_head: bool = False
    quality_visibility_heads_only: bool = False
    grounding_query_loss_weight: float = 0.0
    grounding_query_temperature: float = 1.0
    grounding_query_loss_downsample: int = 1
    grounding_text_embeddings: str = DEFAULT_SIGLIP2_TEXT_EMBEDDINGS

    # HCD Codec
    codec_type: str = "hcd"  # "hcd", "direct", or "identity" for no-codec ablations
    bottleneck_dim: int = 64
    dual_stream: bool = True
    symmetric_decoder: bool = False
    codec_hidden_normalization: str = "legacy_group"
    codec_final_normalization: str = "legacy_group"
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 3

    # FeatSharp
    featsharp_mode: str = "analytical"  # "none", "analytical", "learned", "multiview"
    featsharp_strength: float = 0.5
    featsharp_num_source_views: int = 2

    # Screen-space refiner
    use_refiner: bool = False
    refiner_hidden_dim: int = 128
    refiner_num_blocks: int = 4
    refiner_dropout: float = 0.1
    refiner_rgb_guide: bool = False  # Use RGB as additional input to refiner
    refiner_depth_guide: bool = False  # Use rendered depth as guide (always available)
    refiner_depth_grad: bool = False   # Use depth gradients (3ch: depth+dx+dy) instead of 1ch
    refiner_alpha_guide: bool = False  # Use opacity/alpha as a guide channel
    refiner_boundary_guide: bool = False  # Use depth/alpha boundary cue as a guide
    refiner_depth_grad_scale: float = 10.0  # Scale factor for depth-gradient guide channels
    refiner_norm_type: str = "gn"      # "gn" (GroupNorm, stable) or "bn" (BatchNorm, legacy)
    self_guided: bool = False  # Use rendered RGB (not GT) as refiner guide
    lr_refiner: float = 5e-4

    # Joint RGB training (V10)
    train_sh: bool = False  # Unfreeze SH coefficients for joint RGB training
    rgb_loss_weight: float = 0.0  # Weight for RGB reconstruction loss
    lr_sh: float = 5e-4  # Learning rate for SH coefficients

    # Rendering
    image_height: int = 480
    image_width: int = 640
    fx: float = 320.0
    fy: float = 320.0
    cx: float = 319.5
    cy: float = 239.5
    feature_height: int = 30  # H/16 for RADIO patch_size=16
    feature_width: int = 40  # W/16
    max_channels_per_chunk: int = 32
    use_2dgs: bool = False

    # Training
    epochs: int = 100
    batch_size: int = 4
    train_shuffle: bool = True
    lr_features: float = 1e-3
    lr_decoder: float = 1e-4
    lr_hash: float = 1e-3
    lr_heads: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 10.0
    warmup_epochs: int = 5
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    train_mode: str = "decoded"  # "decoded" (1280d) or "latent" (64d)

    # Loss weights
    l2_weight: float = 1.0
    cosine_weight: float = 0.5
    consistency_weight: float = 0.1
    adaptor_weight: float = 0.1
    samclip_mask_loss_weight: float = 0.0
    samclip_contrastive_loss_weight: float = 0.0
    samclip_background_loss_weight: float = 0.0
    samclip_contrastive_temperature: float = 0.07
    samclip_mask_min_pixels: int = 16
    samclip_mask_max_regions: int = 64
    samclip_mask_cache_size: int = 8
    samclip_feature_level: int = 0
    samclip_language_feature_dir: str = ""
    siglip_alignment_weight: float = 0.0
    # Legacy invalid proxy: the SigLIP summary head is not a spatial adaptor.
    # Kept only so old YAML files load; active training fails if this is nonzero.
    siglip_summary_alignment_weight: float = 0.0
    siglip_summary_head_weights: str = "checkpoints/siglip2_summary_head.pth"
    radio_adaptor_alignment_names: str = ""
    radio_adaptor_alignment_weight: float = 0.0
    radio_adaptor_alignment_kind: str = "feature_projection"
    radio_adaptor_alignment_checkpoint: str = "/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar"
    radio_adaptor_relation_names: str = ""
    radio_adaptor_relation_weight: float = 0.0
    radio_adaptor_relation_downsample: int = 1
    radio_adaptor_relation_max_tokens: int = 512
    radio_adaptor_relation_temperature: float = 1.0
    radio_adaptor_local_affinity_names: str = ""
    radio_adaptor_local_affinity_weight: float = 0.0
    radio_adaptor_local_affinity_downsample: int = 1
    radio_adaptor_local_affinity_radius: int = 1
    radio_adaptor_token_contrast_names: str = ""
    radio_adaptor_token_contrast_weight: float = 0.0
    radio_adaptor_token_contrast_downsample: int = 1
    radio_adaptor_token_contrast_max_tokens: int = 512
    radio_adaptor_token_contrast_temperature: float = 0.07
    radio_adaptor_peak_background_names: str = ""
    radio_adaptor_peak_background_weight: float = 0.0
    radio_adaptor_peak_background_downsample: int = 1
    radio_adaptor_peak_background_max_tokens: int = 512
    radio_adaptor_peak_background_num_anchors: int = 16
    radio_adaptor_peak_background_temperature: float = 0.2
    radio_adaptor_peak_background_anchor_strategy: str = "linspace"
    radio_adaptor_region_names: str = ""
    radio_adaptor_region_weight: float = 0.0
    radio_adaptor_region_downsample: int = 1
    radio_adaptor_region_max_tokens: int = 512
    radio_adaptor_region_num_anchors: int = 16
    radio_adaptor_region_temperature: float = 0.07
    radio_adaptor_mask_logit_names: str = ""
    radio_adaptor_mask_logit_weight: float = 0.0
    radio_adaptor_mask_logit_downsample: int = 1
    radio_adaptor_mask_logit_max_tokens: int = 512
    radio_adaptor_mask_logit_num_anchors: int = 16
    radio_adaptor_mask_logit_temperature: float = 0.07
    radio_adaptor_cross_view_names: str = ""
    radio_adaptor_cross_view_weight: float = 0.0
    radio_adaptor_cross_view_downsample: int = 2
    radio_adaptor_cross_view_max_tokens: int = 256
    radio_adaptor_cross_view_temperature: float = 1.0
    radio_adaptor_cross_view_objective: str = "mse"
    radio_adaptor_cross_view_propagation_names: str = ""
    radio_adaptor_cross_view_propagation_weight: float = 0.0
    radio_adaptor_cross_view_propagation_downsample: int = 2
    radio_adaptor_cross_view_propagation_max_tokens: int = 256
    radio_adaptor_cross_view_propagation_num_anchors: int = 16
    radio_adaptor_cross_view_propagation_temperature: float = 0.2
    radio_adaptor_cross_view_propagation_anchor_strategy: str = "linspace"
    radio_adaptor_cross_view_mask_propagation_names: str = ""
    radio_adaptor_cross_view_mask_propagation_weight: float = 0.0
    radio_adaptor_cross_view_mask_propagation_downsample: int = 2
    radio_adaptor_cross_view_mask_propagation_max_tokens: int = 256
    radio_adaptor_cross_view_mask_propagation_num_anchors: int = 16
    radio_adaptor_cross_view_mask_propagation_temperature: float = 0.2
    radio_adaptor_cross_view_mask_propagation_anchor_strategy: str = "linspace"
    foundation_cache_root: str = ""
    foundation_cache_weight: float = 0.0
    foundation_cache_heads: str = ""
    foundation_cache_mask_logit_weight: float = 0.0
    foundation_cache_mask_boundary_weight: float = 0.0
    foundation_cache_token_weight: float = 0.0
    foundation_cache_region_consistency_weight: float = 0.0
    foundation_cache_region_separation_weight: float = 0.0
    foundation_cache_feature_boundary_weight: float = 0.0
    foundation_cache_region_score_threshold: float = 0.0
    foundation_cache_region_max_masks: int = 16
    foundation_cache_region_separation_margin: float = 0.25
    foundation_cache_require_official: bool = False
    foundation_cache_mask_projector_hidden_dim: int = 256
    foundation_cache_mask_projector_masks: int = 32
    text_heatmap_distill_weight: float = 0.0
    text_heatmap_distill_embeddings: str = ""
    text_heatmap_distill_downsample: int = 2
    text_heatmap_distill_temperature: float = 20.0
    text_heatmap_distill_mode: str = "query"
    tv_weight: float = 0.01
    feat_norm_weight: float = 0.0
    gradient_loss_weight: float = 0.0
    gradient_loss_type: str = "sobel"
    depth_guided_feature_weight: float = 0.0
    geometric_edge_loss_weight: float = 0.0
    boundary_aware_loss_weight: float = 0.0
    boundary_aware_sharpness_weight: float = 1.0
    boundary_aware_smoothness_weight: float = 1.0
    boundary_aware_edge_threshold: float = 0.1
    channel_std_weight: float = 0.0
    hybrid_semantic_aux_weight: float = 0.0
    quality_loss_weight: float = 0.0
    visibility_loss_weight: float = 0.0
    visibility_target_binary: bool = False
    visibility_alpha_threshold: float = 0.02
    direct_point_loss_weight: float = 0.0
    direct_point_sample_count: int = 2048
    direct_point_sample_strategy: str = "uniform"  # "uniform" or "class_balanced"
    direct_point_query_mode: str = "gaussian_index"  # "gaussian_index" or "knn"
    direct_point_gaussian_position_mode: str = "label_point"  # "gaussian_center" or "label_point"
    direct_point_source: str = "gaussian"  # "gaussian", "label_ply", or "points3d"
    direct_point_ply_path: str = ""  # optional override; may contain {scene}
    direct_point_teacher_cache: str = ""  # optional .pt cache from eval_scannet_pointcloud_radio_teacher.py
    direct_point_teacher_cache_feature_key: str = ""  # ""/auto, "features", or "summary_features"
    direct_point_teacher_cache_feature_space: str = ""  # ""/auto, "radio", or "siglip_summary"
    direct_point_feature_key: str = "features"  # "features", "fused", "semantic", or "geometry"
    direct_point_k: int = 8
    direct_point_candidate_k: int = 0
    direct_point_depth_tolerance: float = 0.08
    direct_point_relative_depth_tolerance: float = 0.02
    direct_point_alpha_threshold: float = 0.02
    direct_point_summary_alignment_weight: float = 0.0
    direct_point_relation_weight: float = 0.0
    direct_point_relation_max_points: int = 256
    direct_point_summary_adapter_weight: float = 0.0
    direct_point_text_loss_weight: float = 0.0
    direct_point_adapter_text_loss_weight: float = 0.0
    direct_point_adapter_text_distill_weight: float = 0.0
    direct_point_text_pseudo_ce_weight: float = 0.0
    direct_point_text_pseudo_ce_confidence_threshold: float = 0.0
    direct_point_text_pseudo_ce_logit_scale: float = 1.0
    direct_point_text_pseudo_ce_center_logits: bool = False
    direct_point_text_pseudo_ce_splits: str = ""
    direct_point_adapter_text_pseudo_ce_weight: float = 0.0
    direct_point_adapter_text_pseudo_ce_confidence_threshold: float = 0.0
    direct_point_adapter_text_pseudo_ce_logit_scale: float = 1.0
    direct_point_adapter_text_pseudo_ce_center_logits: bool = False
    direct_point_adapter_text_pseudo_ce_splits: str = ""
    direct_point_adapter_decoder_anchor_weight: float = 0.0
    direct_point_text_embeddings: str = "checkpoints/siglip2_scannet_og_text_embeddings.pt"
    direct_point_text_split: str = "19"
    direct_point_text_temperature: float = 0.07
    direct_point_text_ce_weighting: str = "none"  # "none", "inverse_batch", "inverse_pool", or "sqrt_inverse_pool_capped"
    direct_point_text_ce_min_weight: float = 0.5
    direct_point_text_ce_max_weight: float = 3.0
    direct_point_text_distill_weight: float = 0.0
    direct_point_text_distill_temperature: float = 1.0
    direct_point_text_distill_confidence_threshold: float = 0.0
    direct_point_query_logit_distill_weight: float = 0.0
    direct_point_query_logit_distill_embeddings: str = ""
    direct_point_query_logit_distill_temperature: float = 1.0
    direct_point_query_logit_distill_confidence_threshold: float = 0.0
    direct_point_query_support_distill_weight: float = 0.0
    direct_point_query_support_distill_embeddings: str = ""
    direct_point_query_support_distill_temperature: float = 0.25
    direct_point_query_support_distill_confidence_threshold: float = 0.0
    direct_point_query_support_distill_logit_norm: str = "none"  # "none", "center", or "zscore"
    direct_point_view_count_weighting: str = "none"  # "none", "log", or "clipped_log"
    direct_point_view_count_min_weight: float = 0.0
    direct_point_view_count_percentile_low: float = 5.0
    direct_point_view_count_percentile_high: float = 95.0
    direct_point_text_contrast_weight: float = 0.0
    direct_point_text_contrast_temperature: float = 0.1
    direct_point_text_contrast_confidence_threshold: float = 0.0
    direct_point_text_contrast_pair_weighting: str = "none"  # "none" or "visibility"
    direct_point_text_contrast_max_points: int = 4096
    direct_point_text_contrast_center_logits: bool = False
    direct_point_render_consistency_weight: float = 0.0
    direct_point_render_consistency_mode: str = "cosine"  # "cosine" or "mse"
    direct_point_cached_visible_fraction: float = 0.0
    direct_point_cached_visible_candidate_multiplier: int = 1
    direct_point_cached_visible_balance: bool = False
    direct_point_proposal_consistency_weight: float = 0.0
    direct_point_proposal_contrast_weight: float = 0.0
    direct_point_proposal_contrast_temperature: float = 0.07
    direct_point_proposal_voxel_size: float = 0.05
    direct_point_proposal_min_count: int = 2
    direct_point_proposal_space: str = "auto"  # "auto", "adapter", or "decoder"
    point_summary_adapter_hidden_dim: int = 512
    point_summary_adapter_num_layers: int = 2
    point_summary_adapter_dropout: float = 0.0
    point_summary_adapter_context_features: str = ""
    lr_point_summary_adapter: float = 1e-4
    depth_loss_weight: float = 0.0
    geom_depth_loss_weight: float = 0.0
    geom_depth_detach: bool = True
    depth_alpha_threshold: float = 0.05
    depth_supervision_loss_type: str = "scale_invariant"
    geom_depth_supervision_loss_type: str = "scale_invariant"
    siglip_projection_weights: str = DEFAULT_SIGLIP2_PROJECTION_WEIGHTS

    # Data
    feature_dir: str = ""  # pre-extracted RADIO features
    val_feature_dir: str = ""
    pose_file: str = ""
    pose_dir: str = ""
    val_pose_file: str = ""
    val_pose_dir: str = ""
    rgb_dir: str = ""
    val_rgb_dir: str = ""
    depth_dir: str = ""
    val_depth_dir: str = ""
    semantics_dir: str = ""
    val_semantics_dir: str = ""
    instance_dir: str = ""
    val_instance_dir: str = ""
    train_split: str = "Sequence_1"
    val_split: str = "Sequence_2"
    train_frame_ids_path: str = ""
    val_frame_ids_path: str = ""
    mixed_split: bool = False
    mixed_train_ratio: float = 0.8
    mixed_seed: int = 42
    num_workers: int = 4
    grounding_source: str = "replica"
    grounding_annotations_path: str = ""

    # Downstream tasks
    depth_head_type: str = "mlp"  # "linear", "mlp", "dpt"
    depth_head_hidden_dim: int = 256
    depth_head_num_layers: int = 3
    depth_num_classes: int = 1
    seg_num_classes: int = 40
    seg_head_type: str = "mlp"
    seg_head_hidden_dim: int = 256
    seg_head_num_layers: int = 2
    seg_loss_weight: float = 0.0
    seg_loss_type: str = "ce"
    seg_ignore_index: int = 255
    grounding_use_adaptor: bool = True

    # Frozen depth head supervision (core innovation)
    frozen_depth_head_weight: float = 0.0
    frozen_depth_head_path: str = ""
    frozen_depth_head_type: str = "mlp"
    frozen_depth_head_hidden_dim: int = 256
    frozen_depth_head_num_layers: int = 3
    frozen_depth_loss_type: str = "scale_invariant"
    frozen_depth_teacher: str = "geom_depth"  # "geom_depth" or "gt_features"
    frozen_depth_gradient_weight: float = 0.0
    frozen_depth_warmup_epochs: int = 0  # Curriculum: ramp FDH weight from 0→target over N epochs

    # Frozen segmentation head supervision
    frozen_seg_head_weight: float = 0.0
    frozen_seg_head_path: str = ""
    frozen_seg_head_type: str = "mlp"
    frozen_seg_head_hidden_dim: int = 256
    frozen_seg_head_num_layers: int = 3
    frozen_seg_num_classes: int = 40
    frozen_seg_loss_type: str = "kl"  # "kl" or "mse"
    frozen_seg_temperature: float = 1.0

    # Best-checkpoint selection
    best_metric: str = "cosine"  # cosine, psnr, depth_gt, depth_geom, frozen_depth, seg_aux_miou, ground_query_acc, proxy_depth
    best_metric_mode: str = "auto"  # auto, min, max

    # Checkpointing
    save_every: int = 10
    save_periodic_every: int = 0  # Save epoch_XXX.pth every N epochs (0=disabled)
    eval_every: int = 5
    log_every: int = 100  # iterations
    resume_from: str = ""  # Resume training from checkpoint (model + optimizer)
    warmstart_from: str = ""  # Warmstart model weights only


def config_to_dict(config: RadioGSConfig) -> Dict[str, Any]:
    """Convert config dataclass to a plain dict."""
    return asdict(config)


def _coerce_value(field_type: type, value: Any) -> Any:
    """Coerce a parsed YAML value to the expected dataclass field type."""
    if field_type is bool:
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    return field_type(value)


def _load_raw_config(
    yaml_path: Union[str, Path],
    seen: Optional[Set[Path]] = None,
) -> Dict[str, Any]:
    path = Path(yaml_path).expanduser().resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Recursive base_config detected for {path}")
    seen.add(path)
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    base_config = raw.pop("base_config", None)
    if not base_config:
        return raw
    base_path = Path(str(base_config)).expanduser()
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    merged = _load_raw_config(base_path, seen)
    merged.update(raw)
    return merged


def load_config(yaml_path: str) -> RadioGSConfig:
    """Load a RadioGSConfig from a YAML file.

    Unknown keys in the YAML are silently ignored so that experiment
    configs can carry extra metadata without breaking the loader.
    """
    raw = _load_raw_config(yaml_path)

    valid_fields = {fld.name: fld for fld in fields(RadioGSConfig)}
    kwargs: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in valid_fields:
            kwargs[key] = _coerce_value(valid_fields[key].type, value)

    return RadioGSConfig(**kwargs)


def save_config(config: RadioGSConfig, path: str) -> None:
    """Save a RadioGSConfig to a YAML file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_to_dict(config), f, default_flow_style=False, sort_keys=False)


def override_from_args(
    config: RadioGSConfig, args: argparse.Namespace
) -> RadioGSConfig:
    """Override config fields with non-None values from argparse.

    Only fields that exist in both the config and the Namespace are
    considered, so extra CLI flags (e.g. ``--config``) are harmless.
    """
    valid_fields = {fld.name: fld for fld in fields(RadioGSConfig)}
    updates = config_to_dict(config)
    for key, value in vars(args).items():
        if value is not None and key in valid_fields:
            updates[key] = _coerce_value(valid_fields[key].type, value)
    return RadioGSConfig(**updates)


# ---------------------------------------------------------------------------
# Quick CLI: ``python -m radio_gs.config <yaml>`` prints the resolved config
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys

    parser = argparse.ArgumentParser(description="Print resolved RadioGSConfig")
    parser.add_argument("yaml", nargs="?", help="Path to YAML config file")
    cli_args = parser.parse_args()

    if cli_args.yaml:
        cfg = load_config(cli_args.yaml)
    else:
        cfg = RadioGSConfig()

    print(json.dumps(config_to_dict(cfg), indent=2))
