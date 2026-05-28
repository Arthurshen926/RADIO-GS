"""Auxiliary supervision helpers for RADIO-GS feature-field training."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from radio_gs.artifact_paths import (
    DEFAULT_SIGLIP2_TEXT_EMBEDDINGS,
    resolve_siglip_text_embeddings_path,
)
from radio_gs.config import RadioGSConfig
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
from radio_gs.losses.direct_point_query_logit_distill_loss import (
    compute_direct_point_query_logit_distill_loss,
    compute_direct_point_query_support_distill_loss,
)
from radio_gs.losses.text_heatmap_distill_loss import compute_text_heatmap_distill_loss
from radio_gs.models.foundation_cache import (
    compute_foundation_cache_supervision_loss,
    load_foundation_cache,
)
from radio_gs.models.proposal_memory import (
    build_proposal_memory_from_labels,
    build_voxel_proposal_labels,
    compute_region_prototype_contrast_loss,
)
from radio_gs.models.point_summary_adapter import append_point_summary_context
from radio_gs.replica_constants import GROUNDING_QUERIES
from radio_gs.scannet_constants import (
    NYU40_ID_TO_NAME,
    OPENGAUSSIAN_NYU40_CLASS_SPLITS,
)
from radio_gs.training.feature_training_utils import (
    resolve_foundation_cache_path,
    sample_multiview_radio_targets as _sample_multiview_radio_targets_impl,
    select_visible_gaussian_indices as _select_visible_gaussian_indices_impl,
)
from radio_gs.training.tensor_cache_io import load_training_tensor_cache


def _train_script_export(name: str, fallback):
    module = sys.modules.get("radio_gs.scripts.train_feature_field")
    if module is None:
        return fallback
    return getattr(module, name, fallback)


def _sample_multiview_radio_targets(*args, **kwargs):
    return _train_script_export(
        "sample_multiview_radio_targets",
        _sample_multiview_radio_targets_impl,
    )(*args, **kwargs)


def _select_visible_gaussian_indices(*args, **kwargs):
    return _train_script_export(
        "select_visible_gaussian_indices",
        _select_visible_gaussian_indices_impl,
    )(*args, **kwargs)


def _direct_point_view_count_weights(
    view_counts: torch.Tensor,
    *,
    mode: str = "none",
    min_weight: float = 0.0,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
) -> Optional[torch.Tensor]:
    """Build normalized confidence weights from registration view counts."""
    normalized_mode = str(mode or "none").lower()
    if normalized_mode == "none":
        return None
    counts = view_counts.float().clamp_min(0.0)
    active = counts > 0
    if not active.any():
        return torch.zeros_like(counts)
    if normalized_mode not in {"log", "clipped_log"}:
        raise ValueError("direct_point_view_count_weighting must be one of: none, log, clipped_log")
    weights = torch.log1p(counts)
    if normalized_mode == "clipped_log":
        active_weights = weights[active]
        lo_q = float(percentile_low) / 100.0
        hi_q = float(percentile_high) / 100.0
        lo = torch.quantile(active_weights, max(0.0, min(lo_q, 1.0)))
        hi = torch.quantile(active_weights, max(0.0, min(hi_q, 1.0)))
        if float(hi.detach().cpu()) > float(lo.detach().cpu()):
            weights = (weights.clamp(min=lo, max=hi) - lo) / (hi - lo).clamp_min(1e-6)
        else:
            weights = torch.where(active, torch.ones_like(weights), torch.zeros_like(weights))
    weights = torch.where(active, weights, torch.zeros_like(weights))
    if min_weight > 0:
        weights = torch.where(active, weights.clamp_min(float(min_weight)), weights)
    mean = weights[active].mean().clamp_min(1e-6)
    return weights / mean


def _direct_point_weight_mask(point_map: torch.Tensor, sample_weights: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if sample_weights is None:
        return None
    weights = sample_weights.to(device=point_map.device, dtype=point_map.dtype)
    if point_map.ndim != 4:
        raise ValueError(f"Expected point map [B,C,H,W], got {tuple(point_map.shape)}")
    if point_map.shape[0] == weights.numel() and point_map.shape[-2:] == (1, 1):
        return weights.view(-1, 1, 1, 1)
    if point_map.shape[0] == 1 and point_map.shape[2] == weights.numel() and point_map.shape[-1] == 1:
        return weights.view(1, 1, -1, 1)
    raise ValueError(
        f"Cannot align {weights.numel()} sample weights with point map {tuple(point_map.shape)}"
    )


def _weighted_vector_mean(values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    if weights is None:
        return values.mean()
    weights = weights.to(device=values.device, dtype=values.dtype).view(-1)
    if values.shape[0] != weights.numel():
        raise ValueError(f"Expected {values.shape[0]} weights, got {weights.numel()}")
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


class FeatureSupervisionMixin:
    def _point_summary_adapter_input(
        self,
        compact: torch.Tensor,
        gaussian_indices: torch.Tensor,
        view_counts: Optional[torch.Tensor],
    ) -> torch.Tensor:
        context_features = str(
            getattr(self, "point_summary_adapter_context_features", "") or ""
        )
        if not context_features:
            return compact.float()
        if gaussian_indices.ndim != 1:
            gaussian_indices = gaussian_indices.reshape(-1)
        opacity = None
        scales = None
        if "opacity" in context_features:
            opacity = self.model.get_opacity()[gaussian_indices].to(compact.device)
        if "scale_log" in context_features:
            scales = self.model.get_scaling()[gaussian_indices].to(compact.device)
        return append_point_summary_context(
            compact,
            context_features=context_features,
            opacity=opacity,
            scales=scales,
            view_counts=view_counts.to(compact.device) if view_counts is not None else None,
            view_count_max=getattr(self, "point_summary_adapter_view_count_max", None),
        )

    def _project_point_summary_adapter(
        self,
        compact: torch.Tensor,
        gaussian_indices: torch.Tensor,
        view_counts: Optional[torch.Tensor],
    ) -> torch.Tensor:
        adapter_input = self._point_summary_adapter_input(
            compact,
            gaussian_indices,
            view_counts,
        )
        return F.normalize(self.point_summary_adapter(adapter_input), dim=-1)

    def _subsample_direct_point_indices(
        self,
        source_indices: torch.Tensor,
        sample_count: int,
    ) -> torch.Tensor:
        """Subsample visible direct-supervision points for the current batch."""
        if source_indices.numel() <= sample_count:
            return source_indices
        strategy = getattr(self, "direct_point_sample_strategy", "uniform")
        if strategy == "uniform":
            return source_indices[:sample_count]

        if strategy == "teacher_balanced":
            source_labels = self._direct_point_teacher_pseudo_labels(source_indices)
            if source_labels is None:
                return source_indices[:sample_count]
            class_ids = torch.unique(source_labels).detach().cpu().tolist()
            class_ids = [int(class_id) for class_id in class_ids]
            return self._balanced_subsample_by_labels(
                source_indices,
                source_labels.long(),
                class_ids,
                sample_count,
            )

        if strategy != "class_balanced":
            return source_indices[:sample_count]

        labels = getattr(self, "direct_point_pool_labels", None)
        if labels is None:
            return source_indices[:sample_count]
        source_labels = labels[source_indices].long()
        split_ids = [int(v) for v in getattr(self, "direct_point_text_split_ids", [])]
        if split_ids:
            class_ids = [
                raw_id
                for raw_id in split_ids
                if bool((source_labels == raw_id).any().item())
            ]
        else:
            present = torch.unique(source_labels[source_labels > 0]).detach().cpu().tolist()
            class_ids = [int(raw_id) for raw_id in present]
        if not class_ids:
            return source_indices[:sample_count]

        return self._balanced_subsample_by_labels(
            source_indices,
            source_labels,
            class_ids,
            sample_count,
        )

    def _balanced_subsample_by_labels(
        self,
        source_indices: torch.Tensor,
        source_labels: torch.Tensor,
        class_ids: list[int],
        sample_count: int,
    ) -> torch.Tensor:
        """Subsample source indices approximately evenly over the provided labels."""
        per_class = max(1, (sample_count + len(class_ids) - 1) // len(class_ids))
        parts: list[torch.Tensor] = []
        for raw_id in class_ids:
            class_indices = source_indices[source_labels == int(raw_id)]
            if class_indices.numel() == 0:
                continue
            if class_indices.numel() > per_class:
                order = torch.randperm(class_indices.numel(), device=source_indices.device)[:per_class]
                class_indices = class_indices[order]
            parts.append(class_indices)
        if not parts:
            return source_indices[:sample_count]

        sampled = torch.cat(parts, dim=0)
        if sampled.numel() > sample_count:
            order = torch.randperm(sampled.numel(), device=source_indices.device)[:sample_count]
            sampled = sampled[order]
        elif sampled.numel() < sample_count:
            remaining_mask = ~torch.isin(source_indices, sampled)
            remaining = source_indices[remaining_mask]
            if remaining.numel() > 0:
                take = min(sample_count - sampled.numel(), remaining.numel())
                order = torch.randperm(remaining.numel(), device=source_indices.device)[:take]
                sampled = torch.cat([sampled, remaining[order]], dim=0)
        return sampled[:sample_count]

    def _direct_point_teacher_pseudo_labels(
        self,
        source_indices: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return teacher text pseudo-labels for source indices without GT labels."""
        teacher_features = getattr(self, "direct_point_teacher_features", None)
        if (
            teacher_features is None
            or getattr(self, "siglip_summary_head", None) is None
        ):
            return None

        cached = getattr(self, "direct_point_teacher_pseudo_label_cache", None)
        if cached is not None and int(cached.shape[0]) == int(teacher_features.shape[0]):
            return cached.to(source_indices.device)[source_indices].long()

        banks = getattr(self, "direct_point_text_pseudo_ce_banks", None) or []
        if banks:
            _split, _split_ids, text_embeddings = banks[0]
        else:
            text_embeddings = getattr(self, "direct_point_text_embeddings", None)
            if text_embeddings is None or not getattr(self, "direct_point_text_split_ids", []):
                return None

        text = F.normalize(text_embeddings.to(teacher_features.device).float(), dim=-1)
        chunk_size = max(
            1,
            int(getattr(self, "direct_point_teacher_pseudo_label_chunk_size", 8192)),
        )
        logit_scale = float(getattr(self, "direct_point_text_pseudo_ce_logit_scale", 1.0))
        logits_parts: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, int(teacher_features.shape[0]), chunk_size):
                end = min(start + chunk_size, int(teacher_features.shape[0]))
                features = teacher_features[start:end].float()
                feature_map = features[:, :, None, None].contiguous()
                summary_map = self._project_summary_head_features(feature_map)
                summary = self._direct_point_map_to_rows(summary_map)
                logits_parts.append((summary.float() @ text.T) * logit_scale)
            logits = torch.cat(logits_parts, dim=0)
            if bool(getattr(self, "direct_point_text_pseudo_ce_center_logits", False)):
                logits = logits - logits.mean(dim=0, keepdim=True)
            labels = logits.argmax(dim=-1).long()
        self.direct_point_teacher_pseudo_label_cache = labels
        return labels.to(source_indices.device)[source_indices].long()

    def _decode_direct_point_map(
        self,
        compact: torch.Tensor,
        target_points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode compact point features without coupling points in one pseudo image."""
        if hasattr(self.codec, "decode_points"):
            decoded_rows = self.codec.decode_points(compact.float())
            return (
                decoded_rows[:, :, None, None].contiguous(),
                target_points.float()[:, :, None, None].contiguous(),
            )
        compact_map = compact.T.reshape(1, compact.shape[1], compact.shape[0], 1)
        target_map = target_points.T.reshape(
            1,
            target_points.shape[1],
            target_points.shape[0],
            1,
        )
        return self.codec.decoder(compact_map.float()), target_map.float()

    def _num_model_gaussians(self) -> int:
        value = getattr(self.model, "num_gaussians", None)
        if callable(value):
            value = value()
        if value is None:
            return int(self.model.get_xyz().shape[0])
        return int(value)

    @staticmethod
    def _direct_point_map_to_rows(point_map: torch.Tensor) -> torch.Tensor:
        """Convert legacy [1,C,N,1] or pointwise [N,C,1,1] direct maps to [N,C]."""
        if point_map.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W] point map, got {tuple(point_map.shape)}")
        if point_map.shape[0] == 1 and point_map.shape[-1] == 1:
            return point_map.squeeze(0).squeeze(-1).T.contiguous()
        if point_map.shape[-2:] == (1, 1):
            return point_map[:, :, 0, 0].contiguous()
        raise ValueError(f"Unsupported direct point map shape: {tuple(point_map.shape)}")

    def _compute_direct_point_loss(
        self,
        batch: Dict[str, torch.Tensor],
        render_result: Dict[str, torch.Tensor],
        target_features: Optional[torch.Tensor],
        rendered_compact: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        zero = (
            target_features.sum() * 0.0
            if target_features is not None
            else torch.tensor(0.0, device=self.device)
        )
        stats = {
            "loss": zero,
            "valid_ratio": zero,
            "summary": zero,
            "summary_adapter": zero,
            "text": zero,
            "text_valid_ratio": zero,
            "text_acc": zero,
            "adapter_text": zero,
            "adapter_text_valid_ratio": zero,
            "adapter_text_acc": zero,
            "text_distill": zero,
            "text_distill_valid_ratio": zero,
            "text_distill_teacher_conf": zero,
            "text_distill_agreement": zero,
            "text_pseudo_ce": zero,
            "text_pseudo_ce_valid_ratio": zero,
            "text_pseudo_ce_teacher_conf": zero,
            "text_pseudo_ce_agreement": zero,
            "text_contrast": zero,
            "text_contrast_valid_ratio": zero,
            "text_contrast_teacher_conf": zero,
            "text_contrast_agreement": zero,
            "render_consistency": zero,
            "render_consistency_valid_ratio": zero,
            "adapter_text_distill": zero,
            "adapter_text_distill_valid_ratio": zero,
            "adapter_text_distill_teacher_conf": zero,
            "adapter_text_distill_agreement": zero,
            "adapter_text_pseudo_ce": zero,
            "adapter_text_pseudo_ce_valid_ratio": zero,
            "adapter_text_pseudo_ce_teacher_conf": zero,
            "adapter_text_pseudo_ce_agreement": zero,
            "adapter_decoder_anchor": zero,
            "query_logit_distill": zero,
            "query_logit_distill_valid_ratio": zero,
            "query_logit_distill_teacher_conf": zero,
            "query_logit_distill_agreement": zero,
            "query_support_distill": zero,
            "query_support_distill_valid_ratio": zero,
            "query_support_distill_teacher_conf": zero,
            "query_support_distill_top1_agreement": zero,
            "proposal_consistency": zero,
            "proposal_consistency_valid_ratio": zero,
            "proposal_contrast": zero,
            "proposal_contrast_valid_ratio": zero,
            "proposal_contrast_num_proposals": zero,
            "view_weight_mean": zero,
            "view_weight_max": zero,
        }
        teacher_features = getattr(self, "direct_point_teacher_features", None)
        has_teacher_cache = teacher_features is not None
        if (
            self.direct_point_loss_weight <= 0
            or (target_features is None and not has_teacher_cache)
            or not self._is_hybrid
            or self.direct_point_sample_count <= 0
        ):
            return stats

        points_all = (
            self.direct_point_pool
            if self.direct_point_pool is not None
            else self.model.get_xyz().to(device=self.device, dtype=torch.float32)
        )
        num_source_points = int(points_all.shape[0])
        if num_source_points <= 0:
            return stats
        sample_count = min(self.direct_point_sample_count, num_source_points)

        if has_teacher_cache:
            assert teacher_features is not None
            if int(teacher_features.shape[0]) != num_source_points:
                raise RuntimeError(
                    "direct_point_teacher_cache point count does not match direct point pool: "
                    f"{int(teacher_features.shape[0])} vs {num_source_points}"
                )
            with torch.no_grad():
                source_indices = torch.arange(num_source_points, device=self.device)
                teacher_valid = getattr(self, "direct_point_teacher_valid", None)
                if teacher_valid is not None:
                    source_indices = source_indices[teacher_valid.to(self.device).bool()]
                if source_indices.numel() == 0:
                    return stats
                sample_count = min(sample_count, int(source_indices.numel()))
                visible_indices = None
                visible_fraction = float(
                    getattr(self, "direct_point_cached_visible_fraction", 0.0) or 0.0
                )
                if (
                    visible_fraction > 0.0
                    and rendered_compact is not None
                    and getattr(self, "direct_point_render_consistency_weight", 0.0) > 0
                    and "pose_w2c" in batch
                    and getattr(self, "renderer", None) is not None
                ):
                    spatial_size = rendered_compact.shape[-2:]
                    depth_map = self._canonicalize_spatial_map(
                        render_result.get("depth_map"),
                        batch_size=rendered_compact.shape[0],
                        spatial_size=spatial_size,
                    )
                    alpha_map = self._canonicalize_spatial_map(
                        render_result.get("alpha_map"),
                        batch_size=rendered_compact.shape[0],
                        spatial_size=spatial_size,
                    )
                    K = self.renderer.K.float()
                    if spatial_size != (self.renderer.image_height, self.renderer.image_width):
                        scale_x = float(spatial_size[1]) / float(self.renderer.image_width)
                        scale_y = float(spatial_size[0]) / float(self.renderer.image_height)
                        K = K.clone()
                        K[0, 0] *= scale_x
                        K[0, 2] *= scale_x
                        K[1, 1] *= scale_y
                        K[1, 2] *= scale_y

                    visible_quota = min(
                        sample_count,
                        max(1, int(round(float(sample_count) * visible_fraction))),
                    )
                    balance_visible = bool(
                        getattr(self, "direct_point_cached_visible_balance", False)
                    )
                    visible_candidate_multiplier = max(
                        1,
                        int(
                            getattr(
                                self,
                                "direct_point_cached_visible_candidate_multiplier",
                                1,
                            )
                            or 1
                        ),
                    )
                    candidate_count = min(
                        int(source_indices.numel()),
                        max(sample_count, visible_quota * 64),
                    )
                    if source_indices.numel() > candidate_count:
                        order = torch.randperm(source_indices.numel(), device=self.device)[
                            :candidate_count
                        ]
                        candidate_indices = source_indices[order]
                    else:
                        candidate_indices = source_indices
                    visible_select_count = visible_quota
                    if balance_visible:
                        visible_select_count = min(
                            candidate_count,
                            max(visible_quota, visible_quota * visible_candidate_multiplier),
                        )
                    visible_rel, _ = _select_visible_gaussian_indices(
                        points_all[candidate_indices],
                        batch["pose_w2c"].to(self.device).float(),
                        K,
                        image_height=spatial_size[0],
                        image_width=spatial_size[1],
                        sample_count=visible_select_count,
                        depth_map=depth_map,
                        alpha_map=alpha_map,
                        depth_tolerance=self.direct_point_depth_tolerance,
                        relative_depth_tolerance=self.direct_point_relative_depth_tolerance,
                        alpha_threshold=self.direct_point_alpha_threshold,
                    )
                    if visible_rel.numel() > 0:
                        visible_candidates = candidate_indices[visible_rel]
                        if (
                            balance_visible
                            and visible_candidates.numel() > visible_quota
                        ):
                            visible_indices = self._subsample_direct_point_indices(
                                visible_candidates,
                                visible_quota,
                            )
                        else:
                            visible_indices = visible_candidates[:visible_quota]

                if visible_indices is not None and visible_indices.numel() > 0:
                    visible_indices = visible_indices[:sample_count]
                    remaining_count = sample_count - int(visible_indices.numel())
                    if remaining_count > 0:
                        remaining_mask = ~torch.isin(source_indices, visible_indices)
                        remaining_source = source_indices[remaining_mask]
                        if remaining_source.numel() > remaining_count:
                            order = torch.randperm(
                                remaining_source.numel(),
                                device=self.device,
                            )
                            remaining_source = remaining_source[order]
                        remaining_indices = self._subsample_direct_point_indices(
                            remaining_source,
                            remaining_count,
                        )
                        indices = torch.cat([visible_indices, remaining_indices], dim=0)
                    else:
                        indices = visible_indices
                else:
                    if source_indices.numel() > sample_count:
                        order = torch.randperm(source_indices.numel(), device=self.device)
                        source_indices = source_indices[order]
                    indices = self._subsample_direct_point_indices(source_indices, sample_count)
                points = points_all[indices]
                point_targets = teacher_features[indices].float()
                valid = torch.ones(indices.shape[0], device=self.device, dtype=torch.bool)
                view_counts_all = getattr(self, "direct_point_teacher_view_counts", None)
                if view_counts_all is None:
                    view_counts = valid.float()
                else:
                    view_counts = view_counts_all[indices].float()
        else:
            assert target_features is not None
            target = target_features.detach()
            spatial_size = target.shape[-2:]
            depth_map = self._canonicalize_spatial_map(
                render_result.get("depth_map"),
                batch_size=target.shape[0],
                spatial_size=spatial_size,
            )
            alpha_map = self._canonicalize_spatial_map(
                render_result.get("alpha_map"),
                batch_size=target.shape[0],
                spatial_size=spatial_size,
            )
            K = self.renderer.K.float()
            if spatial_size != (self.renderer.image_height, self.renderer.image_width):
                scale_x = float(spatial_size[1]) / float(self.renderer.image_width)
                scale_y = float(spatial_size[0]) / float(self.renderer.image_height)
                K = K.clone()
                K[0, 0] *= scale_x
                K[0, 2] *= scale_x
                K[1, 1] *= scale_y
                K[1, 2] *= scale_y

            with torch.no_grad():
                candidate_count = min(num_source_points, max(sample_count, sample_count * 64))
                candidate_indices = torch.randperm(num_source_points, device=self.device)[:candidate_count]
                candidate_points = points_all[candidate_indices]
                visible_rel, _ = _select_visible_gaussian_indices(
                    candidate_points,
                    batch["pose_w2c"].to(self.device).float(),
                    K,
                    image_height=spatial_size[0],
                    image_width=spatial_size[1],
                    sample_count=sample_count,
                    depth_map=depth_map,
                    alpha_map=alpha_map,
                    depth_tolerance=self.direct_point_depth_tolerance,
                    relative_depth_tolerance=self.direct_point_relative_depth_tolerance,
                    alpha_threshold=self.direct_point_alpha_threshold,
                )
                if visible_rel.numel() > 0:
                    indices = candidate_indices[visible_rel]
                else:
                    indices = candidate_indices[:sample_count]
                indices = self._subsample_direct_point_indices(indices, sample_count)
                points = points_all[indices]
                point_targets, valid, view_counts = _sample_multiview_radio_targets(
                    points,
                    target.float(),
                    batch["pose_w2c"].to(self.device).float(),
                    K,
                    depth_map=depth_map,
                    alpha_map=alpha_map,
                    depth_tolerance=self.direct_point_depth_tolerance,
                    relative_depth_tolerance=self.direct_point_relative_depth_tolerance,
                    alpha_threshold=self.direct_point_alpha_threshold,
                )
        valid_ratio = view_counts.float().gt(0).float().mean()
        stats["valid_ratio"] = valid_ratio
        if not valid.any():
            return stats

        valid_indices = indices[valid]
        point_labels = None
        direct_point_pool_labels = getattr(self, "direct_point_pool_labels", None)
        if direct_point_pool_labels is not None:
            point_labels = direct_point_pool_labels[valid_indices]
        direct_point_feature_key = getattr(self, "direct_point_feature_key", "features")
        if self.direct_point_query_mode == "knn":
            query_kwargs = {"k": self.direct_point_k}
            if getattr(self, "direct_point_candidate_k", 0) > 0:
                query_kwargs["candidate_k"] = self.direct_point_candidate_k
            if direct_point_feature_key == "features":
                compact = self.model.query_compact_points(
                    points[valid],
                    **query_kwargs,
                )
            else:
                compact_aux = self.model.query_compact_points(
                    points[valid],
                    return_aux=True,
                    **query_kwargs,
                )
                if direct_point_feature_key not in compact_aux:
                    raise KeyError(
                        f"Requested direct_point_feature_key='{direct_point_feature_key}', "
                        f"available={sorted(compact_aux.keys())}"
                    )
                compact = compact_aux[direct_point_feature_key]
        else:
            if self.direct_point_pool is not None:
                model_count = self._num_model_gaussians()
                if num_source_points != model_count:
                    raise RuntimeError(
                        "direct_point_query_mode=gaussian_index with "
                        "direct_point_source=label_ply requires row-aligned point "
                        f"and Gaussian counts, got {num_source_points} points vs "
                        f"{model_count} Gaussians"
                    )
            gaussian_points = (
                points[valid]
                if getattr(
                    self,
                    "direct_point_gaussian_position_mode",
                    "gaussian_center",
                )
                == "label_point"
                else None
            )
            if direct_point_feature_key == "features":
                if gaussian_points is None:
                    compact = self.model.query_gaussian_points(valid_indices)
                else:
                    compact = self.model.query_gaussian_points(
                        valid_indices,
                        points_xyz=gaussian_points,
                    )
            else:
                if gaussian_points is None:
                    compact_aux = self.model.query_gaussian_points(
                        valid_indices,
                        return_aux=True,
                    )
                else:
                    compact_aux = self.model.query_gaussian_points(
                        valid_indices,
                        points_xyz=gaussian_points,
                        return_aux=True,
                    )
                if direct_point_feature_key not in compact_aux:
                    raise KeyError(
                        f"Requested direct_point_feature_key='{direct_point_feature_key}', "
                        f"available={sorted(compact_aux.keys())}"
                    )
                compact = compact_aux[direct_point_feature_key]
        point_target_rows = point_targets[valid].float()
        teacher_feature_space = str(
            getattr(self, "direct_point_teacher_feature_space", "radio") or "radio"
        ).lower()
        summary_teacher_points = None
        if teacher_feature_space in {"siglip", "siglip2", "summary", "siglip_summary"}:
            teacher_feature_space = "siglip_summary"
            summary_teacher_points = F.normalize(point_target_rows, dim=-1)
        elif teacher_feature_space not in {"radio", "teacher", "teacher_1280"}:
            raise ValueError(
                "direct_point_teacher_feature_space must be 'radio' or "
                f"'siglip_summary', got {teacher_feature_space!r}"
            )
        decoded_points, target_map = self._decode_direct_point_map(
            compact,
            point_target_rows,
        )
        target_summary = None
        target_summary_points_for_losses = summary_teacher_points

        def _target_summary_points() -> torch.Tensor:
            nonlocal target_summary, target_summary_points_for_losses
            if target_summary_points_for_losses is not None:
                return target_summary_points_for_losses
            if self.siglip_summary_head is None:
                raise RuntimeError(
                    "SigLIP summary projection is required for RADIO-space direct "
                    "point targets"
                )
            if target_summary is None:
                with torch.no_grad():
                    target_summary = self._project_summary_head_features(target_map.float())
            target_summary_points_for_losses = self._direct_point_map_to_rows(target_summary)
            return target_summary_points_for_losses

        sample_weights = _direct_point_view_count_weights(
            view_counts[valid],
            mode=getattr(self, "direct_point_view_count_weighting", "none"),
            min_weight=float(getattr(self, "direct_point_view_count_min_weight", 0.0)),
            percentile_low=float(getattr(self, "direct_point_view_count_percentile_low", 5.0)),
            percentile_high=float(getattr(self, "direct_point_view_count_percentile_high", 95.0)),
        )
        if sample_weights is not None:
            sample_weights = sample_weights.to(device=self.device, dtype=torch.float32)
            stats["view_weight_mean"] = sample_weights.mean().detach()
            stats["view_weight_max"] = sample_weights.max().detach()
        weight_mask = _direct_point_weight_mask(decoded_points, sample_weights)
        if teacher_feature_space == "siglip_summary":
            distill = {"total": decoded_points.sum() * 0.0}
        elif weight_mask is None:
            distill = self.distill_loss_fn(decoded_points, target_map.float())
        else:
            distill = self.distill_loss_fn(decoded_points, target_map.float(), mask=weight_mask)
        summary_loss = decoded_points.sum() * 0.0
        pred_summary = None
        if (
            self.direct_point_summary_alignment_weight > 0
            and self.siglip_summary_head is not None
        ):
            pred_summary = self._project_summary_head_features(decoded_points)
            pred_summary_points = self._direct_point_map_to_rows(pred_summary)
            target_summary_points = _target_summary_points()
            summary_per_point = 1.0 - (pred_summary_points * target_summary_points).sum(dim=1)
            summary_loss = _weighted_vector_mean(summary_per_point, sample_weights)
        adapter_loss = decoded_points.sum() * 0.0
        pred_summary_points = None
        if (
            getattr(self, "direct_point_summary_adapter_weight", 0.0) > 0
            and getattr(self, "point_summary_adapter", None) is not None
            and (
                self.siglip_summary_head is not None
                or summary_teacher_points is not None
            )
        ):
            pred_summary_points = F.normalize(
                self._project_point_summary_adapter(compact, valid_indices, view_counts[valid]),
                dim=-1,
            )
            target_summary_points = _target_summary_points()
            adapter_per_point = 1.0 - (pred_summary_points * target_summary_points).sum(dim=-1)
            adapter_loss = _weighted_vector_mean(adapter_per_point, sample_weights)
        text_loss = decoded_points.sum() * 0.0
        text_valid_ratio = text_loss.detach()
        text_acc = text_loss.detach()
        if (
            getattr(self, "direct_point_text_loss_weight", 0.0) > 0
            and self.siglip_summary_head is not None
        ):
            if pred_summary is None:
                pred_summary = self._project_summary_head_features(decoded_points)
            pred_summary_text = self._direct_point_map_to_rows(pred_summary)
            text_loss, text_valid_ratio, text_acc = self._compute_direct_point_text_ce(
                pred_summary_text,
                point_labels,
            )
        adapter_text_loss = decoded_points.sum() * 0.0
        adapter_text_valid_ratio = adapter_text_loss.detach()
        adapter_text_acc = adapter_text_loss.detach()
        if (
            getattr(self, "direct_point_adapter_text_loss_weight", 0.0) > 0
            and getattr(self, "point_summary_adapter", None) is not None
        ):
            if pred_summary_points is None:
                pred_summary_points = F.normalize(
                    self._project_point_summary_adapter(compact, valid_indices, view_counts[valid]),
                    dim=-1,
                )
            (
                adapter_text_loss,
                adapter_text_valid_ratio,
                adapter_text_acc,
            ) = self._compute_direct_point_text_ce(
                pred_summary_points,
                point_labels,
            )
        text_distill_loss = decoded_points.sum() * 0.0
        text_distill_valid_ratio = text_distill_loss.detach()
        text_distill_teacher_conf = text_distill_loss.detach()
        text_distill_agreement = text_distill_loss.detach()
        text_pseudo_ce_loss = decoded_points.sum() * 0.0
        text_pseudo_ce_valid_ratio = text_pseudo_ce_loss.detach()
        text_pseudo_ce_teacher_conf = text_pseudo_ce_loss.detach()
        text_pseudo_ce_agreement = text_pseudo_ce_loss.detach()
        text_contrast_loss = decoded_points.sum() * 0.0
        text_contrast_valid_ratio = text_contrast_loss.detach()
        text_contrast_teacher_conf = text_contrast_loss.detach()
        text_contrast_agreement = text_contrast_loss.detach()
        render_consistency_loss = decoded_points.sum() * 0.0
        render_consistency_valid_ratio = render_consistency_loss.detach()
        adapter_text_distill_loss = decoded_points.sum() * 0.0
        adapter_text_distill_valid_ratio = adapter_text_distill_loss.detach()
        adapter_text_distill_teacher_conf = adapter_text_distill_loss.detach()
        adapter_text_distill_agreement = adapter_text_distill_loss.detach()
        adapter_text_pseudo_ce_loss = decoded_points.sum() * 0.0
        adapter_text_pseudo_ce_valid_ratio = adapter_text_pseudo_ce_loss.detach()
        adapter_text_pseudo_ce_teacher_conf = adapter_text_pseudo_ce_loss.detach()
        adapter_text_pseudo_ce_agreement = adapter_text_pseudo_ce_loss.detach()
        adapter_decoder_anchor_loss = decoded_points.sum() * 0.0
        query_logit_distill_loss = decoded_points.sum() * 0.0
        query_logit_distill_valid_ratio = query_logit_distill_loss.detach()
        query_logit_distill_teacher_conf = query_logit_distill_loss.detach()
        query_logit_distill_agreement = query_logit_distill_loss.detach()
        query_support_distill_loss = decoded_points.sum() * 0.0
        query_support_distill_valid_ratio = query_support_distill_loss.detach()
        query_support_distill_teacher_conf = query_support_distill_loss.detach()
        query_support_distill_top1_agreement = query_support_distill_loss.detach()
        proposal_consistency_loss = decoded_points.sum() * 0.0
        proposal_consistency_valid_ratio = proposal_consistency_loss.detach()
        proposal_contrast_loss = decoded_points.sum() * 0.0
        proposal_contrast_valid_ratio = proposal_contrast_loss.detach()
        proposal_contrast_num_proposals = proposal_contrast_loss.detach()
        if (
            getattr(self, "direct_point_text_distill_weight", 0.0) > 0
            and self.siglip_summary_head is not None
        ):
            if pred_summary is None:
                pred_summary = self._project_summary_head_features(decoded_points)
            pred_summary_points_for_distill = self._direct_point_map_to_rows(pred_summary)
            if target_summary is None:
                target_summary_points = _target_summary_points()
            else:
                target_summary_points = self._direct_point_map_to_rows(target_summary)
            (
                text_distill_loss,
                text_distill_valid_ratio,
                text_distill_teacher_conf,
                text_distill_agreement,
            ) = self._compute_direct_point_text_distill_kl(
                pred_summary_points_for_distill,
                target_summary_points,
                sample_weights=sample_weights,
            )
        if (
            getattr(self, "direct_point_text_contrast_weight", 0.0) > 0
            and self.siglip_summary_head is not None
        ):
            if pred_summary is None:
                pred_summary = self._project_summary_head_features(decoded_points)
            pred_summary_points_for_contrast = self._direct_point_map_to_rows(pred_summary)
            if target_summary is None:
                target_summary_points = _target_summary_points()
            else:
                target_summary_points = self._direct_point_map_to_rows(target_summary)
            (
                text_contrast_loss,
                text_contrast_valid_ratio,
                text_contrast_teacher_conf,
                text_contrast_agreement,
            ) = self._compute_direct_point_text_contrast(
                pred_summary_points_for_contrast,
                target_summary_points,
                sample_weights=sample_weights,
            )
        if (
            getattr(self, "direct_point_render_consistency_weight", 0.0) > 0
            and rendered_compact is not None
        ):
            render_consistency_loss, render_consistency_valid_ratio = (
                self._compute_direct_point_render_consistency(
                    compact,
                    points[valid],
                    batch=batch,
                    render_result=render_result,
                    rendered_compact=rendered_compact,
                    sample_weights=sample_weights,
                )
            )
        if (
            getattr(self, "direct_point_text_pseudo_ce_weight", 0.0) > 0
            and self.siglip_summary_head is not None
        ):
            if pred_summary is None:
                pred_summary = self._project_summary_head_features(decoded_points)
            pred_summary_points_for_pseudo_ce = self._direct_point_map_to_rows(pred_summary)
            if target_summary is None:
                target_summary_points = _target_summary_points()
            else:
                target_summary_points = self._direct_point_map_to_rows(target_summary)
            (
                text_pseudo_ce_loss,
                text_pseudo_ce_valid_ratio,
                text_pseudo_ce_teacher_conf,
                text_pseudo_ce_agreement,
            ) = self._compute_multi_split_direct_point_text_pseudo_ce(
                pred_summary_points_for_pseudo_ce,
                target_summary_points,
                logit_scale=getattr(self, "direct_point_text_pseudo_ce_logit_scale", 1.0),
                confidence_threshold=getattr(
                    self,
                    "direct_point_text_pseudo_ce_confidence_threshold",
                    0.0,
                ),
                center_logits=getattr(
                    self,
                    "direct_point_text_pseudo_ce_center_logits",
                    False,
                ),
                banks=getattr(self, "direct_point_text_pseudo_ce_banks", []),
            )
        if (
            getattr(self, "direct_point_adapter_text_distill_weight", 0.0) > 0
            and getattr(self, "point_summary_adapter", None) is not None
            and (
                self.siglip_summary_head is not None
                or summary_teacher_points is not None
            )
        ):
            if pred_summary_points is None:
                pred_summary_points = F.normalize(
                    self._project_point_summary_adapter(compact, valid_indices, view_counts[valid]),
                    dim=-1,
                )
            if target_summary is None:
                target_summary_points = _target_summary_points()
            else:
                target_summary_points = self._direct_point_map_to_rows(target_summary)
            (
                adapter_text_distill_loss,
                adapter_text_distill_valid_ratio,
                adapter_text_distill_teacher_conf,
                adapter_text_distill_agreement,
            ) = self._compute_direct_point_text_distill_kl(
                pred_summary_points,
                target_summary_points,
                sample_weights=sample_weights,
            )
        if (
            getattr(self, "direct_point_adapter_text_pseudo_ce_weight", 0.0) > 0
            and getattr(self, "point_summary_adapter", None) is not None
            and (
                self.siglip_summary_head is not None
                or summary_teacher_points is not None
            )
        ):
            if pred_summary_points is None:
                pred_summary_points = F.normalize(
                    self._project_point_summary_adapter(compact, valid_indices, view_counts[valid]),
                    dim=-1,
                )
            if target_summary is None:
                target_summary_points = _target_summary_points()
            else:
                target_summary_points = self._direct_point_map_to_rows(target_summary)
            (
                adapter_text_pseudo_ce_loss,
                adapter_text_pseudo_ce_valid_ratio,
                adapter_text_pseudo_ce_teacher_conf,
                adapter_text_pseudo_ce_agreement,
            ) = self._compute_multi_split_direct_point_text_pseudo_ce(
                pred_summary_points,
                target_summary_points,
                logit_scale=getattr(
                    self,
                    "direct_point_adapter_text_pseudo_ce_logit_scale",
                    1.0,
                ),
                confidence_threshold=getattr(
                    self,
                    "direct_point_adapter_text_pseudo_ce_confidence_threshold",
                    0.0,
                ),
                center_logits=getattr(
                    self,
                    "direct_point_adapter_text_pseudo_ce_center_logits",
                    False,
                ),
                banks=getattr(self, "direct_point_adapter_text_pseudo_ce_banks", []),
            )
        if (
            getattr(self, "direct_point_adapter_decoder_anchor_weight", 0.0) > 0
            and getattr(self, "point_summary_adapter", None) is not None
            and self.siglip_summary_head is not None
        ):
            if pred_summary_points is None:
                pred_summary_points = F.normalize(
                    self._project_point_summary_adapter(compact, valid_indices, view_counts[valid]),
                    dim=-1,
                )
            if pred_summary is None:
                with torch.no_grad():
                    decoder_anchor_summary = self._project_summary_head_features(decoded_points)
            else:
                decoder_anchor_summary = pred_summary.detach()
            decoder_anchor_points = self._direct_point_map_to_rows(decoder_anchor_summary)
            adapter_decoder_anchor_loss = (
                1.0 - (pred_summary_points * decoder_anchor_points).sum(dim=-1).mean()
            )
        if (
            getattr(self, "direct_point_query_logit_distill_weight", 0.0) > 0
            and getattr(self, "direct_point_query_logit_distill_embeddings", None) is not None
        ):
            if pred_summary_points is None:
                if (
                    getattr(self, "point_summary_adapter", None) is not None
                    and (
                        self.siglip_summary_head is not None
                        or summary_teacher_points is not None
                    )
                ):
                    pred_summary_points = F.normalize(
                        self._project_point_summary_adapter(compact, valid_indices, view_counts[valid]),
                        dim=-1,
                    )
                else:
                    if pred_summary is None:
                        pred_summary = self._project_summary_head_features(decoded_points)
                    pred_summary_points = self._direct_point_map_to_rows(pred_summary)
            teacher_query_summary = _target_summary_points()
            query_loss, query_stats = compute_direct_point_query_logit_distill_loss(
                pred_summary_points,
                teacher_query_summary,
                self.direct_point_query_logit_distill_embeddings,
                temperature=getattr(self, "direct_point_query_logit_distill_temperature", 1.0),
                confidence_threshold=getattr(
                    self,
                    "direct_point_query_logit_distill_confidence_threshold",
                    0.0,
                ),
                sample_weights=sample_weights,
            )
            query_logit_distill_loss = query_loss
            query_logit_distill_valid_ratio = query_stats["valid_ratio"]
            query_logit_distill_teacher_conf = query_stats["teacher_conf"]
            query_logit_distill_agreement = query_stats["agreement"]
        if (
            getattr(self, "direct_point_query_support_distill_weight", 0.0) > 0
            and getattr(self, "direct_point_query_support_distill_embeddings", None) is not None
        ):
            if pred_summary_points is None:
                if (
                    getattr(self, "point_summary_adapter", None) is not None
                    and (
                        self.siglip_summary_head is not None
                        or summary_teacher_points is not None
                    )
                ):
                    pred_summary_points = F.normalize(
                        self._project_point_summary_adapter(compact, valid_indices, view_counts[valid]),
                        dim=-1,
                    )
                else:
                    if pred_summary is None:
                        pred_summary = self._project_summary_head_features(decoded_points)
                    pred_summary_points = self._direct_point_map_to_rows(pred_summary)
            teacher_query_summary = _target_summary_points()
            support_loss, support_stats = compute_direct_point_query_support_distill_loss(
                pred_summary_points,
                teacher_query_summary,
                self.direct_point_query_support_distill_embeddings,
                temperature=getattr(self, "direct_point_query_support_distill_temperature", 0.25),
                confidence_threshold=getattr(
                    self,
                    "direct_point_query_support_distill_confidence_threshold",
                    0.0,
                ),
                sample_weights=sample_weights,
                support_logit_norm=getattr(
                    self,
                    "direct_point_query_support_distill_logit_norm",
                    "none",
                ),
            )
            query_support_distill_loss = support_loss
            query_support_distill_valid_ratio = support_stats["valid_ratio"]
            query_support_distill_teacher_conf = support_stats["teacher_conf"]
            query_support_distill_top1_agreement = support_stats["top1_agreement"]
        if (
            getattr(self, "direct_point_proposal_consistency_weight", 0.0) > 0
            or getattr(self, "direct_point_proposal_contrast_weight", 0.0) > 0
        ):
            proposal_space = str(
                getattr(self, "direct_point_proposal_space", "auto") or "auto"
            )
            proposal_summary = None
            if proposal_space not in {"auto", "adapter", "decoder"}:
                raise ValueError(
                    "direct_point_proposal_space must be one of: auto, adapter, decoder"
                )
            if (
                proposal_space in {"auto", "adapter"}
                and getattr(self, "point_summary_adapter", None) is not None
            ):
                if pred_summary_points is None:
                    pred_summary_points = F.normalize(
                        self._project_point_summary_adapter(compact, valid_indices, view_counts[valid]),
                        dim=-1,
                    )
                proposal_summary = pred_summary_points
            elif self.siglip_summary_head is not None:
                if pred_summary is None:
                    pred_summary = self._project_summary_head_features(decoded_points)
                proposal_summary = self._direct_point_map_to_rows(pred_summary)
            if proposal_summary is not None:
                proposal_labels = build_voxel_proposal_labels(
                    points[valid],
                    voxel_size=float(
                        getattr(self, "direct_point_proposal_voxel_size", 0.05)
                    ),
                )
                memory = build_proposal_memory_from_labels(
                    proposal_summary,
                    proposal_labels,
                    confidence=sample_weights,
                )
                assigned = memory.row_to_proposal >= 0
                min_count = max(
                    1, int(getattr(self, "direct_point_proposal_min_count", 2) or 1)
                )
                if min_count > 1 and bool(assigned.any()):
                    assigned_indices = memory.row_to_proposal[assigned]
                    count_mask = memory.counts[assigned_indices] >= min_count
                    full_mask = torch.zeros_like(assigned)
                    full_mask[assigned] = count_mask
                    assigned = full_mask
                proposal_consistency_valid_ratio = assigned.float().mean().detach()
                if bool(assigned.any()):
                    prototypes = F.normalize(
                        memory.pooled_values[memory.row_to_proposal[assigned]].detach(),
                        dim=-1,
                    )
                    per_point = 1.0 - (
                        F.normalize(proposal_summary[assigned].float(), dim=-1)
                        * prototypes.float()
                    ).sum(dim=-1)
                    proposal_weights = (
                        sample_weights[assigned] if sample_weights is not None else None
                    )
                    proposal_consistency_loss = _weighted_vector_mean(
                        per_point,
                        proposal_weights,
                    )
                if getattr(self, "direct_point_proposal_contrast_weight", 0.0) > 0:
                    proposal_contrast_loss, proposal_contrast_stats = (
                        compute_region_prototype_contrast_loss(
                            proposal_summary,
                            proposal_labels,
                            confidence=sample_weights,
                            min_count=min_count,
                            temperature=float(
                                getattr(
                                    self,
                                    "direct_point_proposal_contrast_temperature",
                                    0.07,
                                )
                            ),
                        )
                    )
                    proposal_contrast_valid_ratio = proposal_contrast_stats[
                        "valid_ratio"
                    ]
                    proposal_contrast_num_proposals = proposal_contrast_stats[
                        "num_proposals"
                    ]
        stats["summary"] = summary_loss
        stats["summary_adapter"] = adapter_loss
        stats["text"] = text_loss
        stats["text_valid_ratio"] = text_valid_ratio
        stats["text_acc"] = text_acc
        stats["adapter_text"] = adapter_text_loss
        stats["adapter_text_valid_ratio"] = adapter_text_valid_ratio
        stats["adapter_text_acc"] = adapter_text_acc
        stats["text_distill"] = text_distill_loss
        stats["text_distill_valid_ratio"] = text_distill_valid_ratio
        stats["text_distill_teacher_conf"] = text_distill_teacher_conf
        stats["text_distill_agreement"] = text_distill_agreement
        stats["text_pseudo_ce"] = text_pseudo_ce_loss
        stats["text_pseudo_ce_valid_ratio"] = text_pseudo_ce_valid_ratio
        stats["text_pseudo_ce_teacher_conf"] = text_pseudo_ce_teacher_conf
        stats["text_pseudo_ce_agreement"] = text_pseudo_ce_agreement
        stats["text_contrast"] = text_contrast_loss
        stats["text_contrast_valid_ratio"] = text_contrast_valid_ratio
        stats["text_contrast_teacher_conf"] = text_contrast_teacher_conf
        stats["text_contrast_agreement"] = text_contrast_agreement
        stats["render_consistency"] = render_consistency_loss
        stats["render_consistency_valid_ratio"] = render_consistency_valid_ratio
        stats["adapter_text_distill"] = adapter_text_distill_loss
        stats["adapter_text_distill_valid_ratio"] = adapter_text_distill_valid_ratio
        stats["adapter_text_distill_teacher_conf"] = adapter_text_distill_teacher_conf
        stats["adapter_text_distill_agreement"] = adapter_text_distill_agreement
        stats["adapter_text_pseudo_ce"] = adapter_text_pseudo_ce_loss
        stats["adapter_text_pseudo_ce_valid_ratio"] = adapter_text_pseudo_ce_valid_ratio
        stats["adapter_text_pseudo_ce_teacher_conf"] = adapter_text_pseudo_ce_teacher_conf
        stats["adapter_text_pseudo_ce_agreement"] = adapter_text_pseudo_ce_agreement
        stats["adapter_decoder_anchor"] = adapter_decoder_anchor_loss
        stats["query_logit_distill"] = query_logit_distill_loss
        stats["query_logit_distill_valid_ratio"] = query_logit_distill_valid_ratio
        stats["query_logit_distill_teacher_conf"] = query_logit_distill_teacher_conf
        stats["query_logit_distill_agreement"] = query_logit_distill_agreement
        stats["query_support_distill"] = query_support_distill_loss
        stats["query_support_distill_valid_ratio"] = query_support_distill_valid_ratio
        stats["query_support_distill_teacher_conf"] = query_support_distill_teacher_conf
        stats["query_support_distill_top1_agreement"] = query_support_distill_top1_agreement
        stats["proposal_consistency"] = proposal_consistency_loss
        stats["proposal_consistency_valid_ratio"] = proposal_consistency_valid_ratio
        stats["proposal_contrast"] = proposal_contrast_loss
        stats["proposal_contrast_valid_ratio"] = proposal_contrast_valid_ratio
        stats["proposal_contrast_num_proposals"] = proposal_contrast_num_proposals
        stats["loss"] = (
            distill["total"]
            + self.direct_point_summary_alignment_weight * summary_loss
            + getattr(self, "direct_point_summary_adapter_weight", 0.0) * adapter_loss
            + getattr(self, "direct_point_text_loss_weight", 0.0) * text_loss
            + getattr(self, "direct_point_adapter_text_loss_weight", 0.0) * adapter_text_loss
            + getattr(self, "direct_point_text_distill_weight", 0.0) * text_distill_loss
            + getattr(self, "direct_point_text_pseudo_ce_weight", 0.0)
            * text_pseudo_ce_loss
            + getattr(self, "direct_point_text_contrast_weight", 0.0)
            * text_contrast_loss
            + getattr(self, "direct_point_render_consistency_weight", 0.0)
            * render_consistency_loss
            + getattr(self, "direct_point_adapter_text_distill_weight", 0.0)
            * adapter_text_distill_loss
            + getattr(self, "direct_point_adapter_text_pseudo_ce_weight", 0.0)
            * adapter_text_pseudo_ce_loss
            + getattr(self, "direct_point_adapter_decoder_anchor_weight", 0.0)
            * adapter_decoder_anchor_loss
            + getattr(self, "direct_point_query_logit_distill_weight", 0.0)
            * query_logit_distill_loss
            + getattr(self, "direct_point_query_support_distill_weight", 0.0)
            * query_support_distill_loss
            + getattr(self, "direct_point_proposal_consistency_weight", 0.0)
            * proposal_consistency_loss
            + getattr(self, "direct_point_proposal_contrast_weight", 0.0)
            * proposal_contrast_loss
        )
        return stats

    def _compute_direct_point_render_consistency(
        self,
        direct_compact: torch.Tensor,
        points_xyz: torch.Tensor,
        *,
        batch: Dict[str, torch.Tensor],
        render_result: Dict[str, torch.Tensor],
        rendered_compact: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align direct 3-D compact readout with the same field rendered in 2-D."""
        zero = direct_compact.sum() * 0.0
        if direct_compact.numel() == 0 or points_xyz.numel() == 0:
            return zero, zero.detach()
        if rendered_compact.ndim != 4:
            raise ValueError(
                f"rendered_compact must be [B,C,H,W], got {tuple(rendered_compact.shape)}"
            )
        if direct_compact.shape[1] != rendered_compact.shape[1]:
            raise ValueError(
                "direct compact feature dimension must match rendered compact map: "
                f"{direct_compact.shape[1]} vs {rendered_compact.shape[1]}"
            )

        spatial_size = rendered_compact.shape[-2:]
        depth_map = self._canonicalize_spatial_map(
            render_result.get("depth_map"),
            batch_size=rendered_compact.shape[0],
            spatial_size=spatial_size,
        )
        alpha_map = self._canonicalize_spatial_map(
            render_result.get("alpha_map"),
            batch_size=rendered_compact.shape[0],
            spatial_size=spatial_size,
        )
        K = self.renderer.K.float()
        if spatial_size != (self.renderer.image_height, self.renderer.image_width):
            scale_x = float(spatial_size[1]) / float(self.renderer.image_width)
            scale_y = float(spatial_size[0]) / float(self.renderer.image_height)
            K = K.clone()
            K[0, 0] *= scale_x
            K[0, 2] *= scale_x
            K[1, 1] *= scale_y
            K[1, 2] *= scale_y

        rendered_targets, render_valid, _view_counts = _sample_multiview_radio_targets(
            points_xyz,
            rendered_compact.float(),
            batch["pose_w2c"].to(self.device).float(),
            K,
            depth_map=depth_map,
            alpha_map=alpha_map,
            depth_tolerance=getattr(self, "direct_point_depth_tolerance", 0.08),
            relative_depth_tolerance=getattr(
                self, "direct_point_relative_depth_tolerance", 0.02
            ),
            alpha_threshold=getattr(self, "direct_point_alpha_threshold", 0.0),
        )
        valid_ratio = render_valid.float().mean()
        if not render_valid.any():
            return zero, valid_ratio.detach()

        pred = direct_compact[render_valid].float()
        target = rendered_targets[render_valid].float()
        mode = str(getattr(self, "direct_point_render_consistency_mode", "cosine"))
        if mode == "mse":
            per_point = (pred - target).pow(2).mean(dim=-1)
        elif mode == "cosine":
            per_point = 1.0 - (
                F.normalize(pred, dim=-1) * F.normalize(target, dim=-1)
            ).sum(dim=-1)
        else:
            raise ValueError(
                "direct_point_render_consistency_mode must be one of: cosine, mse"
            )
        active_weights = sample_weights[render_valid] if sample_weights is not None else None
        return _weighted_vector_mean(per_point, active_weights), valid_ratio.detach()

    def _compute_siglip_alignment_loss(
        self,
        decoded: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            self.siglip_projection is None
            or self.siglip_alignment_weight <= 0
            or decoded is None
            or target is None
        ):
            return torch.tensor(0.0, device=self.device)
        if decoded.shape[-2:] != target.shape[-2:]:
            target = self._resize_map(target, decoded.shape[-2:])
        pred_siglip = self._project_siglip_features(decoded)
        with torch.no_grad():
            target_siglip = self._project_siglip_features(target)
        return self.siglip_alignment_weight * F.mse_loss(pred_siglip, target_siglip)

    def _compute_foundation_cache_loss(
        self,
        batch: Dict[str, torch.Tensor],
        decoded: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        zero = torch.tensor(0.0, device=self.device)
        stats: Dict[str, torch.Tensor] = {
            "loss": zero,
            "enabled": zero.detach(),
            "heads": zero.detach(),
            "missing": zero.detach(),
            "skipped": zero.detach(),
        }
        uses_direct_region_cache = (
            self.foundation_cache_region_consistency_weight > 0
            or self.foundation_cache_region_separation_weight > 0
            or self.foundation_cache_feature_boundary_weight > 0
        )
        uses_projected_cache = (
            self.foundation_cache_token_weight > 0
            or self.foundation_cache_mask_logit_weight > 0
            or self.foundation_cache_mask_boundary_weight > 0
        )
        if (
            self.foundation_cache_weight <= 0
            or not self.foundation_cache_root
            or decoded is None
            or (not uses_direct_region_cache and not uses_projected_cache)
            or (uses_projected_cache and not self.foundation_cache_projectors and not uses_direct_region_cache)
        ):
            return stats
        frame_idx = batch.get("frame_idx")
        if frame_idx is None:
            return stats

        frame_values = frame_idx.detach().cpu().reshape(-1).tolist()
        losses: list[torch.Tensor] = []
        head_count = 0
        skipped_count = 0
        missing_count = 0
        enabled_count = 0
        selected_heads = self.foundation_cache_heads or None
        for batch_idx, frame_value in enumerate(frame_values):
            cache_path = resolve_foundation_cache_path(self.foundation_cache_root, frame_value)
            if cache_path is None:
                missing_count += 1
                continue
            try:
                cache = load_foundation_cache(
                    cache_path,
                    require_official=self.foundation_cache_require_official,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                self._log(f"Skipping invalid foundation cache {cache_path}: {exc}")
                skipped_count += 1
                continue
            sample_loss, sample_stats = compute_foundation_cache_supervision_loss(
                decoded_features=decoded[batch_idx : batch_idx + 1].float(),
                cache=cache,
                projectors=self.foundation_cache_projectors,
                heads=selected_heads,
                mask_logit_weight=self.foundation_cache_mask_logit_weight,
                mask_boundary_weight=self.foundation_cache_mask_boundary_weight,
                token_weight=self.foundation_cache_token_weight,
                region_consistency_weight=self.foundation_cache_region_consistency_weight,
                region_separation_weight=self.foundation_cache_region_separation_weight,
                feature_boundary_weight=self.foundation_cache_feature_boundary_weight,
                region_score_threshold=self.foundation_cache_region_score_threshold,
                region_max_masks=self.foundation_cache_region_max_masks,
                region_separation_margin=self.foundation_cache_region_separation_margin,
            )
            head_count += int(sample_stats.get("heads", 0))
            skipped_count += int(sample_stats.get("skipped_heads", 0))
            if int(sample_stats.get("enabled", 0)) > 0:
                losses.append(sample_loss)
                enabled_count += 1

        if losses:
            stats["loss"] = self.foundation_cache_weight * torch.stack(losses).mean()
        stats["enabled"] = torch.tensor(float(enabled_count), device=self.device)
        stats["heads"] = torch.tensor(float(head_count), device=self.device)
        stats["missing"] = torch.tensor(float(missing_count), device=self.device)
        stats["skipped"] = torch.tensor(float(skipped_count), device=self.device)
        return stats

    def _compute_radio_adaptor_alignment_loss(
        self,
        decoded: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            self.radio_adaptor_alignment_weight <= 0
            or not self.radio_adaptor_alignment_adaptors
            or decoded is None
            or target is None
        ):
            device = decoded.device if decoded is not None else self.device
            return torch.tensor(0.0, device=device)
        if decoded.shape[-2:] != target.shape[-2:]:
            target = self._resize_map(target, decoded.shape[-2:])
        adaptor_loss, _ = compute_radio_adaptor_alignment_loss(
            decoded.float(),
            target.float(),
            self.radio_adaptor_alignment_adaptors,
        )
        return self.radio_adaptor_alignment_weight * adaptor_loss

    def _radio_adaptor_subset(self, names: list[str]) -> dict[str, nn.Module]:
        return {
            name: self.radio_adaptor_alignment_adaptors[name]
            for name in names
            if name in self.radio_adaptor_alignment_adaptors
        }

    def _compute_radio_adaptor_relation_loss(
        self,
        decoded: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        adaptors = self._radio_adaptor_subset(self.radio_adaptor_relation_names)
        if (
            self.radio_adaptor_relation_weight <= 0
            or not adaptors
            or decoded is None
            or target is None
        ):
            device = decoded.device if decoded is not None else self.device
            return torch.tensor(0.0, device=device)
        if decoded.shape[-2:] != target.shape[-2:]:
            target = self._resize_map(target, decoded.shape[-2:])
        relation_loss, _ = compute_radio_adaptor_relation_loss(
            decoded.float(),
            target.float(),
            adaptors,
            downsample=self.radio_adaptor_relation_downsample,
            max_tokens=self.radio_adaptor_relation_max_tokens,
            temperature=self.radio_adaptor_relation_temperature,
        )
        return self.radio_adaptor_relation_weight * relation_loss

    def _compute_radio_adaptor_local_affinity_loss(
        self,
        decoded: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        adaptors = self._radio_adaptor_subset(self.radio_adaptor_local_affinity_names)
        if (
            self.radio_adaptor_local_affinity_weight <= 0
            or not adaptors
            or decoded is None
            or target is None
        ):
            device = decoded.device if decoded is not None else self.device
            return torch.tensor(0.0, device=device)
        if decoded.shape[-2:] != target.shape[-2:]:
            target = self._resize_map(target, decoded.shape[-2:])
        local_loss, _ = compute_radio_adaptor_local_affinity_loss(
            decoded.float(),
            target.float(),
            adaptors,
            downsample=self.radio_adaptor_local_affinity_downsample,
            radius=self.radio_adaptor_local_affinity_radius,
        )
        return self.radio_adaptor_local_affinity_weight * local_loss

    def _compute_radio_adaptor_token_contrast_loss(
        self,
        decoded: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        adaptors = self._radio_adaptor_subset(self.radio_adaptor_token_contrast_names)
        if (
            self.radio_adaptor_token_contrast_weight <= 0
            or not adaptors
            or decoded is None
            or target is None
        ):
            device = decoded.device if decoded is not None else self.device
            return torch.tensor(0.0, device=device)
        if decoded.shape[-2:] != target.shape[-2:]:
            target = self._resize_map(target, decoded.shape[-2:])
        token_contrast_loss, _ = compute_radio_adaptor_token_contrast_loss(
            decoded.float(),
            target.float(),
            adaptors,
            downsample=self.radio_adaptor_token_contrast_downsample,
            max_tokens=self.radio_adaptor_token_contrast_max_tokens,
            temperature=self.radio_adaptor_token_contrast_temperature,
        )
        return self.radio_adaptor_token_contrast_weight * token_contrast_loss

    def _compute_radio_adaptor_peak_background_loss(
        self,
        decoded: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        adaptors = self._radio_adaptor_subset(self.radio_adaptor_peak_background_names)
        if (
            self.radio_adaptor_peak_background_weight <= 0
            or not adaptors
            or decoded is None
            or target is None
        ):
            device = decoded.device if decoded is not None else self.device
            return torch.tensor(0.0, device=device)
        if decoded.shape[-2:] != target.shape[-2:]:
            target = self._resize_map(target, decoded.shape[-2:])
        peak_loss, _ = compute_radio_adaptor_peak_background_loss(
            decoded.float(),
            target.float(),
            adaptors,
            downsample=self.radio_adaptor_peak_background_downsample,
            max_tokens=self.radio_adaptor_peak_background_max_tokens,
            num_anchors=self.radio_adaptor_peak_background_num_anchors,
            temperature=self.radio_adaptor_peak_background_temperature,
            anchor_strategy=self.radio_adaptor_peak_background_anchor_strategy,
        )
        return self.radio_adaptor_peak_background_weight * peak_loss

    def _compute_radio_adaptor_region_loss(
        self,
        decoded: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        adaptors = self._radio_adaptor_subset(self.radio_adaptor_region_names)
        if (
            self.radio_adaptor_region_weight <= 0
            or not adaptors
            or decoded is None
            or target is None
        ):
            device = decoded.device if decoded is not None else self.device
            return torch.tensor(0.0, device=device)
        if decoded.shape[-2:] != target.shape[-2:]:
            target = self._resize_map(target, decoded.shape[-2:])
        region_loss, _ = compute_radio_adaptor_region_loss(
            decoded.float(),
            target.float(),
            adaptors,
            downsample=self.radio_adaptor_region_downsample,
            max_tokens=self.radio_adaptor_region_max_tokens,
            num_anchors=self.radio_adaptor_region_num_anchors,
            temperature=self.radio_adaptor_region_temperature,
        )
        return self.radio_adaptor_region_weight * region_loss

    def _compute_radio_adaptor_mask_logit_loss(
        self,
        decoded: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        adaptors = self._radio_adaptor_subset(self.radio_adaptor_mask_logit_names)
        if (
            self.radio_adaptor_mask_logit_weight <= 0
            or not adaptors
            or decoded is None
            or target is None
        ):
            device = decoded.device if decoded is not None else self.device
            return torch.tensor(0.0, device=device)
        if decoded.shape[-2:] != target.shape[-2:]:
            target = self._resize_map(target, decoded.shape[-2:])
        mask_logit_loss, _ = compute_radio_adaptor_mask_logit_loss(
            decoded.float(),
            target.float(),
            adaptors,
            downsample=self.radio_adaptor_mask_logit_downsample,
            max_tokens=self.radio_adaptor_mask_logit_max_tokens,
            num_anchors=self.radio_adaptor_mask_logit_num_anchors,
            temperature=self.radio_adaptor_mask_logit_temperature,
        )
        return self.radio_adaptor_mask_logit_weight * mask_logit_loss

    def _compute_radio_adaptor_cross_view_loss(
        self,
        decoded: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        adaptors = self._radio_adaptor_subset(self.radio_adaptor_cross_view_names)
        if (
            self.radio_adaptor_cross_view_weight <= 0
            or not adaptors
            or decoded is None
            or target is None
            or decoded.shape[0] < 2
        ):
            device = decoded.device if decoded is not None else self.device
            return torch.tensor(0.0, device=device)
        if decoded.shape[-2:] != target.shape[-2:]:
            target = self._resize_map(target, decoded.shape[-2:])
        cross_view_loss, _ = compute_radio_adaptor_cross_view_loss(
            decoded.float(),
            target.float(),
            adaptors,
            downsample=self.radio_adaptor_cross_view_downsample,
            max_tokens=self.radio_adaptor_cross_view_max_tokens,
            temperature=self.radio_adaptor_cross_view_temperature,
            objective=self.radio_adaptor_cross_view_objective,
        )
        return self.radio_adaptor_cross_view_weight * cross_view_loss

    def _compute_radio_adaptor_cross_view_propagation_loss(
        self,
        decoded: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        adaptors = self._radio_adaptor_subset(
            self.radio_adaptor_cross_view_propagation_names
        )
        if (
            self.radio_adaptor_cross_view_propagation_weight <= 0
            or not adaptors
            or decoded is None
            or target is None
            or decoded.shape[0] < 2
        ):
            device = decoded.device if decoded is not None else self.device
            return torch.tensor(0.0, device=device)
        if decoded.shape[-2:] != target.shape[-2:]:
            target = self._resize_map(target, decoded.shape[-2:])
        propagation_loss, _ = compute_radio_adaptor_cross_view_propagation_loss(
            decoded.float(),
            target.float(),
            adaptors,
            downsample=self.radio_adaptor_cross_view_propagation_downsample,
            max_tokens=self.radio_adaptor_cross_view_propagation_max_tokens,
            num_anchors=self.radio_adaptor_cross_view_propagation_num_anchors,
            temperature=self.radio_adaptor_cross_view_propagation_temperature,
            anchor_strategy=self.radio_adaptor_cross_view_propagation_anchor_strategy,
        )
        return self.radio_adaptor_cross_view_propagation_weight * propagation_loss

    def _compute_radio_adaptor_cross_view_mask_propagation_loss(
        self,
        decoded: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        adaptors = self._radio_adaptor_subset(
            self.radio_adaptor_cross_view_mask_propagation_names
        )
        if (
            self.radio_adaptor_cross_view_mask_propagation_weight <= 0
            or not adaptors
            or decoded is None
            or target is None
            or decoded.shape[0] < 2
        ):
            device = decoded.device if decoded is not None else self.device
            return torch.tensor(0.0, device=device)
        if decoded.shape[-2:] != target.shape[-2:]:
            target = self._resize_map(target, decoded.shape[-2:])
        propagation_loss, _ = compute_radio_adaptor_cross_view_mask_propagation_loss(
            decoded.float(),
            target.float(),
            adaptors,
            downsample=self.radio_adaptor_cross_view_mask_propagation_downsample,
            max_tokens=self.radio_adaptor_cross_view_mask_propagation_max_tokens,
            num_anchors=self.radio_adaptor_cross_view_mask_propagation_num_anchors,
            temperature=self.radio_adaptor_cross_view_mask_propagation_temperature,
            anchor_strategy=self.radio_adaptor_cross_view_mask_propagation_anchor_strategy,
        )
        return self.radio_adaptor_cross_view_mask_propagation_weight * propagation_loss

    def _project_summary_head_features(self, features: torch.Tensor) -> torch.Tensor:
        """Project [B, C, H, W] features through frozen SigLIP2SummaryHead to text-aligned space."""
        assert self.siglip_summary_head is not None
        B, C, H, W = features.shape
        feat_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, C).float()
        projected = self.siglip_summary_head(feat_flat)
        projected = projected.permute(0, 2, 1).reshape(B, -1, H, W)
        return F.normalize(projected, dim=1)

    def _resolve_direct_point_text_embedding_path(
        self,
        raw_path: str,
        *,
        split: Optional[str] = None,
    ) -> Path:
        raw = Path(raw_path).expanduser()
        split = str(split or self.direct_point_text_split)

        def with_split_suffix(path: Path) -> Path:
            return path.with_name(f"{path.stem}_split{split}{path.suffix}")

        repo_root = Path(__file__).resolve().parents[2]
        candidates: list[Path] = []
        for base in (raw, repo_root / raw if not raw.is_absolute() else raw):
            candidates.append(with_split_suffix(base))
            candidates.append(base)
        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                return candidate
        return candidates[0]

    def _load_direct_point_text_embeddings(
        self,
        config: RadioGSConfig,
        *,
        split: Optional[str] = None,
    ) -> tuple[list[int], torch.Tensor]:
        split = str(split or self.direct_point_text_split)
        split_ids = OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]
        class_names = [NYU40_ID_TO_NAME[class_id] for class_id in split_ids]
        text_path = self._resolve_direct_point_text_embedding_path(
            getattr(
                config,
                "direct_point_text_embeddings",
                "checkpoints/siglip2_scannet_og_text_embeddings.pt",
            ),
            split=split,
        )
        if not text_path.exists():
            raise FileNotFoundError(
                f"Direct point text embeddings not found: {text_path}"
            )
        data = load_training_tensor_cache(
            text_path,
            map_location="cpu",
            purpose="direct point text embeddings",
        )
        queries = [str(query) for query in data.get("queries", [])]
        embeddings = data["embeddings"].float()
        if queries:
            bank = {
                query: F.normalize(embedding.float(), dim=0)
                for query, embedding in zip(queries, embeddings)
            }
            missing = [name for name in class_names if name not in bank]
            if missing:
                raise ValueError(
                    f"Direct point text bank {text_path} is missing classes: {missing}"
                )
            text_embeddings = torch.stack([bank[name] for name in class_names])
        elif embeddings.shape[0] == len(class_names):
            text_embeddings = F.normalize(embeddings, dim=-1)
        else:
            raise ValueError(
                f"Direct point text bank {text_path} has no query names and "
                f"{embeddings.shape[0]} embeddings; expected {len(class_names)}"
            )
        return list(split_ids), text_embeddings.to(self.device)

    def _load_direct_point_text_embedding_banks(
        self,
        config: RadioGSConfig,
        splits: list[str],
    ) -> list[tuple[str, list[int], torch.Tensor]]:
        banks: list[tuple[str, list[int], torch.Tensor]] = []
        for split in splits:
            split_ids, embeddings = self._load_direct_point_text_embeddings(
                config,
                split=split,
            )
            banks.append((split, split_ids, embeddings))
        return banks

    def _remap_direct_point_text_labels(
        self,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = torch.full_like(labels, -1, dtype=torch.long)
        valid = torch.zeros_like(labels, dtype=torch.bool)
        for out_idx, raw_id in enumerate(self.direct_point_text_split_ids):
            mask = labels == int(raw_id)
            target[mask] = int(out_idx)
            valid |= mask
        return target, valid

    def _direct_point_text_ce_weights(
        self,
        targets: torch.Tensor,
        num_classes: int,
    ) -> Optional[torch.Tensor]:
        mode = getattr(self, "direct_point_text_ce_weighting", "none")
        if mode == "none" or targets.numel() == 0:
            return None
        use_pool_counts = mode in {"inverse_pool", "sqrt_inverse_pool_capped"}
        if use_pool_counts and getattr(self, "direct_point_pool_labels", None) is not None:
            pool_targets, pool_valid = self._remap_direct_point_text_labels(
                self.direct_point_pool_labels.to(targets.device).long()
            )
            count_targets = pool_targets[pool_valid]
            if count_targets.numel() == 0:
                count_targets = targets
        else:
            count_targets = targets
        counts = torch.bincount(
            count_targets.long(),
            minlength=int(num_classes),
        ).to(device=targets.device, dtype=torch.float32)
        present = counts > 0
        weights = torch.zeros_like(counts)
        if present.any():
            total = counts[present].sum()
            weights[present] = total / (float(num_classes) * counts[present])
        if mode == "sqrt_inverse_pool_capped":
            weights[present] = weights[present].sqrt()
            min_weight = float(getattr(self, "direct_point_text_ce_min_weight", 0.5))
            max_weight = float(getattr(self, "direct_point_text_ce_max_weight", 3.0))
            weights[present] = weights[present].clamp(min=min_weight, max=max_weight)
        return weights

    def _compute_direct_point_text_ce(
        self,
        point_summary: torch.Tensor,
        point_labels: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = point_summary.sum() * 0.0
        if (
            point_labels is None
            or getattr(self, "direct_point_text_embeddings", None) is None
            or not getattr(self, "direct_point_text_split_ids", [])
        ):
            return zero, zero.detach(), zero.detach()
        targets, label_valid = self._remap_direct_point_text_labels(point_labels.long())
        valid_ratio = label_valid.float().mean()
        if not label_valid.any():
            return zero, valid_ratio.detach(), zero.detach()
        text_embeddings = self.direct_point_text_embeddings.to(point_summary.device)
        logits = (
            point_summary[label_valid].float() @ text_embeddings.T.float()
        ) / self.direct_point_text_temperature
        valid_targets = targets[label_valid]
        weights = self._direct_point_text_ce_weights(
            valid_targets,
            num_classes=int(text_embeddings.shape[0]),
        )
        loss = F.cross_entropy(logits, valid_targets, weight=weights)
        pred = logits.argmax(dim=-1)
        acc = (pred == valid_targets).float().mean()
        return loss, valid_ratio.detach(), acc.detach()

    def _compute_direct_point_text_distill_kl(
        self,
        point_summary: torch.Tensor,
        teacher_summary: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = point_summary.sum() * 0.0
        if (
            getattr(self, "direct_point_text_embeddings", None) is None
            or not getattr(self, "direct_point_text_split_ids", [])
        ):
            return zero, zero.detach(), zero.detach(), zero.detach()
        if point_summary.shape != teacher_summary.shape:
            raise ValueError(
                "point_summary and teacher_summary must have the same shape, got "
                f"{tuple(point_summary.shape)} vs {tuple(teacher_summary.shape)}"
            )

        text_embeddings = self.direct_point_text_embeddings.to(point_summary.device).float()
        temperature = max(
            1e-6,
            float(getattr(self, "direct_point_text_distill_temperature", 1.0)),
        )
        student_logits = point_summary.float() @ text_embeddings.T
        with torch.no_grad():
            teacher_logits = teacher_summary.float() @ text_embeddings.T
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
            teacher_conf = teacher_probs.max(dim=-1).values
            threshold = float(
                getattr(self, "direct_point_text_distill_confidence_threshold", 0.0)
            )
            valid = teacher_conf >= threshold
        valid_ratio = valid.float().mean()
        if not valid.any():
            return zero, valid_ratio.detach(), zero.detach(), zero.detach()

        log_probs = F.log_softmax(student_logits[valid] / temperature, dim=-1)
        per_point_kl = F.kl_div(
            log_probs,
            teacher_probs[valid],
            reduction="none",
        ).sum(dim=-1)
        loss = _weighted_vector_mean(
            per_point_kl,
            sample_weights[valid] if sample_weights is not None else None,
        )
        if temperature != 1.0:
            loss = loss * (temperature ** 2)
        with torch.no_grad():
            student_pred = student_logits[valid].argmax(dim=-1)
            teacher_pred = teacher_probs[valid].argmax(dim=-1)
            agreement = (student_pred == teacher_pred).float().mean()
            mean_conf = teacher_conf[valid].mean()
        return loss, valid_ratio.detach(), mean_conf.detach(), agreement.detach()

    def _compute_direct_point_text_contrast(
        self,
        point_summary: torch.Tensor,
        teacher_summary: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = point_summary.sum() * 0.0
        text_embeddings = getattr(self, "direct_point_text_embeddings", None)
        if text_embeddings is None or not getattr(self, "direct_point_text_split_ids", []):
            return zero, zero.detach(), zero.detach(), zero.detach()
        if point_summary.shape != teacher_summary.shape:
            raise ValueError(
                "point_summary and teacher_summary must have the same shape, got "
                f"{tuple(point_summary.shape)} vs {tuple(teacher_summary.shape)}"
            )

        text_embeddings = F.normalize(
            text_embeddings.to(point_summary.device).float(),
            dim=-1,
        )
        with torch.no_grad():
            teacher_logits = teacher_summary.float() @ text_embeddings.T
            if bool(getattr(self, "direct_point_text_contrast_center_logits", False)):
                teacher_logits = teacher_logits - teacher_logits.mean(dim=0, keepdim=True)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            teacher_conf, labels = teacher_probs.max(dim=-1)
            threshold = float(
                getattr(self, "direct_point_text_contrast_confidence_threshold", 0.0)
            )
            valid = teacher_conf >= threshold
        valid_ratio = valid.float().mean()
        if int(valid.sum().item()) < 3:
            return zero, valid_ratio.detach(), zero.detach(), zero.detach()

        labels = labels[valid]
        z = F.normalize(point_summary[valid].float(), dim=-1)
        active_sample_weights = (
            sample_weights[valid].to(device=z.device, dtype=z.dtype)
            if sample_weights is not None
            else None
        )
        max_points = max(
            0,
            int(getattr(self, "direct_point_text_contrast_max_points", 4096) or 0),
        )
        if max_points > 0 and z.shape[0] > max_points:
            if active_sample_weights is not None:
                probs = active_sample_weights.float().clamp_min(0.0)
                if float(probs.sum().item()) > 0:
                    keep = torch.multinomial(probs, max_points, replacement=False)
                else:
                    keep = torch.randperm(z.shape[0], device=z.device)[:max_points]
            else:
                keep = torch.randperm(z.shape[0], device=z.device)[:max_points]
            labels = labels[keep]
            z = z[keep]
            if active_sample_weights is not None:
                active_sample_weights = active_sample_weights[keep]
        logits = z @ z.T / max(
            1e-6,
            float(getattr(self, "direct_point_text_contrast_temperature", 0.1)),
        )
        self_mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(self_mask, -1e4)
        positive = labels[:, None].eq(labels[None, :]) & ~self_mask
        has_positive = positive.any(dim=1)
        if not has_positive.any():
            return zero, valid_ratio.detach(), teacher_conf[valid].mean().detach(), zero.detach()

        weighted_logits = logits
        pair_weighting = str(
            getattr(self, "direct_point_text_contrast_pair_weighting", "none") or "none"
        )
        if pair_weighting == "visibility" and active_sample_weights is not None:
            active_weights = active_sample_weights.to(device=logits.device, dtype=logits.dtype)
            pair_weights = torch.sqrt(
                active_weights[:, None].clamp_min(1e-6)
                * active_weights[None, :].clamp_min(1e-6)
            )
            weighted_logits = logits + pair_weights.log().clamp_min(-20.0)
            weighted_logits = weighted_logits.masked_fill(self_mask, -1e4)
        elif pair_weighting != "none":
            raise ValueError(
                "direct_point_text_contrast_pair_weighting must be one of: none, visibility"
            )

        log_denom = torch.logsumexp(weighted_logits[has_positive], dim=1)
        log_pos = torch.logsumexp(
            weighted_logits[has_positive].masked_fill(~positive[has_positive], -1e4),
            dim=1,
        )
        per_point = -(log_pos - log_denom)
        active_weights = (
            active_sample_weights[has_positive]
            if active_sample_weights is not None
            else None
        )
        loss = _weighted_vector_mean(per_point, active_weights)
        with torch.no_grad():
            nearest = logits.argmax(dim=-1)
            agreement = labels.eq(labels[nearest]).float().mean()
            mean_conf = teacher_conf[valid].mean()
        return loss, valid_ratio.detach(), mean_conf.detach(), agreement.detach()

    def _compute_direct_point_text_pseudo_ce(
        self,
        point_summary: torch.Tensor,
        teacher_summary: torch.Tensor,
        *,
        logit_scale: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
        center_logits: bool = False,
        split_ids: Optional[list[int]] = None,
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = point_summary.sum() * 0.0
        active_split_ids = (
            split_ids
            if split_ids is not None
            else getattr(self, "direct_point_text_split_ids", [])
        )
        active_text_embeddings = (
            text_embeddings
            if text_embeddings is not None
            else getattr(self, "direct_point_text_embeddings", None)
        )
        if (
            active_text_embeddings is None
            or not active_split_ids
        ):
            return zero, zero.detach(), zero.detach(), zero.detach()
        if point_summary.shape != teacher_summary.shape:
            raise ValueError(
                "point_summary and teacher_summary must have the same shape, got "
                f"{tuple(point_summary.shape)} vs {tuple(teacher_summary.shape)}"
            )

        text_embeddings = F.normalize(
            active_text_embeddings.to(point_summary.device).float(),
            dim=-1,
        )
        active_logit_scale = float(
            getattr(self, "direct_point_adapter_text_pseudo_ce_logit_scale", 1.0)
            if logit_scale is None
            else logit_scale
        )
        student_logits = (point_summary.float() @ text_embeddings.T) * active_logit_scale
        with torch.no_grad():
            teacher_logits = (teacher_summary.float() @ text_embeddings.T) * active_logit_scale
            teacher_bias = (
                teacher_logits.mean(dim=0, keepdim=True)
                if bool(center_logits)
                else None
            )
            if teacher_bias is not None:
                teacher_logits = teacher_logits - teacher_bias
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            teacher_conf, teacher_targets = teacher_probs.max(dim=-1)
            active_threshold = float(
                getattr(
                    self,
                    "direct_point_adapter_text_pseudo_ce_confidence_threshold",
                    0.0,
                )
                if confidence_threshold is None
                else confidence_threshold
            )
            valid = teacher_conf >= float(
                active_threshold
            )
        if bool(center_logits) and teacher_bias is not None:
            student_logits = student_logits - teacher_bias.to(
                device=student_logits.device,
                dtype=student_logits.dtype,
            )
        valid_ratio = valid.float().mean()
        if not valid.any():
            return zero, valid_ratio.detach(), zero.detach(), zero.detach()

        loss = F.cross_entropy(student_logits[valid], teacher_targets[valid])
        with torch.no_grad():
            student_pred = student_logits[valid].argmax(dim=-1)
            agreement = (student_pred == teacher_targets[valid]).float().mean()
            mean_conf = teacher_conf[valid].mean()
        return loss, valid_ratio.detach(), mean_conf.detach(), agreement.detach()

    def _compute_multi_split_direct_point_text_pseudo_ce(
        self,
        point_summary: torch.Tensor,
        teacher_summary: torch.Tensor,
        *,
        logit_scale: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
        center_logits: bool = False,
        banks: Optional[list[tuple[str, list[int], torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = point_summary.sum() * 0.0
        active_banks = (
            banks
            if banks is not None
            else getattr(self, "direct_point_text_pseudo_ce_banks", None)
        )
        if not active_banks:
            default_embeddings = getattr(self, "direct_point_text_embeddings", None)
            default_split_ids = getattr(self, "direct_point_text_split_ids", [])
            if default_embeddings is not None and default_split_ids:
                active_banks = [
                    (
                        str(getattr(self, "direct_point_text_split", "default")),
                        default_split_ids,
                        default_embeddings,
                    )
                ]
            else:
                active_banks = []
        outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for _split, split_ids, text_embeddings in active_banks:
            outputs.append(
                self._compute_direct_point_text_pseudo_ce(
                    point_summary,
                    teacher_summary,
                    logit_scale=logit_scale,
                    confidence_threshold=confidence_threshold,
                    center_logits=center_logits,
                    split_ids=split_ids,
                    text_embeddings=text_embeddings,
                )
            )
        if not outputs:
            return zero, zero.detach(), zero.detach(), zero.detach()

        averaged = []
        for metric_idx in range(4):
            averaged.append(torch.stack([out[metric_idx] for out in outputs]).mean())
        return tuple(averaged)  # type: ignore[return-value]

    def _compute_summary_alignment_loss(
        self,
        decoded: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Cosine distance loss in the text-aligned SigLIP2 summary head space.

        Uses 1 - cos_sim (averaged over spatial locations) instead of MSE
        to avoid the loss being diluted by the high dimensionality (1536-d).
        """
        if (
            self.siglip_summary_head is None
            or self.siglip_summary_alignment_weight <= 0
            or decoded is None
            or target is None
        ):
            return torch.tensor(0.0, device=self.device)
        if decoded.shape[-2:] != target.shape[-2:]:
            target = self._resize_map(target, decoded.shape[-2:])
        pred_summary = self._project_summary_head_features(decoded)  # [B, C, H, W] L2-normed
        with torch.no_grad():
            target_summary = self._project_summary_head_features(target)
        # Cosine similarity per spatial location, then mean cosine distance
        cos_sim = (pred_summary * target_summary).sum(dim=1).mean()  # dot product of unit vecs
        return self.siglip_summary_alignment_weight * (1.0 - cos_sim)

    def _compute_text_heatmap_distill_loss(
        self,
        decoded: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            self.text_heatmap_distill_weight <= 0
            or self.siglip_summary_head is None
            or self.text_heatmap_distill_embeddings is None
            or decoded is None
            or target is None
        ):
            return torch.tensor(0.0, device=self.device)
        if decoded.shape[-2:] != target.shape[-2:]:
            target = self._resize_map(target, decoded.shape[-2:])
        pred_summary = self._project_summary_head_features(decoded)
        with torch.no_grad():
            target_summary = self._project_summary_head_features(target)
        heatmap_loss, _ = compute_text_heatmap_distill_loss(
            pred_summary,
            target_summary,
            self.text_heatmap_distill_embeddings,
            downsample=self.text_heatmap_distill_downsample,
            temperature=self.text_heatmap_distill_temperature,
            mode=self.text_heatmap_distill_mode,
        )
        return self.text_heatmap_distill_weight * heatmap_loss

    def _load_grounding_text_embeddings(
        self,
        config: RadioGSConfig,
    ) -> Tuple[List[str], List[int], torch.Tensor]:
        text_path = resolve_siglip_text_embeddings_path(
            getattr(
                config,
                "grounding_text_embeddings",
                DEFAULT_SIGLIP2_TEXT_EMBEDDINGS,
            )
        )
        if not text_path.exists():
            raise FileNotFoundError(f"Grounding text embeddings not found: {text_path}")
        data = load_training_tensor_cache(
            text_path,
            map_location="cpu",
            purpose="grounding text embeddings",
        )
        bank = {
            query: F.normalize(embedding.float(), dim=0)
            for query, embedding in zip(data["queries"], data["embeddings"])
        }
        selected = [
            (query, class_id)
            for query, class_id in sorted(GROUNDING_QUERIES.items(), key=lambda x: x[1])
            if query in bank
        ]
        if not selected:
            raise ValueError(
                f"No Replica grounding queries from {list(GROUNDING_QUERIES)} found in {text_path}"
            )
        query_names = [query for query, _ in selected]
        query_class_ids = [class_id for _, class_id in selected]
        text_embeddings = torch.stack([bank[query] for query in query_names]).to(self.device)
        return query_names, query_class_ids, text_embeddings

    def _load_text_heatmap_distill_embeddings(
        self,
        config: RadioGSConfig,
    ) -> torch.Tensor:
        raw_path = self.text_heatmap_distill_embeddings_path or getattr(
            config,
            "grounding_text_embeddings",
            DEFAULT_SIGLIP2_TEXT_EMBEDDINGS,
        )
        text_path = resolve_siglip_text_embeddings_path(raw_path)
        if not text_path.exists():
            raise FileNotFoundError(
                f"Text heatmap distillation embeddings not found: {text_path}"
            )
        data = load_training_tensor_cache(
            text_path,
            map_location="cpu",
            purpose="text heatmap distillation embeddings",
        )
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise ValueError(f"Missing 'embeddings' tensor in {text_path}")
        embeddings = F.normalize(embeddings.float(), dim=1)
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected text heatmap embeddings [Q, C], got {tuple(embeddings.shape)}"
            )
        return embeddings.to(self.device)

    def _load_direct_point_query_logit_distill_embeddings(
        self,
        config: RadioGSConfig,
        raw_path: str = "",
        purpose: str = "direct point query-logit distillation embeddings",
    ) -> torch.Tensor:
        raw_path = raw_path or getattr(self, "direct_point_query_logit_distill_embeddings_path", "") or getattr(
            config,
            "direct_point_query_logit_distill_embeddings",
            "",
        )
        if not raw_path:
            raw_path = getattr(config, "text_heatmap_distill_embeddings", "") or DEFAULT_SIGLIP2_TEXT_EMBEDDINGS
        text_path = resolve_siglip_text_embeddings_path(raw_path)
        if not text_path.exists():
            raise FileNotFoundError(
                f"{purpose} not found: {text_path}"
            )
        data = load_training_tensor_cache(
            text_path,
            map_location="cpu",
            purpose=purpose,
        )
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise ValueError(f"Missing 'embeddings' tensor in {text_path}")
        embeddings = F.normalize(embeddings.float(), dim=1)
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected query-logit embeddings [Q, C], got {tuple(embeddings.shape)}"
            )
        return embeddings.to(self.device)

    def _compute_grounding_query_loss(
        self,
        batch: Dict[str, torch.Tensor],
        decoded: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        zero = torch.tensor(0.0, device=self.device)
        stats = {
            "loss": zero,
            "accuracy": zero.detach(),
            "valid_ratio": zero.detach(),
        }
        if (
            self.grounding_query_loss_fn is None
            or self.grounding_query_loss_weight <= 0
            or self.grounding_text_embeddings is None
            or decoded is None
        ):
            return stats
        gt_sem = batch.get("semantics")
        if gt_sem is None:
            return stats
        gt_sem = gt_sem.to(self.device).long()
        target_size = decoded.shape[-2:]
        downsample = max(1, int(getattr(self.cfg, "grounding_query_loss_downsample", 1)))
        decoded_for_loss = decoded
        if downsample > 1:
            target_size = (
                max(1, target_size[0] // downsample),
                max(1, target_size[1] // downsample),
            )
            decoded_for_loss = F.interpolate(
                decoded.float(),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        if gt_sem.shape[-2:] != target_size:
            gt_sem = self._resize_map(
                gt_sem.unsqueeze(1).float(),
                target_size,
                is_mask=True,
            ).squeeze(1).long()
        if self.siglip_summary_head is None:
            raise RuntimeError(
                "grounding_query_loss requires SigLIP2SummaryHead; "
                "check siglip_summary_head_weights"
            )
        pred_siglip = self._project_summary_head_features(decoded_for_loss)
        return self.grounding_query_loss_fn(
            pred_siglip,
            self.grounding_text_embeddings,
            gt_sem,
            self.grounding_query_class_ids,
        )
