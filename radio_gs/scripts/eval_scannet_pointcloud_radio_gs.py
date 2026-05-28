#!/usr/bin/env python3
"""Evaluate direct RADIO-GS point-cloud understanding on ScanNet.

This evaluator does not render features back to images.  It queries the
HybridFeatureGaussian field directly at label-PLY vertex coordinates, decodes
compact RADIO-GS features to 1280d, projects them into SigLIP2 text space, and
classifies each point against OpenGaussian's NYU40 class splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from radio_gs.artifact_paths import (
    DEFAULT_SIGLIP2_PROJECTION_WEIGHTS,
    resolve_siglip_projection_path,
)
from radio_gs.config import load_config
from radio_gs.models.hcd_codec import HCDCodec
from radio_gs.models.hybrid_gaussian import HybridFeatureGaussian
from radio_gs.models.point_summary_adapter import CompactToSummaryAdapter
from radio_gs.models.proposal_memory import (
    build_voxel_proposal_labels,
    propagate_logits_with_proposals,
)
from radio_gs.models.siglip_projection import SigLIP2FeatureProjection, SigLIP2SummaryHead
from radio_gs.utils.checkpoint_io import load_trusted_checkpoint
from radio_gs.scannet_constants import (
    NYU40_ID_TO_NAME,
    OPENGAUSSIAN_NYU40_CLASS_SPLITS,
    compute_split_metrics,
)
from radio_gs.scripts.eval_lerf_grounding import (
    load_or_generate_prompt_ensemble_embeddings,
    parse_prompt_templates,
)


DEFAULT_PREPARED_ROOT = Path("dataset/scannet_og")
FEATURE_RGB_PROJECTION_SEED = 20260429
QUERY_MODES = ("knn", "nearest", "gaussian_index")
COMPACT_FEATURE_KEYS = ("features", "fused", "semantic", "geometry")
LOGIT_CALIBRATION_MODES = ("none", "scene_mean")
LOGIT_SMOOTHING_MODES = ("none", "spatial_knn")
PROPOSAL_SMOOTHING_MODES = ("none", "voxel")
CLASS_ALIAS_MODES = ("none", "scannet")
OPACITY_FILTER_MODES = ("auto", "label_index", "query_top1", "query_weighted", "off")
GAUSSIAN_INDEX_POSITION_MODES = ("gaussian_center", "label_point")
SCANNET_CLASS_TEXT_ALIASES = {
    "wall": ["wall", "indoor wall", "wall surface"],
    "floor": ["floor", "indoor floor", "floor surface"],
    "cabinet": ["cabinet", "kitchen cabinet", "storage cabinet"],
    "bed": ["bed", "bed frame"],
    "chair": ["chair", "desk chair", "dining chair"],
    "sofa": ["sofa", "couch"],
    "table": ["table", "dining table", "coffee table"],
    "door": ["door", "indoor door", "doorway"],
    "window": ["window", "indoor window"],
    "bookshelf": ["bookshelf", "book shelf", "shelving"],
    "picture": ["picture", "wall picture", "framed picture"],
    "counter": ["counter", "kitchen counter", "countertop"],
    "desk": ["desk", "office desk", "table desk"],
    "curtain": ["curtain", "window curtain"],
    "refrigerator": ["refrigerator", "fridge"],
    "shower curtain": ["shower curtain", "bathroom curtain"],
    "toilet": ["toilet", "toilet bowl"],
    "sink": ["sink", "bathroom sink", "kitchen sink"],
    "bathtub": ["bathtub", "bath tub"],
}


def _format_scene_path(pattern: Optional[str], scene: str) -> Optional[str]:
    if not pattern:
        return None
    return str(pattern).format(scene=scene)


def _parse_splits(raw: str) -> list[str]:
    splits = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    for split in splits:
        if split not in OPENGAUSSIAN_NYU40_CLASS_SPLITS:
            raise ValueError(
                f"Unsupported split '{split}'. Expected one of "
                f"{sorted(OPENGAUSSIAN_NYU40_CLASS_SPLITS)}"
            )
    return splits


def _resolve_class_aliases(class_names: list[str], mode: str) -> list[list[str]]:
    if mode not in CLASS_ALIAS_MODES:
        raise ValueError(f"class_aliases must be one of: {', '.join(CLASS_ALIAS_MODES)}")
    if mode == "none":
        return [[str(name)] for name in class_names]
    alias_groups: list[list[str]] = []
    for name in class_names:
        clean = str(name)
        aliases = SCANNET_CLASS_TEXT_ALIASES.get(clean, [clean])
        deduped = list(dict.fromkeys([clean, *[str(alias) for alias in aliases]]))
        alias_groups.append(deduped)
    return alias_groups


def _load_or_generate_class_text_embeddings(
    class_names: list[str],
    device: torch.device,
    *,
    cache_path: Optional[str],
    prompt_templates: list[str],
    class_aliases: str,
) -> torch.Tensor:
    alias_groups = _resolve_class_aliases(class_names, class_aliases)
    if class_aliases == "none":
        return load_or_generate_prompt_ensemble_embeddings(
            class_names,
            device,
            cache_path=cache_path,
            prompt_templates=prompt_templates,
        )

    def _load_cache() -> Optional[torch.Tensor]:
        if not cache_path or not Path(cache_path).exists():
            return None
        data = torch.load(cache_path, map_location="cpu")
        if (
            [str(q) for q in data.get("class_names", [])] == list(class_names)
            and [[str(a) for a in group] for group in data.get("alias_groups", [])]
            == alias_groups
            and [str(t) for t in data.get("prompt_templates", [])] == list(prompt_templates)
            and str(data.get("class_aliases", "none")) == class_aliases
        ):
            return F.normalize(data["embeddings"].float(), dim=-1).to(device)
        return None

    cached = _load_cache()
    if cached is not None:
        return cached

    flat_aliases = [alias for group in alias_groups for alias in group]
    alias_emb = load_or_generate_prompt_ensemble_embeddings(
        flat_aliases,
        device,
        cache_path=None,
        prompt_templates=prompt_templates,
    )
    class_emb_parts: list[torch.Tensor] = []
    offset = 0
    for group in alias_groups:
        width = len(group)
        class_emb_parts.append(alias_emb[offset : offset + width].mean(dim=0))
        offset += width
    embeddings = F.normalize(torch.stack(class_emb_parts, dim=0).float(), dim=-1)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "class_names": class_names,
                "alias_groups": alias_groups,
                "prompt_templates": prompt_templates,
                "class_aliases": class_aliases,
                "embeddings": embeddings.detach().cpu(),
            },
            cache_path,
        )
    return embeddings


def _read_label_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )
    if "label" not in vertex.data.dtype.names:
        raise ValueError(f"Label PLY has no 'label' property: {path}")
    labels = np.asarray(vertex["label"], dtype=np.int32)
    return xyz, labels


def _subsample_points(
    xyz: np.ndarray,
    labels: np.ndarray,
    max_points: Optional[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if max_points is None or max_points <= 0 or xyz.shape[0] <= max_points:
        return xyz, labels, None
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(xyz.shape[0], size=max_points, replace=False))
    return xyz[indices], labels[indices], indices


def _apply_opacity_label_filter(
    labels: np.ndarray,
    opacities: torch.Tensor | np.ndarray,
    *,
    threshold: float,
    scene: str,
) -> tuple[np.ndarray, dict]:
    """Match OpenGaussian eval by marking low-opacity GT labels invalid."""
    labels = np.asarray(labels, dtype=np.int32)
    opacity_np = (
        opacities.detach().float().cpu().numpy()
        if isinstance(opacities, torch.Tensor)
        else np.asarray(opacities, dtype=np.float32)
    ).reshape(-1)
    if opacity_np.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Opacity/label count mismatch for {scene}: "
            f"opacities={opacity_np.shape[0]} labels={labels.shape[0]}"
        )
    filtered = labels.copy()
    low_opacity = opacity_np < float(threshold)
    filtered[low_opacity] = 0
    return filtered, {
        "enabled": True,
        "mode": "label_index",
        "threshold": float(threshold),
        "num_filtered": int(low_opacity.sum()),
        "num_points": int(labels.shape[0]),
    }


def _resolve_opacity_filter_mode(
    mode: str,
    *,
    query_mode: str,
    label_count: int,
    gaussian_count: int,
) -> str:
    if mode not in OPACITY_FILTER_MODES:
        raise ValueError(f"opacity_filter_mode must be one of: {', '.join(OPACITY_FILTER_MODES)}")
    if mode != "auto":
        return mode
    if query_mode == "gaussian_index" and int(label_count) == int(gaussian_count):
        return "label_index"
    return "query_top1"


def _apply_query_opacity_label_filter(
    labels: np.ndarray,
    query_aux: dict[str, torch.Tensor],
    opacities: torch.Tensor | np.ndarray,
    *,
    threshold: float,
    mode: str,
) -> tuple[np.ndarray, dict]:
    if mode not in {"query_top1", "query_weighted"}:
        raise ValueError("mode must be one of: query_top1, query_weighted")
    labels = np.asarray(labels, dtype=np.int32)
    opacity = (
        opacities.detach().float()
        if isinstance(opacities, torch.Tensor)
        else torch.as_tensor(opacities, dtype=torch.float32)
    ).reshape(-1)
    indices = query_aux.get("gaussian_indices")
    if indices is None:
        raise KeyError("query_aux is missing gaussian_indices")
    indices = indices.detach().long() if isinstance(indices, torch.Tensor) else torch.as_tensor(indices, dtype=torch.long)
    if indices.dim() == 1:
        indices = indices.unsqueeze(1)
    if indices.shape[0] != labels.shape[0]:
        raise ValueError(
            "query opacity filter point count mismatch: "
            f"indices={indices.shape[0]} labels={labels.shape[0]}"
        )
    if int(indices.min()) < 0 or int(indices.max()) >= int(opacity.shape[0]):
        raise IndexError("query opacity filter gaussian index out of range")
    neighbor_opacity = opacity.to(indices.device)[indices]
    if mode == "query_top1":
        weights = query_aux.get("weights")
        if weights is None:
            selected_opacity = neighbor_opacity[:, 0]
        else:
            weights = (
                weights.detach().float()
                if isinstance(weights, torch.Tensor)
                else torch.as_tensor(weights, dtype=torch.float32)
            ).to(indices.device)
            if weights.dim() == 1:
                weights = weights.unsqueeze(1)
            top_pos = weights.argmax(dim=1, keepdim=True)
            selected_opacity = neighbor_opacity.gather(1, top_pos).squeeze(1)
    else:
        weights = query_aux.get("weights")
        if weights is None:
            weights = torch.full_like(neighbor_opacity, 1.0 / max(neighbor_opacity.shape[1], 1))
        else:
            weights = (
                weights.detach().float()
                if isinstance(weights, torch.Tensor)
                else torch.as_tensor(weights, dtype=torch.float32)
            ).to(indices.device)
            if weights.dim() == 1:
                weights = weights.unsqueeze(1)
        weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        selected_opacity = (neighbor_opacity * (weights / weight_sum)).sum(dim=1)
    low_opacity = selected_opacity.detach().cpu().numpy() < float(threshold)
    filtered = labels.copy()
    filtered[low_opacity] = 0
    return filtered, {
        "enabled": True,
        "mode": mode,
        "threshold": float(threshold),
        "num_filtered": int(low_opacity.sum()),
        "num_points": int(labels.shape[0]),
    }


def _build_hybrid_model(config, checkpoint_path: str, device: torch.device):
    if getattr(config, "architecture", "hybrid") != "hybrid":
        raise ValueError("Direct point query currently requires architecture: hybrid")
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
        semantic_adaptor_mode=getattr(config, "hybrid_semantic_adaptor_mode", "confidence"),
        semantic_adaptor_hidden_dim=getattr(config, "hybrid_semantic_adaptor_hidden_dim", 64),
        semantic_adaptor_use_geometry_guidance=getattr(
            config, "hybrid_semantic_adaptor_use_geometry_guidance", True
        ),
        semantic_adaptor_use_depth_guidance=getattr(
            config, "hybrid_semantic_adaptor_use_depth_guidance", False
        ),
        semantic_adaptor_residual=getattr(config, "hybrid_semantic_adaptor_residual", True),
    )
    ply_path = getattr(config, "ply_path", "")
    if not ply_path:
        raise ValueError("Config must define ply_path for direct point query")
    model.load_from_ply(ply_path)
    model = model.to(device).eval()

    codec = HCDCodec(
        input_dim=getattr(config, "radio_feature_dim", 1280),
        bottleneck_dim=getattr(config, "bottleneck_dim", getattr(config, "hybrid_output_dim", 128)),
        dual_stream=getattr(config, "dual_stream", True),
        symmetric_decoder=getattr(config, "symmetric_decoder", False),
    ).to(device).eval()

    ckpt = load_trusted_checkpoint(checkpoint_path, map_location=device)
    _load_state_or_raise(
        model,
        ckpt["model_state_dict"],
        "model",
        required_prefixes=(
            "_latent",
            "hash_field.",
            "fine_decoder.",
            "coarse_decoder.",
            "fusion_head.",
        ),
    )
    _load_state_or_raise(codec, ckpt["codec_state_dict"], "codec")
    return model, codec


def _load_state_or_raise(
    module: torch.nn.Module,
    state_dict: dict,
    name: str,
    required_prefixes: tuple[str, ...] | None = None,
) -> None:
    result = module.load_state_dict(state_dict, strict=False)
    if required_prefixes is None:
        missing_required = list(result.missing_keys)
    else:
        missing_required = [
            key
            for key in result.missing_keys
            if any(key == prefix or key.startswith(prefix) for prefix in required_prefixes)
        ]
    if missing_required or result.unexpected_keys:
        raise RuntimeError(
            f"Incompatible {name} checkpoint state: "
            f"missing_required={missing_required}, "
            f"unexpected={list(result.unexpected_keys)}"
        )


def _decode_points_1280(
    model: HybridFeatureGaussian,
    codec: HCDCodec,
    points_xyz: torch.Tensor,
    k: int,
    *,
    candidate_k: Optional[int] = None,
    return_aux: bool = False,
    compact_feature_key: str = "features",
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if compact_feature_key not in COMPACT_FEATURE_KEYS:
        raise ValueError(
            f"compact_feature_key must be one of: {', '.join(COMPACT_FEATURE_KEYS)}"
        )
    query_kwargs = {"k": k, "return_aux": return_aux}
    if candidate_k is not None:
        query_kwargs["candidate_k"] = candidate_k
    compact_result = model.query_compact_points(points_xyz, **query_kwargs)
    if return_aux:
        assert isinstance(compact_result, dict)
        if compact_feature_key not in compact_result:
            raise KeyError(
                f"Requested compact feature branch '{compact_feature_key}', "
                f"available={sorted(compact_result.keys())}"
            )
        compact = compact_result[compact_feature_key]
    else:
        assert isinstance(compact_result, torch.Tensor)
        compact = compact_result
    if hasattr(codec, "decode_points"):
        decoded_points = codec.decode_points(compact.float())
    else:
        compact_map = compact.T.reshape(1, compact.shape[1], compact.shape[0], 1)
        decoded = codec.decode(compact_map.float())
        decoded_points = decoded.squeeze(0).squeeze(-1).T.contiguous()
    if return_aux:
        assert isinstance(compact_result, dict)
        return decoded_points, compact_result
    return decoded_points


def _decode_gaussian_indices_1280(
    model: HybridFeatureGaussian,
    codec: HCDCodec,
    gaussian_indices: torch.Tensor,
    *,
    points_xyz: Optional[torch.Tensor] = None,
    return_aux: bool = False,
    compact_feature_key: str = "features",
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if compact_feature_key not in COMPACT_FEATURE_KEYS:
        raise ValueError(
            f"compact_feature_key must be one of: {', '.join(COMPACT_FEATURE_KEYS)}"
        )
    compact_result = model.query_gaussian_points(
        gaussian_indices,
        points_xyz=points_xyz,
        return_aux=return_aux,
    )
    if return_aux:
        assert isinstance(compact_result, dict)
        if compact_feature_key not in compact_result:
            raise KeyError(
                f"Requested compact feature branch '{compact_feature_key}', "
                f"available={sorted(compact_result.keys())}"
            )
        compact = compact_result[compact_feature_key]
    else:
        assert isinstance(compact_result, torch.Tensor)
        compact = compact_result
    if hasattr(codec, "decode_points"):
        decoded_points = codec.decode_points(compact.float())
    else:
        compact_map = compact.T.reshape(1, compact.shape[1], compact.shape[0], 1)
        decoded = codec.decode(compact_map.float())
        decoded_points = decoded.squeeze(0).squeeze(-1).T.contiguous()
    if return_aux:
        assert isinstance(compact_result, dict)
        return decoded_points, compact_result
    return decoded_points


def _project_compact_with_summary_adapter(
    compact: torch.Tensor,
    adapter: torch.nn.Module,
) -> torch.Tensor:
    """Project compact point features directly to normalized text-aligned space."""
    return F.normalize(adapter(compact.float()).float(), dim=-1)


def _build_point_summary_adapter(config, checkpoint_path: str, device: torch.device):
    adapter = CompactToSummaryAdapter(
        input_dim=getattr(config, "bottleneck_dim", getattr(config, "hybrid_output_dim", 128)),
        output_dim=1536,
        hidden_dim=getattr(config, "point_summary_adapter_hidden_dim", 512),
        num_layers=getattr(config, "point_summary_adapter_num_layers", 2),
        dropout=getattr(config, "point_summary_adapter_dropout", 0.0),
    ).to(device)
    ckpt = load_trusted_checkpoint(checkpoint_path, map_location=device)
    state = ckpt.get("point_summary_adapter_state_dict")
    if state is None:
        raise KeyError(
            "Checkpoint has no point_summary_adapter_state_dict; train with "
            "direct_point_summary_adapter_weight > 0 first"
        )
    adapter.load_state_dict(state, strict=True)
    return adapter.eval()


def _load_projection(args, device: torch.device):
    if args.use_summary_head:
        head_path = Path(args.summary_head_weights)
        if head_path.exists():
            proj = SigLIP2SummaryHead.from_extracted_weights(str(head_path))
        else:
            proj = SigLIP2SummaryHead.from_radio_checkpoint(args.radio_checkpoint)
    else:
        proj_path = resolve_siglip_projection_path(args.projection_weights)
        if proj_path.exists():
            proj = SigLIP2FeatureProjection()
            proj.load_state_dict(torch.load(str(proj_path), map_location="cpu"))
        else:
            raise FileNotFoundError(f"Projection weights not found: {proj_path}")
    return proj.to(device).eval()


def _project_points(
    features_1280: torch.Tensor,
    projection: torch.nn.Module,
) -> torch.Tensor:
    siglip = projection(features_1280.unsqueeze(0))
    return F.normalize(siglip.squeeze(0).float(), dim=-1)


def _decode_compact_1280(codec: HCDCodec, compact: torch.Tensor) -> torch.Tensor:
    if hasattr(codec, "decode_points"):
        return codec.decode_points(compact.float())
    compact_map = compact.T.reshape(1, compact.shape[1], compact.shape[0], 1)
    decoded = codec.decode(compact_map.float())
    return decoded.squeeze(0).squeeze(-1).T.contiguous()


def _blend_summary_features(
    base_summary: torch.Tensor,
    adapter_summary: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Blend base decoded summary features with adapter summary features."""
    blend_alpha = float(alpha)
    if not 0.0 <= blend_alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    base = F.normalize(base_summary.float(), dim=-1)
    adapter = F.normalize(adapter_summary.float(), dim=-1)
    return F.normalize((1.0 - blend_alpha) * base + blend_alpha * adapter, dim=-1)


def _empty_split_diagnostics(split_ids: Iterable[int]) -> dict:
    split_ids = [int(class_id) for class_id in split_ids]
    return {
        "num_points": 0,
        "top1_margin_sum": 0.0,
        "top1_confidence_sum": 0.0,
        "logit_sum": np.zeros(len(split_ids), dtype=np.float64),
        "logits": [],
        "labels": [],
        "pred_ids": [],
    }


def _update_split_diagnostics(
    diagnostics: dict,
    logits: torch.Tensor,
    labels: np.ndarray,
    pred_ids: np.ndarray,
    *,
    save_logits_npz: bool,
) -> None:
    logits_cpu = logits.detach().float().cpu()
    num_points = int(logits_cpu.shape[0])
    diagnostics["num_points"] += num_points
    if num_points == 0:
        return

    if logits_cpu.shape[1] == 1:
        margin = torch.zeros_like(logits_cpu[:, 0])
    else:
        top2 = torch.topk(logits_cpu, k=2, dim=-1).values
        top1 = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1]
    confidence = F.softmax(logits_cpu, dim=-1).max(dim=-1).values

    diagnostics["top1_margin_sum"] += float(margin.sum().item())
    diagnostics["top1_confidence_sum"] += float(confidence.sum().item())
    diagnostics["logit_sum"] += logits_cpu.sum(dim=0).numpy().astype(np.float64)

    if save_logits_npz:
        diagnostics["logits"].append(logits_cpu.numpy().astype(np.float32, copy=False))
        diagnostics["labels"].append(labels.astype(np.int32, copy=False))
        diagnostics["pred_ids"].append(pred_ids.astype(np.int32, copy=False))


def _finalize_split_diagnostics(diagnostics: dict, split_ids: Iterable[int]) -> dict:
    split_ids = [int(class_id) for class_id in split_ids]
    num_points = int(diagnostics["num_points"])
    if num_points == 0:
        mean_margin = 0.0
        mean_confidence = 0.0
        mean_logits = np.zeros(len(split_ids), dtype=np.float64)
    else:
        mean_margin = float(diagnostics["top1_margin_sum"] / num_points)
        mean_confidence = float(diagnostics["top1_confidence_sum"] / num_points)
        mean_logits = diagnostics["logit_sum"] / num_points
    return {
        "mean_top1_margin": mean_margin,
        "mean_top1_confidence_softmax": mean_confidence,
        "class_mean_logits": {
            NYU40_ID_TO_NAME.get(class_id, f"class_{class_id}"): float(mean_logits[idx])
            for idx, class_id in enumerate(split_ids)
        },
    }


def _apply_logit_calibration(
    logits: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Apply class-wise logit bias without changing the input tensor in-place."""
    if bias is None:
        return logits
    bias = bias.to(device=logits.device, dtype=logits.dtype).reshape(1, -1)
    if bias.shape[1] != logits.shape[1]:
        raise ValueError(
            f"Logit calibration bias has {bias.shape[1]} classes, "
            f"but logits have {logits.shape[1]}"
        )
    return logits - float(alpha) * bias


def _build_spatial_knn_graph(
    xyz: np.ndarray | torch.Tensor,
    *,
    k: int,
    sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    xyz_np = xyz.detach().float().cpu().numpy() if isinstance(xyz, torch.Tensor) else np.asarray(xyz)
    xyz_np = np.asarray(xyz_np, dtype=np.float32)
    if xyz_np.ndim != 2 or xyz_np.shape[1] != 3:
        raise ValueError(f"Expected xyz [N,3], got {tuple(xyz_np.shape)}")
    num_points = int(xyz_np.shape[0])
    if num_points == 0 or k <= 0:
        return (
            np.empty((num_points, 0), dtype=np.int64),
            np.empty((num_points, 0), dtype=np.float32),
            {
                "enabled": False,
                "k": int(k),
                "sigma": float(sigma),
                "mean_neighbor_distance": 0.0,
            },
        )

    try:
        from scipy.spatial import cKDTree
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("spatial_knn logit smoothing requires scipy") from exc

    query_k = min(num_points, int(k) + 1)
    tree = cKDTree(xyz_np)
    try:
        distances, indices = tree.query(xyz_np, k=query_k, workers=-1)
    except TypeError:  # scipy<1.6
        distances, indices = tree.query(xyz_np, k=query_k)
    distances = np.asarray(distances, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    if query_k > 1:
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    if indices.shape[1] == 0:
        weights = np.empty((num_points, 0), dtype=np.float32)
    elif sigma > 0:
        weights = np.exp(-0.5 * np.square(distances / float(sigma))).astype(np.float32)
    else:
        weights = (1.0 / np.maximum(distances, 1e-4)).astype(np.float32)
    if weights.size:
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    stats = {
        "enabled": bool(indices.shape[1] > 0),
        "k": int(k),
        "sigma": float(sigma),
        "mean_neighbor_distance": float(distances.mean()) if distances.size else 0.0,
    }
    return indices, weights, stats


def _apply_spatial_knn_smoothing(
    logits: torch.Tensor,
    indices: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    iterations: int,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"Expected logits [N,C], got {tuple(logits.shape)}")
    if indices.shape[0] != logits.shape[0] or weights.shape != indices.shape:
        raise ValueError(
            "spatial KNN graph must be aligned with logits: "
            f"indices={indices.shape}, weights={weights.shape}, logits={tuple(logits.shape)}"
        )
    if indices.shape[1] == 0 or alpha <= 0.0 or iterations <= 0:
        return logits
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("logit_smoothing_alpha must be in [0, 1]")

    values = logits.detach().float().cpu().numpy().astype(np.float32, copy=True)
    residual = 1.0 - float(alpha)
    blend = float(alpha)
    for _ in range(int(iterations)):
        neigh = values[indices]
        propagated = (neigh * weights[..., None]).sum(axis=1)
        values = residual * values + blend * propagated
    return torch.from_numpy(values).to(device=logits.device, dtype=logits.dtype)


def _smooth_logits_spatial_knn(
    logits: torch.Tensor,
    xyz: np.ndarray | torch.Tensor,
    *,
    k: int,
    alpha: float,
    iterations: int = 1,
    sigma: float = 0.0,
    graph: tuple[np.ndarray, np.ndarray, dict[str, float]] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Propagate text logits over a local 3D graph without using labels.

    This is a readout-time consistency prior for ScanNet direct point queries:
    nearby points should have compatible language scores, but the original
    point logits remain in the residual path so thin structures are not fully
    washed out.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits [N,C], got {tuple(logits.shape)}")
    num_points = int(logits.shape[0])
    if num_points == 0 or k <= 0 or alpha <= 0.0 or iterations <= 0:
        return logits, {
            "enabled": False,
            "k": int(k),
            "alpha": float(alpha),
            "iterations": int(iterations),
            "sigma": float(sigma),
            "mean_neighbor_distance": 0.0,
        }
    if graph is None:
        graph = _build_spatial_knn_graph(xyz, k=k, sigma=sigma)
    indices, weights, graph_stats = graph
    if indices.shape[0] != num_points:
        raise ValueError(
            f"spatial KNN graph has {indices.shape[0]} points, expected {num_points}"
        )
    smoothed = _apply_spatial_knn_smoothing(
        logits,
        indices,
        weights,
        alpha=alpha,
        iterations=iterations,
    )
    stats = dict(graph_stats)
    stats.update({
        "enabled": bool(indices.shape[1] > 0),
        "k": int(k),
        "alpha": float(alpha),
        "iterations": int(iterations),
        "sigma": float(sigma),
    })
    return smoothed, stats


def _build_voxel_proposal_labels_for_points(
    xyz: np.ndarray | torch.Tensor,
    *,
    voxel_size: float,
) -> np.ndarray:
    xyz_tensor = xyz.detach().float().cpu() if isinstance(xyz, torch.Tensor) else torch.as_tensor(xyz, dtype=torch.float32)
    labels = build_voxel_proposal_labels(xyz_tensor, voxel_size=voxel_size)
    return labels.detach().cpu().numpy().astype(np.int64, copy=False)


def _smooth_logits_voxel_proposals(
    logits: torch.Tensor,
    xyz: np.ndarray | torch.Tensor,
    *,
    voxel_size: float,
    alpha: float,
    min_count: int = 2,
    gate: str = "all",
    margin_threshold: float = 0.0,
    confidence_threshold: float = 0.0,
    proposal_consensus_threshold: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Pool logits inside deterministic 3D voxel proposals and blend them back."""
    if logits.ndim != 2:
        raise ValueError(f"Expected logits [N,C], got {tuple(logits.shape)}")
    if logits.shape[0] == 0 or alpha <= 0.0:
        return logits, {
            "enabled": False,
            "mode": "voxel",
            "voxel_size": float(voxel_size),
            "alpha": float(alpha),
            "min_count": int(min_count),
            "gate": gate,
            "margin_threshold": float(margin_threshold),
            "confidence_threshold": float(confidence_threshold),
            "proposal_consensus_threshold": float(proposal_consensus_threshold),
            "num_proposals": 0,
            "num_assigned": 0,
        }
    xyz_tensor = xyz.detach().float().cpu() if isinstance(xyz, torch.Tensor) else torch.as_tensor(xyz, dtype=torch.float32)
    labels = build_voxel_proposal_labels(xyz_tensor, voxel_size=voxel_size).to(logits.device)
    smoothed, stats = propagate_logits_with_proposals(
        logits,
        labels,
        alpha=alpha,
        min_count=min_count,
        gate=gate,
        margin_threshold=margin_threshold,
        confidence_threshold=confidence_threshold,
        proposal_consensus_threshold=proposal_consensus_threshold,
    )
    stats = dict(stats)
    stats.update(
        {
            "mode": "voxel",
            "voxel_size": float(voxel_size),
        }
    )
    return smoothed, stats


def _empty_query_diagnostics() -> dict:
    return {
        "num_points": 0,
        "top1_weight_sum": 0.0,
        "weight_entropy_sum": 0.0,
        "effective_neighbors_sum": 0.0,
        "top1_euclidean_dist_sum": 0.0,
        "top1_mahalanobis_dist2_sum": 0.0,
        "top1_density_sum": 0.0,
    }


def _update_query_diagnostics(diagnostics: dict, aux: dict[str, torch.Tensor]) -> None:
    weights = aux.get("weights")
    if weights is None:
        return
    weights_cpu = weights.detach().float().cpu()
    num_points = int(weights_cpu.shape[0])
    diagnostics["num_points"] += num_points
    if num_points == 0:
        return

    top1_weight, top1_idx = weights_cpu.max(dim=1)
    entropy = -(weights_cpu * weights_cpu.clamp_min(1e-12).log()).sum(dim=1)
    effective_neighbors = 1.0 / weights_cpu.square().sum(dim=1).clamp_min(1e-12)

    diagnostics["top1_weight_sum"] += float(top1_weight.sum().item())
    diagnostics["weight_entropy_sum"] += float(entropy.sum().item())
    diagnostics["effective_neighbors_sum"] += float(effective_neighbors.sum().item())

    for key, target in (
        ("euclidean_dist", "top1_euclidean_dist_sum"),
        ("mahalanobis_dist2", "top1_mahalanobis_dist2_sum"),
        ("density", "top1_density_sum"),
    ):
        values = aux.get(key)
        if values is None:
            continue
        values_cpu = values.detach().float().cpu()
        if values_cpu.ndim == 2 and values_cpu.shape[0] == num_points:
            diagnostics[target] += float(
                values_cpu.gather(1, top1_idx[:, None]).squeeze(1).sum().item()
            )


def _finalize_query_diagnostics(diagnostics: dict) -> dict:
    num_points = int(diagnostics["num_points"])
    if num_points == 0:
        return {
            "num_points": 0,
            "mean_top1_weight": 0.0,
            "mean_weight_entropy": 0.0,
            "mean_effective_neighbors": 0.0,
            "mean_top1_euclidean_dist": 0.0,
            "mean_top1_mahalanobis_dist2": 0.0,
            "mean_top1_density": 0.0,
        }
    return {
        "num_points": num_points,
        "mean_top1_weight": float(diagnostics["top1_weight_sum"] / num_points),
        "mean_weight_entropy": float(diagnostics["weight_entropy_sum"] / num_points),
        "mean_effective_neighbors": float(diagnostics["effective_neighbors_sum"] / num_points),
        "mean_top1_euclidean_dist": float(
            diagnostics["top1_euclidean_dist_sum"] / num_points
        ),
        "mean_top1_mahalanobis_dist2": float(
            diagnostics["top1_mahalanobis_dist2_sum"] / num_points
        ),
        "mean_top1_density": float(diagnostics["top1_density_sum"] / num_points),
    }


def _save_split_logits_npz(
    output_dir: Path,
    scene: str,
    split: str,
    diagnostics: dict,
) -> str:
    logits_path = output_dir / "logits" / scene / f"split_{split}_logits.npz"
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    logits_parts = diagnostics["logits"]
    label_parts = diagnostics["labels"]
    pred_parts = diagnostics["pred_ids"]
    num_classes = int(diagnostics["logit_sum"].shape[0])
    logits = (
        np.concatenate(logits_parts, axis=0)
        if logits_parts
        else np.empty((0, num_classes), dtype=np.float32)
    )
    labels = np.concatenate(label_parts, axis=0) if label_parts else np.empty((0,), dtype=np.int32)
    pred_ids = np.concatenate(pred_parts, axis=0) if pred_parts else np.empty((0,), dtype=np.int32)
    np.savez_compressed(logits_path, logits=logits, labels=labels, pred_ids=pred_ids)
    return str(logits_path)


def _label_colors(labels: np.ndarray) -> np.ndarray:
    colors = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    for raw_id in np.unique(labels):
        raw_id_int = int(raw_id)
        rng = np.random.default_rng(raw_id_int + 17)
        colors[labels == raw_id_int] = rng.integers(40, 255, size=3, dtype=np.uint8)
    return colors


def _save_prediction_ply(
    path: Path,
    xyz: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = _label_colors(pred_labels)
    arr = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("label", "i4"),
            ("pred_label", "i4"),
        ],
    )
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    arr["label"] = gt_labels.astype(np.int32)
    arr["pred_label"] = pred_labels.astype(np.int32)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def _fixed_rgb_projection_matrix(feature_dim: int, seed: int = FEATURE_RGB_PROJECTION_SEED) -> np.ndarray:
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(feature_dim, 3)).astype(np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=0, keepdims=True), 1e-6)
    return matrix


def _project_features_to_rgb_values(
    features: np.ndarray | torch.Tensor,
    projection_matrix: np.ndarray,
) -> np.ndarray:
    if isinstance(features, torch.Tensor):
        features_np = features.detach().float().cpu().numpy()
    else:
        features_np = np.asarray(features, dtype=np.float32)
    if features_np.ndim != 2:
        raise ValueError(f"features must have shape [N, D], got {features_np.shape}")
    if features_np.shape[1] != projection_matrix.shape[0]:
        raise ValueError(
            f"feature dim mismatch: features D={features_np.shape[1]}, "
            f"projection D={projection_matrix.shape[0]}"
        )
    features_np = features_np / np.maximum(np.linalg.norm(features_np, axis=1, keepdims=True), 1e-6)
    return features_np @ projection_matrix


def _normalize_rgb_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    lo = np.percentile(values, 1.0, axis=0)
    hi = np.percentile(values, 99.0, axis=0)
    scale = hi - lo
    if np.any(scale < 1e-6):
        min_v = values.min(axis=0)
        max_v = values.max(axis=0)
        fallback_scale = max_v - min_v
        lo = np.where(scale < 1e-6, min_v, lo)
        scale = np.where(scale < 1e-6, fallback_scale, scale)
    safe_scale = np.where(scale < 1e-6, 1.0, scale)
    normalized = np.clip((values - lo) / safe_scale, 0.0, 1.0)
    normalized[:, scale < 1e-6] = 0.5
    return np.rint(normalized * 255.0).astype(np.uint8)


def _fixed_rgb_project_features(
    features: np.ndarray | torch.Tensor,
    seed: int = FEATURE_RGB_PROJECTION_SEED,
) -> np.ndarray:
    feature_dim = int(features.shape[1])
    matrix = _fixed_rgb_projection_matrix(feature_dim, seed=seed)
    return _normalize_rgb_values(_project_features_to_rgb_values(features, matrix))


def _save_feature_rgb_ply(
    path: Path,
    xyz: np.ndarray,
    gt_labels: np.ndarray,
    colors: np.ndarray,
) -> None:
    if xyz.shape[0] != colors.shape[0]:
        raise ValueError(f"xyz/colors length mismatch: {xyz.shape[0]} vs {colors.shape[0]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("label", "i4"),
        ],
    )
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    arr["label"] = gt_labels.astype(np.int32)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def _save_language_features_npz(
    path: Path,
    xyz: np.ndarray,
    gt_labels: np.ndarray,
    features: np.ndarray | torch.Tensor,
) -> None:
    """Save normalized language-space point features aligned to label vertices."""
    if isinstance(features, torch.Tensor):
        features_np = features.detach().float().cpu().numpy()
    else:
        features_np = np.asarray(features, dtype=np.float32)
    if features_np.ndim != 2:
        raise ValueError(f"features must have shape [N,D], got {features_np.shape}")
    if xyz.shape[0] != features_np.shape[0]:
        raise ValueError(f"xyz/features length mismatch: {xyz.shape[0]} vs {features_np.shape[0]}")
    if gt_labels.shape[0] != features_np.shape[0]:
        raise ValueError(
            f"labels/features length mismatch: {gt_labels.shape[0]} vs {features_np.shape[0]}"
        )
    features_np = features_np / np.maximum(
        np.linalg.norm(features_np, axis=1, keepdims=True),
        1e-6,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        xyz=np.asarray(xyz, dtype=np.float32),
        labels=np.asarray(gt_labels, dtype=np.int32),
        features=features_np.astype(np.float16),
    )


def evaluate_scene(
    scene: str,
    config_path: str,
    checkpoint_path: str,
    label_ply: str,
    projection: torch.nn.Module,
    split_text_embeddings: Dict[str, torch.Tensor],
    device: torch.device,
    *,
    split_names: Iterable[str],
    k: int = 8,
    candidate_k: Optional[int] = None,
    chunk_size: int = 4096,
    max_points: Optional[int] = None,
    sample_seed: int = 42,
    output_dir: Optional[Path] = None,
    save_ply: bool = False,
    query_mode: str = "knn",
    save_logits_npz: bool = False,
    save_feature_rgb_ply: bool = False,
    save_language_features_npz: bool = False,
    feature_rgb_seed: int = FEATURE_RGB_PROJECTION_SEED,
    compact_feature_key: str = "features",
    use_point_summary_adapter: bool = False,
    point_summary_adapter_blend_alpha: float = 1.0,
    gaussian_index_position_mode: str = "label_point",
    opacity_threshold: Optional[float] = 0.1,
    opacity_filter_mode: str = "auto",
    logit_calibration: str = "none",
    logit_calibration_alpha: float = 1.0,
    logit_smoothing: str = "none",
    logit_smoothing_k: int = 8,
    logit_smoothing_alpha: float = 0.0,
    logit_smoothing_iterations: int = 1,
    logit_smoothing_sigma: float = 0.0,
    proposal_smoothing: str = "none",
    proposal_voxel_size: float = 0.08,
    proposal_smoothing_alpha: float = 0.0,
    proposal_min_count: int = 2,
    proposal_smoothing_gate: str = "all",
    proposal_margin_threshold: float = 0.0,
    proposal_confidence_threshold: float = 0.0,
    proposal_consensus_threshold: float = 0.0,
) -> dict:
    if query_mode not in QUERY_MODES:
        raise ValueError(f"query_mode must be one of: {', '.join(QUERY_MODES)}")
    if candidate_k is not None and candidate_k <= 0:
        candidate_k = None
    if logit_calibration not in LOGIT_CALIBRATION_MODES:
        raise ValueError(
            f"logit_calibration must be one of: {', '.join(LOGIT_CALIBRATION_MODES)}"
        )
    if logit_smoothing not in LOGIT_SMOOTHING_MODES:
        raise ValueError(
            f"logit_smoothing must be one of: {', '.join(LOGIT_SMOOTHING_MODES)}"
        )
    if proposal_smoothing not in PROPOSAL_SMOOTHING_MODES:
        raise ValueError(
            f"proposal_smoothing must be one of: {', '.join(PROPOSAL_SMOOTHING_MODES)}"
        )
    if opacity_filter_mode not in OPACITY_FILTER_MODES:
        raise ValueError(
            f"opacity_filter_mode must be one of: {', '.join(OPACITY_FILTER_MODES)}"
        )
    if gaussian_index_position_mode not in GAUSSIAN_INDEX_POSITION_MODES:
        raise ValueError(
            "gaussian_index_position_mode must be one of: "
            f"{', '.join(GAUSSIAN_INDEX_POSITION_MODES)}"
        )
    if save_logits_npz and output_dir is None:
        raise ValueError("output_dir is required when save_logits_npz=True")
    if save_feature_rgb_ply and output_dir is None:
        raise ValueError("output_dir is required when save_feature_rgb_ply=True")
    if save_language_features_npz and output_dir is None:
        raise ValueError("output_dir is required when save_language_features_npz=True")
    if not 0.0 <= float(point_summary_adapter_blend_alpha) <= 1.0:
        raise ValueError("point_summary_adapter_blend_alpha must be in [0, 1]")

    config = load_config(config_path)
    model, codec = _build_hybrid_model(config, checkpoint_path, device)
    point_summary_adapter = (
        _build_point_summary_adapter(config, checkpoint_path, device)
        if use_point_summary_adapter
        else None
    )

    xyz_np, labels_np = _read_label_ply(label_ply)
    loaded_label_count = int(labels_np.shape[0])
    gaussian_count = int(model.num_gaussians)
    active_opacity_filter_mode = "off"
    if opacity_threshold is not None and opacity_threshold >= 0:
        active_opacity_filter_mode = _resolve_opacity_filter_mode(
            opacity_filter_mode,
            query_mode=query_mode,
            label_count=loaded_label_count,
            gaussian_count=gaussian_count,
        )
        if active_opacity_filter_mode == "label_index":
            labels_np, opacity_filter = _apply_opacity_label_filter(
                labels_np,
                model.get_opacity(),
                threshold=float(opacity_threshold),
                scene=scene,
            )
        elif active_opacity_filter_mode == "off":
            opacity_filter = {
                "enabled": False,
                "mode": "off",
                "threshold": float(opacity_threshold),
                "num_filtered": 0,
                "num_points": loaded_label_count,
            }
        else:
            labels_np = labels_np.copy()
            opacity_filter = {
                "enabled": True,
                "mode": active_opacity_filter_mode,
                "threshold": float(opacity_threshold),
                "num_filtered": 0,
                "num_points": loaded_label_count,
            }
    else:
        opacity_filter = {
            "enabled": False,
            "mode": "off",
            "threshold": None,
            "num_filtered": 0,
            "num_points": loaded_label_count,
        }
    if query_mode == "gaussian_index":
        if loaded_label_count != gaussian_count:
            raise ValueError(
                "query_mode='gaussian_index' requires the loaded label point count "
                f"to equal the model gaussian count; got label points={loaded_label_count}, "
                f"gaussians={gaussian_count}"
            )
    xyz_np, labels_np, sample_indices = _subsample_points(
        xyz_np,
        labels_np,
        max_points=max_points,
        seed=sample_seed,
    )
    xyz = torch.from_numpy(xyz_np).to(device=device, dtype=torch.float32)
    if active_opacity_filter_mode in {"query_top1", "query_weighted"}:
        opacity_filter["num_points"] = int(labels_np.shape[0])
    if sample_indices is None:
        gaussian_indices_np = np.arange(loaded_label_count, dtype=np.int64)
    else:
        gaussian_indices_np = sample_indices.astype(np.int64, copy=False)
    gaussian_indices = torch.from_numpy(gaussian_indices_np).to(device=device, dtype=torch.long)

    split_ids = {
        split: OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]
        for split in split_names
    }
    pred_by_split = {
        split: np.full(labels_np.shape, -1, dtype=np.int32)
        for split in split_ids
    }
    diagnostics_by_split = {
        split: _empty_split_diagnostics(ids)
        for split, ids in split_ids.items()
    }
    query_diagnostics = _empty_query_diagnostics()
    feature_rgb_projection: Optional[np.ndarray] = None
    feature_rgb_value_parts: list[np.ndarray] = []
    language_feature_parts: list[np.ndarray] = []
    logit_bias_by_split: dict[str, torch.Tensor | None] = {
        split: None for split in split_ids
    }
    needs_logits_postprocess = (
        (logit_smoothing != "none" and logit_smoothing_alpha > 0.0)
        or (proposal_smoothing != "none" and proposal_smoothing_alpha > 0.0)
    )
    smooth_logits_parts: dict[str, list[torch.Tensor]] | None = (
        {split: [] for split in split_ids}
        if needs_logits_postprocess
        else None
    )
    logit_smoothing_by_split: dict[str, dict[str, float]] = {
        split: {
            "enabled": False,
            "mode": logit_smoothing,
            "k": int(logit_smoothing_k),
            "alpha": float(logit_smoothing_alpha),
            "iterations": int(logit_smoothing_iterations),
            "sigma": float(logit_smoothing_sigma),
            "mean_neighbor_distance": 0.0,
        }
        for split in split_ids
    }
    proposal_smoothing_by_split: dict[str, dict[str, float | int | bool]] = {
        split: {
            "enabled": False,
            "mode": proposal_smoothing,
            "voxel_size": float(proposal_voxel_size),
            "alpha": float(proposal_smoothing_alpha),
            "min_count": int(proposal_min_count),
            "gate": proposal_smoothing_gate,
            "margin_threshold": float(proposal_margin_threshold),
            "confidence_threshold": float(proposal_confidence_threshold),
            "proposal_consensus_threshold": float(proposal_consensus_threshold),
            "num_proposals": 0,
            "num_assigned": 0,
        }
        for split in split_ids
    }

    if logit_calibration == "scene_mean":
        logit_sum_by_split = {
            split: torch.zeros(len(ids), dtype=torch.float64)
            for split, ids in split_ids.items()
        }
        logit_count = 0
        for start in tqdm(
            range(0, xyz.shape[0], chunk_size),
            desc=f"{scene} logit calibration",
        ):
            end = min(start + chunk_size, xyz.shape[0])
            points = xyz[start:end]
            gaussian_points = (
                points if gaussian_index_position_mode == "label_point" else None
            )
            with torch.no_grad():
                if point_summary_adapter is not None:
                    if query_mode == "gaussian_index":
                        compact_result = model.query_gaussian_points(
                            gaussian_indices[start:end],
                            points_xyz=gaussian_points,
                            return_aux=True,
                        )
                    else:
                        point_k = 1 if query_mode == "nearest" else k
                        point_candidate_k = None if query_mode == "nearest" else candidate_k
                        query_kwargs = {"k": point_k, "return_aux": True}
                        if point_candidate_k is not None:
                            query_kwargs["candidate_k"] = point_candidate_k
                        compact_result = model.query_compact_points(
                            points,
                            **query_kwargs,
                        )
                    assert isinstance(compact_result, dict)
                    if compact_feature_key not in compact_result:
                        raise KeyError(
                            f"Requested compact feature branch '{compact_feature_key}', "
                            f"available={sorted(compact_result.keys())}"
                        )
                    visual = _project_compact_with_summary_adapter(
                        compact_result[compact_feature_key],
                        point_summary_adapter,
                    )
                    if point_summary_adapter_blend_alpha < 1.0:
                        decoded = _decode_compact_1280(
                            codec,
                            compact_result[compact_feature_key],
                        )
                        base_visual = _project_points(decoded, projection)
                        visual = _blend_summary_features(
                            base_visual,
                            visual,
                            alpha=point_summary_adapter_blend_alpha,
                        )
                else:
                    if query_mode == "gaussian_index":
                        decoded = _decode_gaussian_indices_1280(
                            model,
                            codec,
                            gaussian_indices[start:end],
                            points_xyz=gaussian_points,
                            return_aux=False,
                            compact_feature_key=compact_feature_key,
                        )
                    else:
                        point_k = 1 if query_mode == "nearest" else k
                        point_candidate_k = None if query_mode == "nearest" else candidate_k
                        decoded = _decode_points_1280(
                            model,
                            codec,
                            points,
                            k=point_k,
                            candidate_k=point_candidate_k,
                            return_aux=False,
                            compact_feature_key=compact_feature_key,
                        )
                    assert isinstance(decoded, torch.Tensor)
                    visual = _project_points(decoded, projection)
                for split, ids in split_ids.items():
                    text_emb = split_text_embeddings[split].to(device)
                    logits = visual @ text_emb.T
                    logit_sum_by_split[split] += logits.detach().double().cpu().sum(dim=0)
            logit_count += int(end - start)
        if logit_count > 0:
            logit_bias_by_split = {
                split: (values / float(logit_count)).float()
                for split, values in logit_sum_by_split.items()
            }

    for start in tqdm(range(0, xyz.shape[0], chunk_size), desc=f"{scene} point query"):
        end = min(start + chunk_size, xyz.shape[0])
        points = xyz[start:end]
        gaussian_points = points if gaussian_index_position_mode == "label_point" else None
        with torch.no_grad():
            if point_summary_adapter is not None:
                if query_mode == "gaussian_index":
                    compact_result = model.query_gaussian_points(
                        gaussian_indices[start:end],
                        points_xyz=gaussian_points,
                        return_aux=True,
                    )
                else:
                    point_k = 1 if query_mode == "nearest" else k
                    point_candidate_k = None if query_mode == "nearest" else candidate_k
                    query_kwargs = {"k": point_k, "return_aux": True}
                    if point_candidate_k is not None:
                        query_kwargs["candidate_k"] = point_candidate_k
                    compact_result = model.query_compact_points(
                        points,
                        **query_kwargs,
                    )
                assert isinstance(compact_result, dict)
                if compact_feature_key not in compact_result:
                    raise KeyError(
                        f"Requested compact feature branch '{compact_feature_key}', "
                        f"available={sorted(compact_result.keys())}"
                    )
                query_aux = compact_result
                visual = _project_compact_with_summary_adapter(
                    compact_result[compact_feature_key],
                    point_summary_adapter,
                )
                if point_summary_adapter_blend_alpha < 1.0:
                    decoded = _decode_compact_1280(
                        codec,
                        compact_result[compact_feature_key],
                    )
                    base_visual = _project_points(decoded, projection)
                    visual = _blend_summary_features(
                        base_visual,
                        visual,
                        alpha=point_summary_adapter_blend_alpha,
                    )
            else:
                if query_mode == "gaussian_index":
                    decoded, query_aux = _decode_gaussian_indices_1280(
                        model,
                        codec,
                        gaussian_indices[start:end],
                        points_xyz=gaussian_points,
                        return_aux=True,
                        compact_feature_key=compact_feature_key,
                    )
                else:
                    point_k = 1 if query_mode == "nearest" else k
                    point_candidate_k = None if query_mode == "nearest" else candidate_k
                    decoded, query_aux = _decode_points_1280(
                        model,
                        codec,
                        points,
                        k=point_k,
                        candidate_k=point_candidate_k,
                        return_aux=True,
                        compact_feature_key=compact_feature_key,
                    )
                visual = _project_points(decoded, projection)
            _update_query_diagnostics(query_diagnostics, query_aux)
            if active_opacity_filter_mode in {"query_top1", "query_weighted"}:
                filtered_labels, chunk_opacity_filter = _apply_query_opacity_label_filter(
                    labels_np[start:end],
                    query_aux,
                    model.get_opacity(),
                    threshold=float(opacity_threshold),
                    mode=active_opacity_filter_mode,
                )
                labels_np[start:end] = filtered_labels
                opacity_filter["num_filtered"] += chunk_opacity_filter["num_filtered"]
            if save_feature_rgb_ply:
                if feature_rgb_projection is None:
                    feature_rgb_projection = _fixed_rgb_projection_matrix(
                        int(visual.shape[1]),
                        seed=feature_rgb_seed,
                    )
                feature_rgb_value_parts.append(
                    _project_features_to_rgb_values(visual, feature_rgb_projection)
                )
            if save_language_features_npz:
                language_feature_parts.append(
                    visual.detach().float().cpu().numpy().astype(np.float32, copy=False)
                )
            for split, ids in split_ids.items():
                text_emb = split_text_embeddings[split].to(device)
                logits = _apply_logit_calibration(
                    visual @ text_emb.T,
                    logit_bias_by_split.get(split),
                    alpha=logit_calibration_alpha,
                )
                if smooth_logits_parts is not None:
                    smooth_logits_parts[split].append(logits.detach().float().cpu())
                else:
                    pred_idx = logits.argmax(dim=-1).detach().cpu().numpy()
                    raw_ids = np.asarray(ids, dtype=np.int32)
                    pred_ids = raw_ids[pred_idx]
                    pred_by_split[split][start:end] = pred_ids
                    _update_split_diagnostics(
                        diagnostics_by_split[split],
                        logits,
                        labels_np[start:end],
                        pred_ids,
                        save_logits_npz=save_logits_npz,
                    )

    if smooth_logits_parts is not None:
        smoothing_graph = None
        if logit_smoothing == "spatial_knn":
            smoothing_graph = _build_spatial_knn_graph(
                xyz_np,
                k=logit_smoothing_k,
                sigma=logit_smoothing_sigma,
            )
        for split, ids in split_ids.items():
            logits = torch.cat(smooth_logits_parts[split], dim=0)
            if logit_smoothing == "spatial_knn":
                logits, smoothing_stats = _smooth_logits_spatial_knn(
                    logits,
                    xyz_np,
                    k=logit_smoothing_k,
                    alpha=logit_smoothing_alpha,
                    iterations=logit_smoothing_iterations,
                    sigma=logit_smoothing_sigma,
                    graph=smoothing_graph,
                )
            else:  # pragma: no cover - guarded above
                smoothing_stats = logit_smoothing_by_split[split]
            if logit_smoothing != "none":
                smoothing_stats["mode"] = logit_smoothing
                logit_smoothing_by_split[split] = smoothing_stats
            if proposal_smoothing == "voxel" and proposal_smoothing_alpha > 0.0:
                logits, proposal_stats = _smooth_logits_voxel_proposals(
                    logits,
                    xyz_np,
                    voxel_size=proposal_voxel_size,
                    alpha=proposal_smoothing_alpha,
                    min_count=proposal_min_count,
                    gate=proposal_smoothing_gate,
                    margin_threshold=proposal_margin_threshold,
                    confidence_threshold=proposal_confidence_threshold,
                    proposal_consensus_threshold=proposal_consensus_threshold,
                )
                proposal_smoothing_by_split[split] = proposal_stats
            elif proposal_smoothing != "none":  # pragma: no cover - guarded above
                raise ValueError(f"Unsupported proposal_smoothing: {proposal_smoothing}")
            pred_idx = logits.argmax(dim=-1).detach().cpu().numpy()
            raw_ids = np.asarray(ids, dtype=np.int32)
            pred_ids = raw_ids[pred_idx]
            pred_by_split[split][:] = pred_ids
            _update_split_diagnostics(
                diagnostics_by_split[split],
                logits,
                labels_np,
                pred_ids,
                save_logits_npz=save_logits_npz,
            )

    scene_results: dict[str, dict] = {}
    for split, pred_labels in pred_by_split.items():
        metrics = compute_split_metrics(
            pred_labels=pred_labels,
            gt_labels=labels_np,
            split_ids=split_ids[split],
        )
        metrics.update(_finalize_split_diagnostics(diagnostics_by_split[split], split_ids[split]))
        if save_logits_npz and output_dir is not None:
            metrics["logits_npz"] = _save_split_logits_npz(
                output_dir,
                scene,
                split,
                diagnostics_by_split[split],
            )
        scene_results[split] = metrics
        if save_ply and output_dir is not None:
            _save_prediction_ply(
                output_dir / "visualizations" / scene / f"pred_split_{split}.ply",
                xyz_np,
                labels_np,
                pred_labels,
            )

    feature_rgb_ply = None
    if save_feature_rgb_ply and output_dir is not None:
        feature_rgb_ply = output_dir / "visualizations" / scene / "language_feature_rgb.ply"
        feature_rgb_values = (
            np.concatenate(feature_rgb_value_parts, axis=0)
            if feature_rgb_value_parts
            else np.empty((0, 3), dtype=np.float32)
        )
        _save_feature_rgb_ply(
            feature_rgb_ply,
            xyz_np,
            labels_np,
            _normalize_rgb_values(feature_rgb_values),
        )

    language_features_npz = None
    if save_language_features_npz and output_dir is not None:
        language_features_npz = output_dir / "visualizations" / scene / "language_features.npz"
        language_features = (
            np.concatenate(language_feature_parts, axis=0)
            if language_feature_parts
            else np.empty((0, 0), dtype=np.float32)
        )
        _save_language_features_npz(
            language_features_npz,
            xyz_np,
            labels_np,
            language_features,
        )

    return {
        "scene": scene,
        "label_ply": str(label_ply),
        "num_points": int(labels_np.shape[0]),
        "sample_indices": sample_indices.tolist() if sample_indices is not None else None,
        "query_mode": query_mode,
        "gaussian_index_position_mode": gaussian_index_position_mode,
        "candidate_k": int(candidate_k) if candidate_k is not None else None,
        "compact_feature_key": compact_feature_key,
        "feature_projection": (
            (
                "compact_to_summary_adapter"
                if point_summary_adapter_blend_alpha >= 1.0
                else "hcd_decoder_summary_blend_compact_to_summary_adapter"
            )
            if use_point_summary_adapter
            else "hcd_decoder_plus_siglip_summary_head"
        ),
        "point_summary_adapter_blend_alpha": (
            float(point_summary_adapter_blend_alpha)
            if use_point_summary_adapter
            else None
        ),
        "logit_calibration": {
            "mode": logit_calibration,
            "alpha": float(logit_calibration_alpha),
            "bias_by_split": {
                split: (
                    bias.detach().cpu().tolist()
                    if isinstance(bias, torch.Tensor)
                    else None
                )
                for split, bias in logit_bias_by_split.items()
            },
        },
        "logit_smoothing": logit_smoothing_by_split,
        "proposal_smoothing": proposal_smoothing_by_split,
        "opacity_filter": opacity_filter,
        "query_diagnostics": _finalize_query_diagnostics(query_diagnostics),
        "language_feature_rgb_ply": str(feature_rgb_ply) if feature_rgb_ply is not None else None,
        "language_features_npz": (
            str(language_features_npz) if language_features_npz is not None else None
        ),
        "splits": scene_results,
    }


def _discover_scenes(prepared_root: Path) -> list[str]:
    return sorted(path.name for path in prepared_root.glob("scene*") if path.is_dir())


def _parse_scene_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    scenes = [part.strip() for part in str(raw).replace(";", ",").split(",") if part.strip()]
    return list(dict.fromkeys(scenes))


def _default_label_ply(prepared_root: Path, scene: str) -> str:
    scene_root = prepared_root / scene
    preferred = scene_root / f"{scene}_vh_clean_2.labels.ply"
    if preferred.exists():
        return str(preferred)
    matches = sorted(scene_root.glob("*.labels.ply"))
    if not matches:
        raise FileNotFoundError(f"No label PLY found under {scene_root}")
    return str(matches[0])


def _write_csv(path: Path, scene_results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene", "split", "miou", "macc", "num_valid", "num_points"],
        )
        writer.writeheader()
        for scene, result in scene_results.items():
            for split, metrics in result["splits"].items():
                writer.writerow(
                    {
                        "scene": scene,
                        "split": split,
                        "miou": metrics["miou"],
                        "macc": metrics["macc"],
                        "num_valid": metrics["num_valid"],
                        "num_points": result["num_points"],
                    }
                )


def _add_query_mode_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--query_mode",
        choices=QUERY_MODES,
        default="knn",
        help=(
            "Feature query source: kNN at label vertices, nearest single-Gaussian "
            "point query, or direct Gaussian index lookup"
        ),
    )
    parser.add_argument(
        "--save_logits_npz",
        action="store_true",
        help="Save per-split logits, labels, and predicted raw ids for evaluated points",
    )
    parser.add_argument(
        "--compact_feature_key",
        choices=COMPACT_FEATURE_KEYS,
        default="features",
        help=(
            "Compact branch decoded for text classification. Decoupled hybrid "
            "models expose semantic/geometry/fused in addition to the default features."
        ),
    )
    parser.add_argument(
        "--logit_calibration",
        choices=LOGIT_CALIBRATION_MODES,
        default="none",
        help=(
            "Optional label-free class-wise logit calibration. scene_mean subtracts "
            "the mean per-class logit over all evaluated scene points before argmax."
        ),
    )
    parser.add_argument(
        "--logit_calibration_alpha",
        type=float,
        default=1.0,
        help="Scale factor for --logit_calibration scene_mean",
    )
    parser.add_argument(
        "--logit_smoothing",
        choices=LOGIT_SMOOTHING_MODES,
        default="none",
        help=(
            "Optional label-free point-graph logit propagation. spatial_knn "
            "averages text logits over local 3D neighbours before argmax."
        ),
    )
    parser.add_argument(
        "--logit_smoothing_k",
        type=int,
        default=8,
        help="Neighbour count for --logit_smoothing spatial_knn",
    )
    parser.add_argument(
        "--logit_smoothing_alpha",
        type=float,
        default=0.0,
        help="Residual blend for spatial logit smoothing; 0 disables smoothing",
    )
    parser.add_argument(
        "--logit_smoothing_iterations",
        type=int,
        default=1,
        help="Number of spatial logit propagation iterations",
    )
    parser.add_argument(
        "--logit_smoothing_sigma",
        type=float,
        default=0.0,
        help="Gaussian distance sigma for smoothing; <=0 uses inverse-distance weights",
    )
    parser.add_argument(
        "--proposal_smoothing",
        choices=PROPOSAL_SMOOTHING_MODES,
        default="none",
        help=(
            "Optional label-free object/proposal readout. voxel pools logits "
            "inside deterministic 3D voxel proposals before argmax."
        ),
    )
    parser.add_argument(
        "--proposal_voxel_size",
        type=float,
        default=0.08,
        help="Voxel size in scene units for --proposal_smoothing voxel",
    )
    parser.add_argument(
        "--proposal_smoothing_alpha",
        type=float,
        default=0.0,
        help="Residual blend for proposal-pooled logits; 0 disables proposal readout",
    )
    parser.add_argument(
        "--proposal_min_count",
        type=int,
        default=2,
        help="Minimum points per proposal before proposal smoothing is applied",
    )
    parser.add_argument(
        "--proposal_smoothing_gate",
        choices=[
            "all",
            "low_margin",
            "low_confidence",
            "low_margin_or_low_confidence",
            "proposal_consensus",
            "low_margin_and_proposal_consensus",
            "low_confidence_and_proposal_consensus",
        ],
        default="all",
        help=(
            "Apply proposal smoothing to all rows, low-margin rows, "
            "low-confidence rows, either low-margin/low-confidence rows, "
            "or rows inside high-consensus proposals"
        ),
    )
    parser.add_argument(
        "--proposal_margin_threshold",
        type=float,
        default=0.0,
        help="Top1-top2 logit margin threshold for --proposal_smoothing_gate low_margin",
    )
    parser.add_argument(
        "--proposal_confidence_threshold",
        type=float,
        default=0.0,
        help="Softmax top1 confidence threshold for low-confidence proposal smoothing gates",
    )
    parser.add_argument(
        "--proposal_consensus_threshold",
        type=float,
        default=0.0,
        help="Minimum proposal top-1 agreement for proposal-consensus smoothing gates",
    )
    parser.add_argument(
        "--class_aliases",
        choices=CLASS_ALIAS_MODES,
        default="none",
        help="Optional ScanNet class-name alias ensemble for text embeddings",
    )
    parser.add_argument(
        "--gaussian_index_position_mode",
        choices=GAUSSIAN_INDEX_POSITION_MODES,
        default="label_point",
        help=(
            "For query_mode=gaussian_index, decode the hash/coarse branch at the "
            "optimized Gaussian center or at the row-aligned label point coordinate."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct RADIO-GS ScanNet point-cloud understanding evaluator"
    )
    parser.add_argument("--scene", default="scene0000_00", help="Scene id or 'all'")
    parser.add_argument(
        "--scene_list",
        default=None,
        help=(
            "Optional comma- or semicolon-separated scene ids. Overrides --scene "
            "and is useful for fixed published splits such as VALA/OpenGaFF ScanNet."
        ),
    )
    parser.add_argument("--prepared_root", default=str(DEFAULT_PREPARED_ROOT))
    parser.add_argument("--config", required=True, help="Config path; may contain {scene}")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path; may contain {scene}")
    parser.add_argument("--label_ply", default=None, help="Optional label PLY path; may contain {scene}")
    parser.add_argument("--output_dir", default="output/scannet_pointcloud_eval")
    parser.add_argument("--class_splits", default="19,15,10")
    parser.add_argument("--k", type=int, default=8, help="Gaussian neighbours per point")
    parser.add_argument(
        "--candidate_k",
        type=int,
        default=0,
        help=(
            "Optional Euclidean candidate count before density pruning for knn "
            "direct query; 0 keeps the legacy candidate count equal to --k."
        ),
    )
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--max_points", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument(
        "--opacity_threshold",
        type=float,
        default=0.1,
        help=(
            "OpenGaussian-compatible opacity filter. Labels whose matching "
            "Gaussian opacity is below this threshold are treated as invalid; "
            "set a negative value to disable."
        ),
    )
    parser.add_argument(
        "--opacity_filter_mode",
        choices=OPACITY_FILTER_MODES,
        default="auto",
        help=(
            "How to map Gaussian opacity to evaluated labels. auto uses row-aligned "
            "filtering only for gaussian_index and query-neighbor opacity for point "
            "queries."
        ),
    )
    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument(
        "--save_feature_rgb_ply",
        action="store_true",
        help="Save normalized language-aligned point features as a fixed RGB projection PLY",
    )
    parser.add_argument(
        "--save_language_features_npz",
        action="store_true",
        help="Save normalized language-space point features aligned to label vertices",
    )
    parser.add_argument(
        "--feature_rgb_seed",
        type=int,
        default=FEATURE_RGB_PROJECTION_SEED,
        help="Seed for the fixed RGB projection used by --save_feature_rgb_ply",
    )
    _add_query_mode_args(parser)
    parser.add_argument("--prompt_templates", default="{query}")
    parser.add_argument("--text_embedding_cache", default=None)
    parser.add_argument("--projection_weights", default=DEFAULT_SIGLIP2_PROJECTION_WEIGHTS)
    parser.add_argument("--summary_head_weights", default="checkpoints/siglip2_summary_head.pth")
    parser.add_argument("--radio_checkpoint", default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar")
    parser.add_argument("--use_summary_head", action="store_true", default=True)
    parser.add_argument("--no_summary_head", dest="use_summary_head", action="store_false")
    parser.add_argument(
        "--use_point_summary_adapter",
        action="store_true",
        help="Classify compact point features through a trained compact-to-summary adapter",
    )
    parser.add_argument(
        "--point_summary_adapter_blend_alpha",
        type=float,
        default=1.0,
        help=(
            "When using --use_point_summary_adapter, blend decoded base summary "
            "features with adapter summary features: 0=base only, 1=adapter only"
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    prepared_root = Path(args.prepared_root)
    scene_list = _parse_scene_list(args.scene_list)
    scenes = scene_list if scene_list is not None else (
        _discover_scenes(prepared_root) if args.scene == "all" else [args.scene]
    )
    if not scenes:
        raise FileNotFoundError(f"No prepared scenes found under {prepared_root}")
    split_names = _parse_splits(args.class_splits)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    projection = _load_projection(args, device)
    prompt_templates = parse_prompt_templates(args.prompt_templates)
    split_text_embeddings: Dict[str, torch.Tensor] = {}
    for split in split_names:
        class_names = [
            NYU40_ID_TO_NAME[class_id]
            for class_id in OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]
        ]
        cache_path = None
        if args.text_embedding_cache:
            base = Path(args.text_embedding_cache)
            alias_suffix = (
                f"_aliases_{args.class_aliases}" if args.class_aliases != "none" else ""
            )
            cache_path = str(base.with_name(f"{base.stem}_split{split}{alias_suffix}.pt"))
        split_text_embeddings[split] = _load_or_generate_class_text_embeddings(
            class_names,
            device,
            cache_path=cache_path,
            prompt_templates=prompt_templates,
            class_aliases=args.class_aliases,
        )

    print("=" * 72)
    print("  ScanNet Direct RADIO-GS Point-Cloud Evaluation")
    print("=" * 72)
    print(f"  Scenes:      {', '.join(scenes)}")
    print(f"  Splits:      {', '.join(split_names)}")
    print(f"  k:           {args.k}")
    print(f"  candidate_k: {args.candidate_k or 'legacy'}")
    print(f"  Query mode:  {args.query_mode}")
    if args.query_mode == "gaussian_index":
        print(f"  GIdx pos:    {args.gaussian_index_position_mode}")
    print(f"  Opacity:     {args.opacity_filter_mode} @ {args.opacity_threshold:g}")
    print(f"  Branch:      {args.compact_feature_key}")
    print(f"  Aliases:     {args.class_aliases}")
    print(
        f"  Calibration: {args.logit_calibration}"
        + (
            f" (alpha={args.logit_calibration_alpha:g})"
            if args.logit_calibration != "none"
            else ""
        )
    )
    print(
        f"  Smoothing:   {args.logit_smoothing}"
        + (
            f" (k={args.logit_smoothing_k}, alpha={args.logit_smoothing_alpha:g}, "
            f"iters={args.logit_smoothing_iterations}, sigma={args.logit_smoothing_sigma:g})"
            if args.logit_smoothing != "none"
            else ""
        )
    )
    print(
        f"  Proposal:    {args.proposal_smoothing}"
        + (
            f" (voxel={args.proposal_voxel_size:g}, "
            f"alpha={args.proposal_smoothing_alpha:g}, "
            f"min_count={args.proposal_min_count}, "
            f"gate={args.proposal_smoothing_gate}, "
            f"margin={args.proposal_margin_threshold:g}, "
            f"conf={args.proposal_confidence_threshold:g}, "
            f"consensus={args.proposal_consensus_threshold:g})"
            if args.proposal_smoothing != "none"
            else ""
        )
    )
    print(
        "  Projection:  "
        + (
            (
                f"base/adapter blend alpha={args.point_summary_adapter_blend_alpha:g}"
                if args.point_summary_adapter_blend_alpha < 1.0
                else "compact-to-summary adapter"
            )
            if args.use_point_summary_adapter
            else "HCD decoder + SigLIP summary head"
        )
    )
    print(f"  Chunk size:  {args.chunk_size}")
    print(f"  Max points:  {args.max_points or 'all'}")
    print()

    all_results: dict[str, dict] = {}
    for scene in scenes:
        label_ply = _format_scene_path(args.label_ply, scene) or _default_label_ply(prepared_root, scene)
        result = evaluate_scene(
            scene=scene,
            config_path=_format_scene_path(args.config, scene),
            checkpoint_path=_format_scene_path(args.checkpoint, scene),
            label_ply=label_ply,
            projection=projection,
            split_text_embeddings=split_text_embeddings,
            device=device,
            split_names=split_names,
            k=args.k,
            candidate_k=args.candidate_k if args.candidate_k > 0 else None,
            chunk_size=args.chunk_size,
            max_points=args.max_points,
            sample_seed=args.sample_seed,
            output_dir=output_dir,
            save_ply=args.save_ply,
            query_mode=args.query_mode,
            save_logits_npz=args.save_logits_npz,
            save_feature_rgb_ply=args.save_feature_rgb_ply,
            save_language_features_npz=args.save_language_features_npz,
            feature_rgb_seed=args.feature_rgb_seed,
            compact_feature_key=args.compact_feature_key,
            use_point_summary_adapter=args.use_point_summary_adapter,
            point_summary_adapter_blend_alpha=args.point_summary_adapter_blend_alpha,
            gaussian_index_position_mode=args.gaussian_index_position_mode,
            opacity_threshold=args.opacity_threshold,
            opacity_filter_mode=args.opacity_filter_mode,
            logit_calibration=args.logit_calibration,
            logit_calibration_alpha=args.logit_calibration_alpha,
            logit_smoothing=args.logit_smoothing,
            logit_smoothing_k=args.logit_smoothing_k,
            logit_smoothing_alpha=args.logit_smoothing_alpha,
            logit_smoothing_iterations=args.logit_smoothing_iterations,
            logit_smoothing_sigma=args.logit_smoothing_sigma,
            proposal_smoothing=args.proposal_smoothing,
            proposal_voxel_size=args.proposal_voxel_size,
            proposal_smoothing_alpha=args.proposal_smoothing_alpha,
            proposal_min_count=args.proposal_min_count,
            proposal_smoothing_gate=args.proposal_smoothing_gate,
            proposal_margin_threshold=args.proposal_margin_threshold,
            proposal_confidence_threshold=args.proposal_confidence_threshold,
            proposal_consensus_threshold=args.proposal_consensus_threshold,
        )
        all_results[scene] = result
        for split in split_names:
            metrics = result["splits"][split]
            print(
                f"{scene} split{split}: "
                f"mIoU={metrics['miou']:.4f} mAcc={metrics['macc']:.4f} "
                f"valid={metrics['num_valid']}"
            )

    macro = {}
    for split in split_names:
        macro[split] = {
            "miou": float(np.mean([res["splits"][split]["miou"] for res in all_results.values()])),
            "macc": float(np.mean([res["splits"][split]["macc"] for res in all_results.values()])),
        }

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {k: str(v) for k, v in vars(args).items()},
        "prompt_templates": prompt_templates,
        "class_aliases": args.class_aliases,
        "gaussian_index_position_mode": args.gaussian_index_position_mode,
        "proposal_smoothing": {
            "mode": args.proposal_smoothing,
            "voxel_size": float(args.proposal_voxel_size),
            "alpha": float(args.proposal_smoothing_alpha),
            "min_count": int(args.proposal_min_count),
            "gate": args.proposal_smoothing_gate,
            "margin_threshold": float(args.proposal_margin_threshold),
            "confidence_threshold": float(args.proposal_confidence_threshold),
            "proposal_consensus_threshold": float(args.proposal_consensus_threshold),
        },
        "macro": macro,
        "scenes": all_results,
    }
    json_path = output_dir / "scannet_pointcloud_radio_gs_results.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(output_dir / "scannet_pointcloud_radio_gs_results.csv", all_results)

    print("\nMacro:")
    for split, metrics in macro.items():
        print(f"  split{split}: mIoU={metrics['miou']:.4f} mAcc={metrics['macc']:.4f}")
    print(f"\nSaved JSON: {json_path}")


if __name__ == "__main__":
    main()
