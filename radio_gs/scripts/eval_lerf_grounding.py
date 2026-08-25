"""Evaluate text grounding on LERF-OVS with an explicit protocol record.

The evaluator supports raw teacher maps, legacy rendered RADIO maps, and
row-aligned primitive-query capability caches.  Query text, negative prompts,
scoring, threshold convention, localization rule, and optional refinement are
all serialized in the result.  ``vala_paper_2d`` fixes exact ``{query}`` text,
four generic negatives, relevancy logit scale 10, absolute threshold 0.5,
image-resolution masks, bbox-smoothed localization, and no refinement or
test-set calibration.

Usage
-----
    # Rendered features (full evaluation)
    python -m radio_gs.scripts.eval_lerf_grounding \\
        --config  configs/lerf_figurines.yaml \\
        --checkpoint output/lerf_figurines/best.pt \\
        --scene figurines

    # Teacher/oracle RADIO features only (upper-bound evaluation)
    python -m radio_gs.scripts.eval_lerf_grounding \\
        --gt_feature_dir output/radio_features_lerf/figurines/backbone \\
        --scene figurines --gt_only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import string
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, ".")

from radio_gs.artifact_paths import (
    DEFAULT_SIGLIP2_PROJECTION_WEIGHTS,
    resolve_siglip_projection_path,
)
from radio_gs.geometry_utils import resolve_use_2dgs
from radio_gs.data.lerf_dataset import (
    LERFDataset,
    _coerce_polygons,
    _load_annotation_json,
    _rasterize_polygons,
)
from radio_gs.models.siglip_projection import SigLIP2FeatureProjection, SigLIP2SummaryHead
from radio_gs.models.prompt_conditioned_mask_head import PromptConditionedMaskHead
from radio_gs.models.prompt_conditioned_mask_refinement import (
    filter_refined_mask_by_heatmap_support,
    mask_overlap_stats,
    refine_mask_with_prompt_conditioned_sam3_head,
)
from radio_gs.evaluation.openclip_readout import (
    NEGATIVE_PROMPTS,
    load_or_generate_openclip_prompt_ensemble_embeddings,
)
from radio_gs.utils.checkpoint_io import load_trusted_checkpoint
from radio_gs.querying.unified_query import cosine_relevancy_torch

logger = logging.getLogger(__name__)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LERF_OVS_SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")

DEFAULT_LABEL_DIR = "/mnt/pool/sqy/3d_understanding/lerf_ovs/label"
LEGACY_LABEL_DIR = "/mnt/pool/sqy/lerf_ovs/label"
DEFAULT_GT_FEATURE_ROOT = "output/radio_features_lerf"
DEFAULT_PROMPT_TEMPLATES = "{query}"
DEFAULT_SAM3_PROMPT_MASK_HEAD_LOGIT_THRESHOLD = 0.0
DEFAULT_SAM3_PROMPT_MASK_HEAD_MIN_INITIAL_IOU = 0.50
DEFAULT_SAM3_PROMPT_MASK_HEAD_MAX_INITIAL_AREA_FRACTION = 1.0
DEFAULT_SAM3_PROMPT_MASK_HEAD_MIN_REFINED_AREA_RATIO = 0.70
DEFAULT_SAM3_PROMPT_MASK_HEAD_MAX_REFINED_AREA_RATIO = 1.30
DEFAULT_SAM3_PROMPT_MASK_HEAD_SUPPORT_DILATE = 12
DEFAULT_SAM3_PROMPT_MASK_HEAD_COARSE_DILATE = 1
DEFAULT_SAM3_PROMPT_MASK_HEAD_COARSE_THRESHOLD = 0.5


def canonical_lerf_mode(mode: str) -> str:
    """Return the canonical public name for a LERF evaluator feature source."""
    normalized = str(mode).strip().lower()
    if normalized in {"gt", "teacher", "oracle", "oracle_teacher"}:
        return "teacher"
    if normalized == "rendered":
        return "rendered"
    raise ValueError(f"Unsupported LERF evaluator mode: {mode}")


def lerf_mode_tag(mode: str) -> str:
    """Filename tag for a LERF evaluator mode."""
    return canonical_lerf_mode(mode)


def display_lerf_mode(mode: str) -> str:
    """Human-readable label for summaries and figure scripts."""
    canonical = canonical_lerf_mode(mode)
    if canonical == "teacher":
        return "TEACHER RADIO features"
    return "RENDERED RADIO-GS features"


def iter_lerf_report_modes(scene_result: Dict) -> List[str]:
    """Canonical report order, accepting the legacy ``gt`` alias."""
    modes: List[str] = []
    for mode in ("teacher", "rendered"):
        if mode in scene_result or (mode == "teacher" and "gt" in scene_result):
            modes.append(mode)
    return modes


def get_lerf_mode_metrics(scene_result: Dict, mode: str) -> Optional[Dict]:
    """Fetch metrics by canonical mode while accepting the legacy ``gt`` key."""
    canonical = canonical_lerf_mode(mode)
    if canonical in scene_result:
        return scene_result[canonical]
    if canonical == "teacher":
        return scene_result.get("gt")
    return None


def aggregate_lerf_mode_metrics(metrics: List[Dict]) -> Dict:
    """Keep benchmark sample, scene, and category averaging conventions explicit."""

    if not metrics:
        raise ValueError("cannot aggregate an empty LERF metric list")
    sample_counts = [
        int(value.get("n_iou_samples", value["loc_total"])) for value in metrics
    ]
    sample_count = sum(sample_counts)
    loc_total = sum(int(value["loc_total"]) for value in metrics)
    category_values = [
        float(info["miou"])
        for value in metrics
        for info in value["per_category"].values()
        if info["miou"] is not None
    ]
    return {
        "localization_accuracy": sum(int(value["loc_correct"]) for value in metrics)
        / max(1, loc_total),
        "sample_micro_miou": sum(
            float(value["miou"]) * count
            for value, count in zip(metrics, sample_counts)
        )
        / max(1, sample_count),
        "scene_macro_miou": float(np.mean([value["miou"] for value in metrics])),
        "category_macro_miou": (
            float(np.mean(category_values)) if category_values else None
        ),
        "sample_count": sample_count,
        "scene_count": len(metrics),
    }


def neutralize_invalid_primitive_scores_for_render(
    scores: torch.Tensor,
    valid: Optional[torch.Tensor],
) -> torch.Tensor:
    """Use zero semantic support, not a selection sentinel, during rendering.

    Direct 3-D selection can safely assign invalid primitives a very negative
    logit. Alpha compositing cannot: that sentinel is mixed into visible
    pixels and overwhelms every valid score. A zero row keeps the invalid
    primitive as an opacity occluder while contributing no query support.
    """

    if valid is None:
        return scores
    mask = torch.as_tensor(valid, device=scores.device, dtype=torch.bool).reshape(-1)
    if mask.shape != (scores.shape[0],):
        raise ValueError("primitive validity mask does not align with score rows")
    result = scores.clone()
    result[~mask] = 0.0
    return result


def normalize_primitive_scores_by_valid_mass(
    rendered_scores_and_validity: torch.Tensor,
    *,
    eps: float = 1e-6,
    coverage_power: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate semantic score from valid-capability coverage.

    The last rendered channel is ``sum(w*v)/sum(w)`` and preceding channels
    are ``sum(w*v*s_q)/sum(w)``.  Dividing them yields the intended
    ``sum(w*v*s_q)/sum(w*v)`` without changing geometric transmittance.
    """

    rendered = torch.as_tensor(rendered_scores_and_validity)
    if rendered.ndim != 3 or rendered.shape[0] < 2:
        raise ValueError(
            "valid-conditioned primitive render needs score channels plus validity"
        )
    if not np.isfinite(coverage_power) or coverage_power < 0:
        raise ValueError("coverage_power must be finite and non-negative")
    validity = rendered[-1]
    supported = validity > float(eps)
    scores = torch.where(
        supported[None],
        rendered[:-1] / validity.clamp_min(float(eps))[None],
        torch.zeros_like(rendered[:-1]),
    )
    if coverage_power != 0:
        scores = scores * validity.clamp(0.0, 1.0).pow(float(coverage_power))[None]
    return scores, validity


def apply_primitive_semantic_confidence(
    scores: torch.Tensor,
    confidence: Optional[torch.Tensor],
) -> torch.Tensor:
    """Suppress uncertain primitive support before alpha compositing.

    Confidence is a row-aligned, query-independent field property stored by
    the semantic cache.  It is deliberately applied to scalar support after
    text scoring; scaling a normalized descriptor itself would have no effect
    on cosine similarity.
    """
    if confidence is None:
        return scores
    values = torch.as_tensor(
        confidence, device=scores.device, dtype=scores.dtype
    ).reshape(-1)
    if values.shape != (scores.shape[0],):
        raise ValueError("primitive semantic confidence does not align with scores")
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("primitive semantic confidence is non-finite")
    if bool(((values < 0.0) | (values > 1.0)).any()):
        raise ValueError("primitive semantic confidence must lie in [0,1]")
    return scores * values[:, None]


def monotonic_logit_calibration(
    probabilities: torch.Tensor,
    *,
    scale: float = 1.0,
    bias: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply an order-preserving Platt map to a typed Bernoulli posterior.

    ``scale`` is constrained to be strictly positive, so this operation can
    calibrate an output domain without changing proposal identity, ranking, or
    the shared Gaussian posterior's topology.  Exact zero and one endpoints
    remain endpoints; this matters for explicit absence and certain support.
    """

    values = torch.as_tensor(probabilities)
    if (
        not values.is_floating_point()
        or not bool(torch.isfinite(values).all())
        or bool(((values < 0.0) | (values > 1.0)).any())
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
        or not math.isfinite(float(bias))
        or not 0.0 < float(eps) < 0.5
    ):
        raise ValueError("monotonic posterior calibration inputs differ")
    interior = values.clamp(float(eps), 1.0 - float(eps))
    calibrated = torch.sigmoid(
        float(scale) * torch.logit(interior) + float(bias)
    )
    calibrated = torch.where(values == 0.0, torch.zeros_like(calibrated), calibrated)
    calibrated = torch.where(values == 1.0, torch.ones_like(calibrated), calibrated)
    return calibrated


def blend_primary_with_uncovered_fallback(
    primary_heatmaps: torch.Tensor,
    fallback_heatmaps: torch.Tensor,
    primary_coverage: torch.Tensor,
) -> torch.Tensor:
    """Add fallback semantics only in image regions lacking primary support."""
    if primary_heatmaps.shape != fallback_heatmaps.shape:
        raise ValueError("primary and fallback heatmaps must have equal shape")
    coverage = torch.as_tensor(
        primary_coverage,
        device=primary_heatmaps.device,
        dtype=primary_heatmaps.dtype,
    )
    if coverage.ndim == 3 and coverage.shape[0] == 1:
        coverage = coverage[0]
    if coverage.shape != primary_heatmaps.shape[-2:]:
        raise ValueError("primary coverage must align with heatmap pixels")
    if not bool(torch.isfinite(coverage).all()):
        raise FloatingPointError("primary coverage is non-finite")
    return primary_heatmaps + fallback_heatmaps * (1.0 - coverage.clamp(0.0, 1.0))


def blend_primary_with_dominant_fallback(
    primary_heatmaps: torch.Tensor,
    fallback_heatmaps: torch.Tensor,
    primary_coverage: torch.Tensor,
    fallback_coverage: torch.Tensor,
) -> torch.Tensor:
    """Route fallback support only to pixels where its geometry dominates."""
    if primary_heatmaps.shape != fallback_heatmaps.shape:
        raise ValueError("primary and fallback heatmaps must have equal shape")
    coverages = []
    for name, value in (
        ("primary", primary_coverage),
        ("fallback", fallback_coverage),
    ):
        coverage = torch.as_tensor(
            value,
            device=primary_heatmaps.device,
            dtype=primary_heatmaps.dtype,
        )
        if coverage.ndim == 3 and coverage.shape[0] == 1:
            coverage = coverage[0]
        if coverage.shape != primary_heatmaps.shape[-2:]:
            raise ValueError(f"{name} coverage must align with heatmap pixels")
        if not bool(torch.isfinite(coverage).all()):
            raise FloatingPointError(f"{name} coverage is non-finite")
        coverages.append(coverage)
    fallback_gate = (coverages[1] > coverages[0]).to(primary_heatmaps.dtype)
    return primary_heatmaps + fallback_heatmaps * fallback_gate


def blend_primary_first(
    primary_heatmaps: torch.Tensor,
    fallback_heatmaps: torch.Tensor,
    *,
    semantic_threshold: float,
) -> torch.Tensor:
    """Use completion only for queries with no positive primary support.

    The fixed relevancy decision threshold is already part of the declared
    evaluation/readout protocol.  Reusing it here adds no fitted parameter and
    makes every supported query exactly follow the established primary path.
    """
    if primary_heatmaps.shape != fallback_heatmaps.shape:
        raise ValueError("primary and fallback heatmaps must have equal shape")
    if primary_heatmaps.ndim != 3:
        raise ValueError("query heatmaps must be [Q,H,W]")
    primary_supported = primary_heatmaps.amax(dim=(-2, -1)) >= float(
        semantic_threshold
    )
    completed = primary_heatmaps + fallback_heatmaps
    return torch.where(
        primary_supported[:, None, None], primary_heatmaps, completed
    )


def blend_strongest_source(
    primary_heatmaps: torch.Tensor,
    fallback_heatmaps: torch.Tensor,
) -> torch.Tensor:
    """Complete a query only when fallback has stronger peak evidence."""
    if primary_heatmaps.shape != fallback_heatmaps.shape:
        raise ValueError("primary and fallback heatmaps must have equal shape")
    if primary_heatmaps.ndim != 3:
        raise ValueError("query heatmaps must be [Q,H,W]")
    fallback_stronger = fallback_heatmaps.amax(dim=(-2, -1)) > primary_heatmaps.amax(
        dim=(-2, -1)
    )
    completed = primary_heatmaps + fallback_heatmaps
    return torch.where(
        fallback_stronger[:, None, None], completed, primary_heatmaps
    )


def validate_primitive_support_cache(
    payload: Mapping[str, Any],
    model_xyz: torch.Tensor,
    categories: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate a solved, row-aligned primitive support cache.

    The strict contract prevents an unary score cache, a cache with a different
    query list, or a cache from different Gaussian geometry from silently
    entering the paper-facing ``primitive_support`` readout.
    """
    scores = payload.get("query_scores", payload.get("features"))
    valid = payload.get("valid")
    cached_xyz = payload.get("xyz")
    metadata = dict(payload.get("metadata", {}))
    query_names = [str(value) for value in metadata.get("query_names", [])]
    expected_shape = (int(model_xyz.shape[0]), len(categories))
    if not isinstance(scores, torch.Tensor) or tuple(scores.shape) != expected_shape:
        raise ValueError(f"Primitive support scores must be {expected_shape}")
    if not isinstance(valid, torch.Tensor) or tuple(valid.shape) != (expected_shape[0],):
        raise ValueError("Primitive support cache requires row-aligned valid")
    if not isinstance(cached_xyz, torch.Tensor) or cached_xyz.shape != model_xyz.shape:
        raise ValueError("Primitive support cache requires row-aligned xyz")
    if query_names != list(categories):
        raise ValueError(
            f"Primitive support query order mismatch: {query_names} vs {categories}"
        )
    if metadata.get("construction") != "shared_3d_support_solver_probabilities":
        raise ValueError(
            "primitive_support requires shared_3d_support_solver_probabilities"
        )
    xyz_error = (cached_xyz.float() - model_xyz.detach().cpu().float()).norm(dim=-1)
    if xyz_error.numel() and float(xyz_error.max()) > 1e-6:
        raise ValueError(
            f"Primitive support cache xyz mismatch: max_l2={float(xyz_error.max()):.3e}"
        )
    values = scores.float().cpu()
    mask = valid.bool().cpu()
    if bool(mask.any()):
        supported = values[mask]
        if not bool(torch.isfinite(supported).all()):
            raise ValueError("Primitive support cache contains non-finite probabilities")
        if float(supported.min()) < 0.0 or float(supported.max()) > 1.0:
            raise ValueError("Primitive support probabilities must lie in [0,1]")
    return values, mask


def validate_primitive_unary_cache(
    payload: Mapping[str, Any],
    model_xyz: torch.Tensor,
    categories: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate frozen query-set-invariant primitive text unaries."""

    scores = payload.get("query_scores", payload.get("features"))
    valid = payload.get("valid")
    cached_xyz = payload.get("xyz")
    metadata = dict(payload.get("metadata", {}))
    expected_shape = (int(model_xyz.shape[0]), len(categories))
    if not isinstance(scores, torch.Tensor) or tuple(scores.shape) != expected_shape:
        raise ValueError(f"Primitive unary scores must be {expected_shape}")
    if not isinstance(valid, torch.Tensor) or tuple(valid.shape) != (expected_shape[0],):
        raise ValueError("Primitive unary cache requires row-aligned valid")
    if not isinstance(cached_xyz, torch.Tensor) or cached_xyz.shape != model_xyz.shape:
        raise ValueError("Primitive unary cache requires row-aligned xyz")
    if [str(value) for value in metadata.get("query_names", [])] != list(categories):
        raise ValueError("Primitive unary query order mismatch")
    if metadata.get("feature_space") != "primitive_text_query_scores":
        raise ValueError("primitive_unary requires primitive_text_query_scores")
    if metadata.get("scoring") != "cosine":
        raise ValueError("paper-facing primitive_unary requires independent cosine")
    xyz_error = (cached_xyz.float() - model_xyz.detach().cpu().float()).norm(dim=-1)
    if xyz_error.numel() and float(xyz_error.max()) > 1e-6:
        raise ValueError(
            f"Primitive unary cache xyz mismatch: max_l2={float(xyz_error.max()):.3e}"
        )
    values = scores.float().cpu(); mask = valid.bool().cpu()
    if bool(mask.any()):
        selected = values[mask]
        if not bool(torch.isfinite(selected).all()):
            raise ValueError("Primitive unary cache contains non-finite scores")
        if float(selected.min()) < -1.0001 or float(selected.max()) > 1.0001:
            raise ValueError("Cosine primitive unaries must lie in [-1,1]")
    return values, mask


def validate_primitive_posterior_cache(
    payload: Mapping[str, Any],
    model_xyz: torch.Tensor,
    categories: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate a typed Gaussian Query Posterior for scalar rendering."""

    scores = payload.get("query_scores")
    valid = payload.get("valid")
    cached_xyz = payload.get("xyz")
    metadata = dict(payload.get("metadata", {}))
    expected_shape = (int(model_xyz.shape[0]), len(categories))
    if not isinstance(scores, torch.Tensor) or tuple(scores.shape) != expected_shape:
        raise ValueError(f"Primitive posterior scores must be {expected_shape}")
    if not isinstance(valid, torch.Tensor) or tuple(valid.shape) != (expected_shape[0],):
        raise ValueError("Primitive posterior cache requires row-aligned valid")
    if not isinstance(cached_xyz, torch.Tensor) or cached_xyz.shape != model_xyz.shape:
        raise ValueError("Primitive posterior cache requires row-aligned xyz")
    if [str(value) for value in metadata.get("query_names", [])] != list(categories):
        raise ValueError("Primitive posterior query order mismatch")
    if metadata.get("query_family") != "text_object_extent":
        raise ValueError("primitive_posterior requires text_object_extent query family")
    typed_posterior = str(metadata.get("typed_posterior", ""))
    if not (
        typed_posterior.startswith(
            "official_sam3_siglip2_identity_extent_factorization_"
        )
        or typed_posterior.startswith(
            "object_aware_universal_field_v2_text_object_posterior_"
        )
    ):
        raise ValueError("primitive_posterior method identity differs")
    if metadata.get("persistent_second_semantic_field") is not False:
        raise ValueError("primitive_posterior introduced a persistent semantic field")
    if any(
        bool(metadata.get(key, False))
        for key in (
            "benchmark_images_opened",
            "benchmark_masks_opened",
            "evaluation_rgb_opened",
        )
    ):
        raise ValueError("primitive_posterior opened forbidden evaluation information")
    xyz_error = (cached_xyz.float() - model_xyz.detach().cpu().float()).norm(dim=-1)
    if xyz_error.numel() and float(xyz_error.max()) > 1e-6:
        raise ValueError(
            f"Primitive posterior xyz mismatch: max_l2={float(xyz_error.max()):.3e}"
        )
    values = scores.float().cpu()
    mask = valid.bool().cpu()
    if bool(mask.any()):
        selected = values[mask]
        if not bool(torch.isfinite(selected).all()):
            raise ValueError("Primitive posterior contains non-finite probabilities")
        if float(selected.min()) < 0.0 or float(selected.max()) > 1.0:
            raise ValueError("Primitive posterior probabilities must lie in [0,1]")
    return values, mask


def validate_primitive_posterior_identity_cache(
    payload: Mapping[str, Any],
    model_xyz: torch.Tensor,
    categories: List[str],
) -> Optional[torch.Tensor]:
    """Validate the identity half of an identity/extent typed posterior.

    Older extent-only caches remain readable and return ``None``.  A v3 cache
    must carry a row-aligned identity posterior so localization never uses the
    flat interior of an instance mask as an accidental identity score.
    """

    metadata = dict(payload.get("metadata", {}))
    identity = payload.get("identity_query_scores")
    if identity is None:
        if metadata.get("separate_identity_localization") is True:
            raise ValueError("separated primitive posterior lacks identity scores")
        return None
    expected_shape = (int(model_xyz.shape[0]), len(categories))
    if not isinstance(identity, torch.Tensor) or tuple(identity.shape) != expected_shape:
        raise ValueError(f"Primitive identity scores must be {expected_shape}")
    if metadata.get("separate_identity_localization") is not True:
        raise ValueError("primitive identity scores lack separated-output contract")
    if metadata.get("localization_authority") not in {
        "field_siglip2_relevancy_identity",
        "direct_source_view_language_response_exact_mpr",
    }:
        raise ValueError("primitive localization authority differs")
    values = identity.float().cpu()
    valid = torch.as_tensor(payload.get("valid")).bool().cpu()
    if bool(valid.any()):
        selected = values[valid]
        if not bool(torch.isfinite(selected).all()):
            raise ValueError("Primitive identity posterior contains non-finite values")
        if float(selected.min()) < 0.0 or float(selected.max()) > 1.0:
            raise ValueError("Primitive relevancy identity must lie in [0,1]")
    return values


# ---------------------------------------------------------------------------
# Text-embedding generation (SigLIP2 via ``transformers``)
# ---------------------------------------------------------------------------
_SIGLIP2_MODEL_NAME = "google/siglip2-giant-opt-patch16-384"
_SIGLIP2_TEXT_CANONICALIZATION = "official_c_radio_siglip2_g"


def _canonicalize_siglip2_text(text: str) -> str:
    """Match the official C-RADIO ``siglip2-g`` text pre-processing.

    NVIDIA's released adaptor lower-cases text, removes ASCII punctuation,
    replaces underscores with spaces, and collapses whitespace before the
    official SigLIP2 tokenizer is called.  Keeping that operation here makes
    the lightweight frozen-cache path equivalent to the adaptor's tokenizer
    instead of relying on tokenizer-version-specific implicit behaviour.
    """

    value = str(text).replace("_", " ")
    value = value.translate(str.maketrans("", "", string.punctuation)).lower()
    return " ".join(value.split()).strip()


def _restore_siglip2_text_head_from_state(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> bool:
    """Restore SigLIP2's 1536d text projection head when transformers builds 1152d.

    Some transformers versions instantiate ``google/siglip2-giant-opt-patch16-384``
    with ``text_model.head`` shaped 1152→1152 even though the checkpoint and
    config define a 1152→1536 text-aligned head.  That silently breaks text
    embeddings if loaded with ``ignore_mismatched_sizes=True``.
    """
    text_model = getattr(model, "text_model", None)
    head = getattr(text_model, "head", None)
    text_config = getattr(getattr(model, "config", None), "text_config", None)
    if text_model is None or head is None or text_config is None:
        return False

    weight = state.get("text_model.head.weight")
    bias = state.get("text_model.head.bias")
    if weight is None or weight.ndim != 2:
        return False

    projection_size = int(getattr(text_config, "projection_size", weight.shape[0]))
    hidden_size = int(getattr(text_config, "hidden_size", weight.shape[1]))
    if tuple(weight.shape) != (projection_size, hidden_size):
        return False
    if bias is not None and tuple(bias.shape) != (projection_size,):
        return False

    head_weight = getattr(head, "weight", None)
    device = head_weight.device if head_weight is not None else torch.device("cpu")
    dtype = head_weight.dtype if head_weight is not None else weight.dtype
    restored = nn.Linear(hidden_size, projection_size, bias=bias is not None)
    restored = restored.to(device=device, dtype=dtype)
    with torch.no_grad():
        restored.weight.copy_(weight.to(device=device, dtype=dtype))
        if bias is not None and restored.bias is not None:
            restored.bias.copy_(bias.to(device=device, dtype=dtype))
    text_model.head = restored
    return True


def _load_siglip2_text_head_state(model_name: str) -> Optional[Dict[str, torch.Tensor]]:
    """Load only SigLIP2 text-head tensors from local/HF safetensors shards."""
    try:
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
    except Exception:
        return None

    patterns = [
        "model.safetensors",
        "model-*.safetensors",
        "model.safetensors.index.json",
    ]
    try:
        # Prefer the already verified local shards.  Requiring a Hub metadata
        # request here made exact-query evaluation fail during transient TLS
        # outages even though every necessary tensor was present on disk.
        snapshot = Path(
            snapshot_download(
                model_name,
                allow_patterns=patterns,
                local_files_only=True,
            )
        )
    except Exception:
        try:
            snapshot = Path(
                snapshot_download(model_name, allow_patterns=patterns)
            )
        except Exception as exc:
            logger.warning(
                "Could not locate SigLIP2 safetensors for text-head restore: %s", exc
            )
            return None

    state: Dict[str, torch.Tensor] = {}
    for shard in sorted(snapshot.glob("*.safetensors")):
        try:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for key in ("text_model.head.weight", "text_model.head.bias"):
                    if key in handle.keys() and key not in state:
                        state[key] = handle.get_tensor(key)
        except Exception as exc:
            logger.warning("Could not read SigLIP2 safetensors shard %s: %s", shard, exc)
    return state or None


def _load_siglip2_model_for_text(model_name: str, device: torch.device) -> nn.Module:
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_name)
    text_config = getattr(config, "text_config", None)
    projection_size = int(getattr(text_config, "projection_size", 0) or 0)
    hidden_size = int(getattr(text_config, "hidden_size", 0) or 0)
    needs_head_restore = projection_size and hidden_size and projection_size != hidden_size

    if needs_head_restore:
        model = AutoModel.from_pretrained(model_name, ignore_mismatched_sizes=True)
        state = _load_siglip2_text_head_state(model_name)
        if state is None or not _restore_siglip2_text_head_from_state(model, state):
            raise RuntimeError(
                "SigLIP2 text head has mismatched projection size and could not be "
                "restored from checkpoint safetensors."
            )
    else:
        model = AutoModel.from_pretrained(model_name)
    return model.to(device).eval()


def _resolve_siglip2_text_max_length(config: object) -> int:
    text_config = getattr(config, "text_config", None)
    max_length = int(getattr(text_config, "max_position_embeddings", 0) or 0)
    return max_length if max_length > 0 else 64


def _tokenize_siglip2_text(
    queries: List[str],
    model_name: str,
) -> Dict[str, torch.Tensor]:
    canonical_queries = [_canonicalize_siglip2_text(query) for query in queries]
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name)
        inputs = processor(
            text=canonical_queries,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
    except Exception as exc:
        logger.warning("AutoProcessor text tokenization failed, falling back to AutoTokenizer: %s", exc)
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        inputs = tokenizer(
            canonical_queries,
            padding="max_length",
            truncation=True,
            max_length=_resolve_siglip2_text_max_length(config),
            return_tensors="pt",
        )
    return {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}


@torch.no_grad()
def encode_text_siglip2(
    queries: List[str],
    device: torch.device,
    model_name: str = _SIGLIP2_MODEL_NAME,
) -> torch.Tensor:
    """Encode text queries into normalised SigLIP2 embeddings.

    Returns:
        ``[N, 1536]`` L2-normalised float32 tensor on *device*.

    Raises:
        ``RuntimeError`` when the ``transformers`` model cannot be loaded.
    """
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The `transformers` library is required to generate SigLIP2 text "
            "embeddings.  Install with: pip install transformers"
        ) from exc

    model = _load_siglip2_model_for_text(model_name, device)
    inputs = _tokenize_siglip2_text(queries, model_name)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    text_emb = model.get_text_features(**inputs)  # [N, D]
    text_emb = F.normalize(text_emb.float(), dim=-1)
    return text_emb


def load_or_generate_text_embeddings(
    queries: List[str],
    device: torch.device,
    cache_path: Optional[str] = None,
) -> torch.Tensor:
    """Generate or load cached SigLIP2 text embeddings for *queries*.

    Tries on-the-fly encoding first.  If that fails (model not downloaded,
    no GPU memory, etc.) and a *cache_path* exists, falls back to the cached
    bank – but **only** if every query is present.
    """
    # Prefer an exact cache hit. Loading SigLIP2 text towers is expensive and
    # can occupy the same GPU we want to use for rendering.
    if cache_path and Path(cache_path).exists():
        data = torch.load(cache_path, map_location="cpu")
        bank = {q: e for q, e in zip(data["queries"], data["embeddings"])}
        missing = [q for q in queries if q not in bank]
        cached_templates = [str(value) for value in data.get("prompt_templates", [])]
        template_compatible = not cached_templates or cached_templates == ["{query}"]
        encoder_compatible = str(data.get("text_encoder", "siglip2")) == "siglip2"
        if not missing and template_compatible and encoder_compatible:
            emb = torch.stack([bank[q] for q in queries])
            return F.normalize(emb.float(), dim=-1).to(device)

    # Try on-the-fly generation
    try:
        emb = encode_text_siglip2(queries, device)
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "queries": queries,
                    "prompt_templates": ["{query}"],
                    "text_encoder": "siglip2",
                    "model_name": _SIGLIP2_MODEL_NAME,
                    "text_canonicalization": _SIGLIP2_TEXT_CANONICALIZATION,
                    "embeddings": emb.cpu(),
                },
                cache_path,
            )
            logger.info("Cached text embeddings → %s", cache_path)
        return emb
    except Exception as exc:
        logger.warning("On-the-fly SigLIP2 text encoding failed: %s", exc)

    # Fallback: pre-computed bank
    if cache_path and Path(cache_path).exists():
        data = torch.load(cache_path, map_location="cpu")
        bank = {q: e for q, e in zip(data["queries"], data["embeddings"])}
        missing = [q for q in queries if q not in bank]
        cached_templates = [str(value) for value in data.get("prompt_templates", [])]
        if missing or (cached_templates and cached_templates != ["{query}"]):
            raise RuntimeError(
                f"Cached text-embedding bank at {cache_path} is incompatible "
                f"with raw {{query}} evaluation (missing={missing}, "
                f"prompt_templates={cached_templates or '<missing>'})."
            )
        emb = torch.stack([bank[q] for q in queries])
        return F.normalize(emb.float(), dim=-1).to(device)

    raise RuntimeError(
        "Cannot generate SigLIP2 text embeddings and no cache is available.  "
        "Install `transformers` and ensure network access, or provide a "
        "pre-computed bank via --text_embedding_cache."
    )


def resolve_lerf_label_dir(label_dir: str) -> str:
    """Resolve the new LERF-OVS label root with legacy fallback."""
    requested = Path(label_dir)
    if requested.exists():
        return str(requested)
    if str(label_dir) == DEFAULT_LABEL_DIR and Path(LEGACY_LABEL_DIR).exists():
        logger.warning(
            "Default LERF label dir missing (%s); falling back to legacy %s",
            DEFAULT_LABEL_DIR,
            LEGACY_LABEL_DIR,
        )
        return LEGACY_LABEL_DIR
    return str(requested)


def resolve_lerf_scene_root(scene: str, raw_root: str | Path) -> Path:
    """Resolve scene root across the new 3d_understanding layout and legacy paths."""
    raw_root = Path(raw_root) if raw_root else Path()
    candidates = []
    if raw_root:
        candidates.append(raw_root)
        candidates.append(raw_root / scene)
        candidates.append(raw_root.parent / scene)
    candidates.extend(
        [
            Path("/mnt/pool/sqy/3d_understanding/lerf_ovs") / scene,
            Path("/mnt/pool/sqy/lerf_ovs") / scene,
            Path("dataset") / "lerf" / scene,
        ]
    )
    for candidate in candidates:
        if candidate.exists() and ((candidate / "sparse").exists() or (candidate / "transforms.json").exists()):
            return candidate
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_prompt_templates(raw: str | List[str] | Tuple[str, ...] | None) -> List[str]:
    """Parse prompt templates from a pipe/comma separated CLI value."""
    if raw is None:
        return ["{query}"]
    if isinstance(raw, (list, tuple)):
        templates = [str(item).strip() for item in raw if str(item).strip()]
    else:
        sep = "|" if "|" in raw else ","
        templates = [part.strip() for part in str(raw).split(sep) if part.strip()]
    return templates or ["{query}"]


def build_prompt_variants(query: str, templates: List[str]) -> List[str]:
    """Expand one class/query string through prompt templates."""
    variants: List[str] = []
    for template in templates:
        if "{query}" in template:
            text = template.replace("{query}", query)
        elif "{}" in template:
            text = template.format(query)
        else:
            text = f"{template} {query}".strip()
        variants.append(text)
    return variants


def load_or_generate_prompt_ensemble_embeddings(
    queries: List[str],
    device: torch.device,
    cache_path: Optional[str] = None,
    prompt_templates: Optional[List[str]] = None,
) -> torch.Tensor:
    """Encode prompt ensembles and average them into one embedding per query."""
    templates = prompt_templates or ["{query}"]
    if len(templates) == 1 and templates[0] == "{query}":
        return load_or_generate_text_embeddings(queries, device, cache_path)

    def _load_cache() -> Optional[torch.Tensor]:
        if not cache_path or not Path(cache_path).exists():
            return None
        data = torch.load(cache_path, map_location="cpu")
        cached_queries = [str(q) for q in data.get("queries", [])]
        cached_templates = [str(t) for t in data.get("prompt_templates", ["{query}"])]
        if cached_queries == list(queries) and cached_templates == list(templates):
            return F.normalize(data["embeddings"].float(), dim=-1).to(device)
        return None

    cached = _load_cache()
    if cached is not None:
        return cached

    try:
        flat_prompts: List[str] = []
        for query in queries:
            flat_prompts.extend(build_prompt_variants(query, templates))
        flat_emb = encode_text_siglip2(flat_prompts, device)
        emb = flat_emb.reshape(len(queries), len(templates), -1).mean(dim=1)
        emb = F.normalize(emb.float(), dim=-1)
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "queries": queries,
                    "prompt_templates": templates,
                    "text_encoder": "siglip2",
                    "model_name": _SIGLIP2_MODEL_NAME,
                    "text_canonicalization": _SIGLIP2_TEXT_CANONICALIZATION,
                    "embeddings": emb.cpu(),
                },
                cache_path,
            )
            logger.info("Cached prompt-ensemble text embeddings → %s", cache_path)
        return emb
    except Exception as exc:
        logger.warning("Prompt-ensemble SigLIP2 text encoding failed: %s", exc)

    raise RuntimeError(
        "Cannot generate prompt-ensemble SigLIP2 text embeddings and no matching "
        "cache is available."
    )


def validate_text_embedding_cache_contract(
    cache_path: str | Path,
    *,
    required_queries: List[str],
    prompt_templates: List[str],
    text_encoder: str,
    model_name: str,
    pretrained: str = "",
    exact_queries: bool = False,
) -> None:
    """Fail closed when a frozen-protocol text cache lacks provenance metadata."""
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Frozen-protocol text cache not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("embeddings"), torch.Tensor):
        raise ValueError(f"Invalid text embedding cache payload: {path}")
    cached_queries = [str(value) for value in payload.get("queries", [])]
    required = [str(value) for value in required_queries]
    if exact_queries:
        if cached_queries != required:
            raise ValueError(
                f"Frozen cache query mismatch for {path}: expected exact {required}, "
                f"got {cached_queries}"
            )
    else:
        missing = [query for query in required if query not in cached_queries]
        if missing:
            raise ValueError(f"Frozen cache {path} is missing queries: {missing}")
    cached_templates = [str(value) for value in payload.get("prompt_templates", [])]
    if cached_templates != [str(value) for value in prompt_templates]:
        raise ValueError(
            f"Frozen cache prompt-template mismatch for {path}: "
            f"expected {prompt_templates}, got {cached_templates or '<missing>'}"
        )
    if str(payload.get("text_encoder", "")) != str(text_encoder):
        raise ValueError(
            f"Frozen cache text_encoder mismatch for {path}: "
            f"expected {text_encoder!r}, got {payload.get('text_encoder')!r}"
        )
    if str(payload.get("model_name", "")) != str(model_name):
        raise ValueError(
            f"Frozen cache model_name mismatch for {path}: "
            f"expected {model_name!r}, got {payload.get('model_name')!r}"
        )
    if pretrained and str(payload.get("pretrained", "")) != str(pretrained):
        raise ValueError(
            f"Frozen cache pretrained mismatch for {path}: "
            f"expected {pretrained!r}, got {payload.get('pretrained')!r}"
        )
    if int(payload["embeddings"].shape[0]) != len(cached_queries):
        raise ValueError(f"Frozen cache row/query count mismatch: {path}")


# ---------------------------------------------------------------------------
# LERF-OVS label loading
# ---------------------------------------------------------------------------

def load_lerf_ovs_labels(
    label_dir: str,
    scene: str,
) -> Tuple[Dict[int, List[dict]], List[str], int, int]:
    """Load LERF-OVS JSON polygon labels for a scene.

    Returns:
        frame_annotations: ``{frame_id: [{"category": str, "polygons": [np.ndarray]}]}``
        categories:        sorted unique category list
        img_h, img_w:      image dimensions from the annotation ``info`` block
    """
    scene_dir = Path(label_dir) / scene
    if not scene_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {scene_dir}")

    json_files = sorted(scene_dir.glob("frame_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No frame_*.json labels in {scene_dir}")

    frame_annotations: Dict[int, List[dict]] = {}
    all_categories: set[str] = set()
    img_h, img_w = 0, 0

    for jf in json_files:
        frame_id = int(jf.stem.split("_")[1])
        data = _load_annotation_json(jf)
        info = data.get("info", {})
        img_h = int(info.get("height", img_h))
        img_w = int(info.get("width", img_w))

        objects = []
        for obj in data.get("objects", []):
            cat = str(obj.get("category", "")).strip()
            if not cat:
                continue
            polys = _coerce_polygons(obj.get("segmentation"))
            if polys:
                bbox = np.asarray(obj.get("bbox", []), dtype=np.float32).reshape(-1)
                objects.append(
                    {
                        "category": cat,
                        "polygons": polys,
                        "bbox": bbox[:4] if bbox.size >= 4 else None,
                    }
                )
                all_categories.add(cat)
        frame_annotations[frame_id] = objects

    categories = sorted(all_categories)
    return frame_annotations, categories, img_h, img_w


def build_gt_masks(
    frame_objects: List[dict],
    categories: List[str],
    height: int,
    width: int,
    src_height: int = 0,
    src_width: int = 0,
) -> Dict[str, np.ndarray]:
    """Rasterise polygon annotations into per-category binary masks.

    When *src_height*/*src_width* are given and differ from *height*/*width*,
    polygon coordinates (assumed to be in source resolution) are scaled to
    the target resolution before rasterisation.

    Returns:
        ``{category: np.ndarray(H, W, uint8)}`` – 1 inside polygon, 0 outside.
    """
    need_scale = (src_height > 0 and src_width > 0
                  and (src_height != height or src_width != width))
    sx = width / src_width if need_scale else 1.0
    sy = height / src_height if need_scale else 1.0

    masks: Dict[str, np.ndarray] = {}
    for cat in categories:
        mask = np.zeros((height, width), dtype=np.uint8)
        for obj in frame_objects:
            if obj["category"] == cat:
                polys = obj["polygons"]
                if need_scale:
                    polys = [p * np.array([sx, sy]) for p in polys]
                mask = np.maximum(mask, _rasterize_polygons(polys, height, width))
        masks[cat] = mask
    return masks


# ---------------------------------------------------------------------------
# Feature projection & heatmap helpers
# ---------------------------------------------------------------------------

def project_to_siglip2(
    features_1280: torch.Tensor,
    proj_model: nn.Module,
) -> torch.Tensor:
    """Project ``[1, C, H, W]`` to finite FP32, L2-normalised features.

    Projection modules may deliberately run in FP16, but normalization must
    not: the default ``F.normalize`` epsilon underflows to zero in FP16, so an
    all-zero background vector becomes ``0 / 0`` and contaminates an entire
    relevance map with NaNs.  Keep the public readout contract in FP32 and use
    an epsilon representable by both FP16 and FP32.
    """
    B, C, H, W = features_1280.shape
    feat_flat = features_1280.reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, 1280]
    try:
        first_param = next(proj_model.parameters())
    except StopIteration:
        first_param = None
    if first_param is not None and feat_flat.dtype != first_param.dtype:
        feat_flat = feat_flat.to(dtype=first_param.dtype)
    with torch.no_grad():
        siglip = proj_model(feat_flat)  # [B, HW, 1536]
    siglip = siglip.float()
    if not bool(torch.isfinite(siglip).all()):
        raise FloatingPointError("Projection produced non-finite visual features")
    siglip = F.normalize(siglip, dim=-1, eps=1e-8)
    if not bool(torch.isfinite(siglip).all()):
        raise FloatingPointError("Visual feature normalization produced non-finite values")
    return siglip.permute(0, 2, 1).reshape(B, -1, H, W)  # [B, 1536, H, W]


def compute_relevancy_heatmap(
    visual_feat: torch.Tensor,
    text_emb: torch.Tensor,
    canonical_emb: Optional[torch.Tensor] = None,
    temperature: float = 0.01,
    scoring: str = "cosine",
    all_scene_emb: Optional[torch.Tensor] = None,
    active_scene_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    """Cosine-similarity heatmap per query.

    Args:
        visual_feat: ``[1, D, H, W]`` L2-normalised SigLIP2 visual features.
        text_emb:    ``[N, D]`` L2-normalised text embeddings for active queries.
        canonical_emb: ``[M, D]`` optional canonical phrase embeddings.
        temperature: Softmax temperature scaling factor.
        scoring: ``"cosine"`` (raw similarity), ``"softmax_scene"`` (softmax over
            all scene categories — recommended, matches LangSplat protocol), or
            ``"relevancy"`` (LERF-style canonical normalization).
        all_scene_emb: ``[K, D]`` all scene category embeddings (for softmax_scene).
        active_scene_indices: indices into *all_scene_emb* for the active queries.

    Returns:
        ``[N, H, W]`` similarity maps.
    """
    # Relevance scoring is a probability readout, not part of mixed-precision
    # rendering.  Do every normalization/matmul/softmax in FP32 and reject a
    # corrupt tensor instead of silently reporting zero-valued metrics.
    visual_feat = F.normalize(visual_feat.float(), dim=1, eps=1e-8)
    text_emb = F.normalize(text_emb.float(), dim=-1, eps=1e-8)
    if canonical_emb is not None:
        canonical_emb = F.normalize(canonical_emb.float(), dim=-1, eps=1e-8)
    if all_scene_emb is not None:
        all_scene_emb = F.normalize(all_scene_emb.float(), dim=-1, eps=1e-8)
    inputs = {
        "visual features": visual_feat,
        "text embeddings": text_emb,
        "canonical embeddings": canonical_emb,
        "all-scene embeddings": all_scene_emb,
    }
    for name, tensor in inputs.items():
        if tensor is not None and not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"Non-finite {name} in relevance scoring")

    _, D, H, W = visual_feat.shape
    vis_flat = visual_feat.squeeze(0).reshape(D, H * W)  # [D, HW]

    if scoring == "softmax_scene" and all_scene_emb is not None and active_scene_indices is not None:
        # Softmax over all scene categories (matches LangSplat evaluation protocol)
        all_sim = all_scene_emb @ vis_flat  # [K, HW]
        all_prob = torch.softmax(all_sim * temperature, dim=0)  # [K, HW]
        heatmaps = all_prob[active_scene_indices].reshape(-1, H, W)
        if not bool(torch.isfinite(heatmaps).all()):
            raise FloatingPointError("Non-finite softmax-scene heatmap")
        return heatmaps

    sim = text_emb @ vis_flat  # [N, HW]

    if scoring == "relevancy" and canonical_emb is not None and canonical_emb.shape[0] > 0:
        heatmaps = cosine_relevancy_torch(
            vis_flat.T,
            text_emb,
            canonical_emb,
            logit_scale=1.0 / float(temperature),
            assume_normalized=True,
        ).T.reshape(-1, H, W)
        if not bool(torch.isfinite(heatmaps).all()):
            raise FloatingPointError("Non-finite relevancy heatmap")
        return heatmaps

    heatmaps = sim.reshape(-1, H, W)
    if not bool(torch.isfinite(heatmaps).all()):
        raise FloatingPointError("Non-finite cosine heatmap")
    return heatmaps


def fuse_typed_text_scores(
    primitive_scores: torch.Tensor,
    region_scores: torch.Tensor,
    *,
    region_weight: float = 0.5,
) -> torch.Tensor:
    """Fuse primitive and region probabilities with one global fixed weight."""

    primitive = torch.as_tensor(primitive_scores)
    region = torch.as_tensor(region_scores)
    if primitive.shape != region.shape:
        raise ValueError("primitive and region score maps must have matching shapes")
    weight = float(region_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("region_weight must be in [0,1]")
    return (1.0 - weight) * primitive + weight * region


def apply_readout_confidence_gate(
    heatmaps: torch.Tensor,
    aux_maps: Optional[Dict[str, torch.Tensor]],
    *,
    gate: str = "none",
    gamma: float = 1.0,
) -> torch.Tensor:
    """Apply a rendered feature-field confidence map to query heatmaps.

    The gate is GT-free and query-agnostic: it can only suppress locations that
    the feature field itself predicts as low-quality or poorly visible.  This
    keeps the text-query protocol unchanged while letting trained quality heads
    affect the readout.
    """
    if gate == "none" or aux_maps is None or float(gamma) <= 0:
        return heatmaps

    gate_terms: List[torch.Tensor] = []
    if gate in {"quality", "quality_visibility"} and "quality_logit" in aux_maps:
        gate_terms.append(torch.sigmoid(aux_maps["quality_logit"].float()))
    if gate in {"visibility", "quality_visibility"} and "visibility_logit" in aux_maps:
        gate_terms.append(torch.sigmoid(aux_maps["visibility_logit"].float()))
    if gate == "alpha" and "alpha_map" in aux_maps:
        gate_terms.append(aux_maps["alpha_map"].float().clamp(0.0, 1.0))

    if not gate_terms:
        return heatmaps

    gate_map = gate_terms[0]
    for term in gate_terms[1:]:
        gate_map = gate_map * term
    if gate_map.dim() == 4:
        gate_map = gate_map[0, 0]
    elif gate_map.dim() == 3:
        gate_map = gate_map[0]
    if gate_map.shape[-2:] != heatmaps.shape[-2:]:
        gate_map = F.interpolate(
            gate_map.unsqueeze(0).unsqueeze(0),
            size=tuple(int(v) for v in heatmaps.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    gate_map = gate_map.clamp(1e-6, 1.0).pow(float(gamma)).to(
        device=heatmaps.device,
        dtype=heatmaps.dtype,
    )
    return heatmaps * gate_map.unsqueeze(0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def localization_accuracy(
    heatmap: torch.Tensor,
    gt_mask: np.ndarray,
) -> bool:
    """Check whether the argmax pixel of *heatmap* falls inside *gt_mask*.

    Both operate at the same spatial resolution (feature-resolution or image).
    """
    H, W = heatmap.shape
    flat_idx = heatmap.reshape(-1).argmax().item()
    py, px = divmod(flat_idx, W)

    mH, mW = gt_mask.shape
    if (mH, mW) != (H, W):
        # Scale pixel coordinates when resolutions differ
        py = int(py * mH / H)
        px = int(px * mW / W)

    py = min(py, mH - 1)
    px = min(px, mW - 1)
    return bool(gt_mask[py, px] > 0)


def localization_accuracy_bbox_smoothed(
    heatmap: torch.Tensor,
    bboxes: List[np.ndarray],
    *,
    image_shape: Tuple[int, int],
    kernel_size: int = 30,
) -> bool:
    """VALA/LangSplat localization: box-filtered peak inside any GT bbox."""
    if not bboxes:
        return False
    heat = heatmap.detach().float().cpu().numpy()
    size = max(1, int(kernel_size))
    kernel = np.ones((size, size), dtype=np.float32) / float(size * size)
    smoothed = cv2.filter2D(heat, -1, kernel)
    peak_value = float(smoothed.max())
    peak_yx = np.argwhere(smoothed == peak_value)
    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    heat_h, heat_w = int(smoothed.shape[0]), int(smoothed.shape[1])
    for py, px in peak_yx:
        image_y = float(py) * image_h / max(heat_h, 1)
        image_x = float(px) * image_w / max(heat_w, 1)
        for raw_bbox in bboxes:
            box = np.asarray(raw_bbox, dtype=np.float32).reshape(-1)
            if box.size < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            if (
                min(x1, x2) <= image_x <= max(x1, x2)
                and min(y1, y2) <= image_y <= max(y1, y2)
            ):
                return True
    return False


def heatmap_to_binary_mask(
    heatmap: torch.Tensor,
    *,
    threshold_ratio: float = 0.5,
    threshold_mode: str = "fixed",
    threshold_mean_std_k: float = 1.0,
    threshold_min_ratio: float = 0.0,
    threshold_max_ratio: float = 1.0,
    target_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Binarize a heatmap and optionally resize it to a target H,W."""
    if threshold_mode == "absolute":
        # LERF/LangSplat-style relevance maps are calibrated probabilities.
        # Their published 0.5 rule is an absolute probability threshold, not
        # 0.5 times the maximum response in each evaluated query heatmap.
        source = (heatmap > float(threshold_ratio)).cpu().numpy().astype(np.uint8)
    else:
        hmax = heatmap.max().item()
        if hmax <= 0:
            source = np.zeros(tuple(heatmap.shape), dtype=np.uint8)
        else:
            ratio = resolve_heatmap_threshold_ratio(
                heatmap,
                threshold_ratio,
                mode=threshold_mode,
                mean_std_k=threshold_mean_std_k,
                min_ratio=threshold_min_ratio,
                max_ratio=threshold_max_ratio,
            )
            source = (heatmap > ratio * hmax).cpu().numpy().astype(np.uint8)
    if target_shape is not None and tuple(source.shape) != tuple(target_shape):
        source = cv2.resize(
            source,
            (int(target_shape[1]), int(target_shape[0])),
            interpolation=cv2.INTER_NEAREST,
        )
    return source


def resolve_heatmap_threshold_ratio(
    heatmap: torch.Tensor,
    threshold_ratio: float,
    *,
    mode: str = "fixed",
    mean_std_k: float = 1.0,
    min_ratio: float = 0.0,
    max_ratio: float = 1.0,
) -> float:
    """Return a GT-free peak-relative threshold ratio for one heatmap."""
    fixed = float(threshold_ratio)
    if mode in {"fixed", "absolute"}:
        return fixed
    values = heatmap.detach().float()
    hmax = float(values.max().item()) if values.numel() else 0.0
    if hmax <= 1e-12:
        return fixed
    if mode == "mean_std":
        normalized = (values / hmax).reshape(-1)
        ratio = float(normalized.mean().item()) + float(mean_std_k) * float(normalized.std(unbiased=False).item())
        return float(np.clip(ratio, float(min_ratio), float(max_ratio)))
    raise ValueError(f"Unsupported threshold_mode: {mode}")


def refine_mask_with_rgb_edges(
    rgb_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    iterations: int = 1,
    dilate_pixels: int = 5,
    erode_pixels: int = 2,
) -> np.ndarray:
    """Snap a predicted binary mask to local RGB edges without using GT masks."""
    pred = np.asarray(mask).astype(bool)
    if pred.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {pred.shape}")
    if rgb_bgr.shape[:2] != pred.shape:
        raise ValueError(f"RGB/mask shape mismatch: {rgb_bgr.shape[:2]} vs {pred.shape}")
    if not pred.any() or pred.all():
        return pred.copy()

    dilate_size = max(1, int(dilate_pixels) * 2 + 1)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    pred_u8 = pred.astype(np.uint8)
    support = cv2.dilate(pred_u8, dilate_kernel, iterations=1).astype(bool)
    if int(erode_pixels) > 0:
        erode_size = max(1, int(erode_pixels) * 2 + 1)
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
        sure_fg = cv2.erode(pred_u8, erode_kernel, iterations=1).astype(bool)
    else:
        sure_fg = pred
    if not sure_fg.any():
        sure_fg = pred

    init = np.full(pred.shape, cv2.GC_BGD, dtype=np.uint8)
    init[support] = cv2.GC_PR_FGD
    init[pred] = cv2.GC_PR_FGD
    init[sure_fg] = cv2.GC_FGD
    init[~support] = cv2.GC_BGD
    if not np.any(init == cv2.GC_BGD) or not np.any((init == cv2.GC_FGD) | (init == cv2.GC_PR_FGD)):
        return pred.copy()

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            rgb_bgr,
            init,
            None,
            bgd_model,
            fgd_model,
            max(1, int(iterations)),
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return pred.copy()
    refined = (init == cv2.GC_FGD) | (init == cv2.GC_PR_FGD)
    return refined & support


def heatmap_peak_in_shape(
    heatmap: torch.Tensor,
    target_shape: Tuple[int, int],
) -> Tuple[int, int]:
    """Return the heatmap argmax coordinate scaled into ``target_shape``."""
    H, W = heatmap.shape
    flat_idx = heatmap.reshape(-1).argmax().item()
    py, px = divmod(flat_idx, W)
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    py = min(max(int(py * target_h / max(H, 1)), 0), max(target_h - 1, 0))
    px = min(max(int(px * target_w / max(W, 1)), 0), max(target_w - 1, 0))
    return py, px


def keep_peak_connected_component(
    mask: np.ndarray,
    peak_yx: Tuple[int, int],
) -> np.ndarray:
    """Keep only the predicted connected component containing the query peak."""
    from radio_gs.querying.typed_extent_posterior import (
        PeakAnchoredExtentPolicy,
        apply_peak_anchored_extent,
    )

    return apply_peak_anchored_extent(
        mask,
        peak_yx,
        policy=PeakAnchoredExtentPolicy(
            domain="dense_raster",
            minimum_retained_fraction=0.0,
        ),
    ).mask


def compute_iou(
    heatmap: torch.Tensor,
    gt_mask_np: np.ndarray,
    threshold_ratio: float = 0.5,
    *,
    threshold_mode: str = "fixed",
    threshold_mean_std_k: float = 1.0,
    threshold_min_ratio: float = 0.0,
    threshold_max_ratio: float = 1.0,
    rgb_image: Optional[np.ndarray] = None,
    mask_refinement: str = "none",
    mask_refinement_iters: int = 1,
    mask_refinement_dilate: int = 5,
    mask_refinement_erode: int = 2,
) -> float:
    """IoU between a thresholded heatmap and a GT binary mask.

    Threshold = ``threshold_ratio × max(heatmap)``.  Both inputs must share
    the same spatial resolution.
    """
    gt = gt_mask_np.astype(np.uint8)
    target_shape = tuple(rgb_image.shape[:2]) if rgb_image is not None and mask_refinement != "none" else tuple(gt.shape)
    pred = heatmap_to_binary_mask(
        heatmap,
        threshold_ratio=threshold_ratio,
        threshold_mode=threshold_mode,
        threshold_mean_std_k=threshold_mean_std_k,
        threshold_min_ratio=threshold_min_ratio,
        threshold_max_ratio=threshold_max_ratio,
        target_shape=target_shape,
    )
    if tuple(gt.shape) != tuple(pred.shape):
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
    if mask_refinement in {"peak_component", "peak_component_rgb_grabcut"}:
        pred = keep_peak_connected_component(
            pred,
            heatmap_peak_in_shape(heatmap, tuple(pred.shape)),
        ).astype(np.uint8)
    if mask_refinement in {"rgb_grabcut", "peak_component_rgb_grabcut"} and rgb_image is not None:
        pred = refine_mask_with_rgb_edges(
            rgb_image,
            pred,
            iterations=mask_refinement_iters,
            dilate_pixels=mask_refinement_dilate,
            erode_pixels=mask_refinement_erode,
        ).astype(np.uint8)
    elif mask_refinement not in {"none", "peak_component"}:
        raise ValueError(f"Unsupported mask_refinement: {mask_refinement}")

    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter / union) if union > 0 else 0.0


def build_sam3_prompt_initial_mask(
    heatmap: torch.Tensor,
    *,
    threshold_ratio: float,
    threshold_mode: str,
    threshold_mean_std_k: float,
    threshold_min_ratio: float,
    threshold_max_ratio: float,
    target_shape: Tuple[int, int],
    initial_refinement: str = "none",
) -> np.ndarray:
    """Build the coarse mask that conditions the feature-only SAM3 readout."""
    pred = heatmap_to_binary_mask(
        heatmap,
        threshold_ratio=threshold_ratio,
        threshold_mode=threshold_mode,
        threshold_mean_std_k=threshold_mean_std_k,
        threshold_min_ratio=threshold_min_ratio,
        threshold_max_ratio=threshold_max_ratio,
        target_shape=target_shape,
    ).astype(bool)
    if initial_refinement == "none":
        return pred
    if initial_refinement == "peak_component":
        return keep_peak_connected_component(
            pred,
            heatmap_peak_in_shape(heatmap, tuple(pred.shape)),
        ).astype(bool)
    if initial_refinement == "peak_component_or_seed":
        peak = heatmap_peak_in_shape(heatmap, tuple(pred.shape))
        if pred.any():
            return keep_peak_connected_component(pred, peak).astype(bool)

        # An absolute relevance threshold can legitimately produce no coarse
        # mask even when the query heatmap has a localized maximum.  In that
        # case use only the top ten percent of the heatmap's own dynamic range
        # as a box seed for the official SAM3 decoder.  This is query-driven,
        # fixed across scenes, and never reads an RGB frame or benchmark mask.
        heat_np = (
            heatmap.detach().float().cpu().numpy()
            if isinstance(heatmap, torch.Tensor)
            else np.asarray(heatmap)
        )
        if heat_np.ndim == 3:
            heat_np = heat_np[0]
        heat_np = heat_np.astype(np.float32)
        if heat_np.shape != pred.shape:
            heat_np = cv2.resize(
                heat_np,
                (pred.shape[1], pred.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        finite = np.isfinite(heat_np)
        if not finite.any():
            return pred
        finite_values = heat_np[finite]
        heat_min = float(finite_values.min())
        heat_max = float(finite_values.max())
        if heat_max - heat_min <= 1e-8:
            return pred
        seed_threshold = heat_min + 0.90 * (heat_max - heat_min)
        seed = finite & (heat_np >= seed_threshold)
        return keep_peak_connected_component(seed, peak).astype(bool)
    if initial_refinement == "adaptive_peak":
        peak = keep_peak_connected_component(
            pred,
            heatmap_peak_in_shape(heatmap, tuple(pred.shape)),
        ).astype(bool)
        if not pred.any() or not peak.any() or np.array_equal(pred, peak):
            return pred
        num_labels, _ = cv2.connectedComponents(pred.astype(np.uint8), connectivity=4)
        if num_labels <= 2:
            return pred
        heat_np = heatmap.detach().float().cpu().numpy() if isinstance(heatmap, torch.Tensor) else np.asarray(heatmap)
        if heat_np.ndim == 3:
            heat_np = heat_np[0]
        heat_np = heat_np.astype(np.float32)
        if heat_np.shape != pred.shape:
            heat_np = cv2.resize(
                heat_np,
                (pred.shape[1], pred.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        heat_np = heat_np - float(np.nanmin(heat_np))
        heat_max = float(np.nanmax(heat_np))
        if heat_max > 1e-8:
            heat_np = heat_np / heat_max
        raw_mass = float(heat_np[pred].sum()) if pred.any() else 0.0
        peak_mass = float(heat_np[peak].sum()) if peak.any() else 0.0
        mass_ratio = peak_mass / max(raw_mass, 1e-8)
        area_ratio = float(peak.sum()) / max(float(pred.sum()), 1.0)
        if mass_ratio >= 0.65 and area_ratio >= 0.05:
            return peak
        return pred
    raise ValueError(f"Unsupported SAM3 prompt initial refinement: {initial_refinement}")


def load_prompt_conditioned_mask_head(
    checkpoint_path: str | Path,
    device: torch.device,
) -> Tuple[PromptConditionedMaskHead, Tuple[int, int]]:
    """Load a trained feature-only prompt-conditioned SAM mask head."""

    ckpt = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    state = ckpt.get("prompt_mask_head_state_dict")
    if not isinstance(state, dict):
        raise KeyError(
            f"Checkpoint does not contain prompt_mask_head_state_dict: {checkpoint_path}"
        )
    head = PromptConditionedMaskHead(
        feature_dim=int(ckpt.get("feature_dim", 1280)),
        prompt_dim=int(ckpt.get("prompt_dim", 1536)),
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
    ).to(device)
    head.load_state_dict(state, strict=True)
    head.eval()
    target_size = tuple(int(v) for v in ckpt.get("target_size", (240, 320)))
    if len(target_size) != 2:
        raise ValueError(f"Invalid prompt mask head target size: {target_size}")
    return head.float(), (int(target_size[0]), int(target_size[1]))


# ---------------------------------------------------------------------------
# Rendering pipeline (reuse eval_grounding helpers)
# ---------------------------------------------------------------------------

def _import_render_pipeline():
    """Lazy import of the heavy rendering stack (only needed in rendered mode)."""
    from radio_gs.config import load_config
    from radio_gs.models.hybrid_gaussian import HybridFeatureGaussian
    from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
    from radio_gs.models.hcd_codec import build_feature_codec
    from radio_gs.models.featsharp_3d import FeatSharp3D
    from radio_gs.models.screen_refiner import (
        ScreenSpaceRefiner,
        build_refiner_guide,
        compute_refiner_extra_channels,
    )
    from radio_gs.rendering.feature_renderer import FeatureFieldRenderer

    return {
        "load_config": load_config,
        "HybridFeatureGaussian": HybridFeatureGaussian,
        "ExplicitFeatureGaussian": ExplicitFeatureGaussian,
        "build_feature_codec": build_feature_codec,
        "FeatSharp3D": FeatSharp3D,
        "ScreenSpaceRefiner": ScreenSpaceRefiner,
        "build_refiner_guide": build_refiner_guide,
        "compute_refiner_extra_channels": compute_refiner_extra_channels,
        "FeatureFieldRenderer": FeatureFieldRenderer,
    }


def load_render_pipeline(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    *,
    strict_checkpoint_contract: bool = False,
    load_ply_rgb_features: bool = True,
    expected_checkpoint_sha256: str | None = None,
):
    """Load the trained RADIO-GS model, codec, renderer, and refiner.

    Returns:
        ``(model, codec, renderer, sharpener, refiner, config, is_hybrid)``
    """
    R = _import_render_pipeline()
    config = R["load_config"](config_path)

    architecture = getattr(config, "architecture", "explicit")
    is_hybrid = architecture == "hybrid"
    if is_hybrid:
        latent_dim = getattr(config, "hybrid_latent_dim", 16)
        model = R["HybridFeatureGaussian"](
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
            use_quality_head=getattr(config, "hybrid_quality_head", False),
            use_visibility_head=getattr(config, "hybrid_visibility_head", False),
        )
    else:
        latent_dim = getattr(config, "latent_dim", 64)
        model = R["ExplicitFeatureGaussian"](latent_dim=latent_dim)

    ply_path = getattr(config, "ply_path", "")
    ckpt = None
    if is_hybrid and not load_ply_rgb_features:
        # Feature-only checkpoints already contain every geometry buffer.  Seed
        # matching shapes directly from the trusted state dict so large remote
        # PLY files are not parsed a second time merely to allocate tensors.
        ckpt = load_trusted_checkpoint(
            checkpoint_path,
            map_location="cpu",
            expected_sha256=expected_checkpoint_sha256,
        )
        model_state = ckpt.get("model_state_dict", {})
        buffer_names = (
            "_xyz",
            "_rotation",
            "_scaling",
            "_opacity",
            "_features_dc",
            "_features_rest",
        )
        missing_geometry = [name for name in (*buffer_names, "_latent") if name not in model_state]
        if missing_geometry:
            raise KeyError(
                "Feature-only checkpoint lacks geometry state: "
                + ", ".join(missing_geometry)
            )
        for name in buffer_names:
            setattr(model, name, torch.empty_like(model_state[name], device="cpu"))
        model._latent = nn.Parameter(
            torch.empty_like(model_state["_latent"], device="cpu")
        )
        rest = model_state["_features_rest"]
        model._sh_degree = int(math.sqrt(int(rest.shape[1]) + 1)) - 1 if rest.ndim == 3 else 0
        print(
            f"[HybridFeatureGaussian] Geometry shapes from checkpoint: "
            f"{int(model_state['_xyz'].shape[0])} Gaussians"
        )
    elif ply_path:
        if is_hybrid:
            model.load_from_ply(ply_path, load_rgb_features=True)
        else:
            model.load_from_ply(ply_path)
    model = model.to(device).eval()
    use_2dgs = resolve_use_2dgs(config, ply_path)

    codec = R["build_feature_codec"](
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
    ).to(device).eval()

    fH = getattr(config, "feature_height", 30)
    fW = getattr(config, "feature_width", 40)
    renderer = R["FeatureFieldRenderer"](
        image_height=fH, image_width=fW,
        fx=getattr(config, "fx", 320.0) * fW / getattr(config, "image_width", 640),
        fy=getattr(config, "fy", 320.0) * fH / getattr(config, "image_height", 480),
        cx=getattr(config, "cx", 319.5) * fW / getattr(config, "image_width", 640),
        cy=getattr(config, "cy", 239.5) * fH / getattr(config, "image_height", 480),
        max_channels_per_chunk=getattr(config, "max_channels_per_chunk", 32),
        use_2dgs=use_2dgs,
    ).to(device)

    sharpener = R["FeatSharp3D"](
        mode=getattr(config, "featsharp_mode", "analytical"),
        feature_dim=latent_dim,
        strength=getattr(config, "featsharp_strength", 0.3),
    ).to(device).eval()

    refiner = None
    if getattr(config, "use_refiner", False):
        extra_ch = R["compute_refiner_extra_channels"](
            rgb_guide=getattr(config, "refiner_rgb_guide", False),
            depth_guide=getattr(config, "refiner_depth_guide", False),
            depth_grad=getattr(config, "refiner_depth_grad", False),
            alpha_guide=getattr(config, "refiner_alpha_guide", False),
            boundary_guide=getattr(config, "refiner_boundary_guide", False),
        )
        refiner = R["ScreenSpaceRefiner"](
            latent_dim=latent_dim,
            hidden_dim=getattr(config, "refiner_hidden_dim", 128),
            num_blocks=getattr(config, "refiner_num_blocks", 4),
            dropout=getattr(config, "refiner_dropout", 0.1),
            extra_channels=extra_ch,
            norm_type=getattr(config, "refiner_norm_type", "gn"),
        ).to(device).eval()

    if ckpt is None:
        ckpt = load_trusted_checkpoint(
            checkpoint_path,
            map_location=device,
            expected_sha256=expected_checkpoint_sha256,
        )
    model_status = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    codec_status = codec.load_state_dict(ckpt["codec_state_dict"], strict=False)
    contract = {
        "model_missing_keys": list(model_status.missing_keys),
        "model_unexpected_keys": list(model_status.unexpected_keys),
        "codec_missing_keys": list(codec_status.missing_keys),
        "codec_unexpected_keys": list(codec_status.unexpected_keys),
    }
    contract_errors = [
        key for key, value in contract.items() if isinstance(value, list) and value
    ]

    def _load_optional_module(module: Optional[nn.Module], checkpoint_key: str, prefix: str) -> None:
        if module is None or not module.state_dict():
            contract[f"{prefix}_status"] = "not_applicable"
            return
        if checkpoint_key not in ckpt:
            contract[f"{prefix}_status"] = "missing_state_dict"
            contract_errors.append(f"{prefix}_missing_state_dict")
            return
        status = module.load_state_dict(ckpt[checkpoint_key], strict=False)
        contract[f"{prefix}_status"] = "loaded"
        contract[f"{prefix}_missing_keys"] = list(status.missing_keys)
        contract[f"{prefix}_unexpected_keys"] = list(status.unexpected_keys)
        if status.missing_keys:
            contract_errors.append(f"{prefix}_missing_keys")
        if status.unexpected_keys:
            contract_errors.append(f"{prefix}_unexpected_keys")

    _load_optional_module(sharpener, "sharpener_state_dict", "sharpener")
    _load_optional_module(refiner, "refiner_state_dict", "refiner")
    contract["errors"] = contract_errors
    if strict_checkpoint_contract and contract_errors:
        raise RuntimeError(
            "Frozen evaluator checkpoint/config contract mismatch: "
            + json.dumps(contract, sort_keys=True)
        )
    setattr(config, "checkpoint_contract", contract)

    return model, codec, renderer, sharpener, refiner, config, is_hybrid


def render_1280d(
    model, codec, renderer, sharpener, refiner, viewmat,
    *, is_hybrid=False, config=None, device=None, rgb_image=None,
    return_aux: bool = False,
):
    """Render a single frame's decoded 1280-d features.

    Args:
        viewmat: ``[1, 4, 4]`` world-to-camera matrix.
        rgb_image: Optional ``[1, 3, H, W]`` RGB image for refiner guide.
    """
    from radio_gs.models.screen_refiner import build_refiner_guide

    with torch.no_grad():
        result = renderer.render_features(model, viewmat.squeeze(0))
        aux: Dict[str, torch.Tensor] = {}
        latent = result["feature_map"].unsqueeze(0)  # [1, D, H, W]
        latent = sharpener(latent)

        if refiner is not None:
            # Build RGB guide if config requires it
            rgb_guide = None
            if getattr(config, "refiner_rgb_guide", False) if config else False:
                if rgb_image is not None:
                    _, _, fH, fW = latent.shape
                    rgb_guide = F.interpolate(
                        rgb_image.float(), size=(fH, fW), mode="bilinear", align_corners=False
                    )
            guide = build_refiner_guide(
                result,
                rgb_guide=rgb_guide,
                use_depth_guide=getattr(config, "refiner_depth_guide", False) if config else False,
                use_depth_grad=getattr(config, "refiner_depth_grad", False) if config else False,
                depth_grad_scale=getattr(config, "refiner_depth_grad_scale", 10.0) if config else 10.0,
                use_alpha_guide=getattr(config, "refiner_alpha_guide", False) if config else False,
                use_boundary_guide=getattr(config, "refiner_boundary_guide", False) if config else False,
            )
            latent = refiner(latent, guide=guide)

        if is_hybrid:
            from radio_gs.models.hybrid_gaussian import unproject_depth_to_positions
            depth_map = result["depth_map"].float()
            H, W = depth_map.shape[0], depth_map.shape[1]
            if depth_map.dim() == 2:
                depth_map = depth_map.unsqueeze(0).unsqueeze(0)
            elif depth_map.dim() == 3:
                depth_map = depth_map.unsqueeze(0)
            dH, dW = depth_map.shape[2], depth_map.shape[3]
            position_map = unproject_depth_to_positions(
                depth_map, viewmat.float(), renderer.K.float(), dH, dW,
            )
            xyz = model.get_xyz()
            margin = 0.1
            lo = xyz.min(dim=0).values - margin
            hi = xyz.max(dim=0).values + margin
            extent = (hi - lo).clamp(min=1e-6)
            position_map = (
                (position_map - lo.view(1, 3, 1, 1)) / extent.view(1, 3, 1, 1)
            ).clamp(0, 1)
            decoded_or_aux = model.decode_screen_space(
                latent.float(), position_map, depth_map=depth_map, return_aux=return_aux,
            )
            if return_aux and isinstance(decoded_or_aux, dict):
                latent = decoded_or_aux["features"]
                for key in ("quality_logit", "visibility_logit"):
                    if key in decoded_or_aux:
                        aux[key] = decoded_or_aux[key].detach()
            else:
                latent = decoded_or_aux

        decoded = codec.decode(latent)  # [1, 1280, H, W]
        if return_aux:
            depth_map = result.get("depth_map")
            if depth_map is not None:
                if depth_map.dim() == 2:
                    aux["depth_map"] = depth_map.unsqueeze(0).unsqueeze(0).detach()
                elif depth_map.dim() == 3:
                    aux["depth_map"] = depth_map.unsqueeze(0).detach()
                else:
                    aux["depth_map"] = depth_map.detach()
            alpha_map = result.get("alpha_map")
            if alpha_map is not None:
                if alpha_map.dim() == 2:
                    aux["alpha_map"] = alpha_map.unsqueeze(0).unsqueeze(0).detach()
                elif alpha_map.dim() == 3:
                    aux["alpha_map"] = alpha_map.unsqueeze(0).detach()
                else:
                    aux["alpha_map"] = alpha_map.detach()
    if return_aux:
        return decoded, aux
    return decoded


@torch.inference_mode()
def decode_primitive_query_rows(
    model: nn.Module,
    codec: nn.Module,
    projection: nn.Module,
    *,
    is_hybrid: bool,
    device: torch.device,
    chunk_size: int = 8192,
    store_on_cpu: bool = False,
) -> torch.Tensor:
    """Decode every Gaussian into the frozen text-aligned query space.

    These exact rows are shared by direct 3-D querying and primitive-first 2-D
    rendering; no screen-space decoder or refiner is involved.
    """
    rows: List[torch.Tensor] = []
    num_gaussians = int(model.get_xyz().shape[0])
    try:
        projection_dtype = next(projection.parameters()).dtype
    except StopIteration:
        projection_dtype = torch.float32
    for start in tqdm(
        range(0, num_gaussians, max(1, int(chunk_size))),
        desc="  decode primitive query rows",
        leave=False,
    ):
        stop = min(start + max(1, int(chunk_size)), num_gaussians)
        indices = torch.arange(start, stop, device=device, dtype=torch.long)
        if is_hybrid and hasattr(model, "query_gaussian_points"):
            compact = model.query_gaussian_points(indices)
        else:
            compact = model.get_features()[indices]
        decoded = codec.decode_points(compact.float())
        projected = projection(decoded.unsqueeze(0).to(dtype=projection_dtype)).squeeze(0)
        normalized = F.normalize(projected.float(), dim=-1, eps=1e-8)
        rows.append(normalized.half().cpu() if store_on_cpu else normalized)
        del compact, decoded, projected, normalized
    return torch.cat(rows, dim=0)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _slugify_vis_name(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower()).strip("_")
    return slug or "query"


def save_heatmap_vis(
    heatmaps: Dict[str, np.ndarray],
    gt_masks: Dict[str, np.ndarray],
    frame_id: int,
    out_dir: Path,
    tag: str = "",
    source_label: Optional[str] = None,
    rgb_image: Optional[np.ndarray] = None,
    save_per_query: bool = False,
) -> None:
    """Save a grid visualisation of heatmaps vs GT masks for one frame.

    If ``rgb_image`` is provided, append RGB/overlay columns for qualitative
    inspection.  The image is expected in OpenCV BGR channel order.  Overlay
    panels preserve the RGB aspect ratio by resizing masks/heatmaps to RGB
    resolution instead of resizing the RGB image to feature-map resolution.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cats = [c for c in sorted(heatmaps.keys()) if gt_masks.get(c) is not None and gt_masks[c].any()]
    if not cats:
        return

    suffix = f"_{tag}" if tag else ""
    labels = ["query", "GT mask", source_label or f"{tag or 'feature'} heatmap"]
    if rgb_image is not None:
        labels.extend(["RGB", "GT/RGB", "heatmap/RGB"])
    num_cols = len(labels)

    def add_header(rows: List[np.ndarray]) -> np.ndarray:
        grid = np.concatenate(rows, axis=0)
        col_w = grid.shape[1] // num_cols
        header = np.zeros((28, grid.shape[1], 3), dtype=np.uint8)
        for col, label in enumerate(labels):
            cv2.putText(
                header,
                label,
                (col * col_w + 4, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return np.concatenate([header, grid], axis=0)

    rows = []
    for cat in cats[:8]:  # max 8 rows
        hm = heatmaps[cat]
        mask = gt_masks[cat]
        hmin, hmax = hm.min(), hm.max()
        if hmax - hmin > 1e-6:
            hm_norm = ((hm - hmin) / (hmax - hmin) * 255).astype(np.uint8)
        else:
            hm_norm = np.zeros_like(hm, dtype=np.uint8)
        mask_u8 = (mask > 0).astype(np.uint8)

        if rgb_image is not None:
            target_h, target_w = rgb_image.shape[:2]
            hm_norm = cv2.resize(hm_norm, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            mask_u8 = cv2.resize(mask_u8, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        mask_vis = cv2.applyColorMap((mask_u8 * 255).astype(np.uint8), cv2.COLORMAP_BONE)

        label_img = np.zeros_like(hm_color)
        cv2.putText(label_img, cat, (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        row_parts = [label_img, mask_vis, hm_color]
        if rgb_image is not None:
            rgb_resized = rgb_image.copy()
            mask_rgb = rgb_resized.copy()
            mask_overlay = np.zeros_like(mask_rgb)
            mask_overlay[:, :, 1] = mask_u8 * 255
            mask_rgb = cv2.addWeighted(mask_rgb, 0.65, mask_overlay, 0.35, 0.0)
            heat_rgb = cv2.addWeighted(rgb_resized, 0.55, hm_color, 0.45, 0.0)
            row_parts.extend([rgb_resized, mask_rgb, heat_rgb])
        row = np.concatenate(row_parts, axis=1)
        rows.append(row)

        if save_per_query:
            per_query_grid = add_header([row])
            query_suffix = _slugify_vis_name(cat)
            cv2.imwrite(
                str(out_dir / f"lerf_grounding_frame_{frame_id:05d}{suffix}_{query_suffix}.png"),
                per_query_grid,
            )

    cv2.imwrite(str(out_dir / f"lerf_grounding_frame_{frame_id:05d}{suffix}.png"), add_header(rows))


def save_prediction_masks(
    *,
    out_dir: Path,
    scene: str,
    mode: str,
    frame_id: int,
    query: str,
    gt_mask: np.ndarray,
    final_mask: np.ndarray,
    initial_mask: Optional[np.ndarray] = None,
    rgb_image: Optional[np.ndarray] = None,
) -> None:
    """Save per-query binary masks and lightweight RGB overlays for qualitative figures."""
    query_slug = _slugify_vis_name(query)
    target_dir = out_dir / scene / mode
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{frame_id:05d}_{query_slug}"

    masks: dict[str, np.ndarray] = {
        "gt": gt_mask,
        "final": final_mask,
    }
    if initial_mask is not None:
        masks["initial"] = initial_mask

    for name, mask in masks.items():
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
        cv2.imwrite(str(target_dir / f"{stem}_{name}.png"), mask_u8)

    if rgb_image is None:
        return
    rgb = rgb_image
    for name in ("initial", "final"):
        if name not in masks:
            continue
        mask_u8 = (np.asarray(masks[name]) > 0).astype(np.uint8)
        if mask_u8.shape[:2] != rgb.shape[:2]:
            mask_u8 = cv2.resize(mask_u8, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = (rgb.astype(np.float32) * 0.35).astype(np.uint8)
        color = np.zeros_like(rgb, dtype=np.uint8)
        color[:, :, 1] = mask_u8 * 255
        blended = cv2.addWeighted(overlay, 1.0, color, 0.65, 0.0)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0, 255, 255), 2)
        cv2.imwrite(str(target_dir / f"{stem}_{name}_overlay.png"), blended)


def load_lerf_rgb_frame(scene: str, frame_id: int, scene_root_hint: str | Path = "") -> Optional[np.ndarray]:
    """Load an RGB frame as an OpenCV BGR image for visual overlays."""
    scene_root = resolve_lerf_scene_root(scene, scene_root_hint)
    candidates = [
        scene_root / "images" / f"frame_{frame_id:05d}.jpg",
        scene_root / "images" / f"frame_{frame_id:05d}.png",
        scene_root / f"frame_{frame_id:05d}.jpg",
        scene_root / f"frame_{frame_id:05d}.png",
    ]
    for path in candidates:
        if path.exists():
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                return image
    return None


# ---------------------------------------------------------------------------
# Single-scene evaluation
# ---------------------------------------------------------------------------

def evaluate_scene(
    scene: str,
    label_dir: str,
    proj: nn.Module,
    text_embeddings: torch.Tensor,
    categories: List[str],
    device: torch.device,
    *,
    gt_feature_dir: Optional[str] = None,
    region_feature_dir: Optional[str] = None,
    region_score_weight: float = 0.5,
    render_pipeline: Optional[tuple] = None,
    lerf_dataset: Optional[LERFDataset] = None,
    vis_dir: Optional[Path] = None,
    iou_threshold: float = 0.5,
    threshold_mode: str = "fixed",
    threshold_mean_std_k: float = 1.0,
    threshold_min_ratio: float = 0.0,
    threshold_max_ratio: float = 1.0,
    canonical_emb: Optional[torch.Tensor] = None,
    temperature: float = 50.0,
    scoring: str = "softmax_scene",
    heatmap_upsample: int = 1,
    eval_at_image_resolution: bool = False,
    localization_mode: str = "polygon_argmax",
    localization_smoothing_kernel: int = 30,
    readout_confidence_gate: str = "none",
    readout_confidence_gamma: float = 1.0,
    save_overlay_vis: bool = False,
    save_per_query_vis: bool = False,
    pred_mask_dir: Optional[Path] = None,
    mask_refinement: str = "none",
    rgb_refinement_source: str = "dataset_frame",
    mask_refinement_iters: int = 1,
    mask_refinement_dilate: int = 5,
    mask_refinement_erode: int = 2,
    sam3_box_refiner: Optional[Any] = None,
    sam3_box_initial_refinement: str = "none",
    sam3_box_min_heatmap_mean_ratio: float = 0.0,
    sam3_box_min_heatmap_mass_ratio: float = 0.0,
    sam3_box_require_peak_in_refined: bool = False,
    sam3_prompt_mask_head: Optional[PromptConditionedMaskHead] = None,
    sam3_prompt_mask_head_target_size: Tuple[int, int] = (240, 320),
    sam3_prompt_mask_head_logit_threshold: float = DEFAULT_SAM3_PROMPT_MASK_HEAD_LOGIT_THRESHOLD,
    sam3_prompt_mask_head_min_initial_iou: float = DEFAULT_SAM3_PROMPT_MASK_HEAD_MIN_INITIAL_IOU,
    sam3_prompt_mask_head_max_initial_area_fraction: float = DEFAULT_SAM3_PROMPT_MASK_HEAD_MAX_INITIAL_AREA_FRACTION,
    sam3_prompt_mask_head_min_refined_area_ratio: float = DEFAULT_SAM3_PROMPT_MASK_HEAD_MIN_REFINED_AREA_RATIO,
    sam3_prompt_mask_head_max_refined_area_ratio: float = DEFAULT_SAM3_PROMPT_MASK_HEAD_MAX_REFINED_AREA_RATIO,
    sam3_prompt_mask_head_support_dilate: int = DEFAULT_SAM3_PROMPT_MASK_HEAD_SUPPORT_DILATE,
    sam3_prompt_mask_head_coarse_dilate: int = DEFAULT_SAM3_PROMPT_MASK_HEAD_COARSE_DILATE,
    sam3_prompt_mask_head_coarse_threshold: float = DEFAULT_SAM3_PROMPT_MASK_HEAD_COARSE_THRESHOLD,
    sam3_prompt_mask_head_min_heatmap_mean_ratio: float = 0.0,
    sam3_prompt_mask_head_min_heatmap_mass_ratio: float = 0.0,
    sam3_prompt_mask_head_require_peak_in_refined: bool = False,
    sam3_prompt_mask_head_initial_refinement: str = "none",
    sam3_prompt_mask_head_apply_to: str = "rendered",
    sam3_prompt_mask_head_feature_dir: Optional[str] = None,
    render_readout: str = "screen_decode",
    primitive_chunk_size: int = 8192,
    primitive_query_cache: str = "",
    primitive_score_cache: str = "",
    primitive_confidence: str = "cache",
    primitive_fallback_blend: str = "uncovered",
    feature_contribution_gamma: float = 1.0,
    primitive_valid_normalization: bool = False,
    primitive_valid_coverage_power: float = 0.0,
    primitive_posterior_visibility_mass: bool = False,
    primitive_posterior_calibration_scale: float = 1.0,
    primitive_posterior_calibration_bias: float = 0.0,
) -> Dict:
    """Evaluate one LERF-OVS scene.

    Exactly one of *gt_feature_dir* or *render_pipeline* must be provided (or
    both for a joint GT + rendered evaluation).

    Returns:
        dict with ``loc_acc``, ``miou``, per-category breakdowns, etc.
    """
    frame_annotations, scene_categories, img_h, img_w = load_lerf_ovs_labels(label_dir, scene)
    if (
        region_feature_dir is not None
        and not 0.0 <= float(region_score_weight) <= 1.0
    ):
        raise ValueError("region_score_weight must be in [0,1]")

    # Map each scene category to an index in the global embedding tensor
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    scene_cat_indices = {c: cat_to_idx[c] for c in scene_categories if c in cat_to_idx}

    results: Dict[str, Dict] = {}
    for mode in ("gt", "rendered"):
        if mode == "gt" and gt_feature_dir is None:
            continue
        if mode == "rendered" and render_pipeline is None:
            continue

        loc_correct = 0
        loc_total = 0
        ious: List[float] = []
        initial_ious: List[float] = []
        refinement_reports: List[Dict[str, object]] = []
        per_cat_loc: Dict[str, List[bool]] = {c: [] for c in scene_categories}
        per_cat_iou: Dict[str, List[float]] = {c: [] for c in scene_categories}

        if mode == "rendered":
            model, codec, renderer, sharpener, refiner, config, is_hybrid = render_pipeline
            scene_root_hint = getattr(config, "scene_root", "")
            if lerf_dataset is None:
                raise RuntimeError(f"Missing rendered pose dataset for scene={scene!r}")
            missing_pose_frames = sorted(
                set(frame_annotations) - set(lerf_dataset.pose_by_frame_idx)
            )
            if missing_pose_frames:
                raise RuntimeError(
                    f"Missing camera poses for labeled {scene} frames: {missing_pose_frames}"
                )
            primitive_query_rows = None
            primitive_query_valid = None
            primitive_query_confidence = None
            primitive_query_primary = None
            primitive_support_rows = None
            primitive_support_valid = None
            primitive_localization_rows = None
            if render_readout in {"primitive_query", "primitive_score"}:
                if primitive_query_cache:
                    payload = torch.load(primitive_query_cache, map_location="cpu")
                    primitive_query_rows = payload.get(
                        "summary_features", payload.get("features")
                    )
                    primitive_query_valid = payload.get("valid")
                    primitive_query_primary = payload.get("primary_valid")
                    if primitive_query_primary is not None:
                        primitive_query_primary = torch.as_tensor(
                            primitive_query_primary
                        ).bool()
                        if primitive_query_primary.shape != (
                            model.get_xyz().shape[0],
                        ):
                            raise ValueError(
                                "Primitive query primary mask row-count mismatch"
                            )
                        valid_mask = torch.as_tensor(primitive_query_valid).bool()
                        if bool((primitive_query_primary & ~valid_mask).any()):
                            raise ValueError(
                                "Primitive query primary rows must be valid"
                            )
                    if primitive_confidence == "cache":
                        primitive_query_confidence = payload.get(
                            "semantic_confidence"
                        )
                        if primitive_query_confidence is not None:
                            confidence_shape = tuple(
                                torch.as_tensor(primitive_query_confidence).shape
                            )
                            if confidence_shape != (model.get_xyz().shape[0],):
                                raise ValueError(
                                    "Primitive query confidence row-count mismatch"
                                )
                    cached_xyz = payload.get("xyz")
                    if not isinstance(primitive_query_rows, torch.Tensor):
                        raise ValueError("Primitive query cache lacks feature rows")
                    if primitive_query_rows.shape[0] != model.get_xyz().shape[0]:
                        raise ValueError("Primitive query cache row-count mismatch")
                    if not isinstance(cached_xyz, torch.Tensor) or cached_xyz.shape != model.get_xyz().shape:
                        raise ValueError("Primitive query cache lacks row-aligned xyz")
                    xyz_error = (
                        cached_xyz.float() - model.get_xyz().detach().cpu().float()
                    ).norm(dim=-1)
                    if float(xyz_error.max()) > 1e-6:
                        raise ValueError(
                            f"Primitive query cache xyz mismatch: max_l2={float(xyz_error.max()):.3e}"
                        )
                    if render_readout == "primitive_query":
                        primitive_query_rows = primitive_query_rows.to(
                            device=device, dtype=torch.float32
                        )
                else:
                    primitive_query_rows = decode_primitive_query_rows(
                        model,
                        codec,
                        proj,
                        is_hybrid=is_hybrid,
                        device=device,
                        chunk_size=primitive_chunk_size,
                        store_on_cpu=render_readout == "primitive_score",
                    )
            elif render_readout in {"primitive_support", "primitive_unary", "primitive_posterior"}:
                if not primitive_score_cache:
                    raise ValueError(
                        f"{render_readout} requires --primitive_score_cache"
                    )
                support_payload = torch.load(primitive_score_cache, map_location="cpu")
                if render_readout == "primitive_support":
                    primitive_support_rows, primitive_support_valid = validate_primitive_support_cache(
                        support_payload, model.get_xyz(), categories
                    )
                elif render_readout == "primitive_unary":
                    primitive_support_rows, primitive_support_valid = validate_primitive_unary_cache(
                        support_payload, model.get_xyz(), categories
                    )
                else:
                    primitive_support_rows, primitive_support_valid = validate_primitive_posterior_cache(
                        support_payload, model.get_xyz(), categories
                    )
                    primitive_localization_rows = validate_primitive_posterior_identity_cache(
                        support_payload, model.get_xyz(), categories
                    )
                    if primitive_query_cache:
                        identity_payload = torch.load(
                            primitive_query_cache, map_location="cpu"
                        )
                        primitive_query_rows = identity_payload.get(
                            "summary_features", identity_payload.get("features")
                        )
                        primitive_query_valid = torch.as_tensor(
                            identity_payload.get("valid")
                        ).bool()
                        identity_xyz = identity_payload.get("xyz")
                        if (
                            not isinstance(primitive_query_rows, torch.Tensor)
                            or primitive_query_rows.shape
                            != (model.get_xyz().shape[0], 1536)
                            or primitive_query_valid.shape
                            != (model.get_xyz().shape[0],)
                            or not isinstance(identity_xyz, torch.Tensor)
                            or identity_xyz.shape != model.get_xyz().shape
                        ):
                            raise ValueError(
                                "primitive posterior identity feature cache differs"
                            )
                        identity_xyz_error = (
                            identity_xyz.float()
                            - model.get_xyz().detach().cpu().float()
                        ).norm(dim=-1)
                        if float(identity_xyz_error.max()) > 1e-6:
                            raise ValueError(
                                "primitive posterior identity feature xyz differs"
                            )
        else:
            scene_root_hint = ""
            primitive_query_rows = None
            primitive_query_valid = None
            primitive_query_confidence = None
            primitive_query_primary = None
            primitive_support_rows = None
            primitive_support_valid = None
            primitive_localization_rows = None
        canonical_mode = canonical_lerf_mode(mode)
        mode_mask_refinement = mask_refinement
        if (
            mask_refinement == "sam3_prompt_mask_head"
            and sam3_prompt_mask_head_apply_to == "rendered"
            and canonical_mode != "rendered"
        ):
            mode_mask_refinement = "none"
        if mask_refinement == "sam3_box" and canonical_mode != "rendered":
            mode_mask_refinement = "none"
        rgb_mask_renderer = None
        render_rgb_refinement_frame_fn = None
        if (
            canonical_mode == "rendered"
            and rgb_refinement_source == "rendered"
            and mode_mask_refinement
            in {"rgb_grabcut", "peak_component_rgb_grabcut", "sam3_box"}
        ):
            from radio_gs.scripts.eval_lerf_direct_3d_selection import (
                build_mask_renderer,
                render_rgb_refinement_frame,
            )

            rgb_mask_renderer = build_mask_renderer(
                config,
                height=img_h,
                width=img_w,
                device=device,
            )
            render_rgb_refinement_frame_fn = render_rgb_refinement_frame
        mode_confidence_gate = readout_confidence_gate if canonical_mode == "rendered" else "none"
        for frame_id, frame_objects in tqdm(
            sorted(frame_annotations.items()),
            desc=f"  {scene}/{canonical_mode}",
            leave=False,
        ):
            region_siglip_feat = None
            localization_heatmaps = None
            # --- obtain 1280-d features ---
            if mode == "gt":
                feat_path = Path(gt_feature_dir) / f"rgb_{frame_id}.pt"
                if not feat_path.exists():
                    # Try backbone/ subdirectory
                    feat_path = Path(gt_feature_dir) / "backbone" / f"rgb_{frame_id}.pt"
                if not feat_path.exists():
                    logger.warning("GT features missing for frame %d – skipping", frame_id)
                    continue
                feat_1280 = torch.load(feat_path, map_location=device).float()
                if feat_1280.dim() == 3:
                    feat_1280 = feat_1280.unsqueeze(0)  # [1, 1280, H, W]
            else:  # rendered
                if lerf_dataset is None:
                    logger.warning("No LERFDataset for rendered mode – skipping")
                    continue
                pose_w2c = lerf_dataset.pose_by_frame_idx.get(frame_id)
                if pose_w2c is None:
                    logger.warning("No pose for frame %d – skipping", frame_id)
                    continue
                viewmat = torch.from_numpy(pose_w2c.copy()).float().to(device).unsqueeze(0)
                render_aux = None
                if render_readout == "primitive_query":
                    assert primitive_query_rows is not None
                    primitive_render = renderer.render_feature_rows(
                        model,
                        viewmat.squeeze(0),
                        primitive_query_rows,
                        alpha_normalize=True,
                        contribution_gamma=feature_contribution_gamma,
                    )
                    siglip_feat = F.normalize(
                        primitive_render["feature_map"].unsqueeze(0).float(),
                        dim=1,
                        eps=1e-8,
                    )
                    render_aux = {
                        "alpha_map": primitive_render["alpha_map"][None, None]
                    }
                    feat_1280 = None
                elif render_readout in {"primitive_score", "primitive_support", "primitive_unary", "primitive_posterior"}:
                    if render_readout == "primitive_score":
                        assert primitive_query_rows is not None
                    siglip_feat = None
                    feat_1280 = None
                # Load RGB image for refiner guide if needed
                rgb_tensor = None
                if getattr(config, "refiner_rgb_guide", False):
                    scene_root = resolve_lerf_scene_root(scene, getattr(config, "scene_root", ""))
                    img_path = scene_root / "images" / f"frame_{frame_id:05d}.jpg"
                    if not img_path.exists():
                        img_path = scene_root / "images" / f"frame_{frame_id:05d}.png"
                    if img_path.exists():
                        import torchvision.transforms.functional as TF
                        from PIL import Image
                        rgb_pil = Image.open(img_path).convert("RGB")
                        rgb_tensor = TF.to_tensor(rgb_pil).unsqueeze(0).to(device)
                if render_readout in {
                    "primitive_query",
                    "primitive_score",
                    "primitive_support",
                    "primitive_unary",
                    "primitive_posterior",
                }:
                    pass
                elif mode_confidence_gate != "none":
                    feat_1280, render_aux = render_1280d(
                        model, codec, renderer, sharpener, refiner, viewmat,
                        is_hybrid=is_hybrid, config=config, device=device,
                        rgb_image=rgb_tensor,
                        return_aux=True,
                    )
                else:
                    feat_1280 = render_1280d(
                        model, codec, renderer, sharpener, refiner, viewmat,
                        is_hybrid=is_hybrid, config=config, device=device,
                        rgb_image=rgb_tensor,
                    )

            # Project into the selected text space.  ``proj`` is an identity
            # module for native 512-D OpenCLIP/SAM-CLIP feature fields.
            if not (
                mode == "rendered"
                and render_readout
                in {"primitive_query", "primitive_score", "primitive_support", "primitive_unary", "primitive_posterior"}
            ):
                siglip_feat = project_to_siglip2(feat_1280.half(), proj)
            if mode == "gt" and region_feature_dir is not None:
                region_path = Path(region_feature_dir) / f"rgb_{frame_id}.pt"
                if not region_path.exists():
                    region_path = (
                        Path(region_feature_dir)
                        / "backbone"
                        / f"rgb_{frame_id}.pt"
                    )
                if not region_path.is_file():
                    raise FileNotFoundError(
                        f"typed region feature is missing for frame {frame_id}: "
                        f"{region_path}"
                    )
                region_features = torch.load(
                    region_path, map_location=device
                ).float()
                if region_features.ndim == 3:
                    region_features = region_features[None]
                region_siglip_feat = project_to_siglip2(
                    region_features, nn.Identity().to(device)
                )

            # Only evaluate categories present in this frame
            frame_cats = {obj["category"] for obj in frame_objects}
            active_indices = []
            active_cats = []
            for cat in sorted(frame_cats):
                if cat in scene_cat_indices:
                    active_indices.append(scene_cat_indices[cat])
                    active_cats.append(cat)
            if not active_indices:
                continue

            active_emb = text_embeddings[active_indices].to(device)  # [K, 1536]
            if not (mode == "rendered" and render_readout in {"primitive_support", "primitive_unary", "primitive_posterior"}):
                visual_dim = (
                    int(primitive_query_rows.shape[1])
                    if mode == "rendered" and render_readout == "primitive_score"
                    else int(siglip_feat.shape[1])
                )
                if visual_dim != int(active_emb.shape[1]):
                    raise ValueError(
                        "Visual/text feature dimension mismatch: "
                        f"visual={visual_dim}, text={int(active_emb.shape[1])}. "
                        "Use --text_encoder openclip for a 512-D SAM-CLIP field or "
                        "--text_encoder siglip2 for a RADIO/SigLIP2 field."
                    )

            # Build all-scene embeddings for softmax_scene scoring
            all_scene_emb = None
            active_scene_idx = None
            if scoring == "softmax_scene":
                scene_emb_global_idx = [cat_to_idx[c] for c in sorted(scene_cat_indices.keys())]
                all_scene_emb = text_embeddings[scene_emb_global_idx].to(device)
                scene_cats_sorted = sorted(scene_cat_indices.keys())
                active_scene_idx = [scene_cats_sorted.index(c) for c in active_cats]

            if mode == "rendered" and render_readout == "primitive_score":
                assert primitive_query_rows is not None
                primitive_score_parts: List[torch.Tensor] = []
                for start in range(0, primitive_query_rows.shape[0], primitive_chunk_size):
                    query_chunk = primitive_query_rows[
                        start : start + primitive_chunk_size
                    ].to(device=device, dtype=torch.float32)
                    chunk_heatmaps = compute_relevancy_heatmap(
                        query_chunk.T[None, :, :, None],
                        active_emb,
                        canonical_emb=canonical_emb,
                        temperature=temperature,
                        scoring=scoring,
                        all_scene_emb=all_scene_emb,
                        active_scene_indices=active_scene_idx,
                    )
                    primitive_score_parts.append(chunk_heatmaps.squeeze(-1).T)
                primitive_score_rows = torch.cat(primitive_score_parts, dim=0).contiguous()
                primitive_score_rows = neutralize_invalid_primitive_scores_for_render(
                    primitive_score_rows,
                    primitive_query_valid,
                )
                primitive_score_rows = apply_primitive_semantic_confidence(
                    primitive_score_rows,
                    primitive_query_confidence,
                )
                if (
                    primitive_fallback_blend
                    in {"uncovered", "dominant", "primary_first", "strongest"}
                    and primitive_query_primary is not None
                ):
                    if primitive_valid_normalization:
                        raise ValueError(
                            "valid normalization with split primary/fallback "
                            "rendering requires --primitive_fallback_blend direct"
                        )
                    primary_mask = primitive_query_primary.to(device=device)
                    primary_rows = primitive_score_rows * primary_mask[:, None]
                    fallback_rows = primitive_score_rows * (~primary_mask)[:, None]
                    primary_render = renderer.render_feature_rows(
                        model,
                        viewmat.squeeze(0),
                        primary_rows,
                        alpha_normalize=True,
                        contribution_gamma=feature_contribution_gamma,
                    )
                    fallback_render = renderer.render_feature_rows(
                        model,
                        viewmat.squeeze(0),
                        fallback_rows,
                        alpha_normalize=True,
                        contribution_gamma=feature_contribution_gamma,
                    )
                    if primitive_fallback_blend == "primary_first":
                        heatmaps = blend_primary_first(
                            primary_render["feature_map"].float(),
                            fallback_render["feature_map"].float(),
                            semantic_threshold=float(iou_threshold),
                        )
                    elif primitive_fallback_blend == "strongest":
                        heatmaps = blend_strongest_source(
                            primary_render["feature_map"].float(),
                            fallback_render["feature_map"].float(),
                        )
                    else:
                        coverage_render = renderer.render_feature_rows(
                            model,
                            viewmat.squeeze(0),
                            primary_mask.float()[:, None],
                            alpha_normalize=True,
                            contribution_gamma=feature_contribution_gamma,
                        )
                    if primitive_fallback_blend == "dominant":
                        fallback_coverage_render = renderer.render_feature_rows(
                            model,
                            viewmat.squeeze(0),
                            (~primary_mask).float()[:, None]
                            * torch.as_tensor(
                                primitive_query_valid,
                                device=device,
                                dtype=torch.float32,
                            )[:, None],
                            alpha_normalize=True,
                            contribution_gamma=feature_contribution_gamma,
                        )
                        heatmaps = blend_primary_with_dominant_fallback(
                            primary_render["feature_map"].float(),
                            fallback_render["feature_map"].float(),
                            coverage_render["feature_map"].float(),
                            fallback_coverage_render["feature_map"].float(),
                        )
                    elif primitive_fallback_blend == "uncovered":
                        heatmaps = blend_primary_with_uncovered_fallback(
                            primary_render["feature_map"].float(),
                            fallback_render["feature_map"].float(),
                            coverage_render["feature_map"].float(),
                        )
                    render_aux = {
                        "alpha_map": primary_render["alpha_map"][None, None]
                    }
                else:
                    render_rows = primitive_score_rows
                    if primitive_valid_normalization:
                        if primitive_query_valid is None:
                            raise ValueError(
                                "valid normalization requires primitive query validity"
                            )
                        render_rows = torch.cat(
                            [
                                primitive_score_rows,
                                torch.as_tensor(
                                    primitive_query_valid,
                                    device=device,
                                    dtype=primitive_score_rows.dtype,
                                )[:, None],
                            ],
                            dim=1,
                        )
                    score_render = renderer.render_feature_rows(
                        model,
                        viewmat.squeeze(0),
                        render_rows,
                        alpha_normalize=True,
                        contribution_gamma=feature_contribution_gamma,
                    )
                    if primitive_valid_normalization:
                        heatmaps, semantic_coverage = (
                            normalize_primitive_scores_by_valid_mass(
                                score_render["feature_map"].float(),
                                coverage_power=primitive_valid_coverage_power,
                            )
                        )
                    else:
                        heatmaps = score_render["feature_map"].float()
                        semantic_coverage = None
                    render_aux = {
                        "alpha_map": score_render["alpha_map"][None, None]
                    }
                    if semantic_coverage is not None:
                        render_aux["semantic_coverage"] = semantic_coverage[
                            None, None
                        ]
            elif mode == "rendered" and render_readout in {"primitive_support", "primitive_unary", "primitive_posterior"}:
                assert primitive_support_rows is not None
                support_rows = primitive_support_rows[:, active_indices]
                if render_readout == "primitive_posterior":
                    support_rows = monotonic_logit_calibration(
                        support_rows,
                        scale=primitive_posterior_calibration_scale,
                        bias=primitive_posterior_calibration_bias,
                    )
                support_rows = neutralize_invalid_primitive_scores_for_render(
                    support_rows,
                    primitive_support_valid,
                ).to(device=device, dtype=torch.float32)
                localization_rows = None
                if render_readout == "primitive_posterior":
                    if primitive_query_rows is not None:
                        identity_score_parts: List[torch.Tensor] = []
                        for start in range(
                            0, primitive_query_rows.shape[0], primitive_chunk_size
                        ):
                            identity_chunk = primitive_query_rows[
                                start : start + primitive_chunk_size
                            ].to(device=device, dtype=torch.float32)
                            identity_heatmaps = compute_relevancy_heatmap(
                                identity_chunk.T[None, :, :, None],
                                active_emb,
                                canonical_emb=canonical_emb,
                                temperature=temperature,
                                scoring=scoring,
                                all_scene_emb=all_scene_emb,
                                active_scene_indices=active_scene_idx,
                            )
                            identity_score_parts.append(
                                identity_heatmaps.squeeze(-1).T
                            )
                        localization_rows = torch.cat(
                            identity_score_parts, dim=0
                        ).contiguous()
                        localization_rows = neutralize_invalid_primitive_scores_for_render(
                            localization_rows,
                            primitive_query_valid,
                        )
                    elif primitive_localization_rows is not None:
                        localization_rows = primitive_localization_rows[
                            :, active_indices
                        ]
                        localization_rows = neutralize_invalid_primitive_scores_for_render(
                            localization_rows,
                            primitive_support_valid,
                        ).to(device=device, dtype=torch.float32)
                render_rows = support_rows
                if primitive_valid_normalization:
                    if primitive_support_valid is None:
                        raise ValueError(
                            "valid normalization requires primitive support validity"
                        )
                    render_rows = torch.cat(
                        [
                            render_rows,
                            torch.as_tensor(
                                primitive_support_valid,
                                device=device,
                                dtype=render_rows.dtype,
                            )[:, None],
                        ],
                        dim=1,
                    )
                visibility_mass_projection = bool(
                    render_readout == "primitive_posterior"
                    and primitive_posterior_visibility_mass
                )
                support_render = renderer.render_feature_rows(
                    model,
                    viewmat.squeeze(0),
                    render_rows,
                    alpha_normalize=not visibility_mass_projection,
                    contribution_gamma=feature_contribution_gamma,
                )
                if visibility_mass_projection:
                    # Preserve residual transmittance as the Bernoulli
                    # background event: P(y_x=1)=sum_i T_i alpha_i P(y_i=1).
                    # Alpha normalization would amplify transparent tails.
                    rendered_semantics = support_render["feature_map"].float()
                    if primitive_valid_normalization:
                        semantic_coverage = rendered_semantics[-1]
                        rendered_semantics = rendered_semantics[:-1]
                    else:
                        semantic_coverage = None
                elif primitive_valid_normalization:
                    rendered_semantics, semantic_coverage = (
                        normalize_primitive_scores_by_valid_mass(
                            support_render["feature_map"].float(),
                            coverage_power=primitive_valid_coverage_power,
                        )
                    )
                else:
                    rendered_semantics = support_render["feature_map"].float()
                    semantic_coverage = None
                heatmaps = rendered_semantics[: len(active_indices)]
                localization_heatmaps = (
                    renderer.render_feature_rows(
                        model,
                        viewmat.squeeze(0),
                        localization_rows,
                        alpha_normalize=True,
                        contribution_gamma=feature_contribution_gamma,
                    )["feature_map"].float()
                    if localization_rows is not None else None
                )
                render_aux = {
                    "alpha_map": support_render["alpha_map"][None, None]
                }
                if semantic_coverage is not None:
                    render_aux["semantic_coverage"] = semantic_coverage[
                        None, None
                    ]
            else:
                heatmaps = compute_relevancy_heatmap(
                    siglip_feat, active_emb,
                    canonical_emb=canonical_emb,
                    temperature=temperature,
                    scoring=scoring,
                    all_scene_emb=all_scene_emb,
                    active_scene_indices=active_scene_idx,
                )
                if region_siglip_feat is not None:
                    region_heatmaps = compute_relevancy_heatmap(
                        region_siglip_feat,
                        active_emb,
                        canonical_emb=canonical_emb,
                        temperature=temperature,
                        scoring=scoring,
                        all_scene_emb=all_scene_emb,
                        active_scene_indices=active_scene_idx,
                    )
                    if region_heatmaps.shape[-2:] != heatmaps.shape[-2:]:
                        region_heatmaps = F.interpolate(
                            region_heatmaps[None],
                            size=heatmaps.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )[0]
                    heatmaps = fuse_typed_text_scores(
                        heatmaps,
                        region_heatmaps,
                        region_weight=region_score_weight,
                    )
            if mode_confidence_gate != "none":
                heatmaps = apply_readout_confidence_gate(
                    heatmaps,
                    render_aux,
                    gate=mode_confidence_gate,
                    gamma=readout_confidence_gamma,
                )

            # Published LERF evaluators compare image-resolution masks.  Keep
            # the legacy integer upsample path available for old artifacts,
            # while the frozen protocol evaluates at the annotation size.
            if eval_at_image_resolution and tuple(heatmaps.shape[-2:]) != (img_h, img_w):
                heatmaps = F.interpolate(
                    heatmaps.unsqueeze(0),
                    size=(img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                if localization_heatmaps is not None:
                    localization_heatmaps = F.interpolate(
                        localization_heatmaps.unsqueeze(0),
                        size=(img_h, img_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
            elif heatmap_upsample > 1:
                K = heatmaps.shape[0]
                heatmaps = F.interpolate(
                    heatmaps.unsqueeze(0),  # [1, K, fH, fW]
                    scale_factor=heatmap_upsample,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)  # [K, fH*up, fW*up]
                if localization_heatmaps is not None:
                    localization_heatmaps = F.interpolate(
                        localization_heatmaps.unsqueeze(0),
                        scale_factor=heatmap_upsample,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

            fH, fW = heatmaps.shape[1], heatmaps.shape[2]
            prompt_mask_feature = None
            if mode_mask_refinement == "sam3_prompt_mask_head" and sam3_prompt_mask_head is not None:
                if feat_1280 is None and sam3_prompt_mask_head_feature_dir:
                    feature_path = (
                        Path(sam3_prompt_mask_head_feature_dir)
                        / f"rgb_{frame_id}.pt"
                    )
                    if not feature_path.is_file():
                        feature_path = (
                            Path(sam3_prompt_mask_head_feature_dir)
                            / "backbone"
                            / f"rgb_{frame_id}.pt"
                        )
                    if not feature_path.is_file():
                        raise FileNotFoundError(
                            "feature-only SAM3 boundary input is missing: "
                            f"{feature_path}"
                        )
                    feat_1280 = torch.load(feature_path, map_location=device).float()
                    if feat_1280.ndim == 3:
                        feat_1280 = feat_1280.unsqueeze(0)
                    if feat_1280.ndim != 4 or int(feat_1280.shape[1]) != 1280:
                        raise ValueError(
                            "feature-only SAM3 boundary input must have shape "
                            f"[1,1280,H,W], got {tuple(feat_1280.shape)}"
                        )
                if feat_1280 is None:
                    raise ValueError(
                        "sam3_prompt_mask_head requires rendered 1280D features; "
                        "set --sam3_prompt_mask_head_feature_dir for primitive_score"
                    )
                prompt_mask_feature = F.interpolate(
                    feat_1280.float(),
                    size=tuple(int(v) for v in sam3_prompt_mask_head_target_size),
                    mode="bilinear",
                    align_corners=False,
                )

            # Build GT masks at full image resolution
            gt_masks_full = build_gt_masks(frame_objects, active_cats, img_h, img_w)
            # Scale polygons from image coords to feature resolution
            gt_masks_feat = build_gt_masks(frame_objects, active_cats, fH, fW,
                                           src_height=img_h, src_width=img_w)
            rgb_image_for_masks = None
            sam3_box_state = None
            if (
                mode_mask_refinement
                in {
                    "rgb_grabcut",
                    "peak_component_rgb_grabcut",
                    "sam3_box",
                }
                or save_overlay_vis
            ):
                if (
                    mode_mask_refinement != "none"
                    and rgb_refinement_source == "rendered"
                    and canonical_mode == "rendered"
                    and rgb_mask_renderer is not None
                    and render_rgb_refinement_frame_fn is not None
                ):
                    rgb_image_for_masks = render_rgb_refinement_frame_fn(
                        model,
                        rgb_mask_renderer,
                        viewmat.squeeze(0),
                    )
                else:
                    rgb_image_for_masks = load_lerf_rgb_frame(
                        scene,
                        frame_id,
                        scene_root_hint,
                    )
                if (
                    mode_mask_refinement == "sam3_box"
                    and sam3_box_refiner is not None
                    and rgb_image_for_masks is not None
                ):
                    sam3_box_state = sam3_box_refiner.set_image(rgb_image_for_masks)

            hm_vis: Dict[str, np.ndarray] = {}
            gt_vis: Dict[str, np.ndarray] = {}
            for ki, cat in enumerate(active_cats):
                hm = heatmaps[ki]  # [fH, fW]
                hm_localization = (
                    localization_heatmaps[ki]
                    if mode == "rendered"
                    and render_readout == "primitive_posterior"
                    and localization_heatmaps is not None
                    else hm
                )
                gt_full = gt_masks_full[cat]
                gt_feat = gt_masks_feat[cat]
                initial_pred_for_save: Optional[np.ndarray] = None
                pred_for_save: Optional[np.ndarray] = None

                if gt_full.sum() == 0:
                    continue

                # Localization: map feature-level argmax to image coordinates
                if localization_mode == "bbox_smoothed_peak":
                    cat_bboxes = [
                        obj["bbox"]
                        for obj in frame_objects
                        if obj["category"] == cat and obj.get("bbox") is not None
                    ]
                    is_correct = localization_accuracy_bbox_smoothed(
                        hm_localization,
                        cat_bboxes,
                        image_shape=(img_h, img_w),
                        kernel_size=localization_smoothing_kernel,
                    )
                elif localization_mode == "polygon_argmax":
                    is_correct = localization_accuracy(hm_localization, gt_full)
                else:
                    raise ValueError(f"Unsupported localization_mode: {localization_mode}")
                loc_correct += int(is_correct)
                loc_total += 1
                per_cat_loc[cat].append(is_correct)

                # mIoU at feature resolution by default; optional RGB refinement
                # evaluates the refined mask at image resolution without using GT.
                if mode_mask_refinement == "sam3_box" and sam3_box_refiner is not None:
                    initial_pred = build_sam3_prompt_initial_mask(
                        hm,
                        threshold_ratio=iou_threshold,
                        threshold_mode=threshold_mode,
                        threshold_mean_std_k=threshold_mean_std_k,
                        threshold_min_ratio=threshold_min_ratio,
                        threshold_max_ratio=threshold_max_ratio,
                        target_shape=(img_h, img_w),
                        initial_refinement=sam3_box_initial_refinement,
                    )
                    initial_pred_for_save = initial_pred
                    initial_overlap = mask_overlap_stats(initial_pred, gt_full)
                    if sam3_box_state is None:
                        pred = initial_pred.copy()
                        report = {
                            "attempted": False,
                            "accepted": False,
                            "fallback_reason": "missing_sam3_state",
                        }
                    else:
                        pred, report = sam3_box_refiner.refine_from_state_with_report(
                            sam3_box_state,
                            initial_pred,
                        )
                    if report.get("accepted", False) and (
                        sam3_box_require_peak_in_refined
                        or sam3_box_min_heatmap_mean_ratio > 0
                        or sam3_box_min_heatmap_mass_ratio > 0
                    ):
                        pred, support_report = filter_refined_mask_by_heatmap_support(
                            initial_pred,
                            pred,
                            hm,
                            min_mean_ratio=sam3_box_min_heatmap_mean_ratio,
                            min_mass_ratio=sam3_box_min_heatmap_mass_ratio,
                            require_peak_in_refined=sam3_box_require_peak_in_refined,
                        )
                        report["heatmap_support"] = support_report
                        if not support_report.get("accepted", False):
                            report["accepted"] = False
                            report["fallback_reason"] = support_report.get(
                                "fallback_reason",
                                "heatmap_support_rejected",
                            )
                    iou = float(mask_overlap_stats(pred, gt_full)["iou"])
                    pred_for_save = pred
                    initial_ious.append(float(initial_overlap["iou"]))
                    refinement_reports.append(report)
                elif mode_mask_refinement == "sam3_prompt_mask_head" and prompt_mask_feature is not None:
                    initial_pred = build_sam3_prompt_initial_mask(
                        hm,
                        threshold_ratio=iou_threshold,
                        threshold_mode=threshold_mode,
                        threshold_mean_std_k=threshold_mean_std_k,
                        threshold_min_ratio=threshold_min_ratio,
                        threshold_max_ratio=threshold_max_ratio,
                        target_shape=(img_h, img_w),
                        initial_refinement=sam3_prompt_mask_head_initial_refinement,
                    )
                    initial_pred_for_save = initial_pred
                    initial_overlap = mask_overlap_stats(initial_pred, gt_full)
                    pred, report = refine_mask_with_prompt_conditioned_sam3_head(
                        feature_map=prompt_mask_feature,
                        prompt_embedding=text_embeddings[cat_to_idx[cat]].detach().float(),
                        coarse_mask=initial_pred,
                        head=sam3_prompt_mask_head,
                        logit_threshold=sam3_prompt_mask_head_logit_threshold,
                        min_initial_iou=sam3_prompt_mask_head_min_initial_iou,
                        max_initial_area_fraction=sam3_prompt_mask_head_max_initial_area_fraction,
                        min_refined_area_ratio=sam3_prompt_mask_head_min_refined_area_ratio,
                        max_refined_area_ratio=sam3_prompt_mask_head_max_refined_area_ratio,
                        support_dilate=sam3_prompt_mask_head_support_dilate,
                        coarse_dilate=sam3_prompt_mask_head_coarse_dilate,
                        coarse_threshold=sam3_prompt_mask_head_coarse_threshold,
                    )
                    if report.get("accepted", False) and (
                        sam3_prompt_mask_head_require_peak_in_refined
                        or sam3_prompt_mask_head_min_heatmap_mean_ratio > 0
                        or sam3_prompt_mask_head_min_heatmap_mass_ratio > 0
                    ):
                        pred, support_report = filter_refined_mask_by_heatmap_support(
                            initial_pred,
                            pred,
                            hm,
                            min_mean_ratio=sam3_prompt_mask_head_min_heatmap_mean_ratio,
                            min_mass_ratio=sam3_prompt_mask_head_min_heatmap_mass_ratio,
                            require_peak_in_refined=sam3_prompt_mask_head_require_peak_in_refined,
                        )
                        report["heatmap_support"] = support_report
                        if not support_report.get("accepted", False):
                            report["accepted"] = False
                            report["fallback_reason"] = support_report.get(
                                "fallback_reason",
                                "heatmap_support_rejected",
                            )
                    iou = float(mask_overlap_stats(pred, gt_full)["iou"])
                    pred_for_save = pred
                    initial_ious.append(float(initial_overlap["iou"]))
                    refinement_reports.append(report)
                elif mode_mask_refinement != "none" and rgb_image_for_masks is not None:
                    iou = compute_iou(
                        hm,
                        gt_full,
                        threshold_ratio=iou_threshold,
                        threshold_mode=threshold_mode,
                        threshold_mean_std_k=threshold_mean_std_k,
                        threshold_min_ratio=threshold_min_ratio,
                        threshold_max_ratio=threshold_max_ratio,
                        rgb_image=rgb_image_for_masks,
                        mask_refinement=mode_mask_refinement,
                        mask_refinement_iters=mask_refinement_iters,
                        mask_refinement_dilate=mask_refinement_dilate,
                        mask_refinement_erode=mask_refinement_erode,
                    )
                else:
                    iou = compute_iou(
                        hm,
                        gt_feat,
                        threshold_ratio=iou_threshold,
                        threshold_mode=threshold_mode,
                        threshold_mean_std_k=threshold_mean_std_k,
                        threshold_min_ratio=threshold_min_ratio,
                        threshold_max_ratio=threshold_max_ratio,
                    )
                ious.append(iou)
                per_cat_iou[cat].append(iou)

                if pred_mask_dir is not None:
                    if pred_for_save is None:
                        target_shape = tuple(gt_full.shape)
                        initial_pred_for_save = build_sam3_prompt_initial_mask(
                            hm,
                            threshold_ratio=iou_threshold,
                            threshold_mode=threshold_mode,
                            threshold_mean_std_k=threshold_mean_std_k,
                            threshold_min_ratio=threshold_min_ratio,
                            threshold_max_ratio=threshold_max_ratio,
                            target_shape=target_shape,
                            initial_refinement="peak_component"
                            if mode_mask_refinement == "peak_component"
                            else "none",
                        )
                        pred_for_save = initial_pred_for_save
                    save_prediction_masks(
                        out_dir=pred_mask_dir,
                        scene=scene,
                        mode=lerf_mode_tag(mode),
                        frame_id=frame_id,
                        query=cat,
                        gt_mask=gt_full,
                        initial_mask=initial_pred_for_save,
                        final_mask=pred_for_save,
                        rgb_image=rgb_image_for_masks,
                    )

                if vis_dir is not None:
                    hm_vis[cat] = hm.cpu().numpy()
                    gt_vis[cat] = gt_feat

            if vis_dir is not None and hm_vis:
                source_label = (
                    "teacher RADIO heatmap"
                    if canonical_mode == "teacher"
                    else "rendered RADIO-GS heatmap"
                )
                rgb_image = rgb_image_for_masks if save_overlay_vis else None
                save_heatmap_vis(
                    hm_vis,
                    gt_vis,
                    frame_id,
                    vis_dir / scene,
                    tag=lerf_mode_tag(mode),
                    source_label=source_label,
                    rgb_image=rgb_image,
                    save_per_query=save_per_query_vis,
                )

        if loc_total == 0 or not ious:
            raise RuntimeError(
                f"Zero-sample LERF evaluation for scene={scene!r}, mode={canonical_mode!r}. "
                "Check labeled frame IDs, camera poses, and requested feature source; "
                "refusing to write a misleading all-zero result."
            )
        loc_acc = loc_correct / loc_total
        miou = float(np.mean(ious))

        per_cat_summary = {}
        for cat in scene_categories:
            cat_loc = per_cat_loc[cat]
            cat_iou = per_cat_iou[cat]
            per_cat_summary[cat] = {
                "loc_acc": float(np.mean(cat_loc)) if cat_loc else None,
                "miou": float(np.mean(cat_iou)) if cat_iou else None,
                "n_samples": len(cat_loc),
            }

        mode_metrics = {
            "loc_acc": loc_acc,
            "localization_mode": localization_mode,
            "localization_smoothing_kernel": int(localization_smoothing_kernel),
            "miou": miou,
            "loc_correct": loc_correct,
            "loc_total": loc_total,
            "n_iou_samples": len(ious),
            "per_category": per_cat_summary,
            "render_readout": render_readout if canonical_mode == "rendered" else "teacher",
        }
        if initial_ious:
            accepted = sum(1 for report in refinement_reports if bool(report.get("accepted", False)))
            attempted = sum(1 for report in refinement_reports if bool(report.get("attempted", True)))
            initial_miou = float(np.mean(initial_ious))
            mode_metrics.update(
                {
                    "initial_miou": initial_miou,
                    "delta_miou": miou - initial_miou,
                    "sam3_prompt_refinement_count": len(refinement_reports),
                    "sam3_prompt_attempt_count": attempted,
                    "sam3_prompt_accept_count": accepted,
                    "sam3_prompt_accept_rate": float(accepted / max(attempted, 1)),
                    "sam3_prompt_mean_initial_overlap": float(np.mean([
                        float(report.get("best_initial_overlap", 0.0))
                        for report in refinement_reports
                    ])) if refinement_reports else 0.0,
                    "sam3_prompt_mean_refined_area_ratio": float(np.mean([
                        float(report.get("refined_area_ratio", 0.0))
                        for report in refinement_reports
                        if "refined_area_ratio" in report
                    ])) if any("refined_area_ratio" in report for report in refinement_reports) else 0.0,
                    "sam3_prompt_fallback_reasons": {
                        str(reason): sum(
                            1
                            for report in refinement_reports
                            if str(report.get("fallback_reason", "")) == str(reason)
                        )
                        for reason in sorted({
                            str(report.get("fallback_reason", ""))
                            for report in refinement_reports
                        })
                    },
                }
            )
        results[canonical_mode] = mode_metrics
        if canonical_mode == "teacher":
            # Backward-compatible JSON alias for existing sweep scripts/results.
            results["gt"] = mode_metrics

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LERF-OVS text grounding & segmentation evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Model / rendering
    parser.add_argument("--config", default=None,
                        help="Path to LERF feature-field config YAML (for rendered mode)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to trained model checkpoint (for rendered mode)")
    # Teacher/oracle RADIO features
    parser.add_argument("--gt_feature_dir", default=None,
                        help="Dir with teacher/oracle RADIO 1280-d .pt files (or parent with backbone/ subdir)")
    parser.add_argument(
        "--region_feature_dir",
        default=None,
        help=(
            "Optional typed 1536-D region-summary map paired with the primary "
            "RADIO map for fixed two-level score fusion."
        ),
    )
    parser.add_argument(
        "--region_score_weight",
        type=float,
        default=0.5,
        help="Fixed region score share; frozen VALA evaluation requires 0.5.",
    )
    parser.add_argument("--gt_only", action="store_true",
                        help="Evaluate teacher/oracle RADIO features only (skip rendered mode)")
    parser.add_argument("--rendered_only", action="store_true",
                        help="Evaluate the rendered feature field only (skip teacher features)")
    parser.add_argument(
        "--render_readout",
        choices=[
            "screen_decode",
            "primitive_query",
            "primitive_score",
            "primitive_support",
            "primitive_unary",
            "primitive_posterior",
        ],
        default="screen_decode",
        help=(
            "Rendered path: decode-after-splat, splat query rows, score "
            "primitives then splat scalar unaries, or splat a shared-3D-solver "
            "support cache."
        ),
    )
    parser.add_argument("--primitive_chunk_size", type=int, default=8192)
    parser.add_argument("--primitive_query_cache", default="", help="Oracle/audit cache of row-aligned per-Gaussian query-space features")
    parser.add_argument(
        "--primitive_confidence",
        choices=["cache", "none"],
        default="cache",
        help=(
            "For primitive_score, apply an optional query-independent "
            "semantic_confidence vector stored in the primitive cache."
        ),
    )
    parser.add_argument(
        "--primitive_fallback_blend",
        choices=[
            "primary_first",
            "strongest",
            "uncovered",
            "dominant",
            "direct",
        ],
        default="primary_first",
        help=(
            "For completion caches, preserve the primary score map whenever "
            "it has positive support and use fallback only for unsupported "
            "queries (default). Other choices are diagnostic ablations."
        ),
    )
    parser.add_argument(
        "--primitive_score_cache",
        default="",
        help=(
            "Row-aligned shared-3D-solver probability cache required by "
            "--render_readout primitive_support."
        ),
    )
    parser.add_argument(
        "--feature_contribution_gamma",
        type=float,
        default=1.0,
        help=(
            "Query-independent exponent for normalized front-to-back feature "
            "mixture weights. 1 preserves ordinary alpha averaging; 2 is the "
            "label-free frozen compositor candidate."
        ),
    )
    parser.add_argument(
        "--primitive_valid_normalization",
        action="store_true",
        help=(
            "Render scalar primitive scores as sum(w*v*s)/sum(w*v), keeping "
            "semantic validity coverage separate from the score value."
        ),
    )
    parser.add_argument(
        "--primitive_valid_coverage_power",
        type=float,
        default=0.0,
        help=(
            "Query-independent coverage abstention after valid-mass "
            "normalization. Zero is purely conditional; one recovers the "
            "original total-alpha score."
        ),
    )
    parser.add_argument(
        "--primitive_posterior_visibility_mass",
        action="store_true",
        help=(
            "Project primitive posterior as sum(T*alpha*P), retaining residual "
            "transmittance as background. Valid only for primitive_posterior."
        ),
    )
    parser.add_argument(
        "--primitive_posterior_calibration_scale",
        type=float,
        default=1.0,
        help=(
            "Positive output-domain Platt scale applied to the shared Gaussian "
            "posterior before rendering; one with zero bias is identity."
        ),
    )
    parser.add_argument(
        "--primitive_posterior_calibration_bias",
        type=float,
        default=0.0,
        help=(
            "Output-domain Platt bias applied without changing Gaussian/proposal "
            "ranking; it must be source-trained for a formal benchmark claim."
        ),
    )
    # Scene selection
    parser.add_argument("--scene", default="all",
                        help="Scene name or 'all' (default: all)")
    # Paths
    parser.add_argument("--label_dir", default=DEFAULT_LABEL_DIR,
                        help="LERF-OVS label root dir")
    parser.add_argument("--output_dir", default="output/lerf_ovs_eval",
                        help="Where to save JSON report")
    parser.add_argument("--projection_weights", default=DEFAULT_SIGLIP2_PROJECTION_WEIGHTS,
                        help="SigLIP2 feature projection weights")
    parser.add_argument("--summary_head_weights", default="checkpoints/siglip2_summary_head.pth",
                        help="SigLIP2 summary head weights (text-aligned, preferred for grounding)")
    parser.add_argument("--use_summary_head", action="store_true", default=True,
                        help="Use the text-aligned summary head instead of spatial projection (default)")
    parser.add_argument("--no_summary_head", dest="use_summary_head", action="store_false",
                        help="Use spatial feature projection instead of summary head")
    parser.add_argument(
        "--preprojected_text_features",
        action="store_true",
        help=(
            "Input maps are already normalized 1536-D SigLIP2 text-space "
            "descriptors produced by the typed region-summary readout."
        ),
    )
    parser.add_argument("--text_embedding_cache", default=None,
                        help="Path to cache/load pre-computed text embeddings")
    parser.add_argument("--canonical_embedding_cache", default=None,
                        help="Optional cache for fixed generic negative/canonical text embeddings")
    parser.add_argument("--text_encoder", choices=["siglip2", "openclip"], default="siglip2",
                        help="Text/readout space. OpenCLIP expects a native 512-D SAM-CLIP field")
    parser.add_argument("--openclip_model", default="ViT-B-16",
                        help="OpenCLIP model used with --text_encoder openclip")
    parser.add_argument("--openclip_pretrained", default="laion2b_s34b_b88k",
                        help="OpenCLIP pretrained tag used with --text_encoder openclip")
    parser.add_argument("--prompt_templates", default=DEFAULT_PROMPT_TEMPLATES,
                        help="Prompt templates separated by '|'. Use {query} as placeholder")
    # Evaluation
    parser.add_argument("--protocol_preset", choices=["none", "vala_paper_2d"], default="none",
                        help="Frozen paper-level preset: relevance>0.5, image-resolution evaluation, and no refinement")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="Mask threshold. Its units are selected by --threshold_mode")
    parser.add_argument("--threshold_mode", choices=["fixed", "absolute", "mean_std"], default="fixed",
                        help="Mask threshold rule: fixed/mean_std are peak-relative; absolute applies score > --iou_threshold")
    parser.add_argument("--threshold_mean_std_k", type=float, default=1.0,
                        help="k for --threshold_mode mean_std")
    parser.add_argument("--threshold_min_ratio", type=float, default=0.0,
                        help="Lower clamp for adaptive threshold ratios")
    parser.add_argument("--threshold_max_ratio", type=float, default=1.0,
                        help="Upper clamp for adaptive threshold ratios")
    parser.add_argument("--scoring", choices=["cosine", "softmax_scene", "relevancy"],
                        default="softmax_scene",
                        help="Scoring: 'softmax_scene' (recommended, softmax over scene categories), "
                             "'cosine' (raw similarity), or 'relevancy' (LERF-style canonical)")
    parser.add_argument("--relevancy_temp", type=float, default=50.0,
                        help="Logit scale for softmax_scene (default 50); denominator temperature for relevancy")
    parser.add_argument("--save_vis", action="store_true",
                        help="Save heatmap visualisations")
    parser.add_argument("--save_overlay_vis", action="store_true",
                        help="When saving heatmaps, append RGB, GT/RGB, and heatmap/RGB overlay columns")
    parser.add_argument("--save_per_query_vis", action="store_true",
                        help="Also write one visualisation PNG per frame/query. Enabled automatically with --save_overlay_vis")
    parser.add_argument("--save_pred_masks", action="store_true",
                        help="Save per-query GT, initial, and final binary masks/overlays for qualitative figures")
    parser.add_argument("--heatmap_upsample", type=int, default=4,
                        help="Upsample heatmaps by this factor before localization (default 4)")
    parser.add_argument("--eval_at_image_resolution", action="store_true",
                        help="Bilinearly resize heatmaps to annotation resolution before mask evaluation")
    parser.add_argument(
        "--localization_mode",
        choices=["polygon_argmax", "bbox_smoothed_peak"],
        default="polygon_argmax",
        help="Localization metric readout; bbox_smoothed_peak matches released VALA/LangSplat",
    )
    parser.add_argument("--localization_smoothing_kernel", type=int, default=30,
                        help="Box-filter kernel for bbox_smoothed_peak localization")
    parser.add_argument(
        "--readout_confidence_gate",
        choices=["none", "quality", "visibility", "quality_visibility", "alpha"],
        default="none",
        help="GT-free rendered-readout confidence gating before mask thresholding.",
    )
    parser.add_argument(
        "--readout_confidence_gamma",
        type=float,
        default=1.0,
        help="Exponent for --readout_confidence_gate; 0 disables gating.",
    )
    parser.add_argument("--mask_refinement", choices=["none", "rgb_grabcut", "peak_component", "peak_component_rgb_grabcut", "sam3_box", "sam3_prompt_mask_head"], default="none",
                        help="Optional GT-free RGB boundary snapping after heatmap binarisation")
    parser.add_argument(
        "--rgb_refinement_source",
        choices=["rendered", "dataset_frame"],
        default="dataset_frame",
        help=(
            "RGB used by GrabCut/official SAM3 refinement. 'rendered' uses only "
            "the Gaussian scene; 'dataset_frame' preserves historical behavior."
        ),
    )
    parser.add_argument("--mask_refinement_iters", type=int, default=1,
                        help="GrabCut iterations for rgb_grabcut mask refinement")
    parser.add_argument("--mask_refinement_dilate", type=int, default=5,
                        help="Pixel dilation radius defining rgb_grabcut support band")
    parser.add_argument("--mask_refinement_erode", type=int, default=2,
                        help="Pixel erosion radius defining sure foreground for rgb_grabcut")
    parser.add_argument("--sam3_checkpoint_path", default="checkpoints/sam3_modelscope/sam3.pt")
    parser.add_argument("--sam3_confidence_threshold", type=float, default=0.0)
    parser.add_argument("--sam3_resolution", type=int, default=1008)
    parser.add_argument("--sam3_amp_dtype", choices=["auto", "off", "bfloat16"], default="auto")
    parser.add_argument("--sam3_box_padding", type=int, default=8)
    parser.add_argument("--sam3_min_initial_iou", type=float, default=0.05)
    parser.add_argument(
        "--sam3_box_initial_refinement",
        choices=["none", "peak_component", "peak_component_or_seed"],
        default="none",
        help=(
            "GT-free coarse prompt cleanup. peak_component_or_seed falls back "
            "to the peak-connected top 10%% heatmap band only when the fixed "
            "threshold produces an empty mask."
        ),
    )
    parser.add_argument("--sam3_box_min_heatmap_mean_ratio", type=float, default=0.0)
    parser.add_argument("--sam3_box_min_heatmap_mass_ratio", type=float, default=0.0)
    parser.add_argument("--sam3_box_require_peak_in_refined", action="store_true")
    parser.add_argument("--sam3_prompt_mask_head_checkpoint", default="",
                        help="Feature-only prompt-conditioned SAM mask head checkpoint. Supports {scene}.")
    parser.add_argument(
        "--sam3_prompt_mask_head_logit_threshold",
        type=float,
        default=DEFAULT_SAM3_PROMPT_MASK_HEAD_LOGIT_THRESHOLD,
    )
    parser.add_argument(
        "--sam3_prompt_mask_head_min_initial_iou",
        type=float,
        default=DEFAULT_SAM3_PROMPT_MASK_HEAD_MIN_INITIAL_IOU,
    )
    parser.add_argument(
        "--sam3_prompt_mask_head_max_initial_area_fraction",
        type=float,
        default=DEFAULT_SAM3_PROMPT_MASK_HEAD_MAX_INITIAL_AREA_FRACTION,
    )
    parser.add_argument(
        "--sam3_prompt_mask_head_min_refined_area_ratio",
        type=float,
        default=DEFAULT_SAM3_PROMPT_MASK_HEAD_MIN_REFINED_AREA_RATIO,
    )
    parser.add_argument(
        "--sam3_prompt_mask_head_max_refined_area_ratio",
        type=float,
        default=DEFAULT_SAM3_PROMPT_MASK_HEAD_MAX_REFINED_AREA_RATIO,
    )
    parser.add_argument(
        "--sam3_prompt_mask_head_support_dilate",
        type=int,
        default=DEFAULT_SAM3_PROMPT_MASK_HEAD_SUPPORT_DILATE,
    )
    parser.add_argument(
        "--sam3_prompt_mask_head_coarse_dilate",
        type=int,
        default=DEFAULT_SAM3_PROMPT_MASK_HEAD_COARSE_DILATE,
    )
    parser.add_argument(
        "--sam3_prompt_mask_head_coarse_threshold",
        type=float,
        default=DEFAULT_SAM3_PROMPT_MASK_HEAD_COARSE_THRESHOLD,
    )
    parser.add_argument(
        "--sam3_prompt_mask_head_min_heatmap_mean_ratio",
        type=float,
        default=0.0,
        help="GT-free acceptance guard: require refined mask mean heatmap support / initial mean support to exceed this value.",
    )
    parser.add_argument(
        "--sam3_prompt_mask_head_min_heatmap_mass_ratio",
        type=float,
        default=0.0,
        help="GT-free acceptance guard: require refined mask total heatmap support / initial total support to exceed this value.",
    )
    parser.add_argument(
        "--sam3_prompt_mask_head_require_peak_in_refined",
        action="store_true",
        help="Reject feature-only SAM refinements that drop the query heatmap peak.",
    )
    parser.add_argument(
        "--sam3_prompt_mask_head_initial_refinement",
        choices=["none", "peak_component", "adaptive_peak"],
        default="none",
        help="GT-free refinement applied to the heatmap coarse mask before feature-only SAM3 decoding.",
    )
    parser.add_argument("--sam3_prompt_mask_head_apply_to", choices=["rendered", "all"], default="rendered")
    parser.add_argument(
        "--sam3_prompt_mask_head_feature_dir",
        default="",
        help=(
            "Optional current-field rendered 1280D feature directory used by "
            "the feature-only SAM3 head when the identity readout is primitive_score"
        ),
    )
    # Hardware
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device id")

    args = parser.parse_args()
    if not np.isfinite(args.feature_contribution_gamma) or args.feature_contribution_gamma <= 0:
        parser.error("--feature_contribution_gamma must be finite and positive")
    if (
        not np.isfinite(args.primitive_valid_coverage_power)
        or args.primitive_valid_coverage_power < 0
    ):
        parser.error(
            "--primitive_valid_coverage_power must be finite and non-negative"
        )
    if (
        args.primitive_valid_coverage_power != 0
        and not args.primitive_valid_normalization
    ):
        parser.error(
            "--primitive_valid_coverage_power requires "
            "--primitive_valid_normalization"
        )
    if (
        args.primitive_posterior_visibility_mass
        and args.render_readout != "primitive_posterior"
    ):
        parser.error(
            "--primitive_posterior_visibility_mass requires "
            "--render_readout primitive_posterior"
        )
    if args.gt_only and args.rendered_only:
        parser.error("--gt_only and --rendered_only are mutually exclusive")
    if args.preprojected_text_features and not args.gt_only:
        parser.error(
            "--preprojected_text_features is only valid for a typed GT cache"
        )
    if args.preprojected_text_features and args.region_feature_dir:
        parser.error(
            "a preprojected region cache cannot also be the primitive level "
            "of a two-level readout"
        )
    if args.region_feature_dir and not args.gt_feature_dir:
        parser.error("--region_feature_dir requires --gt_feature_dir")
    if args.render_readout in {
        "primitive_query",
        "primitive_score",
        "primitive_support",
        "primitive_unary",
        "primitive_posterior",
    } and args.text_encoder != "siglip2":
        parser.error("primitive-first render readouts currently require --text_encoder siglip2")
    if args.render_readout in {"primitive_support", "primitive_unary", "primitive_posterior"} and not args.primitive_score_cache:
        parser.error(f"{args.render_readout} requires --primitive_score_cache")
    if (
        args.primitive_valid_normalization
        and args.render_readout
        not in {
            "primitive_query",
            "primitive_score",
            "primitive_support",
            "primitive_unary",
            "primitive_posterior",
        }
    ):
        parser.error(
            "--primitive_valid_normalization requires a primitive-first render readout"
        )
    if (
        args.feature_contribution_gamma != 1.0
        and args.render_readout == "screen_decode"
    ):
        parser.error(
            "contribution sharpening currently requires a primitive-first render readout"
        )
    if args.scene == "all" and not args.gt_only:
        parser.error(
            "Rendered --scene all is unsafe because one config/checkpoint cannot "
            "represent four scene-specific fields; run one command per scene."
        )
    if args.protocol_preset == "vala_paper_2d":
        # Freeze every paper-facing readout choice.  In particular, this does
        # not inspect LERF annotations to choose a threshold, temperature,
        # checkpoint, component rule, or boundary post-process.
        args.scoring = "relevancy"
        args.relevancy_temp = 0.1
        args.threshold_mode = "absolute"
        args.iou_threshold = 0.5
        args.mask_refinement = "none"
        args.readout_confidence_gate = "none"
        args.prompt_templates = "{query}"
        args.heatmap_upsample = 1
        args.eval_at_image_resolution = True
        args.localization_mode = "bbox_smoothed_peak"
        args.localization_smoothing_kernel = 30
        # RADIO/SigLIP fields use the declared frozen summary projection;
        # native SAM-CLIP/OpenCLIP fields are already in their text space and
        # therefore use the identity projection below.
        args.use_summary_head = args.text_encoder == "siglip2"
        if args.region_feature_dir and args.region_score_weight != 0.5:
            parser.error(
                "vala_paper_2d freezes primitive/region score fusion at 0.5"
            )
    args.label_dir = resolve_lerf_label_dir(args.label_dir)
    prompt_templates = parse_prompt_templates(args.prompt_templates)

    # Validate arguments
    if not args.gt_only and (args.config is None or args.checkpoint is None):
        if args.gt_feature_dir is None:
            parser.error(
                "Provide --config + --checkpoint for rendered evaluation, "
                "or --gt_feature_dir (+ --gt_only) for GT-only evaluation."
            )
        args.gt_only = True
    if args.mask_refinement == "sam3_prompt_mask_head" and not args.sam3_prompt_mask_head_checkpoint:
        parser.error("--mask_refinement sam3_prompt_mask_head requires --sam3_prompt_mask_head_checkpoint")
    if args.mask_refinement == "sam3_box" and not args.sam3_checkpoint_path:
        parser.error("--mask_refinement sam3_box requires --sam3_checkpoint_path")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    scenes = LERF_OVS_SCENES if args.scene == "all" else (args.scene,)

    print("=" * 70)
    print("  LERF-OVS Text Grounding Evaluation")
    print("=" * 70)
    print(f"  Scenes:     {', '.join(scenes)}")
    print(f"  Label dir:  {args.label_dir}")
    mode_label = "Teacher only" if args.gt_only else ("Rendered only" if args.rendered_only else "Teacher + Rendered")
    print(f"  Mode:       {mode_label}")
    print(f"  Readout:    {args.render_readout}")
    print(f"  Protocol:   {args.protocol_preset}")
    print(f"  Text enc.:  {args.text_encoder}")
    print(f"  IoU thresh: {args.iou_threshold}")
    print(f"  Thr mode:   {args.threshold_mode}")
    print(f"  Heatmap ↑:  {args.heatmap_upsample}×")
    print(f"  Loc mode:   {args.localization_mode}")
    print(f"  Conf gate:  {args.readout_confidence_gate} (γ={args.readout_confidence_gamma})")
    if args.render_readout == "primitive_score":
        print(f"  Prim. conf: {args.primitive_confidence}")
        print(f"  Fallback:   {args.primitive_fallback_blend}")
    print(f"  Mask ref.:  {args.mask_refinement}")
    if args.mask_refinement in {"rgb_grabcut", "peak_component_rgb_grabcut", "sam3_box"}:
        print(f"  RGB source: {args.rgb_refinement_source}")
    print(f"  Prompts:    {len(prompt_templates)} template(s)")
    print()

    # ------------------------------------------------------------------
    # 1. Discover all categories across requested scenes
    # ------------------------------------------------------------------
    all_categories: set[str] = set()
    for scene in scenes:
        scene_dir = Path(args.label_dir) / scene
        if not scene_dir.exists():
            logger.warning("Label dir missing for scene '%s' – skipping", scene)
            continue
        for jf in scene_dir.glob("frame_*.json"):
            data = _load_annotation_json(jf)
            for obj in data.get("objects", []):
                cat = str(obj.get("category", "")).strip()
                if cat:
                    all_categories.add(cat)
    categories = sorted(all_categories)
    print(f"Total unique categories: {len(categories)}")
    for i, c in enumerate(categories):
        print(f"  [{i:2d}] {c}")
    print()

    if args.protocol_preset == "vala_paper_2d":
        if not args.text_embedding_cache or not args.canonical_embedding_cache:
            parser.error(
                "vala_paper_2d requires explicit --text_embedding_cache and "
                "--canonical_embedding_cache with frozen provenance metadata"
            )
        if (
            args.text_encoder == "siglip2"
            and args.use_summary_head
            and not args.preprojected_text_features
            and not Path(args.summary_head_weights).exists()
        ):
            parser.error(
                "vala_paper_2d requires the declared --summary_head_weights file; "
                "frozen evaluation forbids implicit checkpoint fallback"
            )
        encoder_model = (
            args.openclip_model if args.text_encoder == "openclip" else _SIGLIP2_MODEL_NAME
        )
        encoder_pretrained = args.openclip_pretrained if args.text_encoder == "openclip" else ""
        validate_text_embedding_cache_contract(
            args.text_embedding_cache,
            required_queries=categories,
            prompt_templates=prompt_templates,
            text_encoder=args.text_encoder,
            model_name=encoder_model,
            pretrained=encoder_pretrained,
        )
        validate_text_embedding_cache_contract(
            args.canonical_embedding_cache,
            required_queries=list(NEGATIVE_PROMPTS),
            prompt_templates=["{query}"],
            text_encoder=args.text_encoder,
            model_name=encoder_model,
            pretrained=encoder_pretrained,
            exact_queries=True,
        )

    # ------------------------------------------------------------------
    # 2. Load the visual-to-text projection model
    # ------------------------------------------------------------------
    if args.preprojected_text_features:
        if args.text_encoder != "siglip2":
            parser.error(
                "--preprojected_text_features requires SigLIP2 text embeddings"
            )
        proj = nn.Identity()
        print("Using identity projection for native SigLIP2 text-space features")
    elif args.text_encoder == "openclip":
        proj = nn.Identity()
        print("Using identity projection for native OpenCLIP/SAM-CLIP features")
    elif args.use_summary_head:
        head_path = Path(args.summary_head_weights)
        if head_path.exists():
            proj = SigLIP2SummaryHead.from_extracted_weights(str(head_path))
            print(f"Loaded SigLIP2 summary head (text-aligned) from {head_path}")
        else:
            proj = SigLIP2SummaryHead.from_radio_checkpoint(
                "/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar"
            )
            print("Loaded SigLIP2 summary head from RADIO checkpoint")
    else:
        proj_path = resolve_siglip_projection_path(args.projection_weights)
        if proj_path.exists():
            proj = SigLIP2FeatureProjection()
            proj.load_state_dict(torch.load(str(proj_path), map_location="cpu"))
            print(f"Loaded SigLIP2 spatial projection from {proj_path}")
        else:
            raise FileNotFoundError(f"Projection weights not found: {proj_path}")
    proj = proj.to(device)
    proj = proj.half() if device.type == "cuda" else proj.float()
    proj = proj.eval()

    # ------------------------------------------------------------------
    # 3. Generate / load text embeddings
    # ------------------------------------------------------------------
    print(f"Generating {args.text_encoder} text embeddings …")
    t0 = time.time()
    if args.text_encoder == "openclip":
        text_embeddings = load_or_generate_openclip_prompt_ensemble_embeddings(
            categories,
            device,
            cache_path=args.text_embedding_cache,
            prompt_templates=prompt_templates,
            model_name=args.openclip_model,
            pretrained=args.openclip_pretrained,
        )
    else:
        text_embeddings = load_or_generate_prompt_ensemble_embeddings(
            categories,
            device,
            cache_path=args.text_embedding_cache,
            prompt_templates=prompt_templates,
        )
    text_embeddings = text_embeddings.half() if device.type == "cuda" else text_embeddings.float()
    print(f"  {text_embeddings.shape[0]} embeddings ({text_embeddings.shape[1]}-d), "
          f"{time.time() - t0:.1f}s")

    # Generate canonical phrase embeddings for relevancy scoring
    canonical_emb = None
    if args.scoring == "relevancy":
        if args.protocol_preset == "vala_paper_2d" or args.text_encoder == "openclip":
            default_name = f"{args.text_encoder}_lerf_negative_embeddings.pt"
            canon_cache = Path(args.canonical_embedding_cache or f"checkpoints/{default_name}")
            if args.text_encoder == "openclip":
                canonical_emb = load_or_generate_openclip_prompt_ensemble_embeddings(
                    list(NEGATIVE_PROMPTS),
                    device,
                    cache_path=canon_cache,
                    prompt_templates=["{query}"],
                    model_name=args.openclip_model,
                    pretrained=args.openclip_pretrained,
                )
            else:
                canonical_emb = load_or_generate_prompt_ensemble_embeddings(
                    list(NEGATIVE_PROMPTS),
                    device,
                    cache_path=str(canon_cache),
                    prompt_templates=["{query}"],
                )
            canonical_emb = F.normalize(canonical_emb.float(), dim=-1).to(device)
            canonical_emb = canonical_emb.half() if device.type == "cuda" else canonical_emb.float()
            print(f"  Fixed generic negatives {list(NEGATIVE_PROMPTS)}: {canonical_emb.shape}")
        else:
            canon_cache = Path(args.canonical_embedding_cache or "checkpoints/siglip2_canonical_embeddings.pt")
            if not canon_cache.exists():
                raise FileNotFoundError(
                    f"Canonical embedding cache not found: {canon_cache}. "
                    "Use --protocol_preset vala_paper_2d to generate fixed generic negatives."
                )
            cdata = torch.load(canon_cache, map_location="cpu")
            canonical_emb = F.normalize(cdata["embeddings"].float(), dim=-1).to(device)
            canonical_emb = canonical_emb.half() if device.type == "cuda" else canonical_emb.float()
            print(f"  Loaded canonical embeddings from {canon_cache}: {canonical_emb.shape}")
        print(f"  Scoring: LERF-style relevancy")
    elif args.scoring == "softmax_scene":
        print("  Scoring: scene-category softmax")
    else:
        print("  Scoring: raw cosine similarity")

    # ------------------------------------------------------------------
    # 4. Optionally load rendering pipeline
    # ------------------------------------------------------------------
    render_pipeline = None
    lerf_datasets: Dict[str, LERFDataset] = {}
    if not args.gt_only:
        print("Loading rendering pipeline …")
        render_pipeline = load_render_pipeline(
            args.config,
            args.checkpoint,
            device,
            strict_checkpoint_contract=args.protocol_preset == "vala_paper_2d",
            # LERF feature evaluation never calls Gaussian RGB/SH rendering;
            # optional RGB guides are loaded from the source image directly.
            load_ply_rgb_features=False,
        )
        config = render_pipeline[5]
        fH = getattr(config, "feature_height", 30)
        fW = getattr(config, "feature_width", 40)
        # Build LERFDataset per scene for pose access
        for scene in scenes:
            scene_root_cand = resolve_lerf_scene_root(
                scene,
                getattr(config, "scene_root", ""),
            )
            feat_dir_cand = Path(args.gt_feature_dir or DEFAULT_GT_FEATURE_ROOT) / scene
            if not feat_dir_cand.exists():
                feat_dir_cand = Path(DEFAULT_GT_FEATURE_ROOT) / scene
            try:
                ds = LERFDataset(
                    scene_root=str(scene_root_cand),
                    feature_dir=str(feat_dir_cand),
                    annotation_dir=str(Path(args.label_dir) / scene),
                    feature_height=fH,
                    feature_width=fW,
                    allow_empty_features=True,
                )
                lerf_datasets[scene] = ds
            except Exception as e:
                logger.warning("Could not build LERFDataset for '%s': %s", scene, e)

    # ------------------------------------------------------------------
    # 5. Evaluate each scene
    # ------------------------------------------------------------------
    vis_root = Path(args.output_dir) / "visualisations" if args.save_vis else None
    all_results: Dict[str, Dict] = {}

    for scene in scenes:
        print(f"\n{'─' * 70}")
        print(f"  Scene: {scene}")
        print(f"{'─' * 70}")

        gt_feat_dir = None
        if args.rendered_only:
            gt_feat_dir = None
        elif args.gt_feature_dir:
            candidate = Path(args.gt_feature_dir)
            if candidate.name == scene or (candidate / "backbone").exists():
                gt_feat_dir = str(candidate)
            else:
                gt_feat_dir = str(candidate / scene)
        else:
            gt_feat_dir = str(Path(DEFAULT_GT_FEATURE_ROOT) / scene)

        # Verify GT feature dir exists (non-fatal if missing)
        if gt_feat_dir and not Path(gt_feat_dir).exists():
            backbone_cand = Path(gt_feat_dir) / "backbone"
            if not backbone_cand.exists():
                logger.warning("GT feature dir not found: %s", gt_feat_dir)
                gt_feat_dir = None

        region_feat_dir = None
        if args.region_feature_dir:
            region_candidate = Path(args.region_feature_dir)
            if (
                region_candidate.name == scene
                or (region_candidate / "backbone").exists()
            ):
                region_feat_dir = str(region_candidate)
            else:
                region_feat_dir = str(region_candidate / scene)
            if not Path(region_feat_dir).exists():
                raise FileNotFoundError(
                    f"typed region feature directory is missing: {region_feat_dir}"
                )

        prompt_mask_feature_dir = None
        if args.sam3_prompt_mask_head_feature_dir:
            prompt_candidate = Path(
                args.sam3_prompt_mask_head_feature_dir.format(scene=scene)
            )
            if (
                prompt_candidate.name == scene
                or (prompt_candidate / "backbone").is_dir()
            ):
                prompt_mask_feature_dir = str(prompt_candidate)
            else:
                prompt_mask_feature_dir = str(prompt_candidate / scene)
            if not Path(prompt_mask_feature_dir).is_dir():
                raise FileNotFoundError(
                    "feature-only SAM3 boundary directory is missing: "
                    f"{prompt_mask_feature_dir}"
                )

        prompt_mask_head = None
        prompt_mask_target_size = (240, 320)
        sam3_box_refiner = None
        if args.mask_refinement == "sam3_prompt_mask_head":
            prompt_head_path = args.sam3_prompt_mask_head_checkpoint.format(scene=scene)
            prompt_mask_head, prompt_mask_target_size = load_prompt_conditioned_mask_head(
                prompt_head_path,
                device,
            )
        elif args.mask_refinement == "sam3_box":
            from radio_gs.scripts.eval_lerf_direct_3d_selection import Sam3BoxMaskRefiner

            sam3_box_refiner = Sam3BoxMaskRefiner(
                checkpoint_path=args.sam3_checkpoint_path,
                device=str(device),
                confidence_threshold=args.sam3_confidence_threshold,
                resolution=args.sam3_resolution,
                amp_dtype=args.sam3_amp_dtype,
                box_padding_pixels=args.sam3_box_padding,
                min_initial_iou=args.sam3_min_initial_iou,
            )

        scene_results = evaluate_scene(
            scene=scene,
            label_dir=args.label_dir,
            proj=proj,
            text_embeddings=text_embeddings,
            categories=categories,
            device=device,
            gt_feature_dir=gt_feat_dir if not args.gt_only or gt_feat_dir else gt_feat_dir,
            region_feature_dir=region_feat_dir,
            region_score_weight=args.region_score_weight,
            render_pipeline=render_pipeline if not args.gt_only else None,
            lerf_dataset=lerf_datasets.get(scene),
            vis_dir=vis_root,
            iou_threshold=args.iou_threshold,
            threshold_mode=args.threshold_mode,
            threshold_mean_std_k=args.threshold_mean_std_k,
            threshold_min_ratio=args.threshold_min_ratio,
            threshold_max_ratio=args.threshold_max_ratio,
            canonical_emb=canonical_emb,
            temperature=args.relevancy_temp,
            scoring=args.scoring,
            heatmap_upsample=args.heatmap_upsample,
            eval_at_image_resolution=args.eval_at_image_resolution,
            localization_mode=args.localization_mode,
            localization_smoothing_kernel=args.localization_smoothing_kernel,
            readout_confidence_gate=args.readout_confidence_gate,
            readout_confidence_gamma=args.readout_confidence_gamma,
            save_overlay_vis=args.save_overlay_vis,
            save_per_query_vis=args.save_per_query_vis or args.save_overlay_vis,
            pred_mask_dir=Path(args.output_dir) / "pred_masks" if args.save_pred_masks else None,
            mask_refinement=args.mask_refinement,
            rgb_refinement_source=args.rgb_refinement_source,
            mask_refinement_iters=args.mask_refinement_iters,
            mask_refinement_dilate=args.mask_refinement_dilate,
            mask_refinement_erode=args.mask_refinement_erode,
            sam3_box_refiner=sam3_box_refiner,
            sam3_box_initial_refinement=args.sam3_box_initial_refinement,
            sam3_box_min_heatmap_mean_ratio=args.sam3_box_min_heatmap_mean_ratio,
            sam3_box_min_heatmap_mass_ratio=args.sam3_box_min_heatmap_mass_ratio,
            sam3_box_require_peak_in_refined=args.sam3_box_require_peak_in_refined,
            sam3_prompt_mask_head=prompt_mask_head,
            sam3_prompt_mask_head_target_size=prompt_mask_target_size,
            sam3_prompt_mask_head_logit_threshold=args.sam3_prompt_mask_head_logit_threshold,
            sam3_prompt_mask_head_min_initial_iou=args.sam3_prompt_mask_head_min_initial_iou,
            sam3_prompt_mask_head_max_initial_area_fraction=args.sam3_prompt_mask_head_max_initial_area_fraction,
            sam3_prompt_mask_head_min_refined_area_ratio=args.sam3_prompt_mask_head_min_refined_area_ratio,
            sam3_prompt_mask_head_max_refined_area_ratio=args.sam3_prompt_mask_head_max_refined_area_ratio,
            sam3_prompt_mask_head_support_dilate=args.sam3_prompt_mask_head_support_dilate,
            sam3_prompt_mask_head_coarse_dilate=args.sam3_prompt_mask_head_coarse_dilate,
            sam3_prompt_mask_head_coarse_threshold=args.sam3_prompt_mask_head_coarse_threshold,
            sam3_prompt_mask_head_min_heatmap_mean_ratio=args.sam3_prompt_mask_head_min_heatmap_mean_ratio,
            sam3_prompt_mask_head_min_heatmap_mass_ratio=args.sam3_prompt_mask_head_min_heatmap_mass_ratio,
            sam3_prompt_mask_head_require_peak_in_refined=args.sam3_prompt_mask_head_require_peak_in_refined,
            sam3_prompt_mask_head_initial_refinement=args.sam3_prompt_mask_head_initial_refinement,
            sam3_prompt_mask_head_apply_to=args.sam3_prompt_mask_head_apply_to,
            sam3_prompt_mask_head_feature_dir=prompt_mask_feature_dir,
            render_readout=args.render_readout,
            primitive_chunk_size=args.primitive_chunk_size,
            primitive_query_cache=args.primitive_query_cache,
            primitive_score_cache=args.primitive_score_cache,
            primitive_confidence=args.primitive_confidence,
            primitive_fallback_blend=args.primitive_fallback_blend,
            feature_contribution_gamma=args.feature_contribution_gamma,
            primitive_valid_normalization=args.primitive_valid_normalization,
            primitive_valid_coverage_power=args.primitive_valid_coverage_power,
            primitive_posterior_visibility_mass=(
                args.primitive_posterior_visibility_mass
            ),
            primitive_posterior_calibration_scale=(
                args.primitive_posterior_calibration_scale
            ),
            primitive_posterior_calibration_bias=(
                args.primitive_posterior_calibration_bias
            ),
        )
        all_results[scene] = scene_results

    # ------------------------------------------------------------------
    # 6. Print summary tables
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  LERF-OVS RESULTS SUMMARY")
    print("=" * 70)

    for mode in ("teacher", "rendered"):
        scene_metrics = [
            (s, get_lerf_mode_metrics(r, mode))
            for s, r in all_results.items()
            if get_lerf_mode_metrics(r, mode) is not None
        ]
        if not scene_metrics:
            continue

        print(f"\n  [{display_lerf_mode(mode)}]")
        print(f"  {'Scene':<20} {'Loc Acc':>10} {'mIoU':>10} {'Samples':>10}")
        print(f"  {'─' * 50}")

        for scene_name, m in scene_metrics:
            print(f"  {scene_name:<20} {m['loc_acc']:>10.4f} {m['miou']:>10.4f} "
                  f"{m['loc_total']:>10d}")
        aggregate = aggregate_lerf_mode_metrics([value for _, value in scene_metrics])
        print(f"  {'─' * 50}")
        print(
            f"  {'OVERALL (samples)':<20} "
            f"{aggregate['localization_accuracy']:>10.4f} "
            f"{aggregate['sample_micro_miou']:>10.4f} "
            f"{aggregate['sample_count']:>10d}"
        )
        if aggregate["category_macro_miou"] is not None:
            print(
                f"  {'CATEGORY MACRO':<20} {'':>10} "
                f"{aggregate['category_macro_miou']:>10.4f} {'':>10}"
            )

        # Per-category breakdown (across scenes)
        print(f"\n  Per-category breakdown ({canonical_lerf_mode(mode)}):")
        print(f"  {'Category':<30} {'Loc Acc':>10} {'mIoU':>10} {'N':>6}")
        print(f"  {'─' * 56}")
        for cat in categories:
            cat_locs: List[bool] = []
            cat_ious: List[float] = []
            for _, m in scene_metrics:
                ci = m["per_category"].get(cat, {})
                if ci.get("loc_acc") is not None and ci["n_samples"] > 0:
                    cat_locs.extend([ci["loc_acc"]] * ci["n_samples"])
                if ci.get("miou") is not None and ci["n_samples"] > 0:
                    cat_ious.append(ci["miou"])
            if not cat_locs:
                continue
            print(f"  {cat:<30} {np.mean(cat_locs):>10.4f} "
                  f"{np.mean(cat_ious):>10.4f} {len(cat_locs):>6d}")

    # ------------------------------------------------------------------
    # 7. Save JSON report
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance_paths = {
        "evaluator_source": str(Path(__file__).resolve()),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "text_embedding_cache": args.text_embedding_cache,
        "canonical_embedding_cache": args.canonical_embedding_cache,
        "primitive_query_cache": args.primitive_query_cache,
        "primitive_score_cache": args.primitive_score_cache,
        "primitive_confidence": args.primitive_confidence,
        "primitive_fallback_blend": args.primitive_fallback_blend,
    }
    if (
        args.text_encoder == "siglip2"
        and args.use_summary_head
        and not args.preprojected_text_features
    ):
        provenance_paths["summary_head_weights"] = args.summary_head_weights
    if args.sam3_prompt_mask_head_feature_dir:
        feature_root = Path(
            args.sam3_prompt_mask_head_feature_dir.format(scene=args.scene)
        )
        feature_manifest = feature_root / "canonical_render_manifest.json"
        if feature_manifest.is_file():
            provenance_paths["sam3_prompt_mask_head_feature_manifest"] = str(
                feature_manifest
            )

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {k: str(v) for k, v in vars(args).items()},
        "provenance": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for name, path in provenance_paths.items()
            if path and Path(path).exists()
        },
        "checkpoint_contract": (
            getattr(render_pipeline[5], "checkpoint_contract", {})
            if render_pipeline is not None
            else {}
        ),
        "prompt_templates": prompt_templates,
        "feature_observation_operator": {
            "type": (
                "front_to_back_posterior_visibility_mass"
                if args.primitive_posterior_visibility_mass
                else "normalized_front_to_back_contribution_power"
                if float(args.feature_contribution_gamma) != 1.0
                else "alpha_normalized_mean"
            ),
            "gamma": float(args.feature_contribution_gamma),
            "primitive_valid_normalization": bool(
                args.primitive_valid_normalization
            ),
            "semantic_score_formula": (
                "sum(w*v*s); residual_transmittance_is_background"
                if args.primitive_posterior_visibility_mass
                else "sum(w*v*s)/sum(w*v) * coverage**coverage_power"
                if args.primitive_valid_normalization
                else "sum(w*v*s)/sum(w)"
            ),
            "semantic_coverage_power": (
                float(args.primitive_valid_coverage_power)
                if args.primitive_valid_normalization
                else None
            ),
            "posterior_monotonic_calibration": {
                "family": "positive_scale_platt",
                "scale": float(args.primitive_posterior_calibration_scale),
                "bias": float(args.primitive_posterior_calibration_bias),
                "changes_proposal_selection": False,
            },
            "typed_text_readout": (
                {
                    "levels": ["primitive", "region_summary"],
                    "score_fusion": "fixed_convex_mean",
                    "region_weight": float(args.region_score_weight),
                }
                if args.region_feature_dir
                else {"levels": ["region_summary"]}
                if args.preprojected_text_features
                else {"levels": ["primitive"]}
            ),
            "query_dependent": False,
            "changes_geometry_or_alpha": False,
        },
        "categories": categories,
        "aggregates": {},
        "scenes": {},
    }
    for scene_name, scene_res in all_results.items():
        scene_report: Dict = {}
        for mode in iter_lerf_report_modes(scene_res):
            m = get_lerf_mode_metrics(scene_res, mode)
            if m is None:
                continue
            # Convert per_category values for JSON serialisation
            per_cat_json = {}
            for cat, info in m["per_category"].items():
                per_cat_json[cat] = {
                    "loc_acc": info["loc_acc"],
                    "miou": info["miou"],
                    "n_samples": info["n_samples"],
                }
            scene_report[mode] = {
                "loc_acc": m["loc_acc"],
                "miou": m["miou"],
                "loc_correct": m["loc_correct"],
                "loc_total": m["loc_total"],
                "n_iou_samples": m.get("n_iou_samples"),
                "localization_mode": m.get("localization_mode"),
                "localization_smoothing_kernel": m.get("localization_smoothing_kernel"),
                "per_category": per_cat_json,
            }
            for key in (
                "initial_miou",
                "delta_miou",
                "sam3_prompt_refinement_count",
                "sam3_prompt_attempt_count",
                "sam3_prompt_accept_count",
                "sam3_prompt_accept_rate",
                "sam3_prompt_mean_initial_overlap",
                "sam3_prompt_mean_refined_area_ratio",
                "sam3_prompt_fallback_reasons",
            ):
                if key in m:
                    scene_report[mode][key] = m[key]
            if mode == "teacher":
                scene_report["gt"] = scene_report[mode]
        report["scenes"][scene_name] = scene_report

    for mode in ("teacher", "rendered"):
        metrics = [
            get_lerf_mode_metrics(value, mode) for value in all_results.values()
        ]
        metrics = [value for value in metrics if value is not None]
        if not metrics:
            continue
        report["aggregates"][mode] = aggregate_lerf_mode_metrics(metrics)

    report_path = out_dir / "lerf_ovs_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()
