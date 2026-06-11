"""Evaluate text grounding on the LERF-OVS benchmark.

Pipeline
--------
1. Load LERF-OVS polygon annotations for the requested scene(s).
2. Generate SigLIP2 text embeddings for every object category on-the-fly
   (falls back to a pre-computed bank if the ``transformers`` model is
   unavailable).
3. For each labeled frame:
   a. **Teacher mode** – load pre-extracted RADIO 1280-d features from real RGB.
   b. **Rendered mode** – render latent features from the trained 3DGS
      feature field and decode to 1280-d.
4. Project 1280-d features → SigLIP2 1536-d using a frozen projection head.
5. Compute cosine-similarity heatmaps for every query.
6. Evaluate:
   - **Localization accuracy** – argmax pixel inside GT polygon?
   - **mIoU** – binarised heatmap (0.5 × max) vs. GT polygon mask.
7. Print per-scene / per-query tables and save a JSON report.

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
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

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
from radio_gs.utils.checkpoint_io import load_trusted_checkpoint

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Text-embedding generation (SigLIP2 via ``transformers``)
# ---------------------------------------------------------------------------
_SIGLIP2_MODEL_NAME = "google/siglip2-giant-opt-patch16-384"


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

    try:
        snapshot = Path(
            snapshot_download(
                model_name,
                allow_patterns=[
                    "model.safetensors",
                    "model-*.safetensors",
                    "model.safetensors.index.json",
                ],
            )
        )
    except Exception as exc:
        logger.warning("Could not locate SigLIP2 safetensors for text-head restore: %s", exc)
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
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name)
        inputs = processor(text=queries, padding="max_length", return_tensors="pt")
    except Exception as exc:
        logger.warning("AutoProcessor text tokenization failed, falling back to AutoTokenizer: %s", exc)
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        inputs = tokenizer(
            queries,
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
        if not missing:
            emb = torch.stack([bank[q] for q in queries])
            return F.normalize(emb.float(), dim=-1).to(device)

    # Try on-the-fly generation
    try:
        emb = encode_text_siglip2(queries, device)
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"queries": queries, "embeddings": emb.cpu()}, cache_path)
            logger.info("Cached text embeddings → %s", cache_path)
        return emb
    except Exception as exc:
        logger.warning("On-the-fly SigLIP2 text encoding failed: %s", exc)

    # Fallback: pre-computed bank
    if cache_path and Path(cache_path).exists():
        data = torch.load(cache_path, map_location="cpu")
        bank = {q: e for q, e in zip(data["queries"], data["embeddings"])}
        missing = [q for q in queries if q not in bank]
        if missing:
            raise RuntimeError(
                f"Cached text-embedding bank at {cache_path} is missing "
                f"queries: {missing}.  Please ensure the SigLIP2 model is "
                f"downloadable or provide a complete cache."
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
                objects.append({"category": cat, "polygons": polys})
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
    """Project ``[1, 1280, H, W]`` to L2-normalised ``[1, 1536, H, W]``."""
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
    siglip = F.normalize(siglip, dim=-1)
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
    _, D, H, W = visual_feat.shape
    vis_flat = visual_feat.squeeze(0).reshape(D, H * W)  # [D, HW]

    if scoring == "softmax_scene" and all_scene_emb is not None and active_scene_indices is not None:
        # Softmax over all scene categories (matches LangSplat evaluation protocol)
        all_sim = all_scene_emb @ vis_flat  # [K, HW]
        all_prob = torch.softmax(all_sim * temperature, dim=0)  # [K, HW]
        return all_prob[active_scene_indices].reshape(-1, H, W)

    sim = text_emb @ vis_flat  # [N, HW]

    if scoring == "relevancy" and canonical_emb is not None and canonical_emb.shape[0] > 0:
        canon_sim = canonical_emb @ vis_flat  # [M, HW]
        canon_max = canon_sim.max(dim=0, keepdim=True).values  # [1, HW]
        sim_scaled = sim / temperature
        canon_scaled = canon_max.expand_as(sim) / temperature
        max_val = torch.maximum(sim_scaled, canon_scaled)
        relevancy = torch.exp(sim_scaled - max_val) / (
            torch.exp(sim_scaled - max_val) + torch.exp(canon_scaled - max_val) + 1e-8
        )
        return relevancy.reshape(-1, H, W)

    return sim.reshape(-1, H, W)


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
    if mode == "fixed":
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
    pred = np.asarray(mask).astype(bool)
    if pred.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {pred.shape}")
    if not pred.any():
        return pred.copy()
    peak_y = min(max(int(peak_yx[0]), 0), pred.shape[0] - 1)
    peak_x = min(max(int(peak_yx[1]), 0), pred.shape[1] - 1)
    num_labels, labels = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return pred.copy()
    peak_label = int(labels[peak_y, peak_x])
    if peak_label > 0:
        return labels == peak_label

    # If the argmax is just outside the thresholded support, snap to the nearest
    # foreground component.  This keeps the readout peak-conditioned without
    # silently falling back to a global largest-component heuristic.
    ys, xs = np.nonzero(pred)
    if ys.size == 0:
        return pred.copy()
    nearest_idx = int(np.argmin((ys - peak_y) ** 2 + (xs - peak_x) ** 2))
    nearest_label = int(labels[int(ys[nearest_idx]), int(xs[nearest_idx])])
    if nearest_label <= 0:
        return pred.copy()
    return labels == nearest_label


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


def load_render_pipeline(config_path: str, checkpoint_path: str, device: torch.device):
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
    if ply_path:
        model.load_from_ply(ply_path)
    model = model.to(device).eval()
    use_2dgs = resolve_use_2dgs(config, ply_path)

    codec = R["build_feature_codec"](
        input_dim=getattr(config, "radio_feature_dim", 1280),
        bottleneck_dim=getattr(config, "bottleneck_dim", 64),
        codec_type=getattr(config, "codec_type", "hcd"),
        dual_stream=getattr(config, "dual_stream", True),
        symmetric_decoder=getattr(config, "symmetric_decoder", False),
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

    ckpt = load_trusted_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    codec.load_state_dict(ckpt["codec_state_dict"], strict=False)
    if "sharpener_state_dict" in ckpt:
        sharpener.load_state_dict(ckpt["sharpener_state_dict"], strict=False)
    if refiner is not None and "refiner_state_dict" in ckpt:
        refiner.load_state_dict(ckpt["refiner_state_dict"], strict=False)

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
    readout_confidence_gate: str = "none",
    readout_confidence_gamma: float = 1.0,
    save_overlay_vis: bool = False,
    save_per_query_vis: bool = False,
    pred_mask_dir: Optional[Path] = None,
    mask_refinement: str = "none",
    mask_refinement_iters: int = 1,
    mask_refinement_dilate: int = 5,
    mask_refinement_erode: int = 2,
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
) -> Dict:
    """Evaluate one LERF-OVS scene.

    Exactly one of *gt_feature_dir* or *render_pipeline* must be provided (or
    both for a joint GT + rendered evaluation).

    Returns:
        dict with ``loc_acc``, ``miou``, per-category breakdowns, etc.
    """
    frame_annotations, scene_categories, img_h, img_w = load_lerf_ovs_labels(label_dir, scene)

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
        else:
            scene_root_hint = ""
        canonical_mode = canonical_lerf_mode(mode)
        mode_mask_refinement = mask_refinement
        if (
            mask_refinement == "sam3_prompt_mask_head"
            and sam3_prompt_mask_head_apply_to == "rendered"
            and canonical_mode != "rendered"
        ):
            mode_mask_refinement = "none"
        mode_confidence_gate = readout_confidence_gate if canonical_mode == "rendered" else "none"
        for frame_id, frame_objects in tqdm(
            sorted(frame_annotations.items()),
            desc=f"  {scene}/{canonical_mode}",
            leave=False,
        ):
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
                render_aux = None
                if mode_confidence_gate != "none":
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

            # --- project to SigLIP2 ---
            siglip_feat = project_to_siglip2(feat_1280.half(), proj)  # [1, 1536, H, W]

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

            # Build all-scene embeddings for softmax_scene scoring
            all_scene_emb = None
            active_scene_idx = None
            if scoring == "softmax_scene":
                scene_emb_global_idx = [cat_to_idx[c] for c in sorted(scene_cat_indices.keys())]
                all_scene_emb = text_embeddings[scene_emb_global_idx].to(device)
                scene_cats_sorted = sorted(scene_cat_indices.keys())
                active_scene_idx = [scene_cats_sorted.index(c) for c in active_cats]

            heatmaps = compute_relevancy_heatmap(
                siglip_feat, active_emb,
                canonical_emb=canonical_emb,
                temperature=temperature,
                scoring=scoring,
                all_scene_emb=all_scene_emb,
                active_scene_indices=active_scene_idx,
            )
            if mode_confidence_gate != "none":
                heatmaps = apply_readout_confidence_gate(
                    heatmaps,
                    render_aux,
                    gate=mode_confidence_gate,
                    gamma=readout_confidence_gamma,
                )

            # Sub-pixel localization via heatmap upsampling
            if heatmap_upsample > 1:
                K = heatmaps.shape[0]
                heatmaps = F.interpolate(
                    heatmaps.unsqueeze(0),  # [1, K, fH, fW]
                    scale_factor=heatmap_upsample,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)  # [K, fH*up, fW*up]

            fH, fW = heatmaps.shape[1], heatmaps.shape[2]
            prompt_mask_feature = None
            if mode_mask_refinement == "sam3_prompt_mask_head" and sam3_prompt_mask_head is not None:
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
            if mode_mask_refinement != "none" or save_overlay_vis:
                rgb_image_for_masks = load_lerf_rgb_frame(scene, frame_id, scene_root_hint)

            hm_vis: Dict[str, np.ndarray] = {}
            gt_vis: Dict[str, np.ndarray] = {}
            for ki, cat in enumerate(active_cats):
                hm = heatmaps[ki]  # [fH, fW]
                gt_full = gt_masks_full[cat]
                gt_feat = gt_masks_feat[cat]
                initial_pred_for_save: Optional[np.ndarray] = None
                pred_for_save: Optional[np.ndarray] = None

                if gt_full.sum() == 0:
                    continue

                # Localization: map feature-level argmax to image coordinates
                is_correct = localization_accuracy(hm, gt_full)
                loc_correct += int(is_correct)
                loc_total += 1
                per_cat_loc[cat].append(is_correct)

                # mIoU at feature resolution by default; optional RGB refinement
                # evaluates the refined mask at image resolution without using GT.
                if mode_mask_refinement == "sam3_prompt_mask_head" and prompt_mask_feature is not None:
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

        loc_acc = loc_correct / max(loc_total, 1)
        miou = float(np.mean(ious)) if ious else 0.0

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
            "miou": miou,
            "loc_correct": loc_correct,
            "loc_total": loc_total,
            "n_iou_samples": len(ious),
            "per_category": per_cat_summary,
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
    parser.add_argument("--gt_only", action="store_true",
                        help="Evaluate teacher/oracle RADIO features only (skip rendered mode)")
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
    parser.add_argument("--text_embedding_cache", default=None,
                        help="Path to cache/load pre-computed text embeddings")
    parser.add_argument("--prompt_templates", default=DEFAULT_PROMPT_TEMPLATES,
                        help="Prompt templates separated by '|'. Use {query} as placeholder")
    # Evaluation
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="Threshold ratio (fraction of max) for mIoU binarisation")
    parser.add_argument("--threshold_mode", choices=["fixed", "mean_std"], default="fixed",
                        help="GT-free mask threshold rule. 'fixed' uses --iou_threshold; "
                             "'mean_std' uses mean+k*std of the peak-normalized heatmap.")
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
    parser.add_argument("--mask_refinement", choices=["none", "rgb_grabcut", "peak_component", "peak_component_rgb_grabcut", "sam3_prompt_mask_head"], default="none",
                        help="Optional GT-free RGB boundary snapping after heatmap binarisation")
    parser.add_argument("--mask_refinement_iters", type=int, default=1,
                        help="GrabCut iterations for rgb_grabcut mask refinement")
    parser.add_argument("--mask_refinement_dilate", type=int, default=5,
                        help="Pixel dilation radius defining rgb_grabcut support band")
    parser.add_argument("--mask_refinement_erode", type=int, default=2,
                        help="Pixel erosion radius defining sure foreground for rgb_grabcut")
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
    # Hardware
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device id")

    args = parser.parse_args()
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

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    scenes = LERF_OVS_SCENES if args.scene == "all" else (args.scene,)

    print("=" * 70)
    print("  LERF-OVS Text Grounding Evaluation (RADIO-GS → SigLIP2)")
    print("=" * 70)
    print(f"  Scenes:     {', '.join(scenes)}")
    print(f"  Label dir:  {args.label_dir}")
    print(f"  Mode:       {'Teacher only' if args.gt_only else 'Teacher + Rendered'}")
    print(f"  IoU thresh: {args.iou_threshold}")
    print(f"  Thr mode:   {args.threshold_mode}")
    print(f"  Heatmap ↑:  {args.heatmap_upsample}×")
    print(f"  Conf gate:  {args.readout_confidence_gate} (γ={args.readout_confidence_gamma})")
    print(f"  Mask ref.:  {args.mask_refinement}")
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

    # ------------------------------------------------------------------
    # 2. Load SigLIP2 projection model
    # ------------------------------------------------------------------
    if args.use_summary_head:
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
    print("Generating SigLIP2 text embeddings …")
    t0 = time.time()
    text_embeddings = load_or_generate_prompt_ensemble_embeddings(
        categories,
        device,
        cache_path=args.text_embedding_cache,
        prompt_templates=prompt_templates,
    )  # [N, 1536]
    text_embeddings = text_embeddings.half() if device.type == "cuda" else text_embeddings.float()
    print(f"  {text_embeddings.shape[0]} embeddings ({text_embeddings.shape[1]}-d), "
          f"{time.time() - t0:.1f}s")

    # Generate canonical phrase embeddings for relevancy scoring
    canonical_emb = None
    if args.scoring == "relevancy":
        canon_cache = Path("checkpoints/siglip2_canonical_embeddings.pt")
        if canon_cache.exists():
            cdata = torch.load(canon_cache, map_location="cpu")
            canonical_emb = F.normalize(cdata["embeddings"].float(), dim=-1).to(device)
            canonical_emb = canonical_emb.half() if device.type == "cuda" else canonical_emb.float()
            print(f"  Loaded canonical embeddings from {canon_cache}: {canonical_emb.shape}")
        else:
            # Fallback: use mean of all category embeddings as canonical
            canonical_emb = F.normalize(
                text_embeddings.float().mean(dim=0, keepdim=True), dim=-1
            ).to(device)
            canonical_emb = canonical_emb.half() if device.type == "cuda" else canonical_emb.float()
            print(f"  Using mean-category canonical embedding: {canonical_emb.shape}")
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
        render_pipeline = load_render_pipeline(args.config, args.checkpoint, device)
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
        if args.gt_feature_dir:
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

        prompt_mask_head = None
        prompt_mask_target_size = (240, 320)
        if args.mask_refinement == "sam3_prompt_mask_head":
            prompt_head_path = args.sam3_prompt_mask_head_checkpoint.format(scene=scene)
            prompt_mask_head, prompt_mask_target_size = load_prompt_conditioned_mask_head(
                prompt_head_path,
                device,
            )

        scene_results = evaluate_scene(
            scene=scene,
            label_dir=args.label_dir,
            proj=proj,
            text_embeddings=text_embeddings,
            categories=categories,
            device=device,
            gt_feature_dir=gt_feat_dir if not args.gt_only or gt_feat_dir else gt_feat_dir,
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
            readout_confidence_gate=args.readout_confidence_gate,
            readout_confidence_gamma=args.readout_confidence_gamma,
            save_overlay_vis=args.save_overlay_vis,
            save_per_query_vis=args.save_per_query_vis or args.save_overlay_vis,
            pred_mask_dir=Path(args.output_dir) / "pred_masks" if args.save_pred_masks else None,
            mask_refinement=args.mask_refinement,
            mask_refinement_iters=args.mask_refinement_iters,
            mask_refinement_dilate=args.mask_refinement_dilate,
            mask_refinement_erode=args.mask_refinement_erode,
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

        agg_loc_correct = 0
        agg_loc_total = 0
        agg_ious: List[float] = []

        for scene_name, m in scene_metrics:
            print(f"  {scene_name:<20} {m['loc_acc']:>10.4f} {m['miou']:>10.4f} "
                  f"{m['loc_total']:>10d}")
            agg_loc_correct += m["loc_correct"]
            agg_loc_total += m["loc_total"]

            # Accumulate per-category IoUs for overall mean
            for cat_info in m["per_category"].values():
                if cat_info["miou"] is not None:
                    agg_ious.append(cat_info["miou"])

        overall_loc = agg_loc_correct / max(agg_loc_total, 1)
        overall_miou = float(np.mean(agg_ious)) if agg_ious else 0.0
        print(f"  {'─' * 50}")
        print(f"  {'OVERALL':<20} {overall_loc:>10.4f} {overall_miou:>10.4f} "
              f"{agg_loc_total:>10d}")

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
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {k: str(v) for k, v in vars(args).items()},
        "prompt_templates": prompt_templates,
        "categories": categories,
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

    report_path = out_dir / "lerf_ovs_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()
