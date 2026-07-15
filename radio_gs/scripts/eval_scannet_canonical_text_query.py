#!/usr/bin/env python3
"""Evaluate canonical primitive semantics on the official ScanNet mesh domain.

The evaluator is intentionally calibration-free.  It projects the frozen,
query-independent primitive descriptors to ScanNet label vertices with a fixed
inverse-distance kNN readout, classifies them by cosine similarity to frozen
SigLIP2 text embeddings, and delegates metric computation to the same NYU40
implementation used by the legacy RADIO-GS ScanNet evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from radio_gs.scannet_constants import (
    NYU40_ID_TO_NAME,
    OPENGAUSSIAN_NYU40_CLASS_SPLITS,
    compute_split_metrics,
)
from radio_gs.scripts.eval_lerf_grounding import parse_prompt_templates
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import (
    _default_label_ply,
    _parse_splits,
    _read_label_ply,
)


DEFAULT_PROMPTS = (
    "{query}|a photo of a {query}|a 3d scan of a {query}|"
    "a point cloud of a {query}|an indoor scene containing a {query}"
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_text_cache(
    path: str | Path,
    *,
    class_names: list[str],
    prompt_templates: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Load an exact frozen cache without silently encoding or overwriting it."""

    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"frozen text cache not found: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid text cache payload: {cache_path}")
    queries = [str(value) for value in payload.get("queries", [])]
    templates = [str(value) for value in payload.get("prompt_templates", [])]
    if queries != list(class_names):
        raise ValueError(
            f"frozen text cache query mismatch: expected {class_names}, got {queries}"
        )
    if templates != list(prompt_templates):
        raise ValueError(
            f"frozen text cache template mismatch: expected {prompt_templates}, got {templates}"
        )
    if payload.get("text_encoder") not in (None, "siglip2"):
        raise ValueError("frozen text cache is not a SigLIP2 cache")
    embeddings = torch.as_tensor(payload.get("embeddings"))
    if embeddings.shape != (len(class_names), 1536):
        raise ValueError(
            "frozen SigLIP2 text embeddings must be "
            f"[{len(class_names)},1536], got {tuple(embeddings.shape)}"
        )
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("frozen text cache contains NaN or infinity")
    return F.normalize(embeddings.float(), dim=-1).to(device)


def load_primitive_semantic_cache(
    path: str | Path,
    *,
    allow_mpr_oracle: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    """Load a query-free, official-SigLIP2 primitive semantic cache."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported primitive semantic cache: {path}")
    required = {"xyz", "valid", "metadata"}
    if not required.issubset(payload):
        raise ValueError(f"semantic cache lacks keys: {sorted(required - set(payload))}")
    features_value = payload.get("summary_features", payload.get("features"))
    if features_value is None:
        raise ValueError("semantic cache lacks summary_features/features")
    xyz = torch.as_tensor(payload["xyz"]).float().cpu()
    valid = torch.as_tensor(payload["valid"]).bool().cpu()
    features = torch.as_tensor(features_value).cpu()
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("semantic cache metadata must be a mapping")
    count = int(xyz.shape[0]) if xyz.ndim == 2 else -1
    if xyz.ndim != 2 or xyz.shape[1] != 3 or valid.shape != (count,):
        raise ValueError("semantic cache xyz/valid rows are malformed")
    if features.ndim != 2 or features.shape[0] != count:
        raise ValueError("semantic cache features do not align with geometry")
    if not bool(valid.any()):
        raise ValueError("semantic cache has no valid primitive rows")
    if not bool(torch.isfinite(xyz).all()) or not bool(torch.isfinite(features).all()):
        raise ValueError("semantic cache contains NaN or infinity")
    source = str(metadata.get("source", ""))
    if source not in {
        "canonical_radio_primitive_neighborhood",
        "mpr_radio_primitive_neighborhood",
    }:
        raise ValueError(f"unsupported semantic cache source: {source}")
    if source.startswith("mpr_") and not allow_mpr_oracle:
        raise ValueError("MPR semantic caches are oracle diagnostics, not method outputs")
    if metadata.get("official_summary_head") is not True:
        raise ValueError("semantic cache must use the official SigLIP2 summary head")
    if metadata.get("custom_text_projection") is not False:
        raise ValueError("semantic cache must not use a custom text projection")
    for forbidden in ("benchmark_images_opened", "benchmark_masks_opened", "text_queries_opened"):
        if metadata.get(forbidden) is not False:
            raise ValueError(f"semantic cache must be query-free ({forbidden}=False)")
    return xyz, valid, features, metadata


@torch.no_grad()
def project_primitive_semantics_to_points(
    primitive_xyz: torch.Tensor,
    primitive_valid: torch.Tensor,
    primitive_features: torch.Tensor,
    query_xyz: np.ndarray,
    *,
    k: int = 8,
    distance_epsilon: float = 1e-4,
    chunk_size: int = 2048,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Lift normalized primitive descriptors to arbitrary 3-D query points."""

    if k <= 0 or distance_epsilon <= 0 or chunk_size <= 0:
        raise ValueError("k, distance_epsilon, and chunk_size must be positive")
    xyz = torch.as_tensor(primitive_xyz).float().cpu()
    valid = torch.as_tensor(primitive_valid).bool().cpu()
    features = torch.as_tensor(primitive_features).cpu()
    points = np.asarray(query_xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("primitive_xyz and query_xyz must be [N,3]")
    if valid.shape != (xyz.shape[0],) or features.ndim != 2 or features.shape[0] != xyz.shape[0]:
        raise ValueError("primitive validity/features must align with primitive_xyz")
    global_rows = torch.where(valid)[0]
    active_k = min(int(k), int(global_rows.numel()))
    tree = cKDTree(xyz[global_rows].numpy())
    distances, local_indices = tree.query(points, k=active_k, workers=-1)
    distances = np.asarray(distances, dtype=np.float32)
    local_indices = np.asarray(local_indices, dtype=np.int64)
    if active_k == 1:
        distances = distances[:, None]
        local_indices = local_indices[:, None]
    weights = 1.0 / np.maximum(distances, float(distance_epsilon))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    global_indices = global_rows.numpy()[local_indices]

    target_device = torch.device(device)
    output = torch.empty(points.shape[0], features.shape[1], dtype=torch.float16)
    for start in range(0, points.shape[0], int(chunk_size)):
        stop = min(points.shape[0], start + int(chunk_size))
        rows = torch.from_numpy(global_indices[start:stop]).long()
        neighbor_features = features[rows].to(target_device).float()
        neighbor_weights = torch.from_numpy(weights[start:stop]).to(target_device).float()
        fused = (neighbor_features * neighbor_weights.unsqueeze(-1)).sum(dim=1)
        output[start:stop] = F.normalize(fused, dim=-1, eps=1e-8).half().cpu()
    return output


def evaluate(
    *,
    scene: str,
    label_ply: str | Path,
    semantic_cache: str | Path,
    split_text_embeddings: Mapping[str, torch.Tensor],
    split_names: list[str],
    projection_k: int,
    distance_epsilon: float,
    chunk_size: int,
    device: torch.device,
    allow_mpr_oracle: bool = False,
) -> dict[str, Any]:
    primitive_xyz, primitive_valid, primitive_features, metadata = (
        load_primitive_semantic_cache(
            semantic_cache,
            allow_mpr_oracle=allow_mpr_oracle,
        )
    )
    mesh_xyz, gt_labels = _read_label_ply(label_ply)
    mesh_features = project_primitive_semantics_to_points(
        primitive_xyz,
        primitive_valid,
        primitive_features,
        mesh_xyz,
        k=projection_k,
        distance_epsilon=distance_epsilon,
        chunk_size=chunk_size,
        device=device,
    )
    results: dict[str, dict[str, Any]] = {}
    for split in split_names:
        class_ids = OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]
        text = F.normalize(split_text_embeddings[split].float().to(device), dim=-1)
        pred_parts: list[np.ndarray] = []
        for start in range(0, mesh_features.shape[0], int(chunk_size)):
            stop = min(mesh_features.shape[0], start + int(chunk_size))
            visual = mesh_features[start:stop].float().to(device)
            indices = (visual @ text.T).argmax(dim=-1).cpu().numpy()
            pred_parts.append(np.asarray(class_ids, dtype=np.int32)[indices])
        pred_labels = np.concatenate(pred_parts)
        results[split] = compute_split_metrics(pred_labels, gt_labels, class_ids)
    return {
        "schema_version": 1,
        "scene": scene,
        "label_ply": str(label_ply),
        "semantic_cache": str(semantic_cache),
        "semantic_cache_sha256": _sha256_file(semantic_cache),
        "semantic_source": str(metadata.get("source", "")),
        "num_mesh_vertices": int(mesh_xyz.shape[0]),
        "num_primitives": int(primitive_xyz.shape[0]),
        "num_valid_primitives": int(primitive_valid.sum()),
        "protocol": {
            "evaluation_domain": "official_scannet_label_mesh_vertices",
            "primitive_to_mesh": "inverse_distance_knn",
            "projection_k": int(projection_k),
            "distance_epsilon": float(distance_epsilon),
            "classification": "normalized_cosine_argmax",
            "text_encoder": "official_siglip2_g",
            "logit_calibration": "none",
            "spatial_postprocess": "none",
            "ground_truth_usage": "metrics_only",
        },
        "splits": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="scene0000_00")
    parser.add_argument("--prepared-root", default="dataset/scannet_og")
    parser.add_argument("--label-ply", default="")
    parser.add_argument("--semantic-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--class-splits", default="19,15,10")
    parser.add_argument("--projection-k", type=int, default=8)
    parser.add_argument("--distance-epsilon", type=float, default=1e-4)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--prompt-templates", default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--text-embedding-cache",
        default="checkpoints/siglip2_scannet_og_text_embeddings_ens5.pt",
        help="Base path; _split{split} is inserted before the suffix.",
    )
    parser.add_argument("--allow-mpr-oracle", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    split_names = _parse_splits(args.class_splits)
    templates = parse_prompt_templates(args.prompt_templates)
    text_by_split: dict[str, torch.Tensor] = {}
    for split in split_names:
        class_names = [
            NYU40_ID_TO_NAME[class_id]
            for class_id in OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]
        ]
        base = Path(args.text_embedding_cache)
        cache_path = base.with_name(f"{base.stem}_split{split}{base.suffix}")
        text_by_split[split] = load_frozen_text_cache(
            cache_path,
            class_names=class_names,
            prompt_templates=templates,
            device=device,
        )
    label_ply = args.label_ply or _default_label_ply(Path(args.prepared_root), args.scene)
    report = evaluate(
        scene=args.scene,
        label_ply=label_ply,
        semantic_cache=args.semantic_cache,
        split_text_embeddings=text_by_split,
        split_names=split_names,
        projection_k=args.projection_k,
        distance_epsilon=args.distance_epsilon,
        chunk_size=args.chunk_size,
        device=device,
        allow_mpr_oracle=args.allow_mpr_oracle,
    )
    report["protocol"]["prompt_templates"] = templates
    report["protocol"]["class_aliases"] = "none"
    report["protocol"]["text_embedding_cache_base"] = str(args.text_embedding_cache)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
