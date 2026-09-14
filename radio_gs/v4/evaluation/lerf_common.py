"""Method-neutral LERF data and metric utilities for the isolated v4 path."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.carrier import Camera


GENERIC_NEGATIVES = ("object", "things", "stuff", "texture")


def frame_index(value: str) -> int:
    matches = re.findall(r"\d+", value)
    if not matches:
        raise ValueError(f"no frame index in {value!r}")
    return int(matches[-1])


def validate_source_authority(
    authority: dict[str, Any], scene_label: str
) -> list[int]:
    if authority.get("contract") != "sam3-query-free-source-rgb-authority-v1":
        raise ValueError("source RGB authority contract differs")
    if str(authority.get("scene")) != scene_label:
        raise ValueError("source RGB authority scene differs")
    policy = authority.get("information_policy", {})
    required_false = (
        "benchmark_ground_truth_used",
        "query_text_used",
        "target_or_evaluation_rgb_used",
    )
    if any(policy.get(key) is not False for key in required_false):
        raise ValueError("source RGB authority violates the information policy")
    if policy.get("registered_source_rgb_only") is not True:
        raise ValueError("source RGB authority is not restricted to registered source views")
    frames = [frame_index(str(item["image_id"])) for item in authority.get("images", [])]
    if not frames or len(frames) != len(set(frames)):
        raise ValueError("source RGB authority frames are empty or duplicated")
    return sorted(frames)


def validate_semantic_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    authority: dict[str, Any],
    feature_dir: Path,
    source_frames: list[int],
) -> list[int]:
    backbone = manifest.get("features", {}).get("backbone", {})
    grid = backbone.get("grid")
    if (
        backbone.get("dim") != 1280
        or not isinstance(grid, list)
        or len(grid) != 2
        or any(not isinstance(value, int) or value <= 0 for value in grid)
    ):
        raise ValueError("semantic frame manifest lacks the expected RADIO backbone")
    binding = authority.get("construction", {}).get("frame_manifest", {})
    if binding.get("sha256") != sha256_file(manifest_path):
        raise ValueError("semantic frame manifest differs from source authority binding")
    expected_dir = (manifest_path.parent / str(backbone.get("subdir", ""))).resolve()
    if feature_dir != expected_dir:
        raise ValueError("semantic feature directory differs from its manifest")
    records = manifest.get("frames", [])
    frames = [int(record["frame_idx"]) for record in records]
    if not frames or len(frames) != len(set(frames)):
        raise ValueError("semantic frame manifest frames are empty or duplicated")
    excluded = {frame_index(str(value)) for value in manifest.get("excluded_image_names", [])}
    if excluded & set(frames):
        raise ValueError("semantic frame manifest includes an explicitly excluded frame")
    if not set(source_frames).issubset(frames):
        raise ValueError("semantic frame manifest omits sealed source frames")
    return sorted(frames)


def load_dense_feature(
    path: Path,
    *,
    expected_channels: int,
    expected_height: int | None = None,
    expected_width: int | None = None,
) -> torch.Tensor:
    value = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise ValueError(f"semantic feature must be a [D,H,W] tensor: {path}")
    actual = tuple(int(size) for size in value.shape)
    if actual[0] != expected_channels:
        raise ValueError(f"semantic feature channel mismatch at {path}: {actual[0]}")
    if expected_height is not None and actual[1] != expected_height:
        raise ValueError(f"semantic feature height mismatch at {path}: {actual[1]}")
    if expected_width is not None and actual[2] != expected_width:
        raise ValueError(f"semantic feature width mismatch at {path}: {actual[2]}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"semantic feature contains NaN or Inf: {path}")
    return value


def canonicalize_siglip2_query(query: str) -> str:
    import string
    value = str(query).replace("_", " ").translate(str.maketrans("", "", string.punctuation)).lower()
    return " ".join(value.split())


def load_text_cache(
    path: Path, required_queries: list[str], device: torch.device
) -> torch.Tensor:
    payload = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    if payload.get("text_encoder") != "siglip2":
        raise ValueError(f"text cache is not a SigLIP2 bank: {path}")
    queries = [str(value) for value in payload.get("queries", [])]
    canonical = [canonicalize_siglip2_query(query) for query in queries]
    if canonical != queries and payload.get("canonical_queries") != canonical:
        raise ValueError("text cache lacks explicit canonical query provenance; regenerate with the official text encoder")
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError(f"text cache embeddings must be a matrix: {path}")
    if len(queries) != embeddings.shape[0] or len(queries) != len(set(queries)):
        raise ValueError(f"text cache queries are misaligned or duplicated: {path}")
    if embeddings.shape[1] != 1536 or not bool(torch.isfinite(embeddings).all()):
        raise ValueError(f"text cache embeddings have the wrong dimension or values: {path}")
    if bool((embeddings.float().norm(dim=-1) <= 1e-8).any()):
        raise ValueError("text cache contains zero embeddings")
    bank = dict(zip(queries, embeddings))
    missing = [query for query in required_queries if query not in bank]
    if missing:
        raise ValueError(f"text cache lacks exact queries: {missing}")
    return F.normalize(
        torch.stack([bank[query] for query in required_queries]).float(), dim=-1
    ).to(device)


def accumulate_surface_features(
    feature_sum: torch.Tensor,
    feature_mass: torch.Tensor,
    features: torch.Tensor,
    projection: Any,
    *,
    channel_chunk_size: int = 192,
) -> None:
    if feature_sum.ndim != 2 or feature_mass.shape != (feature_sum.shape[0],):
        raise ValueError("surface feature accumulators must have shapes [E,D] and [E]")
    features = torch.as_tensor(features, dtype=torch.float32, device=feature_sum.device)
    if features.ndim != 3 or features.shape[0] != feature_sum.shape[1]:
        raise ValueError("dense features must have shape [D,H,W]")
    if not bool(torch.isfinite(features).all()) or channel_chunk_size <= 0:
        raise ValueError("dense features and channel chunk size are invalid")
    element_ids = torch.as_tensor(
        projection.element_ids, dtype=torch.long, device=feature_sum.device
    )
    pixel_ids = torch.as_tensor(
        projection.pixel_ids, dtype=torch.long, device=feature_sum.device
    )
    weights = torch.as_tensor(
        projection.weights, dtype=torch.float32, device=feature_sum.device
    )
    if element_ids.ndim != 1 or pixel_ids.shape != element_ids.shape or weights.shape != element_ids.shape:
        raise ValueError("projection samples must be aligned vectors")
    if element_ids.numel():
        if int(element_ids.min()) < 0 or int(element_ids.max()) >= feature_sum.shape[0]:
            raise ValueError("projection contains an out-of-range surface element")
        if int(pixel_ids.min()) < 0 or int(pixel_ids.max()) >= features.shape[1] * features.shape[2]:
            raise ValueError("projection contains an out-of-range raster pixel")
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("projection weights must be finite and non-negative")
    else:
        return
    feature_mass.scatter_add_(0, element_ids, weights)
    pixels = features.flatten(1).T
    for start in range(0, feature_sum.shape[1], channel_chunk_size):
        stop = min(start + channel_chunk_size, feature_sum.shape[1])
        feature_sum[:, start:stop].index_add_(
            0, element_ids, pixels[pixel_ids, start:stop] * weights[:, None]
        )


def retain_top_query_tokens(probability: torch.Tensor, maximum_tokens: int) -> torch.Tensor:
    probability = torch.as_tensor(probability, dtype=torch.float32)
    if probability.ndim != 2 or maximum_tokens <= 0:
        raise ValueError("probability must be [Q,K] and maximum_tokens positive")
    if probability.shape[1] <= maximum_tokens:
        return probability
    _, indices = probability.topk(maximum_tokens, dim=-1)
    retained = torch.zeros_like(probability)
    retained.scatter_(1, indices, probability.gather(1, indices))
    return retained


def prototype_max_token_posterior(
    prototypes: torch.Tensor,
    prototype_token_ids: torch.Tensor,
    queries: torch.Tensor,
    negatives: torch.Tensor,
    *,
    num_tokens: int,
    temperature: float = 0.03,
    null_similarity: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    prototypes = F.normalize(torch.as_tensor(prototypes, dtype=torch.float32), dim=-1)
    queries = F.normalize(
        torch.as_tensor(queries, dtype=torch.float32, device=prototypes.device), dim=-1
    )
    negatives = F.normalize(
        torch.as_tensor(negatives, dtype=torch.float32, device=prototypes.device), dim=-1
    )
    token_ids = torch.as_tensor(
        prototype_token_ids, dtype=torch.long, device=prototypes.device
    )
    if prototypes.ndim != 2 or token_ids.shape != (prototypes.shape[0],):
        raise ValueError("prototypes and their token ids must align")
    if num_tokens <= 0 or temperature <= 0:
        raise ValueError("num_tokens and temperature must be positive")
    positive_similarity = queries @ prototypes.T
    negative_similarity = (prototypes @ negatives.T).max(-1).values
    negative_similarity = torch.maximum(
        negative_similarity, torch.full_like(negative_similarity, null_similarity)
    )
    prototype_probability = torch.sigmoid(
        (positive_similarity - negative_similarity[None]) / temperature
    )
    token_probability = prototypes.new_zeros((queries.shape[0], num_tokens))
    for token_id in range(num_tokens):
        selected = token_ids == token_id
        if bool(selected.any()):
            token_probability[:, token_id] = prototype_probability[:, selected].max(-1).values
    return token_probability, 1.0 - token_probability


def load_labels(
    label_root: Path, scene_label: str
) -> tuple[dict[int, list[dict[str, Any]]], list[str], int, int]:
    scene_dir = label_root / scene_label
    annotations: dict[int, list[dict[str, Any]]] = {}
    categories: set[str] = set()
    source_height = source_width = 0
    for path in sorted(scene_dir.glob("frame_*.json")):
        payload = json.loads(path.read_text())
        info = payload.get("info", {})
        frame_height = int(info.get("height", 0))
        frame_width = int(info.get("width", 0))
        if frame_height <= 0 or frame_width <= 0:
            raise ValueError(f"annotation dimensions must be positive: {path}")
        if source_height and (frame_height, frame_width) != (source_height, source_width):
            raise ValueError("mixed annotation dimensions require per-frame scaling")
        source_height, source_width = frame_height, frame_width
        objects = []
        for item in payload.get("objects", []):
            category = str(item.get("category", "")).strip()
            segmentation = item.get("segmentation", [])
            if isinstance(segmentation, dict):
                segmentation = segmentation.get("polygons", [])
            if segmentation:
                if isinstance(segmentation[0], (int, float)):
                    segmentation = [segmentation]
                elif all(
                    isinstance(point, (list, tuple)) and len(point) == 2
                    and all(isinstance(coordinate, (int, float)) for coordinate in point)
                    for point in segmentation
                ):
                    segmentation = [segmentation]
            polygons = []
            for value in segmentation:
                array = np.asarray(value, dtype=np.float32).reshape(-1, 2)
                if not np.isfinite(array).all():
                    raise ValueError(f"annotation polygon contains nonfinite coordinates: {path}")
                if array.shape[0] >= 3:
                    polygons.append(array)
            if category and polygons:
                categories.add(category)
                objects.append({"category": category, "polygons": polygons})
        annotations[frame_index(path.stem)] = objects
    if not annotations:
        raise FileNotFoundError(f"no LERF annotations under {scene_dir}")
    if not categories:
        raise ValueError(f"LERF annotations under {scene_dir} contain no valid polygons")
    return annotations, sorted(categories), source_height, source_width


def ground_truth_masks(
    objects: list[dict[str, Any]],
    categories: list[str],
    height: int,
    width: int,
    source_height: int,
    source_width: int,
) -> dict[str, torch.Tensor]:
    scale = np.asarray([width / source_width, height / source_height], dtype=np.float32)
    output = {}
    for category in categories:
        mask = np.zeros((height, width), dtype=np.uint8)
        polygons = [
            np.round(polygon * scale).astype(np.int32)
            for item in objects
            if item["category"] == category
            for polygon in item["polygons"]
        ]
        if polygons:
            cv2.fillPoly(mask, polygons, 1)
        output[category] = torch.from_numpy(mask.astype(bool))
    return output


def camera(view: Any, frame_id: int, height: int, width: int) -> Camera:
    intrinsic = view.intrinsic.clone()
    intrinsic[0] *= width / view.width
    intrinsic[1] *= height / view.height
    return Camera(str(frame_id), intrinsic, view.camera_to_world, height, width)


def binary_iou(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[float, int, int]:
    prediction, target = prediction.bool(), target.bool()
    intersection = int((prediction & target).sum())
    union = int((prediction | target).sum())
    return (float(intersection / union) if union else float("nan"), intersection, union)


__all__ = [
    "GENERIC_NEGATIVES",
    "accumulate_surface_features",
    "binary_iou",
    "camera",
    "frame_index",
    "ground_truth_masks",
    "load_dense_feature",
    "load_labels",
    "load_text_cache",
    "prototype_max_token_posterior",
    "retain_top_query_tokens",
    "validate_semantic_manifest",
    "validate_source_authority",
]
