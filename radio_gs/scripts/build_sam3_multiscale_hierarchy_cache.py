#!/usr/bin/env python3
"""Build query-free official-SAM3 multiscale source-region hierarchies.

The input is an explicit source-RGB authority.  This command has no query,
annotation, target-view, or evaluation-view interface.  Every frame cache is
bound to the source bytes, official checkpoint, numerical runtime, producer
source closure, and complete generation contract.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import PIL
from PIL import Image
import torch

import radio_gs.models.sam3_multiscale_hierarchy as hierarchy
from radio_gs.models.sam3_multiscale_hierarchy import (
    CropSpec,
    binary_mask_box,
    build_crop_pyramid,
    canonical_json_sha256,
    containment_aware_deduplicate,
    crop_edge_flags,
    dense_point_grid,
    direct_containment_graph,
    pack_masks,
    remap_crop_mask,
    require_sha256,
    validate_multiscale_cache_payload,
    validate_source_authority_payload,
)
import radio_gs.scripts.build_sam3_foundation_cache as foundation
from radio_gs.scripts.build_sam3_foundation_cache import (
    _load_sam3_model,
    set_requested_cuda_device,
    sha256_file,
)


GENERATION_SOURCE = "official_sam3_interactive_multiscale_crop_pyramid_v1"


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def runtime_binding(device: str) -> dict[str, object]:
    requested = torch.device(str(device))
    cuda: dict[str, object] = {
        "requested_device": str(requested),
        "torch_cuda_version": str(torch.version.cuda or ""),
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = requested.index if requested.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        cuda.update(
            {
                "selected_device_index": int(index),
                "device_name": str(properties.name),
                "compute_capability": [int(properties.major), int(properties.minor)],
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    payload: dict[str, object] = {
        "contract": "sam3-multiscale-runtime-binding-v1",
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "torch_git_version": str(getattr(torch.version, "git_version", "") or ""),
        "numpy_version": str(np.__version__),
        "pillow_version": str(PIL.__version__),
        "sam3_package_version": _package_version("sam3"),
        "cuda": cuda,
        "numerical_flags": {
            "default_dtype": str(torch.get_default_dtype()),
            "cudnn_enabled": bool(torch.backends.cudnn.enabled),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        },
    }
    payload["digest"] = canonical_json_sha256(payload)
    return payload


def producer_binding(repository_root: str | Path) -> dict[str, object]:
    root = Path(repository_root).expanduser().resolve()
    sources = (Path(__file__).resolve(), Path(hierarchy.__file__).resolve(), Path(foundation.__file__).resolve())
    files: list[dict[str, str]] = []
    for source in sources:
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"producer source is not one regular file: {source}")
        try:
            relative = source.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"producer source escaped repository root: {source}") from error
        files.append({"relative_path": relative, "sha256": sha256_file(source)})
    files.sort(key=lambda item: item["relative_path"])
    payload: dict[str, object] = {
        "contract": "sam3-multiscale-producer-source-closure-v1",
        "repository_root": str(root),
        "files": files,
    }
    payload["digest"] = canonical_json_sha256(payload)
    return payload


def mask_stability(
    low_resolution_logits: np.ndarray | None,
    masks: np.ndarray,
    *,
    offset: float,
) -> tuple[np.ndarray, bool]:
    candidates = np.asarray(masks, dtype=bool)
    logits = None if low_resolution_logits is None else _as_numpy(low_resolution_logits)
    if (
        logits is None
        or logits.ndim != 3
        or logits.shape[0] != candidates.shape[0]
        or not np.issubdtype(logits.dtype, np.number)
    ):
        return np.ones(candidates.shape[0], dtype=np.float32), False
    high = logits > float(offset)
    low = logits > -float(offset)
    intersection = np.logical_and(high, low).sum(axis=(1, 2), dtype=np.int64)
    union = np.logical_or(high, low).sum(axis=(1, 2), dtype=np.int64)
    return (intersection / np.maximum(union, 1)).astype(np.float32), True


def generation_contract(args: argparse.Namespace, *, checkpoint_sha256: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "source": GENERATION_SOURCE,
        "official_decoder": True,
        "query_free": True,
        "information_inputs": ["registered_source_or_mapping_rgb"],
        "forbidden_inputs": ["query_text", "benchmark_ground_truth", "target_or_evaluation_rgb"],
        "checkpoint_sha256": require_sha256(checkpoint_sha256, label="checkpoint sha256"),
        "resolution": int(args.resolution),
        "dtype": str(args.dtype),
        "crop_pyramid": {
            "layers_after_full_image": int(args.crop_layers),
            "crops_per_axis_at_layer": "2**layer",
            "overlap_ratio": float(args.crop_overlap_ratio),
            "box_convention": "full_image_xyxy_half_open_edge_anchored",
        },
        "point_grid": {
            "points_per_side_at_full_image": int(args.points_per_side),
            "downscale_factor_per_layer": int(args.point_grid_downscale_factor),
            "placement": "deterministic_crop_pixel_cell_centres_row_major",
        },
        "filtering": {
            "minimum_quality": float(args.minimum_quality),
            "minimum_stability": float(args.minimum_stability),
            "stability_offset": float(args.stability_offset),
            "minimum_crop_area_fraction": float(args.minimum_crop_area_fraction),
            "minimum_full_image_area_fraction": float(args.minimum_full_image_area_fraction),
            "maximum_full_image_area_fraction": float(args.maximum_full_image_area_fraction),
            "artificial_crop_edge_policy": "reject",
            "crop_edge_tolerance_pixels": int(args.crop_edge_tolerance_pixels),
        },
        "deduplication": {
            "type": "per_crop_then_global_quality_stability_ranked_near_identical_only",
            "iou_threshold": float(args.dedup_iou),
            "near_equal_area_ratio": float(args.dedup_near_equal_area_ratio),
            "maximum_masks": int(args.maximum_masks),
            "proper_containment_preserved": True,
        },
        "hierarchy": {
            "type": "direct_smallest-containing-parent_forest",
            "containment_threshold": float(args.containment_threshold),
            "minimum_parent_area_ratio": float(args.minimum_parent_area_ratio),
        },
        "mask_tensor_semantics": "binary_probability_thresholded_by_official_decoder",
        "full_image_remapping": "exact_integer_crop_embedding_without_resize",
    }
    payload["digest"] = canonical_json_sha256(payload)
    return payload


def _validate_arguments(args: argparse.Namespace) -> None:
    if int(args.resolution) != 1008:
        raise ValueError("official SAM3 hierarchy generation requires resolution=1008")
    if int(args.crop_layers) < 0 or int(args.points_per_side) <= 0:
        raise ValueError("crop layers/point grid are invalid")
    if int(args.point_grid_downscale_factor) <= 0:
        raise ValueError("point grid downscale factor must be positive")
    for name in ("minimum_quality", "minimum_stability", "minimum_crop_area_fraction", "minimum_full_image_area_fraction", "maximum_full_image_area_fraction", "dedup_iou", "dedup_near_equal_area_ratio", "containment_threshold"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0,1]")
    if float(args.minimum_parent_area_ratio) <= 1.0:
        raise ValueError("minimum_parent_area_ratio must exceed one")
    if float(args.minimum_full_image_area_fraction) > float(args.maximum_full_image_area_fraction):
        raise ValueError("full-image area limits are reversed")
    if int(args.crop_edge_tolerance_pixels) < 0 or int(args.maximum_masks) < 0:
        raise ValueError("edge tolerance/maximum masks must be non-negative")


@torch.inference_mode()
def automatic_multiscale_hierarchy(
    processor: Any,
    image: Image.Image,
    args: argparse.Namespace,
) -> dict[str, object]:
    width, height = image.size
    crops = build_crop_pyramid(
        image_width=width,
        image_height=height,
        crop_layers=int(args.crop_layers),
        overlap_ratio=float(args.crop_overlap_ratio),
        points_per_side=int(args.points_per_side),
        point_grid_downscale_factor=int(args.point_grid_downscale_factor),
    )
    records: list[dict[str, object]] = []
    prompt_index = 0
    decoder_logits_available_for_all_prompts = True
    rejected_artificial_edge = 0
    candidates_before_filtering = 0
    accepted_before_per_crop_deduplication = 0
    for crop in crops:
        crop_records: list[dict[str, object]] = []
        crop_image = image.crop(crop.box_xyxy)
        state = processor.set_image(crop_image)
        local_points, full_points = dense_point_grid(crop)
        for seed_local, seed_full in zip(local_points, full_points):
            candidates, quality, low_resolution = processor.model.predict_inst(
                state,
                point_coords=np.asarray([seed_local], dtype=np.float32),
                point_labels=np.ones(1, dtype=np.int32),
                multimask_output=True,
            )
            candidates = _as_numpy(candidates)
            if candidates.ndim == 4 and candidates.shape[1] == 1:
                candidates = candidates[:, 0]
            candidates = np.asarray(candidates, dtype=bool)
            quality = _as_numpy(quality).astype(np.float32, copy=False).reshape(-1)
            if (
                candidates.ndim != 3
                or candidates.shape[0] != quality.size
                or tuple(candidates.shape[1:]) != (crop.height, crop.width)
            ):
                raise ValueError("official SAM3 crop masks/quality/raster do not align")
            stability, has_logits = mask_stability(
                low_resolution, candidates, offset=float(args.stability_offset)
            )
            decoder_logits_available_for_all_prompts &= has_logits
            candidates_before_filtering += int(candidates.shape[0])
            for candidate_index, (local_mask, score, stable) in enumerate(
                zip(candidates, quality, stability)
            ):
                if not local_mask.any():
                    continue
                crop_area_fraction = float(local_mask.mean())
                full_area_fraction = float(local_mask.sum(dtype=np.int64) / (height * width))
                if score < float(args.minimum_quality) or stable < float(args.minimum_stability):
                    continue
                if crop_area_fraction < float(args.minimum_crop_area_fraction):
                    continue
                if not float(args.minimum_full_image_area_fraction) <= full_area_fraction <= float(args.maximum_full_image_area_fraction):
                    continue
                local_box = binary_mask_box(local_mask)
                touches, artificial = crop_edge_flags(
                    local_box,
                    crop=crop,
                    image_width=width,
                    image_height=height,
                    tolerance_pixels=int(args.crop_edge_tolerance_pixels),
                )
                if any(artificial):
                    rejected_artificial_edge += 1
                    continue
                crop_records.append(
                    {
                        "mask": local_mask.copy(),
                        "quality": float(score),
                        "stability": float(stable),
                        "seed_xy_full": tuple(float(v) for v in seed_full),
                        "seed_xy_crop": tuple(float(v) for v in seed_local),
                        "prompt_index": int(prompt_index),
                        "candidate_index": int(candidate_index),
                        "crop_index": int(crop.index),
                        "crop_layer": int(crop.layer),
                        "crop_grid_side": int(crop.grid_side),
                        "crop_box": tuple(int(v) for v in crop.box_xyxy),
                        "crop_scale_xy": (float(crop.width / width), float(crop.height / height)),
                        "crop_window_area_fraction": float(crop.width * crop.height / (width * height)),
                        "proposal_area_fraction": full_area_fraction,
                        "crop_area_fraction": crop_area_fraction,
                        "touches_crop_edge": touches,
                        "touches_artificial_crop_edge": artificial,
                    }
                )
            prompt_index += 1

        accepted_before_per_crop_deduplication += len(crop_records)
        crop_ranking = [
            float(record["quality"]) * float(record["stability"])
            for record in crop_records
        ]
        crop_kept = containment_aware_deduplicate(
            [np.asarray(record["mask"]) for record in crop_records],
            crop_ranking,
            iou_threshold=float(args.dedup_iou),
            near_equal_area_ratio=float(args.dedup_near_equal_area_ratio),
            maximum_masks=0,
        )
        for index in crop_kept:
            record = dict(crop_records[index])
            full_mask = remap_crop_mask(
                np.asarray(record["mask"]),
                crop_box_xyxy=crop.box_xyxy,
                full_height=height,
                full_width=width,
            )
            record["mask"] = full_mask
            record["box"] = binary_mask_box(full_mask)
            records.append(record)

    ranking = [float(r["quality"]) * float(r["stability"]) for r in records]
    kept = containment_aware_deduplicate(
        [np.asarray(record["mask"]) for record in records],
        ranking,
        iou_threshold=float(args.dedup_iou),
        near_equal_area_ratio=float(args.dedup_near_equal_area_ratio),
        maximum_masks=int(args.maximum_masks),
    )
    selected = [records[index] for index in kept]
    masks = (
        np.stack([np.asarray(record["mask"], dtype=bool) for record in selected])
        if selected
        else np.empty((0, height, width), dtype=bool)
    )
    graph = direct_containment_graph(
        [mask for mask in masks],
        [float(record["quality"]) for record in selected],
        containment_threshold=float(args.containment_threshold),
        minimum_parent_area_ratio=float(args.minimum_parent_area_ratio),
    )

    def tensor(key: str, *, dtype: torch.dtype, width_: int | None = None) -> torch.Tensor:
        raw = [record[key] for record in selected]
        result = torch.tensor(raw, dtype=dtype)
        if width_ is not None:
            result = result.reshape(-1, width_)
        return result

    return {
        "schema_version": 1,
        "packed_masks": pack_masks(masks),
        "mask_shape": [int(height), int(width)],
        "quality": tensor("quality", dtype=torch.float32),
        "stability": tensor("stability", dtype=torch.float32),
        "seed_xy_full": tensor("seed_xy_full", dtype=torch.float32, width_=2),
        "seed_xy_crop": tensor("seed_xy_crop", dtype=torch.float32, width_=2),
        "prompt_index": tensor("prompt_index", dtype=torch.int64),
        "candidate_index": tensor("candidate_index", dtype=torch.int16),
        "crop_index": tensor("crop_index", dtype=torch.int32),
        "crop_layer": tensor("crop_layer", dtype=torch.int16),
        "crop_grid_side": tensor("crop_grid_side", dtype=torch.int16),
        "crop_boxes_xyxy": tensor("crop_box", dtype=torch.int32, width_=4),
        "crop_scale_xy": tensor("crop_scale_xy", dtype=torch.float32, width_=2),
        "crop_window_area_fraction": tensor("crop_window_area_fraction", dtype=torch.float32),
        "boxes_xyxy": tensor("box", dtype=torch.int32, width_=4),
        "proposal_area_fraction": tensor("proposal_area_fraction", dtype=torch.float32),
        "crop_area_fraction": tensor("crop_area_fraction", dtype=torch.float32),
        "touches_crop_edge": tensor("touches_crop_edge", dtype=torch.bool, width_=4),
        "touches_artificial_crop_edge": tensor("touches_artificial_crop_edge", dtype=torch.bool, width_=4),
        **graph,
        "statistics": {
            "crop_count": len(crops),
            "prompt_count": int(prompt_index),
            "candidate_count_before_filtering": int(candidates_before_filtering),
            "accepted_count_before_per_crop_deduplication": int(accepted_before_per_crop_deduplication),
            "proposal_count_before_global_deduplication": len(records),
            "proposal_count_after_deduplication": len(selected),
            "rejected_artificial_crop_edge": int(rejected_artificial_edge),
            "decoder_logits_available_for_all_prompts": bool(decoder_logits_available_for_all_prompts),
        },
    }


def _load_source_authority(path: Path, expected_sha256: str) -> tuple[dict, tuple[dict, ...]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"source authority is not one regular file: {path}")
    expected = require_sha256(expected_sha256, label="source authority sha256")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError("source authority SHA-256 differs")
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("source authority must be one JSON object")
    records = validate_source_authority_payload(payload)
    return dict(payload), records


def _selected_records(records: tuple[dict, ...], raw_ids: str, maximum: int) -> list[dict]:
    requested = {part for part in str(raw_ids).replace(",", " ").split() if part}
    by_id = {record["image_id"]: record for record in records}
    if requested:
        missing = requested - set(by_id)
        if missing:
            raise ValueError(f"source image ids are absent from authority: {sorted(missing)}")
        selected = [by_id[image_id] for image_id in sorted(requested)]
    else:
        selected = list(records)
    if int(maximum) > 0:
        selected = selected[: int(maximum)]
    if not selected:
        raise ValueError("no source images selected")
    return selected


def _atomic_torch_save(payload: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_write(payload: object, output: Path) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    handle = tempfile.NamedTemporaryFile(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False)
    temporary = Path(handle.name)
    try:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    try:
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> dict[str, object]:
    _validate_arguments(args)
    repository_root = Path(args.repository_root).expanduser().resolve()
    authority_path = Path(args.source_authority).expanduser().resolve()
    _, authority_records = _load_source_authority(authority_path, args.source_authority_sha256)
    selected = _selected_records(authority_records, args.image_ids, int(args.maximum_images))
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise ValueError(f"official SAM3 checkpoint is not one regular file: {checkpoint_path}")
    checkpoint_sha = require_sha256(args.checkpoint_sha256, label="checkpoint sha256")
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise ValueError("official SAM3 checkpoint SHA-256 differs")
    set_requested_cuda_device(args.device)
    runtime = runtime_binding(args.device)
    producer = producer_binding(repository_root)
    contract = generation_contract(args, checkpoint_sha256=checkpoint_sha)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    processor = _load_sam3_model(
        checkpoint_path=str(checkpoint_path),
        device=args.device,
        confidence_threshold=0.0,
        dtype=args.dtype,
        resolution=int(args.resolution),
        point_only=True,
    )
    reports: list[dict[str, object]] = []
    for record in selected:
        image_id = str(record["image_id"])
        raw_path = Path(str(record["path"])).expanduser()
        image_path = (authority_path.parent / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
        if not image_path.is_file():
            raise ValueError(f"source image is absent: {image_path}")
        source_bytes = image_path.read_bytes()
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        if source_sha != record["sha256"]:
            raise ValueError(f"source image SHA-256 differs: {image_id}")
        metadata: dict[str, object] = {
            "generation_contract": contract,
            "source_authority": {"path": str(authority_path), "sha256": str(args.source_authority_sha256)},
            "source_image": {"image_id": image_id, "path": str(image_path), "sha256": source_sha, "rgb_role": "registered_source_or_mapping_view"},
            "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
            "runtime_binding": runtime,
            "producer_binding": producer,
        }
        output = output_root / f"{image_id}.pt"
        if output.exists():
            if not args.skip_existing:
                raise FileExistsError(f"multiscale SAM3 cache exists: {output}")
            existing = torch.load(output, map_location="cpu", weights_only=False)
            count = validate_multiscale_cache_payload(existing, expected_metadata=metadata)
            reports.append({"image_id": image_id, "output": str(output), "output_sha256": sha256_file(output), "proposal_count": count, "reused_after_validation": True})
            continue
        image = Image.open(io.BytesIO(source_bytes)).convert("RGB")
        payload = automatic_multiscale_hierarchy(processor, image, args)
        payload["metadata"] = metadata
        validate_multiscale_cache_payload(payload, expected_metadata=metadata)
        _atomic_torch_save(payload, output)
        reports.append({"image_id": image_id, "output": str(output), "output_sha256": sha256_file(output), "proposal_count": int(payload["quality"].numel()), "reused_after_validation": False})

    manifest: dict[str, object] = {
        "schema_version": 1,
        "contract": "official-sam3-query-free-multiscale-hierarchy-manifest-v1",
        "source_authority": {"path": str(authority_path), "sha256": str(args.source_authority_sha256)},
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "generation_contract": contract,
        "runtime_binding": runtime,
        "producer_binding": producer,
        "selected_image_ids": [record["image_id"] for record in selected],
        "images": reports,
    }
    manifest["content_digest"] = canonical_json_sha256(manifest)
    manifest_path = output_root / str(args.manifest_name)
    if manifest_path.name != str(args.manifest_name) or manifest_path.suffix != ".json":
        raise ValueError("manifest name must be one JSON basename")
    if manifest_path.exists():
        if not args.skip_existing:
            raise FileExistsError(f"multiscale SAM3 manifest exists: {manifest_path}")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise ValueError("existing multiscale SAM3 manifest identity differs")
    else:
        _atomic_json_write(manifest, manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-authority", required=True)
    parser.add_argument("--source-authority-sha256", required=True)
    parser.add_argument("--image-ids", default="", help="Optional authority image-id subset; never arbitrary image paths.")
    parser.add_argument("--maximum-images", type=int, default=0)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--repository-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--crop-layers", type=int, default=2)
    parser.add_argument("--crop-overlap-ratio", type=float, default=0.25)
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--point-grid-downscale-factor", type=int, default=2)
    parser.add_argument("--minimum-quality", type=float, default=0.70)
    parser.add_argument("--minimum-stability", type=float, default=0.0)
    parser.add_argument("--stability-offset", type=float, default=1.0)
    parser.add_argument("--minimum-crop-area-fraction", type=float, default=0.001)
    parser.add_argument("--minimum-full-image-area-fraction", type=float, default=0.0001)
    parser.add_argument("--maximum-full-image-area-fraction", type=float, default=0.90)
    parser.add_argument("--crop-edge-tolerance-pixels", type=int, default=2)
    parser.add_argument("--dedup-iou", type=float, default=0.85)
    parser.add_argument("--dedup-near-equal-area-ratio", type=float, default=0.90)
    parser.add_argument("--maximum-masks", type=int, default=0)
    parser.add_argument("--containment-threshold", type=float, default=0.90)
    parser.add_argument("--minimum-parent-area-ratio", type=float, default=1.05)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
