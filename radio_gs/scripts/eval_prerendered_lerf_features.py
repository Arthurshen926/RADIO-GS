#!/usr/bin/env python3
"""Evaluate pre-rendered LERF feature maps with a fixed OpenCLIP readout."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.evaluation.openclip_readout import NEGATIVE_PROMPTS, OpenCLIPTextScorer
from radio_gs.scripts.eval_opengaussian_lerf_baseline import (
    SCENE_GT_FRAMES,
    _coerce_polygons,
    _rasterize_polygons,
)


PROTOCOL_PROFILES: dict[str, dict[str, object]] = {
    "langsplatv2_released": {
        "mask_thresh": 0.4,
        "activation_kernel": 29,
        "smooth_kernel": 7,
        "feature_mode": "normalized",
        "filter_implementation": "torch_avg_pool",
        "mask_smoothing_implementation": "strict_majority_avg_pool",
        "resize_policy": "error_on_mismatch",
        "source": "LangSplatV2 released eval_lerf.py",
    },
    "occam_langsplat_paper": {
        "mask_thresh": 0.5,
        "activation_kernel": 30,
        "smooth_kernel": 7,
        "feature_mode": "raw",
        "filter_implementation": "opencv_filter2d",
        "mask_smoothing_implementation": "langsplat_legacy",
        "resize_policy": "error_on_mismatch",
        "source": "OccamLGS paper/README plus released LangSplat evaluate_iou_loc.py",
    },
}


@dataclass(frozen=True)
class LerfObject:
    frame: str
    query: str
    mask: np.ndarray
    bboxes: list[tuple[float, float, float, float]]


def _bbox_tuple(raw_bbox: object) -> tuple[float, float, float, float] | None:
    try:
        values = [float(value) for value in raw_bbox]  # type: ignore[arg-type]
    except TypeError:
        return None
    if len(values) != 4:
        return None
    return (values[0], values[1], values[2], values[3])


def load_lerf_objects(label_root: Path, *, frames: Sequence[str] | None = None) -> dict[str, list[LerfObject]]:
    if frames is None:
        frames = sorted(path.stem for path in label_root.glob("frame_*.json"))

    output: dict[str, list[LerfObject]] = {}
    for frame in frames:
        json_path = label_root / f"{frame}.json"
        if not json_path.exists():
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        info = payload.get("info", {})
        height = int(info["height"])
        width = int(info["width"])
        masks_by_query: dict[str, np.ndarray] = {}
        bboxes_by_query: dict[str, list[tuple[float, float, float, float]]] = {}
        for obj in payload.get("objects", []):
            query = str(obj.get("category", "")).strip()
            polygons = _coerce_polygons(obj.get("segmentation"))
            if not query or not polygons:
                continue
            mask = _rasterize_polygons(polygons, height, width)
            if query in masks_by_query:
                masks_by_query[query] = np.logical_or(masks_by_query[query], mask)
            else:
                masks_by_query[query] = mask
            bbox = _bbox_tuple(obj.get("bbox", []))
            if bbox is not None:
                bboxes_by_query.setdefault(query, []).append(bbox)
        output[frame] = [
            LerfObject(
                frame=frame,
                query=query,
                mask=masks_by_query[query].astype(bool),
                bboxes=bboxes_by_query.get(query, []),
            )
            for query in sorted(masks_by_query)
        ]
    return output


def _avg_pool(map_2d: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return map_2d.float()
    padding = kernel_size // 2
    return F.avg_pool2d(
        map_2d.float().unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    ).squeeze(0).squeeze(0)


def _opencv_filter2d(map_2d: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return map_2d.float()
    import cv2

    source = map_2d.detach().float().cpu().numpy()
    # Match released LangSplat exactly: np.ones defaults to float64, while
    # cv2.filter2D preserves the float32 source depth because ddepth=-1.
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    filtered = cv2.filter2D(source, ddepth=-1, kernel=kernel)
    return torch.as_tensor(filtered, dtype=torch.float32, device=map_2d.device)


def _box_filter(
    map_2d: torch.Tensor,
    kernel_size: int,
    implementation: str,
) -> torch.Tensor:
    if implementation == "torch_avg_pool":
        return _avg_pool(map_2d, kernel_size)
    if implementation == "opencv_filter2d":
        return _opencv_filter2d(map_2d, kernel_size)
    raise ValueError(f"unknown filter implementation: {implementation}")


def _smooth_mask_langsplat_legacy(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Vectorized equivalent of LangSplat eval/utils.py::smooth, including edges."""

    if kernel_size <= 1:
        return mask.to(torch.bool)
    if kernel_size % 2 == 0:
        raise ValueError("LangSplat legacy smoothing requires an odd kernel")
    height, width = mask.shape
    radius = kernel_size // 2

    # The released loop uses min(i + radius + 1, height - 1) as an
    # exclusive bound, so the final source row/column is never included.
    source = mask.to(torch.int64)[: max(height - 1, 0), : max(width - 1, 0)]
    integral = F.pad(source.cumsum(0).cumsum(1), (1, 0, 1, 0))
    rows = torch.arange(height, device=mask.device)
    cols = torch.arange(width, device=mask.device)
    row0 = torch.clamp(rows - radius, min=0)
    row1 = torch.clamp(rows + radius + 1, max=max(height - 1, 0))
    col0 = torch.clamp(cols - radius, min=0)
    col1 = torch.clamp(cols + radius + 1, max=max(width - 1, 0))
    ones = (
        integral[row1[:, None], col1[None, :]]
        - integral[row0[:, None], col1[None, :]]
        - integral[row1[:, None], col0[None, :]]
        + integral[row0[:, None], col0[None, :]]
    )
    area = (row1 - row0)[:, None] * (col1 - col0)[None, :]
    # np.argmax(np.bincount(...)) chooses zero on an exact tie.
    return (2 * ones > area).to(torch.bool)


def _smooth_mask(
    mask: torch.Tensor,
    kernel_size: int,
    implementation: str = "strict_majority_avg_pool",
) -> torch.Tensor:
    if kernel_size <= 1:
        return mask.to(torch.bool)
    if implementation == "strict_majority_avg_pool":
        filtered = _avg_pool(mask.float(), kernel_size)
        return (filtered > 0.5).to(torch.bool)
    if implementation == "langsplat_legacy":
        return _smooth_mask_langsplat_legacy(mask, kernel_size)
    raise ValueError(f"unknown mask smoothing implementation: {implementation}")


def _normalize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    output = heatmap.float() - heatmap.float().min()
    output = output / (output.max() + 1e-9)
    return torch.clip(output * 2.0 - 1.0, 0, 1)


def _mask_iou(mask_gt: torch.Tensor, mask_pred: torch.Tensor) -> float:
    intersection = torch.logical_and(mask_gt, mask_pred).sum()
    union = torch.logical_or(mask_gt, mask_pred).sum()
    if int(union.item()) == 0:
        return 0.0
    return float((intersection.float() / union.float()).item())


def _resize_relevance_to_mask(
    relevance: torch.Tensor,
    mask_shape: tuple[int, int],
    resize_policy: str,
) -> torch.Tensor:
    if tuple(relevance.shape[-2:]) == mask_shape:
        return relevance.float()
    if resize_policy == "error_on_mismatch":
        raise ValueError(
            f"relevance shape {tuple(relevance.shape[-2:])} != annotation mask shape {mask_shape}"
        )
    if resize_policy != "bilinear_compat":
        raise ValueError(f"unknown resize policy: {resize_policy}")
    levels, prompts, height, width = relevance.shape
    resized = F.interpolate(
        relevance.float().reshape(levels * prompts, 1, height, width),
        size=mask_shape,
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(levels, prompts, mask_shape[0], mask_shape[1])


def _localization_hit(coords: torch.Tensor, bboxes: Sequence[tuple[float, float, float, float]]) -> bool:
    for x1, y1, x2, y2 in bboxes:
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        for coord in coords:
            if coord[1] >= x_min and coord[1] <= x_max and coord[0] >= y_min and coord[0] <= y_max:
                return True
    return False


def evaluate_relevance_maps(
    frames: Mapping[str, list[LerfObject]],
    relevance_maps: Mapping[str, np.ndarray],
    *,
    mask_thresh: float,
    activation_kernel: int = 29,
    smooth_kernel: int = 7,
    filter_implementation: str = "torch_avg_pool",
    mask_smoothing_implementation: str = "strict_majority_avg_pool",
    resize_policy: str = "error_on_mismatch",
) -> dict[str, object]:
    frame_results: dict[str, object] = {}
    all_ious: list[float] = []
    total_hits = 0
    total_objects = 0

    for frame, objects in frames.items():
        if frame not in relevance_maps:
            raise FileNotFoundError(frame)
        relevance = torch.as_tensor(relevance_maps[frame])
        if relevance.ndim != 4:
            raise ValueError(f"{frame}: expected [levels, prompts, H, W], got {tuple(relevance.shape)}")
        if relevance.shape[1] != len(objects):
            raise ValueError(f"{frame}: prompt count {relevance.shape[1]} != object count {len(objects)}")
        if objects:
            relevance = _resize_relevance_to_mask(
                relevance,
                objects[0].mask.shape,
                resize_policy,
            )

        object_results: list[dict[str, object]] = []
        for object_idx, obj in enumerate(objects):
            iou_by_level: list[float] = []
            score_by_level: list[float] = []
            mask_gt = torch.as_tensor(obj.mask.astype(bool))
            for level_idx in range(relevance.shape[0]):
                raw_relevance = relevance[level_idx, object_idx].float()
                filtered = _box_filter(raw_relevance, activation_kernel, filter_implementation)
                activated = 0.5 * (filtered + raw_relevance)
                output = _normalize_heatmap(activated)
                mask_pred = _smooth_mask(
                    output > mask_thresh,
                    smooth_kernel,
                    mask_smoothing_implementation,
                )
                iou_by_level.append(_mask_iou(mask_gt, mask_pred))
                # LangSplat selects the feature level from the activated raw
                # relevance peak. Selecting from the per-level min-max output
                # degenerates because every non-constant level has max == 1.
                score_by_level.append(float(activated.max().item()))
            chosen_level = int(np.argmax(score_by_level))

            loc_score_by_level: list[float] = []
            loc_coords_by_level: list[torch.Tensor] = []
            for level_idx in range(relevance.shape[0]):
                filtered = _box_filter(
                    relevance[level_idx, object_idx],
                    activation_kernel,
                    filter_implementation,
                )
                loc_score_by_level.append(float(filtered.max().item()))
                loc_coords_by_level.append(torch.nonzero(filtered == filtered.max()))
            loc_level = int(np.argmax(loc_score_by_level))
            hit = _localization_hit(loc_coords_by_level[loc_level], obj.bboxes)

            iou = iou_by_level[chosen_level]
            all_ious.append(iou)
            total_hits += int(hit)
            total_objects += 1
            object_results.append(
                {
                    "query": obj.query,
                    "iou": iou,
                    "level_scores": score_by_level,
                    "chosen_level": chosen_level,
                    "loc_level": loc_level,
                    "loc_hit": hit,
                }
            )

        frame_results[frame] = {
            "objects": object_results,
            "mean_iou": float(np.mean([item["iou"] for item in object_results])) if object_results else 0.0,
            "loc_acc": float(sum(int(item["loc_hit"]) for item in object_results) / len(object_results)) if object_results else 0.0,
        }

    query_micro = {
        "miou": float(np.mean(all_ious)) if all_ious else 0.0,
        "loc_acc": float(total_hits / total_objects) if total_objects else 0.0,
        "objects": total_objects,
        "aggregation": "query_weighted_micro",
    }
    return {
        "frames": frame_results,
        # Backward-compatible alias. Historical files called this "macro",
        # although it is a query-weighted micro average within one scene.
        "macro": dict(query_micro),
        "query_micro": dict(query_micro),
        # One evaluator invocation covers one scene. Across multiple scene
        # reports, use aggregate_scene_results() for the paper's scene-equal
        # overall metric.
        "scene_macro": {
            **query_micro,
            "aggregation": "scene_equal_macro",
            "scenes": 1,
        },
    }


def aggregate_scene_results(scene_results: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Aggregate per-scene reports with explicit paper and micro semantics."""

    rows: dict[str, dict[str, object]] = {}
    for scene, result in scene_results.items():
        metrics = result.get("query_micro", result.get("macro"))
        if not isinstance(metrics, Mapping):
            raise ValueError(f"{scene}: missing query_micro/macro metrics")
        rows[scene] = {
            "miou": float(metrics["miou"]),
            "loc_acc": float(metrics["loc_acc"]),
            "objects": int(metrics["objects"]),
        }

    if not rows:
        raise ValueError("no scene results")
    object_total = sum(int(row["objects"]) for row in rows.values())
    scene_macro = {
        "miou": float(np.mean([float(row["miou"]) for row in rows.values()])),
        "loc_acc": float(np.mean([float(row["loc_acc"]) for row in rows.values()])),
        "scenes": len(rows),
        "aggregation": "scene_equal_macro",
    }
    query_micro = {
        "miou": (
            sum(float(row["miou"]) * int(row["objects"]) for row in rows.values()) / object_total
            if object_total
            else 0.0
        ),
        "loc_acc": (
            sum(float(row["loc_acc"]) * int(row["objects"]) for row in rows.values()) / object_total
            if object_total
            else 0.0
        ),
        "objects": object_total,
        "aggregation": "query_weighted_micro",
    }
    return {
        "scenes": rows,
        "scene_macro": scene_macro,
        "query_micro": query_micro,
        "macro": dict(query_micro),
    }


def _load_feature_tensor(
    frame: str,
    feature_dirs: Sequence[Path],
    *,
    device: torch.device,
    dtype: torch.dtype,
    normalize: bool,
) -> torch.Tensor:
    levels: list[torch.Tensor] = []
    for feature_dir in feature_dirs:
        arr = np.load(feature_dir / f"{frame}.npy", mmap_mode="r")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable.*")
            levels.append(torch.as_tensor(arr, dtype=dtype).to(device, non_blocking=True))
    sem_feat = torch.stack(levels, dim=0)
    if normalize:
        sem_feat = sem_feat / (sem_feat.float().norm(dim=-1, keepdim=True).to(sem_feat.dtype) + 1e-6)
    return sem_feat


def evaluate_feature_dirs(
    label_root: Path,
    feature_dirs: Sequence[Path],
    *,
    frames: Sequence[str],
    mask_thresh: float,
    activation_kernel: int,
    smooth_kernel: int,
    filter_implementation: str,
    mask_smoothing_implementation: str,
    resize_policy: str,
    device: torch.device,
    dtype: torch.dtype,
    normalize: bool,
    scorer_factory: Callable[[torch.device], OpenCLIPTextScorer],
) -> dict[str, object]:
    objects_by_frame = load_lerf_objects(label_root, frames=frames)
    scorer = scorer_factory(device)
    relevance_maps: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for frame, objects in objects_by_frame.items():
            sem_feat = _load_feature_tensor(frame, feature_dirs, device=device, dtype=dtype, normalize=normalize)
            relevance = scorer.relevance(sem_feat, [obj.query for obj in objects])
            relevance_maps[frame] = relevance.detach().cpu().float().numpy()
            del sem_feat, relevance
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return evaluate_relevance_maps(
        objects_by_frame,
        relevance_maps,
        mask_thresh=mask_thresh,
        activation_kernel=activation_kernel,
        smooth_kernel=smooth_kernel,
        filter_implementation=filter_implementation,
        mask_smoothing_implementation=mask_smoothing_implementation,
        resize_policy=resize_policy,
    )


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    return torch.float32


def resolve_protocol_config(args: argparse.Namespace) -> dict[str, Any]:
    profile = dict(PROTOCOL_PROFILES[args.protocol_profile])
    for key in ("mask_thresh", "activation_kernel", "smooth_kernel", "feature_mode"):
        value = getattr(args, key)
        if value is not None:
            profile[key] = value
    profile.update(
        {
            "name": args.protocol_profile,
            "relevance_readout": (
                "minimum positive-vs-each-negative two-way softmax probability; "
                "temperature=10; negatives=object,things,stuff,texture"
            ),
            "activation_transform": (
                "0.5 * raw relevance + 0.5 * box-filtered relevance, then per-level "
                "min-max, rescale [-1,1], clip [0,1]"
            ),
            "level_selection": "argmax activated raw-relevance peak",
            "localization_selection": "argmax box-filtered raw-relevance peak",
            "mask_threshold_comparison": "strict_greater_than",
            "mask_smoothing": (
                f"{int(profile['smooth_kernel'])}x{int(profile['smooth_kernel'])} strict majority"
            ),
            "feature_resize": (
                "forbidden; rendered relevance and annotation mask shapes must match"
                if profile["resize_policy"] == "error_on_mismatch"
                else "bilinear align_corners=False compatibility resize"
            ),
        }
    )
    return profile


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--feature-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--scene", choices=sorted(SCENE_GT_FRAMES), default=None)
    parser.add_argument("--frames", nargs="+", default=None)
    parser.add_argument(
        "--protocol-profile",
        choices=sorted(PROTOCOL_PROFILES),
        default="langsplatv2_released",
    )
    parser.add_argument("--mask-thresh", type=float, default=None)
    parser.add_argument("--activation-kernel", type=int, default=None)
    parser.add_argument("--smooth-kernel", type=int, default=None)
    parser.add_argument("--feature-mode", choices=("normalized", "raw"), default=None)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--openclip-model", default="ViT-B-16")
    parser.add_argument("--openclip-pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    protocol_config = resolve_protocol_config(args)
    if args.frames is not None:
        frames = args.frames
    elif args.scene is not None:
        frames = SCENE_GT_FRAMES[args.scene]
    else:
        frames = sorted(path.stem for path in args.label_root.glob("frame_*.json"))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    normalize = protocol_config["feature_mode"] == "normalized"

    def scorer_factory(target_device: torch.device) -> OpenCLIPTextScorer:
        return OpenCLIPTextScorer(target_device, model_name=args.openclip_model, pretrained=args.openclip_pretrained)

    result = evaluate_feature_dirs(
        args.label_root,
        args.feature_dirs,
        frames=frames,
        mask_thresh=float(protocol_config["mask_thresh"]),
        activation_kernel=int(protocol_config["activation_kernel"]),
        smooth_kernel=int(protocol_config["smooth_kernel"]),
        filter_implementation=str(protocol_config["filter_implementation"]),
        mask_smoothing_implementation=str(protocol_config["mask_smoothing_implementation"]),
        resize_policy=str(protocol_config["resize_policy"]),
        device=device,
        dtype=_dtype_from_name(args.dtype),
        normalize=normalize,
        scorer_factory=scorer_factory,
    )
    result["protocol"] = "pre-rendered LERF feature OpenCLIP fixed-threshold readout"
    result["protocol_config"] = protocol_config
    result["label_root"] = str(args.label_root)
    result["feature_dirs"] = [str(path) for path in args.feature_dirs]
    result["scene"] = args.scene
    result["feature_mode"] = protocol_config["feature_mode"]
    result["dtype"] = args.dtype
    result["mask_thresh"] = protocol_config["mask_thresh"]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    query_micro = result["query_micro"]
    print(
        f"LocAcc={query_micro['loc_acc']:.4f} "
        f"mIoU={query_micro['miou']:.4f} "
        f"objects={query_micro['objects']} "
        f"aggregation={query_micro['aggregation']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
