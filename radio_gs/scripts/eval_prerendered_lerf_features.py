#!/usr/bin/env python3
"""Evaluate pre-rendered LERF feature maps with a fixed OpenCLIP readout."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.scripts.eval_opengaussian_lerf_baseline import (
    SCENE_GT_FRAMES,
    _coerce_polygons,
    _rasterize_polygons,
)


NEGATIVE_PROMPTS = ("object", "things", "stuff", "texture")


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


def _smooth_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return mask.to(torch.bool)
    filtered = _avg_pool(mask.float(), kernel_size)
    return (filtered > 0.5).to(torch.bool)


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

        object_results: list[dict[str, object]] = []
        for object_idx, obj in enumerate(objects):
            iou_by_level: list[float] = []
            score_by_level: list[float] = []
            mask_gt = torch.as_tensor(obj.mask.astype(bool))
            for level_idx in range(relevance.shape[0]):
                filtered = _avg_pool(relevance[level_idx, object_idx], activation_kernel)
                output = _normalize_heatmap(0.5 * (filtered + relevance[level_idx, object_idx].float()))
                mask_pred = _smooth_mask(output > mask_thresh, smooth_kernel)
                iou_by_level.append(_mask_iou(mask_gt, mask_pred))
                score_by_level.append(float(output.max().item()))
            chosen_level = int(np.argmax(score_by_level))

            loc_score_by_level: list[float] = []
            loc_coords_by_level: list[torch.Tensor] = []
            for level_idx in range(relevance.shape[0]):
                filtered = _avg_pool(relevance[level_idx, object_idx], activation_kernel)
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

    return {
        "frames": frame_results,
        "macro": {
            "miou": float(np.mean(all_ious)) if all_ious else 0.0,
            "loc_acc": float(total_hits / total_objects) if total_objects else 0.0,
            "objects": total_objects,
        },
    }


class OpenCLIPTextScorer:
    def __init__(self, device: torch.device, *, model_name: str, pretrained: str):
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, precision="fp16")
        model.eval()
        self.device = device
        self.model = model.to(device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        with torch.inference_mode():
            tokens = torch.cat([self.tokenizer(prompt) for prompt in NEGATIVE_PROMPTS]).to(device)
            self.neg_embeds = self.model.encode_text(tokens)
            self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

    @torch.inference_mode()
    def _positive_embeds(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = torch.cat([self.tokenizer(prompt) for prompt in prompts]).to(self.device)
        embeds = self.model.encode_text(tokens)
        return embeds / embeds.norm(dim=-1, keepdim=True)

    @torch.inference_mode()
    def relevance(self, sem_map: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        pos_embeds = self._positive_embeds(prompts)
        phrase_embeds = torch.cat([pos_embeds, self.neg_embeds], dim=0).to(sem_map.dtype).to(self.device)
        n_levels, height, width, channels = sem_map.shape
        n_prompts = len(prompts)
        n_negatives = len(NEGATIVE_PROMPTS)
        sem_flat = sem_map.permute(0, 3, 1, 2).reshape(n_levels, channels, -1).permute(0, 2, 1).contiguous()
        sim = torch.einsum("nqc,pc->nqp", sem_flat, phrase_embeds)
        pos_vals = sim[:, :, :n_prompts]
        neg_vals = sim[:, :, n_prompts:]
        repeated_pos = pos_vals.unsqueeze(-1).repeat(1, 1, 1, n_negatives)
        repeated_neg = neg_vals.unsqueeze(2).repeat(1, 1, n_prompts, 1)
        sims = torch.stack([repeated_pos, repeated_neg], dim=-1)
        softmax = torch.softmax(10 * sims, dim=-1)
        min_pos_prob, _ = softmax[..., 0].min(dim=-1)
        return min_pos_prob.permute(0, 2, 1).reshape(n_levels, n_prompts, height, width)


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
    )


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    return torch.float32


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--feature-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--scene", choices=sorted(SCENE_GT_FRAMES), default=None)
    parser.add_argument("--frames", nargs="+", default=None)
    parser.add_argument("--mask-thresh", type=float, default=0.4)
    parser.add_argument("--activation-kernel", type=int, default=29)
    parser.add_argument("--smooth-kernel", type=int, default=7)
    parser.add_argument("--feature-mode", choices=("normalized", "raw"), default="normalized")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--openclip-model", default="ViT-B-16")
    parser.add_argument("--openclip-pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.frames is not None:
        frames = args.frames
    elif args.scene is not None:
        frames = SCENE_GT_FRAMES[args.scene]
    else:
        frames = sorted(path.stem for path in args.label_root.glob("frame_*.json"))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    normalize = args.feature_mode == "normalized"

    def scorer_factory(target_device: torch.device) -> OpenCLIPTextScorer:
        return OpenCLIPTextScorer(target_device, model_name=args.openclip_model, pretrained=args.openclip_pretrained)

    result = evaluate_feature_dirs(
        args.label_root,
        args.feature_dirs,
        frames=frames,
        mask_thresh=args.mask_thresh,
        activation_kernel=args.activation_kernel,
        smooth_kernel=args.smooth_kernel,
        device=device,
        dtype=_dtype_from_name(args.dtype),
        normalize=normalize,
        scorer_factory=scorer_factory,
    )
    result["protocol"] = "pre-rendered LERF feature OpenCLIP fixed-threshold readout"
    result["label_root"] = str(args.label_root)
    result["feature_dirs"] = [str(path) for path in args.feature_dirs]
    result["feature_mode"] = args.feature_mode
    result["dtype"] = args.dtype
    result["mask_thresh"] = args.mask_thresh
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    macro = result["macro"]
    print(f"LocAcc={macro['loc_acc']:.4f} mIoU={macro['miou']:.4f} objects={macro['objects']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
