"""Evaluate DINO/SAM RADIO adaptor spaces on LERF-OVS downstream probes.

This script compares original RGB-extracted RADIO features ("teacher") with
RADIO-GS rendered features after projection through frozen RADIO adaptors such
as dino_v3 and sam3.  It runs two label-only probes:

1. Prototype segmentation: use same-category masks from other frames as support
   prototypes and segment the category in the target frame.
2. Source-target matching: use the first annotated frame for a category as the
   source prototype and retrieve the same category in later frames.

The metrics intentionally reuse the LERF-OVS grounding definitions:
localization accuracy (argmax inside polygon) and thresholded mIoU.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, ".")

from radio_gs.data.lerf_dataset import LERFDataset
from radio_gs.models.radio_adaptors import (
    load_radio_adaptor_from_checkpoint,
    project_feature_map_with_adaptor,
)
from radio_gs.scripts.eval_lerf_grounding import (
    DEFAULT_GT_FEATURE_ROOT,
    DEFAULT_LABEL_DIR,
    LERF_OVS_SCENES,
    build_gt_masks,
    compute_iou,
    load_lerf_ovs_labels,
    load_lerf_rgb_frame,
    load_render_pipeline,
    localization_accuracy,
    render_1280d,
    resolve_lerf_label_dir,
    resolve_lerf_scene_root,
)

logger = logging.getLogger(__name__)

DEFAULT_RADIO_ADAPTOR_CHECKPOINT = "/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower()).strip("_")
    return slug or "item"


def _as_chw(feature_map: torch.Tensor) -> torch.Tensor:
    if feature_map.ndim == 4:
        if feature_map.shape[0] != 1:
            raise ValueError(f"Expected a single feature map, got {tuple(feature_map.shape)}")
        feature_map = feature_map.squeeze(0)
    if feature_map.ndim != 3:
        raise ValueError(f"Expected feature map [C,H,W], got {tuple(feature_map.shape)}")
    return feature_map.float()


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.shape == (height, width):
        return mask_u8
    return cv2.resize(mask_u8, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def build_masked_prototype(feature_map: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    """Average L2-normalised tokens inside a binary mask and normalise again."""
    feature_map = _as_chw(feature_map)
    _, height, width = feature_map.shape
    mask_feat = _resize_mask(mask, height, width).astype(bool)
    if not mask_feat.any():
        raise ValueError("Cannot build a prototype from an empty mask")

    tokens = feature_map[:, mask_feat].transpose(0, 1)  # [N, C]
    tokens = F.normalize(tokens.float(), dim=1)
    prototype = tokens.mean(dim=0)
    return F.normalize(prototype, dim=0)


def compute_prototype_heatmap(feature_map: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
    """Cosine-similarity heatmap for a masked feature prototype."""
    feature_map = _as_chw(feature_map)
    prototype = F.normalize(prototype.float(), dim=0)
    feature_map = F.normalize(feature_map.float(), dim=0)
    return (feature_map * prototype[:, None, None]).sum(dim=0)


def select_source_target_pairs(
    frames_by_category: Mapping[str, Iterable[int]],
) -> List[Tuple[str, int, int]]:
    """Use the first annotated frame as source and all later frames as targets."""
    pairs: List[Tuple[str, int, int]] = []
    for category in sorted(frames_by_category):
        frames = sorted(set(int(frame_id) for frame_id in frames_by_category[category]))
        if len(frames) < 2:
            continue
        source = frames[0]
        for target in frames[1:]:
            pairs.append((category, source, target))
    return pairs


def _empty_accumulator() -> Dict:
    return {
        "loc_correct": 0,
        "loc_total": 0,
        "iou_sum": 0.0,
        "n_iou_samples": 0,
        "per_category": defaultdict(
            lambda: {"loc_correct": 0, "loc_total": 0, "iou_sum": 0.0, "n_iou_samples": 0}
        ),
    }


def _update_accumulator(acc: Dict, category: str, correct: bool, iou: float) -> None:
    acc["loc_correct"] += int(correct)
    acc["loc_total"] += 1
    acc["iou_sum"] += float(iou)
    acc["n_iou_samples"] += 1
    cat_acc = acc["per_category"][category]
    cat_acc["loc_correct"] += int(correct)
    cat_acc["loc_total"] += 1
    cat_acc["iou_sum"] += float(iou)
    cat_acc["n_iou_samples"] += 1


def _merge_accumulator(dst: Dict, src: Dict) -> None:
    dst["loc_correct"] += int(src["loc_correct"])
    dst["loc_total"] += int(src["loc_total"])
    dst["iou_sum"] += float(src["iou_sum"])
    dst["n_iou_samples"] += int(src["n_iou_samples"])
    for category, cat_src in src["per_category"].items():
        cat_dst = dst["per_category"][category]
        cat_dst["loc_correct"] += int(cat_src["loc_correct"])
        cat_dst["loc_total"] += int(cat_src["loc_total"])
        cat_dst["iou_sum"] += float(cat_src["iou_sum"])
        cat_dst["n_iou_samples"] += int(cat_src["n_iou_samples"])


def _finalize_accumulator(acc: Dict) -> Dict:
    per_category = {}
    for category in sorted(acc["per_category"]):
        cat = acc["per_category"][category]
        per_category[category] = {
            "loc_acc": cat["loc_correct"] / max(cat["loc_total"], 1),
            "miou": cat["iou_sum"] / max(cat["n_iou_samples"], 1),
            "n_samples": int(cat["loc_total"]),
        }
    return {
        "loc_acc": acc["loc_correct"] / max(acc["loc_total"], 1),
        "miou": acc["iou_sum"] / max(acc["n_iou_samples"], 1),
        "loc_correct": int(acc["loc_correct"]),
        "loc_total": int(acc["loc_total"]),
        "n_iou_samples": int(acc["n_iou_samples"]),
        "per_category": per_category,
    }


def _resolve_gt_feature_dir(gt_feature_root: Optional[str], scene: str) -> Path:
    root = Path(gt_feature_root or DEFAULT_GT_FEATURE_ROOT)
    if root.name == scene or (root / "backbone").exists() or list(root.glob("rgb_*.pt")):
        return root
    return root / scene


def _load_teacher_feature(gt_feature_dir: Path, frame_id: int, device: torch.device) -> torch.Tensor:
    candidates = [
        gt_feature_dir / f"rgb_{frame_id}.pt",
        gt_feature_dir / "backbone" / f"rgb_{frame_id}.pt",
    ]
    for path in candidates:
        if path.exists():
            feature = torch.load(path, map_location=device).float()
            if feature.ndim == 3:
                feature = feature.unsqueeze(0)
            if feature.ndim != 4:
                raise ValueError(f"Expected feature tensor [1,C,H,W] in {path}, got {tuple(feature.shape)}")
            return feature
    raise FileNotFoundError(f"Missing RADIO reference feature for frame {frame_id} in {gt_feature_dir}")


def _load_rgb_tensor(scene: str, frame_id: int, scene_root_hint: str | Path, device: torch.device) -> Optional[torch.Tensor]:
    image = load_lerf_rgb_frame(scene, frame_id, scene_root_hint)
    if image is None:
        return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor.to(device)


def _render_feature(
    scene: str,
    frame_id: int,
    render_pipeline: tuple,
    lerf_dataset: LERFDataset,
    device: torch.device,
) -> torch.Tensor:
    model, codec, renderer, sharpener, refiner, config, is_hybrid = render_pipeline
    pose_w2c = lerf_dataset.pose_by_frame_idx.get(frame_id)
    if pose_w2c is None:
        raise KeyError(f"No pose for {scene} frame {frame_id}")
    viewmat = torch.from_numpy(pose_w2c.copy()).float().to(device).unsqueeze(0)
    rgb_tensor = None
    if getattr(config, "refiner_rgb_guide", False):
        rgb_tensor = _load_rgb_tensor(scene, frame_id, getattr(config, "scene_root", ""), device)
    return render_1280d(
        model,
        codec,
        renderer,
        sharpener,
        refiner,
        viewmat,
        is_hybrid=is_hybrid,
        config=config,
        device=device,
        rgb_image=rgb_tensor,
    )


def _load_projected_features(
    scene: str,
    frame_ids: List[int],
    adaptors: Mapping[str, torch.nn.Module],
    device: torch.device,
    *,
    gt_feature_dir: Optional[Path],
    render_pipeline: Optional[tuple],
    lerf_dataset: Optional[LERFDataset],
) -> Dict[str, Dict[str, Dict[int, torch.Tensor]]]:
    features: Dict[str, Dict[str, Dict[int, torch.Tensor]]] = {}
    modes = []
    if gt_feature_dir is not None:
        modes.append("teacher")
    if render_pipeline is not None and lerf_dataset is not None:
        modes.append("rendered")

    for mode in modes:
        features[mode] = {name: {} for name in adaptors}
        for frame_id in tqdm(frame_ids, desc=f"  {scene}/{mode}/adaptor features", leave=False):
            try:
                if mode == "teacher":
                    feature_1280 = _load_teacher_feature(gt_feature_dir, frame_id, device)
                else:
                    feature_1280 = _render_feature(scene, frame_id, render_pipeline, lerf_dataset, device)
            except Exception as exc:
                logger.warning("Skipping %s frame %d: %s", mode, frame_id, exc)
                continue

            with torch.no_grad():
                for name, adaptor in adaptors.items():
                    projected = project_feature_map_with_adaptor(feature_1280, adaptor)
                    features[mode][name][frame_id] = projected.squeeze(0).detach().cpu().float()
            del feature_1280
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return features


def _build_frame_masks(
    frame_annotations: Mapping[int, List[dict]],
    img_h: int,
    img_w: int,
    feature_height: int,
    feature_width: int,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, Dict[str, np.ndarray]], Dict[str, List[int]]]:
    full_masks: Dict[int, Dict[str, np.ndarray]] = {}
    feat_masks: Dict[int, Dict[str, np.ndarray]] = {}
    frames_by_category: Dict[str, List[int]] = defaultdict(list)

    for frame_id, objects in frame_annotations.items():
        frame_categories = sorted({obj["category"] for obj in objects})
        if not frame_categories:
            full_masks[frame_id] = {}
            feat_masks[frame_id] = {}
            continue
        full = build_gt_masks(objects, frame_categories, img_h, img_w)
        feat = build_gt_masks(
            objects,
            frame_categories,
            feature_height,
            feature_width,
            src_height=img_h,
            src_width=img_w,
        )
        full_masks[frame_id] = full
        feat_masks[frame_id] = feat
        for category in frame_categories:
            if full[category].sum() > 0 and feat[category].sum() > 0:
                frames_by_category[category].append(frame_id)
    return full_masks, feat_masks, frames_by_category


def _mean_prototypes(prototypes: List[torch.Tensor]) -> Optional[torch.Tensor]:
    if not prototypes:
        return None
    stacked = torch.stack(prototypes, dim=0)
    return F.normalize(stacked.mean(dim=0), dim=0)


def _overlay_heatmap(rgb_bgr: np.ndarray, heatmap: torch.Tensor) -> np.ndarray:
    height, width = rgb_bgr.shape[:2]
    hm_np = heatmap.detach().cpu().numpy().astype(np.float32)
    hmin = float(np.min(hm_np))
    hmax = float(np.max(hm_np))
    if hmax - hmin > 1e-6:
        hm_u8 = ((hm_np - hmin) / (hmax - hmin) * 255.0).astype(np.uint8)
    else:
        hm_u8 = np.zeros_like(hm_np, dtype=np.uint8)
    hm_u8 = cv2.resize(hm_u8, (width, height), interpolation=cv2.INTER_LINEAR)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(rgb_bgr, 0.55, hm_color, 0.45, 0.0)

    flat_idx = int(np.argmax(hm_np.reshape(-1)))
    fy, fx = divmod(flat_idx, hm_np.shape[1])
    px = int(round((fx + 0.5) * width / hm_np.shape[1]))
    py = int(round((fy + 0.5) * height / hm_np.shape[0]))
    cv2.circle(overlay, (min(px, width - 1), min(py, height - 1)), 8, (255, 255, 255), 2)
    cv2.circle(overlay, (min(px, width - 1), min(py, height - 1)), 4, (0, 0, 0), -1)
    return overlay


def _overlay_mask(rgb_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    height, width = rgb_bgr.shape[:2]
    mask_u8 = _resize_mask(mask, height, width)
    color = np.zeros_like(rgb_bgr)
    color[:, :, 1] = mask_u8 * 255
    return cv2.addWeighted(rgb_bgr, 0.65, color, 0.35, 0.0)


def _add_header(parts: List[np.ndarray], labels: List[str], title: str) -> np.ndarray:
    panel_h = min(part.shape[0] for part in parts)
    resized = []
    for part in parts:
        if part.shape[0] != panel_h:
            new_w = int(round(part.shape[1] * panel_h / part.shape[0]))
            part = cv2.resize(part, (new_w, panel_h), interpolation=cv2.INTER_AREA)
        resized.append(part)
    grid = np.concatenate(resized, axis=1)
    header = np.zeros((56, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, title[:110], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    x = 0
    for label, part in zip(labels, resized):
        cv2.putText(header, label, (x + 8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        x += part.shape[1]
    return np.concatenate([header, grid], axis=0)


def _save_visual_samples(
    visual_samples: Mapping[Tuple, Dict],
    out_dir: Path,
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for key, sample in visual_samples.items():
        scene, task, adaptor, category, source_frame, target_frame = key
        rgb = sample.get("rgb")
        if rgb is None:
            mask = sample["mask_full"]
            rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        mask_overlay = _overlay_mask(rgb, sample["mask_full"])
        parts = [rgb, mask_overlay]
        labels = ["RGB", "GT mask"]
        for mode in ("teacher", "rendered"):
            heatmap = sample["heatmaps"].get(mode)
            if heatmap is None:
                continue
            parts.append(_overlay_heatmap(rgb, heatmap))
            labels.append(f"{mode} {sample['metrics'][mode]['loc']:.0f}/{sample['metrics'][mode]['iou']:.2f}")

        title = (
            f"{scene} | {adaptor} | {task} | {category} | "
            f"src {source_frame if source_frame is not None else 'multi'} -> tgt {target_frame}"
        )
        grid = _add_header(parts, labels, title)
        filename = (
            f"{_slugify(scene)}_{_slugify(adaptor)}_{_slugify(task)}_"
            f"{target_frame:05d}_{_slugify(category)}.png"
        )
        path = out_dir / filename
        cv2.imwrite(str(path), grid)
        saved.append(str(path))
    return saved


def _maybe_record_visual(
    visual_samples: Dict[Tuple, Dict],
    visual_counts: Dict[Tuple[str, str], int],
    *,
    scene: str,
    task: str,
    adaptor: str,
    mode: str,
    category: str,
    source_frame: Optional[int],
    target_frame: int,
    rgb_image: Optional[np.ndarray],
    mask_full: np.ndarray,
    heatmap: torch.Tensor,
    loc_correct: bool,
    iou: float,
    max_visuals: int,
) -> None:
    if max_visuals <= 0:
        return
    count_key = (task, adaptor)
    sample_key = (scene, task, adaptor, category, source_frame, target_frame)
    if sample_key not in visual_samples and visual_counts[count_key] >= max_visuals:
        return
    if sample_key not in visual_samples:
        visual_counts[count_key] += 1
        visual_samples[sample_key] = {
            "rgb": rgb_image,
            "mask_full": mask_full,
            "heatmaps": {},
            "metrics": {},
        }
    visual_samples[sample_key]["heatmaps"][mode] = heatmap.detach().cpu()
    visual_samples[sample_key]["metrics"][mode] = {"loc": bool(loc_correct), "iou": float(iou)}


def _score_heatmap(heatmap: torch.Tensor, mask_full: np.ndarray, mask_feat: np.ndarray, threshold_ratio: float) -> Tuple[bool, float]:
    loc = localization_accuracy(heatmap, mask_full)
    iou = compute_iou(heatmap, mask_feat, threshold_ratio=threshold_ratio)
    return loc, iou


def evaluate_scene_downstream(
    scene: str,
    label_dir: str,
    adaptors: Mapping[str, torch.nn.Module],
    device: torch.device,
    *,
    gt_feature_dir: Optional[Path],
    render_pipeline: Optional[tuple],
    lerf_dataset: Optional[LERFDataset],
    output_dir: Path,
    iou_threshold: float = 0.5,
    max_visuals: int = 12,
) -> Dict:
    frame_annotations, _, img_h, img_w = load_lerf_ovs_labels(label_dir, scene)
    frame_ids = sorted(frame_annotations)
    projected = _load_projected_features(
        scene,
        frame_ids,
        adaptors,
        device,
        gt_feature_dir=gt_feature_dir,
        render_pipeline=render_pipeline,
        lerf_dataset=lerf_dataset,
    )

    feature_shape = None
    for mode_features in projected.values():
        for adaptor_features in mode_features.values():
            if adaptor_features:
                first = next(iter(adaptor_features.values()))
                feature_shape = tuple(first.shape[-2:])
                break
        if feature_shape is not None:
            break
    if feature_shape is None:
        raise RuntimeError(f"No projected adaptor features available for {scene}")

    feat_h, feat_w = int(feature_shape[0]), int(feature_shape[1])
    full_masks, feat_masks, frames_by_category = _build_frame_masks(
        frame_annotations,
        img_h,
        img_w,
        feat_h,
        feat_w,
    )

    scene_root_hint = ""
    if render_pipeline is not None:
        scene_root_hint = getattr(render_pipeline[5], "scene_root", "")
    visual_samples: Dict[Tuple, Dict] = {}
    visual_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    accumulators: Dict[str, Dict[str, Dict[str, Dict]]] = {}
    for adaptor_name in adaptors:
        accumulators[adaptor_name] = {
            "prototype_segmentation": defaultdict(_empty_accumulator),
            "source_target_matching": defaultdict(_empty_accumulator),
        }

    for mode, mode_features in projected.items():
        for adaptor_name, frame_features in mode_features.items():
            if not frame_features:
                continue

            prototypes: Dict[Tuple[str, int], torch.Tensor] = {}
            for category, frames in frames_by_category.items():
                for frame_id in frames:
                    feature_map = frame_features.get(frame_id)
                    if feature_map is None:
                        continue
                    try:
                        prototypes[(category, frame_id)] = build_masked_prototype(
                            feature_map,
                            feat_masks[frame_id][category],
                        )
                    except ValueError:
                        continue

            for category, frames in sorted(frames_by_category.items()):
                for target_frame in sorted(frames):
                    target_feature = frame_features.get(target_frame)
                    if target_feature is None:
                        continue
                    support = [
                        prototypes[(category, source_frame)]
                        for source_frame in frames
                        if source_frame != target_frame and (category, source_frame) in prototypes
                    ]
                    prototype = _mean_prototypes(support)
                    if prototype is None:
                        continue
                    heatmap = compute_prototype_heatmap(target_feature, prototype)
                    loc, iou = _score_heatmap(
                        heatmap,
                        full_masks[target_frame][category],
                        feat_masks[target_frame][category],
                        iou_threshold,
                    )
                    task_acc = accumulators[adaptor_name]["prototype_segmentation"][mode]
                    _update_accumulator(task_acc, category, loc, iou)
                    rgb = load_lerf_rgb_frame(scene, target_frame, scene_root_hint)
                    _maybe_record_visual(
                        visual_samples,
                        visual_counts,
                        scene=scene,
                        task="prototype_segmentation",
                        adaptor=adaptor_name,
                        mode=mode,
                        category=category,
                        source_frame=None,
                        target_frame=target_frame,
                        rgb_image=rgb,
                        mask_full=full_masks[target_frame][category],
                        heatmap=heatmap,
                        loc_correct=loc,
                        iou=iou,
                        max_visuals=max_visuals,
                    )

            for category, source_frame, target_frame in select_source_target_pairs(frames_by_category):
                if (category, source_frame) not in prototypes:
                    continue
                target_feature = frame_features.get(target_frame)
                if target_feature is None:
                    continue
                heatmap = compute_prototype_heatmap(target_feature, prototypes[(category, source_frame)])
                loc, iou = _score_heatmap(
                    heatmap,
                    full_masks[target_frame][category],
                    feat_masks[target_frame][category],
                    iou_threshold,
                )
                task_acc = accumulators[adaptor_name]["source_target_matching"][mode]
                _update_accumulator(task_acc, category, loc, iou)
                rgb = load_lerf_rgb_frame(scene, target_frame, scene_root_hint)
                _maybe_record_visual(
                    visual_samples,
                    visual_counts,
                    scene=scene,
                    task="source_target_matching",
                    adaptor=adaptor_name,
                    mode=mode,
                    category=category,
                    source_frame=source_frame,
                    target_frame=target_frame,
                    rgb_image=rgb,
                    mask_full=full_masks[target_frame][category],
                    heatmap=heatmap,
                    loc_correct=loc,
                    iou=iou,
                    max_visuals=max_visuals,
                )

    saved_visuals = _save_visual_samples(visual_samples, output_dir / "visualizations" / scene)

    finalized = {}
    for adaptor_name, tasks in accumulators.items():
        finalized[adaptor_name] = {}
        for task_name, mode_accs in tasks.items():
            finalized[adaptor_name][task_name] = {
                mode: _finalize_accumulator(acc) for mode, acc in sorted(mode_accs.items())
            }
    return {
        "feature_resolution": [feat_h, feat_w],
        "n_frames": len(frame_ids),
        "n_categories": len(frames_by_category),
        "visualizations": saved_visuals,
        "adaptors": finalized,
    }


def _parse_adaptor_names(raw: str) -> List[str]:
    names: List[str] = []
    for part in re.split(r"[,\s]+", raw.strip()):
        if not part or part in names:
            continue
        names.append(part)
    return names


def _print_summary(report: Mapping[str, object]) -> None:
    print("\n" + "=" * 78)
    print("  LERF-OVS DINO/SAM DOWNSTREAM SUMMARY")
    print("=" * 78)
    macro = report.get("macro", {})
    for adaptor_name in sorted(macro):
        print(f"\n[{adaptor_name}]")
        for task_name in ("prototype_segmentation", "source_target_matching"):
            task = macro[adaptor_name].get(task_name, {})
            if not task:
                continue
            print(f"  {task_name}")
            print(f"  {'Mode':<12} {'Loc Acc':>10} {'mIoU':>10} {'Samples':>10}")
            print(f"  {'-' * 46}")
            for mode in ("teacher", "rendered"):
                metrics = task.get(mode)
                if not metrics:
                    continue
                print(
                    f"  {mode:<12} {metrics['loc_acc']:>10.4f} "
                    f"{metrics['miou']:>10.4f} {metrics['loc_total']:>10d}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Rendered RADIO-GS config YAML")
    parser.add_argument("--checkpoint", default=None, help="Rendered RADIO-GS checkpoint")
    parser.add_argument("--scene", default="all", help="LERF scene or 'all'")
    parser.add_argument("--label_dir", default=DEFAULT_LABEL_DIR)
    parser.add_argument("--gt_feature_dir", default=None, help="Teacher feature root or scene directory")
    parser.add_argument("--output_dir", default="output/lerf_adaptor_downstream")
    parser.add_argument("--adaptor_names", default="dino_v3,sam3")
    parser.add_argument("--adaptor_kind", default="feature_projection", choices=["feature_projection", "feature", "projection", "head", "summary"])
    parser.add_argument("--radio_adaptor_checkpoint", default=DEFAULT_RADIO_ADAPTOR_CHECKPOINT)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--max_visuals", type=int, default=12, help="Max visual samples per task/adaptor/scene")
    parser.add_argument("--gt_only", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    label_dir = resolve_lerf_label_dir(args.label_dir)
    scenes = LERF_OVS_SCENES if args.scene == "all" else (args.scene,)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if not args.gt_only and (not args.config or not args.checkpoint):
        parser.error("Provide --config and --checkpoint for rendered mode, or pass --gt_only")

    adaptor_names = _parse_adaptor_names(args.adaptor_names)
    if not adaptor_names:
        parser.error("--adaptor_names resolved to an empty list")

    adaptors = {}
    for name in adaptor_names:
        adaptor = load_radio_adaptor_from_checkpoint(
            args.radio_adaptor_checkpoint,
            name,
            kind=args.adaptor_kind,
        ).to(device)
        adaptor.eval()
        for param in adaptor.parameters():
            param.requires_grad_(False)
        adaptors[name] = adaptor

    print("=" * 78)
    print("  LERF-OVS DINO/SAM Downstream Probes")
    print("=" * 78)
    print(f"  Scenes:    {', '.join(scenes)}")
    print(f"  Adaptors:  {', '.join(adaptor_names)} ({args.adaptor_kind})")
    print(f"  Label dir: {label_dir}")
    print(f"  Device:    {device}")
    print()

    render_pipeline = None
    lerf_datasets: Dict[str, LERFDataset] = {}
    if not args.gt_only:
        print("Loading rendering pipeline ...")
        render_pipeline = load_render_pipeline(args.config, args.checkpoint, device)
        config = render_pipeline[5]
        for scene in scenes:
            gt_dir = _resolve_gt_feature_dir(args.gt_feature_dir, scene)
            scene_root = resolve_lerf_scene_root(scene, getattr(config, "scene_root", ""))
            try:
                lerf_datasets[scene] = LERFDataset(
                    scene_root=str(scene_root),
                    feature_dir=str(gt_dir),
                    annotation_dir=str(Path(label_dir) / scene),
                    feature_height=getattr(config, "feature_height", 30),
                    feature_width=getattr(config, "feature_width", 40),
                )
            except Exception as exc:
                logger.warning("Could not create LERFDataset for %s: %s", scene, exc)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_reports = {}
    macro_accumulators: Dict[str, Dict[str, Dict[str, Dict]]] = {
        name: {
            "prototype_segmentation": defaultdict(_empty_accumulator),
            "source_target_matching": defaultdict(_empty_accumulator),
        }
        for name in adaptor_names
    }

    for scene in scenes:
        print(f"\n{'-' * 78}")
        print(f"Scene: {scene}")
        print(f"{'-' * 78}")
        gt_feature_dir = _resolve_gt_feature_dir(args.gt_feature_dir, scene)
        if not gt_feature_dir.exists():
            logger.warning("Teacher feature dir missing for %s: %s", scene, gt_feature_dir)
            gt_feature_dir = None

        scene_report = evaluate_scene_downstream(
            scene,
            label_dir,
            adaptors,
            device,
            gt_feature_dir=gt_feature_dir,
            render_pipeline=None if args.gt_only else render_pipeline,
            lerf_dataset=None if args.gt_only else lerf_datasets.get(scene),
            output_dir=output_dir,
            iou_threshold=args.iou_threshold,
            max_visuals=args.max_visuals,
        )
        scene_reports[scene] = scene_report

        for adaptor_name, tasks in scene_report["adaptors"].items():
            for task_name, modes in tasks.items():
                for mode, metrics in modes.items():
                    acc = _empty_accumulator()
                    acc["loc_correct"] = metrics["loc_correct"]
                    acc["loc_total"] = metrics["loc_total"]
                    acc["iou_sum"] = metrics["miou"] * metrics["n_iou_samples"]
                    acc["n_iou_samples"] = metrics["n_iou_samples"]
                    for category, cat_metrics in metrics["per_category"].items():
                        cat_acc = acc["per_category"][category]
                        cat_acc["loc_total"] = cat_metrics["n_samples"]
                        cat_acc["loc_correct"] = round(cat_metrics["loc_acc"] * cat_metrics["n_samples"])
                        cat_acc["n_iou_samples"] = cat_metrics["n_samples"]
                        cat_acc["iou_sum"] = cat_metrics["miou"] * cat_metrics["n_samples"]
                    _merge_accumulator(macro_accumulators[adaptor_name][task_name][mode], acc)

    macro = {}
    for adaptor_name, tasks in macro_accumulators.items():
        macro[adaptor_name] = {}
        for task_name, modes in tasks.items():
            macro[adaptor_name][task_name] = {
                mode: _finalize_accumulator(acc) for mode, acc in sorted(modes.items())
            }

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {key: str(value) for key, value in vars(args).items()},
        "scenes": scene_reports,
        "macro": macro,
    }
    report_path = output_dir / "lerf_adaptor_downstream_results.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    _print_summary(report)
    print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()
