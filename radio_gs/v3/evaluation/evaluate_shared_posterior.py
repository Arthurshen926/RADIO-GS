"""Evaluate one SUGM-v3 Gaussian posterior in rendered and 3D-thresholded form."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.query.interface import load_query_interface
from radio_gs.v3.query.packet import QueryPacket
from radio_gs.v3.training.instance_upper_bound import sha256_file


def _polygons(value: object) -> list[np.ndarray]:
    if not isinstance(value, list) or not value:
        return []
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2 and array.shape[1] == 2:
        return [array]
    output = []
    for item in value:
        candidate = np.asarray(item, dtype=np.float32)
        if candidate.ndim == 2 and candidate.shape[1] == 2:
            output.append(candidate)
        elif candidate.ndim == 1 and candidate.size >= 6 and candidate.size % 2 == 0:
            output.append(candidate.reshape(-1, 2))
    return output


def _labels(root: Path, height: int, width: int) -> tuple[dict[int, dict[str, np.ndarray]], set[str]]:
    frames: dict[int, dict[str, np.ndarray]] = {}
    categories: set[str] = set()
    for path in sorted(root.glob("frame_*.json")):
        payload = json.loads(path.read_text())
        source_h = int(payload["info"]["height"])
        source_w = int(payload["info"]["width"])
        frame = int(path.stem.split("_")[-1])
        masks: dict[str, np.ndarray] = {}
        for item in payload.get("objects", []):
            category = str(item.get("category", "")).strip()
            if not category:
                continue
            mask = masks.setdefault(category, np.zeros((height, width), dtype=np.uint8))
            scaled = []
            for polygon in _polygons(item.get("segmentation")):
                points = polygon.copy()
                points[:, 0] *= width / source_w
                points[:, 1] *= height / source_h
                scaled.append(np.rint(points).astype(np.int32))
            if scaled:
                cv2.fillPoly(mask, scaled, color=1)
                categories.add(category)
        frames[frame] = masks
    if not frames or not categories:
        raise ValueError("evaluation label cohort is empty")
    return frames, categories


def _iou(prediction: np.ndarray, target: np.ndarray) -> float:
    union = np.logical_or(prediction, target).sum()
    return float(np.logical_and(prediction, target).sum() / max(1, union))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--responsibility", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--text-embeddings", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--threshold-ratio", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    state_path = Path(args.scene_state).resolve(strict=True)
    responsibility_path = Path(args.responsibility).resolve(strict=True)
    labels_path = Path(args.label_dir).resolve(strict=True)
    text_path = Path(args.text_embeddings).resolve(strict=True)
    authority = json.loads(responsibility_path.read_text())
    interface = load_query_interface(state_path, device=args.device)
    if int(authority["num_gaussians"]) != interface.model.memory.shape[0]:
        raise ValueError("evaluation responsibility row domain differs")
    membership = torch.load(
        Path(torch.load(state_path, map_location="cpu")["metadata"]["inputs"]["membership"]["path"]),
        map_location="cpu",
    )
    height = int(membership["metadata"]["feature_height"])
    width = int(membership["metadata"]["feature_width"])
    if height * width != int(authority["num_pixels"]):
        raise ValueError("evaluation responsibility raster differs")
    annotations, categories = _labels(labels_path, height, width)
    text_payload = torch.load(text_path, map_location="cpu")
    names = [str(item) for item in text_payload["queries"]]
    lookup = {name.casefold(): index for index, name in enumerate(names)}
    embeddings = torch.as_tensor(text_payload["embeddings"]).float()
    missing = sorted(category for category in categories if category.casefold() not in lookup)
    if missing:
        raise ValueError(f"frozen text cache lacks categories: {missing}")
    posteriors = {}
    for category in sorted(categories):
        token = embeddings[lookup[category.casefold()]]
        posterior, _ = interface.posterior_from_packet(
            QueryPacket("text", token=token), scale=args.scale,
            topk=args.topk, temperature=args.temperature,
        )
        posteriors[category] = posterior
    view_by_frame = {int(item["frame_index"]): item for item in authority["views"]}
    records = []
    per_category: dict[str, dict[str, list[float] | int]] = {}
    for frame, masks in sorted(annotations.items()):
        if frame not in view_by_frame:
            raise ValueError(f"evaluation responsibility lacks labeled frame {frame}")
        item = view_by_frame[frame]
        shard_path = responsibility_path.parent / item["relative_path"]
        if sha256_file(shard_path) != item["sha256"]:
            raise ValueError("evaluation responsibility shard hash differs")
        shard = torch.load(shard_path, map_location="cpu")
        gaussian = torch.as_tensor(shard["gaussian_ids"]).long().to(args.device)
        pixel = torch.as_tensor(shard["pixel_ids"]).long().to(args.device)
        weight = torch.as_tensor(shard["base_weights"]).float().to(args.device)
        for category, target in sorted(masks.items()):
            posterior = posteriors[category]
            rendered = interface.render_posterior(
                posterior, gaussian, pixel, weight, num_pixels=height * width
            ).reshape(height, width)
            binary3d = (posterior >= args.threshold_ratio * posterior.max()).float()
            rendered3d = interface.render_posterior(
                binary3d, gaussian, pixel, weight, num_pixels=height * width
            ).reshape(height, width)
            pred2d = (rendered >= args.threshold_ratio * rendered.max()).cpu().numpy()
            pred3d = (rendered3d >= args.threshold_ratio * rendered3d.max()).cpu().numpy()
            flat_peak = int(rendered.argmax())
            peak_hit = bool(target.reshape(-1)[flat_peak])
            iou2d, iou3d = _iou(pred2d, target), _iou(pred3d, target)
            records.append({
                "frame": frame, "category": category, "iou_2d": iou2d,
                "iou_3d": iou3d, "localization_hit": peak_hit,
            })
            bucket = per_category.setdefault(category, {"iou_2d": [], "iou_3d": [], "hits": 0})
            bucket["iou_2d"].append(iou2d)  # type: ignore[union-attr]
            bucket["iou_3d"].append(iou3d)  # type: ignore[union-attr]
            bucket["hits"] = int(bucket["hits"]) + int(peak_hit)
    result_categories = {
        key: {
            "miou_2d": float(np.mean(value["iou_2d"])),
            "miou_3d": float(np.mean(value["iou_3d"])),
            "localization_accuracy": int(value["hits"]) / len(value["iou_2d"]),
            "samples": len(value["iou_2d"]),
        }
        for key, value in per_category.items()
    }
    payload = {
        "schema": "radio_gs.sugm_v3.shared_posterior_evaluation.v1",
        "scene": torch.load(state_path, map_location="cpu")["scene"],
        "metrics": {
            "sample_micro_miou_2d": float(np.mean([item["iou_2d"] for item in records])),
            "sample_micro_miou_3d": float(np.mean([item["iou_3d"] for item in records])),
            "category_macro_miou_2d": float(np.mean([v["miou_2d"] for v in result_categories.values()])),
            "category_macro_miou_3d": float(np.mean([v["miou_3d"] for v in result_categories.values()])),
            "localization_accuracy": sum(item["localization_hit"] for item in records) / len(records),
            "samples": len(records),
        },
        "per_category": result_categories,
        "per_sample": records,
        "method": {
            "same_gaussian_posterior_for_2d_and_3d": True,
            "target_rgb_opened": False,
            "topk": args.topk, "scale": args.scale,
            "temperature": args.temperature, "threshold_ratio": args.threshold_ratio,
            "two_dimensional_conversion": "render_continuous_posterior_then_threshold",
            "three_dimensional_conversion": "threshold_same_posterior_then_render",
        },
        "inputs": {
            "scene_state": {"path": str(state_path), "sha256": sha256_file(state_path)},
            "responsibility": {"path": str(responsibility_path), "sha256": sha256_file(responsibility_path)},
            "labels": str(labels_path),
            "text_embeddings": {"path": str(text_path), "sha256": sha256_file(text_path)},
        },
    }
    write_frozen_json(Path(args.output).resolve(), payload)
    print(payload["metrics"])


if __name__ == "__main__":
    main()
