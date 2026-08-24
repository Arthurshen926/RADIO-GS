#!/usr/bin/env python3
"""Track one NVOS registered prompt to its target with official SAM3 memory.

This is an explicitly RGB-assisted development compiler.  The prompt-view
instance is selected by the already sealed signed-evidence SAM inventory.  Its
box and official positive/negative scribbles initialize one SAM3 video object,
which is propagated along the shorter cyclic registered-camera path.  The
target mask and metric remain unopened until the prediction receipt is sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

RECEIPT_TYPE = "nvos_synchronous_multiview_target_prediction_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON authority is not a mapping: {path}")
    return dict(value)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def shorter_cyclic_path(length: int, start: int, target: int) -> list[int]:
    if length < 2 or not 0 <= start < length or not 0 <= target < length:
        raise ValueError("registered cyclic path indices differ")
    forward = (target - start) % length
    reverse = (start - target) % length
    step = 1 if forward <= reverse else -1
    distance = min(forward, reverse)
    return [(start + step * offset) % length for offset in range(distance + 1)]


def sample_signed_points(
    positive: np.ndarray,
    negative: np.ndarray,
    maximum_per_sign: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic farthest signed points in normalized xy order."""

    if positive.shape != negative.shape or positive.ndim != 2:
        raise ValueError("signed prompt maps differ")
    height, width = positive.shape
    points: list[np.ndarray] = []
    labels: list[int] = []
    for values, label in ((positive, 1), (negative, 0)):
        coordinates = np.argwhere(np.asarray(values) > 0.5)
        if not len(coordinates):
            continue
        scores = np.asarray(values)[coordinates[:, 0], coordinates[:, 1]]
        first = int(np.argmax(scores))
        chosen = [first]
        normalized = np.stack(
            (
                coordinates[:, 1] / max(width - 1, 1),
                coordinates[:, 0] / max(height - 1, 1),
            ),
            axis=1,
        ).astype(np.float32)
        minimum_squared = ((normalized - normalized[first]) ** 2).sum(axis=1)
        while len(chosen) < min(int(maximum_per_sign), len(coordinates)):
            index = int(np.argmax(minimum_squared))
            if index in chosen:
                break
            chosen.append(index)
            distance = ((normalized - normalized[index]) ** 2).sum(axis=1)
            minimum_squared = np.minimum(minimum_squared, distance)
        points.extend(normalized[chosen])
        labels.extend([label] * len(chosen))
    if not points or 1 not in labels:
        raise ValueError("registered prompt has no positive signed point")
    return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _scene(manifest: Mapping[str, Any], scene_id: str) -> Mapping[str, Any]:
    values = [row for row in manifest.get("scenes", []) if row.get("scene_id") == scene_id]
    if len(values) != 1:
        raise ValueError("NVOS scene authority differs")
    return values[0]


def predict(args: argparse.Namespace) -> dict[str, Any]:
    from sam3.model_builder import build_sam3_video_model

    plan_path = Path(args.plan).expanduser().resolve(strict=True)
    inventory_path = Path(args.inventory).expanduser().resolve(strict=True)
    manifest_path = Path(args.manifest).expanduser().resolve(strict=True)
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    for path, expected, label in (
        (plan_path, args.expected_plan_sha256, "plan"),
        (inventory_path, args.expected_inventory_sha256, "inventory"),
        (checkpoint, args.expected_checkpoint_sha256, "SAM3 checkpoint"),
    ):
        if sha256_file(path) != str(expected):
            raise ValueError(f"{label} hash differs")
    plan = _load_json(plan_path)
    inventory = _load_json(inventory_path)
    manifest = _load_json(manifest_path)
    scene_id = str(args.scene)
    scene = _scene(manifest, scene_id)
    if (
        plan.get("scene_id") != scene_id
        or inventory.get("scene_id") != scene_id
        or plan.get("target_mask_opened") is not False
        or inventory.get("target_mask_opened") is not False
        or plan.get("view_selection") is not False
        or inventory.get("candidate_count") != 1
    ):
        raise ValueError("registered prompt video authority differs")

    prompt_id = str(scene["prompt"]["frame_id"])
    target_ids = [str(value) for value in scene.get("evaluation_frame_ids", [])]
    if len(target_ids) != 1:
        raise ValueError("NVOS video sentinel requires one target frame")
    target_id = target_ids[0]
    plan_views = [dict(row) for row in plan["candidates"][0]["views"]]
    frame_ids = [str(row["frame_id"]) for row in plan_views]
    if len(frame_ids) != int(plan.get("registered_camera_count", -1)) or len(set(frame_ids)) != len(frame_ids):
        raise ValueError("registered RGB cohort is incomplete")
    prompt_index = frame_ids.index(prompt_id)
    target_index = frame_ids.index(target_id)
    path_indices = shorter_cyclic_path(len(frame_ids), prompt_index, target_index)
    prompt_view = plan_views[prompt_index]
    prompt_digest = str(prompt_view["view_digest"])
    inventory_views = [dict(row) for row in inventory["candidates"][0]["views"]]
    matches = [row for row in inventory_views if str(row.get("view_digest")) == prompt_digest]
    if len(matches) != 1:
        raise ValueError("prompt-view SAM extent is absent")
    probability_record = matches[0]["probability"]
    probability_path = Path(str(probability_record["path"])).resolve(strict=True)
    if sha256_file(probability_path) != str(probability_record["sha256"]):
        raise ValueError("prompt-view SAM probability hash differs")
    prompt_probability = np.load(probability_path, allow_pickle=False)
    seed = np.asarray(prompt_probability) >= float(args.seed_threshold)
    if seed.ndim != 2 or not bool(seed.any()):
        raise ValueError("prompt-view SAM extent is empty")
    ys, xs = np.where(seed)
    height, width = seed.shape
    box = np.asarray(
        [[xs.min() / width, ys.min() / height, (xs.max() + 1) / width, (ys.max() + 1) / height]],
        dtype=np.float32,
    )
    positive_record = prompt_view["positive_authority"]
    negative_record = prompt_view["negative_authority"]
    positive_path = Path(str(positive_record["path"])).resolve(strict=True)
    negative_path = Path(str(negative_record["path"])).resolve(strict=True)
    if (
        sha256_file(positive_path) != str(positive_record["sha256"])
        or sha256_file(negative_path) != str(negative_record["sha256"])
    ):
        raise ValueError("registered signed prompt hash differs")
    points, labels = sample_signed_points(
        np.load(positive_path, allow_pickle=False),
        np.load(negative_path, allow_pickle=False),
        int(args.points_per_sign),
    )

    image_paths: list[Path] = []
    for index in path_indices:
        record = plan_views[index]["rgb"]
        path = Path(str(record["path"])).resolve(strict=True)
        if sha256_file(path) != str(record["sha256"]):
            raise ValueError("registered RGB hash differs")
        image_paths.append(path)
    with Image.open(image_paths[0]) as image:
        image_width, image_height = image.size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("registered prompt RGB shape differs")

    target_mask: np.ndarray | None = None
    target_score: float | None = None
    with tempfile.TemporaryDirectory(prefix=f"radio_gs_nvos_{scene_id}_video_") as temporary:
        frame_root = Path(temporary)
        for index, path in enumerate(image_paths):
            os.symlink(path, frame_root / f"{index:05d}.jpg")
        model = build_sam3_video_model(
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            strict_state_dict_loading=True,
            apply_temporal_disambiguation=True,
            device=args.device,
            compile=False,
        )
        tracker = model.tracker
        tracker.backbone = model.detector.backbone
        state = tracker.init_state(video_path=str(frame_root))
        tracker.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
            box=box,
        )
        for frame_index, object_ids, _low, masks, object_scores in tracker.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=len(image_paths),
            reverse=False,
            propagate_preflight=True,
        ):
            if int(frame_index) != len(image_paths) - 1:
                continue
            positions = [i for i, value in enumerate(object_ids) if int(value) == 1]
            if len(positions) != 1:
                raise RuntimeError("tracked NVOS object identity differs")
            position = positions[0]
            target_mask = masks[position].detach().float().cpu().squeeze().numpy() > 0.0
            target_score = float(torch.as_tensor(object_scores[position]).detach().cpu())
        if hasattr(tracker, "reset_state"):
            tracker.reset_state(state)
        del state, tracker, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if target_mask is None:
        raise RuntimeError("official SAM3 video tracker returned no target mask")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "target_probability.npy"
    temporary_prediction = output_dir / f".target_probability.{os.getpid()}.npy"
    with temporary_prediction.open("wb") as handle:
        np.save(handle, target_mask.astype(np.float32), allow_pickle=False)
    os.replace(temporary_prediction, prediction_path)
    receipt = {
        "schema_version": 1,
        "artifact_type": RECEIPT_TYPE,
        "scene_id": scene_id,
        "target_frame_id": target_id,
        "shape": list(target_mask.shape),
        "threshold": 0.5,
        "prediction": {"path": str(prediction_path), "sha256": sha256_file(prediction_path)},
        "prediction_sealed_before_target_ground_truth": True,
        "target_mask_opened": False,
        "target_metric_opened": False,
        "development_contract": "protocol_authorized_registered_rgb_sam3_video_memory",
        "strict_unseen_eligible": False,
        "tracker_score": target_score,
        "registered_path_frame_ids": [frame_ids[index] for index in path_indices],
        "registered_path_direction": "cyclic_shortest",
        "prompt_frame_id": prompt_id,
        "prompt_points": int(len(points)),
        "prompt_authority_shape": [height, width],
        "prompt_rgb_shape": [image_height, image_width],
        "inputs": {
            "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
            "inventory": {"path": str(inventory_path), "sha256": sha256_file(inventory_path)},
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        },
    }
    receipt_path = output_dir / "prediction_receipt.json"
    _atomic_json(receipt_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-threshold", type=float, default=0.5)
    parser.add_argument("--points-per-sign", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    receipt = predict(parser.parse_args(argv))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
