#!/usr/bin/env python3
"""Render and diagnose a frozen carrier-native LUDVIG bridge on NVOS."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from radio_gs.config import load_config
from radio_gs.evaluation.promptable_segmentation import (
    compute_binary_metrics,
    load_ground_truth_mask,
    resize_mask_nearest,
)
from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.scripts.eval_nvos_gaussian_first import _resize_nvos_score_for_evaluation
from radio_gs.scripts.materialize_nvos_ludvig_carrier_native_bridge import (
    ARTIFACT_TYPE,
    _sha256,
    _write_json,
    _write_numpy,
)
from radio_gs.scripts.render_promptable_nvs_features import resolve_protocol_views


def _bound(path: str | Path, digest: str, label: str) -> Path:
    source = Path(path).expanduser().resolve(strict=True)
    if len(digest) != 64 or _sha256(source) != digest:
        raise ValueError(f"{label} SHA-256 differs")
    return source


def _xyz_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _scene(manifest: Mapping[str, Any], scene_id: str) -> Mapping[str, Any]:
    matches = [x for x in manifest.get("scenes", []) if x.get("scene_id") == scene_id]
    if len(matches) != 1:
        raise ValueError("benchmark scene authority differs")
    return matches[0]


def _frame(scene: Mapping[str, Any], frame_id: str) -> Mapping[str, Any]:
    raw = scene.get("frames", [])
    values = raw.values() if isinstance(raw, Mapping) else raw
    matches = [x for x in values if str(x.get("frame_id")) == frame_id]
    if len(matches) != 1:
        raise ValueError("benchmark target frame authority differs")
    return matches[0]


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    receipt_path = _bound(args.bridge_receipt, args.bridge_receipt_sha256, "bridge receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("artifact_type") != ARTIFACT_TYPE
        or receipt.get("scene_id") != args.scene_id
        or any(receipt.get("safety", {}).get(key) is not False for key in (
            "target_mask_opened", "target_metric_opened", "nearest_neighbor_transfer",
            "carrier_modified", "second_persistent_scene_field_created",
        ))
    ):
        raise ValueError("bridge receipt authority differs")
    state_path = _bound(args.bridge_state, args.bridge_state_sha256, "bridge state")
    if receipt["outputs"]["primitive_state"] != {
        "path": str(state_path), "sha256": args.bridge_state_sha256
    }:
        raise ValueError("bridge receipt does not bind state")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    primitive = state.get("primitive_probability")
    if (
        state.get("artifact_type") != ARTIFACT_TYPE
        or state.get("scene_id") != args.scene_id
        or not torch.is_tensor(primitive)
        or primitive.dtype != torch.float32
        or primitive.ndim != 1
        or not bool(torch.isfinite(primitive).all())
        or bool(((primitive < 0) | (primitive > 1)).any())
    ):
        raise ValueError("bridge primitive state is invalid")

    manifest_path = _bound(args.manifest, args.manifest_sha256, "benchmark manifest")
    if receipt["inputs"]["dataset_manifest"] != {
        "path": str(manifest_path), "sha256": args.manifest_sha256
    }:
        raise ValueError("bridge and evaluator manifests differ")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene = _scene(manifest, args.scene_id)
    targets = list(scene.get("evaluation_frame_ids", []))
    if len(targets) != 1:
        raise ValueError("NVOS bridge evaluator requires one target frame")
    target_frame = str(targets[0])
    target_record = _frame(scene, target_frame)

    queue_scene = Path(args.queue_root).expanduser().resolve(strict=True) / "scenes" / args.scene_id
    config_path = queue_scene / "gaussfm_main_track.yaml"
    checkpoint_path = queue_scene / "feature_field" / "checkpoints" / "best.pth"
    camera_path = queue_scene / "rgb_to_colmap_camera_mapping.json"
    authority = receipt["inputs"]
    if (
        _sha256(config_path) != authority["current_config"]["sha256"]
        or _sha256(camera_path) != authority["camera_mapping"]["sha256"]
    ):
        raise ValueError("current renderer config/camera authority differs")
    checkpoint_load_path = checkpoint_path
    if args.geometry_checkpoint_local_copy:
        checkpoint_load_path = _bound(
            args.geometry_checkpoint_local_copy,
            args.expected_geometry_checkpoint_sha256,
            "local geometry checkpoint copy",
        )
        if checkpoint_load_path.stat().st_size != checkpoint_path.stat().st_size:
            raise ValueError("local geometry checkpoint copy size differs")
    config = load_config(str(config_path))
    camera_mapping = json.loads(camera_path.read_text(encoding="utf-8"))
    views = resolve_protocol_views(
        manifest, scene_id=args.scene_id,
        scene_root=Path(str(config.scene_root)).resolve(), camera_mapping=camera_mapping,
    )
    selected = [x for x in views if str(x["frame_id"]) == target_frame]
    if len(selected) != 1 or selected[0].get("role") != "evaluation":
        raise ValueError("registered target camera authority differs")
    device = torch.device(args.device)
    model, _codec, renderer, _sharpener, refiner, _cfg, _hybrid = load_render_pipeline(
        str(config_path), str(checkpoint_load_path), device,
        strict_checkpoint_contract=True, load_ply_rgb_features=False,
    )
    if refiner is not None or model.get_xyz().shape[0] != primitive.numel():
        raise ValueError("current carrier rows/refiner differ")
    if _xyz_sha256(model.get_xyz()) != state.get("geometry_xyz_sha256"):
        raise ValueError("current carrier geometry hash differs")
    height, width = int(receipt["render"]["height"]), int(receipt["render"]["width"])
    pose_cpu = torch.from_numpy(np.asarray(selected[0]["w2c"], dtype="<f4").copy()).contiguous()
    rendered = renderer.render_feature_rows(
        model, pose_cpu.to(device), primitive[:, None].to(device),
        feature_height=height, feature_width=width, alpha_normalize=True,
    )
    score = rendered["feature_map"][0].float().clamp(0, 1).cpu().contiguous()
    alpha = rendered["alpha_map"].float().cpu().contiguous()
    if score.shape != (height, width) or alpha.shape != (height, width):
        raise ValueError("target render shape differs")
    output = Path(args.output_dir).expanduser().resolve()
    score_path = output / "current_carrier_target_probability.npy"
    score_sha = _write_numpy(score_path, score.numpy())
    del rendered, model, renderer
    gc.collect()
    torch.cuda.empty_cache()

    # No target-derived selector or ground-truth bytes are opened before the
    # current-carrier score is atomically persisted and hashed above.
    native_selector_path = _bound(
        args.native_target_selector, args.native_target_selector_sha256,
        "native target selector",
    )
    method_path = _bound(args.method_v1_prediction, args.method_v1_prediction_sha256, "Method-v1 prediction")
    target_path = Path(str(target_record.get("ground_truth") or target_record.get("gt_mask_path"))).resolve(strict=True)
    declared_gt_sha = str(target_record.get("ground_truth_sha256") or "")
    if _sha256(target_path) != declared_gt_sha:
        raise ValueError("target ground-truth hash differs")
    ground_truth = load_ground_truth_mask(target_path)
    resized = _resize_nvos_score_for_evaluation(
        score.numpy(), tuple(map(int, ground_truth.shape)), registered_forward_unary="none"
    )
    threshold = float(args.threshold_parameter) / 255.0
    bridge_prediction = resized > threshold
    native = np.asarray(Image.open(native_selector_path)) > 0
    method_margin = np.load(method_path, allow_pickle=False)
    if (
        native.shape != ground_truth.shape
        or method_margin.shape != score.shape
        or not np.isfinite(method_margin).all()
    ):
        raise ValueError("paired diagnostic masks do not match target shape")
    method_prediction = _resize_nvos_score_for_evaluation(
        method_margin, tuple(map(int, ground_truth.shape)),
        registered_forward_unary="none",
    ) >= 0.0
    native_coarse = resize_mask_nearest(native, score.shape).astype(bool)
    bridge_coarse = score.numpy() > threshold
    method_coarse = method_margin >= 0.0
    intersection_coarse = np.logical_and(method_coarse, bridge_coarse)
    intersection_margin = np.where(intersection_coarse, 0.5, -0.5).astype(np.float32)
    intersection = _resize_nvos_score_for_evaluation(
        intersection_margin, tuple(map(int, ground_truth.shape)),
        registered_forward_unary="none",
    ) >= 0.0
    union = np.logical_or(bridge_coarse, native_coarse)
    agreement_iou = float(
        np.logical_and(bridge_coarse, native_coarse).sum() / max(1, union.sum())
    )
    report = {
        "schema_version": 1,
        "artifact_type": "nvos_ludvig_carrier_native_bridge_target_diagnostic_v1",
        "scene_id": args.scene_id,
        "target_frame": target_frame,
        "bridge_receipt": {"path": str(receipt_path), "sha256": args.bridge_receipt_sha256},
        "bridge_state": {"path": str(state_path), "sha256": args.bridge_state_sha256},
        "target_render": {
            "path": str(score_path), "sha256": score_sha,
            "pose_sha256": tensor_sha256(pose_cpu),
            "alpha_supported_fraction": float((alpha > 0).double().mean()),
            "score_min": float(score.min()), "score_max": float(score.max()),
            "score_mean": float(score.mean()), "threshold": threshold,
        },
        "paired_development": {
            "carrier_native_bridge_metrics": compute_binary_metrics(bridge_prediction, ground_truth),
            "native_image_space_selector_metrics": compute_binary_metrics(native, ground_truth),
            "method_v1_metrics": compute_binary_metrics(method_prediction, ground_truth),
            "method_v1_intersection_bridge_metrics": compute_binary_metrics(intersection, ground_truth),
            "bridge_vs_native_selector_intersection_over_union": agreement_iou,
            "bridge_foreground_fraction": float(bridge_prediction.mean()),
            "native_foreground_fraction": float(native.mean()),
        },
        "safety": {
            "score_frozen_before_target_selector_or_mask": True,
            "target_rgb_bytes_directly_opened": False,
            "upstream_native_scalar_target_rgb_all_view_assisted": True,
            "native_target_selector_bytes_opened_after_score_freeze": True,
            "target_mask_opened_for_scoring_only": True,
            "nearest_neighbor_transfer": False,
            "development_only": True,
            "strict_unseen_eligible": False,
            "eligibility": "development_all_view_target_rgb_assisted_only",
        },
    }
    _write_json(output / "target_diagnostic.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--bridge-receipt", required=True)
    parser.add_argument("--bridge-receipt-sha256", required=True)
    parser.add_argument("--bridge-state", required=True)
    parser.add_argument("--bridge-state-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--geometry-checkpoint-local-copy")
    parser.add_argument("--expected-geometry-checkpoint-sha256")
    parser.add_argument("--native-target-selector", required=True)
    parser.add_argument("--native-target-selector-sha256", required=True)
    parser.add_argument("--method-v1-prediction", required=True)
    parser.add_argument("--method-v1-prediction-sha256", required=True)
    parser.add_argument("--threshold-parameter", type=float, default=75.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    print(json.dumps(evaluate(parser.parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
