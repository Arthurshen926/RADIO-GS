#!/usr/bin/env python3
"""Attribute a persisted LUDVIG-SAM query field with the frozen renderer.

This is a diagnostic, not a strict-unseen benchmark submission.  The input
field is query-specific and was built with all registered RGB views.  Target
masks remain sealed until the reference-selected predictions are persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from plyfile import PlyData

from radio_gs.config import load_config
from radio_gs.evaluation.promptable_segmentation import load_ground_truth_mask
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.scripts.render_promptable_nvs_features import resolve_protocol_views


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene(manifest: dict, scene_id: str) -> dict:
    rows = [x for x in manifest["scenes"] if str(x["scene_id"]) == scene_id]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one scene {scene_id!r}")
    return rows[0]


def _view(views: list[dict], frame_id: str) -> dict:
    rows = [x for x in views if str(x["frame_id"]) == frame_id]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one protocol view {frame_id!r}")
    return rows[0]


def _ply_render_keys(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertex = PlyData.read(str(path))["vertex"].data
    names = set(vertex.dtype.names or ())
    wanted = ["x", "y", "z", "opacity"]
    wanted.extend(sorted(x for x in names if x.startswith("scale_")))
    wanted.extend(sorted(x for x in names if x.startswith("rot_")))
    missing = [x for x in ("x", "y", "z") if x not in names]
    if missing:
        raise ValueError(f"PLY lacks geometry fields: {missing}")
    fields = [x for x in wanted if x in names]
    matrix = np.ascontiguousarray(
        np.stack([vertex[x].astype("<f4", copy=False) for x in fields], axis=1)
    )
    keys = matrix.view(np.dtype((np.void, matrix.shape[1] * 4))).reshape(-1)
    return matrix[:, :3], keys


def _exact_subset_mapping(full_keys: np.ndarray, subset_keys: np.ndarray) -> np.ndarray:
    order = np.argsort(full_keys, kind="stable")
    ordered = full_keys[order]
    offsets = np.searchsorted(ordered, subset_keys)
    safe = np.minimum(offsets, max(0, len(ordered) - 1))
    valid = (offsets < len(ordered)) & (ordered[safe] == subset_keys)
    if not bool(valid.all()):
        raise ValueError(
            f"SAM PLY is not an exact carrier subset: {int(valid.sum())}/{len(valid)}"
        )
    mapped = order[offsets]
    if len(np.unique(mapped)) != len(mapped):
        raise ValueError("SAM PLY to carrier mapping is not one-to-one")
    return mapped.astype(np.int64, copy=False)


def _normalize_candidates(values: torch.Tensor) -> torch.Tensor:
    flat = values.flatten(1)
    lower = flat.amin(dim=1)[:, None, None]
    upper = flat.amax(dim=1)[:, None, None]
    return (values - lower) / (upper - lower).clamp_min(1e-8)


def _resolve_render_resolution(
    config: object,
    mode: str,
    *,
    registered_resolution: tuple[int, int] | None = None,
) -> tuple[int, int]:
    if mode == "feature":
        height = int(getattr(config, "feature_height"))
        width = int(getattr(config, "feature_width"))
    elif mode == "native":
        height = int(getattr(config, "image_height"))
        width = int(getattr(config, "image_width"))
    elif mode == "registered":
        if registered_resolution is None:
            raise ValueError("registered render resolution was not provided")
        height, width = map(int, registered_resolution)
    else:
        raise ValueError(f"unknown render-resolution mode: {mode!r}")
    if height <= 0 or width <= 0:
        raise ValueError("render resolution must be positive")
    return height, width


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return float(intersection / union) if union else 1.0


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene = _scene(manifest, args.scene_id)
    queue_scene = args.queue_root.resolve() / "scenes" / args.scene_id
    camera_mapping = json.loads(
        (queue_scene / "rgb_to_colmap_camera_mapping.json").read_text(encoding="utf-8")
    )
    carrier_config = load_config(str(args.carrier_config.resolve()))
    prompt_frame = str(scene["prompt_frame_ids"][0])
    reference_mask = load_ground_truth_mask(scene["prompt"]["mask_path"]).astype(bool)
    render_height, render_width = _resolve_render_resolution(
        carrier_config,
        args.render_resolution,
        registered_resolution=tuple(reference_mask.shape),
    )
    views = resolve_protocol_views(
        manifest,
        scene_id=args.scene_id,
        scene_root=Path(str(carrier_config.scene_root)).resolve(),
        camera_mapping=camera_mapping,
    )
    device = torch.device(args.device)
    model, _, renderer, _, _, _, _ = load_render_pipeline(
        str(args.carrier_config.resolve()),
        str(args.carrier_checkpoint.resolve()),
        device,
        strict_checkpoint_contract=True,
        load_ply_rgb_features=False,
    )

    carrier_ply = Path(str(carrier_config.ply_path)).resolve()
    carrier_xyz, carrier_keys = _ply_render_keys(carrier_ply)
    sam_xyz, sam_keys = _ply_render_keys(args.sam_ply.resolve())
    features = np.load(args.sam_features.resolve(), allow_pickle=False)
    if features.shape != (len(sam_keys), 3) or features.dtype != np.float32:
        raise ValueError("SAM features must be float32 [sam_ply_rows,3]")
    model_xyz = model.get_xyz().detach().float().cpu().numpy()
    if not np.array_equal(model_xyz, carrier_xyz):
        raise ValueError("frozen checkpoint rows differ from its declared carrier PLY")
    mapping = _exact_subset_mapping(carrier_keys, sam_keys)
    if args.require_full_alignment and len(mapping) != len(carrier_keys):
        raise ValueError("the registered primary diagnostic requires full row alignment")

    rows = torch.zeros(len(carrier_keys), 4, dtype=torch.float32, device=device)
    index = torch.from_numpy(mapping).to(device)
    rows[index, :3] = torch.from_numpy(features).to(device)
    rows[index, 3] = 1.0

    def render(view: dict) -> np.ndarray:
        pose = torch.from_numpy(view["w2c"].copy()).float().to(device)
        raw = renderer.render_feature_rows(
            model,
            pose,
            rows,
            feature_height=render_height,
            feature_width=render_width,
        )["feature_map"]
        candidates = raw[:3]
        if args.missing_row_policy == "valid_normalized":
            candidates = torch.where(
                raw[3:].expand_as(candidates) > 1e-6,
                candidates / raw[3:].clamp_min(1e-6),
                torch.zeros_like(candidates),
            )
        return _normalize_candidates(candidates).cpu().numpy().astype(np.float32)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference_scores = render(_view(views, prompt_frame))
    reference_score_path = output / "reference_candidates.npy"
    np.save(reference_score_path, reference_scores, allow_pickle=False)
    thresholds = np.arange(0.99, 0.02, -0.01, dtype=np.float64)
    best = None
    calibration = []
    for candidate in range(3):
        score = cv2.resize(
            reference_scores[candidate],
            (reference_mask.shape[1], reference_mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        candidate_best = (-1.0, None)
        for threshold in thresholds:
            value = _iou(score > float(threshold), reference_mask)
            if value > candidate_best[0]:
                candidate_best = (value, float(threshold))
        calibration.append(
            {
                "candidate": candidate,
                "reference_iou": candidate_best[0],
                "threshold": candidate_best[1],
            }
        )
        if best is None or candidate_best[0] > best[0]:
            best = (candidate_best[0], candidate, candidate_best[1])
    assert best is not None
    receipt = {
        "schema_version": "spin9_sam_query_interface_reference_receipt_v1",
        "scene_id": args.scene_id,
        "selected_candidate": best[1],
        "selected_threshold": best[2],
        "selected_reference_iou": best[0],
        "calibration": calibration,
        "target_masks_opened": False,
    }
    receipt_path = output / "reference_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    predictions: dict[str, np.ndarray] = {}
    score_paths: dict[str, str] = {}
    for frame_id in map(str, scene["evaluation_frame_ids"]):
        score = render(_view(views, frame_id))[best[1]]
        path = output / "scores" / f"{frame_id}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, score, allow_pickle=False)
        predictions[frame_id] = score
        score_paths[frame_id] = str(path)

    frame_metrics = []
    for frame_id in map(str, scene["evaluation_frame_ids"]):
        frame = next(x for x in scene["frames"] if str(x["frame_id"]) == frame_id)
        gt = load_ground_truth_mask(frame["ground_truth"]).astype(bool)
        score = cv2.resize(
            predictions[frame_id],
            (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        pred = score > float(best[2])
        frame_metrics.append(
            {
                "frame_id": frame_id,
                "foreground_iou": _iou(pred, gt),
                "pixel_accuracy": float((pred == gt).mean()),
            }
        )
    report = {
        "schema_version": "spin9_sam_query_interface_frozen_renderer_result_v1",
        "claim_scope": "posthoc_component_attribution_only",
        "scene_id": args.scene_id,
        "protocol_hash": manifest["protocol_hash"],
        "manifest_sha256": _sha256(manifest_path),
        "sam_features": str(args.sam_features.resolve()),
        "sam_features_sha256": _sha256(args.sam_features.resolve()),
        "sam_ply": str(args.sam_ply.resolve()),
        "sam_ply_sha256": _sha256(args.sam_ply.resolve()),
        "carrier_ply": str(carrier_ply),
        "carrier_ply_sha256": _sha256(carrier_ply),
        "carrier_rows": len(carrier_keys),
        "mapped_sam_rows": len(mapping),
        "mapped_fraction": float(len(mapping) / len(carrier_keys)),
        "missing_row_policy": args.missing_row_policy,
        "render_resolution_mode": args.render_resolution,
        "renderer_resolution": [render_height, render_width],
        "reference_receipt": receipt,
        "reference_receipt_path": str(receipt_path),
        "score_paths": score_paths,
        "prediction_persisted_before_target_mask_access": True,
        "persisted_field_used_target_rgb_during_original_generation": True,
        "target_masks_used_during_original_generation": False,
        "foreground_iou": float(np.mean([x["foreground_iou"] for x in frame_metrics])),
        "pixel_accuracy": float(np.mean([x["pixel_accuracy"] for x in frame_metrics])),
        "frames": frame_metrics,
    }
    report_path = output / f"{args.scene_id}_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--carrier-config", type=Path, required=True)
    parser.add_argument("--carrier-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-features", type=Path, required=True)
    parser.add_argument("--sam-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--render-resolution",
        choices=("feature", "native", "registered"),
        default="feature",
        help=(
            "Raster resolution for the query-specific scalar field. 'feature' "
            "preserves the legacy RADIO patch-grid diagnostic; 'native' uses "
            "the carrier config's original image resolution; 'registered' uses "
            "the permitted reference-mask resolution shared by the benchmark views."
        ),
    )
    parser.add_argument(
        "--missing-row-policy", choices=("zero_fill", "valid_normalized"), default="zero_fill"
    )
    parser.add_argument("--require-full-alignment", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"scene_id": report["scene_id"], "foreground_iou": report["foreground_iou"]}))


if __name__ == "__main__":
    main()
