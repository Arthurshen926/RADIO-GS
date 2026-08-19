#!/usr/bin/env python3
"""Build a target-view exact-W consensus rerender for NVOS round two.

This is a query-transient RGB-assisted readout stage.  It consumes only
already sealed Method-v1 field/SAM predictions and the registered target
camera.  It never opens target masks or metrics.  For each frozen target view
it materializes the exact accepted-hit 3DGS compositor in memory, applies
``W.T`` with exact visibility normalization, takes a coordinate-wise median
of clipped field/box/point log odds, and rerenders that primitive posterior
with the same ``W``.  The sparse compositor itself is discarded after each
scene so a second persistent semantic field is not created.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.config import load_config
from radio_gs.rendering.contribution_compositor import (
    rasterize_single_view_contributions,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.scripts.predict_nvos_method_v1_transient_sam import (
    DEFAULT_EVALUATION_CONTRACT,
    DEFAULT_METHOD_AUTHORITY,
    SAM_HEIGHT,
    SAM_WIDTH,
    _sha256,
    load_signed_field_prompt,
)
from radio_gs.scripts.render_promptable_nvs_features import resolve_protocol_views


CANDIDATE_ID = "nvos-method-v1-two-round-exact-logodds-sam3-v1"
COMPOSITOR_MODE = (
    "gsplat_exact_accepted_hits_front_to_back_alpha_times_exclusive_transmittance"
)
CONSENSUS_MODE = "coordinatewise_median_of_three_visibility_normalized_logodds"
PROBABILITY_EPSILON = 0.05


def robust_logodds_consensus(
    probabilities: Sequence[torch.Tensor], *, epsilon: float = PROBABILITY_EPSILON
) -> torch.Tensor:
    """Return the fixed coordinate-wise median of finite probability logits."""

    if len(probabilities) != 3:
        raise ValueError("fixed NVOS consensus requires field, box, and point inputs")
    values = [torch.as_tensor(value).float() for value in probabilities]
    if not values or any(value.shape != values[0].shape for value in values):
        raise ValueError("consensus probabilities must have one aligned shape")
    if not 0.0 < float(epsilon) < 0.5:
        raise ValueError("epsilon must lie in (0,0.5)")
    if any(
        not bool(torch.isfinite(value).all())
        or bool(((value < 0.0) | (value > 1.0)).any())
        for value in values
    ):
        raise ValueError("consensus probabilities must be finite in [0,1]")
    logits = torch.stack(
        [torch.logit(value.clamp(float(epsilon), 1.0 - float(epsilon))) for value in values],
        dim=0,
    )
    return torch.sigmoid(torch.median(logits, dim=0).values)


def exact_adjoint_probability(
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    weights: torch.Tensor,
    pixel_probability: torch.Tensor,
    *,
    num_gaussians: int,
    precomputed_visible_mass: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply exact ``W.T`` and visibility normalization on one device."""

    gids = torch.as_tensor(gaussian_ids).long().reshape(-1)
    pids = torch.as_tensor(pixel_ids).long().reshape(-1)
    w = torch.as_tensor(weights).float().reshape(-1)
    probability = torch.as_tensor(pixel_probability).float().reshape(-1)
    if gids.shape != pids.shape or gids.shape != w.shape or not gids.numel():
        raise ValueError("exact compositor triplets must be nonempty and aligned")
    if int(num_gaussians) <= 0 or int(gids.min()) < 0 or int(gids.max()) >= int(num_gaussians):
        raise ValueError("Gaussian id falls outside the declared carrier")
    if int(pids.min()) < 0 or int(pids.max()) >= probability.numel():
        raise ValueError("pixel id falls outside the probability raster")
    if not bool(torch.isfinite(w).all()) or bool((w <= 0).any()):
        raise ValueError("exact compositor weights must be finite and positive")
    if not bool(torch.isfinite(probability).all()) or bool(
        ((probability < 0) | (probability > 1)).any()
    ):
        raise ValueError("pixel probability must be finite in [0,1]")
    if precomputed_visible_mass is None:
        visible_mass = torch.zeros(
            int(num_gaussians), device=w.device, dtype=torch.float32
        )
        visible_mass.index_add_(0, gids, w)
    else:
        visible_mass = torch.as_tensor(precomputed_visible_mass).to(
            device=w.device, dtype=torch.float32
        )
        if visible_mass.shape != (int(num_gaussians),) or not bool(
            torch.isfinite(visible_mass).all()
        ) or bool((visible_mass < 0).any()):
            raise ValueError("precomputed exact visibility is invalid")
    weighted_sum = torch.zeros_like(visible_mass)
    weighted_sum.index_add_(0, gids, w * probability[pids])
    primitive = torch.full_like(visible_mass, 0.5)
    visible = visible_mass > 0
    primitive[visible] = weighted_sum[visible] / visible_mass[visible]
    return primitive.clamp(0.0, 1.0), visible_mass


def exact_forward_probability(
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    weights: torch.Tensor,
    primitive_probability: torch.Tensor,
    *,
    height: int,
    width: int,
    unsupported_fallback: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply normalized ``W`` to a primitive probability vector."""

    gids = torch.as_tensor(gaussian_ids).long().reshape(-1)
    pids = torch.as_tensor(pixel_ids).long().reshape(-1)
    w = torch.as_tensor(weights).float().reshape(-1)
    primitive = torch.as_tensor(primitive_probability).float().reshape(-1)
    fallback = torch.as_tensor(unsupported_fallback).float().reshape(-1)
    pixels = int(height) * int(width)
    if fallback.shape != (pixels,):
        raise ValueError("unsupported fallback has the wrong raster shape")
    numerator = torch.zeros(pixels, device=w.device, dtype=torch.float32)
    mass = torch.zeros_like(numerator)
    numerator.index_add_(0, pids, w * primitive[gids])
    mass.index_add_(0, pids, w)
    supported = mass > 0
    output = fallback.clone()
    output[supported] = numerator[supported] / mass[supported]
    return output.reshape(int(height), int(width)).clamp(0.0, 1.0), mass.reshape(
        int(height), int(width)
    )


def _resize_probability(value: np.ndarray, *, height: int, width: int) -> torch.Tensor:
    source = torch.from_numpy(np.asarray(value, dtype=np.float32)).view(1, 1, *value.shape)
    return F.interpolate(source, size=(int(height), int(width)), mode="bilinear", align_corners=False)[
        0, 0
    ]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_numpy(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(value, dtype=np.float32), allow_pickle=False)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _write_torch(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(value), temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _load_manifest(path: str | Path, expected_sha256: str, *, kind: str) -> tuple[dict[str, Any], Path]:
    source = Path(path).expanduser().resolve(strict=True)
    if len(str(expected_sha256)) != 64 or _sha256(source) != str(expected_sha256):
        raise ValueError(f"sealed parent manifest SHA-256 differs: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("kind") != kind:
        raise ValueError(f"sealed parent manifest kind differs: {source}")
    if bool(value.get("target_mask_opened", True)) or bool(value.get("target_metric_opened", True)):
        raise ValueError("parent prediction opened a target mask or metric")
    return value, source


def _prediction_path(manifest: Mapping[str, Any], manifest_path: Path, scene: str, frame: str) -> Path:
    relative = Path(str(manifest["predictions"][scene][frame]))
    root = Path(str(manifest.get("prediction_root", ".")))
    if not root.is_absolute():
        root = manifest_path.parent / root
    source = (relative if relative.is_absolute() else root / relative).resolve(strict=True)
    if _sha256(source) != str(manifest["prediction_sha256"][scene][frame]):
        raise ValueError(f"sealed prediction SHA-256 differs: {scene}/{frame}")
    return source


def _float32_rows_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


@torch.inference_mode()
def build(args: argparse.Namespace) -> dict[str, Any]:
    scene_ids = [str(value) for value in args.scene_ids]
    if not scene_ids or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("--scene-id must be a nonempty unique list")
    if (SAM_HEIGHT, SAM_WIDTH) != (756, 1008):
        raise RuntimeError("frozen Method-v1 SAM raster authority changed")
    box, box_path = _load_manifest(
        args.box_manifest,
        args.expected_box_manifest_sha256,
        kind="promptable_nvs_method_v1_field_box_sam3_predictions",
    )
    point, point_path = _load_manifest(
        args.point_manifest,
        args.expected_point_manifest_sha256,
        kind="promptable_nvs_method_v1_transient_sam_predictions",
    )
    if box.get("protocol_hash") != point.get("protocol_hash"):
        raise ValueError("box and point parents disagree on protocol hash")
    output_root = Path(args.output_dir).expanduser().resolve()
    if (output_root / "consensus_manifest.json").exists():
        raise FileExistsError(output_root / "consensus_manifest.json")
    queue_root = Path(args.queue_root).expanduser().resolve(strict=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("exact target-view compositor requires CUDA")

    outputs: dict[str, dict[str, str]] = {}
    output_hashes: dict[str, dict[str, str]] = {}
    state_assets: dict[str, dict[str, str]] = {}
    receipts: list[dict[str, str]] = []
    started = time.time()
    for scene_id in scene_ids:
        source = load_signed_field_prompt(
            dataset_manifest_path=args.manifest,
            prompt_manifest_path=args.signed_field_prompt_manifest,
            method_authority_path=args.method_authority,
            evaluation_contract_path=args.evaluation_contract,
            scene_id=scene_id,
        )
        frame_id = str(source["frame_id"])
        for parent, label in ((box, "box"), (point, "point")):
            if list(parent.get("predictions", {}).get(scene_id, {})) != [frame_id]:
                raise ValueError(f"{label} parent frame differs for {scene_id}")
        box_score_path = _prediction_path(box, box_path, scene_id, frame_id)
        point_score_path = _prediction_path(point, point_path, scene_id, frame_id)
        box_margin = np.load(box_score_path, allow_pickle=False)
        point_margin = np.load(point_score_path, allow_pickle=False)
        if box_margin.shape != (SAM_HEIGHT, SAM_WIDTH) or point_margin.shape != box_margin.shape:
            raise ValueError(f"sealed SAM parent raster differs for {scene_id}")

        scene_root = queue_root / "scenes" / scene_id
        config_path = scene_root / "gaussfm_main_track.yaml"
        checkpoint_path = scene_root / "feature_field/checkpoints/best.pth"
        camera_map_path = scene_root / "rgb_to_colmap_camera_mapping.json"
        for path in (config_path, checkpoint_path, camera_map_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        dataset_manifest_path = Path(args.manifest).expanduser().resolve(strict=True)
        dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        config = load_config(str(config_path))
        camera_mapping = json.loads(camera_map_path.read_text(encoding="utf-8"))
        views = resolve_protocol_views(
            dataset_manifest,
            scene_id=scene_id,
            scene_root=Path(str(config.scene_root)).resolve(),
            camera_mapping=camera_mapping,
        )
        selected_views = [view for view in views if str(view["frame_id"]) == frame_id]
        if len(selected_views) != 1 or selected_views[0].get("role") != "evaluation":
            raise ValueError(f"registered target view authority differs for {scene_id}/{frame_id}")
        view = selected_views[0]
        model, _codec, renderer, _sharpener, refiner, _field_config, _is_hybrid = load_render_pipeline(
            str(config_path),
            str(checkpoint_path),
            device,
            strict_checkpoint_contract=True,
            load_ply_rgb_features=False,
        )
        if refiner is not None:
            raise ValueError("query-transient exact carrier forbids an RGB screen refiner")
        pose = torch.from_numpy(view["w2c"]).to(device=device, dtype=torch.float32)
        xyz = model.get_xyz()
        num_gaussians = int(xyz.shape[0])
        geometry_xyz_sha256 = _float32_rows_sha256(xyz)
        hits = rasterize_single_view_contributions(
            model, renderer, pose, height=SAM_HEIGHT, width=SAM_WIDTH
        )
        gids = hits["gaussian_ids"]
        pids = hits["pixel_ids"]
        weights = hits["weights"]
        keep = weights > 0
        gids, pids, weights = gids[keep], pids[keep], weights[keep]
        if not gids.numel() or bool((pids[1:] < pids[:-1]).any()):
            raise ValueError("exact target compositor is empty or not pixel-grouped")

        field_probability = torch.sigmoid(
            _resize_probability(
                np.asarray(source["signed_margin"], dtype=np.float32),
                height=SAM_HEIGHT,
                width=SAM_WIDTH,
            ).to(device)
        )
        box_probability = torch.from_numpy(
            np.where(box_margin >= 0.0, 1.0 - PROBABILITY_EPSILON, PROBABILITY_EPSILON).astype(
                np.float32
            )
        ).to(device)
        point_probability = torch.from_numpy(
            np.clip(point_margin + 0.5, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON).astype(
                np.float32
            )
        ).to(device)
        pixel_fallback = robust_logodds_consensus(
            (field_probability, box_probability, point_probability)
        )
        lifted: list[torch.Tensor] = []
        visible_mass: torch.Tensor | None = None
        for probability in (field_probability, box_probability, point_probability):
            primitive, current_visible = exact_adjoint_probability(
                gids,
                pids,
                weights,
                probability,
                num_gaussians=num_gaussians,
                precomputed_visible_mass=visible_mass,
            )
            if visible_mass is None:
                visible_mass = current_visible
            lifted.append(primitive)
        assert visible_mass is not None
        primitive_consensus = robust_logodds_consensus(lifted)
        rerender, pixel_mass = exact_forward_probability(
            gids,
            pids,
            weights,
            primitive_consensus,
            height=SAM_HEIGHT,
            width=SAM_WIDTH,
            unsupported_fallback=pixel_fallback,
        )
        visible = visible_mass > 0
        rerender_path = output_root / "rerender" / scene_id / f"{frame_id}.npy"
        state_path = output_root / "primitive_state" / scene_id / f"{frame_id}.pt"
        rerender_sha = _write_numpy(rerender_path, rerender.cpu().numpy())
        state_sha = _write_torch(
            state_path,
            {
                "scene_id": scene_id,
                "frame_id": frame_id,
                "primitive_probability": primitive_consensus.cpu(),
                "visible_mass": visible_mass.cpu(),
                "geometry_xyz_sha256": geometry_xyz_sha256,
                "compositor_mode": COMPOSITOR_MODE,
                "consensus_mode": CONSENSUS_MODE,
            },
        )
        receipt = {
            "schema_version": 1,
            "artifact_type": "radio_gs_nvos_two_round_exact_consensus_receipt",
            "candidate_id": CANDIDATE_ID,
            "scene_id": scene_id,
            "frame_id": frame_id,
            "parents": {
                "signed_field": {
                    "path": str(source["signed_margin_path"]),
                    "sha256": source["signed_margin_sha256"],
                },
                "box": {"path": str(box_score_path), "sha256": _sha256(box_score_path)},
                "point": {"path": str(point_score_path), "sha256": _sha256(point_score_path)},
            },
            "registered_target_carrier": {
                "queue_scene": str(scene_root),
                "config": {"path": str(config_path), "sha256": _sha256(config_path)},
                "geometry_checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": _sha256(checkpoint_path),
                },
                "camera_mapping": {
                    "path": str(camera_map_path),
                    "sha256": _sha256(camera_map_path),
                },
                "camera_name": view["camera_name"],
                "colmap_camera_name": view["colmap_camera_name"],
                "pose_sha256": hashlib.sha256(
                    pose.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
                ).hexdigest(),
                "geometry_xyz_sha256": geometry_xyz_sha256,
                "num_gaussians": num_gaussians,
                "raster_height": SAM_HEIGHT,
                "raster_width": SAM_WIDTH,
            },
            "exact_transport": {
                "compositor": COMPOSITOR_MODE,
                "hit_count": int(gids.numel()),
                "positive_weight_only": True,
                "visible_gaussians": int(visible.sum()),
                "visible_mass_sum": float(visible_mass.sum()),
                "supported_pixels": int((pixel_mass > 0).sum()),
                "supported_fraction": float((pixel_mass > 0).float().mean()),
                "operator": "W.T_probability_over_W.T_one_then_W_primitive_over_W_one",
            },
            "consensus": {
                "mode": CONSENSUS_MODE,
                "inputs": ["field_signed_margin_as_logit", "box_binary", "point_vote_probability"],
                "probability_epsilon": PROBABILITY_EPSILON,
                "scene_specific_parameter": False,
            },
            "outputs": {
                "rerender_probability": {"path": str(rerender_path), "sha256": rerender_sha},
                "primitive_state": {"path": str(state_path), "sha256": state_sha},
            },
            "authorities": {
                "dataset_manifest": str(source["dataset_manifest"]),
                "dataset_manifest_sha256": source["dataset_manifest_sha256"],
                "protocol_hash": source["protocol_hash"],
                "evaluation_contract": str(source["evaluation_contract"]),
                "evaluation_contract_sha256": source["evaluation_contract_sha256"],
                "evaluation_contract_id": source["evaluation_contract_id"],
            },
            "safety": {
                "target_rgb_opened": False,
                "target_mask_opened": False,
                "target_metric_opened": False,
                "target_metric_used_for_selection": False,
                "target_view_pose_used": True,
                "second_persistent_semantic_field_created": False,
            },
        }
        receipt_path = output_root / "receipts" / f"{scene_id}.json"
        _write_json(receipt_path, receipt)
        relative = rerender_path.relative_to(output_root).as_posix()
        outputs[scene_id] = {frame_id: relative}
        output_hashes[scene_id] = {frame_id: rerender_sha}
        state_assets[scene_id] = {"path": str(state_path), "sha256": state_sha}
        receipts.append(
            {"scene_id": scene_id, "path": str(receipt_path), "sha256": _sha256(receipt_path)}
        )
        del (
            hits,
            gids,
            pids,
            weights,
            model,
            renderer,
            xyz,
            lifted,
            visible_mass,
            primitive_consensus,
            rerender,
            pixel_mass,
        )
        gc.collect()
        torch.cuda.empty_cache()

    manifest = {
        "schema_version": 1,
        "kind": "promptable_nvs_method_v1_two_round_exact_consensus_rerender",
        "candidate_id": CANDIDATE_ID,
        "protocol_hash": box["protocol_hash"],
        "scene_order": scene_ids,
        "prediction_root": ".",
        "predictions": outputs,
        "prediction_sha256": output_hashes,
        "primitive_state": state_assets,
        "receipts": receipts,
        "parent_manifests": {
            "box": {"path": str(box_path), "sha256": args.expected_box_manifest_sha256},
            "point": {"path": str(point_path), "sha256": args.expected_point_manifest_sha256},
        },
        "method": {
            "compositor": COMPOSITOR_MODE,
            "lifting": "exact_W_transpose_with_exact_visibility_normalization",
            "consensus": CONSENSUS_MODE,
            "rerender": "same_exact_W_with_pixel_mass_normalization",
            "probability_epsilon": PROBABILITY_EPSILON,
        },
        "elapsed_seconds": float(time.time() - started),
        "evaluation_performed": False,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_opened": False,
        "all_requested_rerenders_sealed": True,
    }
    manifest_path = output_root / "consensus_manifest.json"
    _write_json(manifest_path, manifest)
    return {**manifest, "consensus_manifest_path": str(manifest_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--signed-field-prompt-manifest", required=True)
    parser.add_argument("--box-manifest", required=True)
    parser.add_argument("--expected-box-manifest-sha256", required=True)
    parser.add_argument("--point-manifest", required=True)
    parser.add_argument("--expected-point-manifest-sha256", required=True)
    parser.add_argument("--scene-id", action="append", dest="scene_ids", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-authority", default=str(DEFAULT_METHOD_AUTHORITY))
    parser.add_argument("--evaluation-contract", default=str(DEFAULT_EVALUATION_CONTRACT))
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = build(build_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "consensus_manifest": report["consensus_manifest_path"],
                "scenes": report["scene_order"],
                "evaluation_performed": False,
                "target_mask_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
