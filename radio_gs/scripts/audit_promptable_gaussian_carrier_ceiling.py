#!/usr/bin/env python3
"""Audit one shared scalar membership on frozen NVOS/SPIn Gaussian carriers.

This script intentionally opens evaluation masks and is diagnostic-only.  It
reuses the frozen promptable-NVS cameras, renderer checkpoint, exact
front-to-back contributions, and current continuous-score resize path.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch

from radio_gs.config import load_config
from radio_gs.evaluation.gaussian_carrier_ceiling import (
    binary_membership_entropy,
    weighted_carrier_mixing_summary,
)
from radio_gs.evaluation.promptable_segmentation import load_ground_truth_mask
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.rendering.camera_clearance import (
    CAMERA_PLANE_CLEARANCE_CONTRACT,
    camera_plane_clearance_confidence,
)
from radio_gs.scripts.audit_gaussian_carrier_ceiling import (
    _boundary_band,
    _grouped_csr,
    _optimize_memberships,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.scripts.eval_nvos_gaussian_first import (
    _resize_nvos_score_for_evaluation,
)
from radio_gs.scripts.render_promptable_nvs_features import resolve_protocol_views
from radio_gs.utils.immutable_artifacts import (
    file_record,
    sha256_file,
    write_frozen_json,
)


AUDIT_CONTRACT = "promptable_nvs_frozen_gaussian_scalar_membership_ceiling_v1"


def _scene_record(manifest: Mapping[str, Any], scene_id: str) -> dict[str, Any]:
    matches = [row for row in manifest.get("scenes", []) if row.get("scene_id") == scene_id]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest scene {scene_id!r}")
    return dict(matches[0])


def _frame_record(scene: Mapping[str, Any], frame_id: str) -> dict[str, Any]:
    raw = scene.get("frames", [])
    rows = raw.values() if isinstance(raw, Mapping) else raw
    matches = [row for row in rows if str(row.get("frame_id")) == str(frame_id)]
    if len(matches) != 1:
        raise ValueError(f"expected one frame record {frame_id!r}")
    return dict(matches[0])


def _view_record(views: list[dict], frame_id: str) -> dict:
    matches = [row for row in views if str(row.get("frame_id")) == str(frame_id)]
    if len(matches) != 1:
        raise ValueError(f"expected one resolved view {frame_id!r}")
    return matches[0]


def _rows_sha256(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load_and_validate_mask(frame: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(frame.get("ground_truth") or frame.get("gt_mask_path"))).resolve()
    declared = str(frame.get("ground_truth_sha256") or "")
    if declared and sha256_file(path) != declared:
        raise ValueError(f"ground-truth SHA-256 differs: {path}")
    return load_ground_truth_mask(path).astype(bool)


def _native_target(mask: np.ndarray, height: int, width: int) -> torch.Tensor:
    value = cv2.resize(
        np.asarray(mask, dtype=np.uint8),
        (int(width), int(height)),
        interpolation=cv2.INTER_NEAREST,
    )
    return torch.from_numpy(value.reshape(-1)).bool()


def _evaluate_scores(
    scores: np.ndarray,
    target: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    resized = _resize_nvos_score_for_evaluation(
        np.asarray(scores, dtype=np.float32),
        tuple(map(int, target.shape)),
        registered_forward_unary="none",
    )
    truth = np.asarray(target).astype(bool)
    fixed = resized >= 0.5
    union = np.logical_or(fixed, truth).sum()
    fixed_iou = float(np.logical_and(fixed, truth).sum() / union) if union else 1.0
    soft_intersection = float((resized * truth).sum())
    soft_union = float(resized.sum() + truth.sum() - soft_intersection)
    soft_iou = soft_intersection / soft_union if soft_union else 1.0
    curve = []
    for threshold in thresholds:
        prediction = resized >= float(threshold)
        union = np.logical_or(prediction, truth).sum()
        curve.append(
            float(np.logical_and(prediction, truth).sum() / union) if union else 1.0
        )
    return fixed_iou, soft_iou, np.asarray(curve, dtype=np.float64)


def _finalize_method(rows: list[dict[str, Any]], thresholds: np.ndarray) -> dict[str, Any]:
    curves = np.stack([row.pop("curve") for row in rows], axis=0)
    mean_curve = curves.mean(axis=0)
    best = int(mean_curve.argmax())
    return {
        "frame_count": len(rows),
        "fixed_0p5_miou": float(np.mean([row["fixed_0p5_iou"] for row in rows])),
        "soft_miou": float(np.mean([row["soft_iou"] for row in rows])),
        "global_oracle_threshold_miou": float(mean_curve[best]),
        "global_oracle_threshold": float(thresholds[best]),
        "per_frame_oracle_threshold_miou": float(curves.max(axis=1).mean()),
        "threshold_curve": [
            {"threshold": float(threshold), "miou": float(mean_curve[index])}
            for index, threshold in enumerate(thresholds)
        ],
        "frames": rows,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if not bool(args.allow_benchmark_mask_oracle):
        raise ValueError("pass --allow-benchmark-mask-oracle for this diagnostic")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("exact promptable carrier audit requires CUDA")
    torch.manual_seed(int(args.seed))
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene = _scene_record(manifest, args.scene_id)
    base_scene_id = str(scene.get("base_scene_id") or args.scene_id)
    queue_scene = Path(args.queue_root).resolve() / "scenes" / args.scene_id
    if not queue_scene.is_dir():
        queue_scene = Path(args.queue_root).resolve() / "scenes" / base_scene_id
    config_path = (queue_scene / "gaussfm_main_track.yaml").resolve()
    checkpoint_path = (queue_scene / "feature_field" / "checkpoints" / "best.pth").resolve()
    camera_map_path = (queue_scene / "rgb_to_colmap_camera_mapping.json").resolve()
    config = load_config(str(config_path))
    camera_mapping = json.loads(camera_map_path.read_text(encoding="utf-8"))
    views = resolve_protocol_views(
        manifest,
        scene_id=args.scene_id,
        scene_root=Path(str(config.scene_root)).resolve(),
        camera_mapping=camera_mapping,
    )
    evaluation_frame_ids = [str(value) for value in scene.get("evaluation_frame_ids", [])]
    if not evaluation_frame_ids:
        raise RuntimeError("scene has no evaluation frames")
    prompt_type = str(scene.get("prompt", {}).get("type", ""))
    carrier_frame_ids = list(evaluation_frame_ids)
    if prompt_type == "reference_binary_mask":
        prompt_ids = [str(value) for value in scene.get("prompt_frame_ids", [])]
        carrier_frame_ids = list(dict.fromkeys([*prompt_ids, *carrier_frame_ids]))

    model, codec, renderer, sharpener, refiner, _field_config, _hybrid = load_render_pipeline(
        str(config_path),
        str(checkpoint_path),
        device,
        strict_checkpoint_contract=True,
        load_ply_rgb_features=False,
    )
    del codec, sharpener, refiner
    gc.collect()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    resolution_authority_record = None
    if str(args.resolution_authority).strip():
        resolution_authority_path = Path(args.resolution_authority).resolve()
        resolution_payload = json.loads(
            resolution_authority_path.read_text(encoding="utf-8")
        )
        authority = resolution_payload.get("authority")
        if (
            not isinstance(authority, Mapping)
            or str(authority.get("scene_id")) != str(args.scene_id)
            or int(authority.get("height", 0)) <= 0
            or int(authority.get("width", 0)) <= 0
        ):
            raise ValueError("resolution authority does not bind this scene and shape")
        height = int(authority["height"])
        width = int(authority["width"])
        resolution_authority_record = file_record(resolution_authority_path)
    else:
        height = int(renderer.image_height)
        width = int(renderer.image_width)
    num_pixels = height * width
    num_gaussians = int(model.get_xyz().shape[0])
    foreground_mass = torch.zeros(num_gaussians, 1, device=device)
    total_mass = torch.zeros_like(foreground_mass)
    boundary_mass = torch.zeros_like(foreground_mass)
    sample_columns: list[torch.Tensor] = []
    sample_values: list[torch.Tensor] = []
    sample_counts: list[torch.Tensor] = []
    sample_targets: list[torch.Tensor] = []
    stage1_frames: list[dict[str, Any]] = []

    print(
        f"[promptable-carrier] stage1 scene={args.scene_id} rows={num_gaussians} "
        f"views={len(carrier_frame_ids)} resolution={width}x{height}",
        flush=True,
    )
    with torch.inference_mode():
        for frame_position, frame_id in enumerate(carrier_frame_ids):
            frame = _frame_record(scene, frame_id)
            mask = _load_and_validate_mask(frame)
            native_cpu = _native_target(mask, height, width)
            boundary_native = _boundary_band(
                native_cpu.reshape(height, width).numpy(), int(args.boundary_radius)
            )
            view = _view_record(views, frame_id)
            pose = torch.from_numpy(view["w2c"].copy()).float().to(device)
            clearance = None
            if float(args.camera_clearance_sigma) > 0:
                clearance = camera_plane_clearance_confidence(
                    model.get_xyz(),
                    model.get_rotation(),
                    model.get_scaling(),
                    pose,
                    near_plane=float(renderer.near_plane),
                    support_sigma=float(args.camera_clearance_sigma),
                )
            hits = rasterize_single_view_contributions(
                model,
                renderer,
                pose,
                height=height,
                width=width,
                opacity_scale=(clearance.confidence if clearance is not None else None),
            )
            operator = _grouped_csr(
                hits["gaussian_ids"],
                hits["pixel_ids"],
                hits["weights"],
                num_pixels=num_pixels,
                num_gaussians=num_gaussians,
            )
            target = native_cpu.to(device=device, dtype=torch.float32)[:, None]
            boundary = torch.from_numpy(boundary_native.reshape(-1)).to(
                device=device, dtype=torch.float32
            )[:, None]
            foreground_mass += torch.sparse.mm(operator.transpose(0, 1), target)
            total_mass += torch.sparse.mm(
                operator.transpose(0, 1),
                torch.ones(num_pixels, 1, device=device),
            )
            boundary_mass += torch.sparse.mm(operator.transpose(0, 1), boundary)

            count = 0
            sampled_hits = 0
            if int(args.steps) > 0:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(args.seed) + frame_position)
                count = min(int(args.random_pixel_cap), num_pixels)
                selected_cpu = torch.randperm(num_pixels, generator=generator)[
                    :count
                ].sort().values
                selected = selected_cpu.to(device)
                remap = torch.full((num_pixels,), -1, dtype=torch.long, device=device)
                remap[selected] = torch.arange(selected.numel(), device=device)
                keep = remap[hits["pixel_ids"]] >= 0
                local_pixels = remap[hits["pixel_ids"][keep]]
                alpha = hits["accumulated_alpha"].reshape(-1)
                normalized = hits["weights"][keep] / alpha[
                    hits["pixel_ids"][keep]
                ].clamp_min(float(args.alpha_eps))
                sample_columns.append(hits["gaussian_ids"][keep])
                sample_values.append(normalized)
                sample_counts.append(torch.bincount(local_pixels, minlength=count))
                sample_targets.append(native_cpu[selected_cpu, None].to(device))
                sampled_hits = int(keep.sum())
            stage1_frames.append(
                {
                    "frame_id": frame_id,
                    "exact_hits": int(hits["weights"].numel()),
                    "sampled_pixels": int(count),
                    "sampled_hits": sampled_hits,
                    "camera_clearance_rejected_rows": (
                        int((clearance.confidence == 0).sum())
                        if clearance is not None
                        else 0
                    ),
                }
            )
            print(
                f"[promptable-carrier] frame={frame_id} hits={hits['weights'].numel()} "
                f"sample_hits={sampled_hits}",
                flush=True,
            )
            del hits, operator, target, boundary
            torch.cuda.empty_cache()

    initial_probability = torch.where(
        total_mass > float(args.mass_eps),
        foreground_mass / total_mass.clamp_min(float(args.mass_eps)),
        torch.zeros_like(total_mass),
    ).clamp(0.0, 1.0)
    if int(args.steps) > 0:
        counts = torch.cat(sample_counts)
        crow = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)]
        )
        columns = torch.cat(sample_columns)
        values = torch.cat(sample_values).float()
        sample_operator = torch.sparse_csr_tensor(
            crow,
            columns,
            values,
            size=(int(counts.numel()), num_gaussians),
            device=device,
            dtype=torch.float32,
        )
        sample_target = torch.cat(sample_targets).bool()
        sample_available = torch.ones_like(sample_target)
        del counts, crow, columns, values
        optimized_probability, history = _optimize_memberships(
            sample_operator,
            initial_probability,
            sample_target,
            sample_available,
            steps=int(args.steps),
            learning_rate=float(args.learning_rate),
            dice_weight=float(args.dice_weight),
            loss_mode="uniform_bce_dice",
            log_every=int(args.log_every),
        )
        del sample_operator, sample_target, sample_available
    else:
        optimized_probability = initial_probability.clone()
        history = []
    del sample_counts, sample_columns, sample_values, sample_targets
    torch.cuda.empty_cache()

    threshold_tensor = torch.arange(
        float(args.threshold_min),
        float(args.threshold_max) + 0.5 * float(args.threshold_step),
        float(args.threshold_step),
    )
    thresholds = threshold_tensor.numpy().astype(np.float32)
    probabilities = {
        "exact_adjoint_ratio": initial_probability,
        "optimized_scalar_membership": optimized_probability,
    }
    method_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in probabilities}
    print("[promptable-carrier] stage2 frozen evaluation views", flush=True)
    with torch.inference_mode():
        for frame_id in evaluation_frame_ids:
            frame = _frame_record(scene, frame_id)
            target_mask = _load_and_validate_mask(frame)
            view = _view_record(views, frame_id)
            pose = torch.from_numpy(view["w2c"].copy()).float().to(device)
            clearance = None
            if float(args.camera_clearance_sigma) > 0:
                clearance = camera_plane_clearance_confidence(
                    model.get_xyz(),
                    model.get_rotation(),
                    model.get_scaling(),
                    pose,
                    near_plane=float(renderer.near_plane),
                    support_sigma=float(args.camera_clearance_sigma),
                )
            hits = rasterize_single_view_contributions(
                model,
                renderer,
                pose,
                height=height,
                width=width,
                opacity_scale=(clearance.confidence if clearance is not None else None),
            )
            alpha = hits["accumulated_alpha"].reshape(-1)
            normalized = hits["weights"] / alpha[hits["pixel_ids"]].clamp_min(
                float(args.alpha_eps)
            )
            operator = _grouped_csr(
                hits["gaussian_ids"],
                hits["pixel_ids"],
                normalized,
                num_pixels=num_pixels,
                num_gaussians=num_gaussians,
            )
            status = []
            for name, probability in probabilities.items():
                score = torch.sparse.mm(operator, probability).reshape(height, width)
                score_np = score.detach().cpu().numpy().astype(np.float32)
                fixed, soft, curve = _evaluate_scores(score_np, target_mask, thresholds)
                method_rows[name].append(
                    {
                        "frame_id": frame_id,
                        "fixed_0p5_iou": fixed,
                        "soft_iou": soft,
                        "curve": curve,
                        "gt_pixels": int(target_mask.sum()),
                    }
                )
                status.append(f"{name}={fixed:.4f}")
            print(f"[promptable-carrier] evaluated {frame_id} " + " ".join(status), flush=True)
            del hits, alpha, normalized, operator, score
            torch.cuda.empty_cache()

    methods = {
        name: _finalize_method(rows, thresholds) for name, rows in method_rows.items()
    }
    feasible = {
        name: float(values["global_oracle_threshold_miou"])
        for name, values in methods.items()
    }
    best_method = max(feasible, key=feasible.get)
    membership, entropy = binary_membership_entropy(
        foreground_mass[:, 0].cpu(), total_mass[:, 0].cpu(), eps=float(args.mass_eps)
    )
    mixing = weighted_carrier_mixing_summary(
        foreground_mass[:, 0].cpu(),
        total_mass[:, 0].cpu(),
        ambiguity_low=float(args.ambiguity_low),
        ambiguity_high=float(args.ambiguity_high),
        eps=float(args.mass_eps),
    )
    observed = total_mass[:, 0].cpu() > float(args.mass_eps)
    ambiguous = observed & (membership > float(args.ambiguity_low)) & (
        membership < float(args.ambiguity_high)
    )
    edge = boundary_mass[:, 0].cpu()
    mixing.update(
        {
            "boundary_mass_carried_by_ambiguous_rows": float(
                edge[ambiguous].sum() / edge.sum().clamp_min(float(args.mass_eps))
            ),
            "observed_entropy_p90": float(torch.quantile(entropy[observed], 0.9)),
        }
    )
    report = {
        "schema_version": 1,
        "audit": AUDIT_CONTRACT,
        "protocol": {
            "diagnostic_only": True,
            "valid_benchmark_method": False,
            "benchmark_masks_opened": True,
            "text_queries_opened": False,
            "geometry_frozen": True,
            "one_shared_scalar_membership_per_gaussian": True,
            "full_reference_included_when_available": prompt_type == "reference_binary_mask",
            "score_resize": "frozen_registered_forward_unary_none_continuous_resize",
            "camera_clearance": (
                {
                    "contract": CAMERA_PLANE_CLEARANCE_CONTRACT,
                    "support_sigma": float(args.camera_clearance_sigma),
                    "query_independent": True,
                }
                if float(args.camera_clearance_sigma) > 0
                else None
            ),
        },
        "benchmark": str(manifest.get("benchmark", "")),
        "scene_id": args.scene_id,
        "prompt_type": prompt_type,
        "carrier_frame_ids": carrier_frame_ids,
        "evaluation_frame_ids": evaluation_frame_ids,
        "render_resolution": [height, width],
        "render_resolution_source": (
            "explicit_responsibility_authority"
            if resolution_authority_record is not None
            else "frozen_renderer"
        ),
        "num_gaussians": num_gaussians,
        "geometry_xyz_sha256": _rows_sha256(model.get_xyz()),
        "artifacts": {
            "manifest": file_record(manifest_path),
            "config": file_record(config_path),
            "geometry_checkpoint": file_record(checkpoint_path),
            "camera_mapping": file_record(camera_map_path),
            "preregistration": file_record(Path(args.preregistration).resolve()),
            "resolution_authority": resolution_authority_record,
        },
        "optimization": {
            "steps": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "dice_weight": float(args.dice_weight),
            "uniform_pixels_per_view": int(args.random_pixel_cap),
            "history": history,
        },
        "methods": methods,
        "best_feasible_lower_bound": {
            "method": best_method,
            "global_oracle_threshold_miou": feasible[best_method],
            "semantics": "lower bound on the unknown shared scalar-membership optimum",
        },
        "mixing": mixing,
        "stage1_frames": stage1_frames,
        "runtime": {
            "device": str(device),
            "maximum_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
    }
    write_frozen_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--resolution-authority", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-benchmark-mask-oracle", action="store_true")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--random-pixel-cap", type=int, default=196608)
    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument("--ambiguity-low", type=float, default=0.1)
    parser.add_argument("--ambiguity-high", type=float, default=0.9)
    parser.add_argument("--alpha-eps", type=float, default=1e-8)
    parser.add_argument("--mass-eps", type=float, default=1e-10)
    parser.add_argument("--camera-clearance-sigma", type=float, default=0.0)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    report = audit(args)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "scene_id": args.scene_id,
                "methods": {
                    name: {
                        key: values[key]
                        for key in (
                            "fixed_0p5_miou",
                            "global_oracle_threshold_miou",
                            "global_oracle_threshold",
                        )
                    }
                    for name, values in report["methods"].items()
                },
                "best_feasible_lower_bound": report["best_feasible_lower_bound"],
                "mixing": report["mixing"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
