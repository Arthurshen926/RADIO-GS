#!/usr/bin/env python3
"""Audit raw, official-DINO, and official-SAM3 held-out fidelity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from radio_gs.config import load_config
from radio_gs.evaluation.capability_fidelity import (
    dense_cosine_values,
    dense_fidelity_summary,
    local_affinity_pairs,
    relation_fidelity_summary,
)
from radio_gs.field import (
    load_boundary_screen_residual_checkpoint,
    load_canonical_field_checkpoint,
    load_view_residual_checkpoint,
)
from radio_gs.models.radio_adaptors import (
    load_radio_adaptor_from_checkpoint,
    project_feature_map_with_adaptor,
)
from radio_gs.rendering.coefficient_renderer import (
    render_boundary_conditioned_radio,
    render_canonical_radio,
    render_view_conditioned_radio,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.training.feature_training_utils import SimpleRadioDataset


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor_rows(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _parse_frame_ids(raw: str) -> list[int]:
    value = str(raw or "").strip()
    path = Path(value)
    if path.is_file():
        value = path.read_text(encoding="utf-8")
    tokens: list[str] = []
    for line in value.splitlines():
        tokens.extend(line.split("#", 1)[0].replace(",", " ").split())
    result = sorted({int(token) for token in tokens})
    if not result:
        raise ValueError("--frame-ids is empty")
    return result


def _dataset(config, renderer, frame_ids: list[int] | None = None) -> SimpleRadioDataset:
    feature_dir = Path(str(getattr(config, "feature_dir", "")))
    raw_pose_file = str(getattr(config, "pose_file", "") or "").strip()
    pose_file = raw_pose_file if raw_pose_file and Path(raw_pose_file).is_file() else None
    raw_pose_dir = str(getattr(config, "pose_dir", "") or "").strip()
    fallback = feature_dir / "poses_w2c"
    pose_dir = (
        raw_pose_dir
        if raw_pose_dir and Path(raw_pose_dir).is_dir()
        else str(fallback) if fallback.is_dir() else None
    )
    return SimpleRadioDataset(
        feature_dir=str(feature_dir),
        pose_file=pose_file,
        pose_dir=pose_dir,
        feature_size=(
            int(getattr(config, "feature_height", renderer.image_height)),
            int(getattr(config, "feature_width", renderer.image_width)),
        ),
        split="validation",
        dataset_type=str(getattr(config, "dataset_type", "lerf")),
        frame_ids=frame_ids,
    )


def audit(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    config = load_config(args.config)
    model, _codec, renderer, _sharpener, _refiner, _cfg, _hybrid = load_render_pipeline(
        args.config,
        args.geometry_checkpoint,
        device,
        strict_checkpoint_contract=True,
        load_ply_rgb_features=False,
    )
    geometry_hash = _sha256_tensor_rows(model.get_xyz())
    field, field_payload = load_canonical_field_checkpoint(
        args.field_checkpoint, map_location="cpu"
    )
    if str(field_payload.get("geometry_fingerprint", {}).get("xyz_sha256", "")) != geometry_hash:
        raise ValueError("canonical field/geometry row fingerprint mismatch")
    field = field.to(device).eval()
    residual = None
    residual_payload = None
    boundary_residual = None
    boundary_payload = None
    if str(args.view_residual_checkpoint).strip() and str(args.boundary_residual_checkpoint).strip():
        raise ValueError("view and boundary residual checkpoints are mutually exclusive")
    if str(args.view_residual_checkpoint).strip():
        residual, residual_payload = load_view_residual_checkpoint(
            args.view_residual_checkpoint, map_location="cpu"
        )
        if str(residual_payload.get("base_field_sha256", "")) != _sha256_file(
            args.field_checkpoint
        ):
            raise ValueError("view residual was trained over another field")
        residual = residual.to(device).eval()
    if str(args.boundary_residual_checkpoint).strip():
        boundary_residual, boundary_payload = load_boundary_screen_residual_checkpoint(
            args.boundary_residual_checkpoint, map_location="cpu"
        )
        boundary_residual = boundary_residual.to(device).eval()

    frames = _parse_frame_ids(args.frame_ids)
    mpr_training = {
        int(value)
        for value in field_payload.get("mpr_cache_metadata", {}).get(
            "selected_frame_indices", []
        )
    }
    overlap = sorted(set(frames).intersection(mpr_training))
    if overlap:
        raise ValueError(f"capability audit frames overlap MPR training: {overlap}")
    included_frames = _parse_frame_ids(args.include_frame_ids) if str(args.include_frame_ids).strip() else []
    dataset = _dataset(config, renderer, included_frames or None)
    frame_to_index = {int(frame): index for index, frame in enumerate(dataset.frame_indices)}
    missing = sorted(set(frames) - set(frame_to_index))
    if missing:
        raise ValueError(f"capability audit frames unavailable: {missing}")

    radio_checkpoint = Path(args.radio_checkpoint)
    dino = load_radio_adaptor_from_checkpoint(
        radio_checkpoint, "dino_v3", kind="feature_projection"
    ).to(device).eval()
    sam3 = load_radio_adaptor_from_checkpoint(
        radio_checkpoint, "sam3", kind="feature_projection"
    ).to(device).eval()
    for module in (dino, sam3):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    spaces = ("raw_radio", "official_dino_v3", "official_sam3")
    cosine_values: dict[str, list[torch.Tensor]] = {space: [] for space in spaces}
    predicted_affinity: dict[str, list[torch.Tensor]] = {space: [] for space in spaces}
    target_affinity: dict[str, list[torch.Tensor]] = {space: [] for space in spaces}
    per_frame: list[dict] = []
    with torch.inference_mode():
        for frame in frames:
            sample = dataset[frame_to_index[frame]]
            pose = sample["pose_w2c"].to(device)
            if boundary_residual is not None:
                rendered = render_boundary_conditioned_radio(
                    renderer,
                    model,
                    field,
                    boundary_residual,
                    pose,
                    feature_height=sample["radio_features"].shape[1],
                    feature_width=sample["radio_features"].shape[2],
                )
            elif residual is None:
                rendered = render_canonical_radio(
                    renderer,
                    model,
                    field,
                    pose,
                    feature_height=sample["radio_features"].shape[1],
                    feature_width=sample["radio_features"].shape[2],
                )
            else:
                rendered = render_view_conditioned_radio(
                    renderer,
                    model,
                    field,
                    residual,
                    pose,
                    feature_height=sample["radio_features"].shape[1],
                    feature_width=sample["radio_features"].shape[2],
                )
            predicted_raw = rendered["feature_map"].float()
            teacher_raw = sample["radio_features"].to(device).float()
            valid = rendered["alpha_map"] >= float(args.alpha_threshold)
            combined = torch.stack([predicted_raw, teacher_raw], dim=0)
            dino_combined = project_feature_map_with_adaptor(combined, dino)
            sam_combined = project_feature_map_with_adaptor(combined, sam3)
            maps = {
                "raw_radio": (predicted_raw, teacher_raw),
                "official_dino_v3": (dino_combined[0], dino_combined[1]),
                "official_sam3": (sam_combined[0], sam_combined[1]),
            }
            frame_report: dict[str, dict] = {}
            for space, (prediction, target) in maps.items():
                cosine = dense_cosine_values(prediction, target, valid)
                pred_relation = local_affinity_pairs(prediction, valid)
                target_relation = local_affinity_pairs(target, valid)
                cosine_values[space].append(cosine.cpu())
                predicted_affinity[space].append(pred_relation.cpu())
                target_affinity[space].append(target_relation.cpu())
                frame_report[space] = dense_fidelity_summary(cosine)
            per_frame.append({"frame_id": frame, "spaces": frame_report})
            print(
                f"[capability] frame {frame}: raw={frame_report['raw_radio']['mean_cosine']:.4f} "
                f"dino={frame_report['official_dino_v3']['mean_cosine']:.4f} "
                f"sam3={frame_report['official_sam3']['mean_cosine']:.4f}",
                flush=True,
            )

    aggregate = {}
    for space in spaces:
        cosine = torch.cat(cosine_values[space])
        pred_relation = torch.cat(predicted_affinity[space])
        teacher_relation = torch.cat(target_affinity[space])
        aggregate[space] = {
            **dense_fidelity_summary(cosine),
            "local_relation": relation_fidelity_summary(
                pred_relation,
                teacher_relation,
                boundary_quantile=float(args.boundary_quantile),
            ),
        }
    report = {
        "schema_version": 1,
        "audit": "canonical_capability_fidelity_v1",
        "protocol": {
            "held_out_from_mpr": True,
            "frame_ids": frames,
            "raw_teacher_metric_target_only": True,
            "official_adaptors_frozen": True,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "screen_residual_only": boundary_residual is not None,
            "primitive_queries_unchanged": boundary_residual is not None,
            "boundary_definition": (
                "lowest/highest teacher-adaptor local-affinity quantiles; no task labels"
            ),
        },
        "artifacts": {
            "field_checkpoint": str(Path(args.field_checkpoint).resolve()),
            "view_residual_checkpoint": (
                str(Path(args.view_residual_checkpoint).resolve()) if residual is not None else ""
            ),
            "boundary_residual_checkpoint": (
                str(Path(args.boundary_residual_checkpoint).resolve())
                if boundary_residual is not None else ""
            ),
            "radio_checkpoint": str(radio_checkpoint.resolve()),
            "radio_checkpoint_sha256": _sha256_file(radio_checkpoint),
            "geometry_xyz_sha256": geometry_hash,
        },
        "aggregate": aggregate,
        "per_frame": per_frame,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--view-residual-checkpoint", default="")
    parser.add_argument("--boundary-residual-checkpoint", default="")
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--frame-ids", required=True)
    parser.add_argument("--include-frame-ids", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha-threshold", type=float, default=0.02)
    parser.add_argument("--boundary-quantile", type=float, default=0.2)
    args = parser.parse_args()
    report = audit(args)
    print(json.dumps({"output": str(Path(args.output).resolve()), "aggregate": report["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
