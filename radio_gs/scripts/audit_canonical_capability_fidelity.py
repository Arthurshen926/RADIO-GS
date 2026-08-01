#!/usr/bin/env python3
"""Audit raw, official-DINO, and official-SAM3 held-out fidelity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F

from radio_gs.config import config_to_dict, load_config
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
    render_scalar_support,
    render_view_conditioned_radio,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
    _load_bundle_feature_maps,
    _resolve_extracted_capability_source,
    _validated_feature_bundle,
)
from radio_gs.training.feature_training_utils import SimpleRadioDataset
from radio_gs.utils.immutable_artifacts import sha256_file


def _sha256_file(path: str | Path) -> str:
    return sha256_file(path)


def _sha256_tensor_rows(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _resolved_config_sha256(config) -> str:
    payload = json.dumps(
        config_to_dict(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _dataset(
    config,
    renderer,
    frame_ids: list[int] | None = None,
    *,
    feature_subdir: str = "backbone",
) -> SimpleRadioDataset:
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
        feature_subdir=feature_subdir,
        split="validation",
        dataset_type=str(getattr(config, "dataset_type", "lerf")),
        frame_ids=frame_ids,
    )


def _capability_fidelity_maps(
    predicted_raw: torch.Tensor,
    teacher_raw: torch.Tensor,
    dino: torch.nn.Module,
    sam3: torch.nn.Module,
    *,
    official_targets: dict[str, torch.Tensor] | None = None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if predicted_raw.shape != teacher_raw.shape or predicted_raw.ndim != 3:
        raise ValueError("raw prediction/teacher must be matching [C,H,W]")
    predicted = {
        "dino_v3": project_feature_map_with_adaptor(
            predicted_raw[None], dino
        )[0],
        "sam3": project_feature_map_with_adaptor(
            predicted_raw[None], sam3
        )[0],
    }
    if official_targets is None:
        targets = {
            "dino_v3": project_feature_map_with_adaptor(
                teacher_raw[None], dino
            )[0],
            "sam3": project_feature_map_with_adaptor(
                teacher_raw[None], sam3
            )[0],
        }
    else:
        if set(official_targets) != {"dino_v3", "sam3"}:
            raise ValueError("official targets must contain DINOv3 and SAM3")
        targets = {
            name: F.normalize(
                torch.as_tensor(value, device=predicted_raw.device).float(),
                dim=0,
                eps=1e-8,
            )
            for name, value in official_targets.items()
        }
    for name in predicted:
        if predicted[name].shape != targets[name].shape:
            raise ValueError(
                f"official {name} teacher/prediction shape mismatch: "
                f"{tuple(targets[name].shape)} vs {tuple(predicted[name].shape)}"
            )
    return {
        "raw_radio": (predicted_raw, teacher_raw),
        "official_dino_v3": (predicted["dino_v3"], targets["dino_v3"]),
        "official_sam3": (predicted["sam3"], targets["sam3"]),
    }


def audit(args: argparse.Namespace) -> dict:
    if not 0.0 <= float(args.alpha_threshold) <= 1.0:
        raise ValueError("alpha-threshold must lie in [0,1]")
    if float(args.support_eps) < 0.0:
        raise ValueError("support-eps cannot be negative")
    if not 0.0 < float(args.boundary_quantile) < 0.5:
        raise ValueError("boundary-quantile must lie in (0,0.5)")
    expected_hashes = {
        "config": str(getattr(args, "expected_config_sha256", "")),
        "geometry": str(
            getattr(args, "expected_geometry_checkpoint_sha256", "")
        ),
        "field": str(getattr(args, "expected_field_checkpoint_sha256", "")),
        "radio": str(getattr(args, "expected_radio_checkpoint_sha256", "")),
    }
    actual_paths = {
        "config": args.config,
        "geometry": args.geometry_checkpoint,
        "field": args.field_checkpoint,
        "radio": args.radio_checkpoint,
    }
    for name, path in actual_paths.items():
        expected = expected_hashes[name]
        if len(expected) != 64 or _sha256_file(path) != expected:
            raise ValueError(f"{name} artifact differs from caller authority")
    expected_feature_bundle_sha256 = str(
        getattr(args, "expected_feature_output_bundle_sha256", "")
    )
    if len(expected_feature_bundle_sha256) != 64:
        raise ValueError("audit requires caller-trusted feature output bundle SHA-256")
    device = torch.device(args.device)
    config = load_config(args.config)
    model, _codec, renderer, _sharpener, _refiner, _cfg, _hybrid = load_render_pipeline(
        args.config,
        args.geometry_checkpoint,
        device,
        strict_checkpoint_contract=True,
        load_ply_rgb_features=False,
        expected_checkpoint_sha256=expected_hashes["geometry"],
    )
    geometry_hash = _sha256_tensor_rows(model.get_xyz())
    field, field_payload = load_canonical_field_checkpoint(
        args.field_checkpoint,
        map_location="cpu",
        expected_sha256=expected_hashes["field"],
    )
    if (
        field_payload.get("feature_output_bundle_sha256")
        != expected_feature_bundle_sha256
    ):
        raise ValueError("canonical field belongs to another feature output bundle")
    if str(field_payload.get("geometry_fingerprint", {}).get("xyz_sha256", "")) != geometry_hash:
        raise ValueError("canonical field/geometry row fingerprint mismatch")
    reliability = torch.as_tensor(field_payload.get("reliability")).float()
    if (
        reliability.ndim != 2
        or reliability.shape[0] != model.get_xyz().shape[0]
        or reliability.shape[1] == 0
    ):
        raise ValueError(
            "canonical field checkpoint lacks row-aligned MPR reliability"
        )
    row_supported = (reliability > 0).all(dim=-1).to(device)
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
    capability_map_source = str(
        getattr(args, "capability_map_source", "project_raw")
    )
    radio_checkpoint = Path(args.radio_checkpoint)
    radio_checkpoint_sha256 = expected_hashes["radio"]
    feature_dir = Path(str(getattr(config, "feature_dir", "")))
    (
        _feature_manifest,
        feature_bundle_validation,
        feature_tensor_records,
    ) = _validated_feature_bundle(
        feature_dir,
        expected_output_bundle_sha256=expected_feature_bundle_sha256,
    )
    feature_size = (
        int(getattr(config, "feature_height", renderer.image_height)),
        int(getattr(config, "feature_width", renderer.image_width)),
    )
    raw_teacher_maps = _load_bundle_feature_maps(
        feature_dir=feature_dir,
        selected_frame_indices=frames,
        subdir="backbone",
        expected_dim=1280,
        feature_size=feature_size,
        tensor_records=feature_tensor_records,
        normalize=False,
        output_dtype=torch.float32,
    )
    raw_teacher_by_frame = {
        frame: raw_teacher_maps[index]
        for index, frame in enumerate(frames)
    }
    capability_teacher_by_frame: dict[str, dict[int, torch.Tensor]] = {}
    capability_source_provenance: dict[str, dict[str, object]] = {}
    if capability_map_source == "official_extracted":
        for name in ("dino_v3", "sam3"):
            source = _resolve_extracted_capability_source(
                feature_dir,
                name,
                expected_radio_checkpoint_sha256=(
                    radio_checkpoint_sha256
                ),
                expected_scene=str(getattr(config, "scene", "")),
                expected_image_dir=(
                    Path(str(getattr(config, "scene_root", "")))
                    / "color"
                ),
                expected_frame_indices=[
                    int(value) for value in dataset.frame_indices
                ],
                expected_output_bundle_sha256=(
                    expected_feature_bundle_sha256
                ),
                include_tensor_records=True,
            )
            maps = _load_bundle_feature_maps(
                feature_dir=feature_dir,
                selected_frame_indices=frames,
                subdir=str(source["subdir"]),
                expected_dim=int(source["output_dim"]),
                feature_size=feature_size,
                tensor_records=dict(source["tensor_records"]),
                normalize=True,
                output_dtype=torch.float16,
            )
            capability_teacher_by_frame[name] = {
                frame: maps[index]
                for index, frame in enumerate(frames)
            }
            source.pop("tensor_records", None)
            capability_source_provenance[name] = source
    elif capability_map_source != "project_raw":
        raise ValueError(
            "capability_map_source must be project_raw or official_extracted"
        )

    dino = load_radio_adaptor_from_checkpoint(
        radio_checkpoint,
        "dino_v3",
        kind="feature_projection",
        expected_sha256=radio_checkpoint_sha256,
    ).to(device).eval()
    sam3 = load_radio_adaptor_from_checkpoint(
        radio_checkpoint,
        "sam3",
        kind="feature_projection",
        expected_sha256=radio_checkpoint_sha256,
    ).to(device).eval()
    for module in (dino, sam3):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    spaces = ("raw_radio", "official_dino_v3", "official_sam3")
    cosine_values: dict[str, list[torch.Tensor]] = {space: [] for space in spaces}
    predicted_affinity: dict[str, list[torch.Tensor]] = {space: [] for space in spaces}
    target_affinity: dict[str, list[torch.Tensor]] = {space: [] for space in spaces}
    per_frame: list[dict] = []
    supported_visible_pixels = 0
    total_visible_pixels = 0
    with torch.inference_mode():
        for frame in frames:
            dataset_index = frame_to_index[frame]
            pose = torch.from_numpy(dataset.poses_w2c[dataset_index]).float().to(device)
            teacher_raw_cpu = raw_teacher_by_frame[frame]
            if boundary_residual is not None:
                rendered = render_boundary_conditioned_radio(
                    renderer,
                    model,
                    field,
                    boundary_residual,
                    pose,
                    feature_height=teacher_raw_cpu.shape[1],
                    feature_width=teacher_raw_cpu.shape[2],
                )
            elif residual is None:
                rendered = render_canonical_radio(
                    renderer,
                    model,
                    field,
                    pose,
                    feature_height=teacher_raw_cpu.shape[1],
                    feature_width=teacher_raw_cpu.shape[2],
                )
            else:
                rendered = render_view_conditioned_radio(
                    renderer,
                    model,
                    field,
                    residual,
                    pose,
                    feature_height=teacher_raw_cpu.shape[1],
                    feature_width=teacher_raw_cpu.shape[2],
                )
            predicted_raw = rendered["feature_map"].float()
            teacher_raw = teacher_raw_cpu.to(device).float()
            valid = rendered["alpha_map"] >= float(args.alpha_threshold)
            support_render = render_scalar_support(
                renderer,
                model,
                row_supported.float(),
                pose,
                feature_height=teacher_raw.shape[1],
                feature_width=teacher_raw.shape[2],
            )
            if not torch.allclose(
                support_render["alpha_map"],
                rendered["alpha_map"],
                atol=2e-5,
                rtol=2e-5,
            ):
                raise RuntimeError(
                    "support rendering changed the fixed geometry alpha map"
                )
            supported = valid & (
                support_render["feature_map"][0] > float(args.support_eps)
            )
            frame_supported = int(supported.sum())
            frame_visible = int(valid.sum())
            supported_visible_pixels += frame_supported
            total_visible_pixels += frame_visible
            official_targets = None
            if capability_teacher_by_frame:
                official_targets = {
                    name: capability_teacher_by_frame[name][frame].to(device)
                    for name in ("dino_v3", "sam3")
                }
            maps = _capability_fidelity_maps(
                predicted_raw,
                teacher_raw,
                dino,
                sam3,
                official_targets=official_targets,
            )
            frame_report: dict[str, dict] = {}
            for space, (prediction, target) in maps.items():
                cosine = dense_cosine_values(prediction, target, valid)
                pred_relation = local_affinity_pairs(prediction, valid)
                target_relation = local_affinity_pairs(target, valid)
                cosine_values[space].append(cosine.cpu())
                predicted_affinity[space].append(pred_relation.cpu())
                target_affinity[space].append(target_relation.cpu())
                frame_report[space] = dense_fidelity_summary(cosine)
            per_frame.append(
                {
                    "frame_id": frame,
                    "support_fraction_on_visible": (
                        frame_supported / max(frame_visible, 1)
                    ),
                    "spaces": frame_report,
                }
            )
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
    aggregate["support_fraction_on_visible"] = (
        supported_visible_pixels / max(total_visible_pixels, 1)
    )
    aggregate["supported_visible_pixels"] = supported_visible_pixels
    aggregate["total_visible_pixels"] = total_visible_pixels
    report = {
        "schema_version": 1,
        "audit": "canonical_capability_fidelity_v1",
        "protocol": {
            "held_out_from_mpr": True,
            "frame_ids": frames,
            "raw_teacher_metric_target_only": True,
            "official_adaptors_frozen": True,
            "capability_map_source": capability_map_source,
            "capability_teacher_projection_order": (
                "official_runtime_adaptor_then_registration_resample"
                if capability_teacher_by_frame
                else "registration_resample_then_frozen_adaptor_proxy"
            ),
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "alpha_threshold": float(args.alpha_threshold),
            "support_eps": float(args.support_eps),
            "boundary_quantile": float(args.boundary_quantile),
            "residual_mode": (
                "view"
                if residual is not None
                else "boundary"
                if boundary_residual is not None
                else "none"
            ),
            "screen_residual_only": boundary_residual is not None,
            "primitive_queries_unchanged": boundary_residual is not None,
            "boundary_definition": (
                "lowest/highest teacher-adaptor local-affinity quantiles; no task labels"
            ),
        },
        "artifacts": {
            "config": str(Path(args.config).resolve()),
            "config_sha256": _sha256_file(args.config),
            "resolved_config_sha256": _resolved_config_sha256(config),
            "geometry_checkpoint": str(
                Path(args.geometry_checkpoint).resolve()
            ),
            "geometry_checkpoint_sha256": _sha256_file(
                args.geometry_checkpoint
            ),
            "field_checkpoint": str(Path(args.field_checkpoint).resolve()),
            "field_checkpoint_sha256": _sha256_file(
                args.field_checkpoint
            ),
            "view_residual_checkpoint": (
                str(Path(args.view_residual_checkpoint).resolve()) if residual is not None else ""
            ),
            "boundary_residual_checkpoint": (
                str(Path(args.boundary_residual_checkpoint).resolve())
                if boundary_residual is not None else ""
            ),
            "radio_checkpoint": str(radio_checkpoint.resolve()),
            "radio_checkpoint_sha256": _sha256_file(radio_checkpoint),
            "feature_frame_manifest_sha256": feature_bundle_validation[
                "manifest_sha256"
            ],
            "feature_output_bundle_sha256": expected_feature_bundle_sha256,
            "geometry_xyz_sha256": geometry_hash,
            "official_capability_sources": capability_source_provenance,
        },
        "aggregate": aggregate,
        "per_frame": per_frame,
    }
    output = Path(args.output)
    for name, path in actual_paths.items():
        if _sha256_file(path) != expected_hashes[name]:
            raise RuntimeError(f"{name} artifact changed during fidelity audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(report, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--expected-geometry-checkpoint-sha256", required=True)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--expected-field-checkpoint-sha256", required=True)
    parser.add_argument("--view-residual-checkpoint", default="")
    parser.add_argument("--boundary-residual-checkpoint", default="")
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--expected-radio-checkpoint-sha256", required=True)
    parser.add_argument(
        "--expected-feature-output-bundle-sha256",
        required=True,
    )
    parser.add_argument(
        "--capability-map-source",
        choices=["project_raw", "official_extracted"],
        default="project_raw",
        help=(
            "Use the legacy adaptor-after-resampling proxy or the native "
            "official C-RADIO adaptor maps recorded by feature extraction."
        ),
    )
    parser.add_argument("--frame-ids", required=True)
    parser.add_argument("--include-frame-ids", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha-threshold", type=float, default=0.02)
    parser.add_argument(
        "--support-eps",
        type=float,
        default=1e-6,
        help="Minimum valid-MPR contribution mass for a visible pixel.",
    )
    parser.add_argument("--boundary-quantile", type=float, default=0.2)
    args = parser.parse_args()
    report = audit(args)
    print(json.dumps({"output": str(Path(args.output).resolve()), "aggregate": report["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
