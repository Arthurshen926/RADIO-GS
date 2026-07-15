#!/usr/bin/env python3
"""Select a query-free feature compositor on frozen held-out teacher views.

The audit compares ordinary alpha-normalized blending, contribution
sharpening, top-k surface blending, depth bands, and primitive uncertainty
reweighting.  It never opens task masks, text queries, or category labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.config import load_config
from radio_gs.evaluation.capability_fidelity import (
    dense_cosine_values,
    dense_fidelity_summary,
    local_affinity_pairs,
    relation_fidelity_summary,
    select_query_free_compositor,
)
from radio_gs.evaluation.view_consistency import pearson_spearman
from radio_gs.field import (
    load_canonical_field_checkpoint,
    load_view_residual_checkpoint,
)
from radio_gs.models.radio_adaptors import (
    load_radio_adaptor_from_checkpoint,
    project_feature_map_with_adaptor,
)
from radio_gs.rendering.coefficient_renderer import (
    render_canonical_radio,
    render_view_conditioned_radio,
)
from radio_gs.rendering.contribution_compositor import (
    build_compositing_variants,
    composite_feature_variants,
    rasterize_single_view_contributions,
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


def _parse_values(raw: str, cast) -> tuple:
    value = str(raw or "").strip()
    if not value:
        return ()
    return tuple(cast(token) for token in value.replace(",", " ").split())


def _parse_frame_ids(raw: str) -> list[int]:
    value = str(raw or "").strip()
    path = Path(value)
    if path.is_file():
        value = path.read_text(encoding="utf-8")
    tokens: list[str] = []
    for line in value.splitlines():
        tokens.extend(line.split("#", 1)[0].replace(",", " ").split())
    frames = sorted({int(token) for token in tokens})
    if not frames:
        raise ValueError("--frame-ids is empty")
    return frames


def _dataset(config, renderer) -> SimpleRadioDataset:
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
    )


def _filter_variants(
    variants: dict[str, torch.Tensor], requested: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    if not requested:
        return variants
    required = tuple(dict.fromkeys(("alpha_mean", *requested)))
    missing = sorted(set(required) - set(variants))
    if missing:
        raise ValueError(f"unknown compositing variants: {missing}; available={sorted(variants)}")
    return {name: variants[name] for name in required}


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
    fingerprint = dict(field_payload.get("geometry_fingerprint", {}))
    if (
        str(fingerprint.get("xyz_sha256", "")) != geometry_hash
        or int(fingerprint.get("num_gaussians", -1)) != int(model.get_xyz().shape[0])
    ):
        raise ValueError("canonical field/geometry row fingerprint mismatch")
    field = field.to(device).eval()

    residual = None
    residual_path = str(args.view_residual_checkpoint).strip()
    if residual_path:
        residual, residual_payload = load_view_residual_checkpoint(
            residual_path, map_location="cpu"
        )
        if str(residual_payload.get("base_field_sha256", "")) != _sha256_file(
            args.field_checkpoint
        ):
            raise ValueError("view residual was trained over another canonical field")
        residual = residual.to(device).eval()

    primitive_uncertainty = None
    consistency_path = str(args.consistency_cache).strip()
    if consistency_path:
        uncertainty_payload = torch.load(consistency_path, map_location="cpu")
        if str(uncertainty_payload.get("geometry_xyz_sha256", "")) != geometry_hash:
            raise ValueError("view-consistency cache/geometry row fingerprint mismatch")
        primitive_uncertainty = torch.as_tensor(
            uncertainty_payload[args.uncertainty_key]
        ).float().clamp_min(0).to(device)
        if primitive_uncertainty.shape != (model.get_xyz().shape[0],):
            raise ValueError("primitive uncertainty rows do not match geometry")

    frames = _parse_frame_ids(args.frame_ids)
    mpr_training = {
        int(value)
        for value in field_payload.get("mpr_cache_metadata", {}).get(
            "selected_frame_indices", []
        )
    }
    overlap = sorted(set(frames).intersection(mpr_training))
    if overlap:
        raise ValueError(f"compositing audit frames overlap MPR training: {overlap}")
    dataset = _dataset(config, renderer)
    frame_to_index = {int(frame): index for index, frame in enumerate(dataset.frame_indices)}
    missing = sorted(set(frames) - set(frame_to_index))
    if missing:
        raise ValueError(f"compositing audit frames unavailable: {missing}")

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

    gammas = _parse_values(args.gammas, float)
    topk = _parse_values(args.topk, int)
    uncertainty_strengths = _parse_values(args.uncertainty_strengths, float)
    if primitive_uncertainty is None:
        uncertainty_strengths = ()
    requested_variants = _parse_values(args.only_variants, str)
    spaces = ("raw_radio", "official_dino_v3", "official_sam3")
    cosine_values: dict[str, dict[str, list[torch.Tensor]]] = {}
    predicted_affinity: dict[str, dict[str, list[torch.Tensor]]] = {}
    target_affinity: dict[str, list[torch.Tensor]] = {space: [] for space in spaces}
    support_counts: dict[str, list[int]] = {}
    support_totals: dict[str, list[int]] = {}
    selected_masses: dict[str, list[torch.Tensor]] = {}
    per_frame: list[dict] = []
    mixture_uncertainty_values: list[torch.Tensor] = []
    baseline_error_values: list[torch.Tensor] = []
    variant_names: tuple[str, ...] | None = None

    with torch.inference_mode():
        base_coefficients = field.coefficients().detach()
        for frame in frames:
            sample = dataset[frame_to_index[frame]]
            pose = sample["pose_w2c"].to(device)
            teacher_raw = sample["radio_features"].to(device).float()
            height, width = teacher_raw.shape[1:]
            if residual is None:
                row_coefficients = base_coefficients
                baseline = render_canonical_radio(
                    renderer,
                    model,
                    field,
                    pose,
                    feature_height=height,
                    feature_width=width,
                )
            else:
                row_coefficients = base_coefficients + residual(model.get_xyz(), pose)
                baseline = render_view_conditioned_radio(
                    renderer,
                    model,
                    field,
                    residual,
                    pose,
                    feature_height=height,
                    feature_width=width,
                )
            hits = rasterize_single_view_contributions(
                model, renderer, pose, height=height, width=width
            )
            alpha_error = (
                hits["accumulated_alpha"] - baseline["alpha_map"].float()
            ).abs()
            variants = build_compositing_variants(
                hits["pixel_ids"],
                hits["weights"],
                num_pixels=height * width,
                depths=hits["depths"],
                reference_depth=hits["rendered_depth"],
                gammas=gammas,
                topk=topk,
                depth_tolerance=float(args.depth_tolerance),
                relative_depth_tolerance=float(args.relative_depth_tolerance),
                uncertainty=primitive_uncertainty,
                gaussian_ids=hits["gaussian_ids"],
                uncertainty_strengths=uncertainty_strengths,
            )
            variants = _filter_variants(variants, requested_variants)
            current_names = tuple(variants)
            if variant_names is None:
                variant_names = current_names
                for name in variant_names:
                    cosine_values[name] = {space: [] for space in spaces}
                    predicted_affinity[name] = {space: [] for space in spaces}
                    support_counts[name] = []
                    support_totals[name] = []
                    selected_masses[name] = []
            elif current_names != variant_names:
                raise RuntimeError("compositor variant set changed between frames")

            coefficient_maps, mass_maps = composite_feature_variants(
                row_coefficients,
                hits["gaussian_ids"],
                hits["pixel_ids"],
                variants,
                height=height,
                width=width,
                channel_chunk_size=int(args.channel_chunk_size),
                variant_chunk_size=int(args.variant_chunk_size),
            )
            visible = baseline["alpha_map"].float() >= float(args.alpha_threshold)
            baseline_coefficient_cosine = F.cosine_similarity(
                coefficient_maps["alpha_mean"].permute(1, 2, 0)[visible],
                baseline["coefficient_map"].float().permute(1, 2, 0)[visible],
                dim=-1,
                eps=1e-8,
            )
            if (
                float(alpha_error.max()) > float(args.max_baseline_alpha_error)
                or float(baseline_coefficient_cosine.mean())
                < float(args.min_baseline_coefficient_cosine)
            ):
                raise RuntimeError(
                    "explicit hit compositor failed to reproduce alpha-normalized baseline"
                )

            teacher_dino = project_feature_map_with_adaptor(
                teacher_raw[None], dino
            )[0]
            teacher_sam = project_feature_map_with_adaptor(
                teacher_raw[None], sam3
            )[0]
            teacher_maps = {
                "raw_radio": teacher_raw,
                "official_dino_v3": teacher_dino,
                "official_sam3": teacher_sam,
            }
            for space, target in teacher_maps.items():
                target_affinity[space].append(local_affinity_pairs(target, visible).cpu())

            frame_variants: dict[str, dict] = {}
            for name in variant_names:
                predicted_raw = field.decoder.decode_map(coefficient_maps[name]).float()
                predicted_dino = project_feature_map_with_adaptor(
                    predicted_raw[None], dino
                )[0]
                predicted_sam = project_feature_map_with_adaptor(
                    predicted_raw[None], sam3
                )[0]
                predicted_maps = {
                    "raw_radio": predicted_raw,
                    "official_dino_v3": predicted_dino,
                    "official_sam3": predicted_sam,
                }
                frame_spaces: dict[str, dict] = {}
                for space in spaces:
                    cosine = dense_cosine_values(
                        predicted_maps[space], teacher_maps[space], visible
                    )
                    relation = local_affinity_pairs(predicted_maps[space], visible)
                    cosine_values[name][space].append(cosine.cpu())
                    predicted_affinity[name][space].append(relation.cpu())
                    frame_spaces[space] = dense_fidelity_summary(cosine)
                supported = visible & (mass_maps[name] > float(args.mass_eps))
                support_counts[name].append(int(supported.sum()))
                support_totals[name].append(int(visible.sum()))
                selected_masses[name].append(mass_maps[name][visible].detach().cpu())
                frame_variants[name] = {
                    "support_fraction_on_visible": float(
                        supported.sum() / visible.sum().clamp_min(1)
                    ),
                    "spaces": frame_spaces,
                }

            squared_norm_maps, _ = composite_feature_variants(
                row_coefficients.square().sum(dim=1, keepdim=True),
                hits["gaussian_ids"],
                hits["pixel_ids"],
                {"alpha_mean": variants["alpha_mean"]},
                height=height,
                width=width,
                channel_chunk_size=1,
                variant_chunk_size=1,
            )
            mean_squared_norm = squared_norm_maps["alpha_mean"][0]
            relative_mixture_variance = (
                mean_squared_norm - coefficient_maps["alpha_mean"].square().sum(dim=0)
            ).clamp_min(0) / mean_squared_norm.clamp_min(1e-8)
            baseline_raw = field.decoder.decode_map(
                coefficient_maps["alpha_mean"]
            ).float()
            baseline_error = 1.0 - dense_cosine_values(
                baseline_raw, teacher_raw, visible
            )
            mixture_uncertainty_values.append(relative_mixture_variance[visible].cpu())
            baseline_error_values.append(baseline_error.cpu())
            per_frame.append(
                {
                    "frame_id": int(frame),
                    "visible_pixels": int(visible.sum()),
                    "raster_hits": int(hits["gaussian_ids"].numel()),
                    "baseline_reproduction": {
                        "max_alpha_abs_error": float(alpha_error.max()),
                        "mean_coefficient_cosine": float(
                            baseline_coefficient_cosine.mean()
                        ),
                        "p05_coefficient_cosine": float(
                            torch.quantile(baseline_coefficient_cosine, 0.05)
                        ),
                    },
                    "variants": frame_variants,
                }
            )
            print(
                f"[compositor] frame {frame}: hits={hits['gaussian_ids'].numel()} "
                f"alpha_err={float(alpha_error.max()):.2e} "
                f"baseline={frame_variants['alpha_mean']['spaces']['raw_radio']['mean_cosine']:.4f}",
                flush=True,
            )

    if variant_names is None:
        raise RuntimeError("no compositing frame was evaluated")
    aggregate: dict[str, dict] = {}
    for name in variant_names:
        variant_report: dict[str, dict | float] = {}
        for space in spaces:
            cosine = torch.cat(cosine_values[name][space])
            predicted_relation = torch.cat(predicted_affinity[name][space])
            teacher_relation = torch.cat(target_affinity[space])
            variant_report[space] = {
                **dense_fidelity_summary(cosine),
                "local_relation": relation_fidelity_summary(
                    predicted_relation,
                    teacher_relation,
                    boundary_quantile=float(args.boundary_quantile),
                ),
            }
        total_support = sum(support_counts[name])
        total_visible = sum(support_totals[name])
        all_mass = torch.cat(selected_masses[name])
        variant_report["support_fraction_on_visible"] = total_support / max(
            total_visible, 1
        )
        variant_report["mean_selected_mass_on_visible"] = float(all_mass.mean())
        aggregate[name] = variant_report

    if str(args.frozen_selected_variant).strip():
        selected = str(args.frozen_selected_variant).strip()
        if selected not in aggregate:
            raise ValueError("frozen selected compositor was not evaluated")
        decision = {
            "selected_variant": selected,
            "selection_source": "externally_frozen_before_these_frames",
            "selection_uses_task_labels": False,
            "benchmark_metrics_do_not_select_variant": True,
        }
    else:
        decision = select_query_free_compositor(
            aggregate,
            max_mean_dense_drop=float(args.max_mean_dense_drop),
            max_p05_dense_drop=float(args.max_p05_dense_drop),
            max_unsupported_fraction=float(args.max_unsupported_fraction),
            min_relation_gain=float(args.min_relation_gain),
        )
        decision["selection_source"] = "these_query_free_development_frames"

    report = {
        "schema_version": 1,
        "audit": "query_free_feature_compositing_v1",
        "protocol": {
            "frame_role": args.frame_role,
            "frame_ids": frames,
            "held_out_from_mpr": True,
            "mpr_training_overlap": overlap,
            "raw_teacher_metric_target_only": True,
            "official_adaptors_frozen": True,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "variant_family": (
                "alpha-normalized mean, contribution gamma, contribution top-k, "
                "front/expected depth band, primitive-disagreement reweighting"
            ),
            "same_geometry_visible_pixels_for_all_variants": True,
            "boundary_definition": (
                "lowest/highest teacher-adaptor local-affinity quantiles; no task labels"
            ),
        },
        "artifacts": {
            "config": str(Path(args.config).resolve()),
            "geometry_checkpoint": str(Path(args.geometry_checkpoint).resolve()),
            "geometry_xyz_sha256": geometry_hash,
            "field_checkpoint": str(Path(args.field_checkpoint).resolve()),
            "field_checkpoint_sha256": _sha256_file(args.field_checkpoint),
            "view_residual_checkpoint": (
                str(Path(residual_path).resolve()) if residual_path else ""
            ),
            "view_residual_checkpoint_sha256": (
                _sha256_file(residual_path) if residual_path else ""
            ),
            "consistency_cache": (
                str(Path(consistency_path).resolve()) if consistency_path else ""
            ),
            "uncertainty_key": args.uncertainty_key,
            "radio_checkpoint": str(radio_checkpoint.resolve()),
            "radio_checkpoint_sha256": _sha256_file(radio_checkpoint),
        },
        "variant_parameters": {
            "gammas": list(gammas),
            "topk": list(topk),
            "depth_tolerance": float(args.depth_tolerance),
            "relative_depth_tolerance": float(args.relative_depth_tolerance),
            "uncertainty_strengths": list(uncertainty_strengths),
        },
        "aggregate": aggregate,
        "mixture_uncertainty_diagnostic": {
            "definition": (
                "alpha-mean relative coefficient variance E||a||^2-||E[a]||^2"
            ),
            "correlation_with_raw_teacher_error": pearson_spearman(
                torch.cat(mixture_uncertainty_values),
                torch.cat(baseline_error_values),
            ),
        },
        "decision": decision,
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
    parser.add_argument(
        "--view-residual-checkpoint",
        default="",
        help="Optional query-free zero-mean view residual; empty audits the canonical field.",
    )
    parser.add_argument(
        "--consistency-cache",
        default="",
        help="Optional training-view disagreement cache for uncertainty variants.",
    )
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--frame-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-role", choices=["development", "benchmark"], default="development")
    parser.add_argument("--frozen-selected-variant", default="")
    parser.add_argument("--only-variants", default="")
    parser.add_argument("--gammas", default="1.25,1.5,2,4")
    parser.add_argument("--topk", default="1,2,4")
    parser.add_argument("--uncertainty-strengths", default="1,2,4")
    parser.add_argument("--uncertainty-key", default="weighted_view_disagreement")
    parser.add_argument("--depth-tolerance", type=float, default=0.08)
    parser.add_argument("--relative-depth-tolerance", type=float, default=0.02)
    parser.add_argument("--alpha-threshold", type=float, default=0.02)
    parser.add_argument("--mass-eps", type=float, default=1e-8)
    parser.add_argument("--boundary-quantile", type=float, default=0.2)
    parser.add_argument("--channel-chunk-size", type=int, default=32)
    parser.add_argument("--variant-chunk-size", type=int, default=4)
    parser.add_argument("--max-baseline-alpha-error", type=float, default=2e-5)
    parser.add_argument("--min-baseline-coefficient-cosine", type=float, default=0.99999)
    parser.add_argument("--max-mean-dense-drop", type=float, default=0.005)
    parser.add_argument("--max-p05-dense-drop", type=float, default=0.01)
    parser.add_argument("--max-unsupported-fraction", type=float, default=0.005)
    parser.add_argument("--min-relation-gain", type=float, default=0.005)
    args = parser.parse_args()
    report = audit(args)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "decision": report["decision"],
                "mixture_uncertainty_diagnostic": report[
                    "mixture_uncertainty_diagnostic"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
