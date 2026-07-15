#!/usr/bin/env python3
"""Audit lifting, coverage, compression, and render-optimization gaps.

This is a query-free reconstruction audit.  It never opens text queries,
benchmark masks, or task labels.  The raw held-out C-RADIO spatial feature map
is the sole evaluation target.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch

from radio_gs.config import load_config
from radio_gs.evaluation.render_ceiling import (
    PixelMetricAccumulator,
    contribution_coverage,
    coverage_bin_masks,
    normalize_premultiplied,
    parse_coverage_edges,
)
from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.training.feature_training_utils import SimpleRadioDataset


def _sha256_tensor_rows(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _parse_frame_ids(raw: str) -> list[int]:
    value = str(raw or "").strip()
    if not value:
        return []
    path = Path(value)
    if path.is_file():
        value = path.read_text(encoding="utf-8")
    tokens: list[str] = []
    for line in value.splitlines():
        tokens.extend(line.split("#", 1)[0].replace(",", " ").split())
    return sorted({int(token) for token in tokens})


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


def _selected_dataset_indices(
    dataset: SimpleRadioDataset,
    *,
    frame_policy: str,
    frame_ids: list[int],
    reference_payload: Mapping,
    mpr_metadata: Mapping,
    max_views: int,
) -> tuple[list[int], list[int]]:
    frame_to_index = {int(frame): index for index, frame in enumerate(dataset.frame_indices)}
    render_metadata = dict(reference_payload.get("render_optimization", {}))
    if frame_ids:
        requested = list(frame_ids)
    elif frame_policy == "render_validation":
        requested = [int(value) for value in render_metadata.get("validation_frames", [])]
    elif frame_policy == "benchmark":
        requested = [
            int(value) for value in render_metadata.get("excluded_benchmark_frames", [])
        ]
        if not requested:
            requested = [int(value) for value in mpr_metadata.get("excluded_frame_ids", [])]
    else:
        training = {
            int(value) for value in mpr_metadata.get("selected_frame_indices", [])
        }
        excluded = {int(value) for value in mpr_metadata.get("excluded_frame_ids", [])}
        requested = [
            int(frame)
            for frame in dataset.frame_indices
            if int(frame) not in training and int(frame) not in excluded
        ]
    missing = sorted(set(requested) - set(frame_to_index))
    if missing:
        raise ValueError(f"requested reconstruction frames are unavailable: {missing}")
    if max_views > 0 and len(requested) > max_views:
        positions = torch.linspace(0, len(requested) - 1, max_views).round().long()
        requested = [requested[int(position)] for position in positions]
    indices = [frame_to_index[frame] for frame in requested]
    if not indices:
        raise RuntimeError(f"no frame is available for policy {frame_policy}")
    return indices, requested


def _validate_field_payload(
    payload: Mapping,
    *,
    expected_geometry_hash: str,
    expected_num_gaussians: int,
    mpr_path: Path,
) -> None:
    fingerprint = dict(payload.get("geometry_fingerprint", {}))
    if (
        str(fingerprint.get("xyz_sha256", "")) != expected_geometry_hash
        or int(fingerprint.get("num_gaussians", -1)) != expected_num_gaussians
    ):
        raise ValueError("canonical field/geometry row fingerprint mismatch")
    recorded_mpr = Path(str(payload.get("mpr_cache", ""))).resolve()
    if recorded_mpr != mpr_path.resolve():
        raise ValueError(
            f"field was not trained from the audited MPR cache: {recorded_mpr} != {mpr_path.resolve()}"
        )


def _metric_value(report: Mapping, method: str, scope: str) -> float:
    value = report["methods"][method][scope]["mean_cosine"]
    if value is None:
        raise RuntimeError(f"metric is undefined for {method}/{scope}")
    return float(value)


def _decision_summary(report: Mapping, *, material_gap: float) -> dict:
    scope = "observed_support"
    mpr_total = _metric_value(report, "mpr_full1280_total_alpha", scope)
    mpr_conditional = _metric_value(report, "mpr_full1280_valid_conditioned", scope)
    pca_total = _metric_value(report, "pca_d384_total_alpha", scope)
    pca_conditional = _metric_value(report, "pca_d384_valid_conditioned", scope)
    render_total = _metric_value(report, "renderft_d384_total_alpha", scope)
    render_conditional = _metric_value(report, "renderft_d384_valid_conditioned", scope)
    mpr_total_rmse = float(
        report["methods"]["mpr_full1280_total_alpha"][scope]["mean_rmse"]
    )
    mpr_conditional_rmse = float(
        report["methods"]["mpr_full1280_valid_conditioned"][scope]["mean_rmse"]
    )
    coverage_error_spearman = report["methods"][
        "mpr_full1280_valid_conditioned"
    ][scope]["coverage_error_spearman"]
    gaps = {
        "lifting_context_compositing_gap_from_raw_teacher": 1.0 - mpr_conditional,
        "coverage_conditioning_cosine_change": mpr_conditional - mpr_total,
        "coverage_conditioning_rmse_reduction": mpr_total_rmse - mpr_conditional_rmse,
        "pca_compression_gap_total_alpha": mpr_total - pca_total,
        "pca_compression_gap_valid_conditioned": mpr_conditional - pca_conditional,
        "render_optimization_gain_total_alpha": render_total - pca_total,
        "render_optimization_gain_valid_conditioned": render_conditional - pca_conditional,
    }
    low_coverage_fraction = float(
        report["coverage"]["visible_fraction_below_0.75"]
    )
    unsupported_fraction = float(
        report["coverage"]["visible_fraction_without_valid_contribution"]
    )
    compression_primary = max(
        gaps["pca_compression_gap_total_alpha"],
        gaps["pca_compression_gap_valid_conditioned"],
    ) >= material_gap
    coverage_material = unsupported_fraction >= 0.05 or (
        low_coverage_fraction >= 0.20
        and coverage_error_spearman is not None
        and float(coverage_error_spearman) <= -0.20
    )
    render_optimization_material = max(
        gaps["render_optimization_gain_total_alpha"],
        gaps["render_optimization_gain_valid_conditioned"],
    ) >= material_gap
    if compression_primary:
        next_test = "compact_representation_or_decoder"
    elif coverage_material:
        next_test = "coverage_aware_mpr_and_masked_holdout_imputation"
    elif gaps["lifting_context_compositing_gap_from_raw_teacher"] >= material_gap:
        next_test = "cross_view_variance_and_compositing_ablation"
    elif render_optimization_material:
        next_test = "continue_query_free_render_optimization_with_frozen_gate"
    else:
        next_test = "downstream_query_calibration_ablation"
    return {
        "material_cosine_gap_threshold": float(material_gap),
        "gaps": gaps,
        "compression_is_primary": compression_primary,
        "coverage_is_material": coverage_material,
        "coverage_decision_evidence": {
            "visible_fraction_without_valid_contribution": unsupported_fraction,
            "visible_fraction_below_0.75": low_coverage_fraction,
            "coverage_error_spearman": coverage_error_spearman,
            "rule": (
                "unsupported>=0.05 OR (coverage<0.75 fraction>=0.20 AND "
                "Spearman(coverage,error)<=-0.20)"
            ),
        },
        "render_optimization_is_material": render_optimization_material,
        "next_causal_test": next_test,
    }


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
    num_gaussians = int(model.get_xyz().shape[0])

    mpr_path = Path(args.mpr_cache)
    mpr_payload = torch.load(mpr_path, map_location="cpu")
    mpr_fingerprint = dict(mpr_payload.get("geometry_fingerprint", {}))
    if (
        str(mpr_fingerprint.get("xyz_sha256", "")) != geometry_hash
        or int(mpr_fingerprint.get("num_gaussians", -1)) != num_gaussians
    ):
        raise ValueError("MPR/geometry row fingerprint mismatch")
    mpr_features_cpu = torch.as_tensor(mpr_payload["features"])
    row_valid_cpu = torch.as_tensor(mpr_payload["valid"]).bool()
    if mpr_features_cpu.shape[0] != num_gaussians or row_valid_cpu.shape != (num_gaussians,):
        raise ValueError("MPR cache rows do not match geometry")
    mpr_metadata = dict(mpr_payload.get("metadata", {}))
    if bool(mpr_metadata.get("benchmark_masks_opened", False)):
        raise ValueError("MPR provenance reports benchmark-mask access")
    if bool(mpr_metadata.get("text_queries_opened", False)):
        raise ValueError("MPR provenance reports text-query access")

    field_specs = {
        "pca_d384": args.pca_field_checkpoint,
        "renderft_d384": args.renderft_field_checkpoint,
    }
    fields: dict[str, object] = {}
    field_metadata: dict[str, dict] = {}
    reference_payload: Mapping | None = None
    for name, path in field_specs.items():
        field, payload = load_canonical_field_checkpoint(path, map_location="cpu")
        _validate_field_payload(
            payload,
            expected_geometry_hash=geometry_hash,
            expected_num_gaussians=num_gaussians,
            mpr_path=mpr_path,
        )
        architecture = dict(payload.get("architecture", {}))
        if int(architecture.get("feature_dim", -1)) != int(mpr_features_cpu.shape[1]):
            raise ValueError(f"{name} field feature dimension differs from MPR")
        if int(architecture.get("coefficient_dim", -1)) != 384:
            raise ValueError(f"{name} is not the declared d384 ladder stage")
        fields[name] = field.to(device).eval()
        render_optimization = dict(payload.get("render_optimization", {}))
        field_metadata[name] = {
            "checkpoint": str(Path(path).resolve()),
            "architecture": architecture,
            "primitive_metrics": dict(payload.get("final_metrics", {})),
            "render_optimization": {
                key: render_optimization[key]
                for key in (
                    "validation_frames",
                    "excluded_benchmark_frames",
                    "initial_validation_cosine",
                    "best_validation_cosine",
                    "initial_mpr_probe_cosine",
                    "best_mpr_probe_cosine",
                    "best_step",
                    "selection_policy",
                    "max_mpr_drop",
                    "benchmark_masks_opened",
                    "text_queries_opened",
                )
                if key in render_optimization
            },
        }
        if name == "renderft_d384":
            # Retain only split metadata; keeping the full payload here would
            # pin a second CPU copy of the large field state dict.
            reference_payload = {
                "render_optimization": dict(payload.get("render_optimization", {}))
            }
        del payload
        gc.collect()
    if reference_payload is None:
        raise RuntimeError("render-finetuned field metadata is required for frozen frame splits")

    dataset = _dataset(config, renderer)
    dataset_indices, frame_ids = _selected_dataset_indices(
        dataset,
        frame_policy=args.frame_policy,
        frame_ids=_parse_frame_ids(args.frame_ids),
        reference_payload=reference_payload,
        mpr_metadata=mpr_metadata,
        max_views=int(args.max_views),
    )
    mpr_training_frames = {
        int(value) for value in mpr_metadata.get("selected_frame_indices", [])
    }
    overlap = sorted(mpr_training_frames.intersection(frame_ids))
    if overlap:
        raise ValueError(f"audit frames overlap MPR construction views: {overlap}")

    row_valid = row_valid_cpu.to(device)
    mpr_features = mpr_features_cpu.to(device=device, dtype=torch.float32)
    mpr_features.mul_(row_valid[:, None])
    del mpr_features_cpu, mpr_payload
    gc.collect()

    coefficients = {
        name: field.coefficients().detach()
        for name, field in fields.items()
    }
    edges = parse_coverage_edges(args.coverage_bins)
    method_names = (
        "mpr_full1280_total_alpha",
        "mpr_full1280_valid_conditioned",
        "pca_d384_total_alpha",
        "pca_d384_valid_conditioned",
        "renderft_d384_total_alpha",
        "renderft_d384_valid_conditioned",
    )
    accumulators: dict[str, dict[str, PixelMetricAccumulator]] = {
        method: {
            "geometry_visible": PixelMetricAccumulator(),
            "observed_support": PixelMetricAccumulator(),
            **{
                f"coverage_{label}": PixelMetricAccumulator()
                for label in coverage_bin_masks(
                    torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.bool), edges
                )
            },
        }
        for method in method_names
    }
    per_frame: list[dict] = []
    coverage_visible: list[torch.Tensor] = []
    alpha_visible: list[torch.Tensor] = []

    with torch.inference_mode():
        for dataset_index, frame_id in zip(dataset_indices, frame_ids):
            sample = dataset[dataset_index]
            pose = sample["pose_w2c"].to(device)
            teacher = sample["radio_features"].to(device=device, dtype=torch.float32)
            height, width = teacher.shape[1:]

            support_render = renderer.render_feature_rows(
                model,
                pose,
                row_valid[:, None].float(),
                feature_height=height,
                feature_width=width,
                alpha_normalize=False,
            )
            total_alpha = support_render["alpha_map"].float()
            valid_mass = support_render["feature_map"][0].float().clamp_min(0.0)
            coverage = contribution_coverage(valid_mass, total_alpha, eps=args.alpha_eps)
            geometry_visible = total_alpha >= float(args.alpha_threshold)
            observed_support = geometry_visible & (valid_mass > float(args.alpha_eps))
            bin_masks = coverage_bin_masks(coverage, geometry_visible, edges)

            mpr_render = renderer.render_feature_rows(
                model,
                pose,
                mpr_features,
                feature_height=height,
                feature_width=width,
                alpha_normalize=False,
            )
            if not torch.allclose(
                mpr_render["alpha_map"], total_alpha, atol=2e-5, rtol=2e-5
            ):
                raise RuntimeError("alpha changed while rendering fixed MPR row features")
            mpr_numerator = mpr_render["feature_map"].float()
            predictions = {
                "mpr_full1280_total_alpha": normalize_premultiplied(
                    mpr_numerator, total_alpha, eps=args.alpha_eps
                ),
                "mpr_full1280_valid_conditioned": normalize_premultiplied(
                    mpr_numerator, valid_mass, eps=args.alpha_eps
                ),
            }

            for field_name, field in fields.items():
                standard_render = renderer.render_feature_rows(
                    model,
                    pose,
                    coefficients[field_name],
                    feature_height=height,
                    feature_width=width,
                    alpha_normalize=False,
                )
                standard_coefficients = normalize_premultiplied(
                    standard_render["feature_map"].float(),
                    total_alpha,
                    eps=args.alpha_eps,
                )
                conditioned_render = renderer.render_feature_rows(
                    model,
                    pose,
                    coefficients[field_name] * row_valid[:, None],
                    feature_height=height,
                    feature_width=width,
                    alpha_normalize=False,
                )
                conditioned_coefficients = normalize_premultiplied(
                    conditioned_render["feature_map"].float(),
                    valid_mass,
                    eps=args.alpha_eps,
                )
                predictions[f"{field_name}_total_alpha"] = field.decoder.decode_map(
                    standard_coefficients
                ).float()
                predictions[f"{field_name}_valid_conditioned"] = field.decoder.decode_map(
                    conditioned_coefficients
                ).float()

            frame_methods: dict[str, dict] = {}
            for method, prediction in predictions.items():
                method_accumulators = accumulators[method]
                # A valid-conditioned feature is undefined when valid_mass is
                # zero.  Do not manufacture a metric there from decoder bias.
                if not method.endswith("valid_conditioned"):
                    method_accumulators["geometry_visible"].update(
                        prediction,
                        teacher,
                        geometry_visible,
                        coverage=coverage,
                    )
                method_accumulators["observed_support"].update(
                    prediction,
                    teacher,
                    observed_support,
                    coverage=coverage,
                )
                frame_accumulator = PixelMetricAccumulator()
                frame_accumulator.update(
                    prediction,
                    teacher,
                    observed_support,
                    coverage=coverage,
                )
                frame_methods[method] = frame_accumulator.summary()
                for label, mask in bin_masks.items():
                    conditional_mask = (
                        mask & observed_support
                        if method.endswith("valid_conditioned")
                        else mask
                    )
                    method_accumulators[f"coverage_{label}"].update(
                        prediction,
                        teacher,
                        conditional_mask,
                        coverage=coverage,
                    )

            visible_coverage = coverage[geometry_visible].detach().cpu()
            visible_alpha = total_alpha[geometry_visible].detach().cpu()
            coverage_visible.append(visible_coverage)
            alpha_visible.append(visible_alpha)
            per_frame.append(
                {
                    "frame_id": int(frame_id),
                    "geometry_visible_pixels": int(geometry_visible.sum()),
                    "observed_support_pixels": int(observed_support.sum()),
                    "mean_total_alpha": float(visible_alpha.mean()) if visible_alpha.numel() else None,
                    "mean_valid_contribution_coverage": (
                        float(visible_coverage.mean()) if visible_coverage.numel() else None
                    ),
                    "fraction_coverage_below_0.75": (
                        float((visible_coverage < 0.75).float().mean())
                        if visible_coverage.numel()
                        else None
                    ),
                    "methods_on_observed_support": frame_methods,
                }
            )
            print(
                f"[{args.frame_policy}] frame {frame_id}: "
                f"coverage={per_frame[-1]['mean_valid_contribution_coverage']:.4f} "
                f"MPR={frame_methods['mpr_full1280_valid_conditioned']['mean_cosine']:.4f} "
                f"PCA={frame_methods['pca_d384_valid_conditioned']['mean_cosine']:.4f} "
                f"FT={frame_methods['renderft_d384_valid_conditioned']['mean_cosine']:.4f}",
                flush=True,
            )

    all_coverage = torch.cat(coverage_visible).float()
    all_alpha = torch.cat(alpha_visible).float()
    methods_report = {
        method: {
            scope: accumulator.summary()
            for scope, accumulator in scopes.items()
        }
        for method, scopes in accumulators.items()
    }
    report = {
        "schema_version": 1,
        "audit": "render_ceiling_coverage_v1",
        "protocol": {
            "query_free": True,
            "raw_2d_teacher_is_metric_target_only": True,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "frame_policy": args.frame_policy,
            "frame_ids": frame_ids,
            "mpr_training_overlap": overlap,
            "alpha_threshold": float(args.alpha_threshold),
            "alpha_eps": float(args.alpha_eps),
            "coverage_definition": (
                "sum(T_i*alpha_i*row_valid_i)/sum(T_i*alpha_i), with identical "
                "all-Gaussian transmittance and occlusion order"
            ),
            "conditional_definition": (
                "premultiplied valid-row feature sum divided by valid-row contribution mass"
            ),
        },
        "artifacts": {
            "config": str(Path(args.config).resolve()),
            "geometry_checkpoint": str(Path(args.geometry_checkpoint).resolve()),
            "geometry_xyz_sha256": geometry_hash,
            "mpr_cache": str(mpr_path.resolve()),
            "fields": field_metadata,
        },
        "num_views": len(frame_ids),
        "coverage": {
            "visible_pixels": int(all_coverage.numel()),
            "mean_total_alpha": float(all_alpha.mean()),
            "mean_valid_contribution_coverage": float(all_coverage.mean()),
            "p05_valid_contribution_coverage": float(torch.quantile(all_coverage, 0.05)),
            "median_valid_contribution_coverage": float(all_coverage.median()),
            "visible_fraction_below_0.25": float((all_coverage < 0.25).float().mean()),
            "visible_fraction_below_0.50": float((all_coverage < 0.50).float().mean()),
            "visible_fraction_below_0.75": float((all_coverage < 0.75).float().mean()),
            "visible_fraction_at_least_0.95": float((all_coverage >= 0.95).float().mean()),
            "visible_fraction_without_valid_contribution": float(
                (all_coverage <= float(args.alpha_eps)).float().mean()
            ),
        },
        "methods": methods_report,
        "per_frame": per_frame,
    }
    report["decision"] = _decision_summary(
        report, material_gap=float(args.material_cosine_gap)
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--mpr-cache", required=True)
    parser.add_argument("--pca-field-checkpoint", required=True)
    parser.add_argument("--renderft-field-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--frame-policy",
        choices=["render_validation", "benchmark", "mpr_heldout"],
        default="render_validation",
    )
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument("--alpha-threshold", type=float, default=0.02)
    parser.add_argument("--alpha-eps", type=float, default=1e-6)
    parser.add_argument("--coverage-bins", default="0,0.25,0.5,0.75,0.95,1.000001")
    parser.add_argument("--material-cosine-gap", type=float, default=0.02)
    args = parser.parse_args()
    report = audit(args)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "num_views": report["num_views"],
                "coverage": report["coverage"],
                "observed_support_cosine": {
                    method: values["observed_support"]["mean_cosine"]
                    for method, values in report["methods"].items()
                },
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
