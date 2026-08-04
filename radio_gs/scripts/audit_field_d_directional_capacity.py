#!/usr/bin/env python3
"""Audit two-mode primitive direction capacity on query-free training views."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch

from radio_gs.config import load_config
from radio_gs.field.directional_distribution import (
    DIRECTIONAL_PROTOTYPE_CONTRACT,
    directional_prototype_observation_cosines,
    fit_two_direction_prototypes,
)
from radio_gs.rendering.contribution_compositor import (
    rasterize_single_view_contributions,
)
from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
    _load_bundle_feature_maps,
    _validated_feature_bundle,
    prepare_raster_view_features,
)
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    accumulate_raster_contribution_features,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.training.feature_training_utils import SimpleRadioDataset
from radio_gs.training.tensor_cache_io import load_mpr_cache
from radio_gs.utils.immutable_artifacts import sha256_file


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _pose_sources(config, feature_dir: Path) -> tuple[str | None, str | None]:
    raw_pose_file = str(getattr(config, "pose_file", "") or "").strip()
    pose_file = raw_pose_file if raw_pose_file and Path(raw_pose_file).is_file() else None
    raw_pose_dir = str(getattr(config, "pose_dir", "") or "").strip()
    fallback = feature_dir / "poses_w2c"
    pose_dir = (
        raw_pose_dir
        if raw_pose_dir and Path(raw_pose_dir).is_dir()
        else str(fallback)
        if fallback.is_dir()
        else None
    )
    return pose_file, pose_dir


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    mpr, mpr_sha, mpr_path = load_mpr_cache(
        args.mpr_cache,
        expected_sha256=args.expected_mpr_cache_sha256,
        expected_feature_space="radio",
        require_reliability=True,
        require_formal_safety=True,
    )
    metadata = dict(mpr["metadata"])
    expected_policy = {
        "aggregation_mode": "raster_exact_center_uncertainty",
        "registration_weight_mode": "exact_front_to_back_adjoint_center",
        "normalize_each_view": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    mismatched = [
        key for key, value in expected_policy.items() if metadata.get(key) != value
    ]
    if mismatched:
        raise ValueError(f"Field-D capacity MPR policy differs: {mismatched}")
    config_path = Path(metadata["config"]).expanduser().resolve()
    geometry_path = Path(metadata["checkpoint"]).expanduser().resolve()
    if sha256_file(geometry_path) != args.expected_geometry_checkpoint_sha256:
        raise ValueError("Field-D geometry checkpoint differs")
    if sha256_file(config_path) != args.expected_config_sha256:
        raise ValueError("Field-D config differs")

    device = torch.device(args.device)
    config = load_config(config_path)
    model, _codec, renderer, _sharpener, _refiner, _cfg, _hybrid = (
        load_render_pipeline(
            str(config_path),
            str(geometry_path),
            device,
            strict_checkpoint_contract=True,
            load_ply_rgb_features=False,
            expected_checkpoint_sha256=args.expected_geometry_checkpoint_sha256,
        )
    )
    xyz = model.get_xyz().detach().float().cpu()
    cached_xyz = torch.as_tensor(mpr["xyz"]).float().cpu()
    if xyz.shape != cached_xyz.shape or not torch.equal(xyz, cached_xyz):
        raise ValueError("Field-D MPR/renderer primitive rows differ")

    feature_height = int(getattr(config, "feature_height", renderer.image_height))
    feature_width = int(getattr(config, "feature_width", renderer.image_width))
    feature_dir = Path(str(getattr(config, "feature_dir", ""))).expanduser().resolve()
    _manifest, validation, tensor_records = _validated_feature_bundle(
        feature_dir,
        expected_output_bundle_sha256=str(metadata["feature_output_bundle_sha256"]),
    )
    if validation["manifest_sha256"] != metadata["feature_frame_manifest_sha256"]:
        raise ValueError("Field-D feature manifest differs")
    pose_file, pose_dir = _pose_sources(config, feature_dir)
    dataset = SimpleRadioDataset(
        feature_dir=str(feature_dir),
        pose_file=pose_file,
        pose_dir=pose_dir,
        feature_size=(feature_height, feature_width),
        split="train",
        dataset_type=str(getattr(config, "dataset_type", "lerf")),
    )
    frame_to_index = {
        int(frame): index for index, frame in enumerate(dataset.frame_indices)
    }
    frames = [int(value) for value in metadata["selected_frame_indices"]]
    if len(frames) != len(set(frames)) or not frames:
        raise ValueError("Field-D selected training views are invalid")
    missing = sorted(set(frames) - set(frame_to_index))
    if missing:
        raise ValueError(f"Field-D training feature frames are missing: {missing}")

    valid_rows = torch.where(torch.as_tensor(mpr["valid"]).bool())[0]
    requested_rows = int(args.sample_rows)
    sample_size = (
        int(valid_rows.numel())
        if requested_rows <= 0
        else min(requested_rows, int(valid_rows.numel()))
    )
    if sample_size <= 0:
        raise ValueError("Field-D capacity audit has no valid rows")
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    chosen = valid_rows[torch.randperm(valid_rows.numel(), generator=generator)[:sample_size]]
    chosen = chosen.sort().values
    global_to_sample = torch.full(
        (xyz.shape[0],), -1, dtype=torch.long, device=device
    )
    global_to_sample[chosen.to(device)] = torch.arange(sample_size, device=device)
    observations = torch.zeros(
        len(frames), sample_size, 1280, dtype=torch.float16
    )
    observation_weight = torch.zeros(len(frames), sample_size, dtype=torch.float32)

    for view_index, frame in enumerate(frames):
        feature_map = _load_bundle_feature_maps(
            feature_dir=feature_dir,
            selected_frame_indices=[frame],
            subdir="backbone",
            expected_dim=1280,
            feature_size=(feature_height, feature_width),
            tensor_records=tensor_records,
            normalize=False,
            output_dtype=torch.float32,
        )
        feature_map = prepare_raster_view_features(
            feature_map.to(device), normalize_each_view=True
        )
        pose = torch.from_numpy(
            dataset.poses_w2c[frame_to_index[frame]]
        ).float().to(device)
        hits = rasterize_single_view_contributions(
            model,
            renderer,
            pose,
            height=feature_height,
            width=feature_width,
        )
        local_ids = global_to_sample[hits["gaussian_ids"].long()]
        keep = local_ids >= 0
        frame_sum, frame_mass = accumulate_raster_contribution_features(
            feature_map,
            local_ids[keep],
            hits["pixel_ids"][keep],
            hits["weights"][keep],
            n_gaussians=sample_size,
        )
        frame_mass = frame_mass.float().cpu()
        active = frame_mass > 0
        if bool(active.any()):
            observations[view_index, active] = (
                frame_sum.float().cpu()[active]
                / frame_mass[active, None].clamp_min(1e-8)
            ).half()
        observation_weight[view_index] = frame_mass
        print(
            json.dumps(
                {
                    "frame": frame,
                    "sample_observed_rows": int(active.sum()),
                    "sample_rows": sample_size,
                }
            ),
            flush=True,
        )
        del feature_map, hits, local_ids, keep, frame_sum, frame_mass
        if device.type == "cuda":
            torch.cuda.empty_cache()

    observation_valid = observation_weight > 0
    supported = observation_valid.sum(dim=0) >= int(args.minimum_views)
    if not bool(supported.any()):
        raise RuntimeError("Field-D audit has no rows with enough views")
    observations = observations[:, supported]
    observation_valid = observation_valid[:, supported]
    observation_weight = observation_weight[:, supported]
    # Equal view mass prevents a large projected footprint from erasing a
    # minority direction; exact visibility still decides whether it exists.
    supported_rows = int(supported.sum())
    prototype_values = torch.zeros(supported_rows, 2, 1280, dtype=torch.float16)
    mixture_values = torch.zeros(supported_rows, 2, dtype=torch.float16)
    resultant_values = torch.zeros(supported_rows, dtype=torch.float32)
    count_values = observation_valid.sum(dim=0).to(torch.int16)
    center_parts: list[torch.Tensor] = []
    prototype_parts: list[torch.Tensor] = []
    mass_parts: list[torch.Tensor] = []
    for start in range(0, supported_rows, int(args.prototype_row_chunk_size)):
        stop = min(start + int(args.prototype_row_chunk_size), supported_rows)
        chunk_observations = observations[:, start:stop]
        chunk_valid = observation_valid[:, start:stop]
        chunk = fit_two_direction_prototypes(
            chunk_observations,
            chunk_valid,
            weights=None,
            iterations=int(args.iterations),
        )
        prototype_values[start:stop] = chunk.prototypes.half()
        mixture_values[start:stop] = chunk.mixture_weight.half()
        resultant_values[start:stop] = chunk.center_resultant
        center_cosine, prototype_cosine, active_mass = (
            directional_prototype_observation_cosines(
                chunk.prototypes,
                chunk_observations,
                chunk_valid,
                weights=None,
            )
        )
        center_parts.append(center_cosine.cpu())
        prototype_parts.append(prototype_cosine.cpu())
        mass_parts.append(active_mass.cpu())
    center_cosine = torch.cat(center_parts)
    prototype_cosine = torch.cat(prototype_parts)
    active_mass = torch.cat(mass_parts)
    coverage = {
        "center_weighted_mean_cosine": (
            center_cosine * active_mass
        ).sum() / active_mass.sum(),
        "prototype_weighted_mean_cosine": (
            prototype_cosine * active_mass
        ).sum() / active_mass.sum(),
        "center_p05_cosine": torch.quantile(center_cosine, 0.05),
        "prototype_p05_cosine": torch.quantile(prototype_cosine, 0.05),
    }
    improvement = {
        "weighted_mean_cosine": float(
            coverage["prototype_weighted_mean_cosine"]
            - coverage["center_weighted_mean_cosine"]
        ),
        "p05_cosine": float(
            coverage["prototype_p05_cosine"] - coverage["center_p05_cosine"]
        ),
    }
    report: dict[str, object] = {
        "schema_version": "field_d_directional_capacity_audit_v1",
        "contract": DIRECTIONAL_PROTOTYPE_CONTRACT,
        "inputs": {
            "mpr": {"path": str(mpr_path), "sha256": mpr_sha},
            "config": {"path": str(config_path), "sha256": args.expected_config_sha256},
            "geometry_checkpoint": {
                "path": str(geometry_path),
                "sha256": args.expected_geometry_checkpoint_sha256,
            },
            "feature_output_bundle_sha256": metadata["feature_output_bundle_sha256"],
            "feature_manifest_sha256": validation["manifest_sha256"],
        },
        "sampling": {
            "seed": int(args.seed),
            "requested_rows": requested_rows,
            "sampled_valid_rows": sample_size,
            "minimum_views": int(args.minimum_views),
            "supported_rows": supported_rows,
            "training_views": frames,
        },
        "coverage": {key: float(value) for key, value in coverage.items()},
        "prototype_minus_center": improvement,
        "center_resultant": {
            "mean": float(resultant_values.mean()),
            "p10": float(torch.quantile(resultant_values, 0.10)),
            "p50": float(torch.quantile(resultant_values, 0.50)),
        },
        "gate": {
            "mean_cosine_improves": improvement["weighted_mean_cosine"] > 0.0,
            "p05_cosine_improves": improvement["p05_cosine"] > 0.0,
        },
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    report["passed"] = bool(all(report["gate"].values()))
    prototype_cache_path = None
    prototype_cache_sha256 = None
    if str(args.prototype_cache_output).strip():
        prototype_cache_path = Path(args.prototype_cache_output).expanduser().resolve()
        prototype_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "contract": DIRECTIONAL_PROTOTYPE_CONTRACT,
                "global_rows": chosen[supported].long(),
                "prototypes": prototype_values,
                "mixture_weight": mixture_values,
                "observation_count": count_values,
                "center_resultant": resultant_values.half(),
                "geometry_fingerprint": mpr["geometry_fingerprint"],
                "metadata": {
                    "source_mpr_path": str(mpr_path),
                    "source_mpr_sha256": mpr_sha,
                    "config_sha256": args.expected_config_sha256,
                    "geometry_checkpoint_sha256": (
                        args.expected_geometry_checkpoint_sha256
                    ),
                    "feature_output_bundle_sha256": metadata[
                        "feature_output_bundle_sha256"
                    ],
                    "training_views": frames,
                    "equal_view_mass": True,
                    "prototype_iterations": int(args.iterations),
                    "minimum_views": int(args.minimum_views),
                    "seed": int(args.seed),
                    "benchmark_images_opened": False,
                    "benchmark_masks_opened": False,
                    "text_queries_opened": False,
                },
            },
            prototype_cache_path,
        )
        prototype_cache_sha256 = sha256_file(prototype_cache_path)
        report["prototype_cache"] = {
            "path": str(prototype_cache_path),
            "sha256": prototype_cache_sha256,
            "rows": supported_rows,
            "shape": list(prototype_values.shape),
        }
    output = Path(args.output).expanduser().resolve()
    _atomic_json(output, report)
    return {**report, "output": str(output), "output_sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpr-cache", required=True)
    parser.add_argument("--expected-mpr-cache-sha256", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-geometry-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rows", type=int, default=2048)
    parser.add_argument("--minimum-views", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--prototype-row-chunk-size", type=int, default=512)
    parser.add_argument("--prototype-cache-output", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
