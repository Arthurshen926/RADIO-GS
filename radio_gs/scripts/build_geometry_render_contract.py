#!/usr/bin/env python3
"""Build a query-free renderer contract from a trained RGB 3DGS PLY.

The resulting config/checkpoint exists only so observation lifting can reuse
the repository's audited Gaussian renderer.  Random compact feature rows and
the codec are never read as teacher targets or method descriptors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
from radio_gs.models.featsharp_3d import FeatSharp3D
from radio_gs.models.hcd_codec import build_feature_codec


def registered_scannet_source_config(scene_root: str | Path) -> dict[str, str]:
    """Return explicit registered RGB-D paths for a materialized ScanNet scene."""

    root = Path(scene_root).resolve()
    required = {
        "rgb_dir": root / "color",
        "depth_dir": root / "depth",
        "pose_dir": root / "pose",
    }
    missing = [str(path) for path in required.values() if not path.is_dir()]
    if missing:
        raise ValueError(
            "materialized ScanNet source lacks registered observation directories: "
            + ", ".join(missing)
        )
    return {
        "rgb_dir": str(required["rgb_dir"]),
        "val_rgb_dir": str(required["rgb_dir"]),
        "depth_dir": str(required["depth_dir"]),
        "val_depth_dir": str(required["depth_dir"]),
        "pose_file": "",
        "val_pose_file": "",
        "pose_dir": str(required["pose_dir"]),
        "val_pose_dir": str(required["pose_dir"]),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_field_source_contract(
    observation_contract: str,
    source_contract: dict[str, object] | None,
    *,
    source_root: str | Path,
) -> None:
    """Fail closed when a named field source lacks matching provenance.

    The renderer is the first place where a field source becomes trainable.
    Checking here prevents a caller from labelling sparse convenience frames as
    a full ScanNet reconstruction only after the expensive Gaussian/MPR work
    has completed.  Historic unnamed contracts remain readable exclusively
    through their legacy/pilot names.
    """

    declared = str(observation_contract).strip()
    required_versions = {
        "dense_pfpr_queryheldout_v1": "scannet-pfpr-query-heldout-field-v1",
        "dense_agile_all_observations_pilot": "scannet-agile-dense-observation-field-v1",
        "scannet_full_observation_v1": "scannet_full_observation_v1",
        "scannet_full_observation_pfpr_queryheldout_v1": (
            "scannet_full_observation_pfpr_queryheldout_v1"
        ),
    }
    expected = required_versions.get(declared)
    if expected is None:
        return
    if source_contract is None or str(source_contract.get("field_contract_version", "")) != expected:
        raise ValueError(
            f"{declared} requires a matching full source contract version {expected!r}"
        )
    if not str(source_contract.get("field_frame_manifest_sha256", "")):
        raise ValueError(f"{declared} source contract lacks a frame-manifest digest")
    full_sens_contracts = {
        "scannet_full_observation_v1",
        "scannet_full_observation_pfpr_queryheldout_v1",
    }
    if declared not in full_sens_contracts:
        return
    if "scannet_frames_25k" in str(Path(source_root).resolve()):
        raise ValueError("scannet_full_observation_v1 cannot use scannet_frames_25k")
    if not str(source_contract.get("source_sens_sha256", "")):
        raise ValueError("scannet_full_observation_v1 source contract lacks a .sens digest")
    if int(source_contract.get("full_sens_frame_count", 0)) <= 0:
        raise ValueError("scannet_full_observation_v1 source contract lacks full frame count")
    if declared == "scannet_full_observation_pfpr_queryheldout_v1":
        if int(source_contract.get("excluded_query_source_frame_count", 0)) <= 0:
            raise ValueError(
                "full-sens PFPR source contract lacks query-frame exclusions"
            )
        if not str(source_contract.get("excluded_query_source_frame_ids_sha256", "")):
            raise ValueError(
                "full-sens PFPR source contract lacks an exclusion-set digest"
            )


def build_contract(
    *,
    ply_path: str | Path,
    scene_root: str | Path,
    feature_dir: str | Path,
    output_config: str | Path,
    output_checkpoint: str | Path,
    latent_dim: int = 8,
    observation_contract: str = "field_only_dense_rgbd_v1",
) -> dict:
    ply_path = Path(ply_path).resolve()
    scene_root = Path(scene_root).resolve()
    feature_dir = Path(feature_dir).resolve()
    output_config = Path(output_config).resolve()
    output_checkpoint = Path(output_checkpoint).resolve()
    observation_contract = str(observation_contract).strip()
    if not observation_contract:
        raise ValueError("observation_contract must be non-empty")
    intrinsic_path = scene_root / "intrinsic" / "intrinsic_depth.txt"
    if not intrinsic_path.is_file():
        intrinsic_path = scene_root / "intrinsics_depth.txt"
    intrinsic = np.loadtxt(intrinsic_path, dtype=np.float32).reshape(4, 4)

    # A benchmark may materialize a deliberately held-out RGB-D source.  The
    # renderer needs only its public, anchor-free provenance digest; retaining
    # it here makes later field/evaluator reports fail closed on a mismatched
    # observation source without copying any query geometry into the model.
    # ``pfpr_field_contract.json`` is the historic held-out query source.
    # New query-independent sources use a neutral filename so AGILE can prove
    # that every valid RGB-D observation was available without conflating it
    # with evaluator-private pose-free query construction.
    source_contract_path = next(
        (
            candidate
            for candidate in (
                scene_root / "field_source_contract.json",
                scene_root / "pfpr_field_contract.json",
            )
            if candidate.is_file()
        ),
        scene_root / "field_source_contract.json",
    )
    source_contract: dict[str, object] | None = None
    if source_contract_path.is_file():
        source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
        forbidden = (
            "anchor_world_xyz",
            "source_depth_pixel_uv",
            "instance_id",
            "semantic_label",
        )
        if any(key in source_contract for key in forbidden):
            raise ValueError("field source contract exposes private evaluator data")
        if bool(source_contract.get("uses_private_anchor", False)) or bool(
            source_contract.get("uses_private_depth_pixel", False)
        ) or bool(source_contract.get("uses_instances_or_semantic_labels", False)):
            raise ValueError("field source contract is not query-free")
    validate_field_source_contract(
        observation_contract,
        source_contract,
        source_root=scene_root,
    )

    # PFIR reconstruction inputs are normalized to the depth-camera raster.
    from PIL import Image

    depth_path = next(iter(sorted((scene_root / "depth").glob("*.png"))))
    with Image.open(depth_path) as image:
        image_width, image_height = image.size
    feature_height = image_height // 8
    feature_width = image_width // 8

    torch.manual_seed(0)
    model = ExplicitFeatureGaussian(latent_dim=int(latent_dim))
    model.load_from_ply(str(ply_path))
    codec = build_feature_codec(
        input_dim=1280,
        bottleneck_dim=int(latent_dim),
        codec_type="direct",
        dual_stream=False,
        symmetric_decoder=False,
    )
    sharpener = FeatSharp3D(
        mode="analytical", feature_dim=int(latent_dim), strength=0.0
    )
    provenance = {
        "schema_version": 1,
        "purpose": "query_free_geometry_render_contract",
        "scene_root": str(scene_root),
        "ply_path": str(ply_path),
        "ply_sha256": _sha256(ply_path),
        "feature_dir": str(feature_dir),
        "observation_contract": observation_contract,
        "random_feature_rows_used_by_method": False,
        "codec_used_by_method": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    if source_contract is not None:
        provenance.update(
            {
                "field_source_contract": str(source_contract_path),
                "field_source_contract_sha256": _sha256(source_contract_path),
                "field_source_contract_version": str(
                    source_contract.get("field_contract_version", "")
                ),
                "field_source_frame_manifest_sha256": str(
                    source_contract.get("field_frame_manifest_sha256", "")
                ),
                "field_source_excluded_query_frame_count": int(
                    source_contract.get("excluded_query_source_frame_count", 0)
                ),
                "field_source_excluded_query_frame_ids_sha256": str(
                    source_contract.get("excluded_query_source_frame_ids_sha256", "")
                ),
            }
        )
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "codec_state_dict": codec.state_dict(),
            "sharpener_state_dict": sharpener.state_dict(),
            "geometry_render_contract": provenance,
        },
        output_checkpoint,
    )
    registered_source = registered_scannet_source_config(scene_root)
    config = {
        "architecture": "explicit",
        "latent_dim": int(latent_dim),
        "radio_feature_dim": 1280,
        "bottleneck_dim": int(latent_dim),
        "codec_type": "direct",
        "dual_stream": False,
        "symmetric_decoder": False,
        "ply_path": str(ply_path),
        "dataset_type": "scannet",
        "scene": scene_root.name,
        "scene_root": str(scene_root),
        "observation_contract": observation_contract,
        "feature_dir": str(feature_dir),
        **registered_source,
        "image_height": int(image_height),
        "image_width": int(image_width),
        "fx": float(intrinsic[0, 0]),
        "fy": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
        "feature_height": int(feature_height),
        "feature_width": int(feature_width),
        "max_channels_per_chunk": 32,
        "use_2dgs": False,
        "featsharp_mode": "analytical",
        "featsharp_strength": 0.0,
        "use_refiner": False,
    }
    if source_contract is not None:
        config.update(
            {
                "field_source_contract_sha256": provenance[
                    "field_source_contract_sha256"
                ],
                "field_source_contract_version": provenance[
                    "field_source_contract_version"
                ],
                "field_source_frame_manifest_sha256": provenance[
                    "field_source_frame_manifest_sha256"
                ],
                "field_source_excluded_query_frame_count": provenance[
                    "field_source_excluded_query_frame_count"
                ],
                "field_source_excluded_query_frame_ids_sha256": provenance[
                    "field_source_excluded_query_frame_ids_sha256"
                ],
            }
        )
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    report = {
        **provenance,
        "config": str(output_config),
        "checkpoint": str(output_checkpoint),
        "checkpoint_sha256": _sha256(output_checkpoint),
        "num_gaussians": int(model.num_gaussians),
        "image_size_hw": [int(image_height), int(image_width)],
        "feature_size_hw": [int(feature_height), int(feature_width)],
    }
    output_checkpoint.with_suffix(output_checkpoint.suffix + ".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply-path", required=True)
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument(
        "--observation-contract",
        default="field_only_dense_rgbd_v1",
        help="auditable registered-RGB-D source contract for this field",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_contract(
                ply_path=args.ply_path,
                scene_root=args.scene_root,
                feature_dir=args.feature_dir,
                output_config=args.output_config,
                output_checkpoint=args.output_checkpoint,
                latent_dim=args.latent_dim,
                observation_contract=args.observation_contract,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
