#!/usr/bin/env python3
"""Inventory and align the reproduced LUDVIG NVOS all-view assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

from radio_gs.interfaces.capability_cache import (
    _load_memory_mapped_capability_payload,
)


SCENES = (
    "fern",
    "flower",
    "fortress",
    "horns_center",
    "horns_left",
    "leaves",
    "orchids",
    "trex",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def only(paths, *, label: str) -> Path:
    resolved = [path for path in paths if path.is_file()]
    if len(resolved) != 1:
        raise ValueError(f"expected one {label}, found {len(resolved)}")
    return resolved[0].resolve()


def asset(path: Path, **metadata: object) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        **metadata,
    }


def xyz_from_ply(path: Path) -> np.ndarray:
    vertex = PlyData.read(str(path))["vertex"]
    return np.column_stack(
        [vertex["x"], vertex["y"], vertex["z"]]
    ).astype(np.float32, copy=False)


def spatial_alignment(ludvig: np.ndarray, canonical: np.ndarray) -> dict[str, object]:
    tree = cKDTree(canonical)
    step = max(1, len(canonical) // 50_000)
    sample = canonical[::step][:50_000]
    local_spacing = float(np.median(tree.query(sample, k=2, workers=-1)[0][:, 1]))
    distance, nearest = tree.query(ludvig, k=1, workers=-1)
    quantiles = np.quantile(distance, [0.0, 0.5, 0.9, 0.99, 1.0])
    return {
        "ludvig_rows": len(ludvig),
        "canonical_rows": len(canonical),
        "row_count_equal": len(ludvig) == len(canonical),
        "direct_row_or_index_mapping": False,
        "canonical_median_nearest_neighbor_spacing_50k_sample": local_spacing,
        "ludvig_to_canonical_nearest_distance_quantiles_0_50_90_99_100": [
            float(value) for value in quantiles
        ],
        "median_distance_over_canonical_spacing": float(quantiles[1] / local_spacing),
        "within_half_spacing_fraction": float(np.mean(distance <= 0.5 * local_spacing)),
        "within_one_spacing_fraction": float(np.mean(distance <= local_spacing)),
        "within_two_spacing_fraction": float(np.mean(distance <= 2.0 * local_spacing)),
        "unique_canonical_nearest_fraction": float(len(np.unique(nearest)) / len(nearest)),
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    repo = Path(args.repo_root).resolve()
    root = (repo / "output/protocol_audit_20260731/ludvig/nvos/released_all_view").resolve()
    field_root = Path(args.canonical_field_root).resolve()
    summary_path = (root.parent / "released_all_view_full8_3seed_summary.json").resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if float(summary["local_scene_macro_iou_percent"]) != 91.25768502741802:
        raise ValueError("reproduced LUDVIG summary differs")

    code_paths = {
        "sam_dataset": Path("/root/baselines/LUDVIG/predictors/sam.py"),
        "uplift": Path("/root/baselines/LUDVIG/ludvig_uplift.py"),
        "segmentation": Path("/root/baselines/LUDVIG/evaluation/spin_nvos/segmentation.py"),
    }
    scenes: dict[str, object] = {}
    for scene in SCENES:
        runs: dict[str, object] = {}
        geometry_hashes: list[str] = []
        for seed in range(3):
            attempt_root = root / scene / f"seed_{seed}" / "attempts"
            feature_path = only(
                attempt_root.glob(f"*/{scene}/sam/features.npy"),
                label=f"{scene} seed {seed} features",
            )
            sam_root = feature_path.parent
            gaussian_path = sam_root / "gaussians.ply"
            config_path = sam_root / "config.yaml"
            result_path = sam_root / "protocol_result.json"
            float_mask = only(
                (sam_root / "masks/float").glob("*.png"),
                label=f"{scene} seed {seed} float selector",
            )
            binary_mask = only(
                (sam_root / "masks/binary").glob("*.png"),
                label=f"{scene} seed {seed} binary selector",
            )
            run_manifest = feature_path.parents[2] / "run_manifest.json"
            values = np.load(feature_path, mmap_mode="r")
            gaussian_rows = int(PlyData.read(str(gaussian_path))["vertex"].count)
            if values.shape != (gaussian_rows, 1) or values.dtype != np.float32:
                raise ValueError(f"{scene} seed {seed} is not scalar SAM uplift")
            geometry = asset(gaussian_path, rows=gaussian_rows)
            geometry_hashes.append(str(geometry["sha256"]))
            manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
            required_flags = {
                "target_rgb_visible_during_gaussian_splatting_training": True,
                "target_rgb_visible_during_uplifting": True,
                "target_view_2d_foundation_model_calls": True,
                "strict_unseen_exact_match": False,
            }
            if any(manifest.get(key) is not value for key, value in required_flags.items()):
                raise ValueError(f"{scene} seed {seed} information flags differ")
            config_text = config_path.read_text(encoding="utf-8")
            if "predictors.sam.SAMDataset" not in config_text or "dino" in config_text.lower():
                raise ValueError(f"{scene} seed {seed} is not the SAM reproduction path")
            runs[str(seed)] = {
                "primitive_scalar_uplift": asset(
                    feature_path,
                    shape=list(values.shape),
                    dtype=str(values.dtype),
                ),
                "native_gaussian_geometry": geometry,
                "fixed_threshold_result": asset(result_path),
                "rendered_float_selector": asset(float_mask),
                "rendered_binary_selector": asset(binary_mask),
                "resolved_sam_config": asset(config_path),
                "run_manifest": asset(run_manifest),
                "selected_iou": json.loads(result_path.read_text(encoding="utf-8"))[
                    "selected_iou"
                ],
            }
        if len(set(geometry_hashes)) != 1:
            raise ValueError(f"{scene} geometry changes across stochastic seeds")

        seed0_geometry = Path(
            runs["0"]["native_gaussian_geometry"]["path"]  # type: ignore[index]
        )
        ludvig_xyz = xyz_from_ply(seed0_geometry)
        capability_path = field_root / scene / "official_dino_sam3_views.pt"
        payload = _load_memory_mapped_capability_payload(capability_path)
        canonical_xyz = np.asarray(payload["xyz"])
        scenes[scene] = {
            "runs": runs,
            "geometry_identical_across_three_seeds": True,
            "all_three_scalar_uplifts_differ": len(
                {
                    str(runs[str(seed)]["primitive_scalar_uplift"]["sha256"])  # type: ignore[index]
                    for seed in range(3)
                }
            )
            == 3,
            "canonical_capability_xyz": {
                "path": str(capability_path.resolve()),
                "sidecar_sha256": sha256_file(Path(str(capability_path) + ".json")),
                "rows": len(canonical_xyz),
            },
            "spatial_alignment": spatial_alignment(ludvig_xyz, canonical_xyz),
        }
        del payload, canonical_xyz, ludvig_xyz

    all_view_files = [
        path.relative_to(root).as_posix().lower()
        for path in root.rglob("*")
        if path.is_file()
    ]
    dino_or_pca_assets = [
        path for path in all_view_files if "dino" in path or "pca" in path
    ]
    result = {
        "schema_version": "nvos_ludvig_published_compatible_asset_audit_v1",
        "reproduced_summary": asset(
            summary_path,
            macro_iou=summary["local_scene_macro_iou_percent"] / 100.0,
            protocol_id=summary["protocol_id"],
            strict_unseen_exact_match=False,
        ),
        "code_authority": {
            name: asset(path) for name, path in code_paths.items()
        },
        "information_boundary": {
            "method": "online_target_RGB_SAM_query_interface_then_scalar_uplift",
            "per_camera_operation": "read that registered camera RGB, render inverse-scribble 3D support into the camera as positive point prompts, run SAM on that RGB, then uplift the returned scalar mask",
            "target_rgb_visible_during_3dgs_training": True,
            "target_rgb_visible_during_uplift": True,
            "target_view_SAM_calls": True,
            "query_free_all_view_field": False,
            "eligible_for_strict_source_only": False,
        },
        "capability_type": {
            "persisted_features": "one float32 SAM-derived scalar per native LUDVIG Gaussian per seed",
            "DINOv2_PCA40_present": False,
            "DINO_or_PCA_named_assets_below_reproduced_all_view_root": dino_or_pca_assets,
            "selector": "the scalar primitive uplift rendered to the target and thresholded with the frozen NVOS parameter 75",
        },
        "scenes": scenes,
        "mapping_conclusion": {
            "direct_row_mapping": False,
            "reason": "all eight native all-view Gaussian row counts differ from the strict canonical geometry; nearest-neighbor transfer is strongly many-to-one and usually many canonical spacings away",
            "nearest_xyz_copy_is_valid": False,
            "minimum_faithful_reuse": "retain each cached native LUDVIG Gaussian geometry and its scalar features; render those scalar fields through registered cameras and apply an exact image-space adjoint onto canonical primitives",
            "minimum_reuse_cost": "eight scene bridges per seed, consisting only of scalar rendering plus canonical adjoint; no repeated SAM call is needed when cached scalar features are accepted",
            "claim_boundary": "the transferred value remains a query-specific, target-RGB-informed compatible sidecar and cannot become a strict or query-free canonical capability",
            "query_free_DINO_relation_reconstruction": "requires a new DINOv2/PCA40 all-view extraction and uplift because no such asset exists in this reproduced SAM run",
        },
        "frozen_evaluator_modified": False,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--canonical-field-root",
        default="/mnt/pool/sqy/results/RADIO-GS/output/evaluation_closeout_20260716/canonical_mpr_v3_nvos8",
    )
    parser.add_argument(
        "--output",
        default="paper/artifacts/nvos_ludvig_published_compatible_asset_audit_20260804.json",
    )
    return parser.parse_args()


def main() -> None:
    result = build(parse_args())
    print(json.dumps({"output_scenes": list(result["scenes"]), "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
