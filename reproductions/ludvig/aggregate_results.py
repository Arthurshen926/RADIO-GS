#!/usr/bin/env python3
"""Aggregate LUDVIG protocol results without mixing scenes, runs, or cohorts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from reproductions.ludvig.run_ludvig_sam import (
    NVOS_GEOMETRY_REGISTERED_IMAGES,
    NVOS_TASK_TO_GEOMETRY_SCENE,
    SPIN_GEOMETRY_REGISTERED_IMAGES,
    SPIN_NVOS_SHARED_LLFF_GEOMETRIES,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "paper" / "artifacts" / "promptable_nvs_protocol_registry.yaml"
OFFICIAL_3DGS_COMMIT = "f7a116fb1397d9842239127d39dc212f93171f70"
LUDVIG_COMMIT = "4461fc515439bb498a75d71738a1e73cf7a452ed"
SAM_VIT_H_CHECKPOINT_SHA256 = (
    "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"
)
UPSTREAM_PATCH_V1_SHA256 = (
    "7c86d5883058fb9e529608ba6cf856c04e66d37d338a69a8ca6863725292e9ac"
)
UPSTREAM_PATCHED_FILE_V1_SHA256 = {
    "evaluation/spin_nvos/segmentation.py": (
        "ecc9d7b1a29faa7d1091e14e9c5a50036180c5ba8671ba396d0292075a559b0e"
    ),
    "ludvig_uplift.py": (
        "ae2eb5af2050e619a8d3a6f5bb04d228b4c090425cfcaf30f25aa9a859cddd3e"
    ),
}
UPSTREAM_PATCH_V2_SHA256 = (
    "20061d98ec591d833ef5080bf7c454bc298135b2bb04acbd9e516b6d8d30e587"
)
UPSTREAM_PATCHED_FILE_V2_SHA256 = {
    "evaluation/spin_nvos/base.py": (
        "7ecd469119b4ee87e3cc9cf5764426ec6c8a1d9118db072beab4d8335ea0d353"
    ),
    "evaluation/spin_nvos/segmentation.py": (
        "ecc9d7b1a29faa7d1091e14e9c5a50036180c5ba8671ba396d0292075a559b0e"
    ),
    "ludvig_uplift.py": (
        "ae2eb5af2050e619a8d3a6f5bb04d228b4c090425cfcaf30f25aa9a859cddd3e"
    ),
}
UPSTREAM_PATCH_SHA256 = (
    "2c21257316c6f65d25eea2bbd98481bd3e42f0d84df23a13c1bd1cb645e7d602"
)
UPSTREAM_PATCHED_FILE_SHA256 = {
    "evaluation/spin_nvos/base.py": (
        "7ecd469119b4ee87e3cc9cf5764426ec6c8a1d9118db072beab4d8335ea0d353"
    ),
    "evaluation/spin_nvos/segmentation.py": (
        "ecc9d7b1a29faa7d1091e14e9c5a50036180c5ba8671ba396d0292075a559b0e"
    ),
    "ludvig_uplift.py": (
        "ae2eb5af2050e619a8d3a6f5bb04d228b4c090425cfcaf30f25aa9a859cddd3e"
    ),
    "predictors/sam.py": (
        "3cbf8bda6f7334086c3ba7c117a1b604ed12757c351fff96054a6a2f484684b9"
    ),
    "utils/image.py": (
        "6047b23c26fcece6bc451961a532b302e6aeb0dcdce0c5c89a7e46f71eed87c1"
    ),
    "utils/solver.py": (
        "6b71c91c5e4dbe50b2995f6b9428c9cd9bd6940ab58b1f16fe788dfc50b1c70c"
    ),
}
APPROVED_UPSTREAM_PATCHED_FILE_SHA256 = {
    UPSTREAM_PATCH_V1_SHA256: UPSTREAM_PATCHED_FILE_V1_SHA256,
    UPSTREAM_PATCH_V2_SHA256: UPSTREAM_PATCHED_FILE_V2_SHA256,
    UPSTREAM_PATCH_SHA256: UPSTREAM_PATCHED_FILE_SHA256,
}


class AggregationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_number(value: Any, path: str) -> float:
    if type(value) not in {int, float}:
        raise AggregationError(f"{path} must be a JSON number, not {type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise AggregationError(f"{path} must be finite")
    return numeric


def _strict_iou(value: Any, path: str) -> float:
    numeric = _strict_number(value, path)
    if not 0.0 <= numeric <= 1.0:
        raise AggregationError(f"{path} is outside [0, 1]")
    return numeric


def _validate_result(
    result: dict[str, Any],
    *,
    benchmark: str,
    scene: str,
    result_path: Path,
) -> float:
    prefix = str(result_path)
    expected_benchmark = "NVOS" if benchmark == "nvos" else "SPIn-NeRF"
    if (
        type(result.get("schema_version")) is not int
        or result.get("schema_version") != 1
        or result.get("benchmark") != expected_benchmark
        or result.get("scene") != scene
        or result.get("metric") != "foreground_iou"
        or result.get("reference_scored") is not False
    ):
        raise AggregationError(f"Result protocol identity mismatch in {result_path}")

    reference_frame = result.get("reference_frame")
    if not isinstance(reference_frame, str) or not reference_frame:
        raise AggregationError(f"Missing reference frame in {result_path}")
    if benchmark == "nvos":
        target_frame = result.get("target_frame")
        if (
            not isinstance(target_frame, str)
            or not target_frame
            or target_frame == reference_frame
            or result.get("threshold_policy") != "fixed_across_nvos_scenes"
        ):
            raise AggregationError(f"NVOS frame/threshold policy mismatch in {result_path}")
        threshold = _strict_number(
            result.get("selected_threshold_parameter"),
            f"{prefix}.selected_threshold_parameter",
        )
        if threshold != 75.0:
            raise AggregationError(
                f"NVOS fixed threshold must be 75 in {result_path}; found {threshold}"
            )
        _strict_iou(result.get("oracle_target_iou"), f"{prefix}.oracle_target_iou")
        _strict_number(
            result.get("oracle_target_threshold_parameter"),
            f"{prefix}.oracle_target_threshold_parameter",
        )
        return _strict_iou(result.get("selected_iou"), f"{prefix}.selected_iou")

    if (
        result.get("threshold_and_candidate_policy")
        != "maximize_reference_mask_iou_per_scene"
    ):
        raise AggregationError(f"SPIn calibration policy mismatch in {result_path}")
    _strict_number(
        result.get("selected_threshold_parameter"),
        f"{prefix}.selected_threshold_parameter",
    )
    candidate = result.get("selected_sam_candidate")
    if type(candidate) is not int or candidate not in {0, 1, 2}:
        raise AggregationError(
            f"SPIn selected_sam_candidate must be 0, 1, or 2 in {result_path}"
        )
    frame_results = result.get("frame_results")
    if not isinstance(frame_results, list) or len(frame_results) < 2:
        raise AggregationError(f"SPIn frame_results are incomplete in {result_path}")
    frames: list[str] = []
    target_ious: list[float] = []
    for index, row in enumerate(frame_results):
        if not isinstance(row, dict):
            raise AggregationError(f"SPIn frame row {index} is invalid in {result_path}")
        frame = row.get("frame")
        expected_role = "reference" if index == 0 else "target"
        if (
            not isinstance(frame, str)
            or not frame
            or row.get("role") != expected_role
        ):
            raise AggregationError(
                f"SPIn frame role/name mismatch at row {index} in {result_path}"
            )
        frames.append(frame)
        frame_iou = _strict_iou(
            row.get("foreground_iou"),
            f"{prefix}.frame_results[{index}].foreground_iou",
        )
        if index > 0:
            target_ious.append(frame_iou)
    if len(set(frames)) != len(frames) or frames[0] != reference_frame:
        raise AggregationError(f"SPIn frame mapping mismatch in {result_path}")
    recomputed_mean = mean(target_ious)
    recorded_mean = _strict_iou(
        result.get("scene_mean_iou"),
        f"{prefix}.scene_mean_iou",
    )
    if not math.isclose(recorded_mean, recomputed_mean, abs_tol=1e-12, rel_tol=1e-12):
        raise AggregationError(
            f"SPIn scene mean does not match target frame_results in {result_path}"
        )
    return recorded_mean


def _verified_upstream_patch_provenance(manifest: dict[str, Any]) -> bool:
    provenance = manifest.get("upstream_patch_provenance")
    if provenance is None:
        return False
    if not isinstance(provenance, dict):
        raise AggregationError("upstream_patch_provenance must be a mapping")
    patch_sha256 = provenance.get("patch_sha256")
    approved_file_hashes = APPROVED_UPSTREAM_PATCHED_FILE_SHA256.get(
        patch_sha256
    )
    if (
        approved_file_hashes is None
        or provenance.get("tracked_diff_sha256") != patch_sha256
        or provenance.get("patched_file_sha256") != approved_file_hashes
        or provenance.get("staged_tracked_changes") is not False
        or provenance.get("other_tracked_changes") is not False
    ):
        raise AggregationError("LUDVIG reproduction patch provenance mismatch")
    return True


def _verify_released_training_provenance(
    manifest: dict,
    manifest_path: Path,
    checkpoint_hash: str,
) -> str:
    provenance = manifest.get("released_all_view_training_provenance")
    if not isinstance(provenance, dict) or provenance.get("verified") is not True:
        raise AggregationError(
            "released_all_view run lacks verified training provenance in "
            f"{manifest_path}"
        )
    training_path = Path(str(provenance.get("training_manifest", "")))
    if not training_path.is_file():
        raise AggregationError(
            f"Missing released-all-view training manifest: {training_path}"
        )
    training_hash = _sha256(training_path)
    if provenance.get("training_manifest_sha256") != training_hash:
        raise AggregationError(
            f"Training-manifest hash changed after launch: {training_path}"
        )
    training = json.loads(training_path.read_text(encoding="utf-8"))
    source = training.get("source_provenance", {})
    protocol = training.get("effective_training_protocol", {})
    output = training.get("training_output", {})
    geometry_scene = manifest.get("geometry_scene")
    if not isinstance(geometry_scene, str) or not geometry_scene:
        raise AggregationError(
            f"Missing verified geometry_scene in {manifest_path}"
        )
    training_geometry_scene = training.get("geometry_scene")
    legacy_geometry_scene_fallback = provenance.get(
        "legacy_geometry_scene_fallback"
    )
    cross_benchmark_reuse = provenance.get("cross_benchmark_asset_reuse")
    staging = manifest.get("colmap_staging")
    staging_sha256 = (
        hashlib.sha256(
            json.dumps(
                staging,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if isinstance(staging, dict)
        else None
    )
    verified_cross_benchmark_reuse = (
        isinstance(cross_benchmark_reuse, dict)
        and cross_benchmark_reuse.get("verified") is True
        and cross_benchmark_reuse.get("training_benchmark") == "NVOS"
        and cross_benchmark_reuse.get("evaluation_benchmark") == "SPIn-NeRF"
        and cross_benchmark_reuse.get("geometry_scene") == geometry_scene
        and geometry_scene in SPIN_NVOS_SHARED_LLFF_GEOMETRIES
        and cross_benchmark_reuse.get("colmap_staging_sha256") == staging_sha256
        and isinstance(staging, dict)
        and staging.get("strategy")
        == "reuse_verified_identical_llff_colmap_undistortion"
        and staging.get("raw_scene_identity_proven") is True
        and manifest.get("benchmark") == "SPIn-NeRF"
        and training.get("benchmark") == "NVOS"
    )
    if training_geometry_scene is None:
        if (
            legacy_geometry_scene_fallback is True
            and manifest.get("scene") == "fern"
            and geometry_scene == "fern"
            and training.get("benchmark") == "NVOS"
            and training.get("scene") == "fern"
            and (
                manifest.get("benchmark") == "NVOS"
                or verified_cross_benchmark_reuse
            )
        ):
            training_geometry_scene = "fern"
        else:
            raise AggregationError(
                "Training manifest must explicitly record geometry_scene; "
                "legacy fallback is restricted to the bound NVOS fern run in "
                f"{training_path}"
            )
    elif legacy_geometry_scene_fallback is True:
        raise AggregationError(
            "Run provenance claims a legacy geometry_scene fallback for a "
            f"manifest that already records it: {training_path}"
        )
    if training_geometry_scene != geometry_scene:
        raise AggregationError(
            "Training geometry_scene does not match the evaluated geometry in "
            f"{training_path}"
        )
    required = {
        "status": "complete",
        "method": "original-3DGS",
        "geometry_protocol": "released_all_view",
        "scene": geometry_scene,
    }
    for key, expected in required.items():
        if training.get(key) != expected:
            raise AggregationError(
                f"Training manifest has incompatible {key} in {training_path}"
            )
    if (
        source.get("commit") != OFFICIAL_3DGS_COMMIT
        or source.get("ludvig_commit") != LUDVIG_COMMIT
    ):
        raise AggregationError(
            f"Training source provenance is not pinned in {training_path}"
        )
    expected_protocol = {
        "held_out_training_views": 0,
        "eval_split_enabled": False,
        "iterations": 30000,
        "resolution_argument": -1,
        "algorithm_source_modified": False,
    }
    if any(
        type(protocol.get(key)) is not type(value) or protocol.get(key) != value
        for key, value in expected_protocol.items()
    ):
        raise AggregationError(
            f"Training protocol is not exact released-all-view in {training_path}"
        )
    registered_views = protocol.get("registered_training_views")
    training_benchmark = training.get("benchmark")
    if training_benchmark == "NVOS":
        expected_registered_views = NVOS_GEOMETRY_REGISTERED_IMAGES.get(
            geometry_scene
        )
    elif training_benchmark == "SPIn-NeRF":
        expected_registered_views = SPIN_GEOMETRY_REGISTERED_IMAGES.get(
            geometry_scene
        )
    else:
        raise AggregationError(
            f"Unsupported training benchmark in {training_path}: "
            f"{training_benchmark}"
        )
    if (
        training_benchmark != manifest.get("benchmark")
        and not verified_cross_benchmark_reuse
    ):
        raise AggregationError(
            "Training benchmark does not match the evaluated benchmark in "
            f"{training_path}"
        )
    if (
        training_benchmark == manifest.get("benchmark")
        and cross_benchmark_reuse is not None
    ):
        raise AggregationError(
            "Same-benchmark run carries unexpected cross-benchmark reuse "
            f"provenance in {manifest_path}"
        )
    if expected_registered_views is None:
        raise AggregationError(
            "Training geometry is outside the frozen benchmark asset contract "
            f"in {training_path}: {geometry_scene}"
        )
    if (
        type(registered_views) is not int
        or registered_views != expected_registered_views
    ):
        raise AggregationError(
            "Training camera count does not match the frozen asset contract in "
            f"{training_path}: expected {expected_registered_views}, found "
            f"{registered_views}"
        )
    cfg_args = output.get("cfg_args", {})
    if (
        output.get("point_cloud_sha256") != checkpoint_hash
        or provenance.get("point_cloud_sha256") != checkpoint_hash
        or type(output.get("registered_all_view_cameras")) is not int
        or output.get("registered_all_view_cameras") != registered_views
        or output.get("target_rgb_visible_during_training") is not True
        or cfg_args.get("eval") is not False
        or type(cfg_args.get("resolution")) is not int
        or cfg_args.get("resolution") != -1
    ):
        raise AggregationError(
            f"Training output does not bind the evaluated checkpoint: {training_path}"
        )
    return training_hash


def _paper_context(registry: dict) -> dict:
    return next(
        row
        for row in registry["reported_context"]
        if row["method_id"] == "marrie_et_al_iccv_2025_ludvig_sam"
    )


def aggregate(input_root: Path, benchmark: str) -> dict:
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    context = _paper_context(registry)
    records: dict[str, dict[int, float]] = defaultdict(dict)
    checkpoint_hashes: dict[str, set[str]] = defaultdict(set)
    training_manifest_hashes: dict[str, set[str]] = defaultdict(set)
    geometry_checkpoint_hashes: dict[str, set[str]] = defaultdict(set)
    geometry_training_manifest_hashes: dict[str, set[str]] = defaultdict(set)
    upstream_patch_verified: dict[str, dict[int, bool]] = defaultdict(dict)
    live_file_hashes: dict[Path, str] = {}
    manifests: list[dict] = []

    def live_sha256(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in live_file_hashes:
            live_file_hashes[resolved] = _sha256(resolved)
        return live_file_hashes[resolved]

    for manifest_path in sorted(input_root.rglob("run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        expected = "NVOS" if benchmark == "nvos" else "SPIn-NeRF"
        if manifest.get("benchmark") != expected:
            continue
        if manifest.get("protocol_id") != "ludvig_official_online_multiview_v1":
            raise AggregationError(f"Unexpected protocol in {manifest_path}")
        if (
            manifest.get("method") != "LUDVIG-SAM"
            or manifest.get("upstream_commit") != LUDVIG_COMMIT
            or manifest.get("sam_checkpoint_sha256")
            != SAM_VIT_H_CHECKPOINT_SHA256
        ):
            raise AggregationError(
                f"Unpinned LUDVIG-SAM implementation in {manifest_path}"
            )
        patch_provenance_verified = _verified_upstream_patch_provenance(manifest)
        if manifest.get("strict_unseen_exact_match") is not False:
            raise AggregationError(f"Unsafe strict-unseen label in {manifest_path}")
        geometry_protocol = manifest.get("geometry_protocol")
        if geometry_protocol not in {
            "released_all_view",
            "strict_geometry_hybrid_diagnostic",
        }:
            raise AggregationError(
                f"Unknown geometry protocol in {manifest_path}: "
                f"{geometry_protocol}"
            )
        expected_training_visibility = geometry_protocol == "released_all_view"
        task_scene = manifest.get("scene")
        if expected == "NVOS":
            expected_geometry_scene = NVOS_TASK_TO_GEOMETRY_SCENE.get(task_scene)
            if expected_geometry_scene is None:
                raise AggregationError(
                    f"Unknown NVOS task/geometry mapping in {manifest_path}"
                )
        else:
            expected_geometry_scene = task_scene
        declared_geometry_scene = manifest.get("geometry_scene")
        if geometry_protocol == "released_all_view":
            if declared_geometry_scene != expected_geometry_scene:
                raise AggregationError(
                    "Released-all-view task/geometry_scene mapping mismatch in "
                    f"{manifest_path}: expected {expected_geometry_scene}, "
                    f"found {declared_geometry_scene}"
                )
        elif (
            declared_geometry_scene is not None
            and declared_geometry_scene != expected_geometry_scene
        ):
            raise AggregationError(
                "Task/geometry_scene mapping mismatch in "
                f"{manifest_path}: expected {expected_geometry_scene}, "
                f"found {declared_geometry_scene}"
            )
        if (
            manifest.get(
                "target_rgb_visible_during_gaussian_splatting_training"
            )
            is not expected_training_visibility
        ):
            raise AggregationError(
                "Geometry label/target-training visibility mismatch in "
                f"{manifest_path}"
            )
        if (
            manifest.get("target_rgb_visible_during_uplifting") is not True
            or manifest.get("target_view_2d_foundation_model_calls") is not True
            or manifest.get("target_masks_scoring_only") is not True
        ):
            raise AggregationError(
                f"Released online-multiview visibility mismatch in {manifest_path}"
            )
        expected_reference_calibration = benchmark == "spin"
        expected_aggregation = (
            "equal_weight_macro_over_8_tasks"
            if benchmark == "nvos"
            else "frame_mean_then_equal_weight_macro_over_10_scenes"
        )
        if (
            manifest.get("reference_mask_calibration")
            is not expected_reference_calibration
            or manifest.get("aggregation") != expected_aggregation
        ):
            raise AggregationError(
                f"Benchmark-specific calibration/aggregation mismatch in "
                f"{manifest_path}"
            )
        expected_scenes = set(
            context["published_per_scene_iou"][
                "nvos" if benchmark == "nvos" else "spin_nerf"
            ]
        )
        declared_cohort = manifest.get("cohort")
        if (
            not isinstance(declared_cohort, list)
            or len(declared_cohort) != len(expected_scenes)
            or set(declared_cohort) != expected_scenes
            or manifest.get("scene") not in expected_scenes
        ):
            raise AggregationError(
                f"Benchmark cohort mismatch in {manifest_path}"
            )
        checkpoint_hash = manifest.get("gs_source_sha256")
        if not isinstance(checkpoint_hash, str) or not checkpoint_hash:
            raise AggregationError(
                f"Missing gs_source_sha256 in {manifest_path}"
            )
        checkpoint_path = Path(str(manifest.get("gs_source", "")))
        if (
            not checkpoint_path.is_file()
            or live_sha256(checkpoint_path) != checkpoint_hash
        ):
            raise AggregationError(
                f"gs_source is missing or changed after launch in {manifest_path}"
            )
        sam_checkpoint_path = Path(str(manifest.get("sam_checkpoint", "")))
        if (
            not sam_checkpoint_path.is_file()
            or live_sha256(sam_checkpoint_path) != SAM_VIT_H_CHECKPOINT_SHA256
        ):
            raise AggregationError(
                f"SAM checkpoint is missing or changed after launch in {manifest_path}"
            )
        if geometry_protocol == "released_all_view":
            training_manifest_hash = _verify_released_training_provenance(
                manifest,
                manifest_path,
                checkpoint_hash,
            )
            training_manifest_hashes[manifest["scene"]].add(
                training_manifest_hash
            )
            geometry_checkpoint_hashes[declared_geometry_scene].add(
                checkpoint_hash
            )
            geometry_training_manifest_hashes[declared_geometry_scene].add(
                training_manifest_hash
            )
        elif manifest.get("released_all_view_training_provenance") is not None:
            raise AggregationError(
                f"Hybrid run carries released-all-view provenance: {manifest_path}"
            )
        result_paths = list(manifest_path.parent.rglob("protocol_result.json"))
        if len(result_paths) != 1:
            raise AggregationError(
                f"Expected exactly one result below {manifest_path.parent}; "
                f"found {len(result_paths)}"
            )
        result = json.loads(result_paths[0].read_text(encoding="utf-8"))
        scene = manifest["scene"]
        score = _validate_result(
            result,
            benchmark=benchmark,
            scene=scene,
            result_path=result_paths[0],
        )
        seed = manifest.get("seed")
        if type(seed) is not int:
            raise AggregationError(f"Missing integer seed in {manifest_path}")
        if seed in records[scene]:
            raise AggregationError(
                f"Duplicate completed scene/seed {scene}/{seed}; paper run "
                f"aggregation requires unique seeds"
            )
        records[scene][seed] = 100.0 * score
        checkpoint_hashes[scene].add(checkpoint_hash)
        upstream_patch_verified[scene][seed] = patch_provenance_verified
        manifests.append(manifest)
    if not records:
        raise AggregationError(f"No completed {benchmark} results under {input_root}")
    geometry_protocols = {
        manifest.get("geometry_protocol") for manifest in manifests
    }
    if len(geometry_protocols) != 1 or None in geometry_protocols:
        raise AggregationError(
            "All aggregated runs must declare one identical geometry_protocol; "
            f"found {sorted(str(item) for item in geometry_protocols)}"
        )
    geometry_protocol = next(iter(geometry_protocols))
    inconsistent_checkpoints = {
        scene: sorted(hashes)
        for scene, hashes in checkpoint_hashes.items()
        if len(hashes) != 1
    }
    if inconsistent_checkpoints:
        raise AggregationError(
            "All seeds of a scene must evaluate the same 3DGS checkpoint; "
            f"found {inconsistent_checkpoints}"
        )
    inconsistent_geometry_checkpoints = {
        scene: sorted(hashes)
        for scene, hashes in geometry_checkpoint_hashes.items()
        if len(hashes) != 1
    }
    inconsistent_geometry_training_manifests = {
        scene: sorted(hashes)
        for scene, hashes in geometry_training_manifest_hashes.items()
        if len(hashes) != 1
    }
    if inconsistent_geometry_checkpoints or inconsistent_geometry_training_manifests:
        raise AggregationError(
            "All tasks sharing a released-all-view geometry must bind the same "
            "checkpoint and training manifest; found checkpoints="
            f"{inconsistent_geometry_checkpoints}, training_manifests="
            f"{inconsistent_geometry_training_manifests}"
        )

    per_scene = {
        scene: {
            "seeds": sorted(seed_values),
            "run_values_iou_percent": [
                seed_values[seed] for seed in sorted(seed_values)
            ],
            "local_mean_iou_percent": mean(seed_values.values()),
            "paper_iou_percent": context["published_per_scene_iou"][
                "nvos" if benchmark == "nvos" else "spin_nerf"
            ][scene],
            "delta_local_minus_paper": mean(seed_values.values())
            - context["published_per_scene_iou"][
                "nvos" if benchmark == "nvos" else "spin_nerf"
            ][scene],
            "num_runs": len(seed_values),
            "gs_source_sha256": next(iter(checkpoint_hashes[scene])),
            "training_manifest_sha256": (
                next(iter(training_manifest_hashes[scene]))
                if training_manifest_hashes[scene]
                else None
            ),
            "upstream_patch_provenance_verified": all(
                upstream_patch_verified[scene].values()
            ),
        }
        for scene, seed_values in sorted(records.items())
    }
    local_macro = mean(row["local_mean_iou_percent"] for row in per_scene.values())
    paper_same_scene_macro = mean(row["paper_iou_percent"] for row in per_scene.values())
    expected_scenes = (
        registry["protocols"]["ludvig_official_online_multiview_v1"]["nvos"]["tasks"]
        if benchmark == "nvos"
        else registry["protocols"][
            "ludvig_spin_nerf_9scene_without_fork_diagnostic_v1"
        ]["scenes"]
    )
    missing = sorted(set(expected_scenes) - set(per_scene))
    extra = sorted(set(per_scene) - set(expected_scenes))
    complete_cohort = not missing and not extra
    all_scenes_have_three_runs = all(
        row["num_runs"] == 3 for row in per_scene.values()
    )
    required_paper_seeds = [0, 1, 2]
    all_scenes_have_required_seeds = all(
        row["seeds"] == required_paper_seeds for row in per_scene.values()
    )
    released_all_view_geometry = geometry_protocol == "released_all_view"
    inconsistent_training_manifests = {
        scene: sorted(hashes)
        for scene, hashes in training_manifest_hashes.items()
        if hashes and len(hashes) != 1
    }
    if inconsistent_training_manifests:
        raise AggregationError(
            "All seeds of a scene must bind the same training manifest; "
            f"found {inconsistent_training_manifests}"
        )
    released_training_provenance_verified = (
        released_all_view_geometry
        and set(training_manifest_hashes) == set(per_scene)
        and all(len(hashes) == 1 for hashes in training_manifest_hashes.values())
    )
    all_runs_have_verified_upstream_patch = all(
        verified
        for scene_values in upstream_patch_verified.values()
        for verified in scene_values.values()
    )
    per_scene_three_seed_check = (
        released_training_provenance_verified
        and all_runs_have_verified_upstream_patch
        and all_scenes_have_required_seeds
    )
    for row in per_scene.values():
        row["eligible_for_three_seed_paper_protocol_check"] = (
            released_training_provenance_verified
            and all_runs_have_verified_upstream_patch
            and row["seeds"] == required_paper_seeds
        )
    summary = {
        "schema_version": 2,
        "method": "LUDVIG-SAM",
        "benchmark": "NVOS" if benchmark == "nvos" else "SPIn-NeRF",
        "protocol_id": "ludvig_official_online_multiview_v1",
        "geometry_protocol": geometry_protocol,
        "metric": "foreground_iou",
        "metric_source": (
            "selected_iou_fixed_threshold"
            if benchmark == "nvos"
            else "scene_mean_iou_after_reference_calibration"
        ),
        "oracle_values_aggregated": False,
        "aggregation": "seed_mean_per_scene_then_equal_weight_scene_macro",
        "strict_unseen_exact_match": False,
        "cohort": list(per_scene),
        "expected_cohort": expected_scenes,
        "missing_scenes": missing,
        "extra_scenes": extra,
        "complete_requested_cohort": complete_cohort,
        "per_scene": per_scene,
        "local_scene_macro_iou_percent": local_macro,
        "paper_same_scene_macro_iou_percent": paper_same_scene_macro,
        "delta_local_minus_paper": local_macro - paper_same_scene_macro,
        "paper_full_benchmark_iou_percent": context["values"][
            "nvos" if benchmark == "nvos" else "spin_nerf"
        ]["miou"],
        "eligible_for_full_cohort_single_run_report": complete_cohort,
        "eligible_for_full_cohort_three_seed_hybrid_report": (
            geometry_protocol == "strict_geometry_hybrid_diagnostic"
            and complete_cohort
            and all_scenes_have_required_seeds
        ),
        "eligible_for_paper_protocol_comparison": (
            benchmark == "nvos"
            and complete_cohort
            and all_scenes_have_required_seeds
            and released_training_provenance_verified
            and all_runs_have_verified_upstream_patch
        ),
        "eligible_for_per_scene_three_seed_paper_protocol_check": (
            per_scene_three_seed_check
        ),
        "eligible_for_strict_unseen_claim": False,
        "paper_requires_three_runs": True,
        "required_paper_seeds": required_paper_seeds,
        "all_scenes_have_three_runs": all_scenes_have_three_runs,
        "all_scenes_have_required_seeds": all_scenes_have_required_seeds,
        "released_all_view_geometry": released_all_view_geometry,
        "released_all_view_training_provenance_verified": (
            released_training_provenance_verified
        ),
        "all_runs_have_verified_upstream_patch": (
            all_runs_have_verified_upstream_patch
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("nvos", "spin"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = aggregate(args.input_root, args.benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
