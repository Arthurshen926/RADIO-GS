#!/usr/bin/env python3
"""Materialize one strict-unseen NVOS scene as the frozen Method-v1 field.

The queue opens only the source RGB cohort declared by the frozen NVOS
authority.  The evaluation target is removed by exact basename before RADIO
or crop-summary extraction, and no masks, scribbles, queries, or target metric
files are opened here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable

import yaml

from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    sha256_file,
    write_frozen_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RADIO_CHECKPOINT = Path(
    "/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar"
)
RADIO_CHECKPOINT_SHA256 = (
    "bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9"
)
NVOS_AUTHORITY = (
    REPO_ROOT
    / "paper/artifacts/nvos_canonical_mpr_v3_strict_unseen_exact_authority_20260802.json"
)
METHOD_AUTHORITY = (
    REPO_ROOT / "paper/artifacts/five_benchmark_method_v1_authority_20260815.json"
)
DATASET_MANIFEST = Path(
    "/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/"
    "nvos_strict_unseen_v1.json"
)
DEFAULT_RUN_ROOT = Path(
    "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/"
    "core_method_v1/nvos"
)
REGION_BRIDGE = Path(
    "/mnt/pool/sqy/results/RADIO-GS/output/semantic_bridge/"
    "global_region_summary_coco15000_full_context_local_scales_imageholdout.pth"
)
GENERIC_RELATION_AUTHORITY = Path(
    "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/"
    "target_blind_typed_text_relation_authority_v1/fit_relation_indices.pt"
)
GENERIC_RELATION_AUTHORITY_SHA256 = (
    "482e363bf31884e190b255cc0cf0996461400bcbb3cb3f8785fbc236da2702a9"
)
STAGES = (
    "source_features",
    "validation_plan",
    "factorized_mpr",
    "exact_raw_mpr",
    "exact_dino_mpr",
    "exact_sam_mpr",
    "capability_authority",
    "base_field",
    "crop_teacher",
    "siglip_stage",
    "region_stage",
    "generic_stage",
    "method_gate",
)
GPU_THERMAL_ENV = {
    "GPU_MAX_POWER_LIMIT_W": "300.5",
    "GPU_POLL_SECONDS": "3",
    "GPU_START_MAX_TEMP_C": "78",
    "GPU_SOFT_PAUSE_TEMP_C": "76",
    # A 6 C hysteresis is sufficient with the 3 s poll cadence. Requiring
    # 68 C can deadlock the second card near 70 C under dual-GPU heat soak even
    # while both guarded children are stopped.
    "GPU_SOFT_RESUME_TEMP_C": "70",
    "GPU_MAX_TEMP_C": "84",
    "GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES": "3",
    "GPU_OWNER_PID_NAMESPACE_MODE": "exclusive-singleton-after-clear-v1",
}


@dataclass(frozen=True)
class SceneAssets:
    scene: str
    base_config: Path
    geometry: Path
    geometry_sha256: str
    image_dir: Path
    exclusion_file: Path
    excluded_stems: tuple[str, ...]
    training_frame_count: int
    feature_height: int
    feature_width: int
    resolution_scale: float


def _authority_scene_records() -> dict[str, dict]:
    authority, _digest, _path = load_json_object(
        NVOS_AUTHORITY,
        label="frozen NVOS exact authority",
    )
    expected_manifest = authority["protocol"]["dataset_manifest"]
    if Path(expected_manifest["path"]).resolve() != DATASET_MANIFEST.resolve():
        raise ValueError("NVOS authority names another dataset manifest")
    if sha256_file(DATASET_MANIFEST) != expected_manifest["sha256"]:
        raise ValueError("frozen NVOS dataset manifest SHA-256 differs")
    return {str(row["scene_id"]): row for row in authority["scenes"]}


def resolve_scene_assets(scene: str) -> SceneAssets:
    records = _authority_scene_records()
    if scene not in records:
        raise ValueError(f"scene is outside frozen NVOS full8: {scene}")
    row = records[scene]
    dataset, _digest, _path = load_json_object(
        DATASET_MANIFEST,
        label="frozen NVOS dataset manifest",
    )
    dataset_rows = {str(value["scene_id"]): value for value in dataset["scenes"]}
    dataset_row = dataset_rows[scene]
    base_config = Path(row["config"]["path"]).resolve()
    geometry = Path(row["geometry_carrier_checkpoint"]["path"]).resolve()
    if sha256_file(base_config) != row["config"]["sha256"]:
        raise ValueError(f"{scene} frozen config SHA-256 differs")
    if sha256_file(geometry) != row["geometry_carrier_checkpoint"]["sha256"]:
        raise ValueError(f"{scene} frozen geometry SHA-256 differs")
    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    scene_root = base_config.parent
    exclusion_file = scene_root / "excluded_image_stems.json"
    exclusion, _exclusion_digest, _source = load_json_object(
        exclusion_file,
        label=f"{scene} exact target-RGB exclusion",
    )
    excluded_stems = tuple(str(value) for value in exclusion["excluded_image_stems"])
    if list(excluded_stems) != list(dataset_row["excluded_training_frame_ids"]):
        raise ValueError(f"{scene} target-RGB exclusion authority differs")
    if dataset_row.get("target_rgb_policy") != (
        "excluded_from_field_training_and_query"
    ):
        raise ValueError(f"{scene} target RGB policy differs")
    return SceneAssets(
        scene=scene,
        base_config=base_config,
        geometry=geometry,
        geometry_sha256=row["geometry_carrier_checkpoint"]["sha256"],
        image_dir=Path(dataset_row["rgb_directory"]).resolve(),
        exclusion_file=exclusion_file,
        excluded_stems=excluded_stems,
        training_frame_count=len(dataset_row["training_frames"]),
        feature_height=int(config["feature_height"]),
        feature_width=int(config["feature_width"]),
        resolution_scale=0.25,
    )


def _write_exact_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"existing generated file differs: {path}")
        return
    path.write_text(text, encoding="utf-8")


def write_runtime_config(assets: SceneAssets, feature_dir: Path, output: Path) -> None:
    text = (
        f"base_config: {assets.base_config}\n"
        f"feature_dir: {feature_dir}\n"
        f"val_feature_dir: {feature_dir}\n"
    )
    _write_exact_text(output, text)


def build_mpr_common_args(
    *,
    config: Path,
    assets: SceneAssets,
    feature_bundle_sha256: str,
    validation_csv: str,
) -> list[str | Path]:
    """Return the exact-marginal MPR policy shared by every capability cache."""

    return [
        "--config",
        config,
        "--checkpoint",
        assets.geometry,
        "--device",
        "cuda:0",
        "--exclude-frame-ids",
        validation_csv,
        "--expected-feature-scene",
        assets.scene,
        "--expected-feature-image-dir",
        assets.image_dir,
        "--expected-geometry-checkpoint-sha256",
        assets.geometry_sha256,
        "--expected-feature-output-bundle-sha256",
        feature_bundle_sha256,
        "--max-views",
        "120",
        "--alpha-threshold",
        "0",
        "--aggregation-mode",
        "raster_marginal_responsibility",
        "--registration-weight-mode",
        "alpha_depth",
        "--raster-view-fusion",
        "contribution_mean",
        "--no-robust-mpr",
        "--raster-channel-chunk-size",
        "128",
    ]


def _run(
    args: argparse.Namespace,
    stage: str,
    command: Iterable[str],
    *,
    gpu: bool,
    log_dir: Path,
) -> None:
    log_path = log_dir / f"{stage}.log"
    command = [str(value) for value in command]
    if gpu:
        environment = os.environ.copy()
        environment.update(
            {
                **GPU_THERMAL_ENV,
                "GPU": str(args.gpu),
                "GPU_TELEMETRY_LOG": str(log_dir / f"gpu{args.gpu}_telemetry.csv"),
                "GPU_OWNER_AUDIT_LOG": str(log_dir / f"gpu{args.gpu}_owner.csv"),
            }
        )
        full_command = [
            "bash",
            str(REPO_ROOT / "radio_gs/scripts/run_with_gpu_thermal_guard.sh"),
            "--",
            "env",
            f"CUDA_VISIBLE_DEVICES={args.gpu}",
            "bash",
            str(REPO_ROOT / "radio_gs/scripts/run_repo_python.sh"),
            *command,
        ]
    else:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        full_command = [
            "bash",
            str(REPO_ROOT / "radio_gs/scripts/run_repo_python.sh"),
            *command,
        ]
    print(f"{args.scene}: running {stage}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(full_command) + "\n")
        log.flush()
        subprocess.run(
            full_command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _require_complete(primary: Path, companions: Iterable[Path], stage: str) -> bool:
    paths = [primary, *companions]
    present = [path.is_file() and path.stat().st_size > 0 for path in paths]
    if all(present):
        return True
    if any(present):
        raise RuntimeError(f"partial {stage} output must be audited: {paths}")
    return False


def _feature_bundle(feature_dir: Path, assets: SceneAssets) -> str:
    manifest, _digest, _source = load_json_object(
        feature_dir / "frame_manifest.json",
        label=f"{assets.scene} source-only RADIO feature manifest",
    )
    radio = manifest.get("radio", {})
    if (
        manifest.get("scene") != assets.scene
        or Path(manifest.get("image_dir", "")).resolve() != assets.image_dir
        or int(manifest.get("num_frames", -1)) != assets.training_frame_count
        or tuple(manifest.get("excluded_image_stems", [])) != assets.excluded_stems
        or manifest.get("frame_id_mode") != "source_rank"
        or radio.get("checkpoint_sha256") != RADIO_CHECKPOINT_SHA256
        or radio.get("checkpoint_provenance") != "explicit_file_sha256"
        or radio.get("requested_adaptors") != ["siglip2-g"]
    ):
        raise ValueError(f"{assets.scene} source-only feature authority differs")
    features = manifest.get("features", {})
    expected_grid = [assets.feature_height, assets.feature_width]
    if (
        float(manifest.get("resolution_scale", -1.0)) != assets.resolution_scale
        or features.get("backbone", {}).get("grid") != expected_grid
    ):
        raise ValueError(
            f"{assets.scene} source feature grid differs from frozen config"
        )
    adaptors = features.get("adaptors", [])
    if not any(
        row.get("name") == "siglip2-g"
        and row.get("subdir") == "siglip2"
        and row.get("grid") == expected_grid
        for row in adaptors
    ):
        raise ValueError(f"{assets.scene} has no official SigLIP2 grid")
    bundle = str(manifest.get("output_bundle_sha256", ""))
    if len(bundle) != 64:
        raise ValueError(f"{assets.scene} feature bundle is not sealed")
    return bundle


def _validation_csv(path: Path) -> str:
    payload, _digest, _source = load_json_object(
        path,
        label="source-only fidelity validation plan",
    )
    values = [int(value) for value in payload["validation_frame_ids"]]
    if len(values) != 4:
        raise ValueError("Method-v1 requires four fidelity validation frames")
    return ",".join(str(value) for value in values)


def run(args: argparse.Namespace) -> dict:
    assets = resolve_scene_assets(args.scene)
    run_root = Path(args.run_root).expanduser().resolve() / assets.scene
    feature_dir = run_root / "source_only_features"
    config = run_root / "method_v1.yaml"
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    write_runtime_config(assets, feature_dir, config)

    validation_plan = run_root / "fidelity_validation_frames.json"
    responsibility = run_root / "exact_marginal_responsibility_heldout4.json"
    factorized = run_root / "factorized_raw_radio_heldout4.pt"
    exact_raw = run_root / "raw_radio_matched_exact_marginal_heldout4.pt"
    exact_dino = run_root / "dino_v3_matched_exact_marginal_heldout4.pt"
    exact_sam = run_root / "sam3_matched_exact_marginal_heldout4.pt"
    cohort = run_root / "method_v1_capability_cohort_authority.json"
    base_field = run_root / "factorized_d512_l512_heldout4.pth"
    crop_root = run_root / "genuine_crop_summary_teacher"
    siglip_field = run_root / "official_siglip2_spatial_w005_s0_64.pth"
    region_field = run_root / "official_siglip2_spatial_region_w005_s0_64.pth"
    final_field = run_root / "generic_text_response_w005_s0_64.pth"

    stop_index = STAGES.index(args.stop_after)
    if not (feature_dir / "frame_manifest.json").is_file():
        _run(
            args,
            "source_features",
            [
                "radio_gs/scripts/extract_radio_features.py",
                "--scene",
                assets.scene,
                "--image_dir",
                assets.image_dir,
                "--output_dir",
                feature_dir,
                "--radio_repo",
                "/root/RADIO",
                "--radio_version",
                "c-radio_v4-h",
                "--radio_checkpoint",
                RADIO_CHECKPOINT,
                "--batch_size",
                "1",
                "--frame-id-mode",
                "source_rank",
                "--exclude-image-stems-file",
                assets.exclusion_file,
                "--extract_adaptors",
                "--adaptor_names",
                "siglip2-g",
                "--resolution_scale",
                str(assets.resolution_scale),
                "--device",
                "cuda",
                "--amp",
                "--resume-partial",
                "--skip_pca_stats",
            ],
            gpu=True,
            log_dir=logs,
        )
    feature_bundle = _feature_bundle(feature_dir, assets)
    if stop_index == 0:
        return {"scene": assets.scene, "completed_stage": STAGES[0]}

    if not validation_plan.is_file():
        _run(
            args,
            "validation_plan",
            [
                "radio_gs/scripts/select_fidelity_validation_frames.py",
                "--feature-dir",
                feature_dir,
                "--output",
                validation_plan,
                "--views",
                "4",
            ],
            gpu=False,
            log_dir=logs,
        )
    validation_csv = _validation_csv(validation_plan)
    if stop_index == 1:
        return {"scene": assets.scene, "completed_stage": STAGES[1]}

    mpr_common = build_mpr_common_args(
        config=config,
        assets=assets,
        feature_bundle_sha256=feature_bundle,
        validation_csv=validation_csv,
    )
    factorized_report = factorized.with_suffix(factorized.suffix + ".json")
    factorized_parts = [
        factorized.is_file() and factorized.stat().st_size > 0,
        factorized_report.is_file() and factorized_report.stat().st_size > 0,
    ]
    if any(factorized_parts) and not all(factorized_parts):
        raise RuntimeError(
            f"partial factorized_mpr output must be audited: "
            f"{factorized}, {factorized_report}"
        )
    factorized_ready = all(factorized_parts) and responsibility.is_file()
    if not factorized_ready:
        if all(factorized_parts) and not responsibility.is_file():
            raise RuntimeError("factorized MPR lacks its responsibility authority")
        if responsibility.is_file():
            responsibility_args = [
                "--responsibility-cache",
                responsibility,
                "--expected-responsibility-cache-sha256",
                sha256_file(responsibility),
            ]
        else:
            responsibility_args = ["--save-responsibility-cache", responsibility]
        _run(
            args,
            "factorized_mpr",
            [
                "radio_gs/scripts/build_gaussian_multiview_teacher_cache.py",
                *mpr_common,
                "--output",
                factorized,
                "--observation-contract",
                "canonical-factorized-radio-v1",
                "--feature-space",
                "radio",
                *responsibility_args,
            ],
            gpu=True,
            log_dir=logs,
        )
    responsibility_sha = sha256_file(responsibility)
    if stop_index == 2:
        return {"scene": assets.scene, "completed_stage": STAGES[2]}

    exact_common = [
        *mpr_common,
        "--observation-contract",
        "canonical-exact-marginal-mpr-v1",
        "--normalize-each-view",
        "--responsibility-cache",
        responsibility,
        "--expected-responsibility-cache-sha256",
        responsibility_sha,
    ]
    if not _require_complete(
        exact_raw,
        [exact_raw.with_suffix(exact_raw.suffix + ".json")],
        "exact_raw_mpr",
    ):
        _run(
            args,
            "exact_raw_mpr",
            [
                "radio_gs/scripts/build_gaussian_multiview_teacher_cache.py",
                *exact_common,
                "--output",
                exact_raw,
                "--feature-space",
                "radio",
            ],
            gpu=True,
            log_dir=logs,
        )
    if stop_index == 3:
        return {"scene": assets.scene, "completed_stage": STAGES[3]}

    for stage, output, feature_space, shard_channels in (
        ("exact_dino_mpr", exact_dino, "dino_v3", "256"),
        ("exact_sam_mpr", exact_sam, "sam3", "256"),
    ):
        if not _require_complete(
            output,
            [output.with_suffix(output.suffix + ".json")],
            stage,
        ):
            _run(
                args,
                stage,
                [
                    "radio_gs/scripts/build_gaussian_multiview_teacher_cache.py",
                    *exact_common,
                    "--output",
                    output,
                    "--feature-space",
                    feature_space,
                    "--radio-checkpoint",
                    RADIO_CHECKPOINT,
                    "--capability-map-source",
                    "project_raw",
                    "--capability-storage",
                    "channel_sharded",
                    "--capability-shard-channels",
                    shard_channels,
                ],
                gpu=True,
                log_dir=logs,
            )
        if stop_index == STAGES.index(stage):
            return {"scene": assets.scene, "completed_stage": stage}

    exact_digests = {
        "radio": sha256_file(exact_raw),
        "dino_v3": sha256_file(exact_dino),
        "sam3": sha256_file(exact_sam),
    }
    cohort_payload = {
        "schema_version": 1,
        "artifact_type": "factorized_capability_cohort_authority",
        "experiment": "canonical-factorized-radio-v1-formal-capability-cohort",
        "benchmark": "NVOS",
        "scene": assets.scene,
        "feature_output_bundle_sha256": feature_bundle,
        "frozen_cache_authorities": {
            "radio": {"path": str(exact_raw), "sha256": exact_digests["radio"]},
            "dino_v3": {
                "path": str(exact_dino),
                "sha256": exact_digests["dino_v3"],
            },
            "sam3": {"path": str(exact_sam), "sha256": exact_digests["sam3"]},
        },
        "target_access": {
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "target_metrics_used_for_selection": False,
        },
    }
    if cohort.is_file():
        existing, _digest, _source = load_json_object(
            cohort,
            label=f"{assets.scene} Method-v1 capability cohort",
        )
        if existing != cohort_payload:
            raise ValueError(f"{assets.scene} capability cohort authority differs")
    else:
        write_frozen_json(cohort, cohort_payload)
    cohort_sha = sha256_file(cohort)
    if stop_index == 6:
        return {"scene": assets.scene, "completed_stage": STAGES[6]}

    if not _require_complete(
        base_field,
        [base_field.with_suffix(base_field.suffix + ".json")],
        "base_field",
    ):
        _run(
            args,
            "base_field",
            [
                "radio_gs/scripts/train_canonical_radio_field.py",
                "--mpr-cache",
                factorized,
                "--expected-mpr-cache-sha256",
                sha256_file(factorized),
                "--observation-contract",
                "canonical-factorized-radio-v1",
                "--radio-checkpoint",
                RADIO_CHECKPOINT,
                "--expected-radio-checkpoint-sha256",
                RADIO_CHECKPOINT_SHA256,
                "--expected-feature-output-bundle-sha256",
                feature_bundle,
                "--output",
                base_field,
                "--device",
                "cuda:0",
                "--coefficient-dim",
                "512",
                "--local-dim",
                "512",
                "--no-fusion-reliability",
                "--freeze-basis",
                "--basis-fit-device",
                "cuda:0",
                "--official-capability-loss",
                "--capability-target-contract",
                "matched_exact_marginal",
                "--dino-mpr-cache",
                exact_dino,
                "--expected-dino-v3-mpr-cache-sha256",
                exact_digests["dino_v3"],
                "--sam3-mpr-cache",
                exact_sam,
                "--expected-sam3-mpr-cache-sha256",
                exact_digests["sam3"],
                "--factorized-capability-reference-mpr-cache",
                exact_raw,
                "--expected-factorized-capability-reference-mpr-cache-sha256",
                exact_digests["radio"],
                "--factorized-capability-cohort-authority",
                cohort,
                "--expected-factorized-capability-cohort-authority-sha256",
                cohort_sha,
                "--epochs",
                "20",
                "--min-epochs",
                "20",
                "--batch-size",
                "4096",
                "--eval-batch-size",
                "16384",
                "--learning-rate",
                "0.002",
                "--weight-decay",
                "0.00001",
                "--validation-fraction",
                "0.05",
                "--target-cosine",
                "0.985",
                "--seed",
                "0",
            ],
            gpu=True,
            log_dir=logs,
        )
    if stop_index == 7:
        return {"scene": assets.scene, "completed_stage": STAGES[7]}

    crop_manifest = crop_root / assets.scene / "manifest.json"
    if not crop_manifest.is_file():
        _run(
            args,
            "crop_teacher",
            [
                "radio_gs/scripts/extract_official_crop_summary_teacher.py",
                "--dataset-root",
                assets.image_dir.parent,
                "--label-dir",
                assets.image_dir.parent,
                "--image-dir",
                assets.image_dir,
                "--output-root",
                crop_root,
                "--scenes",
                assets.scene,
                "--frames",
                "all",
                "--frame-id-mode",
                "source_rank",
                "--exclude-image-stems-file",
                assets.exclusion_file,
                "--output-size",
                f"{assets.feature_height}x{assets.feature_width}",
                "--radio-checkpoint",
                RADIO_CHECKPOINT,
                "--radio-repo",
                "/root/RADIO",
                "--radio-version",
                "c-radio_v4-h",
                "--device",
                "cuda:0",
            ],
            gpu=True,
            log_dir=logs,
        )
    crop_payload, _digest, _source = load_json_object(
        crop_manifest,
        label=f"{assets.scene} genuine crop-summary manifest",
    )
    if (
        crop_payload.get("frame_id_mode") != "source_rank"
        or tuple(crop_payload.get("excluded_image_stems", []))
        != assets.excluded_stems
        or crop_payload["scenes"][assets.scene]["num_frames"]
        != assets.training_frame_count
        or crop_payload.get("benchmark_masks_opened") is not False
    ):
        raise ValueError(f"{assets.scene} crop-summary source authority differs")
    if stop_index == 8:
        return {"scene": assets.scene, "completed_stage": STAGES[8]}

    finetune_common = [
        "--config",
        config,
        "--geometry-checkpoint",
        assets.geometry,
        "--mpr-cache",
        factorized,
        "--device",
        "cuda:0",
        "--steps",
        "64",
        "--mpr-weight",
        "0.10",
        "--max-mpr-drop",
        "0.0002",
        "--max-validation-drop",
        "0.0002",
        "--max-capability-drop",
        "0.002",
        "--siglip-spatial-render-weight",
        "0.05",
        "--radio-checkpoint",
        RADIO_CHECKPOINT,
        "--capability-map-source",
        "official_extracted",
        "--capability-local-affinity-weight",
        "0.25",
        "--capability-local-radius",
        "1",
        "--capability-local-balance-quantile",
        "0.0",
        "--validation-frame-ids",
        validation_csv,
        "--seed",
        "0",
    ]
    if not _require_complete(
        siglip_field,
        [siglip_field.with_suffix(siglip_field.suffix + ".json")],
        "siglip_stage",
    ):
        _run(
            args,
            "siglip_stage",
            [
                "radio_gs/scripts/finetune_canonical_radio_rendering.py",
                *finetune_common,
                "--field-checkpoint",
                base_field,
                "--output",
                siglip_field,
                "--selection-policy",
                "capability_pareto",
            ],
            gpu=True,
            log_dir=logs,
        )
    if stop_index == 9:
        return {"scene": assets.scene, "completed_stage": STAGES[9]}

    semantic_common = [
        "--semantic-weight",
        "0.05",
        "--semantic-centered-weight",
        "1.0",
        "--semantic-teacher-root",
        crop_root,
        "--semantic-scene",
        assets.scene,
        "--semantic-bridge-checkpoint",
        REGION_BRIDGE,
        "--semantic-kernel-sizes",
        "3,7,15",
    ]
    if not _require_complete(
        region_field,
        [region_field.with_suffix(region_field.suffix + ".json")],
        "region_stage",
    ):
        _run(
            args,
            "region_stage",
            [
                "radio_gs/scripts/finetune_canonical_radio_rendering.py",
                *finetune_common,
                *semantic_common,
                "--field-checkpoint",
                siglip_field,
                "--output",
                region_field,
                "--selection-policy",
                "semantic_capability",
            ],
            gpu=True,
            log_dir=logs,
        )
    if stop_index == 10:
        return {"scene": assets.scene, "completed_stage": STAGES[10]}

    if not _require_complete(
        final_field,
        [final_field.with_suffix(final_field.suffix + ".json")],
        "generic_stage",
    ):
        _run(
            args,
            "generic_stage",
            [
                "radio_gs/scripts/finetune_canonical_radio_rendering.py",
                *finetune_common,
                *semantic_common,
                "--field-checkpoint",
                region_field,
                "--output",
                final_field,
                "--generic-text-response-weight",
                "0.05",
                "--generic-text-relation-authority",
                GENERIC_RELATION_AUTHORITY,
                "--expected-generic-text-relation-authority-sha256",
                GENERIC_RELATION_AUTHORITY_SHA256,
                "--selection-policy",
                "text_response_capability",
            ],
            gpu=True,
            log_dir=logs,
        )
    if stop_index == 11:
        return {"scene": assets.scene, "completed_stage": STAGES[11]}

    gate_report = run_root / "method_v1_gate.json"
    if not gate_report.is_file():
        _run(
            args,
            "method_gate",
            [
                "radio_gs/scripts/validate_five_benchmark_method_v1_field.py",
                "--field",
                final_field,
                "--expected-field-sha256",
                sha256_file(final_field),
                "--authority",
                METHOD_AUTHORITY,
            ],
            gpu=False,
            log_dir=logs,
        )
        # The validator writes JSON to its log; keep a small immutable result
        # that downstream queue/evaluation code can discover without parsing logs.
        write_frozen_json(
            gate_report,
            {
                "status": "pass",
                "benchmark": "NVOS",
                "scene": assets.scene,
                "field": str(final_field),
                "field_sha256": sha256_file(final_field),
                "method_authority": str(METHOD_AUTHORITY),
                "method_authority_sha256": sha256_file(METHOD_AUTHORITY),
            },
        )
    return {
        "scene": assets.scene,
        "completed_stage": STAGES[-1],
        "field": str(final_field),
        "field_sha256": sha256_file(final_field),
        "gate": str(gate_report),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--stop-after", choices=STAGES, default=STAGES[-1])
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
