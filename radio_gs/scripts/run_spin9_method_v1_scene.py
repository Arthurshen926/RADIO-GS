#!/usr/bin/env python3
"""Materialize one frozen SPIn-NeRF Available-Nine Method-v1 field.

The persistent field may use all registered RGB views under the frozen SPIn
protocol, but this queue never opens a reference/evaluation mask, benchmark
query, or target metric.  It deliberately rebuilds official C-RADIO features
at each frozen carrier's registered grid instead of inheriting the historical
D256/L128 caches.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import time
from typing import Iterable

from radio_gs.config import load_config
from radio_gs.scripts.run_nvos_method_v1_scene import (
    GENERIC_RELATION_AUTHORITY,
    GENERIC_RELATION_AUTHORITY_SHA256,
    METHOD_AUTHORITY,
    RADIO_CHECKPOINT,
    RADIO_CHECKPOINT_SHA256,
    REGION_BRIDGE,
    STAGES,
    _require_complete,
    _run,
    _validation_csv,
    build_mpr_common_args,
    write_runtime_config,
)
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    sha256_file,
    write_frozen_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPIN_AUTHORITY = (
    REPO_ROOT
    / "paper/artifacts/spin9_canonical_mpr_v3_exact_local9_authority_20260803.json"
)
SPIN_PREREGISTRATION = (
    REPO_ROOT
    / "paper/artifacts/spin9_method_v1_d512_l512_construction_preregistration_20260816.json"
)
DATASET_MANIFEST = (
    REPO_ROOT / "output/unified_query/manifests/"
    "spin_nerf_full_reference_mask_9scene_diagnostic_v1.json"
)
DEFAULT_RUN_ROOT = Path(
    "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260816/"
    "core_method_v1/spin9"
)
CAPABILITY_SHARD_CHANNELS = 512
HOST_MEMORY_STAGE_LOCK = Path("/tmp/radio_gs_spin9_host_memory_stage.lock")
SCENE_RUN_LOCK_NAME = ".run_spin9_method_v1_scene.lock"
HOST_MEMORY_STAGE_SLOTS_ENV = "RADIO_GS_SPIN9_HOST_MEMORY_SLOTS"
DEFAULT_HOST_MEMORY_STAGE_SLOTS = 2
HOST_MEMORY_HEAVY_STAGES = frozenset(
    {
        "factorized_mpr",
        "exact_raw_mpr",
        "exact_dino_mpr",
        "exact_sam_mpr",
        "base_field",
        "siglip_stage",
        "region_stage",
        "generic_stage",
    }
)


@dataclass(frozen=True)
class SceneAssets:
    scene: str
    base_config: Path
    geometry: Path
    geometry_sha256: str
    image_dir: Path
    training_frame_count: int
    feature_height: int
    feature_width: int
    resolution_scale: float = 1.0


def _image_files(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


@contextmanager
def _host_memory_stage_boundary(
    *, scene: str, stage: str
) -> Iterable[None]:
    """Bound concurrent high-RSS stages without serializing every GPU.

    This host has enough RAM for two measured SPIn Method-v1 stages. Each
    process claims one advisory-lock slot, so GPU4 and GPU5 can make progress
    concurrently while a larger accidental fan-out remains fail-safe.
    """

    if stage not in HOST_MEMORY_HEAVY_STAGES:
        yield
        return
    raw_slots = os.environ.get(
        HOST_MEMORY_STAGE_SLOTS_ENV, str(DEFAULT_HOST_MEMORY_STAGE_SLOTS)
    )
    try:
        slot_count = int(raw_slots)
    except ValueError as exc:
        raise ValueError(f"{HOST_MEMORY_STAGE_SLOTS_ENV} must be an integer") from exc
    if slot_count <= 0 or slot_count > 8:
        raise ValueError(f"{HOST_MEMORY_STAGE_SLOTS_ENV} must be in [1,8]")
    HOST_MEMORY_STAGE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"{scene}: waiting for host-memory stage slot ({stage}; capacity={slot_count})",
        flush=True,
    )
    handle = None
    slot_index = -1
    while handle is None:
        for candidate in range(slot_count):
            path = (
                HOST_MEMORY_STAGE_LOCK
                if candidate == 0
                else Path(f"{HOST_MEMORY_STAGE_LOCK}.{candidate}")
            )
            current = path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(current.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                current.close()
                continue
            handle = current
            slot_index = candidate
            break
        if handle is None:
            time.sleep(1.0)
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps({"scene": scene, "stage": stage, "slot": slot_index}) + "\n"
    )
    handle.flush()
    print(
        f"{scene}: acquired host-memory stage slot {slot_index} ({stage})",
        flush=True,
    )
    try:
        yield
    finally:
        handle.seek(0)
        handle.truncate()
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def _scene_run_boundary(*, run_root: Path, scene: str) -> Iterable[None]:
    """Serialize writers for one run-root/scene without blocking other scenes."""

    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / SCENE_RUN_LOCK_NAME
    handle = lock_path.open("a+", encoding="utf-8")
    # The run root is shared by root-squashed and non-squashed containers.
    # Keep the coordination inode writable across their numeric UID mappings.
    os.fchmod(handle.fileno(), 0o666)
    acquired = False
    announced_wait = False
    try:
        while not acquired:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                if not announced_wait:
                    print(
                        f"{scene}: waiting for per-scene runner lock ({lock_path})",
                        flush=True,
                    )
                    announced_wait = True
                time.sleep(1.0)
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "host": os.uname().nodename,
                    "scene": scene,
                    "run_root": str(run_root),
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        print(
            f"{scene}: acquired per-scene runner lock ({lock_path})",
            flush=True,
        )
        yield
    finally:
        if acquired:
            handle.seek(0)
            handle.truncate()
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            print(
                f"{scene}: released per-scene runner lock ({lock_path})",
                flush=True,
            )
        handle.close()


def _run_spin_stage(
    args: argparse.Namespace,
    stage: str,
    command: Iterable[str | Path],
    *,
    gpu: bool,
    log_dir: Path,
) -> None:
    with _host_memory_stage_boundary(scene=str(args.scene), stage=stage):
        _run(args, stage, command, gpu=gpu, log_dir=log_dir)


def resolve_scene_assets(scene: str) -> SceneAssets:
    authority, _digest, _source = load_json_object(
        SPIN_AUTHORITY, label="frozen SPIn exact authority"
    )
    records = {str(row["scene_id"]): row for row in authority["scenes"]}
    if scene not in records:
        raise ValueError(f"scene is outside frozen SPIn Available-Nine: {scene}")
    prereg, _prereg_sha, _prereg_source = load_json_object(
        SPIN_PREREGISTRATION, label="SPIn Method-v1 D512/L512 preregistration"
    )
    if prereg.get("status") != "frozen_before_first_d512_l512_spin9_execution":
        raise ValueError("SPIn Method-v1 construction preregistration is not frozen")
    dataset, _manifest_sha, _manifest_source = load_json_object(
        DATASET_MANIFEST, label="frozen SPIn Available-Nine manifest"
    )
    dataset_rows = {str(row["scene_id"]): row for row in dataset["scenes"]}
    if scene not in dataset_rows:
        raise ValueError(f"{scene} is absent from the SPIn dataset manifest")

    assets = records[scene]["assets"]
    config = Path(assets["config"]["path"]).resolve(strict=True)
    geometry = Path(assets["geometry_checkpoint"]["path"]).resolve(strict=True)
    if sha256_file(config) != assets["config"]["sha256"]:
        raise ValueError(f"{scene} frozen config SHA-256 differs")
    if sha256_file(geometry) != assets["geometry_checkpoint"]["sha256"]:
        raise ValueError(f"{scene} frozen geometry SHA-256 differs")
    image_dir = Path(dataset_rows[scene]["rgb_directory"]).resolve(strict=True)
    images = _image_files(image_dir)
    if len(images) < 2:
        raise ValueError(f"{scene} has too few registered RGB views")
    loaded = load_config(str(config))
    return SceneAssets(
        scene=scene,
        base_config=config,
        geometry=geometry,
        geometry_sha256=assets["geometry_checkpoint"]["sha256"],
        image_dir=image_dir,
        training_frame_count=len(images),
        feature_height=int(loaded.feature_height),
        feature_width=int(loaded.feature_width),
    )


def source_feature_command(assets: SceneAssets, feature_dir: Path) -> list[str | Path]:
    return [
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
    ]


def _feature_bundle(feature_dir: Path, assets: SceneAssets) -> str:
    manifest, _digest, _source = load_json_object(
        feature_dir / "frame_manifest.json",
        label=f"{assets.scene} all-view official RADIO feature manifest",
    )
    radio = manifest.get("radio", {})
    features = manifest.get("features", {})
    expected_grid = [assets.feature_height, assets.feature_width]
    if (
        manifest.get("scene") != assets.scene
        or Path(str(manifest.get("image_dir", ""))).resolve() != assets.image_dir
        or int(manifest.get("num_frames", -1)) != assets.training_frame_count
        or manifest.get("frame_id_mode") != "source_rank"
        or list(manifest.get("excluded_image_stems", []))
        or radio.get("checkpoint_sha256") != RADIO_CHECKPOINT_SHA256
        or radio.get("checkpoint_provenance") != "explicit_file_sha256"
        or radio.get("requested_adaptors") != ["siglip2-g"]
        or float(manifest.get("resolution_scale", -1.0)) != assets.resolution_scale
        or features.get("backbone", {}).get("grid") != expected_grid
    ):
        raise ValueError(f"{assets.scene} all-view feature authority differs")
    if not any(
        row.get("name") == "siglip2-g"
        and row.get("subdir") == "siglip2"
        and row.get("grid") == expected_grid
        for row in features.get("adaptors", [])
    ):
        raise ValueError(f"{assets.scene} has no official SigLIP2 grid")
    bundle = str(manifest.get("output_bundle_sha256", ""))
    if len(bundle) != 64:
        raise ValueError(f"{assets.scene} feature bundle is not sealed")
    return bundle


def _write_cohort_authority(
    *,
    path: Path,
    assets: SceneAssets,
    feature_bundle_sha256: str,
    exact_raw: Path,
    exact_dino: Path,
    exact_sam: Path,
) -> str:
    payload = {
        "schema_version": 1,
        "artifact_type": "factorized_capability_cohort_authority",
        "experiment": "canonical-factorized-radio-v1-formal-capability-cohort",
        "benchmark": "SPIn-NeRF Available-Nine",
        "scene": assets.scene,
        "feature_output_bundle_sha256": feature_bundle_sha256,
        "frozen_cache_authorities": {
            "radio": {"path": str(exact_raw), "sha256": sha256_file(exact_raw)},
            "dino_v3": {"path": str(exact_dino), "sha256": sha256_file(exact_dino)},
            "sam3": {"path": str(exact_sam), "sha256": sha256_file(exact_sam)},
        },
        "target_access": {
            # Under the frozen SPIn protocol these are registered source views,
            # including views whose masks are evaluated later.  They are not
            # the separately gated target/query RGB category used by the
            # generic field trainer and Method-v1 target-access contract.
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "registered_source_rgb_opened": True,
            "registered_source_rgb_authority": "all registered RGB views permitted",
            "reference_masks_opened": False,
            "evaluation_masks_opened": False,
            "text_queries_opened": False,
            "target_metrics_used_for_selection": False,
        },
    }
    if path.is_file():
        existing, _digest, _source = load_json_object(
            path, label=f"{assets.scene} Method-v1 capability cohort"
        )
        if existing != payload:
            raise ValueError(f"{assets.scene} capability cohort authority differs")
    else:
        write_frozen_json(path, payload)
    return sha256_file(path)


def _run_mpr_stage(
    args: argparse.Namespace,
    *,
    stage: str,
    output: Path,
    command: Iterable[str | Path],
    logs: Path,
) -> None:
    if not _require_complete(
        output, [output.with_suffix(output.suffix + ".json")], stage
    ):
        _run_spin_stage(args, stage, command, gpu=True, log_dir=logs)


def _run_with_scene_lock(
    args: argparse.Namespace, *, assets: SceneAssets, run_root: Path
) -> dict:
    feature_dir = run_root / "all_view_features"
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
    cohort = run_root / "method_v1_capability_cohort_authority_v2.json"
    base_field = run_root / "factorized_d512_l512_heldout4.pth"
    crop_root = run_root / "genuine_crop_summary_teacher"
    siglip_field = run_root / "official_siglip2_spatial_w005_s0_64.pth"
    region_field = run_root / "official_siglip2_spatial_region_w005_s0_64.pth"
    final_field = run_root / "generic_text_response_w005_s0_64.pth"
    stop_index = STAGES.index(args.stop_after)

    # Execution-only microbatching for the largest carrier.  This does not
    # alter the D512/L512 objective, data, epochs, or gates; it prevents an
    # otherwise repeatable 2.46-GiB backward allocation failure on shared
    # 24-GiB workers.  All other scenes retain the preregistered default.
    base_batch_size = "2048" if assets.scene == "truck" else "4096"
    base_eval_batch_size = "8192" if assets.scene == "truck" else "16384"

    if not (feature_dir / "frame_manifest.json").is_file():
        _run_spin_stage(
            args,
            "source_features",
            source_feature_command(assets, feature_dir),
            gpu=True,
            log_dir=logs,
        )
    feature_bundle = _feature_bundle(feature_dir, assets)
    if stop_index == 0:
        return {"scene": assets.scene, "completed_stage": STAGES[0]}

    if not validation_plan.is_file():
        _run_spin_stage(
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
            f"partial factorized MPR output must be audited: {factorized}"
        )
    if not (all(factorized_parts) and responsibility.is_file()):
        if all(factorized_parts) or responsibility.is_file():
            raise RuntimeError("factorized MPR/responsibility authority is partial")
        _run_spin_stage(
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
                "--save-responsibility-cache",
                responsibility,
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
    _run_mpr_stage(
        args,
        stage="exact_raw_mpr",
        output=exact_raw,
        command=[
            "radio_gs/scripts/build_gaussian_multiview_teacher_cache.py",
            *exact_common,
            "--output",
            exact_raw,
            "--feature-space",
            "radio",
        ],
        logs=logs,
    )
    if stop_index == 3:
        return {"scene": assets.scene, "completed_stage": STAGES[3]}

    for stage, output, feature_space in (
        ("exact_dino_mpr", exact_dino, "dino_v3"),
        ("exact_sam_mpr", exact_sam, "sam3"),
    ):
        _run_mpr_stage(
            args,
            stage=stage,
            output=output,
            command=[
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
                str(CAPABILITY_SHARD_CHANNELS),
            ],
            logs=logs,
        )
        if stop_index == STAGES.index(stage):
            return {"scene": assets.scene, "completed_stage": stage}

    cohort_sha = _write_cohort_authority(
        path=cohort,
        assets=assets,
        feature_bundle_sha256=feature_bundle,
        exact_raw=exact_raw,
        exact_dino=exact_dino,
        exact_sam=exact_sam,
    )
    if stop_index == 6:
        return {"scene": assets.scene, "completed_stage": STAGES[6]}

    if not _require_complete(
        base_field,
        [base_field.with_suffix(base_field.suffix + ".json")],
        "base_field",
    ):
        _run_spin_stage(
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
                "cpu",
                "--official-capability-loss",
                "--capability-target-contract",
                "matched_exact_marginal",
                "--dino-mpr-cache",
                exact_dino,
                "--expected-dino-v3-mpr-cache-sha256",
                sha256_file(exact_dino),
                "--sam3-mpr-cache",
                exact_sam,
                "--expected-sam3-mpr-cache-sha256",
                sha256_file(exact_sam),
                "--factorized-capability-reference-mpr-cache",
                exact_raw,
                "--expected-factorized-capability-reference-mpr-cache-sha256",
                sha256_file(exact_raw),
                "--factorized-capability-cohort-authority",
                cohort,
                "--expected-factorized-capability-cohort-authority-sha256",
                cohort_sha,
                "--epochs",
                "20",
                "--min-epochs",
                "20",
                "--batch-size",
                base_batch_size,
                "--eval-batch-size",
                base_eval_batch_size,
                "--local-code-training-dtype",
                "float16",
                *(
                    ["--sparse-local-code-gradients"]
                    if assets.scene == "truck"
                    else []
                ),
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
        _run_spin_stage(
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
                "--resume-partial",
            ],
            gpu=True,
            log_dir=logs,
        )
    crop_payload, _digest, _source = load_json_object(
        crop_manifest, label=f"{assets.scene} genuine crop-summary manifest"
    )
    if (
        crop_payload.get("frame_id_mode") != "source_rank"
        or crop_payload.get("excluded_image_stems")
        or crop_payload["scenes"][assets.scene]["num_frames"]
        != assets.training_frame_count
        or crop_payload.get("benchmark_masks_opened") is not False
    ):
        raise ValueError(f"{assets.scene} crop-summary authority differs")
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
        "256",
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
        "--capability-projection-amp",
        "--capability-projection-xformers",
        "--capability-projection-checkpoint",
        "--capability-projection-token-mlp-chunk-size",
        # The token MLP is pointwise, so smaller exact chunks trade only
        # runtime for peak activation memory.  This leaves enough headroom on
        # shared 24-GiB workers for the complete-grid official adaptor.
        "64",
        "--staged-capability-gradient",
        "--offload-capability-adaptors-after-gradient",
        "--column-staged-direct-field-backward",
        "--release-validation-cuda-cache",
        "--offload-optimizer-state",
        "--local-code-training-dtype",
        "float16",
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
        _run_spin_stage(
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
        # The official summary head is pointwise and CPU resident for SPIn.
        # A large host batch is algebraically identical, avoids thousands of
        # tiny GEMMs per view, and does not consume accelerator memory.
        "--semantic-projection-batch-size",
        "2048",
        "--semantic-projector-device",
        "cpu",
    ]
    if not _require_complete(
        region_field,
        [region_field.with_suffix(region_field.suffix + ".json")],
        "region_stage",
    ):
        _run_spin_stage(
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
        _run_spin_stage(
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
        _run_spin_stage(
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
        write_frozen_json(
            gate_report,
            {
                "status": "pass",
                "benchmark": "SPIn-NeRF Available-Nine",
                "scene": assets.scene,
                "field": str(final_field),
                "field_sha256": sha256_file(final_field),
                "method_authority": str(METHOD_AUTHORITY),
                "method_authority_sha256": sha256_file(METHOD_AUTHORITY),
                "construction_preregistration": str(SPIN_PREREGISTRATION),
                "construction_preregistration_sha256": sha256_file(
                    SPIN_PREREGISTRATION
                ),
            },
        )
    return {
        "scene": assets.scene,
        "completed_stage": STAGES[-1],
        "field": str(final_field),
        "field_sha256": sha256_file(final_field),
        "gate": str(gate_report),
    }


def run(args: argparse.Namespace) -> dict:
    assets = resolve_scene_assets(args.scene)
    run_root = Path(args.run_root).expanduser().resolve() / assets.scene
    with _scene_run_boundary(run_root=run_root, scene=assets.scene):
        return _run_with_scene_lock(args, assets=assets, run_root=run_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    # Accept any physical device exposed by the current host.
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--stop-after", choices=STAGES, default=STAGES[-1])
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
