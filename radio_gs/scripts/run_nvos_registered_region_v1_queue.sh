#!/usr/bin/env bash

# Diagnostic NVOS queue for the theory-frozen registered-region-v1 readout.
# The candidate changes only prompt compilation/readout over the already
# frozen canonical-mpr-v3 fields. It does not retrain geometry or capability
# features and never selects a stage or threshold from target masks.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-1}"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT/output/evaluation_closeout_20260716/canonical_mpr_v3_nvos8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/optimization_20260731/nvos_registered_region_v1}"
QUEUE_PLAN="${QUEUE_PLAN:-/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/gaussfm_jobs/nvos_strict_unseen_v1/queue_plan.json}"
MANIFEST="${MANIFEST:-/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json}"
RADIO_CHECKPOINT="${RADIO_CHECKPOINT:-/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar}"
SCENE_NAMES="${SCENE_NAMES:-fern flower fortress horns_center horns_left leaves orchids trex}"
RUN_MANIFEST="$OUTPUT_ROOT/run_manifest.json"
LOG_ROOT="$OUTPUT_ROOT/logs"
LOCK_ROOT="$OUTPUT_ROOT/locks"
THERMAL_GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"
GPU_TELEMETRY_LOG="${GPU_TELEMETRY_LOG:-$OUTPUT_ROOT/gpu1_telemetry.csv}"
GPU_MAX_TEMP_C="${GPU_MAX_TEMP_C:-78}"
GPU_START_MAX_TEMP_C="${GPU_START_MAX_TEMP_C:-70}"
GPU_MAX_POWER_LIMIT_W="${GPU_MAX_POWER_LIMIT_W:-300.5}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-10}"

if [[ "$GPU" != "1" ]]; then
  echo "registered-region-v1 is assigned to physical GPU1; got GPU=$GPU" >&2
  exit 2
fi
for required in \
  "$SOURCE_ROOT" "$QUEUE_PLAN" "$MANIFEST" "$RADIO_CHECKPOINT" \
  "$THERMAL_GUARD"; do
  if [[ ! -e "$required" ]]; then
    echo "missing registered-region-v1 input: $required" >&2
    exit 2
  fi
done

GPU_INFO=""
for candidate in /proc/driver/nvidia/gpus/*/information; do
  if [[ -r "$candidate" ]] \
    && [[ "$(awk '/Device Minor:/ {print $3}' "$candidate")" == "$GPU" ]]; then
    GPU_INFO="$candidate"
    break
  fi
done
if [[ -z "$GPU_INFO" ]]; then
  echo "physical GPU1 has no NVIDIA driver record" >&2
  exit 2
fi
GPU_BUS_ID="$(awk '/Bus Location:/ {print $3}' "$GPU_INFO")"
GPU_CONFIG="/sys/bus/pci/devices/$GPU_BUS_ID/config"
GPU_CONFIG_PREFIX="$(od -An -tx1 -N16 "$GPU_CONFIG" 2>/dev/null | tr -d ' \n')"
if [[ -z "$GPU_CONFIG_PREFIX" || "$GPU_CONFIG_PREFIX" =~ ^f+$ ]]; then
  echo "physical GPU1 PCIe configuration space is not responding" >&2
  exit 2
fi
if ! timeout --kill-after=2s 10s nvidia-smi -i "$GPU" >/dev/null; then
  echo "physical GPU1 is not usable by the current container" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$LOCK_ROOT"

exec {manifest_lock}>"$LOCK_ROOT/run_manifest.lock"
flock "$manifest_lock"
bash radio_gs/scripts/run_repo_python.sh - \
  "$SOURCE_ROOT" "$QUEUE_PLAN" "$MANIFEST" "$RADIO_CHECKPOINT" \
  "$SCENE_NAMES" "$OUTPUT_ROOT" "$RUN_MANIFEST" "$0" \
  "$THERMAL_GUARD" "$GPU_MAX_TEMP_C" "$GPU_START_MAX_TEMP_C" \
  "$GPU_MAX_POWER_LIMIT_W" "$GPU_POLL_SECONDS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import yaml

(
    source_root,
    queue_plan,
    benchmark_manifest,
    radio_checkpoint,
    scene_names,
    output_root,
    run_manifest,
    runner,
    thermal_guard,
    gpu_max_temp_c,
    gpu_start_max_temp_c,
    gpu_max_power_limit_w,
    gpu_poll_seconds,
) = sys.argv[1:]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


repo = Path(runner).resolve().parents[2]
source = Path(source_root).resolve()
scenes = scene_names.split()
expected = [
    "fern",
    "flower",
    "fortress",
    "horns_center",
    "horns_left",
    "leaves",
    "orchids",
    "trex",
]
if scenes != expected:
    raise SystemExit(
        "registered-region-v1 requires the frozen ordered NVOS cohort: "
        + " ".join(expected)
    )
source_artifacts = {}
for scene in scenes:
    records = {}
    for name in (
        "canonical_d256_l128_capability_first.pth",
        "official_dino_sam3_views.pt",
        "shared_support_graph_k16.pt",
    ):
        path = source / scene / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise SystemExit(f"{scene} lacks frozen source artifact {path}")
        records[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        metadata = path.with_suffix(path.suffix + ".json")
        if not metadata.is_file():
            raise SystemExit(f"{scene} lacks source metadata {metadata}")
        records[name]["metadata_path"] = str(metadata)
        records[name]["metadata_sha256"] = sha256(metadata)
    source_artifacts[scene] = records

queue = json.loads(Path(queue_plan).read_text(encoding="utf-8"))
if (
    str(queue.get("benchmark")) != "nvos"
    or str(queue.get("protocol_hash")) != str(
        json.loads(Path(benchmark_manifest).read_text(encoding="utf-8")).get(
            "protocol_hash"
        )
    )
    or [str(row.get("scene_id")) for row in queue.get("scenes", [])]
    != expected
):
    raise SystemExit("queue/manifest do not match the frozen NVOS cohort")
queue_scene_inputs = {}
queue_scene_root = Path(queue_plan).resolve().parent / "scenes"


def config_chain(path: Path) -> list[Path]:
    chain = []
    seen = set()
    current = path.resolve()
    while True:
        if current in seen:
            raise SystemExit(f"cyclic base_config chain at {current}")
        seen.add(current)
        if not current.is_file():
            raise SystemExit(f"missing config in base_config chain: {current}")
        chain.append(current)
        payload = yaml.safe_load(current.read_text(encoding="utf-8")) or {}
        base = str(payload.get("base_config", "")).strip()
        if not base:
            return chain
        base_path = Path(base).expanduser()
        current = (
            base_path
            if base_path.is_absolute()
            else current.parent / base_path
        ).resolve()


for scene in scenes:
    scene_root = queue_scene_root / scene
    records = {}
    config_path = scene_root / "gaussfm_main_track.yaml"
    config_payload = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    ) or {}
    dataset_scene_root = Path(str(config_payload["scene_root"])).resolve()
    colmap_root = dataset_scene_root / "sparse" / "0"
    colmap_files = [
        colmap_root / "cameras.bin",
        colmap_root / "images.bin",
    ]
    optional_hwf = dataset_scene_root / "hwf_cxcy.npy"
    if optional_hwf.is_file():
        colmap_files.append(optional_hwf)
    fixed_files = (
        *config_chain(config_path),
        scene_root / "feature_field" / "checkpoints" / "best.pth",
        scene_root / "rgb_to_colmap_camera_mapping.json",
        scene_root / "feature_pose_mapping.json",
        scene_root / "train_frame_ids.json",
        *colmap_files,
    )
    mapping = json.loads(
        (scene_root / "feature_pose_mapping.json").read_text(encoding="utf-8")
    )
    pose_files = sorted(
        {
            Path(str(record["pose_path"])).resolve()
            for record in mapping.get("records", [])
        }
    )
    for path in (*fixed_files, *pose_files):
        path = path.resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise SystemExit(f"{scene} lacks renderer/view input {path}")
        records[str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    queue_scene_inputs[scene] = records

implementation = {
    relative: sha256(repo / relative)
    for relative in (
        "radio_gs/scripts/eval_nvos_gaussian_first.py",
        "radio_gs/querying/query_spec.py",
        "radio_gs/querying/query_compilers.py",
        "radio_gs/querying/evidence_scorer.py",
        "radio_gs/querying/query_engine.py",
        "radio_gs/querying/score_calibration.py",
        "radio_gs/querying/support_solver.py",
        "radio_gs/rendering/feature_renderer.py",
        "radio_gs/scripts/eval_lerf_direct_3d_selection.py",
        "radio_gs/scripts/eval_lerf_grounding.py",
        "radio_gs/scripts/render_promptable_nvs_features.py",
        "radio_gs/interfaces/capability_cache.py",
        "radio_gs/evaluation/promptable_segmentation.py",
        "radio_gs/config.py",
        "radio_gs/data/lerf_dataset.py",
        "radio_gs/models/explicit_gaussian.py",
        "radio_gs/models/featsharp_3d.py",
        "radio_gs/models/hcd_codec.py",
        "radio_gs/models/hybrid_gaussian.py",
        "radio_gs/models/screen_refiner.py",
        "radio_gs/rendering/contribution_compositor.py",
        "radio_gs/utils/checkpoint_io.py",
        "radio_gs/scripts/aggregate_registered_prompt_closeout.py",
        "radio_gs/scripts/run_with_gpu_thermal_guard.sh",
    )
}
payload = {
    "schema_version": 1,
    "candidate": "registered-region-v1",
    "eligibility": "diagnostic_until_disjoint_registered_prompt_gate",
    "scenes": scenes,
    "source_root": str(source),
    "source_artifacts": source_artifacts,
    "queue_scene_inputs": queue_scene_inputs,
    "queue_plan": str(Path(queue_plan).resolve()),
    "queue_plan_sha256": sha256(Path(queue_plan)),
    "benchmark_manifest": str(Path(benchmark_manifest).resolve()),
    "benchmark_manifest_sha256": sha256(Path(benchmark_manifest)),
    "radio_checkpoint": str(Path(radio_checkpoint).resolve()),
    "radio_checkpoint_sha256": sha256(Path(radio_checkpoint)),
    "method_contract": {
        "support_mode": "canonical_support",
        "region_space": "sam3",
        "prompt_registration": {
            "mode": "raster_adjoint",
            "scale": 1.0,
            "alpha_threshold": 0.0,
            "depth_tolerance": 0.08,
            "relative_depth_tolerance": 0.02,
        },
        "seed_construction": "joint_signed",
        "seed_normalization": "none",
        "observation_fusion": "probability_mixture",
        "registered_seed_unary_weight": 0.0,
        "observation_mass_source": "raw_raster_adjoint_prompt_mass",
        "observation_confidence": "poisson_mass",
        "observation_mass_scale": 1.0,
        "observation_constructed_before_capability_filter": True,
        "prompt_support_threshold": 0.0,
        "prototype_count": 4,
        "prototype_strategy": "spherical_mean_fps",
        "appearance_weight": 1.0,
        "boundary_weight": 0.35,
        "prototype_temperature": 0.07,
        "feature_calibration": "none",
        "background_centroids": 0,
        "score_calibration": "none",
        "negative_spatial_mode": "none",
        "diagnostic_selection_mode": "seeded_component",
        "selection_applied_to_main_output": False,
        "final_readout": "propagated",
        "graph": {
            "policy": "legacy",
            "component_policy": "same",
            "legacy_residual": 0.0,
            "channel_confidence_mode": "none",
        },
        "score_render": {
            "resolution": "prompt_native",
            "scale": 1.0,
            "valid_support_normalization": True,
            "valid_support_coverage_power": 1.0,
            "feature_contribution_gamma": 1.0,
            "score_chunk_size": 8192,
            "pixel_threshold": 0.5,
            "threshold_comparison": "greater_or_equal",
            "resize_to_ground_truth": "cv2.INTER_LINEAR",
        },
        "solver": {
            "type": "confidence_random_walker",
            "iterations": 12,
            "residual": 0.30,
            "unary_temperature": 0.10,
            "support_threshold": 0.50,
            "laplacian_weight": 1.0,
            "cg_iterations": 64,
            "cg_tolerance": 1e-5,
            "hard_seed_threshold": 0.20,
            "hard_seed_conflict_policy": "exclusive_relative",
            "hard_seed_conflict_margin": 0.0,
            "component_edge_threshold": 1e-5,
            "seeded_component_min_weight": 0.20,
        },
        "canonical_reliability_cache": "",
        "diagnostic_graph_affinity_override": "",
        "asset_hash_verification_required": True,
        "uses_target_calibration": False,
    },
    "runner": str(Path(runner).resolve()),
    "runner_sha256": sha256(Path(runner)),
    "thermal_safety_contract": {
        "guard": str(Path(thermal_guard).resolve()),
        "guard_sha256": sha256(Path(thermal_guard)),
        "physical_gpu": 1,
        "maximum_temperature_c": int(gpu_max_temp_c),
        "maximum_start_temperature_c": int(gpu_start_max_temp_c),
        "maximum_power_limit_w": float(gpu_max_power_limit_w),
        "poll_seconds": int(gpu_poll_seconds),
    },
    "implementation_sources": implementation,
}
manifest = Path(run_manifest)
if manifest.is_file():
    previous = json.loads(manifest.read_text(encoding="utf-8"))
    if previous != payload:
        raise SystemExit("OUTPUT_ROOT belongs to another immutable NVOS run")
else:
    allowed_pre_manifest_files = {
        (Path(output_root) / "locks" / "run_manifest.lock").resolve(),
    }
    existing = [
        path
        for path in Path(output_root).rglob("*")
        if (
            path.is_file()
            and path.name != "run_manifest.json"
            and path.resolve() not in allowed_pre_manifest_files
        )
    ]
    if existing:
        raise SystemExit("OUTPUT_ROOT contains artifacts without a run manifest")
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest)
PY
flock -u "$manifest_lock"
exec {manifest_lock}>&-

for scene in $SCENE_NAMES; do
  source="$SOURCE_ROOT/$scene"
  result_root="$OUTPUT_ROOT/$scene/eval_full_mask_random_walker"
  result="$result_root/${scene}_evaluation.json"
  if [[ -s "$result" ]]; then
    continue
  fi
  exec {scene_lock}>"$LOCK_ROOT/$scene.lock"
  flock "$scene_lock"
  if [[ -s "$result" ]]; then
    flock -u "$scene_lock"
    exec {scene_lock}>&-
    continue
  fi
  field="$source/canonical_d256_l128_capability_first.pth"
  capability="$source/official_dino_sam3_views.pt"
  graph="$source/shared_support_graph_k16.pt"
  field_sha="$(sha256sum "$field" | awk '{print $1}')"
  mkdir -p "$result_root"
  GPU="$GPU" CUDA_VISIBLE_DEVICES="$GPU" \
    GPU_TELEMETRY_LOG="$GPU_TELEMETRY_LOG" \
    GPU_MAX_TEMP_C="$GPU_MAX_TEMP_C" \
    GPU_START_MAX_TEMP_C="$GPU_START_MAX_TEMP_C" \
    GPU_MAX_POWER_LIMIT_W="$GPU_MAX_POWER_LIMIT_W" \
    GPU_POLL_SECONDS="$GPU_POLL_SECONDS" \
    bash "$THERMAL_GUARD" -- \
    bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/eval_nvos_gaussian_first.py \
    --manifest "$MANIFEST" \
    --queue-root "$(dirname "$QUEUE_PLAN")" \
    --scene-id "$scene" \
    --output-dir "$result_root" \
    --run-manifest "$RUN_MANIFEST" \
    --device cuda:0 \
    --radio-checkpoint "$RADIO_CHECKPOINT" \
    --region-space sam3 \
    --support-mode canonical_support \
    --canonical-capability-cache "$capability" \
    --canonical-support-graph "$graph" \
    --canonical-field-sha256 "$field_sha" \
    --prompt-registration-mode raster_adjoint \
    --prompt-registration-scale 1.0 \
    --alpha-threshold 0.0 \
    --support-threshold 0.0 \
    --prototype-count 4 \
    --prototype-strategy spherical_mean_fps \
    --registered-seed-construction joint_signed \
    --registered-observation-fusion probability_mixture \
    --registered-observation-confidence poisson_mass \
    --registered-observation-mass-scale 1.0 \
    --registered-seed-unary-weight 0.0 \
    --registered-selection-mode seeded_component \
    --registered-readout-stage propagated \
    --score-render-resolution prompt_native \
    --score-render-scale 1.0 \
    --valid-support-normalization \
    --valid-support-coverage-power 1.0 \
    --feature-contribution-gamma 1.0 \
    --graph-policy legacy \
    --component-graph-policy same \
    --channel-confidence-mode none \
    --negative-spatial-mode none \
    --appearance-weight 1.0 \
    --boundary-weight 0.35 \
    --prototype-temperature 0.07 \
    --feature-calibration none \
    --background-centroids 0 \
    --score-calibration none \
    --score-chunk-size 8192 \
    --solver-type confidence_random_walker \
    --laplacian-weight 1.0 \
    --cg-iterations 64 \
    --cg-tolerance 1e-5 \
    --hard-seed-threshold 0.20 \
    --hard-seed-conflict-policy exclusive_relative \
    --hard-seed-conflict-margin 0.0 \
    --component-edge-threshold 1e-5 \
    --seeded-component-min-weight 0.20 \
    --solver-iterations 12 \
    --solver-residual 0.30 \
    --solver-unary-temperature 0.10 \
    --solver-support-threshold 0.50 \
    --require-asset-hashes \
    >"$LOG_ROOT/$scene.log" 2>&1
  flock -u "$scene_lock"
  exec {scene_lock}>&-
done

exec {aggregate_lock}>"$LOCK_ROOT/aggregate.lock"
flock "$aggregate_lock"
bash radio_gs/scripts/run_repo_python.sh \
  radio_gs/scripts/aggregate_registered_prompt_closeout.py \
  --queue-plan "$QUEUE_PLAN" \
  --result-root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/summary.json" \
  --run-manifest "$RUN_MANIFEST" \
  --require-method-config \
  >"$OUTPUT_ROOT/aggregate.log" 2>&1
flock -u "$aggregate_lock"
exec {aggregate_lock}>&-
