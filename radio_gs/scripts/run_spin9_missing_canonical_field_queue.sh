#!/usr/bin/env bash

# Resume the three missing SPIn local9 query-independent canonical fields on
# a selected physical GPU. RGB geometry is reused from the frozen all-view 3DGS audit;
# prompts and benchmark masks are not opened by any stage in this queue.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-1}"
SCENES="${SCENES:-horns room truck}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/output/optimization_20260802/spin9_exact_local9}"
QUEUE_ROOT="$REPO_ROOT/output/unified_query/spin9_gaussfm_queue_20260712/scenes"
RADIO_CHECKPOINT="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar"
THERMAL_GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"

[[ "$GPU" =~ ^[01]$ ]] || { echo "GPU must be physical index 0 or 1" >&2; exit 2; }
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/carriers" "$RUN_ROOT/fields"

sha256_file() { sha256sum "$1" | awk '{print $1}'; }

run_guarded() {
  local scene="$1" stage="$2"
  shift 2
  local poll_seconds=20 start_temp=78 soft_temp=81 resume_temp=76 hard_temp=84
  if [[ "$stage" == "feature_extraction" ]]; then
    poll_seconds=5
    start_temp=75
    soft_temp=78
    hard_temp=82
    if [[ "$GPU" == "1" ]]; then resume_temp=70; else resume_temp=72; fi
  fi
  echo "[$(date --iso-8601=seconds)] $scene: $stage"
  env CUDA_VISIBLE_DEVICES="$GPU" \
    GPU="$GPU" \
    GPU_MAX_POWER_LIMIT_W=300.5 \
    GPU_POLL_SECONDS="$poll_seconds" \
    GPU_START_MAX_TEMP_C="$start_temp" \
    GPU_SOFT_PAUSE_TEMP_C="$soft_temp" \
    GPU_SOFT_RESUME_TEMP_C="$resume_temp" \
    GPU_MAX_TEMP_C="$hard_temp" \
    GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES=3 \
    GPU_OWNER_PID_NAMESPACE_MODE=exclusive-singleton-after-clear-v1 \
    GPU_TELEMETRY_LOG="$RUN_ROOT/logs/gpu${GPU}_telemetry.csv" \
    GPU_OWNER_AUDIT_LOG="$RUN_ROOT/logs/gpu${GPU}_owner.csv" \
    bash "$THERMAL_GUARD" -- \
      env CUDA_VISIBLE_DEVICES="$GPU" \
      bash radio_gs/scripts/run_repo_python.sh "$@" \
      >"$RUN_ROOT/logs/${scene}.${stage}.log" 2>&1
}

for scene in $SCENES; do
  case "$scene" in
    horns)
      source_root="/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/SPIn-NeRF/source_images/llff_google_drive/extracted/nerf_llff_data/horns"
      ply="$REPO_ROOT/output/protocol_audit_20260731/ludvig/nvos/released_all_view/horns/training/attempts/exact_f7a_allview_30k_v1/model/point_cloud/iteration_30000/point_cloud.ply"
      expected_ply_sha="1f3084daba5e9a70263152f73f704815f452a2f37e81f56cf3a3eb96c3e49803"
      ;;
    room)
      source_root="/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/SPIn-NeRF/source_images/llff_google_drive/extracted/nerf_llff_data/room"
      ply="$REPO_ROOT/output/protocol_audit_20260731/ludvig/spin/released_all_view/room/training/attempts/exact_f7a_allview_30k_v1/model/point_cloud/iteration_30000/point_cloud.ply"
      expected_ply_sha="c12a46a46ab3550905cfe48152916bd75834f5f36485ec4cf1cae03eaef294fd"
      ;;
    truck)
      source_root="/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/SPIn-NeRF/source_images/tandt/extracted/tandt/truck"
      ply="$REPO_ROOT/output/protocol_audit_20260731/ludvig/spin/released_all_view/truck/training/attempts/exact_f7a_allview_30k_v1/model/point_cloud/iteration_30000/point_cloud.ply"
      expected_ply_sha="aa5b6d166277cde9d49632c9ffa427c392a4e52b7c68719c3d6ca312bedc3492"
      ;;
    *) echo "unknown scene: $scene" >&2; exit 2 ;;
  esac
  [[ "$(sha256_file "$ply")" == "$expected_ply_sha" ]] || {
    echo "$scene frozen geometry PLY SHA-256 mismatch" >&2
    exit 3
  }

  queue_scene="$QUEUE_ROOT/$scene"
  feature_dir="$queue_scene/radio_features"
  carrier_dir="$RUN_ROOT/carriers/$scene"
  field_dir="$RUN_ROOT/fields/$scene"
  config="$carrier_dir/config.yaml"
  carrier="$carrier_dir/geometry_carrier.pth"
  responsibility="$field_dir/responsibility.pt"
  raw="$field_dir/raw_radio.pt"
  dino="$field_dir/dino_v3.pt"
  sam3="$field_dir/sam3.pt"
  field="$field_dir/canonical_d256_l128_capability_first.pth"
  capability="$field_dir/official_dino_sam3_views.pt"
  mkdir -p "$carrier_dir" "$field_dir"

  if [[ ! -s "$feature_dir/frame_manifest.json" ]]; then
    run_guarded "$scene" feature_extraction \
      radio_gs/scripts/extract_radio_features.py \
      --scene "$scene" \
      --image_dir "$source_root/images" \
      --output_dir "$feature_dir" \
      --radio_repo /root/RADIO \
      --radio_version c-radio_v4-h \
      --radio_checkpoint "$RADIO_CHECKPOINT" \
      --resolution_scale 1.0 \
      --frame-id-mode source_rank \
      --device cuda \
      --amp \
      --batch_size 1 \
      --resume-partial \
      --skip_pca_stats \
      --radio-thermal-pacing-seconds-per-image 10
  fi

  if [[ ! -s "$carrier" || ! -s "$config" ]]; then
    CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_promptable_geometry_render_contract.py \
      --base-config "$queue_scene/gaussfm_main_track.yaml" \
      --ply-path "$ply" \
      --output-config "$config" \
      --output-checkpoint "$carrier" \
      >"$RUN_ROOT/logs/${scene}.geometry_carrier.log" 2>&1
  fi
  geometry_sha="$(sha256_file "$carrier")"
  radio_checkpoint_sha="$(sha256_file "$RADIO_CHECKPOINT")"
  feature_bundle_sha="$({
    CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh - \
      "$feature_dir/frame_manifest.json" "$radio_checkpoint_sha" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
radio = payload.get("radio", {})
execution = payload.get("execution", {})
if radio.get("checkpoint_sha256") != sys.argv[2]:
    raise SystemExit("feature manifest RADIO checkpoint SHA-256 differs")
if radio.get("checkpoint_provenance") != "explicit_file_sha256":
    raise SystemExit("feature manifest lacks explicit checkpoint provenance")
if execution.get("resume_contract") != ".extract_resume_contract.json":
    raise SystemExit("feature manifest is not strict-resume output")
value = str(payload.get("output_bundle_sha256", ""))
if len(value) != 64:
    raise SystemExit("feature manifest lacks output_bundle_sha256")
print(value)
PY
  })"

  common_mpr=(
    --config "$config"
    --checkpoint "$carrier"
    --observation-contract canonical-mpr-v1
    --max-views 120
    --device cuda:0
    --expected-feature-scene "$scene"
    --expected-feature-image-dir "$source_root/images"
    --expected-geometry-checkpoint-sha256 "$geometry_sha"
    --expected-feature-output-bundle-sha256 "$feature_bundle_sha"
    --raster-channel-chunk-size 128
  )
  raw_storage_args=()
  sam3_storage_args=()
  dino_shard_channels=256
  if [[ "$scene" == "truck" ]]; then
    raw_storage_args=(
      --capability-storage channel_sharded
      --capability-shard-channels 128
    )
    sam3_storage_args=(
      --capability-storage channel_sharded
      --capability-shard-channels 512
    )
    dino_shard_channels=512
  elif [[ "$scene" == "horns" ]]; then
    raw_storage_args=(
      --capability-storage channel_sharded
      --capability-shard-channels 512
    )
    sam3_storage_args=(
      --capability-storage channel_sharded
      --capability-shard-channels 512
    )
    dino_shard_channels=512
  fi
  if [[ ! -s "$raw" ]]; then
    raw_responsibility_args=()
    if [[ -s "$responsibility" ]]; then
      existing_responsibility_sha="$(sha256_file "$responsibility")"
      raw_responsibility_args=(
        --responsibility-cache "$responsibility"
        --expected-responsibility-cache-sha256 "$existing_responsibility_sha"
      )
    else
      raw_responsibility_args=(--save-responsibility-cache "$responsibility")
    fi
    run_guarded "$scene" mpr_raw \
      radio_gs/scripts/build_gaussian_multiview_teacher_cache.py \
      "${common_mpr[@]}" \
      --feature-space radio \
      "${raw_storage_args[@]}" \
      "${raw_responsibility_args[@]}" \
      --output "$raw"
  elif [[ ! -s "$responsibility" ]]; then
    echo "$scene completed raw MPR lacks its responsibility sidecar" >&2
    exit 4
  fi
  responsibility_sha="$(sha256_file "$responsibility")"
  if [[ ! -s "$dino" ]]; then
    run_guarded "$scene" mpr_dino \
      radio_gs/scripts/build_gaussian_multiview_teacher_cache.py \
      "${common_mpr[@]}" \
      --feature-space dino_v3 \
      --capability-storage channel_sharded \
      --capability-shard-channels "$dino_shard_channels" \
      --radio-checkpoint "$RADIO_CHECKPOINT" \
      --capability-map-source project_raw \
      --responsibility-cache "$responsibility" \
      --expected-responsibility-cache-sha256 "$responsibility_sha" \
      --output "$dino"
  fi
  if [[ ! -s "$sam3" ]]; then
    run_guarded "$scene" mpr_sam3 \
      radio_gs/scripts/build_gaussian_multiview_teacher_cache.py \
      "${common_mpr[@]}" \
      --feature-space sam3 \
      "${sam3_storage_args[@]}" \
      --radio-checkpoint "$RADIO_CHECKPOINT" \
      --capability-map-source project_raw \
      --responsibility-cache "$responsibility" \
      --expected-responsibility-cache-sha256 "$responsibility_sha" \
      --output "$sam3"
  fi
  raw_sha="$(sha256_file "$raw")"
  dino_sha="$(sha256_file "$dino")"
  sam3_sha="$(sha256_file "$sam3")"
  if [[ ! -s "$field" ]]; then
    run_guarded "$scene" canonical_field \
      radio_gs/scripts/train_canonical_radio_field.py \
      --mpr-cache "$raw" \
      --expected-mpr-cache-sha256 "$raw_sha" \
      --observation-contract canonical-mpr-v1 \
      --expected-feature-output-bundle-sha256 "$feature_bundle_sha" \
      --radio-checkpoint "$RADIO_CHECKPOINT" \
      --expected-radio-checkpoint-sha256 "$radio_checkpoint_sha" \
      --output "$field" \
      --device cuda:0 \
      --coefficient-dim 256 \
      --local-dim 128 \
      --primitive-fusion \
      --official-capability-loss \
      --dino-mpr-cache "$dino" \
      --expected-dino-v3-mpr-cache-sha256 "$dino_sha" \
      --sam3-mpr-cache "$sam3" \
      --expected-sam3-mpr-cache-sha256 "$sam3_sha" \
      --epochs 20 \
      --min-epochs 5 \
      --target-cosine 0.985 \
      --seed 0
  fi
  if [[ "${BUILD_CAPABILITY:-0}" == "1" && ! -s "$capability" ]]; then
    run_guarded "$scene" capability_views \
      radio_gs/scripts/build_canonical_capability_views.py \
      --field-checkpoint "$field" \
      --mpr-cache "$raw" \
      --radio-checkpoint "$RADIO_CHECKPOINT" \
      --output "$capability" \
      --batch-size 2048 \
      --device cuda:0
  fi
done
