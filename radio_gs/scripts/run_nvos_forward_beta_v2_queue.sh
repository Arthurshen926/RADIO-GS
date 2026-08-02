#!/usr/bin/env bash

# Independent fixed-full-eight queue authority for balanced-residual Beta-v2.
# GPU admission/thermal behavior is shared with the audited v1 driver, while
# every candidate, snapshot, scene receipt, output and aggregation authority is
# v2-specific.  No v1 result or receipt path is accepted.

set -euo pipefail

V2_RUNNER_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
V2_REPO_ROOT="$(cd "$(dirname "$V2_RUNNER_PATH")/../.." && pwd -P)"

export FORWARD_BETA_VARIANT="v2"
export RUNNER_AUTHORITY_PATH="$V2_RUNNER_PATH"
export SNAPSHOT_STAGER="$V2_REPO_ROOT/radio_gs/scripts/stage_nvos_forward_beta_v2_snapshot.py"
export SCENE_AUTHORITY="$V2_REPO_ROOT/radio_gs/scripts/nvos_forward_beta_v2_scene_authority.py"
export AGGREGATOR="$V2_REPO_ROOT/radio_gs/scripts/aggregate_nvos_forward_beta_v2_full8_nonexact.py"
export CANDIDATE_ID="nvos-forward-beta-balanced-residual-v2"
export FORWARD_MODE="beta_balanced_residual_v2"
export STAGER_MODULE="radio_gs.scripts.stage_nvos_forward_beta_v2_snapshot"
export STAGING_MANIFEST_RELATIVE="paper/artifacts/nvos_forward_beta_balanced_residual_v2_snapshot_staging.json"
export CANDIDATE_CONTRACT="$V2_REPO_ROOT/paper/artifacts/nvos_forward_beta_balanced_residual_v2_candidate_20260802.yaml"
export PROTOCOL_AUTHORITY_RECEIPT="$V2_REPO_ROOT/paper/artifacts/nvos_forward_beta_balanced_residual_v2_protocol_authority.json"
export RELIABILITY_CACHE_MANIFEST="${RELIABILITY_CACHE_MANIFEST:-$V2_REPO_ROOT/paper/artifacts/nvos_forward_beta_balanced_residual_v2_reliability_manifest_20260802.json}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/RADIO-GS/output/optimization_20260802/nvos_forward_beta_balanced_residual_v2_full8}"

exec bash "$V2_REPO_ROOT/radio_gs/scripts/run_nvos_forward_beta_coverage_v1_queue.sh" "$@"
