#!/usr/bin/env bash

# CPU-only orchestration for the frozen Surface text-response promotion gate.
# Dev is the only selection split.  Audit files are not even required or read
# unless the immutable dev decision requests one confirmation pass.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=""

SURFACE_ROOT="${SURFACE_ROOT:?set SURFACE_ROOT to the frozen Surface authority root}"
PROMOTION_MANIFEST="${PROMOTION_MANIFEST:?set PROMOTION_MANIFEST to the frozen attention screen}"
PROMOTION_COMPLETION="${PROMOTION_COMPLETION:?set PROMOTION_COMPLETION to its immutable completion}"
RADIO_CHECKPOINT="${RADIO_CHECKPOINT:-/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar}"
DEV_TEXT_BANK="${DEV_TEXT_BANK:?set DEV_TEXT_BANK to the frozen dev-split artifact}"
DEV_TEXT_BANK_MANIFEST="${DEV_TEXT_BANK_MANIFEST:?set DEV_TEXT_BANK_MANIFEST to its sidecar}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a new promotion output directory}"
EVALUATION_PROTOCOL_FREEZE="${EVALUATION_PROTOCOL_FREEZE:-$REPO_ROOT/paper/artifacts/evaluation_protocol_freeze_20260801.yaml}"
COMPANION="$REPO_ROOT/radio_gs/scripts/finalize_surface_text_response_promotion.py"
MATERIALIZER="$REPO_ROOT/radio_gs/scripts/materialize_surface_text_response_descriptors.py"
TEXT_GATE="$REPO_ROOT/radio_gs/scripts/eval_text_response_fidelity_gate.py"
FREEZE_BINDER="$REPO_ROOT/radio_gs/scripts/bind_evaluation_protocol_freeze.py"
FREEZE_BINDING_RECEIPT="$OUTPUT_ROOT/evaluation_protocol_freeze.binding.json"

for required in \
  "$PROMOTION_MANIFEST" "$PROMOTION_COMPLETION" "$RADIO_CHECKPOINT" \
  "$DEV_TEXT_BANK" "$DEV_TEXT_BANK_MANIFEST" "$COMPANION" \
  "$MATERIALIZER" "$TEXT_GATE" "$EVALUATION_PROTOCOL_FREEZE" \
  "$FREEZE_BINDER"; do
  if [[ ! -f "$required" ]]; then
    echo "missing Surface text-response promotion input: $required" >&2
    exit 2
  fi
done

verify_or_publish_freeze_binding() {
  local mode=()
  if [[ -e "$FREEZE_BINDING_RECEIPT" || -L "$FREEZE_BINDING_RECEIPT" ]]; then
    mode+=(--verify-existing)
  fi
  bash radio_gs/scripts/run_repo_python.sh "$FREEZE_BINDER" \
    --freeze "$EVALUATION_PROTOCOL_FREEZE" \
    --repo-root "$REPO_ROOT" \
    --scope external_benchmarks_unopened \
    --output "$FREEZE_BINDING_RECEIPT" \
    "${mode[@]}"
}

# Bind the canonical external-evaluation freeze before even classifying the
# internal dev bank.  This stage is explicitly not an external benchmark run.
verify_or_publish_freeze_binding

# Reject a changed/unregistered dev bank before the expensive descriptor pass.
# This hashes only the small embedding artifact and sidecar; the full loader
# performs the deeper model/vocabulary provenance validation once per split.
DEV_BANK_FAMILY="$(
  bash radio_gs/scripts/run_repo_python.sh - \
    "$DEV_TEXT_BANK" "$DEV_TEXT_BANK_MANIFEST" <<'PY'
import sys
from pathlib import Path

from radio_gs.scripts.eval_text_response_fidelity_gate import (
    classify_formal_text_bank_pair,
)

print(classify_formal_text_bank_pair(Path(sys.argv[1]), Path(sys.argv[2]), "dev"))
PY
)"

SELECTED_CANDIDATE="$({
  bash radio_gs/scripts/run_repo_python.sh - "$PROMOTION_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
selected = payload.get("selected_candidate")
if (
    payload.get("artifact_type")
    == "surface_c1024_attention_pooling_postcache_continuation"
    and payload.get("selected_variant") == "joint_attention_v1"
    and payload.get("selection_status") == "joint_attention_retained"
    and payload.get("promotion_gate_passed") is False
):
    selected = "context_c1024_geometric"
if not isinstance(selected, str) or not selected:
    raise SystemExit("Surface authority lacks a frozen text-response candidate")
print(selected)
PY
})"

RESPONSE_CANDIDATE_ROOT="${RESPONSE_CANDIDATE_ROOT:?set RESPONSE_CANDIDATE_ROOT to the completed three-seed distill root}"
RESPONSE_CHECKPOINTS=()
for seed in 0 1 2; do
  checkpoint="$RESPONSE_CANDIDATE_ROOT/readouts/${SELECTED_CANDIDATE}_text_response_seed${seed}.pt"
  if [[ ! -f "$checkpoint" || ! -f "${checkpoint}.json" ]]; then
    echo "missing response-distill seed-$seed checkpoint/sidecar: $checkpoint" >&2
    exit 2
  fi
  RESPONSE_CHECKPOINTS+=("$checkpoint")
done

mkdir -p "$OUTPUT_ROOT/descriptors" "$OUTPUT_ROOT/dev" "$OUTPUT_ROOT/logs"
PLAN="$OUTPUT_ROOT/promotion_plan.json"
bash radio_gs/scripts/run_repo_python.sh "$COMPANION" preflight \
  --promotion-manifest "$PROMOTION_MANIFEST" \
  --promotion-completion "$PROMOTION_COMPLETION" \
  --response-checkpoint "${RESPONSE_CHECKPOINTS[0]}" \
  --response-checkpoint "${RESPONSE_CHECKPOINTS[1]}" \
  --response-checkpoint "${RESPONSE_CHECKPOINTS[2]}" \
  --radio-checkpoint "$RADIO_CHECKPOINT" \
  --output "$PLAN" \
  >"$OUTPUT_ROOT/logs/preflight.log" 2>&1

mapfile -t VALIDATION_CACHES < <(
  bash radio_gs/scripts/run_repo_python.sh - "$PLAN" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for record in payload["validation_caches"]:
    print(record["path"])
PY
)
if [[ "${#VALIDATION_CACHES[@]}" -eq 0 ]]; then
  echo "promotion plan contains no validation caches" >&2
  exit 2
fi

plan_value() {
  local role="$1"
  local seed="$2"
  local field="$3"
  bash radio_gs/scripts/run_repo_python.sh - "$PLAN" "$role" "$seed" "$field" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
role, seed, field = sys.argv[2], int(sys.argv[3]), sys.argv[4]
rows = {int(row["seed"]): row for row in payload[role]}
print(rows[seed][field])
PY
}

CONTROL_METHOD_ID="$(bash radio_gs/scripts/run_repo_python.sh - "$PLAN" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["control_method_id"])
PY
)"
CANDIDATE_METHOD_ID="$(bash radio_gs/scripts/run_repo_python.sh - "$PLAN" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["candidate_method_id"])
PY
)"

CACHE_ARGS=()
for cache in "${VALIDATION_CACHES[@]}"; do
  CACHE_ARGS+=(--validation-cache "$cache")
done

CONTROL_DESCRIPTORS=()
CANDIDATE_DESCRIPTORS=()
for seed in 0 1 2; do
  control_checkpoint="$(plan_value control "$seed" checkpoint)"
  candidate_checkpoint="$(plan_value candidate "$seed" checkpoint)"
  control_descriptor="$OUTPUT_ROOT/descriptors/control_seed${seed}.pt"
  candidate_descriptor="$OUTPUT_ROOT/descriptors/candidate_seed${seed}.pt"
  if [[ ! -s "$control_descriptor" ]]; then
    bash radio_gs/scripts/run_repo_python.sh "$MATERIALIZER" \
      "${CACHE_ARGS[@]}" \
      --readout-checkpoint "$control_checkpoint" \
      --readout-binding-manifest "$PROMOTION_MANIFEST" \
      --radio-checkpoint "$RADIO_CHECKPOINT" \
      --method-id "$CONTROL_METHOD_ID" \
      --device cpu \
      --output "$control_descriptor" \
      >"$OUTPUT_ROOT/logs/materialize_control_seed${seed}.log" 2>&1
  fi
  if [[ ! -s "$candidate_descriptor" ]]; then
    bash radio_gs/scripts/run_repo_python.sh "$MATERIALIZER" \
      "${CACHE_ARGS[@]}" \
      --readout-checkpoint "$candidate_checkpoint" \
      --radio-checkpoint "$RADIO_CHECKPOINT" \
      --method-id "$CANDIDATE_METHOD_ID" \
      --device cpu \
      --output "$candidate_descriptor" \
      >"$OUTPUT_ROOT/logs/materialize_candidate_seed${seed}.log" 2>&1
  fi
  CONTROL_DESCRIPTORS+=("$control_descriptor")
  CANDIDATE_DESCRIPTORS+=("$candidate_descriptor")
done

CONTROL_DESCRIPTOR_ARGS=()
CANDIDATE_DESCRIPTOR_ARGS=()
for seed in 0 1 2; do
  CONTROL_DESCRIPTOR_ARGS+=(--control-descriptor "${CONTROL_DESCRIPTORS[$seed]}")
  CANDIDATE_DESCRIPTOR_ARGS+=(--candidate-descriptor "${CANDIDATE_DESCRIPTORS[$seed]}")
done

CONTROL_DEV_REPORTS=()
CANDIDATE_DEV_REPORTS=()
DEV_EVALUATE_ARGS=()
for seed in 0 1 2; do
  control_report="$OUTPUT_ROOT/dev/control_seed${seed}.json"
  candidate_report="$OUTPUT_ROOT/dev/candidate_seed${seed}.json"
  if [[ ! -s "$control_report" ]]; then
    DEV_EVALUATE_ARGS+=(--descriptors "${CONTROL_DESCRIPTORS[$seed]}" --output "$control_report")
  fi
  if [[ ! -s "$candidate_report" ]]; then
    DEV_EVALUATE_ARGS+=(--descriptors "${CANDIDATE_DESCRIPTORS[$seed]}" --output "$candidate_report")
  fi
  CONTROL_DEV_REPORTS+=("$control_report")
  CANDIDATE_DEV_REPORTS+=("$candidate_report")
done
if [[ "${#DEV_EVALUATE_ARGS[@]}" -gt 0 ]]; then
  bash radio_gs/scripts/run_repo_python.sh "$TEXT_GATE" evaluate-many \
    "${DEV_EVALUATE_ARGS[@]}" \
    --text-bank "$DEV_TEXT_BANK" \
    --text-bank-manifest "$DEV_TEXT_BANK_MANIFEST" \
    --query-split dev \
    >"$OUTPUT_ROOT/logs/dev_evaluate_many.log" 2>&1
fi

DEV_GATE="$OUTPUT_ROOT/dev/paired_gate.json"
if [[ ! -s "$DEV_GATE" ]]; then
  bash radio_gs/scripts/run_repo_python.sh "$TEXT_GATE" gate \
    --phase dev \
    --control-report "${CONTROL_DEV_REPORTS[0]}" \
    --control-report "${CONTROL_DEV_REPORTS[1]}" \
    --control-report "${CONTROL_DEV_REPORTS[2]}" \
    --candidate-report "${CANDIDATE_DEV_REPORTS[0]}" \
    --candidate-report "${CANDIDATE_DEV_REPORTS[1]}" \
    --candidate-report "${CANDIDATE_DEV_REPORTS[2]}" \
    --required-seeds 0,1,2 \
    --minimum-improved-seeds 2 \
    --bootstrap-samples 2000 \
    --bootstrap-seed 20260731 \
    --quality-noninferiority-tolerance 0.005 \
    --output "$DEV_GATE" \
    >"$OUTPUT_ROOT/logs/dev_gate.log" 2>&1
fi

CONTROL_DEV_REPORT_ARGS=()
CANDIDATE_DEV_REPORT_ARGS=()
for seed in 0 1 2; do
  CONTROL_DEV_REPORT_ARGS+=(--control-report "${CONTROL_DEV_REPORTS[$seed]}")
  CANDIDATE_DEV_REPORT_ARGS+=(--candidate-report "${CANDIDATE_DEV_REPORTS[$seed]}")
done
DEV_MANIFEST="$OUTPUT_ROOT/dev_decision.json"
DEV_COMPLETION="$OUTPUT_ROOT/dev_decision.complete.json"
bash radio_gs/scripts/run_repo_python.sh "$COMPANION" finalize-dev \
  --plan "$PLAN" \
  "${CONTROL_DESCRIPTOR_ARGS[@]}" \
  "${CANDIDATE_DESCRIPTOR_ARGS[@]}" \
  "${CONTROL_DEV_REPORT_ARGS[@]}" \
  "${CANDIDATE_DEV_REPORT_ARGS[@]}" \
  --gate "$DEV_GATE" \
  --text-bank "$DEV_TEXT_BANK" \
  --text-bank-manifest "$DEV_TEXT_BANK_MANIFEST" \
  --output "$DEV_MANIFEST" \
  --completion "$DEV_COMPLETION" \
  >"$OUTPUT_ROOT/logs/finalize_dev.log" 2>&1

verify_or_publish_freeze_binding

DEV_DECISION="$(bash radio_gs/scripts/run_repo_python.sh - "$DEV_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["decision"])
PY
)"
if [[ "$DEV_DECISION" == "reject_no_audit" ]]; then
  date -Iseconds >"$OUTPUT_ROOT/dev_rejected.complete"
  verify_or_publish_freeze_binding
  exit 0
fi
if [[ "$DEV_DECISION" != "promote_audit_required" ]]; then
  echo "unexpected frozen dev decision: $DEV_DECISION" >&2
  exit 1
fi

AUDIT_TEXT_BANK="${AUDIT_TEXT_BANK:?dev promoted; set AUDIT_TEXT_BANK for the one audit confirmation}"
AUDIT_TEXT_BANK_MANIFEST="${AUDIT_TEXT_BANK_MANIFEST:?dev promoted; set AUDIT_TEXT_BANK_MANIFEST}"
for required in "$AUDIT_TEXT_BANK" "$AUDIT_TEXT_BANK_MANIFEST"; do
  if [[ ! -f "$required" ]]; then
    echo "missing audit confirmation input: $required" >&2
    exit 2
  fi
done
AUDIT_BANK_FAMILY="$(
  bash radio_gs/scripts/run_repo_python.sh - \
    "$AUDIT_TEXT_BANK" "$AUDIT_TEXT_BANK_MANIFEST" <<'PY'
import sys
from pathlib import Path

from radio_gs.scripts.eval_text_response_fidelity_gate import (
    classify_formal_text_bank_pair,
)

print(classify_formal_text_bank_pair(Path(sys.argv[1]), Path(sys.argv[2]), "audit"))
PY
)"
if [[ "$AUDIT_BANK_FAMILY" != "$DEV_BANK_FAMILY" ]]; then
  echo "audit text-bank family differs from frozen dev family" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT/audit"

CONTROL_AUDIT_REPORTS=()
CANDIDATE_AUDIT_REPORTS=()
AUDIT_EVALUATE_ARGS=()
for seed in 0 1 2; do
  control_report="$OUTPUT_ROOT/audit/control_seed${seed}.json"
  candidate_report="$OUTPUT_ROOT/audit/candidate_seed${seed}.json"
  if [[ ! -s "$control_report" ]]; then
    AUDIT_EVALUATE_ARGS+=(--descriptors "${CONTROL_DESCRIPTORS[$seed]}" --output "$control_report")
  fi
  if [[ ! -s "$candidate_report" ]]; then
    AUDIT_EVALUATE_ARGS+=(--descriptors "${CANDIDATE_DESCRIPTORS[$seed]}" --output "$candidate_report")
  fi
  CONTROL_AUDIT_REPORTS+=("$control_report")
  CANDIDATE_AUDIT_REPORTS+=("$candidate_report")
done
if [[ "${#AUDIT_EVALUATE_ARGS[@]}" -gt 0 ]]; then
  bash radio_gs/scripts/run_repo_python.sh "$TEXT_GATE" evaluate-many \
    "${AUDIT_EVALUATE_ARGS[@]}" \
    --text-bank "$AUDIT_TEXT_BANK" \
    --text-bank-manifest "$AUDIT_TEXT_BANK_MANIFEST" \
    --query-split audit \
    >"$OUTPUT_ROOT/logs/audit_evaluate_many.log" 2>&1
fi

AUDIT_GATE="$OUTPUT_ROOT/audit/paired_gate.json"
if [[ ! -s "$AUDIT_GATE" ]]; then
  bash radio_gs/scripts/run_repo_python.sh "$TEXT_GATE" gate \
    --phase audit \
    --control-report "${CONTROL_AUDIT_REPORTS[0]}" \
    --control-report "${CONTROL_AUDIT_REPORTS[1]}" \
    --control-report "${CONTROL_AUDIT_REPORTS[2]}" \
    --candidate-report "${CANDIDATE_AUDIT_REPORTS[0]}" \
    --candidate-report "${CANDIDATE_AUDIT_REPORTS[1]}" \
    --candidate-report "${CANDIDATE_AUDIT_REPORTS[2]}" \
    --required-seeds 0,1,2 \
    --minimum-improved-seeds 2 \
    --bootstrap-samples 2000 \
    --bootstrap-seed 20260731 \
    --quality-noninferiority-tolerance 0.005 \
    --output "$AUDIT_GATE" \
    >"$OUTPUT_ROOT/logs/audit_gate.log" 2>&1
fi

CONTROL_AUDIT_REPORT_ARGS=()
CANDIDATE_AUDIT_REPORT_ARGS=()
for seed in 0 1 2; do
  CONTROL_AUDIT_REPORT_ARGS+=(--control-report "${CONTROL_AUDIT_REPORTS[$seed]}")
  CANDIDATE_AUDIT_REPORT_ARGS+=(--candidate-report "${CANDIDATE_AUDIT_REPORTS[$seed]}")
done
bash radio_gs/scripts/run_repo_python.sh "$COMPANION" finalize-audit \
  --plan "$PLAN" \
  "${CONTROL_DESCRIPTOR_ARGS[@]}" \
  "${CANDIDATE_DESCRIPTOR_ARGS[@]}" \
  "${CONTROL_AUDIT_REPORT_ARGS[@]}" \
  "${CANDIDATE_AUDIT_REPORT_ARGS[@]}" \
  --gate "$AUDIT_GATE" \
  --text-bank "$AUDIT_TEXT_BANK" \
  --text-bank-manifest "$AUDIT_TEXT_BANK_MANIFEST" \
  --dev-manifest "$DEV_MANIFEST" \
  --dev-completion "$DEV_COMPLETION" \
  --output "$OUTPUT_ROOT/audit_confirmation.json" \
  --completion "$OUTPUT_ROOT/audit_confirmation.complete.json" \
  >"$OUTPUT_ROOT/logs/finalize_audit.log" 2>&1

verify_or_publish_freeze_binding
date -Iseconds >"$OUTPUT_ROOT/promotion.complete"
