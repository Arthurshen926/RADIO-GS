#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260816/core_method_v1/spin9}"
RELATION_AUTHORITY="${RADIO_GS_GENERIC_RELATION_AUTHORITY:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/target_blind_typed_text_relation_authority_v1/fit_relation_indices.pt}"
EXPECTED_RELATION_SHA256="482e363bf31884e190b255cc0cf0996461400bcbb3cb3f8785fbc236da2702a9"

if [[ ! -r "$RELATION_AUTHORITY" ]] || [[ "$(sha256sum "$RELATION_AUTHORITY" | cut -d' ' -f1)" != "$EXPECTED_RELATION_SHA256" ]]; then
  echo "generic relation authority is unavailable or differs" >&2
  exit 2
fi

export RADIO_GS_GENERIC_RELATION_AUTHORITY="$RELATION_AUTHORITY"
export RADIO_GS_SPIN9_HOST_MEMORY_SLOTS="${RADIO_GS_SPIN9_HOST_MEMORY_SLOTS:-4}"
cd "$REPO_ROOT"

for scene in horns truck leaves fern; do
  echo "[$(date -Is)] $scene priority full start"
  if ! bash radio_gs/scripts/run_repo_python.sh \
    -m radio_gs.scripts.run_spin9_method_v1_scene \
    --scene "$scene" \
    --gpu 4 \
    --run-root "$RUN_ROOT"; then
    echo "[$(date -Is)] $scene priority full FAILED" >&2
  fi
done
