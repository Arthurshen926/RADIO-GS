#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PFPR_RESULT="${PFPR_RESULT:?set complete PFPR results.json}"
AGILE_RESULT="${AGILE_RESULT:?set complete merged AGILE results.json}"
OUTPUT="${OUTPUT:?set final Markdown report path}"

while [[ ! -s "$PFPR_RESULT" || ! -s "$AGILE_RESULT" ]]; do sleep 30; done

bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.scripts.build_full_benchmark_final_table \
  --pfpr-result "$PFPR_RESULT" \
  --agile-result "$AGILE_RESULT" \
  --output "$OUTPUT"
