#!/usr/bin/env bash

# Run official SAM3-only jobs in the environment that supplies its required
# recent PyTorch attention API.  This is intentionally separate from the
# repository's CPython runner: the latter also loads legacy gsplat/ScanNet
# extensions that are ABI-bound to Python 3.9.
set -euo pipefail

SAM3_PYTHON="${RADIO_GS_SAM3_PYTHON:-/root/miniconda3/envs/sam3/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SAM3_SOURCE="${RADIO_GS_SAM3_SOURCE:-/root/external/sam3}"

if [[ ! -x "$SAM3_PYTHON" ]]; then
    echo "Official SAM3 CPython is unavailable; set RADIO_GS_SAM3_PYTHON." >&2
    exit 127
fi
if [[ ! -d "$SAM3_SOURCE" ]]; then
    SAM3_SOURCE="/tmp/radio_gs_official_deps/sam3"
fi
if [[ ! -d "$SAM3_SOURCE" ]]; then
    echo "Official SAM3 source is unavailable; set RADIO_GS_SAM3_SOURCE." >&2
    exit 127
fi

export PYTHONPATH="${SAM3_SOURCE}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "$SAM3_PYTHON" "$@"
