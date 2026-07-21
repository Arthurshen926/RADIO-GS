#!/usr/bin/env bash

set -euo pipefail

LEGACY_PYTHON="/root/miniconda3/pkgs/python-3.9.25-h0dcde21_1/bin/python3.9"
FALLBACK_PYTHON="/root/miniconda3/envs/cybersim_agent/bin/python3.9"

# The historical CPython package is occasionally removed by shared-machine
# maintenance.  Keep its exact environment when it exists, but fail over to
# the compatible CPython environment rather than silently invoking iclpose's
# PyPy interpreter (which cannot reliably load the project's Torch stack).
if [[ -n "${RADIO_GS_PYTHON:-}" ]]; then
    RADIO_GS_PYTHON="$RADIO_GS_PYTHON"
elif [[ -x "$LEGACY_PYTHON" ]]; then
    RADIO_GS_PYTHON="$LEGACY_PYTHON"
elif [[ -x "$FALLBACK_PYTHON" ]]; then
    RADIO_GS_PYTHON="$FALLBACK_PYTHON"
else
    echo "No compatible CPython found; set RADIO_GS_PYTHON explicitly." >&2
    exit 127
fi

RADIO_GS_SITE_PACKAGES="${RADIO_GS_SITE_PACKAGES:-/root/miniconda3/envs/iclpose/lib/python3.9/site-packages}"
RADIO_GS_SAM3_SOURCE="${RADIO_GS_SAM3_SOURCE:-/tmp/radio_gs_official_deps/sam3}"
if [[ -n "${RADIO_GS_LD_LIBRARY_PATH:-}" ]]; then
    RADIO_GS_LD_LIBRARY_PATH="$RADIO_GS_LD_LIBRARY_PATH"
elif [[ "$RADIO_GS_PYTHON" == "$LEGACY_PYTHON" ]]; then
    RADIO_GS_LD_LIBRARY_PATH="/root/miniconda3/envs/iclpose/lib:/root/miniconda3/pkgs/python-3.9.25-h0dcde21_1/lib"
else
    RADIO_GS_LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:/root/miniconda3/envs/iclpose/lib"
fi

export LD_LIBRARY_PATH="${RADIO_GS_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -d "$RADIO_GS_SAM3_SOURCE" ]]; then
    export PYTHONPATH="${RADIO_GS_SAM3_SOURCE}:${RADIO_GS_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
else
    export PYTHONPATH="${RADIO_GS_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
fi

exec "$RADIO_GS_PYTHON" "$@"
