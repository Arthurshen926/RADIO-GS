#!/usr/bin/env bash

set -euo pipefail

# Source snapshots are executed by root on the shared host.  Unix read-only
# mode bits do not prevent root from creating ``__pycache__`` entries, so the
# interpreter itself must disable bytecode writes before importing the package.
# This also keeps runtime-closure permission audits stable across invocations.
export PYTHONDONTWRITEBYTECODE=1
# Numba's ``cache=True`` path is independent of CPython bytecode policy and
# otherwise creates a writable ``__pycache__`` directory beside read-only
# sources.  Keep compiled CPU artifacts in a dedicated external cache.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/root/.cache/radio_gs/numba}"

RADIO_GS_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export RADIO_GS_REPO_ROOT
cd "$RADIO_GS_REPO_ROOT"

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

# CUDA compatibility packages in a Conda environment can expose a newer
# libcuda than the loaded kernel module.  Preload the exact host-driver library
# when it is available so CUDA does not fail with error 804 before Python can
# select a device.  An explicitly set (including empty)
# RADIO_GS_DRIVER_LIBRARY remains authoritative.
if [[ -z "${RADIO_GS_DRIVER_LIBRARY+x}" ]]; then
    NVIDIA_DRIVER_VERSION=""
    if [[ -r /proc/driver/nvidia/version ]]; then
        NVIDIA_DRIVER_VERSION="$(
            awk '
                /Kernel Module/ {
                    for (field_index = 1; field_index <= NF; ++field_index) {
                        if ($field_index ~ /^[0-9]+([.][0-9]+)+$/) {
                            print $field_index
                            exit
                        }
                    }
                }
            ' /proc/driver/nvidia/version
        )"
    fi
    RADIO_GS_DRIVER_LIBRARY=""
    if [[ -n "$NVIDIA_DRIVER_VERSION" ]]; then
        NVIDIA_DRIVER_CANDIDATE="/usr/lib/x86_64-linux-gnu/libcuda.so.${NVIDIA_DRIVER_VERSION}"
        if [[ -f "$NVIDIA_DRIVER_CANDIDATE" ]]; then
            RADIO_GS_DRIVER_LIBRARY="$NVIDIA_DRIVER_CANDIDATE"
        fi
    fi
fi

if [[ -n "${RADIO_GS_LD_LIBRARY_PATH:-}" ]]; then
    RADIO_GS_LD_LIBRARY_PATH="$RADIO_GS_LD_LIBRARY_PATH"
elif [[ "$RADIO_GS_PYTHON" == "$LEGACY_PYTHON" ]]; then
    RADIO_GS_LD_LIBRARY_PATH="/root/miniconda3/envs/iclpose/lib:/root/miniconda3/pkgs/python-3.9.25-h0dcde21_1/lib"
else
    RADIO_GS_LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:/root/miniconda3/envs/iclpose/lib"
fi

export LD_LIBRARY_PATH="${RADIO_GS_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -n "$RADIO_GS_DRIVER_LIBRARY" ]]; then
    export LD_PRELOAD="${RADIO_GS_DRIVER_LIBRARY}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
if [[ -d "$RADIO_GS_SAM3_SOURCE" ]]; then
    export PYTHONPATH="${RADIO_GS_REPO_ROOT}:${RADIO_GS_SAM3_SOURCE}:${RADIO_GS_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
else
    export PYTHONPATH="${RADIO_GS_REPO_ROOT}:${RADIO_GS_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
fi

exec "$RADIO_GS_PYTHON" "$@"
