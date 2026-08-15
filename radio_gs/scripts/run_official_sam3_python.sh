#!/usr/bin/env bash

# Run official SAM3-only jobs in the environment that supplies its required
# recent PyTorch attention API.  This is intentionally separate from the
# repository's CPython runner: the latter also loads legacy gsplat/ScanNet
# extensions that are ABI-bound to Python 3.9.
set -euo pipefail

SAM3_PYTHON="${RADIO_GS_SAM3_PYTHON:-/root/miniconda3/envs/sam3/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SAM3_SOURCE="${RADIO_GS_SAM3_SOURCE:-/root/external/sam3}"
NVIDIA_VERSION_FILE="${RADIO_GS_NVIDIA_VERSION_FILE:-/proc/driver/nvidia/version}"
DRIVER_LIBRARY_DIR="${RADIO_GS_DRIVER_LIBRARY_DIR:-/usr/lib/x86_64-linux-gnu}"

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

# The container's generic libcuda.so.1 can be newer than the loaded host
# kernel module and make PyTorch fail with CUDA error 804.  Bind only this
# process to the exact kernel-matched userspace library when it exists.  An
# explicitly set RADIO_GS_DRIVER_LIBRARY (including an empty value) remains
# authoritative so callers can disable or replace the automatic selection.
if [[ -z "${RADIO_GS_DRIVER_LIBRARY+x}" ]]; then
    NVIDIA_DRIVER_VERSION=""
    if [[ -r "$NVIDIA_VERSION_FILE" ]]; then
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
            ' "$NVIDIA_VERSION_FILE"
        )"
    fi
    RADIO_GS_DRIVER_LIBRARY=""
    if [[ -n "$NVIDIA_DRIVER_VERSION" ]]; then
        DRIVER_CANDIDATE="$DRIVER_LIBRARY_DIR/libcuda.so.$NVIDIA_DRIVER_VERSION"
        if [[ -f "$DRIVER_CANDIDATE" ]]; then
            RADIO_GS_DRIVER_LIBRARY="$DRIVER_CANDIDATE"
        fi
    fi
fi

export PYTHONPATH="${SAM3_SOURCE}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "$RADIO_GS_DRIVER_LIBRARY" ]]; then
    export LD_PRELOAD="${RADIO_GS_DRIVER_LIBRARY}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
exec "$SAM3_PYTHON" "$@"
