#!/usr/bin/env bash

set -euo pipefail

RADIO_GS_PYTHON="${RADIO_GS_PYTHON:-/root/miniconda3/pkgs/python-3.9.25-h0dcde21_1/bin/python3.9}"
RADIO_GS_SITE_PACKAGES="${RADIO_GS_SITE_PACKAGES:-/root/miniconda3/envs/iclpose/lib/python3.9/site-packages}"
RADIO_GS_LD_LIBRARY_PATH="${RADIO_GS_LD_LIBRARY_PATH:-/root/miniconda3/envs/iclpose/lib:/root/miniconda3/pkgs/python-3.9.25-h0dcde21_1/lib}"

export LD_LIBRARY_PATH="${RADIO_GS_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${RADIO_GS_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"

exec "$RADIO_GS_PYTHON" "$@"
