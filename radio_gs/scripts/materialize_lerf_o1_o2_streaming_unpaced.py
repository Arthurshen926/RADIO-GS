#!/usr/bin/env python3
"""Run the frozen O1/O2 streamer without artificial per-batch pacing.

The numerical implementation remains in ``materialize_lerf_o1_o2_streaming``.
This entrypoint only removes the CUDA synchronize + 50 ms sleep that followed
every projection batch.  Thermal safety remains external and fail-closed in
``run_with_gpu_thermal_guard.sh`` (300 s polling, 88 C hard cutoff).

Keeping this as a separate, hash-bound entrypoint preserves the provenance of
the already completed paced Ramen run while making the resource-policy change
explicit for later scenes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from radio_gs.scripts import materialize_lerf_o1_o2_streaming as _core
from radio_gs.utils.immutable_artifacts import canonical_json_sha256, file_record


CORE_IMPLEMENTATION = file_record(Path(_core.__file__).resolve())
PACING_SECONDS_PER_PROJECTION_BATCH = 0.0
_CORE_METHOD_CONTRACT = _core.method_contract
_CORE_PROJECT_VIEW = _core._project_view


def method_contract() -> dict[str, Any]:
    """Return the unchanged method contract plus explicit execution policy."""

    contract = dict(_CORE_METHOD_CONTRACT())
    contract.update(
        {
            "streaming_core_implementation": dict(CORE_IMPLEMENTATION),
            "projection_pacing_seconds_per_batch": (
                PACING_SECONDS_PER_PROJECTION_BATCH
            ),
            "projection_pacing_affects_method_numerics": False,
            "thermal_safety_owner": "external_300s_hard88_guard",
            "core_execution_physical_gpu_field_interpretation": (
                "logical_cuda_ordinal_within_CUDA_VISIBLE_DEVICES"
            ),
            "host_physical_gpu_bound_by_external_thermal_guard": True,
        }
    )
    return contract


def _project_view_unpaced(**kwargs: Any):
    """Delegate the exact projection while disabling only sync/sleep pacing."""

    kwargs["pace"] = False
    return _CORE_PROJECT_VIEW(**kwargs)


def _install_unpaced_contract() -> None:
    # The core validators intentionally resolve these module globals at call
    # time.  Point them at this auditable entrypoint while retaining the core
    # hash inside ``method_contract`` above.
    _core.PACING_SECONDS_PER_PROJECTION_BATCH = (
        PACING_SECONDS_PER_PROJECTION_BATCH
    )
    _core.method_contract = method_contract
    _core.METHOD_CONTRACT_SHA256 = canonical_json_sha256(method_contract())
    _core._project_view = _project_view_unpaced
    _core.__file__ = str(Path(__file__).resolve())


def main() -> None:
    _install_unpaced_contract()
    _core.main()


if __name__ == "__main__":
    main()


__all__ = [
    "CORE_IMPLEMENTATION",
    "PACING_SECONDS_PER_PROJECTION_BATCH",
    "main",
    "method_contract",
]
