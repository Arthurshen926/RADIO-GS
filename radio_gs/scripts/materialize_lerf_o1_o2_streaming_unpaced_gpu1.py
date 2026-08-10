#!/usr/bin/env python3
"""Run the unpaced frozen O1/O2 streamer on host physical GPU 1.

The numerical implementation and no-pacing policy remain hash-bound to
``materialize_lerf_o1_o2_streaming`` and
``materialize_lerf_o1_o2_streaming_unpaced`` respectively.  This entrypoint
only makes the two device namespaces explicit in the frozen authority:

* host/NVML physical GPU: 1;
* ``CUDA_VISIBLE_DEVICES``: ``"1"``;
* program-local PyTorch device: ``cuda:0``.

The external thermal guard owns host-GPU temperature enforcement.  A small
translation is used only while delegating the otherwise unchanged, exhaustive
core input validator, whose historical execution record used logical ordinal
zero in all three fields.
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from radio_gs.scripts import materialize_lerf_o1_o2_streaming as _core
from radio_gs.scripts import materialize_lerf_o1_o2_streaming_unpaced as _unpaced
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    write_frozen_json,
)


UNPACED_IMPLEMENTATION = file_record(Path(_unpaced.__file__).resolve())
HOST_PHYSICAL_GPU = 1
CUDA_VISIBLE_DEVICES = "1"
PROGRAM_DEVICE = "cuda:0"
_CORE_PREPARE_INPUTS = _core.prepare_inputs
_CORE_MATERIALIZE = _core.materialize


def method_contract() -> dict[str, Any]:
    """Bind the unchanged unpaced method plus the explicit GPU namespace."""

    contract = dict(_unpaced.method_contract())
    contract.update(
        {
            "unpaced_streaming_entrypoint": dict(UNPACED_IMPLEMENTATION),
            "execution_authority_physical_gpu_semantics": "host_nvidia_smi_index",
            "host_physical_gpu": HOST_PHYSICAL_GPU,
            "host_cuda_visible_devices": CUDA_VISIBLE_DEVICES,
            "program_logical_device": PROGRAM_DEVICE,
            "device_namespace_affects_method_numerics": False,
        }
    )
    return contract


def expected_execution() -> dict[str, Any]:
    return {
        "physical_gpu": HOST_PHYSICAL_GPU,
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "program_device": PROGRAM_DEVICE,
        "projection_batch_candidates": list(_core.PREFLIGHT_BATCH_CANDIDATES),
        "pacing_seconds_per_projection_batch": (
            _unpaced.PACING_SECONDS_PER_PROJECTION_BATCH
        ),
        "thermal_poll_seconds": 300,
        "soft_pause_temperature_c": 0,
        "maximum_temperature_c": 88,
    }


def _core_logical_execution() -> dict[str, Any]:
    execution = expected_execution()
    execution.update(
        {
            "physical_gpu": 0,
            "cuda_visible_devices": "0",
            "program_device": "cuda:0",
        }
    )
    return execution


def _translate_authority_for_core_validation(
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the host authority and translate only its device namespace."""

    if authority.get("execution") != expected_execution():
        raise ValueError("GPU1 O1/O2 execution authority differs")
    translated = deepcopy(dict(authority))
    translated["execution"] = _core_logical_execution()
    return translated


def prepare_inputs(
    authority_path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    """Validate GPU1 authority, then reuse every core lineage validator."""

    authority, digest, source = load_json_object(
        authority_path,
        expected_sha256=expected_sha256,
        label="GPU1 O1/O2 streaming execution authority",
    )
    if not isinstance(authority, Mapping):
        raise ValueError("GPU1 O1/O2 execution authority must be an object")
    translated = _translate_authority_for_core_validation(authority)
    with tempfile.TemporaryDirectory(prefix="radio_gs_o1_o2_gpu1_validate_") as tmp:
        candidate = Path(tmp) / "logical_core_authority.json"
        write_frozen_json(candidate, translated)
        record = file_record(candidate)
        prepared = _CORE_PREPARE_INPUTS(
            record["path"], expected_sha256=record["sha256"]
        )
    prepared["authority"] = dict(authority)
    prepared["authority_record"] = {"path": str(source), "sha256": digest}
    return prepared


def build_authority(args: Any) -> dict[str, Any]:
    """Build a first-writer-only authority that names host physical GPU 1."""

    authority_output = _core._new(args.authority_output, label="O1/O2 authority")
    output_root = Path(args.output_dir).expanduser().resolve()
    if str(output_root) != str(args.output_dir):
        raise ValueError("O1/O2 output directory must be canonical absolute")
    names = {
        "teacher_mean": f"{args.scene_id}_teacher_mean_fp16.pt",
        "o1_positive": f"{args.scene_id}_o1_positive.pt",
        "o1_negative": f"{args.scene_id}_o1_negative.pt",
        "o2_positive": f"{args.scene_id}_o2_positive.pt",
        "o2_negative": f"{args.scene_id}_o2_negative.pt",
        "result": f"{args.scene_id}_o1_o2_streaming_result.json",
    }
    outputs = {
        name: str(_core._new(output_root / filename, label=f"O1/O2 {name}"))
        for name, filename in names.items()
    }
    inputs: dict[str, Any] = {}
    for name in (
        "base_descriptor",
        "responsibility_authority",
        "feature_manifest",
        "scene_config",
        "renderer_geometry_checkpoint",
        "official_radio_checkpoint",
        "positive_text",
        "negative_text",
        "o0_positive",
        "o0_negative",
        "frozen_metric_config",
    ):
        inputs[name] = _core._record(
            getattr(args, name),
            getattr(args, f"{name}_sha256"),
            label=name.replace("_", " "),
        )
    contract = method_contract()
    authority = {
        "schema": _core.AUTHORITY_SCHEMA,
        "schema_version": _core.SCHEMA_VERSION,
        "status": "authorized_source_only_premetric_o1_o2_streaming",
        "scene_id": str(args.scene_id),
        "implementation": file_record(Path(__file__).resolve()),
        "method_contract": contract,
        "method_contract_sha256": canonical_json_sha256(contract),
        "feature_output_bundle_sha256": str(args.feature_output_bundle_sha256),
        "inputs": inputs,
        "outputs": outputs,
        "execution": expected_execution(),
        "query_free_materialization_authorized": True,
        "metric_execution_authorized": False,
        "access_audit": _core.access_audit(),
    }
    write_frozen_json(authority_output, authority)
    record = file_record(authority_output)
    prepare_inputs(record["path"], expected_sha256=record["sha256"])
    return {"status": "authorized", "authority": record, "outputs": outputs}


def materialize(args: Any) -> dict[str, Any]:
    """Fail closed unless host GPU 1 is the sole visible CUDA device."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != CUDA_VISIBLE_DEVICES:
        raise RuntimeError(
            "GPU1 O1/O2 materialization requires CUDA_VISIBLE_DEVICES=1; "
            f"received {visible!r}"
        )
    return _CORE_MATERIALIZE(args)


def _install_gpu1_contract() -> None:
    _unpaced._install_unpaced_contract()
    _core.method_contract = method_contract
    _core.METHOD_CONTRACT_SHA256 = canonical_json_sha256(method_contract())
    _core.prepare_inputs = prepare_inputs
    _core.build_authority = build_authority
    _core.materialize = materialize
    _core.__file__ = str(Path(__file__).resolve())


def main() -> None:
    _install_gpu1_contract()
    _core.main()


if __name__ == "__main__":
    main()


__all__ = [
    "CUDA_VISIBLE_DEVICES",
    "HOST_PHYSICAL_GPU",
    "PROGRAM_DEVICE",
    "UNPACED_IMPLEMENTATION",
    "build_authority",
    "expected_execution",
    "main",
    "materialize",
    "method_contract",
    "prepare_inputs",
]
