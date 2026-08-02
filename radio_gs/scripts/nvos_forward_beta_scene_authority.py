#!/usr/bin/env python3
"""CPU-only GPU/scene authority for ``nvos-forward-beta-coverage-v1``.

The authority is deliberately separate from the frozen registered-region-v3
authority.  It validates the new candidate and its pre-generated inline
protocol receipt before launch, binds the actual evaluator argv, and closes a
scene only after the result, CUDA identity, exclusive-owner audit, and
owner-clear postcheck agree.  It never derives protocol authority at runtime
and exposes no exactness or hard-seed override.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import io
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from radio_gs.scripts.bind_nvos_forward_beta_protocol_authority import (
    canonical_json_sha256,
    validate_authority_payload,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    stable_descriptor_load,
    validate_file_record,
    write_frozen_json,
)


CANDIDATE_ID = "nvos-forward-beta-coverage-v1"
SCENE_GPU_ASSIGNMENT = {
    "policy": "fixed_before_execution_no_target_metric_input",
    "gpu0": ["fern", "flower", "fortress", "horns_center"],
    "gpu1": ["horns_left", "leaves", "orchids", "trex"],
}
MAXIMUM_CONCURRENT_SCENE_EVALUATORS = 1
HOST_MEMORY_POLICY = "fixed_mapping_single_scene_resident_v1"
SERIAL_SCENE_GPU_PLAN = [
    {"physical_gpu": 0, "scene": "fern"},
    {"physical_gpu": 1, "scene": "horns_left"},
    {"physical_gpu": 0, "scene": "flower"},
    {"physical_gpu": 1, "scene": "leaves"},
    {"physical_gpu": 0, "scene": "fortress"},
    {"physical_gpu": 1, "scene": "orchids"},
    {"physical_gpu": 0, "scene": "horns_center"},
    {"physical_gpu": 1, "scene": "trex"},
]
SCHEMA_VERSION = 1
COMMAND_ARTIFACT = "nvos-forward-beta-scene-command-v1"
POSTCHECK_ARTIFACT = "nvos-forward-beta-scene-postcheck-v1"
RECEIPT_ARTIFACT = "nvos-forward-beta-scene-receipt-v1"
RECEIPT_STATUS = (
    "beta_report_nonpromoted_cuda_attested_exclusive_owner_postchecked"
)
BETA_CUDA_ATTESTATION_ARTIFACT = "nvos-forward-beta-cuda-child-attestation-v1"
CUDA_DEVICE_ORDER = "PCI_BUS_ID"
GPU_OWNER_PID_NAMESPACE_MODE = "exclusive-singleton-after-clear-v1"
CUDA_ATTESTATION_MECHANISM = (
    "torch_cuda0_live_allocation_plus_nvidia_smi_exclusive_owner_"
    "with_container_host_pid_namespace_binding_plus_uuid_pci_v2"
)
OWNER_AUDIT_COLUMNS = (
    "timestamp", "gpu_uuid", "child_pgid", "owner_pids",
    "child_owner_pids", "foreign_owner_pids", "event",
)
TELEMETRY_COLUMNS = (
    "timestamp", "gpu", "bus_id", "temp_c", "power_w", "power_limit_w",
    "util_pct", "memory_mib", "pstate", "event",
)
CUDA_ATTESTATION_FIELDS = {
    "schema_version", "artifact_type", "status", "scene", "observed_epoch",
    "hostname", "environment", "expected_gpu", "torch_cuda",
    "process_namespace_pids", "nvidia_inventory_row",
    "nvidia_preallocation_owner_rows", "nvidia_compute_owner_rows",
    "owner_pid_binding", "attestation_mechanism",
}
EXPECTED_SCORING = {
    "score_semantics": "beta_centered_posterior",
    "prediction_representation": "continuous_beta_centered_posterior",
    "threshold": {"comparison": "greater_or_equal", "value": 0.0},
    "resize": "nearest",
}
EXPECTED_BLOCKERS = [
    "score_semantics_differs",
    "prediction_representation_differs",
]
CORE_ARGV = {
    "--support-mode": "canonical_support",
    "--registered-observation-fusion": "probability_mixture",
    "--registered-seed-unary-weight": 0.0,
    "--registered-readout-stage": "propagated",
    "--prompt-registration-mode": "raster_adjoint",
    "--prompt-registration-scale": 1.0,
    "--alpha-threshold": 0.0,
    "--feature-contribution-gamma": 1.0,
}


@dataclass(frozen=True)
class SceneAuthorityProfile:
    candidate_id: str
    forward_mode: str
    command_artifact: str
    postcheck_artifact: str
    receipt_artifact: str
    receipt_status: str


V1_PROFILE = SceneAuthorityProfile(
    candidate_id=CANDIDATE_ID,
    forward_mode="beta_coverage_v1",
    command_artifact=COMMAND_ARTIFACT,
    postcheck_artifact=POSTCHECK_ARTIFACT,
    receipt_artifact=RECEIPT_ARTIFACT,
    receipt_status=RECEIPT_STATUS,
)
V2_PROFILE = SceneAuthorityProfile(
    candidate_id="nvos-forward-beta-balanced-residual-v2",
    forward_mode="beta_balanced_residual_v2",
    command_artifact="nvos-forward-beta-v2-scene-command-v1",
    postcheck_artifact="nvos-forward-beta-v2-scene-postcheck-v1",
    receipt_artifact="nvos-forward-beta-v2-scene-receipt-v1",
    receipt_status=(
        "beta_v2_report_nonpromoted_cuda_attested_exclusive_owner_postchecked"
    ),
)


def _validate_sized_file_record(record: object, *, label: str) -> Path:
    """Validate an immutable ``path``/``bytes``/``sha256`` file record.

    ``path`` is a locator, so the returned path may have resolved parent-directory
    symlinks.  Authority comes from the exact byte count and digest, while the
    stable descriptor reader still rejects a final-component symlink and any
    identity or metadata change during the read.
    """

    _require(
        isinstance(record, Mapping)
        and set(record) == {"path", "bytes", "sha256"},
        f"{label} file record differs",
    )
    declared_bytes = record.get("bytes")
    _require(
        isinstance(declared_bytes, int)
        and not isinstance(declared_bytes, bool)
        and declared_bytes >= 0,
        f"{label} byte count differs",
    )
    observed_bytes, _, source = stable_descriptor_load(
        str(record.get("path", "")),
        lambda handle: int(os.fstat(handle.fileno()).st_size),
        expected_sha256=str(record.get("sha256", "")),
        label=label,
    )
    _require(observed_bytes == declared_bytes, f"{label} byte count differs")
    return source


FORBIDDEN_ARGV = {
    "--strict-unseen-protocol-exact-match",
    "--strict-unseen-exact-match",
    "--exact-protocol-override",
    "--registered-forward-protocol-authority",
    "--registered-forward-protocol-authority-sha256",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _contains_exact_override(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if "exact_override" in normalized or "caller_exact" in normalized:
                return True
            if _contains_exact_override(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_exact_override(child) for child in value)
    return False


def _gpu_identity(
    physical_index: object,
    gpu_uuid: object,
    gpu_bus_id: object,
) -> dict[str, object]:
    _require(
        isinstance(physical_index, int)
        and not isinstance(physical_index, bool)
        and physical_index in {0, 1},
        "physical GPU index must be GPU0 or GPU1",
    )
    _require(
        isinstance(gpu_uuid, str)
        and re.fullmatch(r"GPU-[0-9a-fA-F-]{32,}", gpu_uuid) is not None,
        "GPU UUID format differs",
    )
    _require(
        isinstance(gpu_bus_id, str)
        and re.fullmatch(
            r"(?:[0-9a-fA-F]{8}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}[.][0-7]",
            gpu_bus_id,
        )
        is not None,
        "GPU PCI bus format differs",
    )
    return {
        "physical_index": physical_index,
        "uuid": gpu_uuid,
        "pci_bus_id": gpu_bus_id,
    }


def _nvidia_query(arguments: Sequence[str]) -> list[list[str]]:
    completed = subprocess.run(
        ["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return [
        [field.strip() for field in row]
        for row in csv.reader(io.StringIO(completed.stdout))
        if row
    ]


def _namespace_process_ids() -> list[int]:
    values = {os.getpid()}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("NSpid:"):
            values.update(int(value) for value in line.split()[1:])
            break
    return sorted(values)


def _load_csv_rows(
    path: str | Path,
    *,
    columns: Sequence[str],
    label: str,
) -> tuple[list[dict[str, str]], str, Path]:
    def load(handle) -> list[dict[str, str]]:
        reader = csv.DictReader(io.StringIO(handle.read().decode("utf-8")))
        _require(tuple(reader.fieldnames or ()) == tuple(columns), f"{label} header differs")
        return [dict(row) for row in reader]

    rows, digest, source = stable_descriptor_load(path, load, label=label)
    _require(bool(rows), f"{label} is empty")
    return rows, digest, source


def _validate_owner_audit(path: str | Path, *, gpu_uuid: str) -> dict[str, Any]:
    rows, digest, source = _load_csv_rows(
        path, columns=OWNER_AUDIT_COLUMNS, label="forward-Beta GPU owner audit"
    )
    allowed_events = {
        "prelaunch_owner_clear", "runtime_owner_audit",
        "runtime_owner_audit_host_pid_singleton", "postexit_owner_clear",
    }
    child_pgids = {row["child_pgid"] for row in rows}
    runtime = [row for row in rows if row["event"].startswith("runtime_owner_audit")]

    def pids(value: str) -> list[str]:
        result = [item for item in value.split(";") if item]
        _require(all(item.isdigit() for item in result), "owner audit PID differs")
        return result

    _require(
        rows[0]["event"] == "prelaunch_owner_clear"
        and rows[-1]["event"] == "postexit_owner_clear"
        and len(child_pgids) == 1
        and next(iter(child_pgids)).isdigit()
        and all(row["gpu_uuid"] == gpu_uuid for row in rows)
        and all(row["event"] in allowed_events for row in rows)
        and not any(row["foreign_owner_pids"] for row in rows)
        and bool(runtime)
        and all(pids(row["owner_pids"]) == pids(row["child_owner_pids"]) for row in runtime)
        and any(pids(row["child_owner_pids"]) for row in runtime)
        and not any(rows[0][key] or rows[-1][key] for key in (
            "owner_pids", "child_owner_pids", "foreign_owner_pids"
        )),
        "owner audit does not prove exclusive child ownership",
    )
    return {
        "path": str(source), "sha256": digest, "sample_count": len(rows),
        "child_pgid": int(next(iter(child_pgids))),
        "child_owner_pids": sorted({pid for row in runtime for pid in pids(row["child_owner_pids"])}),
        "direct_child_owner_pids": sorted({
            pid for row in runtime if row["event"] == "runtime_owner_audit"
            for pid in pids(row["child_owner_pids"])
        }),
        "host_singleton_owner_pids": sorted({
            pid for row in runtime if row["event"] == "runtime_owner_audit_host_pid_singleton"
            for pid in pids(row["child_owner_pids"])
        }),
    }


def _validate_telemetry(
    path: str | Path, *, physical_index: int, gpu_bus_id: str
) -> dict[str, Any]:
    rows, digest, source = _load_csv_rows(
        path, columns=TELEMETRY_COLUMNS, label="forward-Beta GPU telemetry"
    )
    forbidden = ("abort", "failed", "unresponsive", "foreign_compute_owner")
    suffix = ":".join(gpu_bus_id.lower().split(":")[-2:])
    _require(
        all(
            row["gpu"] == str(physical_index)
            and row["bus_id"].lower().endswith(suffix)
            and not any(token in row["event"] for token in forbidden)
            for row in rows
        )
        and any(row["event"] == "cuda_release_verified_no_compute_owner" for row in rows),
        "telemetry GPU identity or guard status differs",
    )
    return {"path": str(source), "sha256": digest, "sample_count": len(rows)}


def _validate_owner_attestation_correlation(
    owner_audit: Mapping[str, Any], attestation: Mapping[str, Any]
) -> None:
    payload = attestation.get("payload")
    _require(isinstance(payload, Mapping), "CUDA attestation payload is absent")
    rows = payload.get("nvidia_compute_owner_rows")
    _require(isinstance(rows, list), "CUDA attestation owner rows are absent")
    attested = sorted({str(row[1]) for row in rows})
    _require(attested == owner_audit.get("child_owner_pids"), "owner PID chain differs")
    expected = (
        owner_audit.get("direct_child_owner_pids")
        if payload.get("owner_pid_binding") == "process_namespace_pid"
        else owner_audit.get("host_singleton_owner_pids")
    )
    _require(attested == expected, "owner PID namespace chain differs")


def write_forward_beta_cuda_child_attestation(
    *,
    output: str | Path,
    scene: str,
    physical_index: int,
    expected_uuid: str,
    expected_bus_id: str,
) -> dict[str, object]:
    """Bind torch ``cuda:0`` to an explicitly selected physical GPU0/GPU1."""

    identity = _gpu_identity(physical_index, expected_uuid, expected_bus_id)
    expected_environment = {
        "CUDA_DEVICE_ORDER": CUDA_DEVICE_ORDER,
        "CUDA_VISIBLE_DEVICES": expected_uuid,
        "GPU_OWNER_PID_NAMESPACE_MODE": GPU_OWNER_PID_NAMESPACE_MODE,
        "NVIDIA_VISIBLE_DEVICES": expected_uuid,
    }
    environment = {key: os.environ.get(key) for key in expected_environment}
    _require(
        environment == expected_environment,
        "forward-Beta CUDA child visibility environment differs",
    )

    preallocation_owner_rows = _nvidia_query(
        ["--query-compute-apps=gpu_uuid,pid,process_name"]
    )
    _require(
        not any(row and row[0] == expected_uuid for row in preallocation_owner_rows),
        "selected physical GPU was not owner-free before torch allocation",
    )

    import torch

    _require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "forward-Beta CUDA child must see exactly one device",
    )
    torch.cuda.set_device(0)
    probe = torch.empty((1,), dtype=torch.uint8, device="cuda:0")
    probe.zero_()
    torch.cuda.synchronize(0)
    _require(torch.cuda.current_device() == 0, "current CUDA device is not cuda:0")
    properties = torch.cuda.get_device_properties(0)

    inventory = _nvidia_query(["--query-gpu=index,uuid,pci.bus_id,name"])
    matches = [
        row
        for row in inventory
        if len(row) == 4
        and row[0] == str(physical_index)
        and row[1] == expected_uuid
        and str(row[2]).lower().endswith(
            ":".join(expected_bus_id.lower().split(":")[-2:])
        )
    ]
    _require(
        len(matches) == 1,
        "selected physical index/UUID/PCI bus inventory differs",
    )
    owner_rows = _nvidia_query(
        ["--query-compute-apps=gpu_uuid,pid,process_name"]
    )
    namespace_pids = _namespace_process_ids()
    target_rows = [row for row in owner_rows if row and row[0] == expected_uuid]
    foreign_rows = [
        row
        for row in owner_rows
        if len(row) >= 2
        and row[0] != expected_uuid
        and row[1].isdigit()
        and int(row[1]) in namespace_pids
    ]
    _require(
        len(target_rows) == 1
        and len(target_rows[0]) == 3
        and target_rows[0][1].isdigit()
        and not foreign_rows,
        "torch cuda:0 owner did not match the selected physical GPU",
    )
    owner_pid = int(target_rows[0][1])
    if owner_pid in namespace_pids:
        owner_pid_binding = "process_namespace_pid"
    elif not Path(f"/proc/{owner_pid}").exists():
        owner_pid_binding = "exclusive_invisible_host_pid_singleton_after_clear"
    else:
        raise RuntimeError(
            "torch CUDA owner is visible but outside the process namespace"
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": BETA_CUDA_ATTESTATION_ARTIFACT,
        "status": "torch_cuda0_live_owner_matches_physical_gpu_uuid_and_pci",
        "scene": str(scene),
        "observed_epoch": int(time.time()),
        "hostname": socket.gethostname(),
        "environment": environment,
        "expected_gpu": identity,
        "torch_cuda": {
            "visible_device_count": int(torch.cuda.device_count()),
            "current_device": int(torch.cuda.current_device()),
            "device": "cuda:0",
            "device_name": str(properties.name),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory": int(properties.total_memory),
            "torch_version": str(torch.__version__),
            "torch_cuda_build": str(torch.version.cuda),
        },
        "process_namespace_pids": namespace_pids,
        "nvidia_inventory_row": matches[0],
        "nvidia_preallocation_owner_rows": [],
        "nvidia_compute_owner_rows": target_rows,
        "owner_pid_binding": owner_pid_binding,
        "attestation_mechanism": CUDA_ATTESTATION_MECHANISM,
    }
    write_frozen_json(output, payload)
    del probe
    return payload


def _stable_manifest(path: str | Path) -> tuple[dict[str, Any], str, Path]:
    payload, digest, source = load_json_object(
        path, label="forward-Beta candidate run manifest"
    )
    return payload, digest, source


def _validate_inline_authority(
    manifest: Mapping[str, Any],
    *,
    method_sha256: str,
) -> tuple[dict[str, Any], str]:
    raw = manifest.get("registered_forward_protocol_authority")
    declared_sha256 = manifest.get(
        "registered_forward_protocol_authority_sha256"
    )
    _require(isinstance(raw, Mapping), "run manifest lacks inline protocol authority")
    authority = json.loads(json.dumps(raw))
    _require(
        isinstance(declared_sha256, str)
        and SHA256_RE.fullmatch(declared_sha256) is not None
        and canonical_json_sha256(authority) == declared_sha256,
        "inline protocol authority SHA256 differs",
    )
    validate_authority_payload(authority)
    candidate = authority.get("candidate")
    _require(
        isinstance(candidate, Mapping)
        and candidate.get("method_contract_sha256") == method_sha256,
        "inline protocol authority candidate method SHA256 differs",
    )
    _require(
        authority.get("scoring_contract") == EXPECTED_SCORING,
        "inline protocol authority scoring differs",
    )
    _require(
        authority.get("strict_unseen_protocol_exact_match") is False
        and authority.get("strict_unseen_exact_match_blockers")
        == EXPECTED_BLOCKERS,
        "inline protocol authority exactness differs",
    )
    return authority, declared_sha256


def _validate_method_contract(
    method: Mapping[str, Any],
    *,
    profile: SceneAuthorityProfile = V1_PROFILE,
) -> None:
    prompt = method.get("prompt_registration")
    score = method.get("score_render")
    solver = method.get("solver")
    forward = method.get("registered_forward_unary")
    graph = method.get("graph")
    _require(isinstance(prompt, Mapping), "candidate prompt contract is absent")
    _require(isinstance(score, Mapping), "candidate score contract is absent")
    _require(isinstance(solver, Mapping), "candidate solver contract is absent")
    _require(isinstance(forward, Mapping), "candidate forward unary contract is absent")
    _require(isinstance(graph, Mapping), "candidate graph contract is absent")
    _require(method.get("support_mode") == "canonical_support", "support mode differs")
    _require(
        method.get("observation_fusion") == "probability_mixture"
        and method.get("registered_seed_unary_weight") == 0.0,
        "probability-mixture observation contract differs",
    )
    _require("strong_unary" not in method, "hard-seed fusion is forbidden")
    _require(
        method.get("final_readout") == "propagated"
        and method.get("selection_applied_to_main_output") is False,
        "connected selection must remain diagnostic",
    )
    _require(
        prompt.get("mode") == "raster_adjoint"
        and float(prompt.get("scale")) == 1.0
        and float(prompt.get("alpha_threshold")) == 0.0,
        "registered forward prompt parameters differ",
    )
    _require(
        float(score.get("feature_contribution_gamma")) == 1.0
        and float(score.get("pixel_threshold")) == 0.0
        and score.get("threshold_comparison") == "greater_or_equal"
        and score.get("resize_to_ground_truth") == "cv2.INTER_NEAREST",
        "registered forward score parameters differ",
    )
    _require(
        forward.get("mode") == profile.forward_mode
        and forward.get("status")
        == "protocol_authority_bound_non_exact_diagnostic"
        and forward.get("strict_unseen_eligible") is False
        and forward.get("selection_applied_to_main_output") is False
        and forward.get("required_final_readout") == "propagated"
        and forward.get("scoring_adapter") == EXPECTED_SCORING,
        "registered forward method contract differs",
    )
    if profile is V2_PROFILE:
        _require(
            method.get("canonical_reliability_cache")
            == "per_scene_source_artifact:canonical_primitive_reliability_v1.pt",
            "v2 canonical reliability source binding differs",
        )
        balance = forward.get("class_balance")
        anchor = forward.get("anchor")
        _require(
            isinstance(balance, Mapping)
            and balance.get("scope") == "global_expected_counts"
            and balance.get("class_prior_from_scribble_area") is False
            and forward.get("field_prior_concentration_formula")
            == "kappa_i=1+reliability_i*observation_coverage_i"
            and forward.get("field_prior_concentration_bounds")
            == {"minimum": 1.0, "maximum": 2.0}
            and forward.get("residual_evidence_concentration_formula")
            == "m_i/(1+m_i)"
            and forward.get("semantic_precision_is_primary_for_nonanchors") is True
            and isinstance(anchor, Mapping)
            and anchor.get("threshold_source") == "solver.hard_seed_threshold"
            and anchor.get("solver_constraint")
            == "promote_matching_seed_weight_to_one"
            and forward.get("uses_target_calibration") is False
            and forward.get("uses_scene_id_branching") is False,
            "v2 balanced residual/precision/anchor contract differs",
        )


def validate_run_manifest(
    path: str | Path,
    *,
    scene: str,
    profile: SceneAuthorityProfile = V1_PROFILE,
) -> dict[str, Any]:
    manifest, manifest_sha256, source = _stable_manifest(path)
    _require(
        manifest.get("candidate") == profile.candidate_id,
        "candidate id differs",
    )
    _require(
        manifest.get("scene_gpu_assignment") == SCENE_GPU_ASSIGNMENT,
        "fixed scene GPU assignment differs",
    )
    _require(
        manifest.get("maximum_concurrent_scene_evaluators")
        == MAXIMUM_CONCURRENT_SCENE_EVALUATORS,
        "scene-evaluator concurrency must be exactly one",
    )
    _require(
        manifest.get("host_memory_policy") == HOST_MEMORY_POLICY,
        "host-memory policy differs",
    )
    _require(
        manifest.get("serial_scene_gpu_plan") == SERIAL_SCENE_GPU_PLAN,
        "serial scene GPU plan differs",
    )
    scenes = manifest.get("scenes")
    assigned_scenes = [
        *SCENE_GPU_ASSIGNMENT["gpu0"],
        *SCENE_GPU_ASSIGNMENT["gpu1"],
    ]
    _require(
        isinstance(scenes, list)
        and scene in scenes
        and scene in assigned_scenes,
        "scene is not in fixed candidate assignment",
    )
    method = manifest.get("method_contract")
    _require(isinstance(method, Mapping), "candidate method contract is absent")
    method_sha256 = canonical_json_sha256(method)
    _require(
        manifest.get("method_contract_sha256") == method_sha256,
        "candidate method contract SHA256 differs",
    )
    _validate_method_contract(method, profile=profile)
    authority, authority_sha256 = _validate_inline_authority(
        manifest,
        method_sha256=method_sha256,
    )
    _require(
        not _contains_exact_override(manifest),
        "candidate manifest contains a forbidden exact override",
    )
    reliability_record = None
    if profile is V2_PROFILE:
        cache_manifest_record = manifest.get("reliability_cache_manifest")
        _require(
            isinstance(cache_manifest_record, Mapping),
            "v2 reliability cache manifest record is absent",
        )
        cache_manifest_path = _validate_sized_file_record(
            cache_manifest_record,
            label="v2 reliability cache manifest",
        )
        cache_manifest, cache_manifest_sha256, _ = load_json_object(
            cache_manifest_path,
            label="v2 reliability cache manifest",
        )
        _require(
            cache_manifest_record.get("sha256") == cache_manifest_sha256,
            "v2 reliability cache manifest SHA256 differs",
        )
        from radio_gs.scripts.bind_nvos_beta_v2_reliability_manifest import (
            validate_manifest_payload as validate_reliability_manifest_payload,
        )

        # The complete ~17GB cohort is descriptor/hash validated exactly once
        # before snapshot authority is published.  Per-scene authority must not
        # repeat that work eight times; it validates the immutable manifest
        # bytes structurally here, then hashes this scene's cache/report below.
        validate_reliability_manifest_payload(
            cache_manifest, verify_files=False
        )
        sources = manifest.get("source_artifacts")
        _require(isinstance(sources, Mapping), "v2 source artifacts are absent")
        scene_sources = sources.get(scene)
        _require(isinstance(scene_sources, Mapping), "v2 scene sources are absent")
        reliability_record = scene_sources.get(
            "canonical_primitive_reliability_v1.pt"
        )
        _require(
            isinstance(reliability_record, Mapping),
            "v2 scene reliability source is absent",
        )
        _require(
            set(reliability_record)
            == {"path", "bytes", "sha256", "metadata_path", "metadata_sha256"},
            "v2 scene reliability source fields differ",
        )
        cache_scenes = cache_manifest.get("scenes")
        _require(isinstance(cache_scenes, Mapping), "v2 cache manifest scenes differ")
        cache_scene = cache_scenes.get(scene)
        _require(isinstance(cache_scene, Mapping), "v2 cache scene record is absent")
        cache_pt = cache_scene.get("reliability_cache")
        cache_json = cache_scene.get("build_report")
        _require(
            isinstance(cache_pt, Mapping)
            and isinstance(cache_json, Mapping)
            and cache_pt.get("bytes") == reliability_record.get("bytes")
            and cache_pt.get("sha256") == reliability_record.get("sha256")
            and cache_json.get("sha256")
            == reliability_record.get("metadata_sha256")
            and Path(str(cache_pt.get("path", ""))).resolve()
            == Path(str(reliability_record.get("path", ""))).resolve()
            and Path(str(cache_json.get("path", ""))).resolve()
            == Path(str(reliability_record.get("metadata_path", ""))).resolve(),
            "v2 cache manifest/source artifact binding differs",
        )
        reliability_path = _validate_sized_file_record(
            cache_pt,
            label=f"{scene} v2 reliability cache",
        )
        metadata_path = _validate_sized_file_record(
            cache_json,
            label=f"{scene} v2 reliability build report",
        )
    return {
        "payload": manifest,
        "path": source,
        "sha256": manifest_sha256,
        "method_contract": dict(method),
        "method_contract_sha256": method_sha256,
        "protocol_authority": authority,
        "protocol_authority_sha256": authority_sha256,
        "reliability_source": dict(reliability_record) if reliability_record else None,
    }


def _argv_options(argv: Sequence[str]) -> dict[str, str | None]:
    values = [str(value) for value in argv]
    _require(bool(values) and all(values), "scene argv is empty")
    options: dict[str, str | None] = {}
    index = 0
    while index < len(values):
        token = values[index]
        if not token.startswith("--"):
            index += 1
            continue
        if "=" in token:
            flag, value = token.split("=", 1)
        else:
            flag = token
            if index + 1 < len(values) and not values[index + 1].startswith("--"):
                index += 1
                value = values[index]
            else:
                value = None
        _require(flag not in options, f"duplicate argv option: {flag}")
        options[flag] = value
        index += 1
    return options


def _equal_option(actual: str | None, expected: object) -> bool:
    if actual is None:
        return False
    if isinstance(expected, float):
        try:
            return float(actual) == expected
        except ValueError:
            return False
    if isinstance(expected, int):
        try:
            return int(actual) == expected
        except ValueError:
            return False
    return actual == str(expected)


def _derived_argv_contract(method: Mapping[str, Any]) -> dict[str, object]:
    prompt = method["prompt_registration"]
    score = method["score_render"]
    solver = method["solver"]
    graph = method["graph"]
    expected: dict[str, object] = {
        **CORE_ARGV,
        "--registered-forward-unary": method["registered_forward_unary"]["mode"],
        "--region-space": method["region_space"],
        "--depth-tolerance": prompt["depth_tolerance"],
        "--relative-depth-tolerance": prompt["relative_depth_tolerance"],
        "--registered-seed-construction": method["seed_construction"],
        "--registered-observation-confidence": method["observation_confidence"],
        "--registered-observation-mass-scale": method["observation_mass_scale"],
        "--support-threshold": method["prompt_support_threshold"],
        "--prototype-count": method["prototype_count"],
        "--prototype-strategy": method["prototype_strategy"],
        "--appearance-weight": method["appearance_weight"],
        "--boundary-weight": method["boundary_weight"],
        "--prototype-temperature": method["prototype_temperature"],
        "--feature-calibration": method["feature_calibration"],
        "--background-centroids": method["background_centroids"],
        "--score-calibration": method["score_calibration"],
        "--negative-spatial-mode": method["negative_spatial_mode"],
        "--registered-selection-mode": method["diagnostic_selection_mode"],
        "--graph-policy": graph["policy"],
        "--component-graph-policy": graph["component_policy"],
        "--graph-legacy-residual": graph["legacy_residual"],
        "--channel-confidence-mode": graph["channel_confidence_mode"],
        "--score-render-resolution": score["resolution"],
        "--score-render-scale": score["scale"],
        "--valid-support-coverage-power": score["valid_support_coverage_power"],
        "--score-chunk-size": score["score_chunk_size"],
        "--solver-support-threshold": solver["support_threshold"],
        "--solver-type": solver["type"],
        "--solver-iterations": solver["iterations"],
        "--solver-residual": solver["residual"],
        "--solver-unary-temperature": solver["unary_temperature"],
        "--laplacian-weight": solver["laplacian_weight"],
        "--cg-iterations": solver["cg_iterations"],
        "--cg-tolerance": solver["cg_tolerance"],
        "--hard-seed-threshold": solver["hard_seed_threshold"],
        "--hard-seed-conflict-policy": solver["hard_seed_conflict_policy"],
        "--hard-seed-conflict-margin": solver["hard_seed_conflict_margin"],
        "--component-edge-threshold": solver["component_edge_threshold"],
        "--seeded-component-min-weight": solver["seeded_component_min_weight"],
    }
    if "observation_coverage_power" in method:
        expected["--registered-observation-coverage-power"] = method[
            "observation_coverage_power"
        ]
    if str(method.get("canonical_reliability_cache", "")) and not str(
        method.get("canonical_reliability_cache", "")
    ).startswith("per_scene_source_artifact:"):
        expected["--canonical-reliability-cache"] = method[
            "canonical_reliability_cache"
        ]
    if str(method.get("diagnostic_graph_affinity_override", "")):
        expected["--diagnostic-graph-affinity-override"] = method[
            "diagnostic_graph_affinity_override"
        ]
    return expected


def validate_scene_argv(
    argv: Sequence[str],
    *,
    manifest: Mapping[str, Any],
    run_manifest_path: Path,
    scene: str,
    result: Path,
    attestation: Path,
    physical_index: int,
    gpu_uuid: str,
    gpu_bus_id: str,
    profile: SceneAuthorityProfile = V1_PROFILE,
) -> list[str]:
    _gpu_identity(physical_index, gpu_uuid, gpu_bus_id)
    values = [str(value) for value in argv]
    options = _argv_options(values)
    _require(not (FORBIDDEN_ARGV & set(options)), "argv contains authority/exact override")
    _require(
        any(Path(value).name == "eval_nvos_gaussian_first.py" for value in values),
        "argv does not execute the NVOS evaluator",
    )
    identity = {
        "--candidate-id": profile.candidate_id,
        "--scene-id": scene,
        "--device": "cuda:0",
        "--expected-gpu-physical-index": physical_index,
        "--expected-gpu-uuid": gpu_uuid,
        "--expected-gpu-bus-id": gpu_bus_id,
    }
    expected_argv = {
        **_derived_argv_contract(manifest["method_contract"]),
        **identity,
    }
    if profile is V2_PROFILE:
        source = manifest.get("source_artifacts", {}).get(scene, {}).get(
            "canonical_primitive_reliability_v1.pt"
        )
        _require(isinstance(source, Mapping), "v2 argv reliability source is absent")
        expected_argv["--canonical-reliability-cache"] = source.get("path")
    for flag, expected in expected_argv.items():
        _require(_equal_option(options.get(flag), expected), f"argv {flag} differs")
    _require(
        options.get("--run-manifest") is not None
        and Path(str(options["--run-manifest"])).resolve() == run_manifest_path,
        "argv run manifest differs",
    )
    _require(
        options.get("--gpu-attestation-output") is not None
        and Path(str(options["--gpu-attestation-output"])).resolve()
        == attestation.resolve(),
        "argv CUDA attestation path differs",
    )
    _require(
        options.get("--output-dir") is not None
        and Path(str(options["--output-dir"])).resolve() == result.parent.resolve(),
        "argv output directory differs",
    )
    valid_support = bool(
        manifest["method_contract"]["score_render"]["valid_support_normalization"]
    )
    _require(
        ("--valid-support-normalization" in options) is valid_support
        and options.get("--valid-support-normalization") is None,
        "argv valid-support normalization flag differs",
    )
    require_hashes = bool(
        manifest["method_contract"]["asset_hash_verification_required"]
    )
    _require(
        ("--require-asset-hashes" in options) is require_hashes
        and options.get("--require-asset-hashes") is None,
        "argv asset-hash verification flag differs",
    )
    return values


def prepare_scene_command(
    *,
    output: str | Path,
    run_manifest: str | Path,
    scene: str,
    result: str | Path,
    telemetry: str | Path,
    owner_audit: str | Path,
    attestation: str | Path,
    postcheck: str | Path,
    receipt: str | Path,
    evaluator_log: str | Path,
    guard: str | Path,
    physical_index: int,
    gpu_uuid: str,
    gpu_bus_id: str,
    command: Sequence[str],
    profile: SceneAuthorityProfile = V1_PROFILE,
) -> dict[str, Any]:
    gpu_identity = _gpu_identity(physical_index, gpu_uuid, gpu_bus_id)
    validated = validate_run_manifest(run_manifest, scene=scene, profile=profile)
    expected_physical_index = (
        0 if scene in SCENE_GPU_ASSIGNMENT["gpu0"] else 1
    )
    _require(
        physical_index == expected_physical_index,
        f"scene is assigned to physical GPU{expected_physical_index}",
    )
    result_path = Path(result).resolve()
    attestation_path = Path(attestation).resolve()
    argv_values = [str(value) for value in command]
    if argv_values[:1] == ["--"]:
        argv_values.pop(0)
    _require(
        bool(argv_values) and all(argv_values),
        "forward-Beta guarded scene command is empty",
    )
    argv = validate_scene_argv(
        argv_values,
        manifest=validated["payload"],
        run_manifest_path=validated["path"],
        scene=scene,
        result=result_path,
        attestation=attestation_path,
        physical_index=physical_index,
        gpu_uuid=gpu_uuid,
        gpu_bus_id=gpu_bus_id,
        profile=profile,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": profile.command_artifact,
        "candidate_id": profile.candidate_id,
        "scene": scene,
        "run_manifest": file_record(validated["path"]),
        "method_contract_sha256": validated["method_contract_sha256"],
        "protocol_authority_sha256": validated["protocol_authority_sha256"],
        "result_path": str(result_path),
        "telemetry_path": str(Path(telemetry).resolve()),
        "owner_audit_path": str(Path(owner_audit).resolve()),
        "cuda_attestation_path": str(attestation_path),
        "postcheck_path": str(Path(postcheck).resolve()),
        "receipt_path": str(Path(receipt).resolve()),
        "evaluator_log_path": str(Path(evaluator_log).resolve()),
        "guard": file_record(guard),
        "gpu_identity": gpu_identity,
        "cuda_environment": {
            "CUDA_DEVICE_ORDER": CUDA_DEVICE_ORDER,
            "CUDA_VISIBLE_DEVICES": gpu_uuid,
            "GPU_OWNER_PID_NAMESPACE_MODE": GPU_OWNER_PID_NAMESPACE_MODE,
            "NVIDIA_VISIBLE_DEVICES": gpu_uuid,
        },
        "argv": argv,
        "argv_sha256": canonical_json_sha256(argv),
    }
    write_frozen_json(output, payload)
    return payload


def _validate_scene_command(
    path: str | Path,
    *,
    profile: SceneAuthorityProfile = V1_PROFILE,
) -> tuple[dict[str, Any], Path]:
    payload, _digest, source = load_json_object(path, label="forward-Beta scene command")
    expected_fields = {
        "schema_version", "artifact_type", "candidate_id", "scene",
        "run_manifest", "method_contract_sha256", "protocol_authority_sha256",
        "result_path", "telemetry_path", "owner_audit_path", "cuda_attestation_path",
        "postcheck_path", "receipt_path", "evaluator_log_path", "guard",
        "gpu_identity", "cuda_environment", "argv",
        "argv_sha256",
    }
    _require(
        set(payload) == expected_fields
        and payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("artifact_type") == profile.command_artifact
        and payload.get("candidate_id") == profile.candidate_id
        and isinstance(payload.get("argv"), list)
        and payload.get("argv_sha256") == canonical_json_sha256(payload["argv"]),
        "forward-Beta scene command differs",
    )
    manifest_path = validate_file_record(
        payload["run_manifest"], label="forward-Beta command manifest"
    )
    validate_file_record(payload["guard"], label="forward-Beta command guard")
    validated = validate_run_manifest(
        manifest_path, scene=str(payload["scene"]), profile=profile
    )
    _require(
        payload["method_contract_sha256"] == validated["method_contract_sha256"]
        and payload["protocol_authority_sha256"]
        == validated["protocol_authority_sha256"],
        "forward-Beta command authority chain differs",
    )
    gpu = payload["gpu_identity"]
    _require(isinstance(gpu, Mapping), "forward-Beta command GPU identity differs")
    validated_gpu = _gpu_identity(
        gpu.get("physical_index"), gpu.get("uuid"), gpu.get("pci_bus_id")
    )
    _require(
        dict(gpu) == validated_gpu
        and payload["cuda_environment"]
        == {
            "CUDA_DEVICE_ORDER": CUDA_DEVICE_ORDER,
            "CUDA_VISIBLE_DEVICES": gpu["uuid"],
            "GPU_OWNER_PID_NAMESPACE_MODE": GPU_OWNER_PID_NAMESPACE_MODE,
            "NVIDIA_VISIBLE_DEVICES": gpu["uuid"],
        },
        "forward-Beta command GPU identity differs",
    )
    validate_scene_argv(
        payload["argv"],
        manifest=validated["payload"],
        run_manifest_path=manifest_path,
        scene=str(payload["scene"]),
        result=Path(payload["result_path"]),
        attestation=Path(payload["cuda_attestation_path"]),
        physical_index=int(gpu["physical_index"]),
        gpu_uuid=str(gpu["uuid"]),
        gpu_bus_id=str(gpu["pci_bus_id"]),
        profile=profile,
    )
    return payload, source


def _validate_beta_cuda_attestation(
    path: str | Path,
    *,
    scene: str,
    physical_index: int,
    gpu_uuid: str,
    gpu_bus_id: str,
) -> dict[str, Any]:
    payload, digest, source = load_json_object(
        path, label="forward-Beta CUDA attestation"
    )
    expected_environment = {
        "CUDA_DEVICE_ORDER": CUDA_DEVICE_ORDER,
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
        "GPU_OWNER_PID_NAMESPACE_MODE": GPU_OWNER_PID_NAMESPACE_MODE,
        "NVIDIA_VISIBLE_DEVICES": gpu_uuid,
    }
    torch_cuda = payload.get("torch_cuda")
    _require(
        set(payload) == CUDA_ATTESTATION_FIELDS
        and payload.get("schema_version") in {1, 2}
        and payload.get("artifact_type") == BETA_CUDA_ATTESTATION_ARTIFACT
        and payload.get("status")
        == "torch_cuda0_live_owner_matches_physical_gpu_uuid_and_pci"
        and payload.get("scene") == scene
        and isinstance(payload.get("observed_epoch"), int)
        and not isinstance(payload.get("observed_epoch"), bool)
        and int(payload["observed_epoch"]) > 0
        and isinstance(payload.get("hostname"), str)
        and bool(payload.get("hostname"))
        and payload.get("environment") == expected_environment
        and payload.get("expected_gpu")
        == {
            "physical_index": physical_index,
            "uuid": gpu_uuid,
            "pci_bus_id": gpu_bus_id,
        }
        and isinstance(torch_cuda, Mapping)
        and torch_cuda.get("visible_device_count") == 1
        and torch_cuda.get("current_device") == 0
        and torch_cuda.get("device") == "cuda:0"
        and isinstance(payload.get("process_namespace_pids"), list)
        and payload.get("process_namespace_pids")
        and payload.get("nvidia_preallocation_owner_rows") == []
        and isinstance(payload.get("nvidia_compute_owner_rows"), list)
        and len(payload["nvidia_compute_owner_rows"]) == 1
        and payload.get("owner_pid_binding")
        in {
            "process_namespace_pid",
            "exclusive_invisible_host_pid_singleton_after_clear",
        }
        and payload.get("attestation_mechanism") == CUDA_ATTESTATION_MECHANISM,
        "forward-Beta CUDA attestation differs",
    )
    inventory = payload.get("nvidia_inventory_row")
    _require(
        isinstance(inventory, list)
        and len(inventory) == 4
        and inventory[0] == str(physical_index)
        and inventory[1] == gpu_uuid
        and str(inventory[2]).lower().endswith(
            ":".join(str(gpu_bus_id).lower().split(":")[-2:])
        ),
        "forward-Beta CUDA inventory identity differs",
    )
    namespace_pids = {
        int(value)
        for value in payload["process_namespace_pids"]
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    }
    owner_row = payload["nvidia_compute_owner_rows"][0]
    _require(
        isinstance(owner_row, list)
        and len(owner_row) == 3
        and owner_row[0] == gpu_uuid
        and str(owner_row[1]).isdigit(),
        "forward-Beta CUDA owner identity differs",
    )
    owner_pid = int(owner_row[1])
    binding = payload["owner_pid_binding"]
    _require(
        (binding == "process_namespace_pid" and owner_pid in namespace_pids)
        or (
            binding == "exclusive_invisible_host_pid_singleton_after_clear"
            and owner_pid not in namespace_pids
        ),
        "forward-Beta CUDA owner PID namespace binding differs",
    )
    return {
        "path": str(source),
        "sha256": digest,
        "payload": payload,
    }


def _validate_result_report(
    path: str | Path,
    *,
    command: Mapping[str, Any],
    manifest: Mapping[str, Any],
    profile: SceneAuthorityProfile = V1_PROFILE,
) -> dict[str, Any]:
    report, digest, source = load_json_object(path, label="forward-Beta result report")
    safety = report.get("safety")
    method = report.get("method_contract")
    _require(report.get("scene_id") == command["scene"], "result scene differs")
    _require(
        report.get("registered_forward_protocol_authority")
        == manifest["registered_forward_protocol_authority"]
        and report.get("registered_forward_protocol_authority_sha256")
        == command["protocol_authority_sha256"],
        "result top-level protocol authority differs",
    )
    _require(isinstance(safety, Mapping), "result safety contract is absent")
    _require(
        safety.get("main_result_eligible") is False
        and safety.get("frozen_diagnostic_eligible") is False
        and safety.get("strict_unseen_protocol_exact_match") is False
        and safety.get("registered_forward_protocol_authority_sha256")
        == command["protocol_authority_sha256"],
        "result promotion safety differs",
    )
    _require(isinstance(method, Mapping), "result method contract is absent")
    _require(
        method.get("candidate_id") == profile.candidate_id
        and method.get("candidate_method_contract_sha256")
        == command["method_contract_sha256"]
        and method.get("registered_forward_protocol_authority_sha256")
        == command["protocol_authority_sha256"]
        and method.get("registered_forward_protocol_authority")
        == manifest["registered_forward_protocol_authority"],
        "result method/authority binding differs",
    )
    shared = method.get("shared_solver")
    _require(
        isinstance(shared, Mapping)
        and shared.get("registered_readout_stage") == "propagated"
        and shared.get("registered_observation_fusion") == "probability_mixture"
        and isinstance(shared.get("registered_forward_unary"), Mapping)
        and shared["registered_forward_unary"].get("mode")
        == profile.forward_mode,
        "result beta solver/readout differs",
    )
    return {"path": str(source), "sha256": digest, "payload": report}


def _validate_postcheck(
    path: str | Path,
    *,
    command: Mapping[str, Any],
    result_record: Mapping[str, str],
    profile: SceneAuthorityProfile = V1_PROFILE,
) -> dict[str, str]:
    payload, digest, source = load_json_object(path, label="forward-Beta scene postcheck")
    expected_fields = {
        "schema_version", "artifact_type", "status", "candidate_id", "scene",
        "observed_epoch", "run_manifest", "result", "gpu_identity",
        "nvidia_inventory_row", "compute_owners",
    }
    _require(
        set(payload) == expected_fields
        and payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("artifact_type") == profile.postcheck_artifact
        and payload.get("status") == "gpu_identity_and_post_owner_clear_verified"
        and payload.get("candidate_id") == profile.candidate_id
        and payload.get("scene") == command["scene"]
        and isinstance(payload.get("observed_epoch"), int)
        and not isinstance(payload.get("observed_epoch"), bool)
        and int(payload["observed_epoch"]) > 0
        and payload.get("run_manifest") == command["run_manifest"]
        and payload.get("result") == result_record
        and payload.get("gpu_identity") == command["gpu_identity"]
        and payload.get("nvidia_inventory_row")
        == [
            str(command["gpu_identity"]["physical_index"]),
            command["gpu_identity"]["uuid"],
            command["gpu_identity"]["pci_bus_id"],
        ]
        and payload.get("compute_owners") == [],
        "forward-Beta scene postcheck differs",
    )
    return {"path": str(source), "sha256": digest}


def write_scene_postcheck(
    *,
    output: str | Path,
    command_record: str | Path,
    profile: SceneAuthorityProfile = V1_PROFILE,
) -> dict[str, Any]:
    """Verify the planned physical GPU is responsive and owner-free post-run."""

    command, _command_source = _validate_scene_command(
        command_record, profile=profile
    )
    output_path = Path(output).resolve()
    _require(
        output_path == Path(command["postcheck_path"]),
        "postcheck destination differs from command",
    )
    result_record = file_record(command["result_path"])
    gpu = command["gpu_identity"]
    inventory = _nvidia_query(["--query-gpu=index,uuid,pci.bus_id"])
    suffix = ":".join(str(gpu["pci_bus_id"]).lower().split(":")[-2:])
    matches = [
        row
        for row in inventory
        if len(row) == 3
        and row[0] == str(gpu["physical_index"])
        and row[1] == gpu["uuid"]
        and str(row[2]).lower().endswith(suffix)
    ]
    owners = _nvidia_query(
        ["--query-compute-apps=gpu_uuid,pid,process_name"]
    )
    matching_owners = [row for row in owners if row and row[0] == gpu["uuid"]]
    _require(
        len(matches) == 1 and not matching_owners,
        "forward-Beta postcheck GPU identity/owner differs",
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": profile.postcheck_artifact,
        "status": "gpu_identity_and_post_owner_clear_verified",
        "candidate_id": profile.candidate_id,
        "scene": command["scene"],
        "observed_epoch": int(time.time()),
        "run_manifest": command["run_manifest"],
        "result": result_record,
        "gpu_identity": gpu,
        "nvidia_inventory_row": [
            str(gpu["physical_index"]),
            gpu["uuid"],
            gpu["pci_bus_id"],
        ],
        "compute_owners": [],
    }
    write_frozen_json(output_path, payload)
    return {"postcheck": file_record(output_path), "payload": payload}


def _receipt_payload(
    command_path: str | Path,
    postcheck_path: str | Path,
    *,
    profile: SceneAuthorityProfile = V1_PROFILE,
) -> dict[str, Any]:
    command, command_source = _validate_scene_command(command_path, profile=profile)
    manifest_path = validate_file_record(
        command["run_manifest"], label="forward-Beta receipt manifest"
    )
    validated_manifest = validate_run_manifest(
        manifest_path, scene=str(command["scene"]), profile=profile
    )
    result = _validate_result_report(
        command["result_path"],
        command=command,
        manifest=validated_manifest["payload"],
        profile=profile,
    )
    result_record = file_record(result["path"])
    gpu = command["gpu_identity"]
    telemetry = _validate_telemetry(
        command["telemetry_path"],
        physical_index=int(gpu["physical_index"]),
        gpu_bus_id=str(gpu["pci_bus_id"]),
    )
    owners = _validate_owner_audit(
        command["owner_audit_path"], gpu_uuid=str(gpu["uuid"])
    )
    attestation = _validate_beta_cuda_attestation(
        command["cuda_attestation_path"],
        scene=str(command["scene"]),
        physical_index=int(gpu["physical_index"]),
        gpu_uuid=str(gpu["uuid"]),
        gpu_bus_id=str(gpu["pci_bus_id"]),
    )
    _validate_owner_attestation_correlation(owners, attestation)
    _require(
        Path(postcheck_path).resolve() == Path(command["postcheck_path"]),
        "postcheck path differs from command",
    )
    postcheck = _validate_postcheck(
        postcheck_path,
        command=command,
        result_record=result_record,
        profile=profile,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": profile.receipt_artifact,
        "status": profile.receipt_status,
        "candidate_id": profile.candidate_id,
        "scene": command["scene"],
        "run_manifest": command["run_manifest"],
        "method_contract_sha256": command["method_contract_sha256"],
        "protocol_authority_sha256": command["protocol_authority_sha256"],
        "result": result_record,
        "command": file_record(command_source),
        "telemetry": telemetry,
        "owner_audit": owners,
        "cuda_attestation": {
            "path": attestation["path"],
            "sha256": attestation["sha256"],
            "attestation_mechanism": CUDA_ATTESTATION_MECHANISM,
        },
        "postcheck": postcheck,
        "evaluator_log": file_record(command["evaluator_log_path"]),
        "guard": command["guard"],
        "gpu_identity": command["gpu_identity"],
        "promotion": {
            "main_result_eligible": False,
            "frozen_diagnostic_eligible": False,
            "strict_unseen_protocol_exact_match": False,
        },
    }


def finalize_scene_receipt(
    *,
    output: str | Path,
    command_record: str | Path,
    postcheck: str | Path,
    profile: SceneAuthorityProfile = V1_PROFILE,
) -> dict[str, Any]:
    command, _command_source = _validate_scene_command(
        command_record, profile=profile
    )
    _require(
        Path(output).resolve() == Path(command["receipt_path"]),
        "receipt destination differs from command",
    )
    payload = _receipt_payload(command_record, postcheck, profile=profile)
    write_frozen_json(output, payload)
    return validate_scene_receipt(
        output,
        run_manifest=validate_file_record(
            payload["run_manifest"], label="forward-Beta finalized manifest"
        ),
        scene=str(payload["scene"]),
        result=validate_file_record(
            payload["result"], label="forward-Beta finalized result"
        ),
        profile=profile,
    )


def validate_scene_receipt(
    path: str | Path,
    *,
    run_manifest: str | Path,
    scene: str,
    result: str | Path,
    profile: SceneAuthorityProfile = V1_PROFILE,
) -> dict[str, Any]:
    receipt, digest, source = load_json_object(path, label="forward-Beta scene receipt")
    command_path = validate_file_record(
        receipt.get("command"), label="forward-Beta receipt command"
    )
    command, _command_source = _validate_scene_command(command_path, profile=profile)
    _require(
        source == Path(command["receipt_path"]),
        "receipt path differs from command",
    )
    postcheck_path = validate_file_record(
        receipt.get("postcheck"), label="forward-Beta receipt postcheck"
    )
    expected = _receipt_payload(command_path, postcheck_path, profile=profile)
    _require(receipt == expected, "forward-Beta scene receipt chain differs")
    _require(
        receipt.get("scene") == scene
        and validate_file_record(
            receipt.get("run_manifest"), label="forward-Beta receipt manifest"
        )
        == Path(run_manifest).resolve()
        and validate_file_record(
            receipt.get("result"), label="forward-Beta receipt result"
        )
        == Path(result).resolve(),
        "forward-Beta receipt belongs to another scene run",
    )
    return {"receipt": {"path": str(source), "sha256": digest}, "payload": receipt}


def main_for_profile(
    profile: SceneAuthorityProfile,
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("--run-manifest", required=True)
    validate.add_argument("--scene", required=True)
    prepare = subparsers.add_parser("prepare-scene")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--run-manifest", type=Path, required=True)
    prepare.add_argument("--scene", required=True)
    prepare.add_argument("--result", type=Path, required=True)
    prepare.add_argument("--telemetry", type=Path, required=True)
    prepare.add_argument("--owner-audit", type=Path, required=True)
    prepare.add_argument("--attestation", type=Path, required=True)
    prepare.add_argument("--postcheck", type=Path, required=True)
    prepare.add_argument("--receipt", type=Path, required=True)
    prepare.add_argument("--evaluator-log", type=Path, required=True)
    prepare.add_argument("--guard", type=Path, required=True)
    prepare.add_argument("--physical-index", type=int, choices=(0, 1), required=True)
    prepare.add_argument("--gpu-uuid", required=True)
    prepare.add_argument("--gpu-bus-id", required=True)
    prepare.add_argument("command_argv", nargs=argparse.REMAINDER)
    postcheck = subparsers.add_parser("postcheck-scene")
    postcheck.add_argument("--output", type=Path, required=True)
    postcheck.add_argument("--command-record", type=Path, required=True)
    finalize = subparsers.add_parser("finalize-scene")
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--command-record", type=Path, required=True)
    finalize.add_argument("--postcheck", type=Path, required=True)
    validate_scene = subparsers.add_parser("validate-scene")
    validate_scene.add_argument("--receipt", type=Path, required=True)
    validate_scene.add_argument("--run-manifest", type=Path, required=True)
    validate_scene.add_argument("--scene", required=True)
    validate_scene.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "validate-manifest":
        validated = validate_run_manifest(
            args.run_manifest, scene=args.scene, profile=profile
        )
        print(json.dumps({
            "candidate_id": profile.candidate_id,
            "method_contract_sha256": validated["method_contract_sha256"],
            "protocol_authority_sha256": validated["protocol_authority_sha256"],
        }, sort_keys=True))
        return 0
    if args.action == "prepare-scene":
        payload = prepare_scene_command(
            output=args.output,
            run_manifest=args.run_manifest,
            scene=args.scene,
            result=args.result,
            telemetry=args.telemetry,
            owner_audit=args.owner_audit,
            attestation=args.attestation,
            postcheck=args.postcheck,
            receipt=args.receipt,
            evaluator_log=args.evaluator_log,
            guard=args.guard,
            physical_index=args.physical_index,
            gpu_uuid=args.gpu_uuid,
            gpu_bus_id=args.gpu_bus_id,
            command=args.command_argv,
            profile=profile,
        )
    elif args.action == "postcheck-scene":
        payload = write_scene_postcheck(
            output=args.output,
            command_record=args.command_record,
            profile=profile,
        )
    elif args.action == "finalize-scene":
        payload = finalize_scene_receipt(
            output=args.output,
            command_record=args.command_record,
            postcheck=args.postcheck,
            profile=profile,
        )
    elif args.action == "validate-scene":
        payload = validate_scene_receipt(
            args.receipt,
            run_manifest=args.run_manifest,
            scene=args.scene,
            result=args.result,
            profile=profile,
        )
    else:
        raise AssertionError(args.action)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return main_for_profile(V1_PROFILE, argv)


if __name__ == "__main__":
    raise SystemExit(main())
