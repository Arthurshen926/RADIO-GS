from __future__ import annotations

import os
import fcntl
import errno
from pathlib import Path
import shutil
import subprocess
import sys
import json
from types import SimpleNamespace

import pytest

from radio_gs.scripts import surface_region_run_guard as run_guard
from radio_gs.scripts import surface_gpu1_lock_supervisor as gpu1_lock
from radio_gs.scripts.surface_region_run_guard import (
    audit_attempt_inventory,
    discover_repo_python_closure,
    summarize_canary_telemetry,
    validate_attempt_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    REPO_ROOT
    / "radio_gs/scripts/run_surface_region_context_recovery_screen.sh"
)
PYTHON_RUNNER = REPO_ROOT / "radio_gs/scripts/run_repo_python.sh"
THERMAL_GUARD = REPO_ROOT / "radio_gs/scripts/run_with_gpu_thermal_guard.sh"


def _row(*, temp: int, event: str, power: float = 140.0, util: int = 0) -> dict[str, str]:
    return {
        "timestamp": "2026-08-01T00:00:00+08:00",
        "gpu": "1",
        "bus_id": "0000:82:00.0",
        "temp_c": str(temp),
        "power_w": str(power),
        "power_limit_w": "300.00",
        "util_pct": str(util),
        "memory_mib": "3344",
        "pstate": "P2",
        "event": event,
    }


def test_surface_runner_defaults_lock_and_fail_closed_canary() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "surface_fixed_teacher_replay_v2_gpu1_p2_canary71_closure" in source
    assert 'GPU_MAX_TEMP_C="${GPU_MAX_TEMP_C:-75}"' in source
    assert 'GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-1}"' in source
    assert 'GPU_SOFT_PAUSE_TEMP_C="${GPU_SOFT_PAUSE_TEMP_C:-0}"' in source
    assert 'GPU_SOFT_RESUME_TEMP_C="${GPU_SOFT_RESUME_TEMP_C:-0}"' in source
    assert 'GPU_PEER_PAUSE_TEMP_C="${GPU_PEER_PAUSE_TEMP_C:-77}"' in source
    assert 'GPU_PEER_RESUME_TEMP_C="${GPU_PEER_RESUME_TEMP_C:-75}"' in source
    assert (
        'RADIO_THERMAL_PACING_SECONDS_PER_IMAGE="${RADIO_THERMAL_PACING_SECONDS_PER_IMAGE:-2.0}"'
        in source
    )
    assert 'SURFACE_CANARY_RESUME="${SURFACE_CANARY_RESUME:-0}"' in source
    assert 'GPU_PEER_ACTIVITY_ACTION="${GPU_PEER_ACTIVITY_ACTION:-terminate}"' in source
    assert "GPU_PEER_INTERRUPT_EXIT_CODE=87" in source
    assert 'GLOBAL_GPU1_LOCK="/root/RADIO-GS/output/.physical_gpu1.lock"' in source
    assert (
        'GPU1_SINGLETON_PROTOCOL="linux-abstract-af-unix-stream-v1:'
        'radio-gs-physical-gpu1-v1"' in source
    )
    assert "RADIO_GS_GPU1_SINGLETON_FD" in source
    assert "--singleton-fd" in source
    assert (
        '"global_gpu_kernel_singleton_protocol": gpu1_singleton_protocol'
        in source
    )
    assert "surface_gpu1_lock_supervisor.py" in source
    assert "verify-inherited" in source
    assert "flock -n" not in source
    assert "--query-compute-apps=gpu_uuid,pid" in source
    assert '"source_snapshot_import_root"' in source
    assert '"source_snapshot_tree_sha256"' in source
    assert '"runtime_closure": runtime_closure' in source
    assert "audit-canary" in source
    assert '"maximum_temperature_c": 71' in source
    assert "default_fail_closed_after_pass" in source
    assert 'verify_run_closure "pre_${stage}"' in source
    assert "--phase final_before_completion" in source
    assert "--full-checkpoint" in source
    assert "--resume-dir \"$resume_dir\"" in source
    assert 'CACHE_RESUME_ROOT="$OUTPUT_ROOT/cache_scene_resume"' in source
    assert 'ATTEMPT_RECEIPT_ROOT="$OUTPUT_ROOT/stage_attempts"' in source
    assert "peer_activity_interrupted_cuda_released_retry_authorized" in source
    assert "attempt_peer_release_interrupt_count" in source
    assert 'verify_run_closure "pre_${stage}_attempt_${attempt_tag}"' in source
    assert 'verify_run_closure "post_${stage}_attempt_${attempt_tag}"' in source
    assert "surface-region-stage-attempt-v1" in source
    assert "capture-kernel-journal" in source
    assert "capture-gpu-release-postflight" in source


def test_surface_peer_interrupt_releases_cuda_with_reserved_status() -> None:
    source = THERMAL_GUARD.read_text(encoding="utf-8")
    assert 'GPU_PEER_ACTIVITY_ACTION="${GPU_PEER_ACTIVITY_ACTION:-pause}"' in source
    assert "GPU_PEER_INTERRUPT_EXIT_CODE=87" in source
    assert 'GPU_PEER_ACTIVITY_ACTION" == "terminate"' in source
    assert "peer_activity_interrupt_release_cuda_" in source
    assert 'exit "$GPU_PEER_INTERRUPT_EXIT_CODE"' in source
    assert "terminate_child_group" in source
    assert "process_group_has_live_members" in source
    assert "cuda_release_verified_no_compute_owner" in source
    # Existing callers retain SIGSTOP behavior unless they opt in.
    assert 'GPU_PEER_ACTIVITY_ACTION:-pause' in source


def test_surface_runtime_closure_contains_critical_transitive_sources() -> None:
    closure = set(run_guard.repo_source_closure(REPO_ROOT)["files"])
    assert {
        "radio_gs/interfaces/frozen_radio_views.py",
        "radio_gs/interfaces/capability_cache.py",
        "radio_gs/interfaces/crop_spatial_alignment.py",
        "radio_gs/interfaces/scale_ordered_relation.py",
        "radio_gs/interfaces/semantic_alignment.py",
        "radio_gs/interfaces/surface_region_contract.py",
        "radio_gs/interfaces/surface_region_summary.py",
        "radio_gs/models/radio_adaptors.py",
        "radio_gs/models/siglip_projection.py",
        "radio_gs/querying/query_spec.py",
        "radio_gs/querying/query_engine.py",
        "radio_gs/querying/unified_query.py",
        "radio_gs/querying/support_solver.py",
        "radio_gs/scripts/build_canonical_support_graph.py",
        "radio_gs/scripts/build_scannet_surface_region_cache.py",
        "radio_gs/scripts/surface_region_scene_resume.py",
        "radio_gs/scripts/surface_gpu1_lock_supervisor.py",
        "radio_gs/scripts/surface_region_run_guard.py",
        "radio_gs/scripts/train_surface_region_summary_readout.py",
        "radio_gs/utils/checkpoint_io.py",
        "radio_gs/utils/immutable_artifacts.py",
    } <= closure


def test_repository_closure_digest_detects_transitive_source_drift(
    tmp_path: Path,
) -> None:
    for relative in run_guard.REPO_PYTHON_ENTRYPOINTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("from __future__ import annotations\n", encoding="utf-8")
    for module_name in run_guard.RUNTIME_IMPORT_MODULES:
        if module_name == "radio_gs":
            relative = Path("radio_gs/__init__.py")
        else:
            relative = Path(*module_name.split(".")).with_suffix(".py")
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(
                "from __future__ import annotations\n",
                encoding="utf-8",
            )
    for relative in run_guard.REPO_SHELL_SOURCES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (tmp_path / "radio_gs/__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "radio_gs/models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "radio_gs/models/__init__.py").write_text("", encoding="utf-8")
    dependency = tmp_path / "radio_gs/models/transitive.py"
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    builder = tmp_path / "radio_gs/scripts/build_scannet_surface_region_cache.py"
    builder.write_text(
        "from radio_gs.models import transitive\n", encoding="utf-8"
    )
    before = run_guard.repo_source_closure(tmp_path)
    dependency.write_text("VALUE = 2\n", encoding="utf-8")
    after = run_guard.repo_source_closure(tmp_path)
    assert before["digest"] != after["digest"]
    assert (
        before["files"]["radio_gs/models/transitive.py"]
        != after["files"]["radio_gs/models/transitive.py"]
    )


def test_gpu1_lock_is_nofollow_single_link_nonblocking_and_inherited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (
        gpu1_lock._singleton_protocol()
        == gpu1_lock.GPU1_SINGLETON_PROTOCOL
    )
    lock_path = tmp_path / "physical_gpu1.lock"
    descriptor = gpu1_lock._open_canonical_lock(lock_path)
    address = f"\0radio-gs-test-inherited-{os.getpid()}".encode("ascii")
    singleton_descriptor = gpu1_lock._open_kernel_singleton(address)
    try:
        monkeypatch.setenv(gpu1_lock.LOCK_PATH_ENV, str(lock_path))
        monkeypatch.setenv(gpu1_lock.LOCK_FD_ENV, str(descriptor))
        record = gpu1_lock.verify_inherited_lock(descriptor, lock_path)
        assert record["path"] == str(lock_path)
        monkeypatch.setenv(
            gpu1_lock.SINGLETON_FD_ENV, str(singleton_descriptor)
        )
        monkeypatch.setenv(
            gpu1_lock.SINGLETON_PROTOCOL_ENV,
            gpu1_lock._singleton_protocol(address),
        )
        singleton_record = gpu1_lock.verify_inherited_singleton(
            singleton_descriptor, address
        )
        assert singleton_record["protocol"] == gpu1_lock._singleton_protocol(
            address
        )
        contender = os.open(lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(contender)
    finally:
        os.close(singleton_descriptor)
        os.close(descriptor)

    symlink = tmp_path / "symlink.lock"
    symlink.symlink_to(lock_path)
    with pytest.raises(OSError):
        gpu1_lock._open_canonical_lock(symlink)

    hardlink = tmp_path / "hardlink.lock"
    os.link(lock_path, hardlink)
    with pytest.raises(ValueError, match="one regular hard link"):
        gpu1_lock._open_canonical_lock(lock_path)


def test_gpu1_kernel_singleton_survives_lock_path_unlink_recreate(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "physical_gpu1.lock"
    address = f"\0radio-gs-test-recreate-{os.getpid()}".encode("ascii")
    first_lock = gpu1_lock._open_canonical_lock(lock_path)
    singleton = gpu1_lock._open_kernel_singleton(address)
    second_lock = -1
    try:
        lock_path.unlink()
        second_lock = gpu1_lock._open_canonical_lock(lock_path)
        with pytest.raises(OSError) as caught:
            gpu1_lock._open_kernel_singleton(address)
        assert caught.value.errno == errno.EADDRINUSE
    finally:
        if second_lock >= 0:
            os.close(second_lock)
        os.close(singleton)
        os.close(first_lock)


def test_gpu1_kernel_singleton_rejects_forged_non_socket_fd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_descriptor, write_descriptor = os.pipe()
    try:
        monkeypatch.setenv(
            gpu1_lock.SINGLETON_FD_ENV, str(read_descriptor)
        )
        monkeypatch.setenv(
            gpu1_lock.SINGLETON_PROTOCOL_ENV,
            gpu1_lock.GPU1_SINGLETON_PROTOCOL,
        )
        with pytest.raises(OSError):
            gpu1_lock.verify_inherited_singleton(read_descriptor)
    finally:
        os.close(read_descriptor)
        os.close(write_descriptor)


def test_runtime_fingerprint_rejects_import_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_guard, "RUNTIME_IMPORT_MODULES", ("radio_gs",))
    with pytest.raises(ValueError, match="escaped source snapshot"):
        run_guard.runtime_fingerprint(tmp_path)


def test_canary_telemetry_requires_safe_peak_and_paired_peer_events() -> None:
    rows = [
        _row(temp=52, event="sample_peer0_t70_p280_m100_u80"),
        _row(
            temp=68,
            event=(
                "soft_pause_peer0_activity_t77_p298_m100_u99_"
                "peer0_t77_p298_m100_u99"
            ),
            util=83,
        ),
        _row(temp=69, event="soft_cooldown_peer0_t76_p295_m100_u99"),
        _row(temp=71, event="soft_resume_peer0_t75_p260_m100_u0", power=196.0),
    ]
    summary = summarize_canary_telemetry(rows)
    assert summary["maximum_temperature_c"] == 71
    assert summary["maximum_power_w"] == 196.0
    assert summary["maximum_utilization_pct"] == 83
    assert summary["peer_pause_count"] == 1
    assert summary["peer_resume_count"] == 1
    assert summary["peer_interrupt_count"] == 0
    assert summary["fault_event_count"] == 0


def test_gpu1_only_canary_accepts_local_hysteresis_and_rejects_peer_events() -> None:
    rows = [
        _row(temp=74, event="sample"),
        _row(temp=75, event="soft_pause_gpu1_t75"),
        _row(temp=72, event="soft_cooldown"),
        _row(temp=70, event="soft_resume"),
    ]
    summary = summarize_canary_telemetry(
        rows,
        expected_gpu=1,
        maximum_temperature_c=78,
        peer_gpu=None,
    )
    assert summary["soft_pause_count"] == 1
    assert summary["soft_resume_count"] == 1
    assert summary["peer_pause_count"] == 0
    assert summary["peer_interrupt_count"] == 0

    with pytest.raises(ValueError, match="forbidden peer interruption"):
        summarize_canary_telemetry(
            [_row(temp=70, event="peer_activity_interrupt_release_cuda_peer0")],
            expected_gpu=1,
            maximum_temperature_c=78,
            peer_gpu=None,
        )


@pytest.mark.parametrize(
    "rows, message",
    [
        ([_row(temp=72, event="sample_peer0_t70_p280_m100_u80")], "above 71C"),
        ([_row(temp=60, event="thermal_abort_peer0_t70_p280_m100_u80")], "fault event"),
        (
            [
                _row(
                    temp=65,
                    event=(
                        "soft_pause_peer0_activity_t77_p298_m100_u99_"
                        "peer0_t77_p298_m100_u99"
                    ),
                )
            ],
            "not paired",
        ),
    ],
)
def test_canary_telemetry_fails_closed(
    rows: list[dict[str, str]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_canary_telemetry(rows)


def test_canary_resume_reuses_existing_report_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    telemetry_path = tmp_path / "gpu1.csv"
    telemetry_path.write_text("unused\n", encoding="utf-8")
    terminal_path = tmp_path / "train_shard0.pt"
    report_path = tmp_path / "control_train_shard0_canary.json"
    report_path.write_text(
        """{
  "telemetry_interval": {"start_line": 11, "end_line": 29},
  "kernel_journal_interval": {"start_epoch": 100, "end_epoch": 200}
}
""",
        encoding="utf-8",
    )
    manifest = {
        "thermal_safety_contract": {"physical_gpu": 1, "peer_gpu": 0},
        "canary_contract": {"maximum_temperature_c": 71},
    }
    closure = {"digest": "c" * 64}
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        run_guard,
        "verify_runtime_closure",
        lambda *_args, **_kwargs: (manifest, closure),
    )

    def fake_rows(_path: Path, *, start_line: int, end_line: int):
        observed["lines"] = (start_line, end_line)
        return [_row(temp=71, event="sample_peer0_t70_p200_m0_u0")]

    def fake_kernel(start_epoch: int, end_epoch: int) -> list[str]:
        observed["epochs"] = (start_epoch, end_epoch)
        return []

    monkeypatch.setattr(run_guard, "_telemetry_rows", fake_rows)
    monkeypatch.setattr(run_guard, "_kernel_faults", fake_kernel)
    monkeypatch.setattr(
        run_guard,
        "_audit_canary_cache",
        lambda *_args, **_kwargs: {"scenes": 8, "regions": 96},
    )
    monkeypatch.setattr(run_guard, "sha256_file", lambda _path: "a" * 64)
    monkeypatch.setattr(
        run_guard,
        "_atomic_publish_json",
        lambda _path, payload: observed.update({"payload": payload}),
    )
    report = run_guard.audit_canary(
        manifest_path=manifest_path,
        telemetry_path=telemetry_path,
        terminal_path=terminal_path,
        report_path=report_path,
        start_line=None,
        end_line=None,
        start_epoch=None,
        end_epoch=None,
    )
    assert observed["lines"] == (11, 29)
    assert observed["epochs"] == (100, 200)
    assert report["status"] == "canary_passed_resume_authorized"
    assert report["thermal_summary"]["maximum_temperature_c"] == 71


def _write_attempt_fixture(
    tmp_path: Path,
    *,
    events: list[str],
    result: str,
    status: int,
) -> tuple[Path, Path, Path, Path, dict]:
    manifest_path = tmp_path / "run_manifest.json"
    telemetry_path = tmp_path / "gpu1.csv"
    log_root = tmp_path / "logs"
    attempt_root = tmp_path / "attempts"
    stage = "cache_control_c256_geometric_train_0"
    stage_root = attempt_root / stage
    log_root.mkdir()
    stage_root.mkdir(parents=True)
    manifest = {
        "thermal_safety_contract": {
            "physical_gpu": 1,
            "peer_gpu": 0,
            "gpu_uuid": "GPU-test-uuid",
            "peer_activity_action": "terminate",
            "peer_activity_interrupt_exit_code": 87,
        },
        "canary_contract": {"maximum_temperature_c": 71},
        "attempt_receipt_contract": {
            "root": str(attempt_root),
            "log_root": str(log_root),
            "telemetry_path": str(telemetry_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    telemetry_path.write_text(
        ",".join(run_guard.TELEMETRY_COLUMNS)
        + "\n"
        + "".join(
            ",".join(
                [
                    "2026-08-01T00:00:00+08:00",
                    "1",
                    "0000:82:00.0",
                    "60",
                    "140",
                    "300",
                    "0",
                    "1",
                    "P2",
                    event,
                ]
            )
            + "\n"
            for event in events
        ),
        encoding="utf-8",
    )
    log = log_root / f"{stage}.attempt_000001.log"
    log.write_text("attempt\n", encoding="utf-8")
    kernel_log = log_root / f"{stage}.attempt_000001.kernel.log"
    kernel_log.write_text("clean kernel interval\n", encoding="utf-8")
    postflight = None
    if result == "peer_activity_interrupted_cuda_released_retry_authorized":
        postflight_path = (
            log_root / f"{stage}.attempt_000001.gpu_release_postflight.json"
        )
        postflight_path.write_text(
            json.dumps(
                {
                    "artifact_type": "surface-region-gpu-release-postflight-v1",
                    "schema_version": 1,
                    "status": "gpu_release_verified_clear",
                    "physical_gpu": 1,
                    "expected_uuid": "GPU-test-uuid",
                    "observed_uuid": "GPU-test-uuid",
                    "bus_id": "0000:82:00.0",
                    "pcie_config_prefix": "00112233",
                    "pcie_responsive": True,
                    "compute_query_succeeded": True,
                    "compute_owner_pids": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        postflight = {
            "capture_status": 0,
            "report": run_guard._artifact_file_record(
                postflight_path, label="postflight"
            ),
        }
    receipt_path = stage_root / "attempt_000001.json"
    receipt = {
        "artifact_type": "surface-region-stage-attempt-v1",
        "schema_version": 1,
        "run_manifest": run_guard._artifact_file_record(
            manifest_path, label="manifest"
        ),
        "stage": stage,
        "attempt_index": 1,
        "command": ["builder.py", "--seed", "0"],
        "command_status": status,
        "result": result,
        "log": run_guard._artifact_file_record(log, label="log"),
        "telemetry_interval": run_guard.telemetry_interval_record(
            telemetry_path,
            start_line=1,
            end_line=len(events) + 1,
        ),
        "kernel_journal": {
            "start_epoch": 100,
            "end_epoch": 200,
            "capture_status": 0,
            "fault_count": 0,
            "file": run_guard._artifact_file_record(
                kernel_log, label="kernel journal"
            ),
        },
        "gpu_release_postflight": postflight,
        "terminal": None,
        "sidecar": None,
        "peer_activity_action": "terminate",
        "peer_activity_interrupt_exit_code": 87,
    }
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    return manifest_path, telemetry_path, attempt_root, log_root, receipt


def test_attempt_receipt_reopen_rejects_command_log_and_symlink_tampering(
    tmp_path: Path,
) -> None:
    manifest, _telemetry, attempt_root, log_root, receipt = _write_attempt_fixture(
        tmp_path,
        events=["peer_activity_interrupt_release_cuda_peer0_activity"],
        result="peer_activity_interrupted_cuda_released_retry_authorized",
        status=87,
    )
    stage = receipt["stage"]
    receipt_path = attempt_root / stage / "attempt_000001.json"
    log = log_root / f"{stage}.attempt_000001.log"
    validate_attempt_receipt(
        manifest_path=manifest,
        receipt_path=receipt_path,
        expected_stage=stage,
        expected_index=1,
        expected_log=log,
        expected_command=receipt["command"],
    )
    with pytest.raises(ValueError, match="command differs"):
        validate_attempt_receipt(
            manifest_path=manifest,
            receipt_path=receipt_path,
            expected_stage=stage,
            expected_index=1,
            expected_log=log,
            expected_command=["different.py"],
        )
    log.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="log record differs"):
        validate_attempt_receipt(
            manifest_path=manifest,
            receipt_path=receipt_path,
            expected_stage=stage,
            expected_index=1,
            expected_log=log,
            expected_command=receipt["command"],
        )
    log.write_text("attempt\n", encoding="utf-8")
    copied = tmp_path / "copied-receipt.json"
    copied.write_bytes(receipt_path.read_bytes())
    receipt_path.unlink()
    receipt_path.symlink_to(copied)
    with pytest.raises(OSError):
        validate_attempt_receipt(
            manifest_path=manifest,
            receipt_path=receipt_path,
            expected_stage=stage,
            expected_index=1,
            expected_log=log,
            expected_command=receipt["command"],
        )


def test_gpu1_only_attempt_inventory_binds_owner_audit_beside_receipt(
    tmp_path: Path,
) -> None:
    manifest_path, _telemetry, attempt_root, log_root, receipt = (
        _write_attempt_fixture(
            tmp_path,
            events=["sample"],
            result="completed",
            status=0,
        )
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["thermal_safety_contract"].update(
        {
            "peer_gpu": None,
            "owner_pid_namespace_mode": (
                "exclusive-singleton-after-clear-v1"
            ),
        }
    )
    manifest["attempt_receipt_contract"].update(
        {
            "owner_audit_required": True,
            "owner_audit_location": "beside_receipt",
        }
    )
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    stage = receipt["stage"]
    receipt_path = attempt_root / stage / "attempt_000001.json"
    owner_audit = receipt_path.with_suffix(".owner_audit.csv")
    owner_audit.write_text(
        "timestamp,gpu_uuid,child_pgid,owner_pids,child_owner_pids,"
        "foreign_owner_pids,event\n"
        "2026-08-01T00:00:00+08:00,GPU-test-uuid,123,,,,"
        "prelaunch_owner_clear\n"
        "2026-08-01T00:00:01+08:00,GPU-test-uuid,123,999,999,,"
        "runtime_owner_audit_host_pid_singleton\n"
        "2026-08-01T00:00:02+08:00,GPU-test-uuid,123,,,,"
        "postexit_owner_clear\n",
        encoding="utf-8",
    )
    receipt["run_manifest"] = run_guard._artifact_file_record(
        manifest_path, label="manifest"
    )
    receipt["owner_audit"] = run_guard._artifact_file_record(
        owner_audit, label="owner audit"
    )
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    inventory = audit_attempt_inventory(
        manifest_path=manifest_path,
        attempt_root=attempt_root,
        log_root=log_root,
    )
    assert inventory["attempts"][0]["owner_audit"] == receipt["owner_audit"]

    owner_audit.write_text(
        owner_audit.read_text(encoding="utf-8").replace(
            ",999,999,,runtime_owner_audit_host_pid_singleton",
            ",999,1000,,runtime_owner_audit_host_pid_singleton",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="owner audit record differs"):
        audit_attempt_inventory(
            manifest_path=manifest_path,
            attempt_root=attempt_root,
            log_root=log_root,
        )


def test_attempt_inventory_rejects_missing_receipt_and_extra_peer_event(
    tmp_path: Path,
) -> None:
    manifest, _telemetry, attempt_root, log_root, receipt = _write_attempt_fixture(
        tmp_path,
        events=[
            "peer_activity_interrupt_release_cuda_peer0_activity",
            "peer_activity_interrupt_release_cuda_peer0_activity",
        ],
        result="peer_activity_interrupted_cuda_released_retry_authorized",
        status=87,
    )
    with pytest.raises(ValueError, match="result/telemetry differs"):
        audit_attempt_inventory(
            manifest_path=manifest,
            attempt_root=attempt_root,
            log_root=log_root,
        )
    receipt_path = attempt_root / receipt["stage"] / "attempt_000001.json"
    receipt_path.unlink()
    with pytest.raises(ValueError, match="empty"):
        audit_attempt_inventory(
            manifest_path=manifest,
            attempt_root=attempt_root,
            log_root=log_root,
        )


def test_attempt_inventory_rejects_overlapping_telemetry_intervals(
    tmp_path: Path,
) -> None:
    manifest, _telemetry, attempt_root, log_root, receipt = _write_attempt_fixture(
        tmp_path,
        events=["sample_peer0_t70_p200_m0_u0"],
        result="completed",
        status=0,
    )
    second_stage = "cache_control_c256_geometric_validation_0"
    second_root = attempt_root / second_stage
    second_root.mkdir()
    second_log = log_root / f"{second_stage}.attempt_000001.log"
    second_log.write_text("attempt two\n", encoding="utf-8")
    second_kernel = log_root / f"{second_stage}.attempt_000001.kernel.log"
    second_kernel.write_text("clean kernel interval\n", encoding="utf-8")
    second = json.loads(json.dumps(receipt))
    second["stage"] = second_stage
    second["log"] = run_guard._artifact_file_record(second_log, label="log")
    second["kernel_journal"]["file"] = run_guard._artifact_file_record(
        second_kernel, label="kernel journal"
    )
    (second_root / "attempt_000001.json").write_text(
        json.dumps(second) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="overlap or repeat"):
        audit_attempt_inventory(
            manifest_path=manifest,
            attempt_root=attempt_root,
            log_root=log_root,
        )


def test_attempt_receipt_rejects_telemetry_kernel_and_postflight_tampering(
    tmp_path: Path,
) -> None:
    manifest, telemetry, attempt_root, log_root, receipt = _write_attempt_fixture(
        tmp_path,
        events=["peer_activity_interrupt_release_cuda_peer0_activity"],
        result="peer_activity_interrupted_cuda_released_retry_authorized",
        status=87,
    )
    stage = receipt["stage"]
    receipt_path = attempt_root / stage / "attempt_000001.json"
    log = log_root / f"{stage}.attempt_000001.log"

    telemetry.write_text(
        telemetry.read_text(encoding="utf-8").replace(
            "peer_activity_interrupt_release_cuda_peer0_activity",
            "sample_peer0_t70_p200_m0_u0",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="telemetry digest differs"):
        validate_attempt_receipt(
            manifest_path=manifest,
            receipt_path=receipt_path,
            expected_stage=stage,
            expected_index=1,
            expected_log=log,
        )
    telemetry.write_text(
        ",".join(run_guard.TELEMETRY_COLUMNS)
        + "\n2026-08-01T00:00:00+08:00,1,0000:82:00.0,60,140,300,0,1,P2,"
        "peer_activity_interrupt_release_cuda_peer0_activity\n",
        encoding="utf-8",
    )
    kernel = log_root / f"{stage}.attempt_000001.kernel.log"
    kernel.write_text("NVRM: Xid 79, GPU has fallen off the bus\n", encoding="utf-8")
    receipt["kernel_journal"]["fault_count"] = 1
    receipt["kernel_journal"]["file"] = run_guard._artifact_file_record(
        kernel, label="kernel journal"
    )
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="result/telemetry differs"):
        validate_attempt_receipt(
            manifest_path=manifest,
            receipt_path=receipt_path,
            expected_stage=stage,
            expected_index=1,
            expected_log=log,
        )

    kernel.write_text("clean kernel interval\n", encoding="utf-8")
    receipt["kernel_journal"]["fault_count"] = 0
    receipt["kernel_journal"]["file"] = run_guard._artifact_file_record(
        kernel, label="kernel journal"
    )
    postflight_path = (
        log_root / f"{stage}.attempt_000001.gpu_release_postflight.json"
    )
    postflight_payload = json.loads(postflight_path.read_text(encoding="utf-8"))
    postflight_payload["status"] = "gpu_release_not_clear"
    postflight_payload["compute_owner_pids"] = [4242]
    postflight_path.write_text(
        json.dumps(postflight_payload) + "\n", encoding="utf-8"
    )
    receipt["gpu_release_postflight"]["report"] = (
        run_guard._artifact_file_record(postflight_path, label="postflight")
    )
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="result/telemetry differs"):
        validate_attempt_receipt(
            manifest_path=manifest,
            receipt_path=receipt_path,
            expected_stage=stage,
            expected_index=1,
            expected_log=log,
        )


def test_kernel_journal_capture_freezes_fault_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoded = b"kernel: clean\nNVRM: Xid 79, GPU has fallen off the bus\n"
    monkeypatch.setattr(
        run_guard.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=encoded),
    )
    output = tmp_path / "attempt.kernel.log"
    record = run_guard.capture_kernel_journal(
        start_epoch=100,
        end_epoch=200,
        output_path=output,
    )
    assert record["capture_status"] == 0
    assert record["fault_count"] == 1
    assert record["file"] == run_guard._artifact_file_record(
        output, label="kernel journal"
    )


def test_canary_rejects_peer_release_event_without_authorized_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, telemetry, attempt_root, _log_root, receipt = (
        _write_attempt_fixture(
            tmp_path,
            events=[
                "peer_activity_interrupt_release_cuda_peer0_activity",
                "sample_peer0_t70_p200_m0_u0",
            ],
            result="completed",
            status=0,
        )
    )
    receipt_path = attempt_root / receipt["stage"] / "attempt_000001.json"
    receipt["telemetry_interval"] = run_guard.telemetry_interval_record(
        telemetry,
        start_line=2,
        end_line=3,
    )
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        run_guard,
        "verify_runtime_closure",
        lambda *_args, **_kwargs: (manifest, {"digest": "c" * 64}),
    )
    monkeypatch.setattr(run_guard, "_kernel_faults", lambda *_args: [])
    monkeypatch.setattr(
        run_guard,
        "_audit_canary_cache",
        lambda *_args, **_kwargs: {"scenes": 8, "regions": 96},
    )
    with pytest.raises(ValueError, match="not paired"):
        run_guard.audit_canary(
            manifest_path=manifest_path,
            telemetry_path=telemetry,
            terminal_path=tmp_path / "terminal.pt",
            report_path=tmp_path / "canary.json",
            start_line=1,
            end_line=3,
            start_epoch=100,
            end_epoch=200,
        )


def test_run_repo_python_prefers_its_source_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "surface_source_snapshot"
    shutil.copytree(REPO_ROOT / "radio_gs", snapshot / "radio_gs")
    command = r"""
import json
from pathlib import Path
from radio_gs.scripts.surface_region_run_guard import runtime_fingerprint

root = Path(__import__('os').environ['RADIO_GS_REPO_ROOT']).resolve()
fingerprint = runtime_fingerprint(root)
for record in fingerprint['imported_modules'].values():
    assert Path(record['path']).resolve().is_relative_to(root)
print(json.dumps({
    'root': str(root),
    'imports': {
        name: value['path']
        for name, value in fingerprint['imported_modules'].items()
    },
}, sort_keys=True))
"""
    environment = os.environ.copy()
    environment["RADIO_GS_PYTHON"] = sys.executable
    completed = subprocess.run(
        [
            "bash",
            str(snapshot / "radio_gs/scripts/run_repo_python.sh"),
            "-c",
            command,
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert f'"root": "{snapshot}"' in completed.stdout
    assert str(REPO_ROOT / "radio_gs") not in completed.stdout
