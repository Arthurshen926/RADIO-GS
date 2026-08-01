from __future__ import annotations

import argparse
import errno
import os
import json
from pathlib import Path
import sys

import pytest

from radio_gs.scripts import finalize_surface_text_response_promotion as promotion
from radio_gs.scripts import surface_gpu1_lock_supervisor as gpu1_lock
from radio_gs.scripts import surface_text_response_distill_authority as authority
from radio_gs.scripts import train_surface_region_text_response_distill as trainer


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "radio_gs/scripts/run_surface_region_text_response_distill.sh"


def _history() -> list[dict[str, float | int]]:
    rows = [
        {
            "epoch": 0,
            "surface_selection_score": 0.96,
            "summary_token_cosine": 0.96,
            "mean_descriptor_cosine": 0.96,
            "all_view_descriptor_cosine": 0.96,
            "text_support_top1_agreement": 0.5,
            "text_response_smooth_l1": 0.02,
            "descriptor_relation_smooth_l1": 0.001,
        },
        {
            "epoch": 1,
            "surface_selection_score": 0.95,
            "summary_token_cosine": 0.959,
            "mean_descriptor_cosine": 0.959,
            "all_view_descriptor_cosine": 0.959,
            "text_support_top1_agreement": 1.0,
            "text_response_smooth_l1": 0.01,
            "descriptor_relation_smooth_l1": 0.003,
        },
        {
            "epoch": 2,
            "surface_selection_score": 0.97,
            "summary_token_cosine": 0.97,
            "mean_descriptor_cosine": 0.97,
            "all_view_descriptor_cosine": 0.97,
            "text_support_top1_agreement": 1.0,
            "text_response_smooth_l1": 0.02,
            "descriptor_relation_smooth_l1": 0.004,
        },
    ]
    return trainer.finalize_response_primary_epoch_selection(rows)[0]


def test_runner_freezes_gpu1_authority_defaults_and_per_seed_receipts() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'LOCK_ROOT="/root/RADIO-GS/output"' in source
    assert 'GLOBAL_GPU1_LOCK="$LOCK_ROOT/.physical_gpu1.lock"' in source
    assert "TEXT_RESPONSE_DISTILL_LOCK_HELD" not in source
    assert "TEXT_RESPONSE_DISTILL_GLOBAL_LOCK_FD" in source
    assert "TEXT_RESPONSE_DISTILL_RUN_LOCK_FD" in source
    assert "RADIO_GS_GPU1_SINGLETON_FD" in source
    assert "RADIO_GS_GPU1_SINGLETON_PROTOCOL" in source
    assert (
        authority.LOCK_ROOT_BINDING_ENV
        == "TEXT_RESPONSE_DISTILL_LOCK_ROOT_BINDING_SHA256"
    )
    assert "linux-abstract-af-unix-stream-v1:radio-gs-physical-gpu1-v1" in source
    assert "--singleton-fd" in source
    assert "verify-lock-fds" in source
    assert "git -C" not in source
    assert 'GPU_MAX_TEMP_C="${GPU_MAX_TEMP_C:-78}"' in source
    assert 'GPU_START_MAX_TEMP_C="${GPU_START_MAX_TEMP_C:-65}"' in source
    assert 'GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-3}"' in source
    assert 'GPU_SOFT_PAUSE_TEMP_C="${GPU_SOFT_PAUSE_TEMP_C:-75}"' in source
    assert 'GPU_SOFT_RESUME_TEMP_C="${GPU_SOFT_RESUME_TEMP_C:-70}"' in source
    assert 'GPU_PEER_INDEX=""' in source
    assert "GPU_PEER_PAUSE_TEMP_C=0" in source
    assert "GPU_PEER_RESUME_TEMP_C=0" in source
    assert "GPU_PEER_MAX_POWER_W=0" in source
    assert "GPU_PEER_MAX_MEMORY_MIB=0" in source
    assert "GPU_PEER_MAX_UTIL_PCT=100" in source
    assert (
        'GPU_OWNER_PID_NAMESPACE_MODE="exclusive-singleton-after-clear-v1"'
        in source
    )
    assert (
        '--gpu-owner-pid-namespace-mode "$GPU_OWNER_PID_NAMESPACE_MODE"'
        in source
    )
    assert (
        'GPU_OWNER_PID_NAMESPACE_MODE="$GPU_OWNER_PID_NAMESPACE_MODE"'
        in source
    )
    assert 'GPU_TELEMETRY_LOG="$telemetry"' in source
    assert 'f"seed{seed}.csv"' not in source
    assert "seed${seed}.csv" in source
    assert "prepare-command" in source and "finalize" in source
    assert "bind_surface_control" in source
    assert "--surface-control-checkpoint" in source
    assert "--surface-control-checkpoint-sha256" in source
    assert "journalctl -k" in source
    assert 'authority verify-manifest "${MANIFEST_ARGUMENTS[@]}"' in source
    assert "finalize-seed" in source and "verify-seed" in source
    assert authority.EPOCH_SELECTION in (
        REPO_ROOT / "radio_gs/scripts/train_surface_region_text_response_distill.py"
    ).read_text(encoding="utf-8")


def _manifest_cli_arguments(
    tmp_path: Path,
    *,
    command: str,
    owner_pid_namespace_mode: str,
) -> list[str]:
    return [
        command,
        "--repo-root",
        str(tmp_path),
        "--lock-root",
        str(tmp_path / "lock-root"),
        "--candidate",
        "context_c1024_geometric",
        "--surface-root",
        str(tmp_path / "surface"),
        "--output-root",
        str(tmp_path / "output"),
        "--train-caches",
        str(tmp_path / "train*.pt"),
        "--validation-caches",
        str(tmp_path / "validation*.pt"),
        "--fit-text-bank",
        str(tmp_path / "fit.pt"),
        "--fit-text-bank-manifest",
        str(tmp_path / "fit.json"),
        "--radio-checkpoint",
        str(tmp_path / "radio.pt"),
        "--calibration-manifest",
        f"0={tmp_path / 'calibration0.json'}",
        "--calibration-audit",
        f"0={tmp_path / 'calibration0.audit.json'}",
        "--calibration-manifest",
        f"1={tmp_path / 'calibration1.json'}",
        "--calibration-audit",
        f"1={tmp_path / 'calibration1.audit.json'}",
        "--calibration-manifest",
        f"2={tmp_path / 'calibration2.json'}",
        "--calibration-audit",
        f"2={tmp_path / 'calibration2.audit.json'}",
        "--gradient-diagnostic",
        str(tmp_path / "gradient-diagnostic.json"),
        "--gradient-diagnostic-sha256",
        "a" * 64,
        "--initial-gpu-preflight",
        str(tmp_path / "gpu.initial.json"),
        "--thermal-guard",
        str(tmp_path / "guard.sh"),
        "--run-manifest",
        str(tmp_path / "run_manifest.json"),
        "--gpu-max-temp-c",
        "78",
        "--gpu-start-max-temp-c",
        "65",
        "--gpu-max-power-limit-w",
        "300.5",
        "--gpu-poll-seconds",
        "3",
        "--gpu-soft-pause-temp-c",
        "75",
        "--gpu-soft-resume-temp-c",
        "70",
        "--gpu-peer-pause-temp-c",
        "0",
        "--gpu-peer-resume-temp-c",
        "0",
        "--gpu-peer-quiet-seconds",
        "0",
        "--gpu-peer-max-power-w",
        "0",
        "--gpu-peer-max-memory-mib",
        "0",
        "--gpu-peer-max-util-pct",
        "100",
        "--gpu-peer-activity-action",
        "terminate",
        "--gpu-owner-pid-namespace-mode",
        owner_pid_namespace_mode,
    ]


def test_manifest_cli_and_thermal_contract_freeze_owner_pid_namespace_mode(
    tmp_path: Path,
) -> None:
    guard = tmp_path / "guard.sh"
    guard.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    parser = authority.build_parser()
    expected = "exclusive-singleton-after-clear-v1"
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    for command in ("create-manifest", "verify-manifest"):
        mode_action = next(
            action
            for action in subparsers.choices[command]._actions
            if action.dest == "gpu_owner_pid_namespace_mode"
        )
        assert mode_action.required is True
        args = parser.parse_args(
            _manifest_cli_arguments(
                tmp_path,
                command=command,
                owner_pid_namespace_mode=expected,
            )
        )
        assert isinstance(args, argparse.Namespace)
        assert args.gpu_owner_pid_namespace_mode == expected
        thermal = authority._thermal_contract(args, args.thermal_guard)
        assert thermal["owner_pid_namespace_mode"] == expected

    for rejected in ("strict", "exclusive-singleton-after-clear-v2"):
        rejected_args = parser.parse_args(
            _manifest_cli_arguments(
                tmp_path,
                command="create-manifest",
                owner_pid_namespace_mode=rejected,
            )
        )
        with pytest.raises(ValueError, match="owner PID namespace contract"):
            authority._thermal_contract(
                rejected_args,
                rejected_args.thermal_guard,
            )


def test_authority_output_index_never_reuses_telemetry_between_seeds(
    tmp_path: Path,
) -> None:
    rows = authority._seed_outputs(tmp_path, "context_c1024_geometric")
    telemetry = [row["guard_telemetry"] for row in rows]
    receipts = [row["guard_receipt"] for row in rows]
    terminals = [row["terminal"] for row in rows]
    assert len(set(telemetry)) == len(set(receipts)) == len(set(terminals)) == 3
    assert [Path(path).name for path in telemetry] == [
        "seed0.csv",
        "seed1.csv",
        "seed2.csv",
    ]


def test_gpu_observation_rejects_owner_bus_and_dead_pci() -> None:
    valid = dict(
        phase="pre_seed0",
        gpu_uuid="GPU-123456789",
        nvidia_bus_id="00000000:82:00.0",
        proc_bus_id="0000:82:00.0",
        pci_prefix="de100022000000000000000000000000",
        compute_owners=[],
        observed_epoch=1_754_000_000,
    )
    payload = authority.gpu_check_payload(**valid)
    assert payload["gpu_identity"]["physical_index"] == 1
    with pytest.raises(ValueError, match="compute owners"):
        authority.gpu_check_payload(**{**valid, "compute_owners": ["1234"]})
    with pytest.raises(ValueError, match="bus identity"):
        authority.gpu_check_payload(**{**valid, "proc_bus_id": "0000:83:00.0"})
    with pytest.raises(ValueError, match="not responding"):
        authority.gpu_check_payload(**{**valid, "pci_prefix": "f" * 32})


def _minimal_training_manifest(tmp_path: Path) -> tuple[dict[str, object], Path]:
    repo = tmp_path / "snapshot"
    repo.mkdir()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for index in range(4):
        (inputs / f"train_{index}.pt").write_bytes(f"train-{index}".encode())
    for index in range(2):
        (inputs / f"validation_{index}.pt").write_bytes(f"validation-{index}".encode())
    fit = inputs / "fit.pt"
    fit_manifest = inputs / "fit.json"
    radio = inputs / "radio.pt"
    for path in (fit, fit_manifest, radio):
        path.write_bytes(path.name.encode())
    calibrations = {}
    calibration_rows = []
    for seed in range(3):
        calibration = inputs / f"calibration{seed}.json"
        audit = inputs / f"calibration{seed}.audit.json"
        calibration.write_bytes(calibration.name.encode())
        audit.write_bytes(audit.name.encode())
        calibrations[str(seed)] = str(calibration)
        calibration_rows.append(
            {
                "seed": seed,
                "manifest": authority.file_record(calibration),
                "audit": authority.file_record(audit),
            }
        )
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text("{}\n", encoding="utf-8")
    arguments = {
        "train_caches": str(inputs / "train_*.pt"),
        "validation_caches": str(inputs / "validation_*.pt"),
        "fit_text_bank": str(fit),
        "fit_text_bank_manifest": str(fit_manifest),
        "calibration_manifests": calibrations,
        "run_manifest": str(run_manifest),
        "radio_checkpoint": str(radio),
        "output_root": str(tmp_path / "output"),
    }
    outputs = authority._seed_outputs(tmp_path / "output", "context_c1024_geometric")
    surface_controls = []
    for seed in range(3):
        control = inputs / f"surface-control-seed{seed}.pt"
        control.write_bytes(f"surface-control-{seed}".encode())
        surface_controls.append(
            {
                "seed": seed,
                "checkpoint": authority.file_record(control),
                "best_epoch": 1,
                "best_selection_score": 0.9,
                "validation": {
                    "summary_token_cosine": 0.8,
                    "mean_descriptor_cosine": 0.9,
                    "all_view_descriptor_cosine": 0.9,
                },
            }
        )
    surface_promotion = {
        "binding_mode": authority.ATTENTION_BINDING_MODE,
        "selected_variant": authority.CONTEXT_POOLING_MODE,
        "selected_readouts": surface_controls,
    }
    manifest: dict[str, object] = {
        "candidate": "context_c1024_geometric",
        "gpu_identity": {
            "physical_index": 1,
            "uuid": "GPU-123456789",
            "pci_bus_id": "00000000:82:00.0",
        },
        "authority_contract": {
            "source_snapshot_root": str(repo),
            "output_root": str(tmp_path / "output"),
        },
        "train_caches": [
            authority.file_record(inputs / f"train_{index}.pt") for index in range(4)
        ],
        "validation_caches": [
            authority.file_record(inputs / f"validation_{index}.pt") for index in range(2)
        ],
        "fit_text_bank": {
            "artifact": authority.file_record(fit),
            "manifest": authority.file_record(fit_manifest),
        },
        "calibrations": calibration_rows,
        "radio_checkpoint": authority.file_record(radio),
        "surface_promotion": surface_promotion,
        "outputs": outputs,
    }
    manifest["training_command_contract"] = authority._build_training_command_contract(
        repo_root=repo,
        arguments=arguments,
        outputs=outputs,
        surface_promotion=surface_promotion,
    )
    return manifest, run_manifest


def test_receipt_command_requires_current_manifest_scene_seed_and_complete_argv(
    tmp_path: Path,
) -> None:
    manifest, run_manifest = _minimal_training_manifest(tmp_path)
    expected = authority.expected_training_argv(
        manifest,
        manifest_path=run_manifest,
        seed=1,
    )
    control = manifest["surface_promotion"]["selected_readouts"][1][
        "checkpoint"
    ]
    assert expected[expected.index("--surface-control-checkpoint") + 1] == control[
        "path"
    ]
    assert expected[
        expected.index("--surface-control-checkpoint-sha256") + 1
    ] == control["sha256"]
    command = {
        "run_manifest": authority.file_record(run_manifest),
        "seed": 1,
        "scene": manifest["candidate"],
        "gpu_identity": manifest["gpu_identity"],
        "argv": expected,
        "argv_sha256": authority.canonical_json_sha256(expected),
        "prepared_epoch": 1_754_000_001,
    }
    assert authority.validate_receipt_training_command(
        command,
        manifest=manifest,
        manifest_path=run_manifest,
        manifest_sha256=authority.sha256_file(run_manifest),
        seed=1,
    ) == 1_754_000_001

    another_manifest = tmp_path / "another_manifest.json"
    another_manifest.write_text("{}\n", encoding="utf-8")
    attacks = [
        {**command, "run_manifest": authority.file_record(another_manifest)},
        {**command, "scene": "another_candidate"},
        {**command, "seed": 2},
        {**command, "argv": ["bash", "arbitrary.py"]},
    ]
    for attack in attacks:
        with pytest.raises(ValueError, match="complete training argv"):
            authority.validate_receipt_training_command(
                attack,
                manifest=manifest,
                manifest_path=run_manifest,
                manifest_sha256=authority.sha256_file(run_manifest),
                seed=1,
            )

    tampered_manifest = json.loads(json.dumps(manifest))
    tampered_manifest["training_command_contract"]["commands"][1]["argv"][-1] = "other.pt"
    with pytest.raises(ValueError, match="was not reproduced"):
        authority.validate_training_command_contract(
            tampered_manifest,
            manifest_path=run_manifest,
        )


def test_journal_and_telemetry_intervals_reject_cross_seed_replay(tmp_path: Path) -> None:
    start = 1_754_000_000
    end = start + 10
    journal = tmp_path / "seed0.kernel.log"
    journal.write_text(
        f"surface_text_response_seed=0\tstart_epoch={start}\tend_epoch={end}\n"
        "kernel interval clean\n",
        encoding="utf-8",
    )
    record = authority._kernel_journal_record(
        journal,
        seed=0,
        start_epoch=start,
        end_epoch=end,
    )
    assert record["fault_count"] == 0
    with pytest.raises(ValueError, match="interval header differs"):
        authority._kernel_journal_record(
            journal,
            seed=1,
            start_epoch=start,
            end_epoch=end,
        )

    telemetry = tmp_path / "seed0.csv"
    first_timestamp = "2026-08-01T12:00:02+08:00"
    last_timestamp = "2026-08-01T12:00:03+08:00"
    telemetry.write_text(
        ",".join(authority.TELEMETRY_COLUMNS)
        + "\n"
        + f"{first_timestamp},1,0000:82:00.0,50,20,300,10,100,P2,running\n"
        + f"{last_timestamp},1,0000:82:00.0,51,21,300,11,101,P2,running\n",
        encoding="utf-8",
    )
    interval = authority._telemetry_interval_record(
        telemetry,
        seed=0,
        receipt_summary={
            "sample_count": 2,
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
        },
    )
    first_epoch = int(interval["first_epoch"])
    authority.validate_seed_execution_timeline(
        seed=0,
        gpu_preflight_epoch=first_epoch - 3,
        command_prepared_epoch=first_epoch - 2,
        journal_start_epoch=first_epoch - 1,
        telemetry_first_epoch=interval["first_epoch"],
        telemetry_last_epoch=interval["last_epoch"],
        journal_end_epoch=first_epoch + 2,
        gpu_postflight_epoch=first_epoch + 3,
    )
    with pytest.raises(ValueError, match="timeline is not strictly bound"):
        authority.validate_seed_execution_timeline(
            seed=1,
            gpu_preflight_epoch=first_epoch + 20,
            command_prepared_epoch=first_epoch + 21,
            journal_start_epoch=first_epoch + 22,
            telemetry_first_epoch=interval["first_epoch"],
            telemetry_last_epoch=interval["last_epoch"],
            journal_end_epoch=first_epoch + 30,
            gpu_postflight_epoch=first_epoch + 31,
        )


def test_completion_rejects_replayed_clean_seed_evidence() -> None:
    rows = []
    for seed in range(3):
        rows.append(
            {
                "guard_command": {"sha256": str(seed) * 64},
                "telemetry_interval": {
                    "sha256": str(seed + 3) * 64,
                    "row_interval_sha256": str(seed + 6) * 64,
                },
                "kernel_journal": {"sha256": str(seed + 9) * 64},
                "execution_timeline": {
                    "journal_start_epoch": 100 + seed * 20,
                    "journal_end_epoch": 110 + seed * 20,
                },
            }
        )
    authority.validate_cross_seed_replay(rows)
    rows[2]["telemetry_interval"] = dict(rows[0]["telemetry_interval"])
    with pytest.raises(ValueError, match="cross-seed replay"):
        authority.validate_cross_seed_replay(rows)


def test_nofollow_nonblocking_lock_rejects_symlink_and_contention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "frozen-source-without-git"
    repo.mkdir()
    lock_parent = tmp_path / "main"
    lock_parent.mkdir()
    lock_root = lock_parent / "output"
    lock_root.mkdir()
    monkeypatch.setattr(authority, "CANONICAL_LOCK_ROOT", lock_root)
    monkeypatch.setattr(
        authority,
        "GPU1_SINGLETON_ADDRESS",
        f"\0radio-gs-text-lock-test-{os.getpid()}".encode("ascii"),
    )
    output = lock_root / "runs/run"
    status = authority.run_locked(
        repo_root=repo,
        lock_root=lock_root,
        output_root=output,
        command=[sys.executable, "-c", "raise SystemExit(0)"],
    )
    assert status == 0
    global_lock = lock_root / ".physical_gpu1.lock"
    assert global_lock.is_file() and not global_lock.is_symlink()

    descriptor = authority._acquire_nofollow_lock(global_lock)
    try:
        with pytest.raises(RuntimeError, match="already held"):
            authority.run_locked(
                repo_root=repo,
                lock_root=lock_root,
                output_root=output,
                command=[sys.executable, "-c", "raise SystemExit(0)"],
            )
    finally:
        os.close(descriptor)

    global_lock.unlink()
    target = tmp_path / "alias.lock"
    target.touch()
    global_lock.symlink_to(target)
    with pytest.raises(OSError):
        authority.run_locked(
            repo_root=repo,
            lock_root=lock_root,
            output_root=output,
            command=[sys.executable, "-c", "raise SystemExit(0)"],
        )


def test_inherited_fd_contract_cannot_be_forged_by_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_root = tmp_path / "main/output"
    output = lock_root / "run"
    (output / "locks").mkdir(parents=True)
    monkeypatch.setattr(authority, "CANONICAL_LOCK_ROOT", lock_root)
    address = f"\0radio-gs-text-inherited-{os.getpid()}".encode("ascii")
    monkeypatch.setattr(authority, "GPU1_SINGLETON_ADDRESS", address)
    global_path = lock_root / ".physical_gpu1.lock"
    run_path = output / "locks/text_response_distill.run.lock"
    global_fd = authority._acquire_nofollow_lock(global_path)
    run_fd = authority._acquire_nofollow_lock(run_path)
    singleton_fd = gpu1_lock._open_kernel_singleton(address)
    try:
        root_binding = authority.inspect_canonical_lock_root(lock_root)
        monkeypatch.setenv(
            authority.LOCK_ROOT_BINDING_ENV,
            authority.canonical_json_sha256(root_binding),
        )
        monkeypatch.setenv(gpu1_lock.SINGLETON_FD_ENV, str(singleton_fd))
        monkeypatch.setenv(
            gpu1_lock.SINGLETON_PROTOCOL_ENV,
            gpu1_lock._singleton_protocol(address),
        )
        verified = authority.verify_inherited_locks(
            lock_root=lock_root,
            output_root=output,
            global_descriptor=global_fd,
            run_descriptor=run_fd,
            singleton_descriptor=singleton_fd,
        )
        assert verified["global_lock"] == str(global_path)
        assert verified["kernel_singleton"]["protocol"] == gpu1_lock._singleton_protocol(
            address
        )
        monkeypatch.setenv(authority.LOCK_ROOT_BINDING_ENV, "0" * 64)
        with pytest.raises(ValueError, match="root binding differs"):
            authority.verify_inherited_locks(
                lock_root=lock_root,
                output_root=output,
                global_descriptor=global_fd,
                run_descriptor=run_fd,
                singleton_descriptor=singleton_fd,
            )
        monkeypatch.setenv(
            authority.LOCK_ROOT_BINDING_ENV,
            authority.canonical_json_sha256(root_binding),
        )
        read_fd, write_fd = os.pipe()
        try:
            with pytest.raises(ValueError, match="does not own"):
                authority.verify_inherited_locks(
                    lock_root=lock_root,
                    output_root=output,
                    global_descriptor=read_fd,
                    run_descriptor=run_fd,
                    singleton_descriptor=singleton_fd,
                )
            monkeypatch.setenv(gpu1_lock.SINGLETON_FD_ENV, str(read_fd))
            with pytest.raises(OSError):
                authority.verify_inherited_locks(
                    lock_root=lock_root,
                    output_root=output,
                    global_descriptor=global_fd,
                    run_descriptor=run_fd,
                    singleton_descriptor=read_fd,
                )
        finally:
            os.close(read_fd)
            os.close(write_fd)
    finally:
        os.close(singleton_fd)
        os.close(run_fd)
        os.close(global_fd)


def test_text_and_surface_singleton_blocks_unlink_recreate_cross_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_root = tmp_path / "main/output"
    lock_root.mkdir(parents=True)
    output = lock_root / "text-run"
    lock_path = lock_root / ".physical_gpu1.lock"
    address = f"\0radio-gs-cross-runner-{os.getpid()}".encode("ascii")
    monkeypatch.setattr(authority, "CANONICAL_LOCK_ROOT", lock_root)
    monkeypatch.setattr(authority, "GPU1_SINGLETON_ADDRESS", address)

    surface_file_lock = gpu1_lock._open_canonical_lock(lock_path)
    surface_singleton = gpu1_lock._open_kernel_singleton(address)
    try:
        lock_path.unlink()
        with pytest.raises(OSError) as caught:
            authority.run_locked(
                repo_root=tmp_path,
                lock_root=lock_root,
                output_root=output,
                command=[sys.executable, "-c", "raise SystemExit(0)"],
            )
        assert caught.value.errno == errno.EADDRINUSE
        assert lock_path.is_file()
    finally:
        os.close(surface_singleton)
        os.close(surface_file_lock)


def test_run_locked_rejects_any_noncanonical_lock_root(tmp_path: Path) -> None:
    source = tmp_path / "snapshot"
    source.mkdir()
    fake = tmp_path / "other/output"
    fake.mkdir(parents=True)
    with pytest.raises(ValueError, match="/root/RADIO-GS/output"):
        authority.run_locked(
            repo_root=source,
            lock_root=fake,
            output_root=fake / "run",
            command=[sys.executable, "-c", "raise SystemExit(0)"],
        )


def test_run_locked_accepts_only_the_controlled_root_symlink_from_readonly_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "frozen-snapshot"
    snapshot.mkdir()
    snapshot.chmod(0o555)
    lexical_parent = tmp_path / "main-repository"
    lexical_parent.mkdir()
    resolved_root = tmp_path / "mounted-output"
    resolved_root.mkdir()
    lexical_root = lexical_parent / "output"
    lexical_root.symlink_to(resolved_root, target_is_directory=True)
    output = lexical_root / "optimization/run"
    address = f"\0radio-gs-text-root-symlink-{os.getpid()}".encode("ascii")
    monkeypatch.setattr(authority, "CANONICAL_LOCK_ROOT", lexical_root)
    monkeypatch.setattr(authority, "GPU1_SINGLETON_ADDRESS", address)

    status = authority.run_locked(
        repo_root=snapshot,
        lock_root=lexical_root,
        output_root=output,
        command=[sys.executable, "-c", "raise SystemExit(0)"],
    )
    assert status == 0
    binding = authority.inspect_canonical_lock_root(lexical_root)
    assert binding["entry_type"] == "controlled_symlink"
    assert binding["resolved_path"] == str(resolved_root.resolve())
    assert (resolved_root / "optimization/run/locks/text_response_distill.run.lock").is_file()
    assert (resolved_root / ".physical_gpu1.lock").is_file()
    assert not (snapshot / ".git").exists()
    output_row = authority._seed_outputs(
        resolved_root / "optimization/run",
        "context_c1024_geometric",
    )[0]
    arguments = {
        "train_caches": "train*.pt",
        "validation_caches": "validation*.pt",
        "fit_text_bank": "fit.pt",
        "fit_text_bank_manifest": "fit.json",
        "calibration_manifests": {
            "0": "calibration0.json",
            "1": "calibration1.json",
            "2": "calibration2.json",
        },
        "run_manifest": str(output / "run_manifest.json"),
        "radio_checkpoint": "radio.pt",
        "output_root": str(output),
    }
    control = tmp_path / "surface-control-seed0.pt"
    control.write_bytes(b"surface-control")
    argv = authority._training_argv(
        repo_root=snapshot,
        arguments=arguments,
        output_row=output_row,
        surface_control={
            "seed": 0,
            "checkpoint": authority.file_record(control),
        },
        seed=0,
    )
    assert argv[argv.index("--output") + 1] == str(
        output / "readouts/context_c1024_geometric_text_response_seed0.pt"
    )


def test_bound_root_rejects_symlink_repoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lexical_parent = tmp_path / "main-repository"
    lexical_parent.mkdir()
    first = tmp_path / "first-output"
    second = tmp_path / "second-output"
    first.mkdir()
    second.mkdir()
    lexical_root = lexical_parent / "output"
    lexical_root.symlink_to(first, target_is_directory=True)
    monkeypatch.setattr(authority, "CANONICAL_LOCK_ROOT", lexical_root)
    frozen = authority.inspect_canonical_lock_root(lexical_root)

    lexical_root.unlink()
    lexical_root.symlink_to(second, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink/target identity changed"):
        authority.validate_canonical_lock_root_binding(
            frozen,
            lock_root=lexical_root,
        )


def test_controlled_root_still_rejects_every_child_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lexical_parent = tmp_path / "main-repository"
    lexical_parent.mkdir()
    resolved_root = tmp_path / "mounted-output"
    resolved_root.mkdir()
    lexical_root = lexical_parent / "output"
    lexical_root.symlink_to(resolved_root, target_is_directory=True)
    escaped = tmp_path / "escaped"
    escaped.mkdir()
    (resolved_root / "optimization").symlink_to(escaped, target_is_directory=True)
    monkeypatch.setattr(authority, "CANONICAL_LOCK_ROOT", lexical_root)
    monkeypatch.setattr(
        authority,
        "GPU1_SINGLETON_ADDRESS",
        f"\0radio-gs-text-child-symlink-{os.getpid()}".encode("ascii"),
    )

    with pytest.raises(ValueError, match="symlink/non-directory output component"):
        authority.run_locked(
            repo_root=tmp_path,
            lock_root=lexical_root,
            output_root=lexical_root / "optimization/run",
            command=[sys.executable, "-c", "raise SystemExit(0)"],
        )
    assert list(escaped.iterdir()) == []


def test_manifest_path_contract_rechecks_output_directory_identity_per_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lexical_parent = tmp_path / "main-repository"
    lexical_parent.mkdir()
    resolved_root = tmp_path / "mounted-output"
    resolved_root.mkdir()
    lexical_root = lexical_parent / "output"
    lexical_root.symlink_to(resolved_root, target_is_directory=True)
    output = lexical_root / "optimization/text-run"
    monkeypatch.setattr(authority, "CANONICAL_LOCK_ROOT", lexical_root)
    root_binding = authority.inspect_canonical_lock_root(lexical_root)
    output_binding = authority._secure_output_directory(
        output,
        root_binding=root_binding,
        create=True,
    )
    directory_bindings = authority._output_directory_bindings(
        output,
        root_binding=root_binding,
        create=True,
    )
    manifest = {
        "authority_contract": {
            "main_output_root": str(lexical_root),
            "main_output_root_binding": root_binding,
            "output_root": str(output),
            "output_root_binding": output_binding,
            "output_directory_bindings": directory_bindings,
            "root_path_protocol": authority.LOCK_ROOT_BINDING_VERSION,
            "global_gpu_lock": str(lexical_root / ".physical_gpu1.lock"),
            "output_run_lock": str(output / "locks/text_response_distill.run.lock"),
        }
    }
    authority.validate_authority_path_contract(manifest)

    readouts = resolved_root / "optimization/text-run/readouts"
    readouts.rename(readouts.with_name("readouts-old"))
    readouts.mkdir()
    with pytest.raises(ValueError, match="output directory identity changed"):
        authority.validate_authority_path_contract(manifest)


def test_real_deployment_lock_root_preflight_is_read_only() -> None:
    deployed_root = Path("/root/RADIO-GS/output")
    if REPO_ROOT != Path("/root/RADIO-GS") or not deployed_root.is_symlink():
        pytest.skip("real mounted RADIO-GS output root is not present")
    before = os.lstat(deployed_root)
    binding = authority.inspect_canonical_lock_root(deployed_root)
    after = os.lstat(deployed_root)
    assert binding["lexical_path"] == str(deployed_root)
    assert binding["entry_type"] == "controlled_symlink"
    assert Path(binding["resolved_path"]) == deployed_root.resolve(strict=True)
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)


def test_authority_and_promotion_independently_recompute_response_selection() -> None:
    history = _history()
    assert authority.recompute_response_epoch_selection(history) == (1, 0.95)
    assert promotion._recompute_response_primary_selection(history) == (1, 0.95)
    tampered = [dict(row) for row in history]
    tampered[0]["selection_score"] = 0.99
    with pytest.raises(ValueError, match="independently reproduced"):
        authority.recompute_response_epoch_selection(tampered)
    with pytest.raises(ValueError, match="independent recomputation"):
        promotion._recompute_response_primary_selection(tampered)


def test_runtime_closure_covers_authority_transitive_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The production runner always exports this before authority closure
    # capture.  Reproduce that contract even when pytest is invoked directly
    # with the environment's non-symlink CPython executable.
    monkeypatch.setenv("RADIO_GS_REPO_ROOT", str(REPO_ROOT))
    closure = authority.build_runtime_closure(REPO_ROOT)
    files = closure["repository_sources"]["files"]
    assert {
        "radio_gs/scripts/run_surface_region_text_response_distill.sh",
        "radio_gs/scripts/surface_text_response_distill_authority.py",
        "radio_gs/scripts/train_surface_region_text_response_distill.py",
        "radio_gs/scripts/finalize_gpu_guard_receipt.py",
        "radio_gs/scripts/finalize_surface_text_response_promotion.py",
        "radio_gs/scripts/surface_gpu1_lock_supervisor.py",
        "radio_gs/interfaces/surface_region_summary.py",
        "radio_gs/models/siglip_projection.py",
        "radio_gs/utils/immutable_artifacts.py",
    } <= set(files)
    assert closure["digest"] == authority.canonical_json_sha256(
        {
            "schema_version": closure["schema_version"],
            "repository_sources": closure["repository_sources"],
            "runtime_fingerprint": closure["runtime_fingerprint"],
        }
    )
    assert promotion._validate_distill_runtime_closure(
        closure,
        source_snapshot_root=REPO_ROOT,
    ) == closure["digest"]
    tampered = {
        **closure,
        "repository_sources": {
            **closure["repository_sources"],
            "files": {
                **closure["repository_sources"]["files"],
                "radio_gs/interfaces/surface_region_summary.py": "0" * 64,
            },
        },
    }
    with pytest.raises(ValueError, match="source closure changed"):
        promotion._validate_distill_runtime_closure(
            tampered,
            source_snapshot_root=REPO_ROOT,
        )


def test_training_inputs_do_not_use_generic_torch_load_fallback() -> None:
    trainer = (
        REPO_ROOT / "radio_gs/scripts/train_surface_region_text_response_distill.py"
    ).read_text(encoding="utf-8")
    base = (
        REPO_ROOT / "radio_gs/scripts/train_surface_region_summary_readout.py"
    ).read_text(encoding="utf-8")
    assert "torch.load(" not in trainer
    assert "torch.load(path, map_location=\"cpu\")" not in base
    assert "load_torch_mapping" in trainer and "load_torch_mapping" in base
    assert "load_surface_region_summary_readout_v2" in trainer


def test_promotion_authority_rejects_bound_kernel_xid(
    tmp_path: Path,
) -> None:
    identity = {
        "physical_index": 1,
        "uuid": "GPU-123456789",
        "pci_bus_id": "00000000:82:00.0",
    }
    evidence: dict[str, object] = {}
    for field in (
        "checkpoint",
        "report",
        "training_log",
        "audit_report",
        "guard_command",
        "guard_telemetry",
        "guard_receipt",
    ):
        path = tmp_path / field
        path.write_text(field, encoding="utf-8")
        evidence[field] = promotion._file_record(path)
    for phase, field in (("pre_seed0", "gpu_preflight"), ("post_seed0", "gpu_postflight")):
        path = tmp_path / f"{field}.json"
        observed_epoch = 1_754_000_000 if field == "gpu_preflight" else 1_754_000_010
        path.write_text(
            json.dumps(
                {
                    "status": "physical_gpu1_idle_and_pcie_responsive",
                    "phase": phase,
                    "gpu_identity": identity,
                    "compute_owners": [],
                    "observed_epoch": observed_epoch,
                }
            ),
            encoding="utf-8",
        )
        evidence[field] = {
            **promotion._file_record(path),
            "observed_epoch": observed_epoch,
        }
    journal = tmp_path / "kernel.log"
    journal.write_text(
        "surface_text_response_seed=0\tstart_epoch=1754000001\tend_epoch=1754000009\n"
        "NVRM: Xid 79, GPU has fallen off the bus\n",
        encoding="utf-8",
    )
    evidence["kernel_journal"] = {
        **promotion._file_record(journal),
        "seed": 0,
        "start_epoch": 1_754_000_001,
        "end_epoch": 1_754_000_009,
        "fault_count": 0,
    }
    with pytest.raises(ValueError, match="Xid/PCIe faults"):
        promotion._validate_authority_seed_evidence(
            evidence,
            seed=0,
            gpu_identity=identity,
        )
