import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from dataclasses import replace

import pytest

import reproductions.ludvig.run_gpu0_thermal_guard as guard


def test_default_policy_is_the_frozen_gpu0_contract() -> None:
    policy = guard.DEFAULT_POLICY

    assert guard.PHYSICAL_GPU_INDEX == 0
    assert guard.GPU_LOCK_PATH == Path("/tmp/radio-gs-gpu0.lock")
    assert policy.poll_seconds == 3.0
    assert policy.query_timeout_seconds == 2.0
    assert policy.warning_temperature_c == 78.0
    assert policy.pause_temperature_c == 81.0
    assert policy.resume_temperature_c == 70.0
    assert policy.resume_stable_samples == 2
    assert policy.query_failures_to_pause == 2
    assert policy.query_failures_to_terminate == 12


def test_thermal_state_pauses_at_81_and_requires_two_cool_samples() -> None:
    state = guard.ThermalState()

    assert state.observe_temperature(77.9).event == "sample"
    assert state.observe_temperature(78.0).event == "thermal_warning"
    assert state.observe_temperature(81.0) == guard.Decision(
        "pause",
        "thermal_pause",
    )
    assert state.observe_temperature(70.0).action == "none"
    assert state.paused is True
    assert state.observe_temperature(70.0) == guard.Decision(
        "resume",
        "thermal_resume",
    )
    assert state.paused is False


def test_cooldown_counter_resets_above_70() -> None:
    state = guard.ThermalState()
    state.observe_temperature(82.0)
    state.observe_temperature(69.0)

    assert state.consecutive_resume_samples == 1
    assert state.observe_temperature(70.1).event == "thermal_cooldown"
    assert state.consecutive_resume_samples == 0
    assert state.observe_temperature(70.0).action == "none"
    assert state.observe_temperature(70.0).action == "resume"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"poll_seconds": 0.0}, "finite and positive"),
        ({"query_timeout_seconds": 0.0}, "finite and positive"),
        (
            {"poll_seconds": 2.0, "query_timeout_seconds": 2.0},
            "less than poll_seconds",
        ),
        ({"warning_temperature_c": 81.0}, "resume < warning < pause"),
        ({"resume_temperature_c": 78.0}, "resume < warning < pause"),
        ({"resume_stable_samples": 0}, "positive integer"),
    ],
)
def test_policy_rejects_unsafe_parameter_combinations(overrides, message) -> None:
    policy = replace(guard.DEFAULT_POLICY, **overrides)

    with pytest.raises(guard.GuardError, match=message):
        policy.validate()


def test_telemetry_failures_pause_at_two_and_terminate_at_twelve() -> None:
    state = guard.ThermalState()

    assert state.observe_failure().action == "none"
    assert state.observe_failure() == guard.Decision(
        "pause",
        "telemetry_failure_pause",
    )
    for _ in range(9):
        assert state.observe_failure().action == "none"
    assert state.consecutive_query_failures == 11
    assert state.observe_failure() == guard.Decision(
        "terminate",
        "telemetry_failure_terminate",
    )


def test_success_resets_failure_count_but_not_fail_closed_pause() -> None:
    state = guard.ThermalState()
    state.observe_failure()
    state.observe_failure()

    assert state.paused is True
    assert state.observe_temperature(63.0).event == "thermal_cooldown"
    assert state.consecutive_query_failures == 0
    assert state.paused is True


def test_parse_telemetry_requires_exactly_one_finite_six_field_row() -> None:
    telemetry = guard.parse_telemetry("63, 221.5, 300.0, 98, 12345, P2\n")

    assert telemetry == guard.Telemetry(63.0, 221.5, 300.0, 98, 12345, "P2")
    with pytest.raises(guard.GuardError, match="one GPU0 telemetry row"):
        guard.parse_telemetry("63, 1, 300, 1, 2, P8\n64, 1, 300, 1, 2, P8")
    with pytest.raises(guard.GuardError, match="non-finite temperature"):
        guard.parse_telemetry("nan, 1, 300, 1, 2, P8")
    with pytest.raises(guard.GuardError, match="invalid utilization"):
        guard.parse_telemetry("63, 1, 300, -1, 2, P8")
    with pytest.raises(guard.GuardError, match="outside 0..150"):
        guard.parse_telemetry("0, 1, 300, 1, 2, P8")
    with pytest.raises(guard.GuardError, match="outside 0..100"):
        guard.parse_telemetry("63, 1, 300, 101, 2, P8")


def test_sample_command_pins_physical_gpu0_without_calling_real_sensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="63, 200, 300, 99, 20000, P2\n",
            stderr="",
        )

    monkeypatch.setattr(guard.subprocess, "run", fake_run)

    telemetry = guard.sample_gpu0("fake-nvidia-smi")

    assert captured["command"][0:3] == ["fake-nvidia-smi", "-i", "0"]
    assert captured["kwargs"]["timeout"] == 2.0
    assert telemetry.temperature_c == 63.0


def test_status_log_records_gpu_lock_and_physical_index(tmp_path: Path) -> None:
    path = tmp_path / "guard" / "status.jsonl"
    guard.StatusLog(path).write("sample", temperature_c=63.0)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["event"] == "sample"
    assert payload["physical_gpu"] == 0
    assert payload["gpu_lock"] == "/tmp/radio-gs-gpu0.lock"
    assert payload["temperature_c"] == 63.0


def test_poll_boundary_counts_query_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = []
    monkeypatch.setattr(guard.time, "monotonic", lambda: 12.0)
    monkeypatch.setattr(guard.time, "sleep", sleeps.append)

    guard._sleep_to_poll_boundary(cycle_started=10.0, poll_seconds=5.0)
    guard._sleep_to_poll_boundary(cycle_started=5.0, poll_seconds=5.0)

    assert sleeps == [3.0]


def test_child_is_a_distinct_session_and_signals_target_only_its_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        group = guard.GuardedProcessGroup(
            process,
            terminate_grace_seconds=0.1,
        )
        signals = []
        monkeypatch.setattr(
            guard.os,
            "killpg",
            lambda process_group, signum: signals.append((process_group, signum)),
        )

        assert group.identity.pid == process.pid
        assert group.identity.process_group == process.pid
        assert group.identity.session == process.pid
        assert group.identity.process_group != os.getpgrp()
        assert group.signal(signal.SIGSTOP) is True
        assert signals == [(process.pid, signal.SIGSTOP)]
    finally:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def test_identity_change_refuses_signal_before_killpg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 12345

        @staticmethod
        def poll():
            return None

    group = object.__new__(guard.GuardedProcessGroup)
    group.process = FakeProcess()
    group.terminate_grace_seconds = 0.1
    group.identity = guard.ProcessIdentity(12345, 12345, 12345, 100)
    monkeypatch.setattr(
        guard,
        "_read_process_identity",
        lambda _pid: guard.ProcessIdentity(12345, 12345, 12345, 101),
    )
    killpg_calls = []
    monkeypatch.setattr(
        guard.os,
        "killpg",
        lambda *args: killpg_calls.append(args),
    )

    with pytest.raises(guard.GuardError, match="reused PID"):
        group.signal(signal.SIGSTOP)
    assert killpg_calls == []


def test_unvalidated_spawn_is_reaped_without_group_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 24680

        def __init__(self):
            self.terminated = False
            self.killed = False
            self.waited = False

        @staticmethod
        def poll():
            return None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            self.waited = True
            return 0

    process = FakeProcess()
    monkeypatch.setattr(guard, "_wait_before_launch", lambda **_kwargs: True)
    monkeypatch.setattr(
        guard.subprocess,
        "Popen",
        lambda command, start_new_session: process,
    )

    def reject_identity(*_args, **_kwargs):
        raise guard.GuardError("identity rejected")

    monkeypatch.setattr(guard, "GuardedProcessGroup", reject_identity)
    with pytest.raises(guard.GuardError, match="identity rejected"):
        guard.run_guarded(
            ["fake-job"],
            status_log=tmp_path / "status.jsonl",
        )

    assert process.terminated is True
    assert process.waited is True
    assert process.killed is False


def test_status_log_failure_after_spawn_terminates_validated_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGroup:
        identity = guard.ProcessIdentity(24681, 24681, 24681, 100)

        def __init__(self):
            self.terminated = False

        def terminate(self, *, resume_first):
            assert resume_first is False
            self.terminated = True

    group = FakeGroup()
    original_write = guard.StatusLog.write

    def flaky_write(self, event, **fields):
        if event == "job_started":
            raise OSError("simulated full status volume")
        return original_write(self, event, **fields)

    monkeypatch.setattr(guard, "_wait_before_launch", lambda **_kwargs: True)
    monkeypatch.setattr(
        guard.subprocess,
        "Popen",
        lambda command, start_new_session: object(),
    )
    monkeypatch.setattr(guard, "GuardedProcessGroup", lambda *_args, **_kwargs: group)
    monkeypatch.setattr(guard.StatusLog, "write", flaky_write)

    with pytest.raises(OSError, match="simulated full status volume"):
        guard.run_guarded(
            ["fake-job"],
            status_log=tmp_path / "status.jsonl",
        )

    assert group.terminated is True


def test_cli_requires_explicit_exact_internal_lock_acknowledgement() -> None:
    args = guard.parse_args(
        [
            "--status-log",
            "/tmp/thermal.jsonl",
            "--job-owns-gpu-lock",
            "/tmp/radio-gs-gpu0.lock",
            "--",
            "python",
            "job.py",
        ]
    )

    assert args.command == ["python", "job.py"]
    assert args.policy == guard.DEFAULT_POLICY
    with pytest.raises(SystemExit):
        guard.parse_args(
            [
                "--status-log",
                "/tmp/thermal.jsonl",
                "--job-owns-gpu-lock",
                "/tmp/wrong.lock",
                "--",
                "python",
                "job.py",
            ]
        )


def test_cli_builds_and_validates_custom_policy() -> None:
    args = guard.parse_args(
        [
            "--status-log",
            "/tmp/thermal.jsonl",
            "--job-owns-gpu-lock",
            "/tmp/radio-gs-gpu0.lock",
            "--sample-interval",
            "4",
            "--query-timeout",
            "1.5",
            "--warn",
            "79",
            "--pause",
            "82",
            "--resume",
            "67",
            "--stable-samples",
            "4",
            "--",
            "python",
            "job.py",
        ]
    )

    assert args.policy == guard.GuardPolicy(
        poll_seconds=4.0,
        query_timeout_seconds=1.5,
        warning_temperature_c=79.0,
        pause_temperature_c=82.0,
        resume_temperature_c=67.0,
        resume_stable_samples=4,
    )

    with pytest.raises(SystemExit):
        guard.parse_args(
            [
                "--status-log",
                "/tmp/thermal.jsonl",
                "--job-owns-gpu-lock",
                "/tmp/radio-gs-gpu0.lock",
                "--sample-interval",
                "2",
                "--query-timeout",
                "2",
                "--",
                "python",
                "job.py",
            ]
        )
