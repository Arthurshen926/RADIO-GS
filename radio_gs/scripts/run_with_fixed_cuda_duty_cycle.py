#!/usr/bin/env python3
"""Run an unchanged CUDA command under a fixed host-side duty cycle.

This is the deliberately conservative last fallback for a producer which has
already tripped the normal 6 x 88 C thermal guard on both available GPUs.  The
production CLI has no knobs for the timing policy: it runs for at most 45
seconds, pauses the complete process group for 30 seconds, and samples NVML at
30-second boundaries.  SIGSTOP/SIGCONT change only host scheduling; the
command, environment, crop order, tensor math, and output paths are passed
through unchanged.

The worker is a new session/process-group leader.  Every group signal is
preceded by a /proc identity check, and cleanup always resumes a stopped group
before TERM/KILL.  NVIDIA's host-namespace compute PID is accepted only under
the same exclusive-singleton-after-clear-v1 authority used by the existing
thermal guard.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence


EXIT_THERMAL_GUARD = 86
EXIT_IDENTITY_GUARD = 87
SCHEDULE_ID = "fixed-duty-cycle-45run-30cool-nvml30-v1"
PID_NAMESPACE_MODE = "exclusive-singleton-after-clear-v1"


class DutyCycleError(RuntimeError):
    """Base class for fail-closed wrapper errors."""


class TelemetryError(DutyCycleError):
    """NVML/nvidia-smi could not provide trustworthy target-GPU state."""


class IdentityError(DutyCycleError):
    """The guarded process group or target-GPU ownership became ambiguous."""


class GuardInterrupted(BaseException):
    """Turn an external signal into orderly whole-group cleanup."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"fixed duty-cycle guard interrupted by signal {signum}")


@dataclass(frozen=True)
class DutyCyclePolicy:
    run_seconds: float = 45.0
    cool_seconds: float = 30.0
    nvml_poll_seconds: float = 30.0
    nvml_query_timeout_seconds: float = 5.0
    start_max_temperature_c: float = 65.0
    hard_abort_temperature_c: float = 88.0
    hard_abort_consecutive_samples: int = 6
    maximum_power_limit_w: float = 300.5
    terminate_grace_seconds: float = 20.0

    def validate(self) -> None:
        positive = (
            self.run_seconds,
            self.cool_seconds,
            self.nvml_poll_seconds,
            self.nvml_query_timeout_seconds,
            self.start_max_temperature_c,
            self.hard_abort_temperature_c,
            self.maximum_power_limit_w,
            self.terminate_grace_seconds,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise DutyCycleError("duty-cycle policy values must be finite and positive")
        if self.nvml_query_timeout_seconds >= min(
            self.run_seconds,
            self.cool_seconds,
            self.nvml_poll_seconds,
        ):
            raise DutyCycleError("NVML timeout must be below every schedule interval")
        if self.start_max_temperature_c >= self.hard_abort_temperature_c:
            raise DutyCycleError("start temperature must be below hard-abort temperature")
        if (
            type(self.hard_abort_consecutive_samples) is not int
            or self.hard_abort_consecutive_samples < 1
        ):
            raise DutyCycleError("hard-abort sample count must be a positive integer")


FROZEN_POLICY = DutyCyclePolicy()


@dataclass(frozen=True)
class GpuIdentity:
    bus_id: str
    uuid: str


@dataclass(frozen=True)
class Telemetry:
    identity: GpuIdentity
    temperature_c: float
    power_w: float
    power_limit_w: float
    utilization_percent: int
    memory_mib: int
    pstate: str


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    # Scheduler state changes between R/S/T during normal run/pause/resume and
    # therefore is not part of the stable anti-PID-reuse identity.
    state: str = field(compare=False)
    process_group: int
    session: int
    start_time_ticks: int


def _normalise_bus_id(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("00000000:"):
        return "0000:" + stripped[len("00000000:") :]
    return stripped


def _finite_float(value: str, label: str) -> float:
    try:
        parsed = float(value.strip())
    except ValueError as error:
        raise TelemetryError(f"invalid {label}: {value!r}") from error
    if not math.isfinite(parsed):
        raise TelemetryError(f"non-finite {label}")
    return parsed


def _nonnegative_integer(value: str, label: str) -> int:
    stripped = value.strip()
    if not stripped.isdigit():
        raise TelemetryError(f"invalid {label}: {value!r}")
    return int(stripped)


def parse_telemetry(raw: str) -> Telemetry:
    rows = [row for row in raw.splitlines() if row.strip()]
    if len(rows) != 1:
        raise TelemetryError(f"expected one telemetry row, found {len(rows)}")
    fields = rows[0].split(",")
    if len(fields) != 8:
        raise TelemetryError(f"expected eight telemetry fields, found {len(fields)}")
    identity = GpuIdentity(
        bus_id=_normalise_bus_id(fields[0]),
        uuid=fields[1].strip(),
    )
    telemetry = Telemetry(
        identity=identity,
        temperature_c=_finite_float(fields[2], "temperature"),
        power_w=_finite_float(fields[3], "power draw"),
        power_limit_w=_finite_float(fields[4], "power limit"),
        utilization_percent=_nonnegative_integer(fields[5], "utilization"),
        memory_mib=_nonnegative_integer(fields[6], "memory"),
        pstate=fields[7].strip(),
    )
    if not identity.bus_id or not identity.uuid.startswith("GPU-"):
        raise TelemetryError("invalid physical-GPU identity")
    if not 0 < telemetry.temperature_c <= 150:
        raise TelemetryError("temperature is outside 0..150 C")
    if telemetry.power_w < 0 or telemetry.power_limit_w <= 0:
        raise TelemetryError("power telemetry is outside its physical range")
    if not 0 <= telemetry.utilization_percent <= 100:
        raise TelemetryError("utilization is outside 0..100 percent")
    if not telemetry.pstate:
        raise TelemetryError("empty pstate")
    return telemetry


class NvmlClient:
    """Small fail-closed nvidia-smi facade (nvidia-smi uses NVML)."""

    def __init__(
        self,
        physical_gpu: int,
        *,
        nvidia_smi: str,
        timeout_seconds: float,
    ) -> None:
        if type(physical_gpu) is not int or physical_gpu < 0:
            raise DutyCycleError("physical GPU index must be a non-negative integer")
        self.physical_gpu = physical_gpu
        self.nvidia_smi = nvidia_smi
        self.timeout_seconds = timeout_seconds

    def _run(self, arguments: Sequence[str], label: str) -> str:
        command = [self.nvidia_smi, "-i", str(self.physical_gpu), *arguments]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise TelemetryError(f"{label} query failed: {error}") from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise TelemetryError(
                f"{label} query exited {completed.returncode}: {detail}"
            )
        return completed.stdout

    def sample(self, expected_identity: GpuIdentity | None = None) -> Telemetry:
        raw = self._run(
            [
                "--query-gpu=pci.bus_id,uuid,temperature.gpu,power.draw,"
                "power.limit,utilization.gpu,memory.used,pstate",
                "--format=csv,noheader,nounits",
            ],
            "telemetry",
        )
        telemetry = parse_telemetry(raw)
        if expected_identity is not None and telemetry.identity != expected_identity:
            raise TelemetryError(
                "physical-GPU identity changed during guarded execution"
            )
        return telemetry

    def compute_owner_pids(self, gpu_uuid: str) -> list[int]:
        raw = self._run(
            [
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            "compute-owner",
        )
        owners: list[int] = []
        for row in raw.splitlines():
            if not row.strip():
                continue
            fields = row.split(",")
            if len(fields) != 2:
                raise TelemetryError("invalid compute-owner row")
            if fields[0].strip() != gpu_uuid:
                continue
            owners.append(_nonnegative_integer(fields[1], "compute-owner PID"))
        return sorted(set(owners))


def _read_process_identity(pid: int) -> ProcessIdentity:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError) as error:
        raise IdentityError(f"process {pid} no longer exists") from error
    close = raw.rfind(")")
    if close < 0:
        raise IdentityError(f"cannot parse /proc/{pid}/stat")
    fields = raw[close + 1 :].split()
    if len(fields) <= 19:
        raise IdentityError(f"truncated /proc/{pid}/stat")
    try:
        return ProcessIdentity(
            pid=pid,
            state=fields[0],
            process_group=int(fields[2]),
            session=int(fields[3]),
            start_time_ticks=int(fields[19]),
        )
    except ValueError as error:
        raise IdentityError(f"invalid numeric identity for process {pid}") from error


def _matching_group_members(process_group: int, session: int) -> list[ProcessIdentity]:
    members: list[ProcessIdentity] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            identity = _read_process_identity(int(entry.name))
        except IdentityError:
            continue
        if (
            identity.state != "Z"
            and identity.process_group == process_group
            and identity.session == session
        ):
            members.append(identity)
    return sorted(members, key=lambda item: item.pid)


def _namespace_pids(local_pid: int) -> set[int]:
    values = {local_pid}
    try:
        lines = Path(f"/proc/{local_pid}/status").read_text(
            encoding="utf-8"
        ).splitlines()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return values
    for line in lines:
        if not line.startswith("NSpid:"):
            continue
        for field in line.split()[1:]:
            if field.isdigit():
                values.add(int(field))
        break
    return values


class GuardedProcessGroup:
    """An identity-checked child session whose signals never target the guard."""

    def __init__(self, process: subprocess.Popen[Any], grace_seconds: float) -> None:
        self.process = process
        self.grace_seconds = grace_seconds
        self.identity = _read_process_identity(process.pid)
        if self.identity.process_group != process.pid:
            raise IdentityError("worker is not its own process-group leader")
        if self.identity.session != process.pid:
            raise IdentityError("worker is not its own session leader")
        if self.identity.process_group == os.getpgrp():
            raise IdentityError("worker and guard share a process group")
        self.paused = False

    @classmethod
    def launch(
        cls,
        command: Sequence[str],
        *,
        grace_seconds: float,
    ) -> "GuardedProcessGroup":
        process = subprocess.Popen(list(command), start_new_session=True)
        try:
            return cls(process, grace_seconds)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise

    def running(self) -> bool:
        return self.process.poll() is None

    def members(self) -> list[ProcessIdentity]:
        return _matching_group_members(
            self.identity.process_group,
            self.identity.session,
        )

    def _verify_leader(self) -> bool:
        if not self.running():
            return False
        current = _read_process_identity(self.identity.pid)
        if current != self.identity:
            raise IdentityError("worker identity changed; refusing reused PGID")
        if current.process_group == os.getpgrp():
            raise IdentityError("refusing to signal the guard process group")
        return True

    def signal(self, signum: int) -> bool:
        if not self._verify_leader():
            return False
        try:
            os.killpg(self.identity.process_group, signum)
        except ProcessLookupError:
            if not self.running():
                return False
            raise IdentityError("worker process group disappeared during signal")
        except OSError as error:
            raise IdentityError(f"cannot signal worker process group: {error}") from error
        return True

    def set_paused(self, paused: bool) -> bool:
        if paused == self.paused:
            return False
        if not self.signal(signal.SIGSTOP if paused else signal.SIGCONT):
            return False
        self.paused = paused
        return True

    @staticmethod
    def _signal_exact_members(
        members: Iterable[ProcessIdentity],
        signum: int,
    ) -> None:
        for expected in members:
            try:
                current = _read_process_identity(expected.pid)
            except IdentityError:
                continue
            if current != expected:
                continue
            try:
                os.kill(expected.pid, signum)
            except ProcessLookupError:
                pass

    def terminate(self) -> None:
        """Resume first, terminate all members, and reap the direct worker."""

        if self.running():
            if self.paused:
                self.signal(signal.SIGCONT)
                self.paused = False
            self.signal(signal.SIGTERM)
        else:
            # The leader cannot be used to authenticate a cached PGID after it
            # exits.  Signal individually identity-checked descendants instead.
            members = self.members()
            self._signal_exact_members(members, signal.SIGCONT)
            self._signal_exact_members(members, signal.SIGTERM)

        deadline = time.monotonic() + self.grace_seconds
        while time.monotonic() < deadline:
            self.process.poll()
            if not self.members():
                break
            time.sleep(min(0.1, self.grace_seconds / 10.0))
        members = self.members()
        if members:
            if self.running():
                self.signal(signal.SIGKILL)
            else:
                self._signal_exact_members(members, signal.SIGKILL)
        try:
            self.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            # The exact Popen handle is safe even if PGID validation failed.
            self.process.kill()
            self.process.wait()
        self.paused = False


class OwnerAudit:
    """Bind NVML host PIDs to the local worker group without PID guessing."""

    def __init__(self) -> None:
        self.bound_invisible_host_pid: int | None = None

    @staticmethod
    def require_prelaunch_clear(owner_pids: Sequence[int]) -> None:
        if owner_pids:
            raise IdentityError(
                f"target GPU has pre-existing compute owner(s): {list(owner_pids)}"
            )

    def audit(
        self,
        owner_pids: Sequence[int],
        group: GuardedProcessGroup,
    ) -> dict[str, Any]:
        namespace_values: set[int] = set()
        for member in group.members():
            namespace_values.update(_namespace_pids(member.pid))
        known = [pid for pid in owner_pids if pid in namespace_values]
        unknown = [pid for pid in owner_pids if pid not in namespace_values]
        event = "runtime_owner_audit"
        if unknown:
            only = unknown[0] if len(unknown) == 1 else None
            singleton_invisible = (
                len(owner_pids) == 1
                and only is not None
                and not Path(f"/proc/{only}").exists()
                and self.bound_invisible_host_pid in (None, only)
            )
            if not singleton_invisible:
                raise IdentityError(
                    "target GPU owner(s) are outside the guarded session: "
                    f"{unknown}"
                )
            self.bound_invisible_host_pid = only
            known.append(only)
            unknown.clear()
            event = "runtime_owner_audit_host_pid_singleton"
        return {
            "owner_audit_event": event,
            "owner_pids": list(owner_pids),
            "worker_owner_pids": sorted(known),
            "foreign_owner_pids": unknown,
            "pid_namespace_mode": PID_NAMESPACE_MODE,
        }


class StatusLog:
    def __init__(self, path: Path, physical_gpu: int) -> None:
        self.path = path.resolve()
        self.physical_gpu = physical_gpu
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "physical_gpu": self.physical_gpu,
            "schedule_id": SCHEDULE_ID,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()


class ExclusiveGpuLock:
    """Serialize this fallback without taking a producer-owned legacy lock."""

    def __init__(self, physical_gpu: int) -> None:
        self.path = Path(f"/tmp/radio-gs-fixed-duty-cycle-gpu{physical_gpu}.lock")
        self._handle: Any | None = None

    def __enter__(self) -> "ExclusiveGpuLock":
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._handle.close()
            self._handle = None
            raise IdentityError(f"fallback lock is already held: {self.path}") from error
        return self

    def __exit__(self, *_args: Any) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


def _telemetry_fields(telemetry: Telemetry) -> dict[str, Any]:
    payload = asdict(telemetry)
    identity = payload.pop("identity")
    return {**identity, **payload}


def _sleep_to_boundary(started: float, interval: float) -> None:
    remaining = started + interval - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def _prelaunch_cooldown(
    *,
    client: NvmlClient,
    status: StatusLog,
    policy: DutyCyclePolicy,
    owner_audit: OwnerAudit,
) -> GpuIdentity:
    consecutive_hot = 0
    expected_identity: GpuIdentity | None = None
    while True:
        sample_started = time.monotonic()
        telemetry = client.sample(expected_identity)
        if expected_identity is None:
            expected_identity = telemetry.identity
        if telemetry.power_limit_w > policy.maximum_power_limit_w:
            raise TelemetryError(
                f"power limit {telemetry.power_limit_w} W exceeds "
                f"{policy.maximum_power_limit_w} W"
            )
        owners = client.compute_owner_pids(expected_identity.uuid)
        owner_audit.require_prelaunch_clear(owners)
        if telemetry.temperature_c >= policy.hard_abort_temperature_c:
            consecutive_hot += 1
        else:
            consecutive_hot = 0
        ready = telemetry.temperature_c <= policy.start_max_temperature_c
        status.write(
            "prelaunch_ready" if ready else "prelaunch_cooldown",
            consecutive_overheat_samples=consecutive_hot,
            owner_pids=owners,
            **_telemetry_fields(telemetry),
        )
        if consecutive_hot >= policy.hard_abort_consecutive_samples:
            raise TelemetryError(
                "prelaunch temperature remained at or above the hard-abort "
                f"threshold for {consecutive_hot} samples"
            )
        if ready:
            return expected_identity
        _sleep_to_boundary(sample_started, policy.nvml_poll_seconds)


def run_guarded(
    command: Sequence[str],
    *,
    physical_gpu: int,
    status_log: Path,
    nvidia_smi: str = "nvidia-smi",
    policy: DutyCyclePolicy = FROZEN_POLICY,
) -> int:
    """Run one command; ``policy`` injection exists for CPU tests, not the CLI."""

    policy.validate()
    if not command:
        raise DutyCycleError("guarded command is empty")
    status = StatusLog(status_log, physical_gpu)
    client = NvmlClient(
        physical_gpu,
        nvidia_smi=nvidia_smi,
        timeout_seconds=policy.nvml_query_timeout_seconds,
    )
    owner_audit = OwnerAudit()
    group: GuardedProcessGroup | None = None
    result = EXIT_IDENTITY_GUARD
    cleanup_failed = False
    previous_handlers: dict[int, Any] = {}

    def interrupt(signum: int, _frame: Any) -> None:
        raise GuardInterrupted(signum)

    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)

    with ExclusiveGpuLock(physical_gpu) as lock:
        status.write(
            "guard_start",
            command=list(command),
            policy=asdict(policy),
            production_policy_fixed=policy == FROZEN_POLICY,
            lock_path=str(lock.path),
            pid_namespace_mode=PID_NAMESPACE_MODE,
            guard_pid=os.getpid(),
            guard_process_group=os.getpgrp(),
        )
        try:
            expected_identity = _prelaunch_cooldown(
                client=client,
                status=status,
                policy=policy,
                owner_audit=owner_audit,
            )
            group = GuardedProcessGroup.launch(
                command,
                grace_seconds=policy.terminate_grace_seconds,
            )
            for signum in handled_signals:
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, interrupt)
            status.write(
                "worker_started",
                child_pid=group.identity.pid,
                child_process_group=group.identity.process_group,
                child_session=group.identity.session,
                child_start_time_ticks=group.identity.start_time_ticks,
                **asdict(expected_identity),
            )

            phase = "run"
            thermal_hold = False
            consecutive_hot = 0
            now = time.monotonic()
            phase_deadline = now + policy.run_seconds
            telemetry_deadline = now + policy.nvml_poll_seconds

            while group.running():
                now = time.monotonic()
                if now >= phase_deadline:
                    phase = "cool" if phase == "run" else "run"
                    phase_deadline += (
                        policy.cool_seconds if phase == "cool" else policy.run_seconds
                    )
                    should_pause = phase == "cool" or thermal_hold
                    group.set_paused(should_pause)
                    status.write(
                        "duty_pause" if should_pause else "duty_resume",
                        phase=phase,
                        thermal_hold=thermal_hold,
                        child_pid=group.identity.pid,
                    )
                    continue

                if now >= telemetry_deadline:
                    telemetry = client.sample(expected_identity)
                    if telemetry.power_limit_w > policy.maximum_power_limit_w:
                        raise TelemetryError(
                            f"power limit {telemetry.power_limit_w} W exceeds "
                            f"{policy.maximum_power_limit_w} W"
                        )
                    owners = client.compute_owner_pids(expected_identity.uuid)
                    audit_fields = owner_audit.audit(owners, group)
                    if telemetry.temperature_c >= policy.hard_abort_temperature_c:
                        consecutive_hot += 1
                        thermal_hold = True
                    else:
                        consecutive_hot = 0
                        thermal_hold = False
                    group.set_paused(phase == "cool" or thermal_hold)
                    event = "telemetry_sample"
                    if thermal_hold:
                        event = (
                            "thermal_hard_abort_"
                            if consecutive_hot >= policy.hard_abort_consecutive_samples
                            else "thermal_hold_"
                        ) + (
                            f"{consecutive_hot}_of_"
                            f"{policy.hard_abort_consecutive_samples}"
                        )
                    status.write(
                        event,
                        phase=phase,
                        paused=group.paused,
                        consecutive_overheat_samples=consecutive_hot,
                        **audit_fields,
                        **_telemetry_fields(telemetry),
                    )
                    if consecutive_hot >= policy.hard_abort_consecutive_samples:
                        raise TelemetryError(
                            f"GPU{physical_gpu} reached "
                            f"{policy.hard_abort_temperature_c} C for "
                            f"{consecutive_hot} consecutive samples"
                        )
                    telemetry_deadline += policy.nvml_poll_seconds
                    continue

                sleep_seconds = min(
                    0.25,
                    max(0.001, phase_deadline - now),
                    max(0.001, telemetry_deadline - now),
                )
                time.sleep(sleep_seconds)

            returncode = group.process.wait()
            leftovers = group.members()
            if leftovers:
                raise IdentityError(
                    "worker leader exited while session members remained: "
                    f"{[member.pid for member in leftovers]}"
                )
            status.write("worker_complete", returncode=returncode)
            result = returncode
        except GuardInterrupted as caught:
            status.write("guard_interrupted", signal=caught.signum)
            result = 128 + caught.signum
        except TelemetryError as caught:
            status.write("fail_closed_abort", error=str(caught))
            result = EXIT_THERMAL_GUARD
        except IdentityError as caught:
            status.write("identity_guard_abort", error=str(caught))
            result = EXIT_IDENTITY_GUARD
        except BaseException as caught:
            try:
                status.write(
                    "guard_internal_abort",
                    error=f"{type(caught).__name__}: {caught}",
                )
            except Exception:
                pass
            result = EXIT_IDENTITY_GUARD
        finally:
            if group is not None and (group.running() or group.members()):
                try:
                    if group.paused:
                        status.write("worker_cleanup_resume_before_terminate")
                    group.terminate()
                    status.write("worker_cleanup_complete")
                except BaseException as caught:
                    cleanup_failed = True
                    try:
                        status.write(
                            "worker_cleanup_failed",
                            error=f"{type(caught).__name__}: {caught}",
                        )
                    except Exception:
                        pass
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)

    return EXIT_IDENTITY_GUARD if cleanup_failed else result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CUDA command with the immutable 45 s run / 30 s cool / "
            "30 s NVML fixed-duty policy."
        )
    )
    parser.add_argument("--gpu", type=int, required=True, help="physical GPU index")
    parser.add_argument("--status-log", type=Path, required=True)
    parser.add_argument(
        "--nvidia-smi",
        default="nvidia-smi",
        help="override only for a controlled CPU test double",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_guarded(
            args.command,
            physical_gpu=args.gpu,
            status_log=args.status_log,
            nvidia_smi=args.nvidia_smi,
            policy=FROZEN_POLICY,
        )
    except DutyCycleError as error:
        print(f"fixed CUDA duty-cycle guard error: {error}", file=sys.stderr)
        return EXIT_IDENTITY_GUARD


if __name__ == "__main__":
    raise SystemExit(main())
