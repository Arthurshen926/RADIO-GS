#!/usr/bin/env python3
"""Run one internally-locked LUDVIG job under a physical-GPU0 watchdog.

The guarded LUDVIG launchers already serialize their GPU sections with
``/tmp/radio-gs-gpu0.lock``.  This watchdog deliberately does *not* acquire
that lock: taking the same lock around a launcher which later takes it itself
would deadlock.  Instead, the required ``--job-owns-gpu-lock`` argument makes
that ownership boundary explicit and records it in the status log.

The child is started in a new session, so the child PID is also its process
group and session ID.  Thermal signals target that entire task group while the
watchdog remains in a different group.  Before every signal, the watchdog
checks the group leader's /proc start time, process group and session.  This
prevents a stale PID from being used after PID reuse and prevents the watchdog
from signalling its own process group.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Sequence


PHYSICAL_GPU_INDEX = 0
GPU_LOCK_PATH = Path("/tmp/radio-gs-gpu0.lock")
EXIT_THERMAL_GUARD = 86
EXIT_IDENTITY_GUARD = 87


class GuardError(RuntimeError):
    """Raised when the watchdog cannot safely identify or control its job."""


class GuardInterrupted(BaseException):
    """Raised by a signal handler so the guarded process group is cleaned up."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"thermal guard interrupted by signal {signum}")


@dataclass(frozen=True)
class GuardPolicy:
    poll_seconds: float = 3.0
    query_timeout_seconds: float = 2.0
    warning_temperature_c: float = 78.0
    pause_temperature_c: float = 81.0
    resume_temperature_c: float = 70.0
    resume_stable_samples: int = 2
    query_failures_to_pause: int = 2
    query_failures_to_terminate: int = 12
    terminate_grace_seconds: float = 20.0

    def validate(self) -> None:
        numeric = (
            self.poll_seconds,
            self.query_timeout_seconds,
            self.warning_temperature_c,
            self.pause_temperature_c,
            self.resume_temperature_c,
            self.terminate_grace_seconds,
        )
        if any(not math.isfinite(value) or value <= 0 for value in numeric):
            raise GuardError("thermal policy numeric values must be finite and positive")
        if self.query_timeout_seconds >= self.poll_seconds:
            raise GuardError(
                "query_timeout_seconds must be less than poll_seconds"
            )
        if not (
            self.resume_temperature_c
            < self.warning_temperature_c
            < self.pause_temperature_c
        ):
            raise GuardError("thermal thresholds must satisfy resume < warning < pause")
        if type(self.resume_stable_samples) is not int or self.resume_stable_samples < 1:
            raise GuardError("resume_stable_samples must be a positive integer")
        if (
            type(self.query_failures_to_pause) is not int
            or type(self.query_failures_to_terminate) is not int
            or self.query_failures_to_pause < 1
            or self.query_failures_to_terminate <= self.query_failures_to_pause
        ):
            raise GuardError(
                "query failure thresholds must be positive with pause < terminate"
            )


DEFAULT_POLICY = GuardPolicy()


@dataclass(frozen=True)
class Telemetry:
    temperature_c: float
    power_w: float
    power_limit_w: float
    utilization_percent: int
    memory_mib: int
    pstate: str


@dataclass(frozen=True)
class Decision:
    action: str
    event: str


class ThermalState:
    """Pure state machine for temperature and telemetry-failure hysteresis."""

    def __init__(self, policy: GuardPolicy = DEFAULT_POLICY) -> None:
        policy.validate()
        self.policy = policy
        self.paused = False
        self.consecutive_query_failures = 0
        self.consecutive_resume_samples = 0

    def observe_failure(self) -> Decision:
        self.consecutive_query_failures += 1
        self.consecutive_resume_samples = 0
        if (
            self.consecutive_query_failures
            >= self.policy.query_failures_to_terminate
        ):
            return Decision("terminate", "telemetry_failure_terminate")
        if (
            self.consecutive_query_failures >= self.policy.query_failures_to_pause
            and not self.paused
        ):
            self.paused = True
            return Decision("pause", "telemetry_failure_pause")
        return Decision(
            "none",
            "telemetry_failure_paused" if self.paused else "telemetry_failure",
        )

    def observe_temperature(self, temperature_c: float) -> Decision:
        if not math.isfinite(temperature_c):
            return self.observe_failure()
        self.consecutive_query_failures = 0

        if temperature_c >= self.policy.pause_temperature_c:
            self.consecutive_resume_samples = 0
            if not self.paused:
                self.paused = True
                return Decision("pause", "thermal_pause")
            return Decision("none", "thermal_paused_hot")

        if self.paused:
            if temperature_c <= self.policy.resume_temperature_c:
                self.consecutive_resume_samples += 1
            else:
                self.consecutive_resume_samples = 0
            if (
                self.consecutive_resume_samples
                >= self.policy.resume_stable_samples
            ):
                self.paused = False
                self.consecutive_resume_samples = 0
                return Decision("resume", "thermal_resume")
            return Decision("none", "thermal_cooldown")

        self.consecutive_resume_samples = 0
        if temperature_c >= self.policy.warning_temperature_c:
            return Decision("none", "thermal_warning")
        return Decision("none", "sample")


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    process_group: int
    session: int
    start_time_ticks: int


def _read_process_identity(pid: int) -> ProcessIdentity:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError) as error:
        raise GuardError(f"guarded process {pid} no longer exists") from error
    close = raw.rfind(")")
    if close < 0:
        raise GuardError(f"cannot parse /proc/{pid}/stat")
    fields = raw[close + 1 :].split()
    # fields[0] is kernel stat field 3 (state), so indexes 2, 3 and 19 are
    # respectively pgrp (5), session (6), and starttime (22).
    if len(fields) <= 19:
        raise GuardError(f"truncated /proc/{pid}/stat")
    try:
        return ProcessIdentity(
            pid=pid,
            process_group=int(fields[2]),
            session=int(fields[3]),
            start_time_ticks=int(fields[19]),
        )
    except ValueError as error:
        raise GuardError(f"invalid numeric identity for process {pid}") from error


def _session_group_members(process_group: int, session: int) -> list[int]:
    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            identity = _read_process_identity(int(entry.name))
        except GuardError:
            continue
        if (
            identity.process_group == process_group
            and identity.session == session
        ):
            members.append(identity.pid)
    return sorted(members)


class GuardedProcessGroup:
    """Signal only a newly-created, identity-checked child process group."""

    def __init__(
        self,
        process: subprocess.Popen[Any],
        *,
        terminate_grace_seconds: float,
    ) -> None:
        self.process = process
        self.terminate_grace_seconds = terminate_grace_seconds
        self.identity = _read_process_identity(process.pid)
        if self.identity.process_group != process.pid:
            raise GuardError("guarded command is not its own process-group leader")
        if self.identity.session != process.pid:
            raise GuardError("guarded command is not its own session leader")
        if self.identity.process_group == os.getpgrp():
            raise GuardError("watchdog and guarded command share a process group")

    def running(self) -> bool:
        return self.process.poll() is None

    def _verify_live_leader(self) -> bool:
        if self.process.poll() is not None:
            return False
        current = _read_process_identity(self.identity.pid)
        if current != self.identity:
            raise GuardError(
                "guarded process identity changed; refusing to signal a reused PID"
            )
        if current.process_group == os.getpgrp():
            raise GuardError("refusing to signal the watchdog's own process group")
        return True

    def signal(self, signum: int) -> bool:
        if not self._verify_live_leader():
            return False
        try:
            os.killpg(self.identity.process_group, signum)
        except ProcessLookupError:
            # The group may disappear between /proc verification and killpg.
            # Treat that as a completed job, never as permission to try a PID.
            if self.process.poll() is not None:
                return False
            raise GuardError("guarded process group disappeared during signalling")
        except OSError as error:
            raise GuardError(
                f"cannot signal guarded process group: {error}"
            ) from error
        return True

    def terminate(self, *, resume_first: bool) -> None:
        if self.running():
            if resume_first:
                self.signal(signal.SIGCONT)
            self.signal(signal.SIGTERM)

        deadline = time.monotonic() + self.terminate_grace_seconds
        while time.monotonic() < deadline:
            # Reap the direct child promptly; otherwise its zombie /proc entry
            # would make every clean SIGTERM wait for the full grace period.
            self.process.poll()
            if not _session_group_members(
                self.identity.process_group,
                self.identity.session,
            ):
                break
            time.sleep(0.1)
        members = _session_group_members(
            self.identity.process_group,
            self.identity.session,
        )
        if members:
            # Never signal a cached PGID after the validated session has become
            # empty.  At least one matching member must exist immediately before
            # escalation; ProcessLookupError means the group won the exit race.
            try:
                os.killpg(self.identity.process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as error:
                raise GuardError(
                    f"cannot kill guarded process group: {error}"
                ) from error
        try:
            self.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass


def _terminate_unvalidated_child(process: subprocess.Popen[Any]) -> None:
    """Reap a just-spawned child if process-group validation cannot complete.

    ``Popen`` remains the parent of this exact, unreaped PID, so signalling the
    PID through that handle cannot hit a recycled process.  Group signalling is
    intentionally avoided here because the group identity was not validated.
    """

    if process.poll() is not None:
        process.wait()
        return
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


class StatusLog:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "physical_gpu": PHYSICAL_GPU_INDEX,
            "gpu_lock": str(GPU_LOCK_PATH),
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()


def _best_effort_status(status: StatusLog, event: str, **fields: Any) -> None:
    """Record cleanup failures without allowing logging to skip cleanup."""

    try:
        status.write(event, **fields)
    except Exception as error:  # The original guard failure remains primary.
        print(
            f"GPU0 thermal guard could not write {event!r}: {error}",
            file=sys.stderr,
        )


def _sleep_to_poll_boundary(cycle_started: float, poll_seconds: float) -> None:
    """Keep query starts one sample interval apart, including query latency."""

    remaining = cycle_started + poll_seconds - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def _finite_float(value: str, label: str) -> float:
    try:
        parsed = float(value.strip())
    except ValueError as error:
        raise GuardError(f"invalid {label} telemetry: {value!r}") from error
    if not math.isfinite(parsed):
        raise GuardError(f"non-finite {label} telemetry")
    return parsed


def _integer(value: str, label: str) -> int:
    stripped = value.strip()
    if not stripped.isdigit():
        raise GuardError(f"invalid {label} telemetry: {value!r}")
    return int(stripped)


def parse_telemetry(raw: str) -> Telemetry:
    rows = [row for row in raw.splitlines() if row.strip()]
    if len(rows) != 1:
        raise GuardError(f"expected one GPU0 telemetry row, found {len(rows)}")
    fields = rows[0].split(",")
    if len(fields) != 6:
        raise GuardError(f"expected six GPU0 telemetry fields, found {len(fields)}")
    telemetry = Telemetry(
        temperature_c=_finite_float(fields[0], "temperature"),
        power_w=_finite_float(fields[1], "power draw"),
        power_limit_w=_finite_float(fields[2], "power limit"),
        utilization_percent=_integer(fields[3], "utilization"),
        memory_mib=_integer(fields[4], "memory"),
        pstate=fields[5].strip(),
    )
    if not 0 < telemetry.temperature_c <= 150:
        raise GuardError("temperature telemetry is outside 0..150 C")
    if telemetry.power_w < 0 or telemetry.power_limit_w <= 0:
        raise GuardError("power telemetry is outside its physical range")
    if not 0 <= telemetry.utilization_percent <= 100:
        raise GuardError("utilization telemetry is outside 0..100 percent")
    if not telemetry.pstate:
        raise GuardError("empty pstate telemetry")
    return telemetry


def sample_gpu0(nvidia_smi: str, policy: GuardPolicy = DEFAULT_POLICY) -> Telemetry:
    command = [
        nvidia_smi,
        "-i",
        str(PHYSICAL_GPU_INDEX),
        "--query-gpu=temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,pstate",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=policy.query_timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GuardError(f"GPU0 telemetry query failed: {error}") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise GuardError(
            f"GPU0 telemetry query exited {completed.returncode}: {detail}"
        )
    return parse_telemetry(completed.stdout)


def _telemetry_fields(telemetry: Telemetry | None) -> dict[str, Any]:
    return {} if telemetry is None else asdict(telemetry)


def _wait_before_launch(
    *,
    state: ThermalState,
    status: StatusLog,
    nvidia_smi: str,
) -> bool:
    """Wait for a trustworthy launch sample; return False after 12 failures."""

    while True:
        cycle_started = time.monotonic()
        telemetry: Telemetry | None = None
        error: str | None = None
        try:
            telemetry = sample_gpu0(nvidia_smi, state.policy)
            decision = state.observe_temperature(telemetry.temperature_c)
        except GuardError as caught:
            error = str(caught)
            decision = state.observe_failure()
        status.write(
            f"prelaunch_{decision.event}",
            paused=state.paused,
            consecutive_query_failures=state.consecutive_query_failures,
            consecutive_resume_samples=state.consecutive_resume_samples,
            error=error,
            **_telemetry_fields(telemetry),
        )
        if decision.action == "terminate":
            return False
        if telemetry is not None and not state.paused:
            return True
        _sleep_to_poll_boundary(cycle_started, state.policy.poll_seconds)


def run_guarded(
    command: Sequence[str],
    *,
    status_log: Path,
    nvidia_smi: str = "nvidia-smi",
    policy: GuardPolicy = DEFAULT_POLICY,
) -> int:
    policy.validate()
    if not command:
        raise GuardError("guarded command is empty")
    status = StatusLog(status_log)
    state = ThermalState(policy)
    status.write(
        "guard_start",
        guard_pid=os.getpid(),
        guard_process_group=os.getpgrp(),
        command=list(command),
        policy=asdict(policy),
        lock_owner="guarded_job_tree",
    )

    if not _wait_before_launch(
        state=state,
        status=status,
        nvidia_smi=nvidia_smi,
    ):
        status.write("prelaunch_telemetry_abort")
        return EXIT_THERMAL_GUARD

    process = subprocess.Popen(list(command), start_new_session=True)
    try:
        group = GuardedProcessGroup(
            process,
            terminate_grace_seconds=policy.terminate_grace_seconds,
        )
    except BaseException:
        _terminate_unvalidated_child(process)
        raise
    thermal_abort = False
    interrupted_signal: int | None = None

    def interrupt(signum: int, _frame: Any) -> None:
        raise GuardInterrupted(signum)

    previous_handlers: dict[int, Any] = {}
    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    try:
        status.write(
            "job_started",
            child_pid=group.identity.pid,
            child_process_group=group.identity.process_group,
            child_session=group.identity.session,
            child_start_time_ticks=group.identity.start_time_ticks,
        )
        for signum in handled_signals:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupt)

        while group.running():
            cycle_started = time.monotonic()
            telemetry: Telemetry | None = None
            error: str | None = None
            try:
                telemetry = sample_gpu0(nvidia_smi, policy)
                decision = state.observe_temperature(telemetry.temperature_c)
            except GuardError as caught:
                error = str(caught)
                decision = state.observe_failure()

            if decision.action == "pause":
                group.signal(signal.SIGSTOP)
            elif decision.action == "resume":
                group.signal(signal.SIGCONT)
            elif decision.action == "terminate":
                thermal_abort = True

            status.write(
                decision.event,
                child_pid=group.identity.pid,
                child_process_group=group.identity.process_group,
                paused=state.paused,
                consecutive_query_failures=state.consecutive_query_failures,
                consecutive_resume_samples=state.consecutive_resume_samples,
                error=error,
                **_telemetry_fields(telemetry),
            )
            if thermal_abort:
                group.terminate(resume_first=state.paused)
                break
            if group.running():
                _sleep_to_poll_boundary(cycle_started, policy.poll_seconds)
    except GuardInterrupted as caught:
        interrupted_signal = caught.signum
        try:
            status.write(
                "guard_interrupted",
                signal=interrupted_signal,
                paused=state.paused,
            )
        finally:
            group.terminate(resume_first=state.paused)
    except GuardError as caught:
        cleanup_error: BaseException | None = None
        try:
            group.terminate(resume_first=state.paused)
        except BaseException as error:
            cleanup_error = error
        _best_effort_status(status, "identity_guard_abort", error=str(caught))
        if cleanup_error is not None:
            _best_effort_status(
                status,
                "identity_guard_cleanup_refused",
                error=str(cleanup_error),
            )
        return EXIT_IDENTITY_GUARD
    except BaseException as caught:
        # Logging, handler installation, and unexpected implementation errors
        # are also fail-closed: never abandon a live GPU process tree.
        cleanup_error: BaseException | None = None
        try:
            group.terminate(resume_first=state.paused)
        except BaseException as error:
            cleanup_error = error
        _best_effort_status(
            status,
            "guard_internal_abort",
            error=f"{type(caught).__name__}: {caught}",
        )
        if cleanup_error is not None:
            _best_effort_status(
                status,
                "guard_internal_cleanup_failed",
                error=f"{type(cleanup_error).__name__}: {cleanup_error}",
            )
        raise
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)

    if interrupted_signal is not None:
        return 128 + interrupted_signal
    if thermal_abort:
        status.write("guard_terminated_job", reason="telemetry_failures")
        return EXIT_THERMAL_GUARD

    returncode = process.wait()
    leftovers = _session_group_members(
        group.identity.process_group,
        group.identity.session,
    )
    if leftovers:
        status.write("orphaned_group_members", pids=leftovers)
        group.terminate(resume_first=state.paused)
        return EXIT_IDENTITY_GUARD
    status.write("job_complete", returncode=returncode)
    return returncode


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-log", type=Path, required=True)
    parser.add_argument(
        "--job-owns-gpu-lock",
        type=Path,
        required=True,
        help=(
            "Must be /tmp/radio-gs-gpu0.lock. The watchdog records but does not "
            "acquire it, because the guarded LUDVIG job owns that flock."
        ),
    )
    parser.add_argument(
        "--nvidia-smi",
        default="nvidia-smi",
        help="Override only for a controlled test double.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=DEFAULT_POLICY.poll_seconds,
        metavar="SECONDS",
        help="Telemetry query start interval (default: 3 seconds).",
    )
    parser.add_argument(
        "--query-timeout",
        type=float,
        default=DEFAULT_POLICY.query_timeout_seconds,
        metavar="SECONDS",
        help="Per-query timeout; must be below --sample-interval (default: 2).",
    )
    parser.add_argument(
        "--warn",
        type=float,
        default=DEFAULT_POLICY.warning_temperature_c,
        metavar="CELSIUS",
        help="Warning threshold (default: 78 C).",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=DEFAULT_POLICY.pause_temperature_c,
        metavar="CELSIUS",
        help="Whole-job process-group pause threshold (default: 81 C).",
    )
    parser.add_argument(
        "--resume",
        type=float,
        default=DEFAULT_POLICY.resume_temperature_c,
        metavar="CELSIUS",
        help="Maximum temperature for a stable resume sample (default: 70 C).",
    )
    parser.add_argument(
        "--stable-samples",
        type=int,
        default=DEFAULT_POLICY.resume_stable_samples,
        metavar="COUNT",
        help="Consecutive cool samples required to resume (default: 2).",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.job_owns_gpu_lock != GPU_LOCK_PATH:
        parser.error(f"--job-owns-gpu-lock must be {GPU_LOCK_PATH}")
    args.policy = GuardPolicy(
        poll_seconds=args.sample_interval,
        query_timeout_seconds=args.query_timeout,
        warning_temperature_c=args.warn,
        pause_temperature_c=args.pause,
        resume_temperature_c=args.resume,
        resume_stable_samples=args.stable_samples,
    )
    try:
        args.policy.validate()
    except GuardError as error:
        parser.error(str(error))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_guarded(
            args.command,
            status_log=args.status_log,
            nvidia_smi=args.nvidia_smi,
            policy=args.policy,
        )
    except GuardError as error:
        print(f"GPU0 thermal guard error: {error}", file=sys.stderr)
        return EXIT_IDENTITY_GUARD


if __name__ == "__main__":
    raise SystemExit(main())
