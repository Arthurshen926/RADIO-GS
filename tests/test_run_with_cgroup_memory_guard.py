from __future__ import annotations

import os
from pathlib import Path
import subprocess


SCRIPT = Path("radio_gs/scripts/run_with_cgroup_memory_guard.sh").resolve()


def _environment(current: Path, log: Path, maximum: int) -> dict[str, str]:
    return {
        **os.environ,
        "HOST_MEMORY_CURRENT_PATH": str(current),
        "HOST_MEMORY_LOG": str(log),
        "HOST_MEMORY_MAX_BYTES": str(maximum),
        "HOST_MEMORY_POLL_SECONDS": "1",
    }


def test_memory_guard_records_pid_peak_and_success(tmp_path: Path) -> None:
    current = tmp_path / "memory.current"
    log = tmp_path / "memory.csv"
    current.write_text("17\n", encoding="utf-8")
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--", "bash", "-c", "exit 0"],
        env=_environment(current, log, 20),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    content = log.read_text(encoding="utf-8")
    assert "child_pgid" in content
    assert ",17,17,20,launch" in content
    assert ",17,17,20,complete" in content


def test_memory_guard_rejects_prelaunch_above_limit(tmp_path: Path) -> None:
    current = tmp_path / "memory.current"
    log = tmp_path / "memory.csv"
    current.write_text("21\n", encoding="utf-8")
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--", "bash", "-c", "exit 0"],
        env=_environment(current, log, 20),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 89
    assert "prelaunch_memory_above_limit" in log.read_text(encoding="utf-8")
