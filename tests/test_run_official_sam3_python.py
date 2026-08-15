from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "radio_gs/scripts/run_official_sam3_python.sh"


def _fake_python(tmp_path: Path) -> Path:
    executable = tmp_path / "python"
    executable.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"${LD_PRELOAD:-}\"\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _runner_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    source = tmp_path / "sam3"
    source.mkdir()
    version = tmp_path / "nvidia-version"
    version.write_text(
        "NVRM version: NVIDIA UNIX x86_64 Kernel Module  535.288.01  test\n",
        encoding="utf-8",
    )
    library_dir = tmp_path / "drivers"
    library_dir.mkdir()
    driver = library_dir / "libcuda.so.535.288.01"
    driver.touch()
    environment = os.environ.copy()
    environment.pop("LD_PRELOAD", None)
    environment.pop("RADIO_GS_DRIVER_LIBRARY", None)
    environment.update(
        {
            "RADIO_GS_SAM3_PYTHON": str(_fake_python(tmp_path)),
            "RADIO_GS_SAM3_SOURCE": str(source),
            "RADIO_GS_NVIDIA_VERSION_FILE": str(version),
            "RADIO_GS_DRIVER_LIBRARY_DIR": str(library_dir),
        }
    )
    return environment, driver


def test_official_sam3_runner_preloads_kernel_matched_libcuda(tmp_path: Path) -> None:
    environment, driver = _runner_environment(tmp_path)
    result = subprocess.run(
        ["bash", str(RUNNER), "ignored.py"],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(driver)


def test_explicit_empty_driver_override_disables_preload(tmp_path: Path) -> None:
    environment, _ = _runner_environment(tmp_path)
    environment["RADIO_GS_DRIVER_LIBRARY"] = ""
    result = subprocess.run(
        ["bash", str(RUNNER), "ignored.py"],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
