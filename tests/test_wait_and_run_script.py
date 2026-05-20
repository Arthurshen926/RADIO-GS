import os
import subprocess
from pathlib import Path


def test_wait_and_run_keeps_waiting_when_no_gpu_matches(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        "printf '0, 1000, 99\\n'\n",
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)
    log_path = tmp_path / "wait.log"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "MIN_FREE_MIB": "12000",
            "MAX_UTIL": "35",
            "CHECK_INTERVAL": "1",
            "GPU_LIST": "0",
            "GPU_LOCK_DIR": str(tmp_path / "locks"),
        }
    )

    result = subprocess.run(
        [
            "timeout",
            "3",
            "bash",
            "radio_gs/scripts/wait_and_run.sh",
            str(log_path),
            "bash",
            "-lc",
            "exit 7",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 124
    assert "No suitable GPU yet" in log_path.read_text(encoding="utf-8")
