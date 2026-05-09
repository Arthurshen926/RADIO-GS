import subprocess
from pathlib import Path


def test_scannet_dino_cv_queue_launcher_prints_config_path() -> None:
    script = Path("radio_gs/scripts/launch_scannet_dino_cv_queue.sh")

    result = subprocess.run(
        ["bash", str(script), "--print-config", "scene0000_00"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert "radio_gs/configs/generated/scannet_dino_cv" in result.stdout
    assert "v67_dino_cv001_b2_s32768_ft20_scene0000_00.yaml" in result.stdout


def test_scannet_dino_cv_queue_launcher_preserves_prompt_braces() -> None:
    script = Path("radio_gs/scripts/launch_scannet_dino_cv_queue.sh")

    result = subprocess.run(
        ["bash", str(script), "--print-prompts"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.stdout.strip().startswith("{query}|a photo")
    assert "containing a {query}" in result.stdout
