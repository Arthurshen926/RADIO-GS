import subprocess
from pathlib import Path


def test_scannet_v67_queue_launcher_reports_warmstart():
    script = Path("radio_gs/scripts/launch_scannet_v67_scene_queue.sh")

    result = subprocess.run(
        ["bash", str(script), "--print-warmstart", "scene0200_00"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert "scannet_og_scene0200_00_v63fair_teacherpce" in result.stdout
    assert result.stdout.strip().endswith("/checkpoints/best.pth")


def test_scannet_v67_queue_launcher_rejects_unknown_scene():
    script = Path("radio_gs/scripts/launch_scannet_v67_scene_queue.sh")

    result = subprocess.run(
        ["bash", str(script), "--print-warmstart", "scene9999_99"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode != 0
    assert "unknown scene" in result.stderr


def test_scannet_v67_queue_launcher_preserves_prompt_braces():
    script = Path("radio_gs/scripts/launch_scannet_v67_scene_queue.sh")

    result = subprocess.run(
        ["bash", str(script), "--print-prompts"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.stdout.strip().startswith("{query}|a photo")
    assert "containing a {query}" in result.stdout
