from __future__ import annotations

import json
from pathlib import Path

from radio_gs.scripts import download_promptable_nvs_data as downloader


def test_unverified_fork_candidate_is_preserved_and_never_completes_cohort(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "SPIn-NeRF" / "source_images"
    fork = target / "fork" / "fork.zip"
    fork.parent.mkdir(parents=True)
    fork.write_bytes(b"user recovery candidate")

    def fake_archive(_opener, *, url, destination, expected_size, **_kwargs):
        return {
            "path": str(destination),
            "bytes": int(expected_size),
            "sha256": "0" * 64,
            "status": "fixture",
            "url": url,
        }

    monkeypatch.setattr(downloader, "_record_existing_or_download", fake_archive)

    result = downloader.download_spin_segmentation_sources(
        tmp_path,
        allow_missing_upstream=True,
    )

    assert result == target
    assert fork.read_bytes() == b"user recovery candidate"
    manifest = json.loads((target / "download_manifest.json").read_text())
    assert manifest["complete_10_scene_rgb_cohort"] is False
    assert manifest["missing_required_assets"] == ["fork_verified_rgb_and_poses"]
    fork_record = next(item for item in manifest["files"] if item["path"] == "fork/fork.zip")
    assert fork_record["observed_bytes"] == len(b"user recovery candidate")
    assert fork_record["formal_evaluation_ready"] is False

