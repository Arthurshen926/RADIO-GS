from pathlib import Path
from types import SimpleNamespace

import pytest

import radio_gs.scripts.run_spin9_method_v1_scene as spin

from radio_gs.scripts.run_spin9_method_v1_scene import (
    CAPABILITY_SHARD_CHANNELS,
    _write_cohort_authority,
    source_feature_command,
    resolve_scene_assets,
)


def test_spin_source_features_use_all_rgb_at_frozen_grid() -> None:
    assets = resolve_scene_assets("orchids")
    command = [str(value) for value in source_feature_command(assets, Path("features"))]

    assert command[command.index("--resolution_scale") + 1] == "1.0"
    assert command[command.index("--frame-id-mode") + 1] == "source_rank"
    assert command[command.index("--adaptor_names") + 1] == "siglip2-g"
    assert "--exclude-image-stems-file" not in command
    assert "--exclude-frame-ids" not in command
    assert assets.feature_height == 189
    assert assets.feature_width == 252
    assert CAPABILITY_SHARD_CHANNELS == 512


def test_spin_scene_outside_available_nine_fails_closed() -> None:
    with pytest.raises(ValueError, match="outside frozen SPIn Available-Nine"):
        resolve_scene_assets("flower")


def test_spin_cohort_classifies_all_views_as_registered_source_rgb(
    tmp_path: Path,
) -> None:
    assets = resolve_scene_assets("orchids")
    caches = []
    for name in ("raw.pt", "dino.pt", "sam.pt"):
        path = tmp_path / name
        path.write_bytes(name.encode("utf-8"))
        caches.append(path)
    authority = tmp_path / "cohort.json"

    _write_cohort_authority(
        path=authority,
        assets=assets,
        feature_bundle_sha256="a" * 64,
        exact_raw=caches[0],
        exact_dino=caches[1],
        exact_sam=caches[2],
    )

    import json

    target_access = json.loads(authority.read_text(encoding="utf-8"))["target_access"]
    assert target_access["registered_source_rgb_opened"] is True
    assert target_access["benchmark_images_opened"] is False
    assert target_access["benchmark_masks_opened"] is False
    assert target_access["reference_masks_opened"] is False
    assert target_access["evaluation_masks_opened"] is False


def test_host_memory_heavy_stage_lock_wraps_only_declared_stages(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(spin, "HOST_MEMORY_STAGE_LOCK", tmp_path / "host.lock")
    monkeypatch.setattr(
        spin,
        "_run",
        lambda _args, stage, _command, *, gpu, log_dir: calls.append(
            (stage, gpu)
        ),
    )
    args = SimpleNamespace(scene="fixture")

    spin._run_spin_stage(
        args, "region_stage", ["command"], gpu=True, log_dir=tmp_path
    )
    spin._run_spin_stage(
        args, "source_features", ["command"], gpu=True, log_dir=tmp_path
    )

    assert calls == [("region_stage", True), ("source_features", True)]
    assert (tmp_path / "host.lock").read_text(encoding="utf-8") == ""
    assert "factorized_mpr" in spin.HOST_MEMORY_HEAVY_STAGES
    assert "generic_stage" in spin.HOST_MEMORY_HEAVY_STAGES
    assert "source_features" not in spin.HOST_MEMORY_HEAVY_STAGES
