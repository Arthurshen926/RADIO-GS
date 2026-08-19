import json
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


def test_spin_finetune_stages_use_exact_staged_capability_backward() -> None:
    source = Path(spin.__file__).read_text(encoding="utf-8")
    assert '"--staged-capability-gradient"' in source
    assert '"--offload-capability-adaptors-after-gradient"' in source
    assert '"--column-staged-direct-field-backward"' in source


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
    monkeypatch.setenv(spin.HOST_MEMORY_STAGE_SLOTS_ENV, "1")
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


def test_host_memory_stage_slot_count_is_fail_closed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(spin, "HOST_MEMORY_STAGE_LOCK", tmp_path / "host.lock")
    monkeypatch.setenv(spin.HOST_MEMORY_STAGE_SLOTS_ENV, "0")
    with pytest.raises(ValueError, match=r"must be in \[1,8\]"):
        with spin._host_memory_stage_boundary(scene="fixture", stage="base_field"):
            pass


def test_scene_runner_lock_waits_then_acquires_with_clear_log(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    scene_root = tmp_path / "scene"
    scene_root.mkdir()
    lock_path = scene_root / spin.SCENE_RUN_LOCK_NAME
    blocker = lock_path.open("a+", encoding="utf-8")
    spin.fcntl.flock(blocker.fileno(), spin.fcntl.LOCK_EX | spin.fcntl.LOCK_NB)
    sleeps = []

    def release_after_wait(seconds: float) -> None:
        sleeps.append(seconds)
        spin.fcntl.flock(blocker.fileno(), spin.fcntl.LOCK_UN)

    monkeypatch.setattr(spin.time, "sleep", release_after_wait)
    with spin._scene_run_boundary(run_root=scene_root, scene="fixture"):
        assert json.loads(lock_path.read_text(encoding="utf-8"))["scene"] == "fixture"
    blocker.close()

    output = capsys.readouterr().out
    assert sleeps == [1.0]
    assert "waiting for per-scene runner lock" in output
    assert "acquired per-scene runner lock" in output
    assert "released per-scene runner lock" in output
    assert lock_path.read_text(encoding="utf-8") == ""
    assert lock_path.stat().st_mode & 0o777 == 0o666


def test_scene_runner_lock_does_not_serialize_different_scenes(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    with spin._scene_run_boundary(run_root=first, scene="first"):
        with spin._scene_run_boundary(run_root=second, scene="second"):
            assert (first / spin.SCENE_RUN_LOCK_NAME).read_text(encoding="utf-8")
            assert (second / spin.SCENE_RUN_LOCK_NAME).read_text(encoding="utf-8")


def test_scene_runner_lock_releases_after_exception(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene"
    with pytest.raises(RuntimeError, match="fixture failure"):
        with spin._scene_run_boundary(run_root=scene_root, scene="fixture"):
            raise RuntimeError("fixture failure")

    with spin._scene_run_boundary(run_root=scene_root, scene="fixture"):
        assert json.loads(
            (scene_root / spin.SCENE_RUN_LOCK_NAME).read_text(encoding="utf-8")
        )["pid"] > 0


def test_run_holds_scene_lock_before_entering_pipeline(
    tmp_path: Path, monkeypatch
) -> None:
    assets = SimpleNamespace(scene="fixture")
    observed = {}
    monkeypatch.setattr(spin, "resolve_scene_assets", lambda _scene: assets)

    def pipeline(_args, *, assets, run_root):
        observed.update(
            json.loads(
                (run_root / spin.SCENE_RUN_LOCK_NAME).read_text(encoding="utf-8")
            )
        )
        return {"scene": assets.scene}

    monkeypatch.setattr(spin, "_run_with_scene_lock", pipeline)
    result = spin.run(
        SimpleNamespace(scene="fixture", run_root=str(tmp_path))
    )

    assert result == {"scene": "fixture"}
    assert observed["scene"] == "fixture"
    assert observed["run_root"] == str((tmp_path / "fixture").resolve())
