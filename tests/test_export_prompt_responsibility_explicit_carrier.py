import hashlib
import json
from pathlib import Path

import pytest

from radio_gs.scripts import export_prompt_responsibility_explicit_carrier as wrapper


LEGACY_EXPORTER_SHA256 = (
    "5ba326b6fcfb5dd1901569f6537ffb19b53760f86dafff787ba92ea9e8ed16e4"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset(tmp_path: Path, name: str, payload: bytes) -> tuple[Path, str]:
    path = tmp_path / name
    path.write_bytes(payload)
    return path, _sha256(path)


def _cli(tmp_path: Path) -> list[str]:
    config, config_sha256 = _asset(tmp_path, "config.yaml", b"config")
    checkpoint, checkpoint_sha256 = _asset(tmp_path, "carrier.pth", b"checkpoint")
    camera_map, camera_map_sha256 = _asset(tmp_path, "camera-map.json", b"camera")
    return [
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--scene-id",
        "room",
        "--scene-config",
        str(config),
        "--scene-config-sha256",
        config_sha256,
        "--scene-checkpoint",
        str(checkpoint),
        "--scene-checkpoint-sha256",
        checkpoint_sha256,
        "--camera-map",
        str(camera_map),
        "--camera-map-sha256",
        camera_map_sha256,
        "--output",
        str(tmp_path / "exact-w.pt"),
        "--report",
        str(tmp_path / "report.json"),
        "--expected-prompt-type",
        "reference_binary_mask",
        "--expected-reference-frame-id",
        "frame-0",
        "--expected-reference-mask-sha256",
        "a" * 64,
        "--expected-native-height",
        "756",
        "--expected-native-width",
        "1008",
    ]


@pytest.mark.parametrize(
    "flag",
    (
        "--scene-config",
        "--scene-config-sha256",
        "--scene-checkpoint",
        "--scene-checkpoint-sha256",
        "--camera-map",
        "--camera-map-sha256",
    ),
)
def test_cli_requires_all_three_carrier_paths_and_hashes(tmp_path, flag):
    cli = _cli(tmp_path)
    position = cli.index(flag)
    del cli[position : position + 2]
    with pytest.raises(SystemExit) as error:
        wrapper.parser().parse_args(cli)
    assert error.value.code == 2


@pytest.mark.parametrize(
    ("path_attribute", "hash_attribute", "label"),
    (
        ("scene_config", "scene_config_sha256", "scene config"),
        ("scene_checkpoint", "scene_checkpoint_sha256", "scene checkpoint"),
        ("camera_map", "camera_map_sha256", "camera map"),
    ),
)
def test_each_carrier_hash_fails_closed_before_legacy_export(
    tmp_path, monkeypatch, path_attribute, hash_attribute, label
):
    args = wrapper.parser().parse_args(_cli(tmp_path))
    setattr(args, hash_attribute, "0" * 64)
    called = False

    def legacy_export_must_not_run(_args):
        nonlocal called
        called = True
        raise AssertionError("legacy export ran before complete carrier verification")

    monkeypatch.setattr(wrapper, "export", legacy_export_must_not_run)
    with pytest.raises(ValueError, match=rf"{label} SHA-256 differs"):
        wrapper.run(args)
    assert called is False
    assert not Path(args.output).exists()
    assert not Path(args.report).exists()
    assert not (Path(args.output).parent / "carrier_overlay_v1").exists()


def test_verified_overlay_preserves_legacy_call_contract(tmp_path, monkeypatch):
    args = wrapper.parser().parse_args(_cli(tmp_path))
    captured = {}

    def inspect_legacy_export(legacy_args):
        captured.update(vars(legacy_args))
        overlay_scene = Path(legacy_args.queue_root) / "scenes" / args.scene_id
        bindings = {
            overlay_scene / "gaussfm_main_track.yaml": Path(args.scene_config),
            overlay_scene / "feature_field" / "checkpoints" / "best.pth": Path(
                args.scene_checkpoint
            ),
            overlay_scene / "rgb_to_colmap_camera_mapping.json": Path(args.camera_map),
        }
        for destination, source in bindings.items():
            assert destination.is_symlink()
            assert destination.resolve() == source.resolve()
        Path(legacy_args.output).write_bytes(b"legacy exact-W artifact")
        return {"legacy_report": True}

    monkeypatch.setattr(wrapper, "export", inspect_legacy_export)
    result = wrapper.run(args)

    assert set(captured) == {
        "manifest",
        "queue_root",
        "scene_id",
        "output",
        "report",
        "device",
        "cpu_staging_lock",
        "telemetry_log",
        "execution_log",
        "expected_prompt_type",
        "expected_reference_frame_id",
        "expected_reference_mask_sha256",
        "expected_native_height",
        "expected_native_width",
        "overwrite",
    }
    assert captured["manifest"] == args.manifest
    assert captured["scene_id"] == args.scene_id
    assert captured["output"] == str(Path(args.output).resolve())
    assert captured["report"] is None
    assert captured["overwrite"] is False
    for name in (
        "device",
        "cpu_staging_lock",
        "telemetry_log",
        "execution_log",
        "expected_prompt_type",
        "expected_reference_frame_id",
        "expected_reference_mask_sha256",
        "expected_native_height",
        "expected_native_width",
    ):
        assert captured[name] == getattr(args, name)

    authority = result["carrier_override_authority"]
    assert authority["registration"] == wrapper.REGISTRATION
    assert authority["all_three_verified_before_legacy_export"] is True
    assert authority["legacy_exporter_modified"] is False
    assert authority["config"] == {
        "path": str(Path(args.scene_config).resolve()),
        "sha256": args.scene_config_sha256,
    }
    assert authority["checkpoint"] == {
        "path": str(Path(args.scene_checkpoint).resolve()),
        "sha256": args.scene_checkpoint_sha256,
    }
    assert authority["camera_map"] == {
        "path": str(Path(args.camera_map).resolve()),
        "sha256": args.camera_map_sha256,
    }
    assert authority["wrapper_sha256"] == _sha256(Path(wrapper.__file__))
    assert json.loads(Path(args.report).read_text(encoding="utf-8")) == result


def test_legacy_exporter_source_is_bitwise_frozen():
    legacy_path = Path(wrapper.export.__module__.replace(".", "/") + ".py")
    repository_root = Path(__file__).resolve().parents[1]
    assert _sha256(repository_root / legacy_path) == LEGACY_EXPORTER_SHA256
