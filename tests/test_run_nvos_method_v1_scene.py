from pathlib import Path

import pytest

from radio_gs.scripts.run_nvos_method_v1_scene import (
    resolve_scene_assets,
    write_runtime_config,
)


def test_all_frozen_nvos_scenes_resolve_to_target_excluded_source_cohorts() -> None:
    for scene in (
        "fern",
        "flower",
        "fortress",
        "horns_center",
        "horns_left",
        "leaves",
        "orchids",
        "trex",
    ):
        assets = resolve_scene_assets(scene)
        assert assets.scene == scene
        assert assets.base_config.is_file()
        assert assets.geometry.is_file()
        assert assets.image_dir.is_dir()
        assert len(assets.excluded_stems) == 1
        assert assets.training_frame_count >= 5
        assert assets.feature_height > 0
        assert assets.feature_width > 0


def test_runtime_config_only_overrides_the_source_feature_bundle(
    tmp_path: Path,
) -> None:
    assets = resolve_scene_assets("fern")
    feature_dir = tmp_path / "features"
    output = tmp_path / "method.yaml"

    write_runtime_config(assets, feature_dir, output)

    text = output.read_text(encoding="utf-8")
    assert f"base_config: {assets.base_config}" in text
    assert f"feature_dir: {feature_dir}" in text
    assert f"val_feature_dir: {feature_dir}" in text
    assert "mask" not in text
    assert "scribble" not in text


def test_scene_outside_frozen_full8_fails_closed() -> None:
    with pytest.raises(ValueError, match="outside frozen NVOS full8"):
        resolve_scene_assets("room")
