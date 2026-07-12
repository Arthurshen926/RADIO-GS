from __future__ import annotations

import json
from pathlib import Path

import yaml

from radio_gs.scripts.generate_openclip_ablation_configs import generate_config


def _write_template(path: Path, scene: str, dataset_type: str) -> None:
    payload = {
        "exp_name": f"{scene}_base",
        "output_dir": "/tmp/base",
        "dataset_type": dataset_type,
        "scene": scene,
        "radio_feature_dim": 1280,
        "codec_type": "hcd",
        "bottleneck_dim": 64,
        "hybrid_output_dim": 128,
        "latent_dim": 64,
        "feature_height": 30,
        "feature_width": 40,
        "feature_dir": "/tmp/radio_features",
        "val_feature_dir": "/tmp/radio_features",
        "samclip_language_feature_dir": "/tmp/samclip",
        "samclip_mask_loss_weight": 1.0,
        "samclip_contrastive_loss_weight": 0.5,
        "samclip_background_loss_weight": 0.25,
        "siglip_alignment_weight": 0.2,
        "radio_adaptor_cross_view_names": "dino_v3",
        "radio_adaptor_cross_view_weight": 0.3,
        "direct_point_loss_weight": 0.4,
        "grounding_query_loss_weight": 0.5,
        "use_refiner": True,
        "self_guided": True,
        "featsharp_mode": "analytical",
        "featsharp_strength": 0.35,
        "tv_weight": 0.01,
        "resume_from": "/tmp/old.pth",
        "warmstart_from": "/tmp/warm.pth",
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_manifest(feature_dir: Path, output_size: tuple[int, int]) -> None:
    feature_dir.mkdir(parents=True)
    (feature_dir / "openclip_dense_manifest.json").write_text(
        json.dumps({"output_size": list(output_size), "feature_dim": 512}),
        encoding="utf-8",
    )


def test_generate_lerf_openclip_config_disables_sam_and_radio_helpers(tmp_path):
    template = tmp_path / "lerf.yaml"
    _write_template(template, scene="figurines", dataset_type="lerf")
    feature_dir = tmp_path / "features" / "figurines" / "vitb16_l14"
    _write_manifest(feature_dir, (14, 14))

    output = generate_config(
        template,
        scene="figurines",
        scene_root=tmp_path / "lerf_ovs" / "figurines",
        feature_dir=feature_dir,
        output_root=tmp_path / "configs",
        variant="openclip_vitb16_l14_e1",
        repo_root=tmp_path / "repo",
        epochs=1,
    )

    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert output.name == "lerf_figurines_openclip_vitb16_l14_e1.yaml"
    assert payload["exp_name"] == "lerf_figurines_openclip_vitb16_l14_e1"
    assert payload["output_dir"] == str(tmp_path / "repo" / "output" / "radio_gs" / "lerf_figurines_openclip_vitb16_l14_e1")
    assert payload["scene_root"] == str(tmp_path / "lerf_ovs" / "figurines")
    assert payload["rgb_dir"] == str(tmp_path / "lerf_ovs" / "figurines" / "images")
    assert payload["radio_feature_dim"] == 512
    assert payload["codec_type"] == "identity"
    assert payload["bottleneck_dim"] == 512
    assert payload["hybrid_output_dim"] == 512
    assert payload["latent_dim"] == 512
    assert payload["feature_height"] == 14
    assert payload["feature_width"] == 14
    assert payload["feature_dir"] == str(feature_dir)
    assert payload["val_feature_dir"] == str(feature_dir)
    assert payload["samclip_language_feature_dir"] == ""
    assert payload["samclip_mask_loss_weight"] == 0.0
    assert payload["samclip_contrastive_loss_weight"] == 0.0
    assert payload["samclip_background_loss_weight"] == 0.0
    assert payload["siglip_alignment_weight"] == 0.0
    assert payload["radio_adaptor_cross_view_names"] == ""
    assert payload["radio_adaptor_cross_view_weight"] == 0.0
    assert payload["direct_point_loss_weight"] == 0.0
    assert payload["grounding_query_loss_weight"] == 0.0
    assert payload["grounding_use_adaptor"] is False
    assert payload["use_refiner"] is False
    assert payload["self_guided"] is False
    assert payload["featsharp_mode"] == "none"
    assert payload["featsharp_strength"] == 0.0
    assert payload["tv_weight"] == 0.0
    assert payload["resume_from"] == ""
    assert payload["warmstart_from"] == ""
    assert payload["epochs"] == 1


def test_generate_scannet_openclip_config_uses_scene_filename(tmp_path):
    scene = "scene0000_00"
    template = tmp_path / "scannet.yaml"
    _write_template(template, scene=scene, dataset_type="scannet")
    feature_dir = tmp_path / "features" / scene / "vitb16_l14"
    _write_manifest(feature_dir, (14, 14))

    output = generate_config(
        template,
        scene=scene,
        feature_dir=feature_dir,
        output_root=tmp_path / "configs",
        variant="openclip_vitb16_l14_e1",
        repo_root=tmp_path / "repo",
        epochs=1,
    )

    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert output.name == "scene0000_00_openclip_vitb16_l14_e1.yaml"
    assert payload["exp_name"] == "scene0000_00_openclip_vitb16_l14_e1"
    assert payload["feature_dir"] == str(feature_dir)
    assert payload["val_feature_dir"] == str(feature_dir)
    assert payload["feature_height"] == 14
    assert payload["feature_width"] == 14
