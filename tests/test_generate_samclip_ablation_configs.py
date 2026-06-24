import json
from pathlib import Path

import yaml

from radio_gs.scripts.generate_samclip_ablation_configs import generate_config


def _write_template(path: Path, scene: str, dataset_type: str) -> None:
    payload = {
        "exp_name": f"radio_gs_{scene}_base",
        "output_dir": "/tmp/base",
        "dataset_type": dataset_type,
        "scene": scene,
        "radio_feature_dim": 1280,
        "feature_height": 30,
        "feature_width": 40,
        "feature_dir": "/tmp/radio_features",
        "val_feature_dir": "/tmp/radio_features",
        "siglip_alignment_weight": 0.2,
        "siglip_summary_alignment_weight": 0.3,
        "text_heatmap_distill_weight": 0.4,
        "radio_adaptor_cross_view_names": "dino_v3",
        "radio_adaptor_cross_view_weight": 0.5,
        "direct_point_loss_weight": 0.6,
        "direct_point_teacher_cache": "output/radio_teacher.pt",
        "direct_point_text_embeddings": "checkpoints/siglip2_text.pt",
        "siglip_projection_weights": "checkpoints/proj.pth",
        "siglip_summary_head_weights": "checkpoints/head.pth",
        "grounding_query_loss_weight": 0.1,
        "grounding_text_embeddings": "checkpoints/siglip2_text_embeddings_v2.pt",
        "grounding_use_adaptor": True,
        "frozen_depth_head_weight": 0.1,
        "frozen_depth_head_path": "output/depth_head.pth",
        "frozen_seg_head_weight": 0.2,
        "frozen_seg_head_path": "output/seg_head.pth",
        "warmstart_from": "output/radio_gs/radio/checkpoints/best.pth",
        "resume_from": "output/radio_gs/radio/checkpoints/last.pth",
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_manifest(root: Path, scene: str, level: int, output_size: tuple[int, int]) -> None:
    level_root = root / scene / f"l{level}"
    level_root.mkdir(parents=True)
    (level_root / "samclip_manifest.json").write_text(
        json.dumps({"output_size": list(output_size), "feature_dim": 512}),
        encoding="utf-8",
    )


def test_generate_lerf_samclip_config_disables_radio_helpers(tmp_path):
    template = tmp_path / "lerf.yaml"
    _write_template(template, scene="figurines", dataset_type="lerf")
    samclip_root = tmp_path / "samclip_lerf"
    _write_manifest(samclip_root, "figurines", 1, (90, 123))

    output = generate_config(
        template,
        scene="figurines",
        scene_root=tmp_path / "lerf_ovs" / "figurines",
        samclip_root=samclip_root,
        level=1,
        output_root=tmp_path / "configs",
        variant="samclip_l1_smoke_e1",
        repo_root=tmp_path / "repo",
        epochs=1,
    )

    assert output.name == "lerf_figurines_samclip_l1_smoke_e1.yaml"
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["exp_name"] == "lerf_figurines_samclip_l1_smoke_e1"
    assert payload["output_dir"] == str(tmp_path / "repo" / "output" / "radio_gs" / "lerf_figurines_samclip_l1_smoke_e1")
    assert payload["scene_root"] == str(tmp_path / "lerf_ovs" / "figurines")
    assert payload["radio_feature_dim"] == 512
    assert payload["feature_height"] == 90
    assert payload["feature_width"] == 123
    assert payload["feature_dir"] == str(samclip_root / "figurines" / "l1")
    assert payload["val_feature_dir"] == str(samclip_root / "figurines" / "l1")
    assert payload["siglip_alignment_weight"] == 0.0
    assert payload["siglip_summary_alignment_weight"] == 0.0
    assert payload["text_heatmap_distill_weight"] == 0.0
    assert payload["radio_adaptor_cross_view_names"] == ""
    assert payload["radio_adaptor_cross_view_weight"] == 0.0
    assert payload["direct_point_loss_weight"] == 0.0
    assert payload["direct_point_teacher_cache"] == ""
    assert payload["direct_point_text_embeddings"] == ""
    assert payload["siglip_projection_weights"] == ""
    assert payload["siglip_summary_head_weights"] == ""
    assert payload["grounding_query_loss_weight"] == 0.0
    assert payload["grounding_text_embeddings"] == ""
    assert payload["grounding_use_adaptor"] is False
    assert payload["frozen_depth_head_weight"] == 0.0
    assert payload["frozen_depth_head_path"] == ""
    assert payload["frozen_seg_head_weight"] == 0.0
    assert payload["frozen_seg_head_path"] == ""
    assert payload["warmstart_from"] == ""
    assert payload["resume_from"] == ""
    assert payload["samclip_feature_level"] == 1
    assert payload["samclip_language_feature_dir"] == str(samclip_root / "figurines" / "l1")
    assert payload["epochs"] == 1


def test_generate_scannet_samclip_config_uses_scene_filename(tmp_path):
    scene = "scene0000_00"
    template = tmp_path / "scannet.yaml"
    _write_template(template, scene=scene, dataset_type="scannet")
    samclip_root = tmp_path / "samclip_scannet"
    _write_manifest(samclip_root, scene, 2, (60, 80))

    output = generate_config(
        template,
        scene=scene,
        samclip_root=samclip_root,
        level=2,
        output_root=tmp_path / "configs",
        variant="samclip_l2_smoke_e1",
        repo_root=tmp_path / "repo",
        epochs=1,
    )

    assert output.name == "scene0000_00_samclip_l2_smoke_e1.yaml"
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["exp_name"] == "scene0000_00_samclip_l2_smoke_e1"
    assert payload["feature_dir"] == str(samclip_root / scene / "l2")
    assert payload["val_feature_dir"] == str(samclip_root / scene / "l2")
    assert payload["feature_height"] == 60
    assert payload["feature_width"] == 80
