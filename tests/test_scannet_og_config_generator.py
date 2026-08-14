import json
from pathlib import Path

import yaml

from radio_gs.config import load_config
from radio_gs.scripts.generate_scannet_og_configs import generate_config, main


def _write_prepared_scene(root: Path, scene: str = "scene0000_00") -> Path:
    scene_root = root / scene
    (scene_root / "color").mkdir(parents=True)
    (scene_root / "splits").mkdir()
    (scene_root / "splits" / "train_frames.txt").write_text("0\n2\n", encoding="utf-8")
    (scene_root / "splits" / "val_frames.txt").write_text("1\n", encoding="utf-8")
    (scene_root / "traj_w_c.txt").write_text(" ".join(["1"] * 16) + "\n", encoding="utf-8")
    (scene_root / "points3d.ply").write_text("ply\n", encoding="utf-8")
    transforms = {
        "w": 640,
        "h": 480,
        "fl_x": 577.0,
        "fl_y": 578.0,
        "cx": 319.5,
        "cy": 239.5,
        "frames": [],
    }
    (scene_root / "transforms.json").write_text(json.dumps(transforms), encoding="utf-8")
    return scene_root


def test_generate_config_targets_prepared_opengaussian_scene(tmp_path):
    scene = "scene0000_00"
    prepared_root = tmp_path / "dataset" / "scannet_og"
    scene_root = _write_prepared_scene(prepared_root, scene)
    output_root = tmp_path / "configs"
    repo_root = tmp_path / "repo"

    output_path = generate_config(
        scene=scene,
        prepared_root=prepared_root,
        output_root=output_root,
        repo_root=repo_root,
        geom_tag="og_rgb_3dgs",
        iters=123,
    )

    assert output_path == output_root / f"scannet_og_hybrid_v14_{scene}.yaml"
    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert payload["dataset_type"] == "scannet"
    assert payload["scene"] == scene
    assert payload["scene_root"] == str(scene_root)
    assert payload["feature_dir"] == str(repo_root / "output" / "radio_features_scannet_og" / scene)
    assert payload["pose_file"] == str(scene_root / "traj_w_c.txt")
    assert payload["val_pose_file"] == str(scene_root / "traj_w_c.txt")
    assert payload["train_frame_ids_path"] == str(scene_root / "splits" / "train_frames.txt")
    assert payload["val_frame_ids_path"] == str(scene_root / "splits" / "val_frames.txt")
    assert payload["rgb_dir"] == str(scene_root / "color")
    assert payload["val_rgb_dir"] == str(scene_root / "color")
    assert payload["ply_path"] == str(
        repo_root
        / "output"
        / "3dgs_models"
        / "scannet_og"
        / scene
        / "og_rgb_3dgs"
        / "point_cloud"
        / "iteration_123"
        / "point_cloud.ply"
    )
    assert payload["depth_dir"] == ""
    assert payload["val_depth_dir"] == ""
    assert payload["semantics_dir"] == ""
    assert payload["val_semantics_dir"] == ""
    assert payload["depth_loss_weight"] == 0.0
    assert payload["geom_depth_loss_weight"] == 0.0
    assert payload["seg_loss_weight"] == 0.0
    assert payload["frozen_depth_head_weight"] == 0.0
    assert payload["frozen_seg_head_weight"] == 0.0

    loaded = load_config(str(output_path))
    assert loaded.direct_point_gaussian_position_mode == "label_point"
    assert loaded.feature_height == 60
    assert loaded.feature_width == 80


def test_generate_config_can_emit_direct_point_variant(tmp_path):
    scene = "scene0000_00"
    prepared_root = tmp_path / "dataset" / "scannet_og"
    _write_prepared_scene(prepared_root, scene)
    output_root = tmp_path / "configs"
    repo_root = tmp_path / "repo"

    output_path = generate_config(
        scene=scene,
        prepared_root=prepared_root,
        output_root=output_root,
        repo_root=repo_root,
        variant="v15_direct",
        epochs=20,
        siglip_spatial_alignment_weight=0.1,
        direct_point_loss_weight=0.25,
        direct_point_sample_count=4096,
        direct_point_sample_strategy="class_balanced",
        direct_point_query_mode="knn",
        direct_point_gaussian_position_mode="label_point",
        direct_point_source="label_ply",
        direct_point_teacher_cache="output/teacher_cache/{scene}.pt",
        direct_point_feature_key="semantic",
        direct_point_candidate_k=32,
        direct_point_summary_alignment_weight=0.4,
        direct_point_text_loss_weight=0.7,
        direct_point_adapter_text_loss_weight=0.2,
        direct_point_adapter_text_distill_weight=0.9,
        direct_point_text_embeddings="checkpoints/siglip2_scannet_og_text_embeddings_ens5.pt",
        direct_point_text_split="19",
        direct_point_text_temperature=0.05,
        direct_point_text_ce_weighting="sqrt_inverse_pool_capped",
        direct_point_text_ce_min_weight=0.8,
        direct_point_text_ce_max_weight=2.0,
        direct_point_text_distill_weight=0.8,
        direct_point_text_distill_temperature=1.5,
        direct_point_text_distill_confidence_threshold=0.6,
        direct_point_text_pseudo_ce_weight=0.6,
        direct_point_text_pseudo_ce_confidence_threshold=0.25,
        direct_point_text_pseudo_ce_logit_scale=18.0,
        direct_point_text_pseudo_ce_center_logits=True,
        direct_point_text_pseudo_ce_splits="19,15,10",
        direct_point_adapter_text_pseudo_ce_weight=1.1,
        direct_point_adapter_text_pseudo_ce_confidence_threshold=0.3,
        direct_point_adapter_text_pseudo_ce_logit_scale=20.0,
        direct_point_adapter_text_pseudo_ce_center_logits=True,
        direct_point_adapter_text_pseudo_ce_splits="10,15",
        direct_point_adapter_decoder_anchor_weight=0.4,
    )

    assert output_path == output_root / f"scannet_og_hybrid_v15_direct_{scene}.yaml"
    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert payload["exp_name"] == f"radio_gs_scannet_og_{scene}_v15_direct"
    assert payload["output_dir"] == str(
        repo_root / "output" / "radio_gs" / f"scannet_og_{scene}_v15_direct"
    )
    assert payload["epochs"] == 20
    assert payload["siglip_alignment_weight"] == 0.1
    assert payload["siglip_summary_alignment_weight"] == 0.0
    assert payload["direct_point_loss_weight"] == 0.25
    assert payload["direct_point_sample_count"] == 4096
    assert payload["direct_point_sample_strategy"] == "class_balanced"
    assert payload["direct_point_query_mode"] == "knn"
    assert payload["direct_point_gaussian_position_mode"] == "label_point"
    assert payload["direct_point_source"] == "label_ply"
    assert payload["direct_point_teacher_cache"] == "output/teacher_cache/{scene}.pt"
    assert payload["direct_point_feature_key"] == "semantic"
    assert payload["direct_point_candidate_k"] == 32
    assert payload["direct_point_summary_alignment_weight"] == 0.4
    assert payload["direct_point_text_loss_weight"] == 0.7
    assert payload["direct_point_adapter_text_loss_weight"] == 0.2
    assert payload["direct_point_adapter_text_distill_weight"] == 0.9
    assert payload["direct_point_text_embeddings"] == "checkpoints/siglip2_scannet_og_text_embeddings_ens5.pt"
    assert payload["direct_point_text_split"] == "19"
    assert payload["direct_point_text_temperature"] == 0.05
    assert payload["direct_point_text_ce_weighting"] == "sqrt_inverse_pool_capped"
    assert payload["direct_point_text_ce_min_weight"] == 0.8
    assert payload["direct_point_text_ce_max_weight"] == 2.0
    assert payload["direct_point_text_distill_weight"] == 0.8
    assert payload["direct_point_text_distill_temperature"] == 1.5
    assert payload["direct_point_text_distill_confidence_threshold"] == 0.6
    assert payload["direct_point_text_pseudo_ce_weight"] == 0.6
    assert payload["direct_point_text_pseudo_ce_confidence_threshold"] == 0.25
    assert payload["direct_point_text_pseudo_ce_logit_scale"] == 18.0
    assert payload["direct_point_text_pseudo_ce_center_logits"] is True
    assert payload["direct_point_text_pseudo_ce_splits"] == "19,15,10"
    assert payload["direct_point_adapter_text_pseudo_ce_weight"] == 1.1
    assert payload["direct_point_adapter_text_pseudo_ce_confidence_threshold"] == 0.3
    assert payload["direct_point_adapter_text_pseudo_ce_logit_scale"] == 20.0
    assert payload["direct_point_adapter_text_pseudo_ce_center_logits"] is True
    assert payload["direct_point_adapter_text_pseudo_ce_splits"] == "10,15"
    assert payload["direct_point_adapter_decoder_anchor_weight"] == 0.4

    loaded = load_config(str(output_path))
    assert loaded.direct_point_gaussian_position_mode == "label_point"
    assert loaded.direct_point_text_pseudo_ce_weight == 0.6
    assert loaded.direct_point_text_pseudo_ce_center_logits is True
    assert loaded.direct_point_text_pseudo_ce_splits == "19,15,10"
    assert loaded.direct_point_adapter_text_pseudo_ce_weight == 1.1
    assert loaded.direct_point_adapter_text_pseudo_ce_center_logits is True
    assert loaded.direct_point_adapter_text_pseudo_ce_splits == "10,15"
    assert loaded.direct_point_adapter_decoder_anchor_weight == 0.4


def test_cli_accepts_teacher_balanced_direct_point_strategy(tmp_path, monkeypatch):
    scene = "scene0000_00"
    prepared_root = tmp_path / "dataset" / "scannet_og"
    _write_prepared_scene(prepared_root, scene)
    output_root = tmp_path / "configs"
    repo_root = tmp_path / "repo"

    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_scannet_og_configs.py",
            "--scenes",
            scene,
            "--prepared_root",
            str(prepared_root),
            "--output_root",
            str(output_root),
            "--repo_root",
            str(repo_root),
            "--variant",
            "v63_teacher",
            "--direct_point_sample_strategy",
            "teacher_balanced",
        ],
    )

    main()

    output_path = output_root / f"scannet_og_hybrid_v63_teacher_{scene}.yaml"
    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert payload["direct_point_sample_strategy"] == "teacher_balanced"
