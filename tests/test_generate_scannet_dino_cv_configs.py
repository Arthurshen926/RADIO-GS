from pathlib import Path

import yaml

from radio_gs.scripts import generate_scannet_dino_cv_configs as gen


def test_generate_config_preserves_v67_protocol_and_adds_dino_cv(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    scene = "scene0000_00"
    base_path = base_dir / f"scannet_og_hybrid_{gen.BASE_VARIANT}_{scene}.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "exp_name": f"radio_gs_scannet_og_{scene}_{gen.BASE_VARIANT}",
                "output_dir": f"/repo/output/radio_gs/scannet_og_{scene}_{gen.BASE_VARIANT}",
                "batch_size": 4,
                "epochs": 20,
                "direct_point_sample_count": 32768,
                "direct_point_query_mode": "gaussian_index",
                "direct_point_gaussian_position_mode": "label_point",
                "direct_point_source": "label_ply",
                "direct_point_teacher_cache": "output/scannet_teacher_cache_norm/{scene}_radio_teacher_features.pt",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "generated"

    output = gen.generate_config(
        scene=scene,
        base_config_dir=base_dir,
        output_config_dir=out_dir,
        repo_root=Path("/repo"),
        variant="v67_dino_cv001_b2_s32768_ft20",
        cross_view_weight=0.001,
    )

    cfg = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert cfg["batch_size"] == 2
    assert cfg["direct_point_sample_count"] == 32768
    assert cfg["direct_point_query_mode"] == "gaussian_index"
    assert cfg["direct_point_gaussian_position_mode"] == "label_point"
    assert cfg["radio_adaptor_cross_view_names"] == "dino_v3"
    assert cfg["radio_adaptor_cross_view_weight"] == 0.001
    assert cfg["direct_point_view_count_weighting"] == "clipped_log"
    assert cfg["direct_point_text_contrast_weight"] == 0.05
    assert cfg["direct_point_text_contrast_pair_weighting"] == "visibility"
    assert cfg["direct_point_text_contrast_max_points"] == 4096
    assert cfg["direct_point_text_contrast_center_logits"] is False
    assert cfg["direct_point_render_consistency_weight"] == 0.05
    assert cfg["direct_point_render_consistency_mode"] == "cosine"
    assert cfg["direct_point_cached_visible_fraction"] == 0.5
    assert cfg["direct_point_cached_visible_candidate_multiplier"] == 1
    assert cfg["direct_point_cached_visible_balance"] is False
    assert cfg["warmstart_from"].endswith(f"scannet_og_{scene}_{gen.BASE_VARIANT}/checkpoints/best.pth")
    assert "v67_dino_cv001_b2_s32768_ft20" in cfg["output_dir"]


def test_scene_from_config_path_preserves_full_scene_id() -> None:
    path = Path(
        "radio_gs/configs/generated/scannet_dino_cv/"
        "scannet_og_hybrid_v67_dino_cv001_b2_s32768_ft20_scene0645_00.yaml"
    )

    assert gen.scene_from_config_path(path, "v67_dino_cv001_b2_s32768_ft20") == "scene0645_00"
