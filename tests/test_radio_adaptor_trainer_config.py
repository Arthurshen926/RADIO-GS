from radio_gs.config import RadioGSConfig, load_config
from radio_gs.scripts.train_feature_field import merge_radio_adaptor_names
from radio_gs.scripts.train_feature_field import parse_radio_adaptor_names


def test_radio_adaptor_config_defaults_are_disabled():
    cfg = RadioGSConfig()

    assert cfg.radio_adaptor_alignment_weight == 0.0
    assert cfg.radio_adaptor_alignment_names == ""
    assert cfg.radio_adaptor_relation_weight == 0.0
    assert cfg.radio_adaptor_relation_names == ""
    assert cfg.radio_adaptor_local_affinity_weight == 0.0
    assert cfg.radio_adaptor_local_affinity_names == ""
    assert cfg.radio_adaptor_local_affinity_downsample == 1
    assert cfg.radio_adaptor_local_affinity_radius == 1
    assert cfg.radio_adaptor_region_weight == 0.0
    assert cfg.radio_adaptor_region_names == ""
    assert cfg.radio_adaptor_mask_logit_weight == 0.0
    assert cfg.radio_adaptor_mask_logit_names == ""
    assert cfg.radio_adaptor_cross_view_weight == 0.0
    assert cfg.radio_adaptor_cross_view_names == ""
    assert cfg.radio_adaptor_cross_view_objective == "mse"
    assert cfg.radio_adaptor_cross_view_propagation_weight == 0.0
    assert cfg.radio_adaptor_cross_view_propagation_anchor_strategy == "linspace"
    assert cfg.point_summary_adapter_context_features == ""
    assert cfg.direct_point_query_logit_distill_weight == 0.0
    assert cfg.radio_adaptor_cross_view_propagation_names == ""
    assert cfg.radio_adaptor_cross_view_mask_propagation_weight == 0.0
    assert cfg.radio_adaptor_cross_view_mask_propagation_names == ""
    assert cfg.radio_adaptor_cross_view_mask_propagation_anchor_strategy == "linspace"
    assert cfg.radio_adaptor_token_contrast_weight == 0.0
    assert cfg.radio_adaptor_token_contrast_names == ""
    assert cfg.radio_adaptor_peak_background_weight == 0.0
    assert cfg.radio_adaptor_peak_background_names == ""
    assert cfg.radio_adaptor_peak_background_anchor_strategy == "linspace"
    assert cfg.text_heatmap_distill_weight == 0.0
    assert cfg.text_heatmap_distill_embeddings == ""
    assert cfg.text_heatmap_distill_downsample == 2
    assert cfg.text_heatmap_distill_temperature == 20.0
    assert cfg.text_heatmap_distill_mode == "query"
    assert cfg.foundation_cache_mask_projector_hidden_dim == 256
    assert cfg.foundation_cache_mask_projector_masks == 32
    assert cfg.foundation_cache_region_consistency_weight == 0.0
    assert cfg.foundation_cache_region_separation_weight == 0.0
    assert cfg.foundation_cache_feature_boundary_weight == 0.0
    assert cfg.foundation_cache_region_score_threshold == 0.0
    assert cfg.foundation_cache_region_max_masks == 16
    assert cfg.foundation_cache_region_separation_margin == 0.25


def test_parse_radio_adaptor_names_deduplicates_and_strips():
    assert parse_radio_adaptor_names("dino_v3, sam3, dino_v3") == ["dino_v3", "sam3"]


def test_merge_radio_adaptor_names_preserves_first_seen_order():
    assert merge_radio_adaptor_names(["dino_v3", "sam3"], ["sam3"], ["dino_v3", "siglip2-g"]) == [
        "dino_v3",
        "sam3",
        "siglip2-g",
    ]


def test_load_config_supports_relative_base_config_overlay(tmp_path):
    base = tmp_path / "base.yaml"
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    child = child_dir / "overlay.yaml"
    base.write_text(
        "scene: figurines\n"
        "epochs: 240\n"
        "foundation_cache_weight: 0.0\n",
        encoding="utf-8",
    )
    child.write_text(
        "base_config: ../base.yaml\n"
        "epochs: 3\n"
        "foundation_cache_weight: 1.0\n"
        "foundation_cache_heads: sam3\n"
        "foundation_cache_region_consistency_weight: 0.05\n",
        encoding="utf-8",
    )

    cfg = load_config(str(child))

    assert cfg.scene == "figurines"
    assert cfg.epochs == 3
    assert cfg.foundation_cache_weight == 1.0
    assert cfg.foundation_cache_heads == "sam3"
    assert cfg.foundation_cache_region_consistency_weight == 0.05
