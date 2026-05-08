from radio_gs.config import RadioGSConfig
from radio_gs.scripts.train_feature_field import merge_radio_adaptor_names
from radio_gs.scripts.train_feature_field import parse_radio_adaptor_names


def test_radio_adaptor_config_defaults_are_disabled():
    cfg = RadioGSConfig()

    assert cfg.radio_adaptor_alignment_weight == 0.0
    assert cfg.radio_adaptor_alignment_names == ""
    assert cfg.radio_adaptor_relation_weight == 0.0
    assert cfg.radio_adaptor_relation_names == ""
    assert cfg.radio_adaptor_region_weight == 0.0
    assert cfg.radio_adaptor_region_names == ""
    assert cfg.radio_adaptor_cross_view_weight == 0.0
    assert cfg.radio_adaptor_cross_view_names == ""
    assert cfg.text_heatmap_distill_weight == 0.0
    assert cfg.text_heatmap_distill_embeddings == ""
    assert cfg.text_heatmap_distill_downsample == 2
    assert cfg.text_heatmap_distill_temperature == 20.0
    assert cfg.text_heatmap_distill_mode == "query"


def test_parse_radio_adaptor_names_deduplicates_and_strips():
    assert parse_radio_adaptor_names("dino_v3, sam3, dino_v3") == ["dino_v3", "sam3"]


def test_merge_radio_adaptor_names_preserves_first_seen_order():
    assert merge_radio_adaptor_names(["dino_v3", "sam3"], ["sam3"], ["dino_v3", "siglip2-g"]) == [
        "dino_v3",
        "sam3",
        "siglip2-g",
    ]
