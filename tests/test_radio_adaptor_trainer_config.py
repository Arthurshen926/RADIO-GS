from radio_gs.config import RadioGSConfig
from radio_gs.scripts.train_feature_field import parse_radio_adaptor_names


def test_radio_adaptor_config_defaults_are_disabled():
    cfg = RadioGSConfig()

    assert cfg.radio_adaptor_alignment_weight == 0.0
    assert cfg.radio_adaptor_alignment_names == ""


def test_parse_radio_adaptor_names_deduplicates_and_strips():
    assert parse_radio_adaptor_names("dino_v3, sam3, dino_v3") == ["dino_v3", "sam3"]
