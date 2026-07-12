import yaml

from radio_gs.config import config_to_dict, load_config


def test_samclip_level_and_language_dir_survive_config_resolution(tmp_path):
    config_path = tmp_path / "samclip_l2.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "samclip_feature_level": 2,
                "samclip_language_feature_dir": "/tmp/samclip/figurines/l2",
            }
        ),
        encoding="utf-8",
    )

    resolved = config_to_dict(load_config(str(config_path)))

    assert resolved["samclip_feature_level"] == 2
    assert resolved["samclip_language_feature_dir"] == "/tmp/samclip/figurines/l2"
