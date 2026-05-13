from radio_gs.scripts import build_lerf_direct_3d_selection_table as table


def test_selection_tags_sorts_top_and_meanstd_tags_stably():
    results = {
        scene: {
            "results": {
                "meanstd2p5": {"miou": 0.2, "acc025": 0.3},
                "top0p02": {"miou": 0.1, "acc025": 0.2},
                "meanstd1": {"miou": 0.3, "acc025": 0.4},
            }
        }
        for scene in table.SCENES
    }

    assert table.selection_tags(results) == ["top0p02", "meanstd1", "meanstd2p5"]


def test_best_fixed_tag_supports_non_top_selection_tags():
    results = {
        scene: {
            "results": {
                "top0p02": {"miou": 0.1, "acc025": 0.2},
                "meanstd2p5": {"miou": 0.4, "acc025": 0.5},
            }
        }
        for scene in table.SCENES
    }

    assert table.best_fixed_tag(results) == "meanstd2p5"


def test_direct_protocol_sentence_mentions_selection_bounds():
    results = {
        scene: {
            "results": {"meanstd2p5": {"miou": 0.4, "acc025": 0.5}},
            "_protocol": {
                "score_source": "registered_view",
                "registration_frame_mode": "all_poses",
                "registration_max_frames": 96,
                "score_aggregation": "voxel_max",
                "score_aggregation_resolution": 80,
                "score_aggregation_blend": 0.5,
                "selection_min_ratio": 0.0,
                "selection_max_ratio": 0.02,
            },
            "_args": {"scoring": "softmax_scene"},
        }
        for scene in table.SCENES
    }

    sentence = table.direct_protocol_sentence(results)
    caption_source = table.direct_caption_feature_source(results)

    assert "selection cap=0.02" in sentence
    assert "selection cap=0.02" in caption_source
