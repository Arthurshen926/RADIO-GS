import pytest

from radio_gs.scripts.build_scannet_vala8_report import _assert_compatible_protocols, build_vala8_summary


def _scene(miou19, miou15, miou10):
    return {
        "splits": {
            "19": {"miou": miou19, "macc": 0.6},
            "15": {"miou": miou15, "macc": 0.7},
            "10": {"miou": miou10, "macc": 0.8},
        }
    }


def _scene_with_classes(class_iou):
    return {
        "splits": {
            "19": {
                "miou": class_iou,
                "macc": 0.6,
                "per_class": {
                    "1": {
                        "name": "wall",
                        "iou": class_iou,
                        "acc": 0.5,
                        "gt_count": 10,
                    },
                    "2": {
                        "name": "floor",
                        "iou": 0.5,
                        "acc": 0.7,
                        "gt_count": 20,
                    },
                },
            },
            "15": {"miou": class_iou, "macc": 0.7, "per_class": {}},
            "10": {"miou": class_iou, "macc": 0.8, "per_class": {}},
        }
    }


def test_build_vala8_summary_filters_fixed_scene_subset():
    payload = {
        "timestamp": "now",
        "args": {"query_mode": "gaussian_index"},
        "scenes": {
            "scene0000_00": _scene(0.2, 0.3, 0.4),
            "scene0062_00": _scene(0.4, 0.5, 0.6),
            "scene0200_00": _scene(0.9, 0.9, 0.9),
        },
    }

    summary = build_vala8_summary(
        payload,
        scenes=("scene0000_00", "scene0062_00"),
        label="test",
    )

    assert summary["scene_count"] == 2
    assert summary["scenes"] == ["scene0000_00", "scene0062_00"]
    assert summary["macro"]["19"]["miou"] == 0.3
    assert summary["macro"]["15"]["miou"] == 0.4
    assert summary["macro"]["10"]["miou"] == 0.5
    assert summary["source_args"]["query_mode"] == "gaussian_index"
    assert "_per_class" not in summary["rows"][0]


def test_build_vala8_summary_requires_all_split_scenes():
    with pytest.raises(KeyError, match="scene0062_00"):
        build_vala8_summary(
            {"scenes": {"scene0000_00": _scene(0.2, 0.3, 0.4)}},
            scenes=("scene0000_00", "scene0062_00"),
            label="test",
        )


def test_build_vala8_summary_can_require_exact_scene_set():
    payload = {
        "scenes": {
            "scene0000_00": _scene(0.2, 0.3, 0.4),
            "scene0062_00": _scene(0.4, 0.5, 0.6),
            "scene0200_00": _scene(0.9, 0.9, 0.9),
        }
    }

    with pytest.raises(ValueError, match="scene0200_00"):
        build_vala8_summary(
            payload,
            scenes=("scene0000_00", "scene0062_00"),
            label="test",
            require_exact_scene_set=True,
        )


def test_build_vala8_summary_checks_expected_source_args():
    payload = {
        "args": {"query_mode": "gaussian_index", "class_splits": "19,15,10"},
        "scenes": {
            "scene0000_00": _scene(0.2, 0.3, 0.4),
            "scene0062_00": _scene(0.4, 0.5, 0.6),
        },
    }

    summary = build_vala8_summary(
        payload,
        scenes=("scene0000_00", "scene0062_00"),
        label="test",
        expected_source_args={"query_mode": "gaussian_index"},
    )

    assert summary["source_protocol_checks"]["expected_source_args"] == {
        "query_mode": "gaussian_index",
    }

    with pytest.raises(ValueError, match="query_mode"):
        build_vala8_summary(
            payload,
            scenes=("scene0000_00", "scene0062_00"),
            label="test",
            expected_source_args={"query_mode": "knn"},
        )


def test_build_vala8_summary_reports_category_macro_stability():
    payload = {
        "scenes": {
            "scene0000_00": _scene_with_classes(0.2),
            "scene0062_00": _scene_with_classes(0.6),
        }
    }

    summary = build_vala8_summary(
        payload,
        scenes=("scene0000_00", "scene0062_00"),
        label="test",
    )

    split19 = summary["category_macro_stability"]["19"]
    assert split19["classes"]["1"]["mean_iou"] == 0.4
    assert split19["classes"]["1"]["std_iou"] == 0.2
    assert split19["classes"]["1"]["scene_count"] == 2
    assert split19["most_unstable_iou_class"]["name"] == "wall"
    assert split19["worst_mean_iou_class"]["name"] == "wall"


def test_multi_input_protocol_check_canonicalizes_scene_paths():
    first = {
        "args": {
            "query_mode": "knn",
            "checkpoint": "output/run_scene0000_00/checkpoints/best.pth",
        }
    }
    second = {
        "args": {
            "query_mode": "knn",
            "checkpoint": "output/run_scene0062_00/checkpoints/best.pth",
        }
    }

    signature = _assert_compatible_protocols([first, second])

    assert signature["checkpoint"] == "output/run_{scene}/checkpoints/best.pth"


def test_multi_input_protocol_check_rejects_mixed_query_modes():
    with pytest.raises(ValueError, match="query_mode"):
        _assert_compatible_protocols(
            [
                {"args": {"query_mode": "gaussian_index"}},
                {"args": {"query_mode": "knn"}},
            ]
        )
