import json

from radio_gs.scripts.aggregate_lerf_sam_dino_tasks import load_scene_results


def _scene_report(scene, n_samples, teacher_iou, rendered_iou, n_matches=5):
    return {
        "protocol": {"dino_v3": "test"},
        "scenes": {
            scene: {
                "sam3": {
                    task: {
                        "teacher": {"loc_acc": 0.5, "miou": teacher_iou, "n_samples": n_samples},
                        "rendered": {"loc_acc": 1.0, "miou": rendered_iou, "n_samples": n_samples},
                    }
                    for task in (
                        "point_prompt_segmentation",
                        "box_prompt_segmentation",
                        "mask_prompt_propagation",
                    )
                },
                "dino_v3": {
                    "dense_matching": {
                        "teacher": {"hit_rate": 0.2, "mean_score": 0.4, "n_matches": n_matches},
                        "rendered": {"hit_rate": 0.6, "mean_score": 0.8, "n_matches": n_matches},
                    },
                    "mask_propagation": {
                        "teacher": {"loc_acc": 0.5, "miou": teacher_iou, "n_samples": n_samples},
                        "rendered": {"loc_acc": 1.0, "miou": rendered_iou, "n_samples": n_samples},
                    },
                },
                "visualizations": [],
            }
        },
        "macro": {},
    }


def test_load_scene_results_uses_sample_weighted_aggregation(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_scene_report("a", 1, 0.0, 1.0)), encoding="utf-8")
    second.write_text(json.dumps(_scene_report("b", 3, 1.0, 0.0)), encoding="utf-8")

    report = load_scene_results([first, second])

    sam_point = report["macro"]["sam3"]["point_prompt_segmentation"]
    dino_mask = report["macro"]["dino_v3"]["mask_propagation"]
    assert sam_point["teacher"]["miou"] == 0.75
    assert sam_point["rendered"]["miou"] == 0.25
    assert dino_mask["teacher"]["n_samples"] == 4
    assert dino_mask["rendered"]["loc_acc"] == 1.0
