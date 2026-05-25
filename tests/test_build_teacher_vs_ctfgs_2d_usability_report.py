import json
from pathlib import Path

from radio_gs.scripts import build_teacher_vs_ctfgs_2d_usability_report as report


def test_build_report_marks_selected_task_wins_and_caveats(tmp_path: Path) -> None:
    controlled = tmp_path / "controlled.json"
    controlled.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "method": "Frame-wise RADIO teacher",
                        "lerf_loc_acc": 0.80,
                        "lerf_miou": 0.46,
                    },
                    {
                        "method": "Nearest-view RADIO cache",
                        "lerf_loc_acc": 0.27,
                        "lerf_miou": 0.15,
                    },
                    {
                        "method": "Per-Gaussian 1280-D RADIO memory",
                        "lerf_loc_acc": 0.56,
                        "lerf_miou": 0.32,
                    },
                    {
                        "method": "Full CTF-GS",
                        "lerf_loc_acc": 0.87,
                        "lerf_miou": 0.52,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    sam_dino = tmp_path / "sam_dino.json"
    sam_dino.write_text(
        json.dumps(
            {
                "macro": {
                    "sam3": {
                        "point_prompt_segmentation": {
                            "teacher": {"loc_acc": 1.0, "miou": 0.37, "n_samples": 10},
                            "rendered": {"loc_acc": 1.0, "miou": 0.42, "n_samples": 10},
                        },
                        "box_prompt_segmentation": {
                            "teacher": {"loc_acc": 0.87, "miou": 0.66, "n_samples": 10},
                            "rendered": {"loc_acc": 0.82, "miou": 0.67, "n_samples": 10},
                        },
                        "mask_prompt_propagation": {
                            "teacher": {"loc_acc": 0.79, "miou": 0.36, "n_samples": 10},
                            "rendered": {"loc_acc": 0.67, "miou": 0.38, "n_samples": 10},
                        },
                    },
                    "dino_v3": {
                        "dense_matching": {
                            "teacher": {
                                "hit_rate": 0.57,
                                "mean_score": 0.85,
                                "n_matches": 30,
                            },
                            "rendered": {
                                "hit_rate": 0.54,
                                "mean_score": 0.90,
                                "n_matches": 30,
                            },
                        },
                        "mask_propagation": {
                            "teacher": {"loc_acc": 0.76, "miou": 0.51, "n_samples": 10},
                            "rendered": {"loc_acc": 0.79, "miou": 0.48, "n_samples": 10},
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    built = report.build_report(controlled, sam_dino)

    assert built["summary"]["primary_rendered_wins"] == 5
    assert built["summary"]["primary_total"] == 6
    assert built["summary"]["universal_superiority"] is False
    assert any(
        "DINOv3 mask propagation mIoU" in item
        for item in built["summary"]["caveats"]
    )


def test_write_report_outputs_markdown_json_and_latex(tmp_path: Path) -> None:
    built = {
        "text_grounding_rows": [
            {
                "method": "Frame-wise RADIO teacher",
                "loc_acc": 0.8,
                "miou": 0.46,
                "delta_loc_acc": 0.0,
                "delta_miou": 0.0,
            },
            {
                "method": "Full CTF-GS",
                "loc_acc": 0.87,
                "miou": 0.52,
                "delta_loc_acc": 0.07,
                "delta_miou": 0.06,
            },
        ],
        "frozen_head_rows": [
            {
                "task": "SAM3 point prompt",
                "primary_metric": "mIoU",
                "teacher_primary": 0.37,
                "rendered_primary": 0.42,
                "delta_primary": 0.05,
                "secondary_metric": "LocAcc",
                "teacher_secondary": 1.0,
                "rendered_secondary": 1.0,
                "delta_secondary": 0.0,
                "n": 10,
                "winner": "rendered",
            },
            {
                "task": "DINOv3 mask propagation",
                "primary_metric": "mIoU",
                "teacher_primary": 0.51,
                "rendered_primary": 0.48,
                "delta_primary": -0.03,
                "secondary_metric": "LocAcc",
                "teacher_secondary": 0.76,
                "rendered_secondary": 0.79,
                "delta_secondary": 0.03,
                "n": 10,
                "winner": "teacher",
            },
        ],
        "summary": {
            "primary_rendered_wins": 1,
            "primary_total": 2,
            "universal_superiority": False,
            "caveats": ["DINOv3 mask propagation mIoU remains teacher-stronger."],
        },
        "sources": {},
    }

    report.write_outputs(
        built,
        tmp_path / "out.json",
        tmp_path / "out.md",
        tmp_path / "out.tex",
    )

    assert "selected downstream tasks" in (tmp_path / "out.md").read_text(
        encoding="utf-8"
    )
    assert "DINOv3 mask propagation" in (tmp_path / "out.tex").read_text(
        encoding="utf-8"
    )
    assert (
        json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["summary"][
            "universal_superiority"
        ]
        is False
    )


def test_build_report_accepts_optional_prototype_schema(tmp_path: Path) -> None:
    controlled = tmp_path / "controlled.json"
    controlled.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "method": "Frame-wise RADIO teacher",
                        "lerf_loc_acc": 0.80,
                        "lerf_miou": 0.46,
                    },
                    {
                        "method": "Nearest-view RADIO cache",
                        "lerf_loc_acc": 0.27,
                        "lerf_miou": 0.15,
                    },
                    {
                        "method": "Per-Gaussian 1280-D RADIO memory",
                        "lerf_loc_acc": 0.56,
                        "lerf_miou": 0.32,
                    },
                    {
                        "method": "Full CTF-GS",
                        "lerf_loc_acc": 0.87,
                        "lerf_miou": 0.52,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    sam_dino = tmp_path / "sam_dino.json"
    sam_dino.write_text(
        json.dumps(
            {
                "macro": {
                    "sam3": {
                        "point_prompt_segmentation": {
                            "teacher": {"loc_acc": 1.0, "miou": 0.37, "n_samples": 10},
                            "rendered": {"loc_acc": 1.0, "miou": 0.42, "n_samples": 10},
                        },
                        "box_prompt_segmentation": {
                            "teacher": {"loc_acc": 0.87, "miou": 0.66, "n_samples": 10},
                            "rendered": {"loc_acc": 0.82, "miou": 0.67, "n_samples": 10},
                        },
                        "mask_prompt_propagation": {
                            "teacher": {"loc_acc": 0.79, "miou": 0.36, "n_samples": 10},
                            "rendered": {"loc_acc": 0.67, "miou": 0.38, "n_samples": 10},
                        },
                    },
                    "dino_v3": {
                        "dense_matching": {
                            "teacher": {"hit_rate": 0.57, "mean_score": 0.85, "n_matches": 30},
                            "rendered": {"hit_rate": 0.54, "mean_score": 0.90, "n_matches": 30},
                        },
                        "mask_propagation": {
                            "teacher": {"loc_acc": 0.76, "miou": 0.51, "n_samples": 10},
                            "rendered": {"loc_acc": 0.79, "miou": 0.48, "n_samples": 10},
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    prototype = tmp_path / "prototype.json"
    prototype.write_text(
        json.dumps(
            {
                "macro": {
                    "sam3": {
                        "prototype_segmentation": {
                            "teacher": {
                                "loc_acc": 0.84,
                                "miou": 0.08,
                                "n_iou_samples": 20,
                            },
                            "rendered": {
                                "loc_acc": 0.66,
                                "miou": 0.06,
                                "n_iou_samples": 20,
                            },
                        },
                    },
                    "dino_v3": {
                        "source_target_matching": {
                            "teacher": {
                                "loc_acc": 0.60,
                                "miou": 0.10,
                                "n_iou_samples": 15,
                            },
                            "rendered": {
                                "loc_acc": 0.50,
                                "miou": 0.10,
                                "n_iou_samples": 15,
                            },
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    built = report.build_report(controlled, sam_dino, prototype_path=prototype)

    appendix = built["diagnostic_rows"]["prototype_adaptor"]
    assert [row["task"] for row in appendix] == [
        "SAM3 prototype segmentation",
        "DINOv3 source-target matching",
    ]
