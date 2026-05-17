import json
from pathlib import Path

from radio_gs.scripts import build_feature_error_text_relevance_report as report


def _write_log(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"[00:00:00] [Val E{idx * 5:03d}] cos_latent={value:.4f} cos_decoded={value:.4f} psnr=5.00"
        for idx, value in enumerate(values, start=1)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_rendered(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "variants": {
                    "0.60": {
                        "rows": [
                            {"scene": "figurines", "miou": 0.7, "loc": 0.9},
                            {"scene": "waldo_kitchen", "miou": 0.4, "loc": 0.6},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_build_summary_correlates_feature_error_with_text_relevance_error(tmp_path: Path) -> None:
    fig_log = tmp_path / "fig" / "training.log"
    waldo_log = tmp_path / "waldo" / "training.log"
    rendered = tmp_path / "rendered.json"
    _write_log(fig_log, [0.8, 0.9])
    _write_log(waldo_log, [0.5, 0.6])
    _write_rendered(rendered)

    summary = report.build_summary(
        rendered,
        log_paths={"figurines": fig_log, "waldo_kitchen": waldo_log},
        rendered_variant="0.60",
    )

    assert summary["rows"][0]["scene"] == "Figurines"
    assert summary["rows"][0]["best_val_cos_decoded"] == 0.9
    assert summary["rows"][0]["feature_error"] == 0.1
    assert summary["rows"][1]["miou_error"] == 0.6
    assert summary["correlations"]["feature_error_vs_miou_error"] == 1.0
    assert summary["correlations"]["feature_error_vs_loc_error"] == 1.0


def test_build_markdown_and_latex_include_feature_error_rows(tmp_path: Path) -> None:
    fig_log = tmp_path / "fig" / "training.log"
    waldo_log = tmp_path / "waldo" / "training.log"
    rendered = tmp_path / "rendered.json"
    _write_log(fig_log, [0.9])
    _write_log(waldo_log, [0.6])
    _write_rendered(rendered)
    summary = report.build_summary(
        rendered,
        log_paths={"figurines": fig_log, "waldo_kitchen": waldo_log},
        rendered_variant="0.60",
    )

    markdown = report.build_markdown(summary)
    latex = report.build_latex_table(summary)

    assert "Feature Error vs Text Relevance Error" in markdown
    assert "| feature error vs mIoU error | 1.0000 |" in markdown
    assert "\\label{tab:feature_error_text_relevance}" in latex
    assert "Waldo Kitchen" in latex


def test_write_outputs_records_all_formats(tmp_path: Path) -> None:
    summary = {
        "rendered_source": "rendered.json",
        "rendered_variant": "0.60",
        "rows": [],
        "correlations": {},
    }

    paths = report.write_outputs(
        summary,
        tmp_path / "report.md",
        tmp_path / "report.json",
        tmp_path / "report.tex",
    )

    assert paths["markdown"].exists()
    assert paths["json"].exists()
    assert paths["latex"].exists()
