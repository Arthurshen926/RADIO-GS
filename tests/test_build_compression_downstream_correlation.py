import json
from pathlib import Path

from radio_gs.scripts import build_compression_downstream_correlation as corr


def _write_storage(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "| Scene | #Gaussians | Direct 1280-D fp16 | Compact ckpt | Saving |",
                "|---|---:|---:|---:|---:|",
                "| Figurines | 10 | 10.0 MiB | 5.0 MiB | 2.00x |",
                "| Waldo Kitchen | 20 | 20.0 MiB | 5.0 MiB | 4.00x |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_rendered(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "variants": {
                    "0.60": {
                        "rows": [
                            {"scene": "figurines", "miou": 0.4, "loc_acc": 0.8},
                            {"scene": "waldo_kitchen", "miou": 0.2, "loc_acc": 0.7},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _write_direct3d(root: Path) -> None:
    for scene, miou in {"figurines": 0.6, "waldo_kitchen": 0.1}.items():
        scene_dir = root / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "lerf_direct_3d_selection_results.json").write_text(
            json.dumps({"scene": {"scene": scene, "results": {"thr0p25": {"miou": miou, "acc025": 0.5}}}}),
            encoding="utf-8",
        )


def test_build_summary_joins_storage_with_rendered_and_direct3d(tmp_path: Path) -> None:
    storage = tmp_path / "storage.md"
    rendered = tmp_path / "rendered.json"
    direct = tmp_path / "direct"
    _write_storage(storage)
    _write_rendered(rendered)
    _write_direct3d(direct)

    summary = corr.build_summary(storage, rendered, direct3d_root=direct, rendered_variant="0.60")

    assert summary["rows"][0]["scene"] == "Figurines"
    assert summary["rows"][0]["saving_ratio"] == 2.0
    assert summary["rows"][1]["rendered_miou"] == 0.2
    assert summary["correlations"]["saving_vs_rendered_miou"] == -1.0
    assert summary["correlations"]["saving_vs_direct3d_miou"] == -1.0


def test_build_markdown_and_latex_include_correlation_rows(tmp_path: Path) -> None:
    storage = tmp_path / "storage.md"
    rendered = tmp_path / "rendered.json"
    direct = tmp_path / "direct"
    _write_storage(storage)
    _write_rendered(rendered)
    _write_direct3d(direct)
    summary = corr.build_summary(storage, rendered, direct3d_root=direct, rendered_variant="0.60")

    markdown = corr.build_markdown(summary)
    latex = corr.build_latex_table(summary)

    assert "Compression vs Downstream Correlation" in markdown
    assert "| saving ratio vs rendered mIoU | -1.0000 |" in markdown
    assert "\\label{tab:compression_downstream_correlation}" in latex
    assert "Waldo Kitchen & 4.00" in latex


def test_write_outputs_records_all_formats(tmp_path: Path) -> None:
    summary = {
        "storage_source": "storage.md",
        "rendered_source": "rendered.json",
        "direct3d_root": "direct",
        "rendered_variant": "0.60",
        "selection": "thr0p25",
        "rows": [],
        "correlations": {},
    }

    paths = corr.write_outputs(
        summary,
        tmp_path / "report.md",
        tmp_path / "report.json",
        tmp_path / "report.tex",
    )

    assert paths["markdown"].exists()
    assert paths["json"].exists()
    assert paths["latex"].exists()
