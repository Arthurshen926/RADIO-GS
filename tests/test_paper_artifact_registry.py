import json
import sys
from pathlib import Path

from radio_gs.scripts import build_consistency_audit, build_paper_assets_manifest


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _minimal_manifest_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    report_dir = repo / "output" / "radio_gs" / "reports"
    profile_dir = repo / "output" / "radio_gs" / "profiles"
    fig_dir = repo / "output" / "radio_gs" / "paper_figures"
    paper_fig_dir = repo / "paper" / "figures"
    profile_dir.mkdir(parents=True)
    fig_dir.mkdir(parents=True)
    paper_fig_dir.mkdir(parents=True)

    _write(
        report_dir / "paper_submission_main_table.md",
        "\n".join(
            [
                "| Method | A | B | C | D | E | Macro |",
                "|---|---|---|---|---|---|---|",
                "| LERF | 0 | 0 | 0 | 0 | 0 | 0.10 |",
                "| LangSplat | 0 | 0 | 0 | 0 | 0 | 0.20 |",
                "| LEGaussians | 0 | 0 | 0 | 0 | 0 | 0.30 |",
                "| RADIO-GS | 0 | 0 | 0 | 0 | 0 | 0.40 |",
            ]
        ),
    )
    _write(report_dir / "submission_readiness_checklist.md", "")

    monkeypatch.setattr(build_paper_assets_manifest, "REPO_ROOT", repo)
    monkeypatch.setattr(build_paper_assets_manifest, "REPORT_DIR", report_dir)
    monkeypatch.setattr(build_paper_assets_manifest, "FIG_DIR", fig_dir)
    monkeypatch.setattr(build_paper_assets_manifest, "PAPER_FIG_DIR", paper_fig_dir)
    monkeypatch.setattr(build_paper_assets_manifest, "PROFILE_DIR", profile_dir)
    return repo, report_dir


def test_manifest_registers_nearest_view_cache_baseline(monkeypatch, tmp_path):
    _, report_dir = _minimal_manifest_repo(tmp_path, monkeypatch)
    output = report_dir / "manifest.json"
    monkeypatch.setattr(sys, "argv", ["build_paper_assets_manifest.py", "--output", str(output)])

    build_paper_assets_manifest.main()

    manifest = json.loads(output.read_text(encoding="utf-8"))
    tables = manifest["main_tables"]
    assert tables["lerf_nearest_view_cache_baseline.md"]["path"].endswith(
        "output/radio_gs/reports/lerf_nearest_view_cache_baseline.md"
    )
    assert tables["lerf_nearest_view_cache_baseline.json"]["path"].endswith(
        "output/radio_gs/reports/lerf_nearest_view_cache_baseline.json"
    )
    assert tables["paper/lerf_nearest_view_cache_baseline_table.tex"]["path"].endswith(
        "paper/lerf_nearest_view_cache_baseline_table.tex"
    )
    assert tables["lerf_per_gaussian_1280d_baseline.md"]["path"].endswith(
        "output/radio_gs/reports/lerf_per_gaussian_1280d_baseline.md"
    )
    assert tables["lerf_per_gaussian_1280d_baseline.json"]["path"].endswith(
        "output/radio_gs/reports/lerf_per_gaussian_1280d_baseline.json"
    )
    assert tables["paper/lerf_per_gaussian_1280d_baseline_table.tex"]["path"].endswith(
        "paper/lerf_per_gaussian_1280d_baseline_table.tex"
    )
    assert tables["build_lerf_nearest_view_cache_baseline.py"]["path"].endswith(
        "radio_gs/scripts/build_lerf_nearest_view_cache_baseline.py"
    )
    assert tables["build_lerf_per_gaussian_1280d_baseline.py"]["path"].endswith(
        "radio_gs/scripts/build_lerf_per_gaussian_1280d_baseline.py"
    )
    assert tables["teacher_vs_ctfgs_2d_usability_20260525.md"]["path"].endswith(
        "paper/artifacts/teacher_vs_ctfgs_2d_usability_20260525.md"
    )
    assert tables["teacher_vs_ctfgs_2d_usability_20260525.json"]["path"].endswith(
        "paper/artifacts/teacher_vs_ctfgs_2d_usability_20260525.json"
    )
    assert tables["paper/tables/teacher_vs_ctfgs_2d_usability_20260525.tex"][
        "path"
    ].endswith("paper/tables/teacher_vs_ctfgs_2d_usability_20260525.tex")
    assert tables["build_teacher_vs_ctfgs_2d_usability_report.py"][
        "path"
    ].endswith("radio_gs/scripts/build_teacher_vs_ctfgs_2d_usability_report.py")
    assert tables["alpha_depth_boundary_alignment_report.md"]["path"].endswith(
        "output/radio_gs/reports/alpha_depth_boundary_alignment_report.md"
    )
    assert tables["alpha_depth_boundary_alignment_report.json"]["path"].endswith(
        "output/radio_gs/reports/alpha_depth_boundary_alignment_report.json"
    )
    assert tables["lerf_sam3_box_global_threshold_sweep_20260517_geometry.md"]["path"].endswith(
        "output/radio_gs/reports/lerf_sam3_box_global_threshold_sweep_20260517_geometry.md"
    )
    assert tables["lerf_sam3_box_global_threshold_sweep_20260517_geometry.json"]["path"].endswith(
        "output/radio_gs/reports/lerf_sam3_box_global_threshold_sweep_20260517_geometry.json"
    )
    assert tables["paper/alpha_depth_boundary_alignment_table.tex"]["path"].endswith(
        "paper/alpha_depth_boundary_alignment_table.tex"
    )
    assert tables["build_alpha_depth_boundary_alignment_report.py"]["path"].endswith(
        "radio_gs/scripts/build_alpha_depth_boundary_alignment_report.py"
    )
    assert tables["alpha_depth_boundary_case_figure_manifest.md"]["path"].endswith(
        "output/radio_gs/reports/alpha_depth_boundary_case_figure_manifest.md"
    )
    assert tables["alpha_depth_boundary_case_figure_manifest.json"]["path"].endswith(
        "output/radio_gs/reports/alpha_depth_boundary_case_figure_manifest.json"
    )
    assert tables["build_alpha_depth_case_figure.py"]["path"].endswith(
        "radio_gs/scripts/build_alpha_depth_case_figure.py"
    )
    assert tables["train_feature_field_audit.md"]["path"].endswith(
        "output/radio_gs/reports/train_feature_field_audit.md"
    )
    assert tables["train_feature_field_audit.json"]["path"].endswith(
        "output/radio_gs/reports/train_feature_field_audit.json"
    )
    assert tables["paper/train_feature_field_audit_table.tex"]["path"].endswith(
        "paper/train_feature_field_audit_table.tex"
    )
    assert tables["build_train_feature_field_audit.py"]["path"].endswith(
        "radio_gs/scripts/build_train_feature_field_audit.py"
    )
    assert "nearest-view measured" in tables["controlled_baseline_gap_audit.md"]["description"]
    assert "per-Gaussian 1280-D explicit measured" in tables["controlled_baseline_gap_audit.md"]["description"]


def test_consistency_audit_requires_nearest_view_cache_artifacts(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    report_dir = repo / "output" / "radio_gs" / "reports"
    report_dir.mkdir(parents=True)
    _write(report_dir / "paper_assets_manifest.json", json.dumps({"route": "test", "pending": []}))
    _write(report_dir / "submission_freeze_manifest.json", json.dumps({"direct3d_readouts": []}))

    monkeypatch.setattr(build_consistency_audit, "REPO_ROOT", repo)
    monkeypatch.setattr(build_consistency_audit, "REPORT_DIR", report_dir)
    output = report_dir / "audit.md"
    monkeypatch.setattr(sys, "argv", ["build_consistency_audit.py", "--output", str(output)])

    build_consistency_audit.main()

    text = output.read_text(encoding="utf-8")
    assert "lerf_nearest_view_cache_baseline.md" in text
    assert "lerf_nearest_view_cache_baseline.json" in text
    assert "lerf_nearest_view_cache_baseline_table.tex" in text
    assert "build_lerf_nearest_view_cache_baseline.py" in text
    assert "lerf_per_gaussian_1280d_baseline.md" in text
    assert "lerf_per_gaussian_1280d_baseline.json" in text
    assert "lerf_per_gaussian_1280d_baseline_table.tex" in text
    assert "build_lerf_per_gaussian_1280d_baseline.py" in text
    assert "alpha_depth_boundary_alignment_report.md" in text
    assert "alpha_depth_boundary_alignment_report.json" in text
    assert "lerf_sam3_box_global_threshold_sweep_20260517_geometry.md" in text
    assert "lerf_sam3_box_global_threshold_sweep_20260517_geometry.json" in text
    assert "alpha_depth_boundary_alignment_table.tex" in text
    assert "build_alpha_depth_boundary_alignment_report.py" in text
    assert "alpha_depth_boundary_case_figure_manifest.md" in text
    assert "alpha_depth_boundary_case_figure_manifest.json" in text
    assert "build_alpha_depth_case_figure.py" in text
    assert "alpha_depth_boundary_cases.png" in text
    assert "train_feature_field_audit.md" in text
    assert "train_feature_field_audit.json" in text
    assert "train_feature_field_audit_table.tex" in text
    assert "build_train_feature_field_audit.py" in text
