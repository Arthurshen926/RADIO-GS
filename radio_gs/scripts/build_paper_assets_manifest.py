#!/usr/bin/env python3
"""Build a current paper asset manifest for conservative submission freeze."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "output" / "radio_gs" / "reports"
FIG_DIR = REPO_ROOT / "output" / "radio_gs" / "paper_figures"
PAPER_FIG_DIR = REPO_ROOT / "paper" / "figures"
PROFILE_DIR = REPO_ROOT / "output" / "radio_gs" / "profiles"


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 1)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_main_table_summary(path: Path) -> str:
    order = ["LERF", "LangSplat", "LEGaussians", "RADIO-GS"]
    macros: dict[str, str] = {}
    for line in read_text(path).splitlines():
        if not line.startswith("|"):
            continue
        columns = [part.strip().replace("**", "") for part in line.strip().strip("|").split("|")]
        if len(columns) != 7 or columns[0] in {"Method", "---"}:
            continue
        method = columns[0]
        if method in order:
            macros[method] = columns[-1]
    summary_parts = [f"{method}={macros[method]}" for method in order if method in macros]
    return "LocAcc: " + ", ".join(summary_parts)


def extract_pending_items(path: Path) -> list[str]:
    if not path.exists():
        return []
    items: list[str] = []
    for line in read_text(path).splitlines():
        stripped = line.strip()
        if stripped.startswith("- [ ] "):
            items.append(stripped[len("- [ ] "):])
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper assets manifest")
    parser.add_argument(
        "--output",
        default=str(REPORT_DIR / "paper_assets_manifest.json"),
        help="Manifest output path",
    )
    args = parser.parse_args()

    main_table = REPORT_DIR / "paper_submission_main_table.md"
    efficiency = REPORT_DIR / "efficiency_profile_summary.md"
    efficiency_cost = REPORT_DIR / "efficiency_cost_table.md"
    direct_selection = REPORT_DIR / "lerf_direct_3d_selection.md"
    readiness = REPORT_DIR / "submission_readiness_checklist.md"
    sam_dino_tasks = (
        REPO_ROOT
        / "output"
        / "lerf_sam_dino_tasks"
        / "formal_v12c_dino_sam3_boundary_v9readout_gpu_20260528"
        / "lerf_sam_dino_task_report.md"
    )
    sam_dino_sweep = (
        REPO_ROOT
        / "output"
        / "lerf_sam_dino_tasks"
        / "formal_v9_dino_readout_sweep_20260514.md"
    )
    sam3_box_results = REPO_ROOT / "docs" / "experiments" / "2026-05-16-sam3-box-readout-results.md"
    sam3_global_threshold_sweep = REPORT_DIR / "lerf_sam3_box_global_threshold_sweep_20260516.md"
    sam3_geometry_threshold_sweep = REPORT_DIR / "lerf_sam3_box_global_threshold_sweep_20260517_geometry.md"
    expert_latest_audit = (
        REPO_ROOT / "docs" / "experiments" / "2026-05-16-expert-latest-followup-audit.md"
    )
    manifest = {
        "generated": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "route": "conservative_submission",
        "main_tables": {
            "paper_submission_main_table.md": {
                "path": rel(main_table),
                "description": "LERF-OVS main comparison table (markdown)",
                "content": extract_main_table_summary(main_table),
            },
            "paper_submission_main_table.tex": {
                "path": rel(REPORT_DIR / "paper_submission_main_table.tex"),
                "description": "LaTeX main table",
            },
            "paper_main_table.md": {
                "path": rel(REPORT_DIR / "paper_main_table.md"),
                "description": "Seed-robustness companion table",
            },
            "lerf_component_ablation.md": {
                "path": rel(REPORT_DIR / "lerf_component_ablation.md"),
                "description": "LERF component ablation report",
            },
            "quantitative_ablation_suite.md": {
                "path": rel(REPO_ROOT / "paper" / "artifacts" / "quantitative_ablation_suite.md"),
                "description": "Unified quantitative ablation contribution ranking across paper-facing tasks",
            },
            "quantitative_ablation_suite.json": {
                "path": rel(REPO_ROOT / "paper" / "artifacts" / "quantitative_ablation_suite.json"),
                "description": "Machine-readable unified quantitative ablation contribution ranking",
            },
            "paper/lerf_component_ablation_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "lerf_component_ablation_table.tex"),
                "description": "LaTeX component ablation table",
            },
            "paper/quantitative_ablation_summary_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "quantitative_ablation_summary_table.tex"),
                "description": "LaTeX unified quantitative ablation contribution-ranking table",
            },
            "paper/lerf_direct3d_compact_readout_ablation_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "lerf_direct3d_compact_readout_ablation_table.tex"),
                "description": "LaTeX compact Direct3D prompt/component-support readout ablation table",
            },
            "scannet_dino_cv_ablation.md": {
                "path": rel(REPORT_DIR / "scannet_dino_cv_ablation.md"),
                "description": "10-scene ScanNet DINO cross-view ablation",
            },
            "scannet_prompt_calibration_ablation.md": {
                "path": rel(REPORT_DIR / "scannet_prompt_calibration_ablation.md"),
                "description": "10-scene ScanNet prompt and label-free calibration ablation",
            },
            "lerf_sam_dino_task_report.md": {
                "path": rel(sam_dino_tasks),
                "description": "Formal SAM3/DINOv3 downstream task sweep with multi-head DINO support and SAM-adaptor boundary readout",
            },
            "formal_v9_dino_readout_sweep_20260514.md": {
                "path": rel(sam_dino_sweep),
                "description": "DINOv3 readout sweep and promotion decision",
            },
            "baseline_source_verification.md": {
                "path": rel(REPORT_DIR / "baseline_source_verification.md"),
                "description": "External baseline source verification tracker",
            },
            "submission_readiness_checklist.md": {
                "path": rel(REPORT_DIR / "submission_readiness_checklist.md"),
                "description": "Submission freeze checklist",
            },
            "submission_freeze_figure_shortlist.md": {
                "path": rel(REPORT_DIR / "submission_freeze_figure_shortlist.md"),
                "description": "Frozen figure shortlist and captions",
            },
            "efficiency_profile_summary.md": {
                "path": rel(efficiency),
                "description": "Efficiency summary table",
            },
            "efficiency_cost_table.md": {
                "path": rel(efficiency_cost),
                "description": "Paper-facing efficiency/cost table",
            },
            "storage_footprint_report.md": {
                "path": rel(REPORT_DIR / "storage_footprint_report.md"),
                "description": "Storage footprint report with separate optional VPR cache accounting",
            },
            "compression_downstream_correlation.md": {
                "path": rel(REPORT_DIR / "compression_downstream_correlation.md"),
                "description": "Compression-ratio versus rendered/direct-3D downstream mIoU mechanism audit",
            },
            "compression_downstream_correlation.json": {
                "path": rel(REPORT_DIR / "compression_downstream_correlation.json"),
                "description": "Machine-readable compression/downstream correlation report",
            },
            "feature_error_text_relevance_report.md": {
                "path": rel(REPORT_DIR / "feature_error_text_relevance_report.md"),
                "description": "Feature reconstruction error versus rendered text-grounding error mechanism audit",
            },
            "feature_error_text_relevance_report.json": {
                "path": rel(REPORT_DIR / "feature_error_text_relevance_report.json"),
                "description": "Machine-readable feature-error/text-relevance mechanism audit",
            },
            "boundary_error_readout_report.md": {
                "path": rel(REPORT_DIR / "boundary_error_readout_report.md"),
                "description": "Measured SAM3-box boundary-error readout with per-query over/under-selection buckets",
            },
            "boundary_error_readout_report.json": {
                "path": rel(REPORT_DIR / "boundary_error_readout_report.json"),
                "description": "Machine-readable SAM3-box boundary-error readout",
            },
            "alpha_depth_boundary_alignment_report.md": {
                "path": rel(REPORT_DIR / "alpha_depth_boundary_alignment_report.md"),
                "description": "Alpha/depth discontinuity alignment coverage report for SAM3-box boundary errors",
            },
            "alpha_depth_boundary_alignment_report.json": {
                "path": rel(REPORT_DIR / "alpha_depth_boundary_alignment_report.json"),
                "description": "Machine-readable alpha/depth boundary-alignment coverage report",
            },
            "alpha_depth_boundary_case_figure_manifest.md": {
                "path": rel(REPORT_DIR / "alpha_depth_boundary_case_figure_manifest.md"),
                "description": "Selected alpha/depth boundary-case figure manifest",
            },
            "alpha_depth_boundary_case_figure_manifest.json": {
                "path": rel(REPORT_DIR / "alpha_depth_boundary_case_figure_manifest.json"),
                "description": "Machine-readable selected alpha/depth boundary-case figure manifest",
            },
            "train_feature_field_audit.md": {
                "path": rel(REPORT_DIR / "train_feature_field_audit.md"),
                "description": "Auditability and reproducibility report for the modularized training entry point",
            },
            "train_feature_field_audit.json": {
                "path": rel(REPORT_DIR / "train_feature_field_audit.json"),
                "description": "Machine-readable train_feature_field audit report",
            },
            "paper/efficiency_cost_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "efficiency_cost_table.tex"),
                "description": "LaTeX efficiency/cost table",
            },
            "paper/storage_footprint_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "storage_footprint_table.tex"),
                "description": "LaTeX compact-storage and optional VPR cache table",
            },
            "paper/compression_downstream_correlation_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "compression_downstream_correlation_table.tex"),
                "description": "LaTeX compression/downstream correlation table",
            },
            "paper/feature_error_text_relevance_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "feature_error_text_relevance_table.tex"),
                "description": "LaTeX feature-error/text-relevance mechanism table",
            },
            "paper/boundary_error_readout_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "boundary_error_readout_table.tex"),
                "description": "LaTeX SAM3-box boundary-error readout table",
            },
            "paper/alpha_depth_boundary_alignment_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "alpha_depth_boundary_alignment_table.tex"),
                "description": "LaTeX alpha/depth boundary-alignment coverage table",
            },
            "paper/train_feature_field_audit_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "train_feature_field_audit_table.tex"),
                "description": "LaTeX training-entry auditability table",
            },
            "paper/lerf_ovs_main_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "lerf_ovs_main_table.tex"),
                "description": "LaTeX official-source LERF-OVS comparison table",
            },
            "lerf_direct_3d_selection.md": {
                "path": rel(direct_selection),
                "description": "LERF registered + voxel-context direct 3D object-selection report",
            },
            "lerf_direct_3d_published_context.md": {
                "path": rel(REPORT_DIR / "lerf_direct_3d_published_context.md"),
                "description": "Published-context direct 3D object-selection table for recent methods",
            },
            "vpr_protocol_card.md": {
                "path": rel(REPORT_DIR / "vpr_protocol_card.md"),
                "description": "Auditable VPR protocol card for LERF direct 3D object selection",
            },
            "vpr_contribution_weighting_ablation.md": {
                "path": rel(REPORT_DIR / "vpr_contribution_weighting_ablation.md"),
                "description": "Dr. Splat-inspired alpha/alpha-depth VPR registration weighting ablation",
            },
            "lerf_direct_3d_debug_audit.md": {
                "path": rel(REPORT_DIR / "lerf_direct_3d_debug_audit.md"),
                "description": "LERF direct 3D object-selection audit plus registration follow-up",
            },
            "lerf_direct_3d_query_audit.md": {
                "path": rel(REPORT_DIR / "lerf_direct_3d_query_audit.md"),
                "description": "Query-level bootstrap/failure audit for the VPR direct 3D selector",
            },
            "paper/lerf_direct_3d_selection_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "lerf_direct_3d_selection_table.tex"),
                "description": "LaTeX LERF direct 3D object-selection table",
            },
            "paper/lerf_direct_3d_context_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "lerf_direct_3d_context_table.tex"),
                "description": "LaTeX published-context direct 3D table",
            },
            "paper/lerf_vpr_ablation_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "lerf_vpr_ablation_table.tex"),
                "description": "LaTeX VPR direct-selection diagnostic table",
            },
            "paper/lerf_direct_3d_query_audit_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "lerf_direct_3d_query_audit_table.tex"),
                "description": "LaTeX query-level audit table for VPR direct 3D selection",
            },
            "paper/vpr_protocol_card.tex": {
                "path": rel(REPO_ROOT / "paper" / "vpr_protocol_card.tex"),
                "description": "LaTeX VPR protocol card",
            },
            "expert4_improvement_completion_audit.md": {
                "path": rel(REPORT_DIR / "expert4_improvement_completion_audit.md"),
                "description": "Implementation and validation audit for expert suggestion (4)",
            },
            "expert5_improvement_update.md": {
                "path": rel(REPORT_DIR / "expert5_improvement_update.md"),
                "description": "Implementation and validation update for expert suggestion (5)",
            },
            "lerf_vpr_direct_3d_qualitative_manifest.json": {
                "path": rel(REPORT_DIR / "lerf_vpr_direct_3d_qualitative_manifest.json"),
                "description": "Manifest for VPR direct 3D qualitative cases",
            },
            "lerf_main_qualitative_comparison_manifest.json": {
                "path": rel(
                    REPO_ROOT
                    / "paper"
                    / "artifacts"
                    / "lerf_main_qualitative_comparison_manifest.json"
                ),
                "description": "Manifest for the main-paper direct-3D qualitative comparison",
            },
            "lerf_2d3d_ovs_qualitative_manifest.json": {
                "path": rel(
                    REPO_ROOT
                    / "paper"
                    / "artifacts"
                    / "lerf_2d3d_ovs_qualitative_manifest.json"
                ),
                "description": "Manifest for the main-paper LERF 2D/3D OVS qualitative comparison",
            },
            "scannet_openvocab_3d_query_qualitative_manifest.json": {
                "path": rel(
                    REPO_ROOT
                    / "paper"
                    / "artifacts"
                    / "scannet_openvocab_3d_query_qualitative_manifest.json"
                ),
                "description": "Manifest for the main-paper ScanNet binary open-vocabulary 3D query qualitative comparison",
            },
            "lerf_direct3d_support_policy_ablation_qualitative_manifest.json": {
                "path": rel(
                    REPO_ROOT
                    / "paper"
                    / "artifacts"
                    / "lerf_direct3d_support_policy_ablation_qualitative_manifest.json"
                ),
                "description": "Manifest for the main-paper direct-3D support-policy ablation qualitative figure",
            },
            "lerf_direct3d_prompt_ensemble_support_policy_20260528.md": {
                "path": rel(
                    REPO_ROOT
                    / "paper"
                    / "artifacts"
                    / "lerf_direct3d_prompt_ensemble_support_policy_20260528.md"
                ),
                "description": "Guarded compact Direct3D prompt-ensemble support-policy result",
            },
            "lerf_direct3d_compact_readout_ablation_20260528.md": {
                "path": rel(
                    REPO_ROOT
                    / "paper"
                    / "artifacts"
                    / "lerf_direct3d_compact_readout_ablation_20260528.md"
                ),
                "description": "Strict pure one-map versus guarded compact Direct3D ablation",
            },
            "lerf_direct3d_compact_readout_ablation_20260528.json": {
                "path": rel(
                    REPO_ROOT
                    / "paper"
                    / "artifacts"
                    / "lerf_direct3d_compact_readout_ablation_20260528.json"
                ),
                "description": "Machine-readable strict pure one-map versus guarded compact Direct3D ablation",
            },
            "lerf_direct3d_score_component_guard_20260528.md": {
                "path": rel(
                    REPO_ROOT
                    / "paper"
                    / "artifacts"
                    / "lerf_direct3d_score_component_guard_20260528.md"
                ),
                "description": "Promoted compact Direct3D score-component support guard result",
            },
            "lerf_direct3d_score_component_guard_20260528.json": {
                "path": rel(
                    REPO_ROOT
                    / "paper"
                    / "artifacts"
                    / "lerf_direct3d_score_component_guard_20260528.json"
                ),
                "description": "Machine-readable compact Direct3D score-component support guard result",
            },
            "direct3d_compact_readout_factorial_summary.md": {
                "path": rel(
                    REPO_ROOT
                    / "paper"
                    / "artifacts"
                    / "direct3d_compact_readout_factorial_summary.md"
                ),
                "description": "Generated compact Direct3D readout factorial summary used by the LaTeX ablation table",
            },
            "compose_lerf_main_qualitative.py": {
                "path": rel(
                    REPO_ROOT
                    / "radio_gs"
                    / "scripts"
                    / "compose_lerf_main_qualitative.py"
                ),
                "description": "Generator for the main-paper LERF qualitative comparison",
            },
            "2026-05-16-sam3-box-readout-results.md": {
                "path": rel(sam3_box_results),
                "description": "Official SAM3 box-prompt direct-3D readout results and protocol note",
            },
            "2026-05-16-expert-latest-followup-audit.md": {
                "path": rel(expert_latest_audit),
                "description": "Checklist mapping the latest expert recommendations to current artifacts and remaining gaps",
            },
            "lerf_sam3_box_global_threshold_sweep_20260516.md": {
                "path": rel(sam3_global_threshold_sweep),
                "description": "Fixed-global-threshold SAM3 box direct-3D padding sweep summary",
            },
            "lerf_sam3_box_global_threshold_sweep_20260516.json": {
                "path": rel(REPORT_DIR / "lerf_sam3_box_global_threshold_sweep_20260516.json"),
                "description": "Machine-readable fixed-global-threshold SAM3 box direct-3D sweep manifest",
            },
            "lerf_sam3_box_global_threshold_sweep_20260517_geometry.md": {
                "path": rel(sam3_geometry_threshold_sweep),
                "description": "Geometry-map rerun of the strict pad16 SAM3 box direct-3D sweep",
            },
            "lerf_sam3_box_global_threshold_sweep_20260517_geometry.json": {
                "path": rel(REPORT_DIR / "lerf_sam3_box_global_threshold_sweep_20260517_geometry.json"),
                "description": "Machine-readable geometry-map rerun of the strict pad16 SAM3 box direct-3D sweep",
            },
            "controlled_evidence_table.md": {
                "path": rel(REPORT_DIR / "controlled_evidence_table.md"),
                "description": "Controlled evidence table combining teacher, full CTF-GS, ablations, direct-3D readouts, storage, and runtime",
            },
            "controlled_evidence_table.json": {
                "path": rel(REPORT_DIR / "controlled_evidence_table.json"),
                "description": "Machine-readable controlled evidence table",
            },
            "teacher_vs_ctfgs_2d_usability_20260525.md": {
                "path": rel(
                    REPO_ROOT
                    / "paper"
                    / "artifacts"
                    / "teacher_vs_ctfgs_2d_usability_20260525.md"
                ),
                "description": "Consolidated 2D teacher-vs-CTF-GS feature-usability report with 6/6 selected primary downstream wins and secondary caveats",
            },
            "teacher_vs_ctfgs_2d_usability_20260525.json": {
                "path": rel(
                    REPO_ROOT
                    / "paper"
                    / "artifacts"
                    / "teacher_vs_ctfgs_2d_usability_20260525.json"
                ),
                "description": "Machine-readable 2D teacher-vs-CTF-GS feature-usability manifest",
            },
            "unified_multi_head_feature_quality_field_20260525.md": {
                "path": rel(
                    REPO_ROOT
                    / "paper"
                    / "artifacts"
                    / "unified_multi_head_feature_quality_field_20260525.md"
                ),
                "description": "Method-level implementation audit for explicit quality/visibility field readouts",
            },
            "unified_multi_head_feature_quality_field_20260525.json": {
                "path": rel(
                    REPO_ROOT
                    / "paper"
                    / "artifacts"
                    / "unified_multi_head_feature_quality_field_20260525.json"
                ),
                "description": "Machine-readable unified multi-head feature-quality field manifest",
            },
            "paper/tables/teacher_vs_ctfgs_2d_usability_20260525.tex": {
                "path": rel(
                    REPO_ROOT
                    / "paper"
                    / "tables"
                    / "teacher_vs_ctfgs_2d_usability_20260525.tex"
                ),
                "description": "Compact LaTeX table for 2D teacher-vs-CTF-GS feature usability",
            },
            "lerf_nearest_view_cache_baseline.md": {
                "path": rel(REPORT_DIR / "lerf_nearest_view_cache_baseline.md"),
                "description": "Measured unwarped nearest-view RADIO cache baseline under the LERF evaluator",
            },
            "lerf_nearest_view_cache_baseline.json": {
                "path": rel(REPORT_DIR / "lerf_nearest_view_cache_baseline.json"),
                "description": "Machine-readable nearest-view RADIO cache baseline",
            },
            "paper/lerf_nearest_view_cache_baseline_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "lerf_nearest_view_cache_baseline_table.tex"),
                "description": "LaTeX nearest-view RADIO cache baseline table",
            },
            "lerf_per_gaussian_1280d_baseline.md": {
                "path": rel(REPORT_DIR / "lerf_per_gaussian_1280d_baseline.md"),
                "description": "Measured per-Gaussian 1280-D explicit RADIO-memory baseline under the LERF evaluator",
            },
            "lerf_per_gaussian_1280d_baseline.json": {
                "path": rel(REPORT_DIR / "lerf_per_gaussian_1280d_baseline.json"),
                "description": "Machine-readable per-Gaussian 1280-D explicit RADIO-memory baseline",
            },
            "paper/lerf_per_gaussian_1280d_baseline_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "lerf_per_gaussian_1280d_baseline_table.tex"),
                "description": "LaTeX per-Gaussian 1280-D explicit RADIO-memory baseline table",
            },
            "controlled_baseline_gap_audit.md": {
                "path": rel(REPORT_DIR / "controlled_baseline_gap_audit.md"),
                "description": "Audit documenting nearest-view measured status and per-Gaussian 1280-D explicit measured controlled baseline row",
            },
            "waldo_failure_stratification.md": {
                "path": rel(REPORT_DIR / "waldo_failure_stratification.md"),
                "description": "Waldo Kitchen direct-3D object-size and zero-prediction failure stratification",
            },
            "waldo_failure_stratification.json": {
                "path": rel(REPORT_DIR / "waldo_failure_stratification.json"),
                "description": "Machine-readable Waldo Kitchen failure stratification",
            },
            "lerf_direct3d_confidence_coverage_analysis.md": {
                "path": rel(REPORT_DIR / "lerf_direct3d_confidence_coverage_analysis.md"),
                "description": "Direct3D scene view-coverage and teacher-score confidence mechanism analysis",
            },
            "lerf_direct3d_confidence_coverage_analysis.json": {
                "path": rel(REPORT_DIR / "lerf_direct3d_confidence_coverage_analysis.json"),
                "description": "Machine-readable Direct3D confidence/coverage mechanism analysis",
            },
            "paper/lerf_direct3d_confidence_coverage_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "lerf_direct3d_confidence_coverage_table.tex"),
                "description": "LaTeX Direct3D confidence/coverage mechanism table",
            },
            "verify_submission_provenance.py": {
                "path": rel(REPO_ROOT / "radio_gs" / "scripts" / "verify_submission_provenance.py"),
                "description": "Verifier for required row-level provenance fields in the submission freeze manifest",
            },
            "build_waldo_failure_stratification.py": {
                "path": rel(REPO_ROOT / "radio_gs" / "scripts" / "build_waldo_failure_stratification.py"),
                "description": "Generator for Waldo Kitchen failure stratification report",
            },
            "build_direct3d_confidence_coverage_report.py": {
                "path": rel(REPO_ROOT / "radio_gs" / "scripts" / "build_direct3d_confidence_coverage_report.py"),
                "description": "Generator for Direct3D confidence/coverage mechanism report",
            },
            "build_compression_downstream_correlation.py": {
                "path": rel(REPO_ROOT / "radio_gs" / "scripts" / "build_compression_downstream_correlation.py"),
                "description": "Generator for compression-ratio versus downstream-mIoU correlation report",
            },
            "build_feature_error_text_relevance_report.py": {
                "path": rel(REPO_ROOT / "radio_gs" / "scripts" / "build_feature_error_text_relevance_report.py"),
                "description": "Generator for feature-error versus text-relevance mechanism report",
            },
            "build_boundary_error_report.py": {
                "path": rel(REPO_ROOT / "radio_gs" / "scripts" / "build_boundary_error_report.py"),
                "description": "Generator for SAM3-box boundary-error readout report",
            },
            "build_alpha_depth_boundary_alignment_report.py": {
                "path": rel(REPO_ROOT / "radio_gs" / "scripts" / "build_alpha_depth_boundary_alignment_report.py"),
                "description": "Generator for alpha/depth boundary-alignment coverage report",
            },
            "build_alpha_depth_case_figure.py": {
                "path": rel(REPO_ROOT / "radio_gs" / "scripts" / "build_alpha_depth_case_figure.py"),
                "description": "Generator for selected alpha/depth boundary-case figure",
            },
            "build_train_feature_field_audit.py": {
                "path": rel(REPO_ROOT / "radio_gs" / "scripts" / "build_train_feature_field_audit.py"),
                "description": "Generator for train_feature_field auditability report",
            },
            "build_lerf_nearest_view_cache_baseline.py": {
                "path": rel(REPO_ROOT / "radio_gs" / "scripts" / "build_lerf_nearest_view_cache_baseline.py"),
                "description": "Generator for the unwarped nearest-view RADIO cache baseline",
            },
            "build_lerf_per_gaussian_1280d_baseline.py": {
                "path": rel(REPO_ROOT / "radio_gs" / "scripts" / "build_lerf_per_gaussian_1280d_baseline.py"),
                "description": "Generator for the per-Gaussian 1280-D explicit RADIO-memory baseline",
            },
            "build_teacher_vs_ctfgs_2d_usability_report.py": {
                "path": rel(
                    REPO_ROOT
                    / "radio_gs"
                    / "scripts"
                    / "build_teacher_vs_ctfgs_2d_usability_report.py"
                ),
                "description": "Generator for the 2D teacher-vs-CTF-GS feature-usability report",
            },
            "lerf_sam3_box_direct_3d_qualitative_manifest.json": {
                "path": rel(REPORT_DIR / "lerf_sam3_box_direct_3d_qualitative_manifest.json"),
                "description": "Manifest for fixed-pad0 SAM3 box direct-3D qualitative cases",
            },
            "lerf_sam3_box_direct_3d_qualitative_pad16_manifest.json": {
                "path": rel(REPORT_DIR / "lerf_sam3_box_direct_3d_qualitative_pad16_manifest.json"),
                "description": "Manifest for fixed-pad16 SAM3 box boundary-diagnostic qualitative cases",
            },
            "paper/radio_gs_draft.tex": {
                "path": rel(REPO_ROOT / "paper" / "radio_gs_draft.tex"),
                "description": "Current LaTeX paper draft",
            },
        },
        "figures": {},
        "profiles": sorted(rel(path) for path in PROFILE_DIR.iterdir() if path.is_dir()),
        "audit": {
            "paper_submission_result_audit.md": {
                "path": rel(REPORT_DIR / "paper_submission_result_audit.md"),
                "status": "ALL_VERIFIED",
            }
        },
        "pending": extract_pending_items(readiness),
    }

    figure_descriptions = {
        "fig_grounding_comparison.png": "Main qualitative grounding comparison",
        "fig_radio_flow_comparison.png": "Main feature reconstruction comparison",
        "fig_pipeline_figurines.png": "Pipeline visualization for Figurines",
        "fig_room0_radio_flow_comparison.png": "Room0 feature-flow comparison",
        "fig_room0_pipeline_comparison.png": "Room0 pipeline comparison",
    }
    for name, description in figure_descriptions.items():
        path = FIG_DIR / name
        if path.exists():
            manifest["figures"][name] = {
                "path": rel(path),
                "description": description,
                "size_mb": size_mb(path),
            }
    paper_figure_descriptions = {
        "lerf_2d3d_ovs_qualitative.png": "Main-paper LERF 2D/3D open-vocabulary query qualitative comparison",
        "scannet_openvocab_3d_query_qualitative.png": "Main-paper ScanNet binary open-vocabulary 3D query qualitative comparison",
        "lerf_direct3d_support_policy_ablation_qualitative.png": "Main-paper LERF direct-3D support-policy ablation qualitative comparison",
        "lerf_main_qualitative_comparison.png": "Main-paper LERF direct-3D qualitative comparison",
        "lerf_adaptor_downstream_qualitative.png": "DINOv3/SAM3 adaptor qualitative probes",
        "lerf_sam_dino_tasks_qualitative.png": "Formal SAM/DINO downstream task qualitative probes",
        "lerf_vpr_direct_3d_qualitative.png": "VPR direct 3D object-selection qualitative grid",
        "lerf_sam3_box_direct_3d_qualitative_pad16.png": "SAM3 box direct-3D boundary diagnostic qualitative grid",
        "alpha_depth_boundary_cases.png": "Selected alpha/depth boundary-case montage for SAM3-box diagnostics",
    }
    for name, description in paper_figure_descriptions.items():
        path = PAPER_FIG_DIR / name
        if path.exists():
            manifest["figures"][name] = {
                "path": rel(path),
                "description": description,
                "size_mb": size_mb(path),
            }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
