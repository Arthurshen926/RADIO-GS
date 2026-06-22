#!/usr/bin/env python3
"""Build a compact consistency audit for paper-facing artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "output" / "radio_gs" / "reports"


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build consistency audit")
    parser.add_argument(
        "--output",
        default=str(REPORT_DIR / "final_consistency_audit.md"),
        help="Audit markdown output",
    )
    args = parser.parse_args()

    main_table = REPORT_DIR / "paper_submission_main_table.md"
    audit = REPORT_DIR / "paper_submission_result_audit.md"
    manifest = REPORT_DIR / "paper_assets_manifest.json"
    freeze_manifest = REPORT_DIR / "submission_freeze_manifest.json"

    manifest_payload = load_json(manifest) if manifest.exists() else {}
    freeze_payload = load_json(freeze_manifest) if freeze_manifest.exists() else {}
    direct3d_readouts = freeze_payload.get("direct3d_readouts", manifest_payload.get("direct3d_readouts", []))

    lines = [
        "# Final Consistency Audit",
        "",
        "## Required Artifacts",
        "",
        "| Artifact | Exists | Path |",
        "|---|---|---|",
    ]
    required_paths = [
        main_table,
        audit,
        manifest,
        REPORT_DIR / "baseline_source_verification.md",
        REPORT_DIR / "efficiency_cost_table.md",
        REPORT_DIR / "controlled_evidence_table.md",
        REPORT_DIR / "controlled_evidence_table.json",
        REPORT_DIR / "lerf_nearest_view_cache_baseline.md",
        REPORT_DIR / "lerf_nearest_view_cache_baseline.json",
        REPORT_DIR / "lerf_per_gaussian_1280d_baseline.md",
        REPORT_DIR / "lerf_per_gaussian_1280d_baseline.json",
        REPORT_DIR / "controlled_baseline_gap_audit.md",
        REPORT_DIR / "storage_footprint_report.md",
        REPORT_DIR / "compression_downstream_correlation.md",
        REPORT_DIR / "compression_downstream_correlation.json",
        REPORT_DIR / "feature_error_text_relevance_report.md",
        REPORT_DIR / "feature_error_text_relevance_report.json",
        REPORT_DIR / "boundary_error_readout_report.md",
        REPORT_DIR / "boundary_error_readout_report.json",
        REPORT_DIR / "alpha_depth_boundary_alignment_report.md",
        REPORT_DIR / "alpha_depth_boundary_alignment_report.json",
        REPORT_DIR / "alpha_depth_boundary_case_figure_manifest.md",
        REPORT_DIR / "alpha_depth_boundary_case_figure_manifest.json",
        REPORT_DIR / "train_feature_field_audit.md",
        REPORT_DIR / "train_feature_field_audit.json",
        REPORT_DIR / "submission_freeze_report.md",
        freeze_manifest,
        REPORT_DIR / "submission_freeze_figure_shortlist.md",
        REPORT_DIR / "lerf_component_ablation.md",
        REPORT_DIR / "scannet_dino_cv_ablation.md",
        REPORT_DIR / "lerf_direct_3d_selection.md",
        REPORT_DIR / "lerf_direct_3d_published_context.md",
        REPORT_DIR / "vpr_protocol_card.md",
        REPORT_DIR / "vpr_contribution_weighting_ablation.md",
        REPORT_DIR / "lerf_direct_3d_debug_audit.md",
        REPORT_DIR / "lerf_direct_3d_query_audit.md",
        REPORT_DIR / "waldo_failure_stratification.md",
        REPORT_DIR / "waldo_failure_stratification.json",
        REPORT_DIR / "lerf_direct3d_confidence_coverage_analysis.md",
        REPORT_DIR / "lerf_direct3d_confidence_coverage_analysis.json",
        REPORT_DIR / "expert4_improvement_completion_audit.md",
        REPORT_DIR / "lerf_vpr_direct_3d_qualitative_manifest.json",
        REPORT_DIR / "lerf_sam3_box_direct_3d_qualitative_manifest.json",
        REPORT_DIR / "lerf_sam3_box_direct_3d_qualitative_pad16_manifest.json",
        REPORT_DIR / "lerf_sam3_box_global_threshold_sweep_20260516.md",
        REPORT_DIR / "lerf_sam3_box_global_threshold_sweep_20260516.json",
        REPORT_DIR / "lerf_sam3_box_global_threshold_sweep_20260517_geometry.md",
        REPORT_DIR / "lerf_sam3_box_global_threshold_sweep_20260517_geometry.json",
        REPO_ROOT / "docs" / "experiments" / "2026-05-16-sam3-box-readout-results.md",
        REPO_ROOT / "docs" / "experiments" / "2026-05-16-expert-latest-followup-audit.md",
        REPO_ROOT / "radio_gs" / "scripts" / "verify_submission_provenance.py",
        REPO_ROOT / "radio_gs" / "scripts" / "build_waldo_failure_stratification.py",
        REPO_ROOT / "radio_gs" / "scripts" / "build_direct3d_confidence_coverage_report.py",
        REPO_ROOT / "radio_gs" / "scripts" / "build_compression_downstream_correlation.py",
        REPO_ROOT / "radio_gs" / "scripts" / "build_feature_error_text_relevance_report.py",
        REPO_ROOT / "radio_gs" / "scripts" / "build_boundary_error_report.py",
        REPO_ROOT / "radio_gs" / "scripts" / "build_alpha_depth_boundary_alignment_report.py",
        REPO_ROOT / "radio_gs" / "scripts" / "build_alpha_depth_case_figure.py",
        REPO_ROOT / "radio_gs" / "scripts" / "build_train_feature_field_audit.py",
        REPO_ROOT / "radio_gs" / "scripts" / "build_lerf_nearest_view_cache_baseline.py",
        REPO_ROOT / "radio_gs" / "scripts" / "build_lerf_per_gaussian_1280d_baseline.py",
        REPO_ROOT / "paper" / "artifacts" / "dino_sam_boundary_and_waldo_recovery_20260528.md",
        REPO_ROOT / "paper" / "artifacts" / "dino_sam_boundary_and_waldo_recovery_20260528.json",
        REPO_ROOT / "paper" / "radio_gs_draft.tex",
        REPO_ROOT / "paper" / "lerf_ovs_main_table.tex",
        REPO_ROOT / "paper" / "lerf_nearest_view_cache_baseline_table.tex",
        REPO_ROOT / "paper" / "lerf_per_gaussian_1280d_baseline_table.tex",
        REPO_ROOT / "paper" / "efficiency_cost_table.tex",
        REPO_ROOT / "paper" / "storage_footprint_table.tex",
        REPO_ROOT / "paper" / "compression_downstream_correlation_table.tex",
        REPO_ROOT / "paper" / "feature_error_text_relevance_table.tex",
        REPO_ROOT / "paper" / "boundary_error_readout_table.tex",
        REPO_ROOT / "paper" / "alpha_depth_boundary_alignment_table.tex",
        REPO_ROOT / "paper" / "train_feature_field_audit_table.tex",
        REPO_ROOT / "paper" / "lerf_direct_3d_selection_table.tex",
        REPO_ROOT / "paper" / "lerf_direct_3d_context_table.tex",
        REPO_ROOT / "paper" / "lerf_vpr_ablation_table.tex",
        REPO_ROOT / "paper" / "lerf_direct_3d_query_audit_table.tex",
        REPO_ROOT / "paper" / "lerf_direct3d_confidence_coverage_table.tex",
        REPO_ROOT / "paper" / "vpr_protocol_card.tex",
        REPO_ROOT / "paper" / "figures" / "lerf_vpr_direct_3d_qualitative.png",
        REPO_ROOT / "paper" / "figures" / "lerf_sam3_box_direct_3d_qualitative_pad16.png",
        REPO_ROOT / "paper" / "figures" / "alpha_depth_boundary_cases.png",
    ]
    for path in required_paths:
        lines.append(f"| {path.name} | {'YES' if path.exists() else 'NO'} | `{rel(path)}` |")

    lines.extend(
        [
            "",
            "## Current Route",
            "",
            f"- route: `{manifest_payload.get('route', 'unknown')}`",
            f"- manifest generated: `{manifest_payload.get('generated', 'missing')}`",
            "- conservative rule: main table remains `LERF / LangSplat / LEGaussians / RADIO-GS` on rendered `LocAcc`.",
            "- external baselines are official-source context rows unless reproduced under the local evaluator.",
            "- ScanNet remains direct-query transfer evidence rather than a full leaderboard claim.",
            "- LERF direct 3D selection now promotes the compact score-component guard row: no VPR cache, no official RGB SAM decoder, with GT-free RGB/GrabCut support snapping reported as a lightweight support policy.",
            "- Strict no-RGB one-map Direct3D and frozen official SAM3-box rows remain ablations/diagnostics rather than the promoted compact score-component row.",
            "- The DINO/SAM downstream task artifact now records 6/6 selected primary wins for rendered GaussFM features, while keeping secondary SAM LocAcc and DINO dense-HitRate caveats separate.",
            "- SAM3 box direct-3D readout now has a fixed-global-threshold padding sweep; post-hoc best fixed and scene-locked selectors remain diagnostic.",
            "- Dr. Splat-inspired alpha/alpha-depth registration weighting was implemented and tested, but is not promoted because it lowers the fixed-protocol VPR result.",
            "- VPR cache memory is reported separately from persistent compact checkpoint storage.",
            "- The nearest-view RADIO cache and per-Gaussian 1280-D explicit RADIO-memory baselines are both measured under the same LERF evaluator.",
            "- Alpha/depth discontinuity instrumentation is populated for the strict pad16 SAM3-box row with 208 per-query geometry-map records and overlays.",
            "- The training entry point is split across `radio_gs/training` support modules and now passes the audit at 3735 entry-point lines, with manifest/split/checkpoint/lock guards and wrapped tensor-cache loads.",
            "",
        ]
    )

    if direct3d_readouts:
        lines.extend(
            [
                "## Direct-3D Readout Registry",
                "",
                "| Readout | Selector | Macro mIoU | Acc@0.25 | Boundary-F | Trimap IoU |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for readout in direct3d_readouts:
            lines.append(
                "| {label} | `{selector}` | {miou:.4f} | {acc:.4f} | {boundary:.4f} | {trimap:.4f} |".format(
                    label=readout.get("label", "unknown"),
                    selector=readout.get("selector_policy", "unknown"),
                    miou=float(readout.get("macro_miou", 0.0)),
                    acc=float(readout.get("macro_acc025", 0.0)),
                    boundary=float(readout.get("macro_boundary_f", 0.0)),
                    trimap=float(readout.get("macro_trimap_iou", 0.0)),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Artifact Open Items",
            "",
        ]
    )

    diagnostic_selector_constraints = [
        (
            f"Do not promote "
            f"`{readout.get('label', 'unknown')}` from diagnostic to strict main-protocol result; "
            f"current selector policy is `best_by_miou`."
        )
        for readout in direct3d_readouts
        if readout.get("selector_policy") == "best_by_miou"
    ]
    pending_items = manifest_payload.get("pending", ["refresh paper_assets_manifest.json"])
    if pending_items:
        for item in pending_items:
            lines.append(f"- {item}")
    else:
        lines.append("- none; required paper-facing artifacts are present.")

    if diagnostic_selector_constraints:
        lines.extend(
            [
                "",
                "## Diagnostic Selector Constraints",
                "",
            ]
        )
        for item in diagnostic_selector_constraints:
            lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Claim Constraints",
            "",
            "- Strict LERF same-evaluator SOTA claims still require locally rerun external direct-3D baselines; OpenGaussian is locally reproduced for ScanNet, but its LERF recipe is blocked by the missing official LangSplat-reannotated language-feature package.",
            "- True Dr. Splat-style rasterizer contribution assignment remains future work; the implemented center-sampled alpha/alpha-depth approximation regressed.",
            "- Official SAM3 code/weights are diagnostic support only. The promoted SAM boundary readouts are feature-only GaussFM/RADIO adaptor readouts and do not call the official RGB SAM decoder at evaluation time.",
            "- Waldo low-support heatmap recovery is positive as a diagnostic but not promoted globally until the recovery floor is adaptive; the fixed 8000-pixel floor regresses Ramen.",
            "- Component-aware VPR was tested with reusable per-Gaussian score caches and did not beat voxel-max; stronger GT-free object proposals remain future work.",
            "- Nearest-view cached RADIO features are not a 3D scene memory; the measured per-Gaussian 1280-D explicit row is the controlled raw-feature scene-memory comparison.",
            "- Per-query alpha/depth discontinuity analysis is now available, but correlations are weak and should be framed as a boundary-error mechanism diagnostic rather than causal proof.",
        ]
    )

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
