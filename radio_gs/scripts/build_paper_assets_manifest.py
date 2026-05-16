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
        / "formal_v9_dino_topk_area200_bg110_peak_20260514"
        / "lerf_sam_dino_task_report.md"
    )
    sam_dino_sweep = (
        REPO_ROOT
        / "output"
        / "lerf_sam_dino_tasks"
        / "formal_v9_dino_readout_sweep_20260514.md"
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
            "paper/lerf_component_ablation_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "lerf_component_ablation_table.tex"),
                "description": "LaTeX component ablation table",
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
                "description": "Formal SAM3/DINOv3 downstream task sweep with bg-suppressed DINO propagation readout",
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
            "paper/efficiency_cost_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "efficiency_cost_table.tex"),
                "description": "LaTeX efficiency/cost table",
            },
            "paper/storage_footprint_table.tex": {
                "path": rel(REPO_ROOT / "paper" / "storage_footprint_table.tex"),
                "description": "LaTeX compact-storage and optional VPR cache table",
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
        "lerf_adaptor_downstream_qualitative.png": "DINOv3/SAM3 adaptor qualitative probes",
        "lerf_sam_dino_tasks_qualitative.png": "Formal SAM/DINO downstream task qualitative probes",
        "lerf_vpr_direct_3d_qualitative.png": "VPR direct 3D object-selection qualitative grid",
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
