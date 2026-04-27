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
    readiness = REPORT_DIR / "submission_readiness_checklist.md"
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
            "paper_ablation_table.md": {
                "path": rel(REPORT_DIR / "paper_ablation_table.md"),
                "description": "LERF three-variant ablation report",
            },
            "paper_ablation_table.tex": {
                "path": rel(REPORT_DIR / "paper_ablation_table.tex"),
                "description": "LaTeX ablation tables",
            },
            "room0_variant_comparison.md": {
                "path": rel(REPORT_DIR / "room0_variant_comparison.md"),
                "description": "Room0 variant comparison report (markdown)",
            },
            "room0_variant_comparison.tex": {
                "path": rel(REPORT_DIR / "room0_variant_comparison.tex"),
                "description": "Room0 variant comparison report (LaTeX)",
            },
            "paper_protocol_freeze.md": {
                "path": rel(REPORT_DIR / "paper_protocol_freeze.md"),
                "description": "Frozen conservative submission protocol and claims",
            },
            "baseline_source_verification.md": {
                "path": rel(REPORT_DIR / "baseline_source_verification.md"),
                "description": "External baseline source verification tracker",
            },
            "submission_readiness_checklist.md": {
                "path": rel(REPORT_DIR / "submission_readiness_checklist.md"),
                "description": "Submission freeze checklist",
            },
            "figure_shortlist.md": {
                "path": rel(REPORT_DIR / "figure_shortlist.md"),
                "description": "Frozen figure shortlist and captions",
            },
            "efficiency_profile_summary.md": {
                "path": rel(efficiency),
                "description": "Efficiency summary table",
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

    output_path = Path(args.output)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
