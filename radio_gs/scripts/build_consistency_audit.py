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

    manifest_payload = load_json(manifest) if manifest.exists() else {}

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
        REPORT_DIR / "submission_freeze_report.md",
        REPORT_DIR / "submission_freeze_figure_shortlist.md",
        REPORT_DIR / "lerf_component_ablation.md",
        REPORT_DIR / "scannet_dino_cv_ablation.md",
        REPORT_DIR / "lerf_direct_3d_selection.md",
        REPORT_DIR / "lerf_direct_3d_debug_audit.md",
        REPO_ROOT / "paper" / "radio_gs_draft.tex",
        REPO_ROOT / "paper" / "lerf_ovs_main_table.tex",
        REPO_ROOT / "paper" / "efficiency_cost_table.tex",
        REPO_ROOT / "paper" / "lerf_direct_3d_selection_table.tex",
        REPO_ROOT / "paper" / "lerf_vpr_ablation_table.tex",
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
            "- LERF direct 3D selection now uses a VPR registered-view + GT-free voxel-context primitive readout; it is promoted as primitive-level evidence with a Waldo/provenance caveat.",
            "",
            "## Open Items",
            "",
        ]
    )

    pending_items = manifest_payload.get("pending", ["refresh paper_assets_manifest.json"])
    if pending_items:
        for item in pending_items:
            lines.append(f"- {item}")
    else:
        lines.append("- none")

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
