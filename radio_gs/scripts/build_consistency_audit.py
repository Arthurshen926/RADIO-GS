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
    room0 = REPORT_DIR / "room0_variant_comparison.md"
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
    for path in [main_table, audit, room0, manifest, REPORT_DIR / "paper_ablation_table.md", REPORT_DIR / "figure_shortlist.md", REPORT_DIR / "efficiency_profile_summary.md"]:
        lines.append(f"| {path.name} | {'YES' if path.exists() else 'NO'} | `{rel(path)}` |")

    lines.extend(
        [
            "",
            "## Current Route",
            "",
            f"- route: `{manifest_payload.get('route', 'unknown')}`",
            f"- manifest generated: `{manifest_payload.get('generated', 'missing')}`",
            "- conservative rule: main table remains `LERF / LangSplat / LEGaussians / RADIO-GS` on rendered `LocAcc`.",
            "- room0 remains supporting analysis only.",
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
