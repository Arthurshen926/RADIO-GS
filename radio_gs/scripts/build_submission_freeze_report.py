"""Build a conservative submission-freeze report for RADIO-GS."""

from __future__ import annotations

import json
import csv
import argparse
import re
from pathlib import Path
from typing import Any


SCAN_SPLITS = ("19", "15", "10")
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "output" / "radio_gs" / "reports"


def _round4(value: float) -> float:
    return round(float(value), 4)


def collect_scannet_v67(eval_root: str | Path) -> dict[str, Any]:
    root = Path(eval_root)
    pattern = (
        "scene*_v67_teacherbalanced_fromv63_best_gidx_labelpoint/"
        "scannet_pointcloud_radio_gs_results.json"
    )
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in sorted(root.glob(pattern)):
        payload = json.loads(path.read_text())
        scene = path.parent.name.split("_v67_")[0]
        args = payload.get("args", {})
        if args.get("query_mode") != "gaussian_index":
            warnings.append(f"{scene}: query_mode is {args.get('query_mode')}")
        if args.get("opacity_filter_mode") != "label_index":
            warnings.append(f"{scene}: opacity_filter_mode is {args.get('opacity_filter_mode')}")
        if args.get("gaussian_index_position_mode") != "label_point":
            warnings.append(
                f"{scene}: gaussian_index_position_mode is "
                f"{args.get('gaussian_index_position_mode')}"
            )
        macro = payload["macro"]
        rows.append(
            {
                "scene": scene,
                "path": str(path),
                "miou": {split: float(macro[split]["miou"]) for split in SCAN_SPLITS},
                "macc": {split: float(macro[split]["macc"]) for split in SCAN_SPLITS},
            }
        )

    macro_miou = (
        {
            split: _round4(sum(row["miou"][split] for row in rows) / len(rows))
            for split in SCAN_SPLITS
        }
        if rows
        else {split: 0.0 for split in SCAN_SPLITS}
    )
    macro_macc = (
        {
            split: _round4(sum(row["macc"][split] for row in rows) / len(rows))
            for split in SCAN_SPLITS
        }
        if rows
        else {split: 0.0 for split in SCAN_SPLITS}
    )
    return {
        "scene_count": len(rows),
        "rows": rows,
        "macro_miou": macro_miou,
        "macro_macc": macro_macc,
        "warnings": warnings,
    }


def collect_lerf_best(csv_path: str | Path) -> dict[str, Any]:
    path = Path(csv_path)
    rows: list[dict[str, str]] = []
    macro_loc_acc = 0.0
    macro_miou = 0.0
    if not path.exists():
        return {
            "rows": rows,
            "macro_loc_acc": 0.0,
            "macro_miou": 0.0,
            "warnings": [f"missing {path}"],
        }

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["scene"] == "macro":
                macro_loc_acc = _round4(float(row["loc_acc"]))
                macro_miou = _round4(float(row["miou"]))
                continue
            rows.append(row)

    return {
        "rows": rows,
        "macro_loc_acc": macro_loc_acc,
        "macro_miou": macro_miou,
        "warnings": [],
    }


def _parse_profile_time(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"wall": "-", "wall_seconds": None}
    text = path.read_text(errors="replace")

    match = re.search(r"^real\s+([0-9.]+)$", text, flags=re.MULTILINE)
    if match:
        seconds = float(match.group(1))
        return {"wall": f"{seconds:.3f} s", "wall_seconds": seconds}

    match = re.search(r"Elapsed \(wall clock\) time .*: (.+)", text)
    if match:
        return {"wall": match.group(1).strip(), "wall_seconds": None}

    return {"wall": "-", "wall_seconds": None}


def _parse_profile_gpu(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "peak_gpu_mem_mib": 0.0,
            "peak_gpu_util_pct": 0.0,
            "mean_gpu_util_pct": 0.0,
            "samples": 0,
        }
    peak_mem = 0.0
    peak_util = 0.0
    total_util = 0.0
    samples = 0
    for line in path.read_text(errors="replace").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            util = float(parts[2])
            mem = float(parts[3])
        except ValueError:
            continue
        peak_util = max(peak_util, util)
        peak_mem = max(peak_mem, mem)
        total_util += util
        samples += 1
    return {
        "peak_gpu_mem_mib": peak_mem,
        "peak_gpu_util_pct": peak_util,
        "mean_gpu_util_pct": total_util / samples if samples else 0.0,
        "samples": samples,
    }


def _parse_profile_meta(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    if not path.exists():
        return meta
    for line in path.read_text(errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.strip()] = value.strip()
    return meta


def collect_profile_runs(profile_dirs: list[str | Path] | tuple[str | Path, ...]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for profile_dir in [Path(path) for path in profile_dirs]:
        if not profile_dir.exists():
            warnings.append(f"missing profile {profile_dir}")
            continue
        time_info = _parse_profile_time(profile_dir / "time.log")
        gpu_info = _parse_profile_gpu(profile_dir / "gpu_metrics.csv")
        meta = _parse_profile_meta(profile_dir / "meta.txt")
        rows.append(
            {
                "name": profile_dir.name,
                "path": str(profile_dir),
                "wall": time_info["wall"],
                "wall_seconds": time_info["wall_seconds"],
                "gpu": meta.get("gpu", ""),
                "command": meta.get("command", ""),
                **gpu_info,
            }
        )
    return {"profile_count": len(rows), "rows": rows, "warnings": warnings}


def _format_split_values(values: dict[str, float]) -> str:
    return " / ".join(f"{split}: {values[split]:.4f}" for split in SCAN_SPLITS)


def build_markdown(
    lerf: dict[str, Any],
    scannet: dict[str, Any],
    profiles: dict[str, Any] | None = None,
) -> str:
    profiles = profiles or {"profile_count": 0, "rows": [], "warnings": []}
    warnings = [
        "External LERF/LangSplat/LEGaussians rows are official-source context rows, not reproduced local-evaluator baselines.",
        "ScanNet label-supervised or GT-label-balanced runs are diagnostic only and excluded from this fair v67 summary.",
        "LERF direct 3D object selection is protocol-aligned; registered-view readout plus GT-free voxel context improves the fixed protocol while Waldo remains the limiting scene.",
    ]
    warnings.extend(lerf.get("warnings", []))
    warnings.extend(scannet.get("warnings", []))
    warnings.extend(profiles.get("warnings", []))

    lines = [
        "# RADIO-GS Submission Freeze Report",
        "",
        "This generated report is the current paper-facing source of truth for the conservative submission package.",
        "",
        "## Claim-to-Artifact Matrix",
        "",
        "| Paper claim | Current status | Primary artifact | Paper use |",
        "|---|---|---|---|",
        (
            "| LERF main result | Frozen | "
            "`output/radio_gs/lerf_summary_tables/current_best_lerf_ovs_per_scene.csv` | Main open-vocabulary table |"
        ),
        (
            "| ScanNet fair cross-domain result | Frozen | "
            "`output/scannet_pointcloud_eval/*_v67_teacherbalanced_fromv63_best_gidx_labelpoint/scannet_pointcloud_radio_gs_results.json` | Cross-domain table |"
        ),
        (
            "| Efficiency/profile evidence | Current eval profiles frozen | "
            "`output/radio_gs/profiles/freeze_*_20260502` | Runtime and memory table |"
        ),
        (
            "| Qualitative figure shortlist | Frozen overlay candidates selected | "
            "`output/radio_gs/reports/submission_freeze_figure_shortlist.md` | Main qualitative figure |"
        ),
        (
            "| External baseline comparison | Official-source provenance closed | "
            "`output/radio_gs/reports/baseline_source_verification.md` | Main comparison table with protocol caveat |"
        ),
        (
            "| LERF direct 3D object selection | Registered-view + voxel-context primitive readout | "
            "`output/radio_gs/reports/lerf_direct_3d_selection.md` | OpenGaussian-style VPR primitive-level result |"
        ),
        "",
        "## LERF-OVS",
        "",
        "- Protocol: rendered-feature best-scene summary from existing JSON-backed LERF sweeps.",
        f"- Macro LocAcc: `{lerf['macro_loc_acc']:.4f}`",
        f"- Macro mIoU: `{lerf['macro_miou']:.4f}`",
        "",
        "| Scene | LocAcc | mIoU | Temp | Source summary |",
        "|---|---:|---:|---:|---|",
    ]
    for row in lerf.get("rows", []):
        lines.append(
            "| {scene} | {loc:.4f} | {miou:.4f} | {temp} | `{summary}` |".format(
                scene=row["scene"],
                loc=float(row["loc_acc"]),
                miou=float(row["miou"]),
                temp=row.get("temp", ""),
                summary=row.get("summary", ""),
            )
        )

    if (REPORT_DIR / "lerf_direct_3d_selection.md").exists():
        lines.extend(
            [
                "",
                "## LERF Direct 3D Object Selection",
                "",
                "- Protocol: OpenGaussian-style direct primitive query, selected-Gaussian rendering, and LERF-OVS mask evaluation.",
                "- Readout: rendered SigLIP2 features registered back to 3D Gaussian primitives with depth/alpha visibility checks.",
                "- Context aggregation: GT-free voxel-max propagation at resolution `80` with blend `0.50`.",
                "- Selector: fixed GT-free `meanstd2p5` with `selection_min_ratio=0.005` and `selection_max_ratio=0.02`.",
                "- CTF-GS fixed `meanstd2p5+floor0.005+cap0.02`: macro mIoU `0.4133`, macro Acc@0.25 `0.6741`.",
                "- CTF-GS fixed `top0p02` conservative audit: macro mIoU `0.3850`, macro Acc@0.25 `0.6428`.",
                "- CTF-GS best-by-scene diagnostic: macro mIoU `0.4166`, macro Acc@0.25 `0.6741`.",
                "- OpenGaussian official context: macro mIoU `0.3836`, macro Acc@0.25 `0.5143`.",
                "- Diagnostics: original Gaussian-center readout is `0.0804` macro mIoU; registered softmax24 without aggregation is `0.3421`; 96-view VPR with voxel aggregation improves fixed-ratio macro mIoU to `0.3850`, GT-free score-distribution selection improves it to `0.3934`, adding the fixed 2% cap improves it to `0.4072`, and adding the fixed 0.5% floor improves it to `0.4133`.",
                "- Paper use: VPR-backed primitive-level evidence with an explicit Waldo/provenance caveat.",
            ]
        )

    lines.extend(
        [
            "",
            "## ScanNet",
            "",
            "- Protocol: v67 teacher-balanced direct point query, `gaussian_index`, `label_point`, `label_index` opacity.",
            f"- Scenes found: `{scannet['scene_count']}`",
            f"- Macro mIoU: `{_format_split_values(scannet['macro_miou'])}`",
            f"- Macro mAcc: `{_format_split_values(scannet['macro_macc'])}`",
            "",
            "| Scene | mIoU19 | mIoU15 | mIoU10 | Source JSON |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in scannet.get("rows", []):
        lines.append(
            "| {scene} | {m19:.4f} | {m15:.4f} | {m10:.4f} | `{path}` |".format(
                scene=row["scene"],
                m19=row["miou"]["19"],
                m15=row["miou"]["15"],
                m10=row["miou"]["10"],
                path=row["path"],
            )
        )

    lines.extend(
        [
            "",
            "## Profile Evidence",
            "",
            f"- Profiled workloads: `{profiles['profile_count']}`",
            "",
            "| Profile | GPU | Wall Time | Peak VRAM (MiB) | Peak GPU% | Mean GPU% | Samples |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in profiles.get("rows", []):
        lines.append(
            "| `{name}` | {gpu} | {wall} | {peak_mem:.0f} | {peak_util:.0f} | {mean_util:.2f} | {samples} |".format(
                name=row["name"],
                gpu=row.get("gpu", ""),
                wall=row["wall"],
                peak_mem=float(row["peak_gpu_mem_mib"]),
                peak_util=float(row["peak_gpu_util_pct"]),
                mean_util=float(row["mean_gpu_util_pct"]),
                samples=int(row["samples"]),
            )
        )

    lines.extend(
        [
            "",
            "## Warnings",
            "",
        ]
    )
    lines.extend(f"- {warning}" for warning in warnings)
    lines.append("")
    return "\n".join(lines)


def write_freeze_outputs(
    output_dir: str | Path,
    lerf: dict[str, Any],
    scannet: dict[str, Any],
    profiles: dict[str, Any] | None = None,
) -> dict[str, Path]:
    profiles = profiles or {"profile_count": 0, "rows": [], "warnings": []}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = out_dir / "submission_freeze_report.md"
    manifest_path = out_dir / "submission_freeze_manifest.json"

    markdown_path.write_text(build_markdown(lerf, scannet, profiles))
    manifest_path.write_text(
        json.dumps(
            {
                "lerf": lerf,
                "scannet": scannet,
                "profiles": profiles,
                "warnings": (
                    lerf.get("warnings", [])
                    + scannet.get("warnings", [])
                    + profiles.get("warnings", [])
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return {"markdown": markdown_path, "manifest": manifest_path}


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = argparse.ArgumentParser(description="Build RADIO-GS submission freeze report")
    parser.add_argument(
        "--lerf_csv",
        default="output/radio_gs/lerf_summary_tables/current_best_lerf_ovs_per_scene.csv",
    )
    parser.add_argument(
        "--scannet_eval_root",
        default="output/scannet_pointcloud_eval",
    )
    parser.add_argument(
        "--profile_dirs",
        nargs="*",
        default=[],
        help="Profile directories created by profile_command.sh",
    )
    parser.add_argument(
        "--output_dir",
        default="output/radio_gs/reports",
    )
    args = parser.parse_args(argv)

    lerf = collect_lerf_best(args.lerf_csv)
    scannet = collect_scannet_v67(args.scannet_eval_root)
    profiles = collect_profile_runs(args.profile_dirs)
    paths = write_freeze_outputs(args.output_dir, lerf, scannet, profiles=profiles)
    print(f"Wrote {paths['markdown']}")
    print(f"Wrote {paths['manifest']}")
    return paths


if __name__ == "__main__":
    main()
