#!/usr/bin/env python3
"""Build paper-facing single-query latency evidence from frozen profiles."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ROOT = REPO_ROOT / "output" / "radio_gs" / "profiles"
DIRECT3D_ROOT = (
    REPO_ROOT / "output" / "radio_gs" / "lerf_direct3d_prompt_ensemble_policy_20260528"
)


@dataclass(frozen=True)
class LatencyRow:
    task: str
    unit: str
    total_seconds: float
    query_count: int
    latency_ms: float
    peak_vram_mib: int
    source: str
    note: str


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_real_seconds(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^real\s+([0-9.]+)$", text, flags=re.MULTILINE)
    if match:
        return float(match.group(1))
    match = re.search(r"Elapsed \(wall clock\) time .*: (.+)", text)
    if not match:
        raise ValueError(f"Could not parse wall time from {path}")
    value = match.group(1).strip()
    if ":" not in value:
        return float(value)
    parts = [float(item) for item in value.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + part
    return seconds


def parse_peak_vram_mib(path: Path) -> int:
    peak = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            peak = max(peak, int(float(parts[3])))
        except ValueError:
            continue
    return peak


def lerf_rendered_latency_row() -> LatencyRow:
    profile_dirs = [
        PROFILE_ROOT / "freeze_lerf_figurines_overlay_20260502",
        PROFILE_ROOT / "freeze_lerf_ramen_overlay_20260502",
        PROFILE_ROOT / "freeze_lerf_teatime_overlay_20260502",
        PROFILE_ROOT / "freeze_lerf_waldo_overlay_20260502",
    ]
    total_seconds = 0.0
    query_count = 0
    peak_vram = 0
    for profile_dir in profile_dirs:
        time_log = profile_dir / "time.log"
        text = time_log.read_text(encoding="utf-8")
        match = re.search(r"LERFDataset: (\d+) frames, (\d+) text queries", text)
        if not match:
            raise ValueError(f"Could not parse frame/query count from {time_log}")
        frames = int(match.group(1))
        queries = int(match.group(2))
        total_seconds += parse_real_seconds(time_log)
        query_count += frames * queries
        peak_vram = max(peak_vram, parse_peak_vram_mib(profile_dir / "gpu_metrics.csv"))
    return LatencyRow(
        task="LERF rendered-view OVS",
        unit="view-query",
        total_seconds=total_seconds,
        query_count=query_count,
        latency_ms=1000.0 * total_seconds / query_count,
        peak_vram_mib=peak_vram,
        source="output/radio_gs/profiles/freeze_lerf_*_overlay_20260502",
        note="conservative profile; includes teacher branch, all queries, and visualization I/O",
    )


def direct3d_latency_row() -> LatencyRow:
    total_seconds = 0.0
    query_count = 0
    peak_vram = 0
    sources: list[str] = []
    for result_path in sorted(DIRECT3D_ROOT.glob("*/lerf_direct_3d_selection_results.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        scene = payload.get("scene", {})
        total_seconds += float(payload.get("elapsed_seconds", 0.0))
        objects = scene.get("objects", []) if isinstance(scene, dict) else []
        if not objects:
            # Fallback to the official LERF-OVS object counts used by the
            # compatibility artifacts.
            scene_name = result_path.parent.name
            query_count += {
                "figurines": 56,
                "ramen": 71,
                "teatime": 59,
                "waldo_kitchen": 22,
            }[scene_name]
        else:
            query_count += len(objects)
        sources.append(_rel(result_path))
    return LatencyRow(
        task="LERF direct 3D OVS",
        unit="object-query",
        total_seconds=total_seconds,
        query_count=query_count,
        latency_ms=1000.0 * total_seconds / query_count,
        peak_vram_mib=peak_vram,
        source=", ".join(sources),
        note="query-select-render evaluation; includes selected-primitive rendering and mask writing",
    )


def scannet_latency_row() -> LatencyRow:
    profile_dir = PROFILE_ROOT / "freeze_scannet_v67_all_eval_20260502"
    time_log = profile_dir / "time.log"
    text = time_log.read_text(encoding="utf-8")
    seen: set[str] = set()
    query_count = 0
    for match in re.finditer(r"(scene\d+_\d+) point query:.*?100%\|.*?\| (\d+)/\2", text):
        scene = match.group(1)
        if scene in seen:
            continue
        seen.add(scene)
        query_count += int(match.group(2))
    if query_count == 0:
        # Fallback for tqdm strings without the repeated numerator pattern.
        for scene, count in re.findall(r"(scene\d+_\d+) point query:.*?\| (\d+)/(\d+)", text):
            if scene not in seen:
                seen.add(scene)
                query_count += int(count)
    total_seconds = parse_real_seconds(time_log)
    return LatencyRow(
        task="ScanNet point query",
        unit="class-query",
        total_seconds=total_seconds,
        query_count=query_count,
        latency_ms=1000.0 * total_seconds / query_count if query_count else 0.0,
        peak_vram_mib=parse_peak_vram_mib(profile_dir / "gpu_metrics.csv"),
        source=_rel(profile_dir),
        note="legacy 10-scene profile; reports point-query class scoring throughput",
    )


def build_rows() -> list[LatencyRow]:
    return [lerf_rendered_latency_row(), direct3d_latency_row(), scannet_latency_row()]


def write_markdown(rows: list[LatencyRow], path: Path) -> None:
    lines = [
        "# Single-Query Latency Evidence",
        "",
        "Lower is better. The current table converts frozen evaluation profiles into "
        "single-query latency units. These are conservative profile-derived values, "
        "not a clean warm-GPU microbenchmark: the LERF rendered profile includes "
        "teacher/evaluator work and visualization I/O, and the direct-3D profile "
        "includes query-select-render mask generation.",
        "",
        "| Task | Unit | #Queries | Total time | Latency / query | Peak VRAM | Source |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.task} | {row.unit} | {row.query_count:,} | "
            f"{row.total_seconds:.3f} s | {row.latency_ms:.1f} ms | "
            f"{row.peak_vram_mib if row.peak_vram_mib else '-'} MiB | `{row.source}` |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- {row.task}: {row.note}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(rows: list[LatencyRow], path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Single-query latency evidence. Lower is better. Values are derived from frozen profile artifacts and are conservative because the profiled commands include evaluator overhead and, for LERF rendered-view runs, visualization/teacher branches. Direct 3D latency includes query-select-render mask generation.}",
        r"\label{tab:efficiency_cost}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Task & Unit & Queries & Latency & Peak VRAM \\",
        r" &  &  & ms/query & MiB \\",
        r"\midrule",
    ]
    for row in rows:
        peak = f"{row.peak_vram_mib}" if row.peak_vram_mib else "--"
        lines.append(
            f"{row.task} & {row.unit} & {row.query_count:,} & "
            f"{row.latency_ms:.1f} & {peak} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_markdown(rows, REPO_ROOT / "paper" / "artifacts" / "query_latency_table.md")
    write_markdown(rows, REPO_ROOT / "paper" / "artifacts" / "efficiency_cost_table.md")
    write_latex(rows, REPO_ROOT / "paper" / "efficiency_cost_table.tex")
    write_markdown(rows, REPO_ROOT / "output" / "radio_gs" / "reports" / "query_latency_table.md")
    write_markdown(rows, REPO_ROOT / "output" / "radio_gs" / "reports" / "efficiency_cost_table.md")
    print("Wrote paper/artifacts/query_latency_table.md")
    print("Wrote paper/artifacts/efficiency_cost_table.md")
    print("Wrote output/radio_gs/reports/query_latency_table.md")
    print("Wrote output/radio_gs/reports/efficiency_cost_table.md")
    print("Wrote paper/efficiency_cost_table.tex")


if __name__ == "__main__":
    main()
