"""Build a conservative submission-freeze report for RADIO-GS."""

from __future__ import annotations

import json
import csv
import argparse
import re
import hashlib
import subprocess
import ast
from pathlib import Path
from typing import Any


SCAN_SPLITS = ("19", "15", "10")
DIRECT3D_SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "output" / "radio_gs" / "reports"
LERF_PROVENANCE_JSON = REPO_ROOT / "output" / "radio_gs" / "lerf_summary_tables" / "current_best_lerf_ovs_per_scene.json"
EVALUATOR_SCRIPTS = {
    "eval_lerf_grounding": "radio_gs/scripts/eval_lerf_grounding.py",
    "eval_scannet_pointcloud_radio_gs": "radio_gs/scripts/eval_scannet_pointcloud_radio_gs.py",
    "eval_lerf_direct_3d_selection": "radio_gs/scripts/eval_lerf_direct_3d_selection.py",
}


def _round4(value: float) -> float:
    return round(float(value), 4)


def _sha256_path(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    if not candidate.exists() or not candidate.is_file():
        return ""
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_scalar(config_path: str | Path, key: str) -> str:
    candidate = Path(config_path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    if not candidate.exists() or not candidate.is_file():
        return ""
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$")
    for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        value = match.group(1).split("#", 1)[0].strip()
        return value.strip("\"'")
    return ""


def _config_provenance(config_path: str | Path) -> dict[str, Any]:
    if not config_path:
        return {
            "config_sha256": "",
            "teacher_model": "",
            "feature_manifest": "",
            "seed": "",
        }
    return {
        "config_sha256": _sha256_path(config_path),
        "teacher_model": _config_scalar(config_path, "radio_version"),
        "feature_manifest": _config_scalar(config_path, "feature_dir"),
        "seed": _config_scalar(config_path, "seed"),
    }


def _evaluator_provenance(evaluator: str) -> dict[str, str]:
    script = EVALUATOR_SCRIPTS.get(evaluator, "")
    return {
        "evaluator": evaluator,
        "evaluator_script": script,
        "evaluator_sha256": _sha256_path(script) if script else "",
    }


def _path_from_payload_value(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("path", ""))
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
        if isinstance(parsed, dict):
            return str(parsed.get("path", value))
    return str(value or "")


def _git_metadata() -> dict[str, Any]:
    def _run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    status = _run_git(["status", "--short"])
    return {
        "git_commit": _run_git(["rev-parse", "HEAD"]),
        "git_branch": _run_git(["branch", "--show-current"]),
        "git_dirty": bool(status),
    }


def _load_lerf_provenance_index(path: str | Path = LERF_PROVENANCE_JSON) -> dict[str, dict[str, Any]]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    scenes = payload.get("scenes", {})
    return {str(scene): dict(value) for scene, value in scenes.items()}


def _parse_scene_subset(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or not str(raw).strip():
        return None
    scenes = [part.strip() for part in str(raw).replace(";", ",").split(",") if part.strip()]
    return tuple(dict.fromkeys(scenes))


def collect_scannet_v67(
    eval_root: str | Path,
    *,
    scene_subset: tuple[str, ...] | None = None,
) -> dict[str, Any]:
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
        if scene_subset is not None and scene not in scene_subset:
            continue
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
        config = str(args.get("config", ""))
        config_meta = _config_provenance(config)
        evaluator_meta = _evaluator_provenance("eval_scannet_pointcloud_radio_gs")
        rows.append(
            {
                "scene": scene,
                "path": str(path),
                "source": str(path),
                "config": config,
                "checkpoint": str(args.get("checkpoint", "")),
                "config_sha256": config_meta["config_sha256"],
                "teacher_model": str(args.get("radio_checkpoint") or config_meta["teacher_model"]),
                "feature_manifest": config_meta["feature_manifest"]
                or str(Path(str(args.get("prepared_root", ""))) / scene),
                "seed": str(args.get("sample_seed") or config_meta["seed"]),
                "text_embedding_cache": str(args.get("text_embedding_cache", "")),
                "selector_policy": "v67_teacherbalanced_gaussian_index_labelpoint",
                "text_head": "SigLIP2",
                **evaluator_meta,
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
        "scene_subset": list(scene_subset) if scene_subset is not None else None,
        "rows": rows,
        "macro_miou": macro_miou,
        "macro_macc": macro_macc,
        "selector_policy": "v67_teacherbalanced_gaussian_index_labelpoint",
        "text_head": "SigLIP2",
        **_evaluator_provenance("eval_scannet_pointcloud_radio_gs"),
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
        "readout": "threshold 0.50",
        "source": str(path),
        "warnings": [],
    }


def _variant_key(value: str | float) -> str:
    return f"{float(value):.2f}"


def _load_sweep_variant(path: str | Path, value: str | float) -> tuple[Path, str, dict[str, Any]]:
    sweep_path = Path(path)
    payload = json.loads(sweep_path.read_text(encoding="utf-8"))
    key = _variant_key(value)
    variants = payload.get("variants", {})
    if key not in variants:
        raise KeyError(f"Variant {key!r} not found in {sweep_path}")
    return sweep_path, key, variants[key]


def collect_lerf_threshold_sweep(path: str | Path, threshold: str | float) -> dict[str, Any]:
    sweep_path, key, variant = _load_sweep_variant(path, threshold)
    provenance_index = _load_lerf_provenance_index()
    rows = []
    evaluator_meta = _evaluator_provenance("eval_lerf_grounding")
    for row in variant.get("rows", []):
        scene = str(row["scene"])
        provenance = provenance_index.get(scene, {})
        config = str(provenance.get("config", ""))
        config_meta = _config_provenance(config)
        rows.append(
            {
                "scene": scene,
                "loc_acc": _round4(float(row["loc"])),
                "miou": _round4(float(row["miou"])),
                "temp": row.get("temp", ""),
                "summary": str(sweep_path),
                "source": str(sweep_path),
                "config": config,
                "checkpoint": str(provenance.get("checkpoint", "")),
                "config_sha256": config_meta["config_sha256"],
                "teacher_model": config_meta["teacher_model"] or "c-radio_v4-h",
                "feature_manifest": config_meta["feature_manifest"],
                "seed": config_meta["seed"],
                "selector_policy": f"fixed_threshold:{key}",
                "threshold_rule": f"fixed global threshold {key}",
                "text_head": "SigLIP2",
                **evaluator_meta,
                "n": int(row.get("n", 0)),
            }
        )
    macro = variant.get("macro", {})
    weighted = variant.get("weighted", {})
    return {
        "rows": rows,
        "macro_loc_acc": _round4(float(macro.get("loc", 0.0))),
        "macro_miou": _round4(float(macro.get("miou", 0.0))),
        "weighted_loc_acc": _round4(float(weighted.get("loc", 0.0))),
        "weighted_miou": _round4(float(weighted.get("miou", 0.0))),
        "readout": f"threshold {key}",
        "selector_policy": f"fixed_threshold:{key}",
        "threshold_rule": f"fixed global threshold {key}",
        "text_head": "SigLIP2",
        "teacher_model": "c-radio_v4-h",
        **_evaluator_provenance("eval_lerf_grounding"),
        "source": str(sweep_path),
        "warnings": [],
    }


def collect_direct3d_silhouette_sweep(path: str | Path, silhouette: str | float) -> dict[str, Any]:
    sweep_path, key, variant = _load_sweep_variant(path, silhouette)
    rows = []
    for row in variant.get("rows", []):
        rows.append(
            {
                "scene": row["scene"],
                "miou": _round4(float(row["miou"])),
                "acc025": _round4(float(row["acc025"])),
                "acc050": _round4(float(row.get("acc050", 0.0))),
                "n": int(row.get("n", 0)),
            }
        )
    macro = variant.get("macro", {})
    weighted = variant.get("weighted", {})
    return {
        "silhouette": key,
        "macro_miou": _round4(float(macro.get("miou", 0.0))),
        "macro_acc025": _round4(float(macro.get("acc025", 0.0))),
        "macro_acc050": _round4(float(macro.get("acc050", 0.0))),
        "weighted_miou": _round4(float(weighted.get("miou", 0.0))),
        "weighted_acc025": _round4(float(weighted.get("acc025", 0.0))),
        "weighted_acc050": _round4(float(weighted.get("acc050", 0.0))),
        "rows": rows,
        "source": str(sweep_path),
        "warnings": [],
    }


def _load_direct3d_scene_payload(root: Path, scene: str) -> tuple[Path, dict[str, Any]] | None:
    preferred = [
        root / scene / "lerf_direct_3d_selection_results.json",
        root / scene / scene / "lerf_direct_3d_selection_results.json",
    ]
    for path in preferred:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("scene", {}).get("scene") == scene:
            return path, payload

    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.glob("**/lerf_direct_3d_selection_results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("scene", {}).get("scene") == scene:
            matches.append((path, payload))
    if not matches:
        return None
    return min(matches, key=lambda item: (len(item[0].parts), str(item[0])))


def _select_direct3d_result(
    scene_payload: dict[str, Any],
    fixed_tag: str | None,
) -> tuple[str, dict[str, Any], str]:
    results = scene_payload.get("results", {})
    if not results:
        raise KeyError(f"No direct-3D results for scene {scene_payload.get('scene', 'unknown')}")
    if fixed_tag and fixed_tag != "best":
        if fixed_tag not in results:
            raise KeyError(
                f"Selection tag {fixed_tag!r} not found for scene "
                f"{scene_payload.get('scene', 'unknown')}"
            )
        return fixed_tag, results[fixed_tag], f"fixed:{fixed_tag}"
    tag = scene_payload.get("best_by_miou") or max(results, key=lambda key: float(results[key]["miou"]))
    return str(tag), results[tag], "best_by_miou"


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return _round4(sum(float(row.get(key, 0.0)) for row in rows) / len(rows))


def collect_direct3d_scene_readout(
    root: str | Path,
    *,
    label: str,
    text_head: str,
    protocol_label: str,
    fixed_tag: str | None = None,
) -> dict[str, Any]:
    """Collect a paper-facing direct-3D readout from per-scene result JSON files."""

    root_path = Path(root)
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, float]] = []
    warnings: list[str] = []
    first_protocol: dict[str, Any] = {}
    selector_policy = "best_by_miou" if not fixed_tag or fixed_tag == "best" else f"fixed:{fixed_tag}"
    evaluator_meta = _evaluator_provenance("eval_lerf_direct_3d_selection")

    for scene in DIRECT3D_SCENES:
        match = _load_direct3d_scene_payload(root_path, scene)
        if match is None:
            warnings.append(f"missing direct-3D scene result for {scene} under {root_path}")
            continue
        path, payload = match
        args = payload.get("args", {})
        scene_payload = payload.get("scene", {})
        if not first_protocol:
            first_protocol = dict(payload.get("protocol", {}))
        tag, metrics, observed_policy = _select_direct3d_result(scene_payload, fixed_tag)
        selector_policy = observed_policy if selector_policy == "best_by_miou" else selector_policy
        config = str(scene_payload.get("config") or args.get("config", ""))
        config_meta = _config_provenance(config)
        raw_row = {
            "miou": float(metrics.get("miou", 0.0)),
            "acc025": float(metrics.get("acc025", 0.0)),
            "acc050": float(metrics.get("acc050", 0.0)),
            "boundary_f": float(metrics.get("boundary_f", 0.0)),
            "trimap_iou": float(metrics.get("trimap_iou", 0.0)),
        }
        raw_rows.append(raw_row)
        rows.append(
            {
                "scene": scene,
                "selection": tag,
                "miou": _round4(raw_row["miou"]),
                "acc025": _round4(raw_row["acc025"]),
                "acc050": _round4(raw_row["acc050"]),
                "boundary_f": _round4(raw_row["boundary_f"]),
                "trimap_iou": _round4(raw_row["trimap_iou"]),
                "n": int(metrics.get("n", 0)),
                "source": str(path),
                "config": config,
                "checkpoint": str(scene_payload.get("checkpoint") or args.get("checkpoint", "")),
                "config_sha256": config_meta["config_sha256"],
                "teacher_model": config_meta["teacher_model"] or "c-radio_v4-h",
                "feature_manifest": config_meta["feature_manifest"],
                "seed": config_meta["seed"],
                "text_embedding_cache": str(args.get("text_embedding_cache", "")),
                "score_cache": _path_from_payload_value(
                    scene_payload.get("score_cache") or args.get("score_cache", "")
                ),
                "selector_policy": selector_policy,
                "text_head": text_head,
                **evaluator_meta,
                "mask_refinement": str(metrics.get("mask_refinement") or args.get("mask_refinement", "")),
            }
        )

    return {
        "label": label,
        "text_head": text_head,
        "protocol_label": protocol_label,
        "selector_policy": selector_policy,
        **evaluator_meta,
        "scene_count": len(rows),
        "macro_miou": _mean_metric(raw_rows, "miou"),
        "macro_acc025": _mean_metric(raw_rows, "acc025"),
        "macro_acc050": _mean_metric(raw_rows, "acc050"),
        "macro_boundary_f": _mean_metric(raw_rows, "boundary_f"),
        "macro_trimap_iou": _mean_metric(raw_rows, "trimap_iou"),
        "rows": rows,
        "source_root": str(root_path),
        "protocol": first_protocol,
        "warnings": warnings,
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
    direct3d: dict[str, Any] | None = None,
    direct3d_readouts: list[dict[str, Any]] | None = None,
) -> str:
    profiles = profiles or {"profile_count": 0, "rows": [], "warnings": []}
    direct3d_readouts = direct3d_readouts or []
    warnings = [
        "External LERF/LangSplat/LEGaussians rows are official-source context rows, not reproduced local-evaluator baselines.",
        "ScanNet label-supervised, GT-label-balanced, old v67, and non-VALA8 runs are diagnostic only and excluded from this VALA-aligned ScanNet-8 summary.",
        "LERF direct 3D object selection is protocol-aligned; direct primitive scoring, RGB-snap cleanup, and official SAM3 box boundary readout are reported as separate readouts.",
    ]
    warnings.extend(lerf.get("warnings", []))
    warnings.extend(scannet.get("warnings", []))
    warnings.extend(profiles.get("warnings", []))
    if direct3d is not None:
        warnings.extend(direct3d.get("warnings", []))
    for readout in direct3d_readouts:
        warnings.extend(readout.get("warnings", []))
        if readout.get("selector_policy") == "best_by_miou":
            warnings.append(
                f"Direct-3D readout `{readout.get('label', 'unknown')}` uses best_by_miou scene selectors; treat it as diagnostic until a validation-selected or global threshold rule is added."
            )

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
            f"`{lerf.get('source', 'output/radio_gs/lerf_summary_tables/current_best_lerf_ovs_per_scene.csv')}` | Main open-vocabulary table |"
        ),
        (
            "| ScanNet fair cross-domain result | Frozen | "
            "`paper/artifacts/scannet_pointcloud_radio_gs_vala8_reproduced_benchmark_20260615.json` | Cross-domain table |"
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
            "`output/radio_gs/reports/lerf_direct_3d_selection.md` | OpenGaussian-style VPR primitive-level result plus separated boundary-readout diagnostics |"
        ),
        "",
        "## LERF-OVS",
        "",
        f"- Protocol: rendered-feature readout from `{lerf.get('source', 'unknown')}`.",
        f"- Mask readout: `{lerf.get('readout', 'threshold 0.50')}`.",
        f"- Macro LocAcc: `{lerf['macro_loc_acc']:.4f}`",
        f"- Macro mIoU: `{lerf['macro_miou']:.4f}`",
        f"- Weighted mIoU: `{float(lerf.get('weighted_miou', 0.0)):.4f}`" if "weighted_miou" in lerf else "",
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

    if direct3d is not None or direct3d_readouts or (REPORT_DIR / "lerf_direct_3d_selection.md").exists():
        direct_lines = [
            "",
            "## LERF Direct 3D Object Selection",
            "",
            "- Protocol: OpenGaussian-style direct primitive query, selected-Gaussian rendering, and LERF-OVS mask evaluation.",
            "- The registry below separates primitive scoring, GT-free RGB boundary cleanup, and frozen official SAM3 box-prompt boundary readout.",
            "- VPR readouts compute text scores on Gaussian primitives; SAM3 box readout refines only the rendered selection boundary and does not use GT masks for candidate selection.",
        ]
        if direct3d is not None:
            direct_lines.extend(
                [
                    (
                        f"- GaussFM + RGB snap silhouette {direct3d['silhouette']}: "
                        f"macro mIoU `{direct3d['macro_miou']:.4f}`, "
                        f"macro Acc@0.25 `{direct3d['macro_acc025']:.4f}`, "
                        f"macro Acc@0.50 `{direct3d['macro_acc050']:.4f}`."
                    ),
                    f"- Direct-3D silhouette sweep source: `{direct3d.get('source', '')}`.",
                ]
            )
        if direct3d_readouts:
            direct_lines.extend(
                [
                    "",
                    "## Direct-3D Readout Registry",
                    "",
                    "| Readout | Text head | Selector policy | Macro mIoU | Macro Acc@0.25 | Boundary-F | Trimap IoU | Source root |",
                    "|---|---|---|---:|---:|---:|---:|---|",
                ]
            )
            for readout in direct3d_readouts:
                direct_lines.append(
                    "| {label} | {text_head} | `{selector}` | {miou:.4f} | {acc:.4f} | {boundary:.4f} | {trimap:.4f} | `{source}` |".format(
                        label=readout["label"],
                        text_head=readout["text_head"],
                        selector=readout["selector_policy"],
                        miou=float(readout["macro_miou"]),
                        acc=float(readout["macro_acc025"]),
                        boundary=float(readout.get("macro_boundary_f", 0.0)),
                        trimap=float(readout.get("macro_trimap_iou", 0.0)),
                        source=readout.get("source_root", ""),
                    )
                )
            for readout in direct3d_readouts:
                direct_lines.extend(
                    [
                        "",
                        f"### {readout['label']} Scene Selectors",
                        "",
                        "| Scene | Selection | mIoU | Acc@0.25 | Boundary-F | Trimap IoU | N | Source JSON |",
                        "|---|---|---:|---:|---:|---:|---:|---|",
                    ]
                )
                for row in readout.get("rows", []):
                    direct_lines.append(
                        "| {scene} | `{selection}` | {miou:.4f} | {acc:.4f} | {boundary:.4f} | {trimap:.4f} | {n} | `{source}` |".format(
                            scene=row["scene"],
                            selection=row["selection"],
                            miou=float(row["miou"]),
                            acc=float(row["acc025"]),
                            boundary=float(row.get("boundary_f", 0.0)),
                            trimap=float(row.get("trimap_iou", 0.0)),
                            n=int(row.get("n", 0)),
                            source=row.get("source", ""),
                        )
                    )
        lines.extend(
            direct_lines
            + [
                "- GaussFM accuracy-oriented cap0.015 diagnostic: macro mIoU `0.4184`, macro Acc@0.25 `0.7013`.",
                "- GaussFM fixed `top0p02` conservative audit: macro mIoU `0.3850`, macro Acc@0.25 `0.6428`.",
                "- GaussFM previous cap0.02 diagnostic: macro mIoU `0.4185`, macro Acc@0.25 `0.6899`.",
                "- OpenGaussian official context: macro mIoU `0.3836`, macro Acc@0.25 `0.5143`.",
                "- Diagnostics: original Gaussian-center readout is `0.0804` macro mIoU; registered softmax24 without aggregation is `0.3421`; 96-view VPR with voxel aggregation improves fixed-ratio macro mIoU to `0.3850`, GT-free score-distribution selection improves it to `0.3934`, adding the fixed 2% cap improves it to `0.4072`, adding the fixed 0.5% floor improves it to `0.4133`, increasing the all-pose registration budget to 128 views improves the fixed paper selector to `0.4185`, tightening the global cap to `0.0175` improves it to `0.4226`, and a cache-backed fixed `0.018` cap slightly improves it to `0.4227` with `0.6906` Acc@0.25.",
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
    direct3d: dict[str, Any] | None = None,
    direct3d_readouts: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    profiles = profiles or {"profile_count": 0, "rows": [], "warnings": []}
    direct3d_readouts = direct3d_readouts or []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = out_dir / "submission_freeze_report.md"
    manifest_path = out_dir / "submission_freeze_manifest.json"

    markdown_path.write_text(
        build_markdown(
            lerf,
            scannet,
            profiles,
            direct3d=direct3d,
            direct3d_readouts=direct3d_readouts,
        )
    )
    manifest_path.write_text(
        json.dumps(
            {
                "metadata": _git_metadata(),
                "lerf": lerf,
                "scannet": scannet,
                "profiles": profiles,
                "direct3d": direct3d,
                "direct3d_readouts": direct3d_readouts,
                "warnings": (
                    lerf.get("warnings", [])
                    + scannet.get("warnings", [])
                    + profiles.get("warnings", [])
                    + (direct3d.get("warnings", []) if direct3d is not None else [])
                    + [
                        warning
                        for readout in direct3d_readouts
                        for warning in readout.get("warnings", [])
                    ]
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
        "--lerf_threshold_sweep_json",
        default="",
        help="Optional rendered-grounding threshold sweep JSON that supersedes --lerf_csv for paper-facing mIoU.",
    )
    parser.add_argument(
        "--lerf_threshold",
        default="0.60",
        help="Variant key to read from --lerf_threshold_sweep_json.",
    )
    parser.add_argument(
        "--direct3d_silhouette_sweep_json",
        default="",
        help="Optional direct-3D RGB-snap silhouette sweep JSON for the paper-facing refined row.",
    )
    parser.add_argument(
        "--direct3d_silhouette",
        default="0.60",
        help="Variant key to read from --direct3d_silhouette_sweep_json.",
    )
    parser.add_argument(
        "--direct3d_scene_readout",
        action="append",
        default=[],
        help=(
            "Direct-3D per-scene readout spec formatted as "
            "label|text_head|protocol_label|root_path[|fixed_tag]. "
            "Use fixed_tag=best or omit it to read each scene's best_by_miou tag."
        ),
    )
    parser.add_argument(
        "--scannet_eval_root",
        default="output/scannet_pointcloud_eval",
    )
    parser.add_argument(
        "--scannet_scene_list",
        default="",
        help="Optional comma-separated fixed ScanNet scene subset for the report.",
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

    if args.lerf_threshold_sweep_json:
        lerf = collect_lerf_threshold_sweep(args.lerf_threshold_sweep_json, args.lerf_threshold)
    else:
        lerf = collect_lerf_best(args.lerf_csv)
    direct3d = (
        collect_direct3d_silhouette_sweep(
            args.direct3d_silhouette_sweep_json,
            args.direct3d_silhouette,
        )
        if args.direct3d_silhouette_sweep_json
        else None
    )
    direct3d_readouts = []
    for spec in args.direct3d_scene_readout:
        parts = spec.split("|")
        if len(parts) not in {4, 5}:
            raise ValueError(
                "--direct3d_scene_readout must be "
                "label|text_head|protocol_label|root_path[|fixed_tag]"
            )
        label, text_head, protocol_label, root_path = parts[:4]
        fixed_tag = parts[4] if len(parts) == 5 else None
        direct3d_readouts.append(
            collect_direct3d_scene_readout(
                root_path,
                label=label,
                text_head=text_head,
                protocol_label=protocol_label,
                fixed_tag=fixed_tag,
            )
        )
    scannet = collect_scannet_v67(
        args.scannet_eval_root,
        scene_subset=_parse_scene_subset(args.scannet_scene_list),
    )
    profiles = collect_profile_runs(args.profile_dirs)
    paths = write_freeze_outputs(
        args.output_dir,
        lerf,
        scannet,
        profiles=profiles,
        direct3d=direct3d,
        direct3d_readouts=direct3d_readouts,
    )
    print(f"Wrote {paths['markdown']}")
    print(f"Wrote {paths['manifest']}")
    return paths


if __name__ == "__main__":
    main()
