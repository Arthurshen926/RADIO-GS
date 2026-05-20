#!/usr/bin/env python3
"""Audit Semantic Gaussians readiness for strict ScanNet reproduction."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Sequence


DEFAULT_REPO = Path("/root/baselines/semantic-gaussians")
DEFAULT_LOCAL_SITE = Path("output/baselines/semantic_gaussians/local_site")
DEFAULT_SCANNET_ROOT = Path("/mnt/pool/sqy/3d_understanding/scannet")
DEFAULT_OUTPUT_ROOT = Path("output/baselines/semantic_gaussians/scannet_compat_20260520")
DEFAULT_SCENES = ("scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00")
DEFAULT_OUTPUT_JSON = Path("paper/artifacts/semantic_gaussians_readiness_audit.json")
DEFAULT_OUTPUT_MD = Path("paper/artifacts/semantic_gaussians_readiness_audit.md")

CORE_DEPENDENCIES = ("MinkowskiEngine", "encoding")
SMOKE_DEPENDENCIES = ("tensorflow", "viser", "torch")
NATIVE_MODULES = ("simple_knn", "rgbd_rasterization", "channel_rasterization")
ENTRYPOINTS = ("train.py", "fusion.py", "distill.py", "eval_segmentation.py", "view_viser.py")
CONFIGS = ("official_train.yaml", "fusion_scannet.yaml", "distill_scannet.yaml", "eval.yaml")


def _extension_present(directory: Path) -> bool:
    return any(path.is_file() for path in directory.glob("_C*.so"))


def _module_available(module: str, local_site: Path) -> bool:
    original_path = list(sys.path)
    original_modules = {
        name: loaded_module
        for name, loaded_module in sys.modules.items()
        if name == module or name.startswith(f"{module}.")
    }
    try:
        sys.path.insert(0, str(local_site.resolve()))
        importlib.invalidate_caches()
        importlib.import_module(module)
    except Exception:
        return False
    finally:
        sys.path[:] = original_path
        importlib.invalidate_caches()
        for name in [
            loaded_name
            for loaded_name in sys.modules
            if loaded_name == module or loaded_name.startswith(f"{module}.")
        ]:
            if name not in original_modules:
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)
    return True


def _check_repo(repo: Path, local_site: Path) -> tuple[bool, bool, list[str]]:
    missing: list[str] = []
    for relative in ENTRYPOINTS:
        if not (repo / relative).is_file():
            missing.append(f"missing {relative}")
    for name in CONFIGS:
        if not (repo / "config" / name).is_file():
            missing.append(f"missing config/{name}")
    native_missing = [
        module for module in NATIVE_MODULES
        if not _extension_present(local_site / module)
    ]
    for module in native_missing:
        missing.append(f"missing {module} extension in local_site")
    if not (local_site / "segment_anything").exists():
        missing.append("missing segment_anything package in local_site")
    repo_ready = not any(item.startswith("missing train.py") or item.startswith("missing fusion.py") for item in missing)
    native_ready = not native_missing and (local_site / "segment_anything").exists()
    return repo_ready, native_ready, missing


def _extracted_scene_ready(scene_dir: Path) -> bool:
    if not scene_dir.is_dir():
        return False
    has_color = (scene_dir / "color").is_dir() or (scene_dir / "images").is_dir()
    has_pose = (scene_dir / "pose").is_dir()
    has_intrinsic = (scene_dir / "intrinsic").is_dir() or (scene_dir / "intrnsic").is_dir()
    has_points = (scene_dir / "points3d.ply").is_file() or (scene_dir / "sparse" / "0").is_dir()
    has_transforms = (scene_dir / "transforms_train.json").is_file() and (scene_dir / "transforms_test.json").is_file()
    scannet_ready = has_color and has_pose and has_intrinsic and has_points
    blender_ready = has_color and has_transforms and (scene_dir / "points3d.ply").is_file()
    return scannet_ready or blender_ready


def _language_features_ready(scene_dir: Path) -> bool:
    language_dir = scene_dir / "language_features"
    if language_dir.is_dir() and any(path.is_file() for path in language_dir.rglob("*")):
        return True
    language_zip = scene_dir / "language_features.zip"
    return language_zip.is_file() and zipfile.is_zipfile(language_zip)


def _glob_any(root: Path, patterns: Sequence[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _check_scene(scannet_root: Path, output_root: Path, scene: str) -> dict[str, Any]:
    scene_dir = scannet_root / scene
    output_dir = output_root / scene
    zip_path = scannet_root / f"{scene}.zip"
    extracted_ready = _extracted_scene_ready(scene_dir)
    language_features_ready = _language_features_ready(scene_dir) if scene_dir.is_dir() else False
    gaussian = _glob_any(output_dir, ("gaussians/**/point_cloud*.ply", "point_cloud/**/point_cloud*.ply"))
    fusion = _glob_any(output_dir, ("fusion/**/*.pt", "fusion/**/*.npy", "fused/**/*.pt", "fused/**/*.npy"))
    distill = _glob_any(output_dir, ("distill/**/*.pth", "distill/**/*.pt", "results_distill/**/*.pth", "results_distill/**/*.pt"))
    eval_result = _glob_any(output_dir, ("eval/**/*.json", "eval_results/**/*.json", "metrics*.json"))

    missing: list[str] = []
    if not scene_dir.is_dir():
        missing.append("missing extracted ScanNet scene directory")
    if not zip_path.is_file():
        missing.append("missing ScanNet raw zip")
    if scene_dir.is_dir() and not extracted_ready:
        missing.append("extracted ScanNet scene is missing color/pose/intrinsic/points data")
    if scene_dir.is_dir() and not language_features_ready:
        missing.append("missing extracted or valid raw language features")
    if gaussian is None:
        missing.append("missing trained RGB Gaussian output")
    if fusion is None:
        missing.append("missing fused 2D semantic features")
    if distill is None:
        missing.append("missing distilled 3D semantic network checkpoint")
    if eval_result is None:
        missing.append("missing segmentation eval metrics")

    return {
        "scene": scene,
        "zip_path": str(zip_path),
        "zip_present": zip_path.is_file(),
        "scene_dir": str(scene_dir),
        "extracted_ready": extracted_ready,
        "language_features_ready": language_features_ready,
        "output_dir": str(output_dir),
        "gaussian_output": str(gaussian) if gaussian else None,
        "gaussian_ready": gaussian is not None,
        "fusion_output": str(fusion) if fusion else None,
        "fusion_ready": fusion is not None,
        "distill_output": str(distill) if distill else None,
        "distill_ready": distill is not None,
        "eval_result": str(eval_result) if eval_result else None,
        "eval_ready": eval_result is not None,
        "ready_for_eval": extracted_ready and language_features_ready and gaussian is not None and fusion is not None and distill is not None and eval_result is not None,
        "missing": missing,
    }


def build_audit(
    *,
    repo: str | Path = DEFAULT_REPO,
    local_site: str | Path = DEFAULT_LOCAL_SITE,
    dependency_site: str | Path | None = None,
    scannet_root: str | Path = DEFAULT_SCANNET_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    scenes: Sequence[str] = DEFAULT_SCENES,
) -> dict[str, Any]:
    repo_path = Path(repo)
    local_site_path = Path(local_site)
    dependency_site_path = Path(dependency_site) if dependency_site is not None else local_site_path
    scannet_root_path = Path(scannet_root)
    output_root_path = Path(output_root)
    repo_ready, native_ready, repo_missing = _check_repo(repo_path, local_site_path)
    dependencies = {
        module: _module_available(module, dependency_site_path)
        for module in (*CORE_DEPENDENCIES, *SMOKE_DEPENDENCIES)
    }
    dependency_ready = all(dependencies[module] for module in CORE_DEPENDENCIES)
    scene_summaries = {
        scene: _check_scene(scannet_root_path, output_root_path, scene)
        for scene in scenes
    }
    environment_ready = repo_ready and native_ready and dependency_ready
    all_scenes_ready = all(scene["ready_for_eval"] for scene in scene_summaries.values())
    strict_ready = environment_ready and all_scenes_ready
    return {
        "method": "Semantic Gaussians",
        "repo": str(repo_path),
        "local_site": str(local_site_path),
        "dependency_site": str(dependency_site_path),
        "scannet_root": str(scannet_root_path),
        "output_root": str(output_root_path),
        "repo_ready": repo_ready,
        "native_ready": native_ready,
        "repo_missing": repo_missing,
        "dependencies": dependencies,
        "dependency_ready": dependency_ready,
        "environment_ready": environment_ready,
        "all_scenes_ready": all_scenes_ready,
        "strict_ready": strict_ready,
        "scenes": scene_summaries,
        "next_action": (
            "Promote completed ScanNet metrics into the external baseline registry."
            if all_scenes_ready
            else (
                "Run or resume RGB-GS/fusion/distill/eval stages."
                if dependency_ready
                else "Resolve MinkowskiEngine/PyTorch-Encoding imports, then run train/fusion/distill/eval stages."
            )
        )
    }


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Semantic Gaussians Readiness Audit",
        "",
        f"- repo ready: {_yes(bool(summary['repo_ready']))}",
        f"- native extensions ready: {_yes(bool(summary['native_ready']))}",
        f"- core dependencies ready: {_yes(bool(summary['dependency_ready']))}",
        f"- strict environment ready: {_yes(bool(summary.get('environment_ready', False)))}",
        f"- all scenes ready: {_yes(bool(summary['all_scenes_ready']))}",
        f"- strict ready: {_yes(bool(summary.get('strict_ready', False)))}",
        f"- next action: {summary['next_action']}",
        "",
        "## Dependencies",
        "",
        "| Module | Importable |",
        "|---|---:|",
    ]
    for module, ready in summary["dependencies"].items():
        lines.append(f"| {module} | {_yes(bool(ready))} |")
    lines.extend([
        "",
        "## Scenes",
        "",
        "| Scene | Zip | Extracted | Lang Feat | RGB GS | Fusion | Distill | Eval | Missing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for scene, item in summary["scenes"].items():
        lines.append(
            "| {scene} | {zip} | {extract} | {lang} | {rgb} | {fusion} | {distill} | {eval} | {missing} |".format(
                scene=scene,
                zip=_yes(bool(item["zip_present"])),
                extract=_yes(bool(item["extracted_ready"])),
                lang=_yes(bool(item["language_features_ready"])),
                rgb=_yes(bool(item["gaussian_ready"])),
                fusion=_yes(bool(item["fusion_ready"])),
                distill=_yes(bool(item["distill_ready"])),
                eval=_yes(bool(item["eval_ready"])),
                missing="; ".join(item["missing"]) if item["missing"] else "none",
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--local-site", type=Path, default=DEFAULT_LOCAL_SITE)
    parser.add_argument("--dependency-site", type=Path, default=None)
    parser.add_argument("--scannet-root", type=Path, default=DEFAULT_SCANNET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--output-json", "--out-json", dest="output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", "--out-md", dest="output_md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = build_audit(
        repo=args.repo,
        local_site=args.local_site,
        dependency_site=args.dependency_site,
        scannet_root=args.scannet_root,
        output_root=args.output_root,
        scenes=args.scenes,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
