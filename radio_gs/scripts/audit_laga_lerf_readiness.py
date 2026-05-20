#!/usr/bin/env python3
"""Audit whether LaGa has the staged outputs needed for LERF mask export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_REPO = Path("/root/baselines/LaGa")
DEFAULT_LOCAL_SITE = Path("output/baselines/laga/local_site")
DEFAULT_LERF_ROOT = Path("/mnt/pool/sqy/3d_understanding/lerf_ovs")
DEFAULT_MODEL_ROOT = Path("output/baselines/laga/lerf_compat_20260519")
DEFAULT_SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")
DEFAULT_OUTPUT_JSON = Path("paper/artifacts/laga_lerf_readiness_audit.json")
DEFAULT_OUTPUT_MD = Path("paper/artifacts/laga_lerf_readiness_audit.md")

DESCRIPTOR_FILES = (
    "multi_lvl_cluster_features.pth",
    "multi_lvl_cluster_feature_weights.pth",
    "multi_lvl_seg_scores.pth",
)


def _first_extension(directory: Path) -> Path | None:
    for path in sorted(directory.glob("_C*.so")):
        if path.exists():
            return path
    return None


def _image_stems(image_dir: Path) -> set[str]:
    stems: set[str] = set()
    if not image_dir.is_dir():
        return stems
    for suffix in ("*.jpg", "*.JPG", "*.jpeg", "*.png"):
        stems.update(path.stem for path in image_dir.glob(suffix))
    return stems


def _label_frames(label_dir: Path) -> list[str]:
    if not label_dir.is_dir():
        return []
    return sorted(path.stem for path in label_dir.glob("*.json"))


def _check_repo(repo: Path, local_site: Path) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for relative in ("train_scene.py", "train_affinity_features.py", "render.py", "inference.ipynb"):
        if not (repo / relative).is_file():
            missing.append(f"missing {relative}")
    for module in ("simple_knn", "diff_gaussian_rasterization", "diff_gaussian_rasterization_contrastive_f"):
        if _first_extension(local_site / module) is None:
            missing.append(f"missing {module} extension in local_site")
    return not missing, missing


def _check_scene(
    lerf_root: Path,
    model_root: Path,
    scene: str,
    scene_iteration: int,
    affinity_iteration: int,
) -> dict[str, Any]:
    scene_dir = lerf_root / scene
    image_dir = scene_dir / "images"
    sparse_dir = scene_dir / "sparse" / "0"
    label_dir = lerf_root / "label" / scene
    model_dir = model_root / scene
    scene_iteration_dir = model_dir / "point_cloud" / f"iteration_{scene_iteration}"
    affinity_iteration_dir = model_dir / "point_cloud" / f"iteration_{affinity_iteration}"
    scene_checkpoint = scene_iteration_dir / "scene_point_cloud.ply"
    affinity_checkpoint = affinity_iteration_dir / "contrastive_feature_point_cloud.ply"
    descriptor_paths = [affinity_iteration_dir / name for name in DESCRIPTOR_FILES]
    labels = _label_frames(label_dir)
    image_stems = _image_stems(image_dir)
    missing_label_images = sorted(frame for frame in labels if frame not in image_stems)

    data_ready = scene_dir.is_dir() and image_dir.is_dir() and sparse_dir.is_dir() and bool(labels) and not missing_label_images
    scene_ready = scene_checkpoint.is_file()
    affinity_ready = affinity_checkpoint.is_file()
    descriptor_ready = all(path.is_file() for path in descriptor_paths)

    missing: list[str] = []
    if not scene_dir.is_dir():
        missing.append("missing scene directory")
    if not image_dir.is_dir():
        missing.append("missing images directory")
    if not sparse_dir.is_dir():
        missing.append("missing COLMAP sparse/0 directory")
    if not labels:
        missing.append("missing official label JSON frames")
    if missing_label_images:
        missing.append(f"missing source images for {len(missing_label_images)} label frames")
    if not scene_ready:
        missing.append(f"missing {scene_checkpoint.name}")
    if not affinity_ready:
        missing.append(f"missing {affinity_checkpoint.name}")
    missing_descriptors = [path.name for path in descriptor_paths if not path.is_file()]
    if missing_descriptors:
        missing.append("missing descriptor files: " + ", ".join(missing_descriptors))

    return {
        "scene": scene,
        "scene_dir": str(scene_dir),
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "model_dir": str(model_dir),
        "scene_iteration_dir": str(scene_iteration_dir),
        "affinity_iteration_dir": str(affinity_iteration_dir),
        "image_count": len(image_stems),
        "label_frame_count": len(labels),
        "test_set": labels,
        "data_ready": data_ready,
        "scene_checkpoint_path": str(scene_checkpoint),
        "scene_checkpoint_ready": scene_ready,
        "affinity_checkpoint_path": str(affinity_checkpoint),
        "affinity_checkpoint_ready": affinity_ready,
        "descriptor_paths": [str(path) for path in descriptor_paths],
        "descriptor_ready": descriptor_ready,
        "ready_for_same_protocol_export": data_ready and scene_ready and affinity_ready and descriptor_ready,
        "missing_label_images": missing_label_images,
        "missing": missing,
    }


def build_audit(
    *,
    repo: str | Path = DEFAULT_REPO,
    local_site: str | Path = DEFAULT_LOCAL_SITE,
    lerf_root: str | Path = DEFAULT_LERF_ROOT,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    scenes: Sequence[str] = DEFAULT_SCENES,
    iteration: int = 30000,
    scene_iteration: int | None = None,
    affinity_iteration: int | None = None,
) -> dict[str, Any]:
    repo_path = Path(repo)
    local_site_path = Path(local_site)
    lerf_root_path = Path(lerf_root)
    model_root_path = Path(model_root)
    scene_iter = scene_iteration if scene_iteration is not None else iteration
    affinity_iter = affinity_iteration if affinity_iteration is not None else iteration
    repo_ready, repo_missing = _check_repo(repo_path, local_site_path)
    scene_summaries = {
        scene: _check_scene(lerf_root_path, model_root_path, scene, scene_iter, affinity_iter)
        for scene in scenes
    }
    all_scenes_ready = repo_ready and all(scene["ready_for_same_protocol_export"] for scene in scene_summaries.values())
    return {
        "method": "LaGa",
        "repo": str(repo_path),
        "local_site": str(local_site_path),
        "lerf_root": str(lerf_root_path),
        "model_root": str(model_root_path),
        "iteration": iteration,
        "scene_iteration": scene_iter,
        "affinity_iteration": affinity_iter,
        "repo_ready": repo_ready,
        "repo_missing": repo_missing,
        "all_scenes_ready": all_scenes_ready,
        "scenes": scene_summaries,
        "next_action": (
            "Train LaGa scene and contrastive affinity feature checkpoints, then convert inference.ipynb descriptor generation into a batch mask export."
            if not all_scenes_ready
            else "Run the batch mask exporter and same-protocol LERF evaluator."
        ),
    }


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LaGa LERF Readiness Audit",
        "",
        f"- repo ready: {_yes(bool(summary['repo_ready']))}",
        f"- all scenes ready: {_yes(bool(summary['all_scenes_ready']))}",
        f"- scene iteration: {summary.get('scene_iteration', summary['iteration'])}",
        f"- affinity iteration: {summary.get('affinity_iteration', summary['iteration'])}",
        f"- next action: {summary['next_action']}",
        "",
        "| Scene | Data | Label frames | Scene ckpt | Affinity ckpt | Descriptors | Missing |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for scene, item in summary["scenes"].items():
        lines.append(
            "| {scene} | {data} | {labels} | {scene_ckpt} | {affinity} | {descriptors} | {missing} |".format(
                scene=scene,
                data=_yes(bool(item["data_ready"])),
                labels=item["label_frame_count"],
                scene_ckpt=_yes(bool(item["scene_checkpoint_ready"])),
                affinity=_yes(bool(item["affinity_checkpoint_ready"])),
                descriptors=_yes(bool(item["descriptor_ready"])),
                missing="; ".join(item["missing"]) if item["missing"] else "none",
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--local-site", type=Path, default=DEFAULT_LOCAL_SITE)
    parser.add_argument("--lerf-root", type=Path, default=DEFAULT_LERF_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument(
        "--scene-iteration",
        type=int,
        default=None,
        help="Scene Gaussian iteration. Defaults to --iteration.",
    )
    parser.add_argument(
        "--affinity-iteration",
        type=int,
        default=None,
        help="Contrastive affinity feature iteration. Defaults to --iteration.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = build_audit(
        repo=args.repo,
        local_site=args.local_site,
        lerf_root=args.lerf_root,
        model_root=args.model_root,
        scenes=args.scenes,
        iteration=args.iteration,
        scene_iteration=args.scene_iteration,
        affinity_iteration=args.affinity_iteration,
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
