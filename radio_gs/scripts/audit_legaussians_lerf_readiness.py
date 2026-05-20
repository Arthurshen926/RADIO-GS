#!/usr/bin/env python3
"""Audit whether LEGaussians is ready for same-protocol LERF evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_REPO = Path("/root/baselines/LEGaussians")
DEFAULT_LOCAL_SITE = Path("output/baselines/legaussians/local_site")
DEFAULT_LERF_ROOT = Path("/mnt/pool/sqy/3d_understanding/lerf_ovs")
DEFAULT_SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")
DEFAULT_OUTPUT_JSON = Path("paper/artifacts/legaussians_lerf_readiness_audit.json")
DEFAULT_OUTPUT_MD = Path("paper/artifacts/legaussians_lerf_readiness_audit.md")


def _first_existing(paths: Sequence[Path]) -> Path | None:
    for path in sorted(paths):
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
    for relative in ("train.py", "render_mask.py", "eval.py"):
        if not (repo / relative).is_file():
            missing.append(f"missing {relative}")
    if _first_existing(list((local_site / "simple_knn").glob("_C*.so"))) is None:
        missing.append("missing simple_knn extension in local_site")
    if _first_existing(list((local_site / "diff_gaussian_rasterization").glob("_C*.so"))) is None:
        missing.append("missing diff_gaussian_rasterization extension in local_site")
    return not missing, missing


def _check_scene(lerf_root: Path, scene: str) -> dict[str, Any]:
    scene_dir = lerf_root / scene
    image_dir = scene_dir / "images"
    label_dir = lerf_root / "label" / scene
    encoding = _first_existing(list(image_dir.glob("*encoding_indices.pt")))
    codebook = _first_existing(list(image_dir.glob("*codebook.pt")))
    labels = _label_frames(label_dir)
    image_stems = _image_stems(image_dir)
    missing_label_images = sorted(frame for frame in labels if frame not in image_stems)

    missing: list[str] = []
    if not scene_dir.is_dir():
        missing.append("missing scene directory")
    if not image_dir.is_dir():
        missing.append("missing images directory")
    if not labels:
        missing.append("missing official label JSON frames")
    if encoding is None:
        missing.append("missing encoding indices from LEGaussians preprocessing")
    if codebook is None:
        missing.append("missing codebook from LEGaussians preprocessing")
    if missing_label_images:
        missing.append(f"missing source images for {len(missing_label_images)} label frames")

    feature_ready = encoding is not None and codebook is not None
    return {
        "scene": scene,
        "scene_dir": str(scene_dir),
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "image_count": len(image_stems),
        "label_frame_count": len(labels),
        "test_set": labels,
        "encoding_indices_path": str(encoding) if encoding else None,
        "codebook_path": str(codebook) if codebook else None,
        "feature_preprocessing_complete": feature_ready,
        "ready_for_training_config": bool(scene_dir.is_dir() and image_dir.is_dir() and labels and feature_ready),
        "missing_label_images": missing_label_images,
        "missing": missing,
    }


def build_audit(
    *,
    repo: str | Path = DEFAULT_REPO,
    local_site: str | Path = DEFAULT_LOCAL_SITE,
    lerf_root: str | Path = DEFAULT_LERF_ROOT,
    scenes: Sequence[str] = DEFAULT_SCENES,
) -> dict[str, Any]:
    repo_path = Path(repo)
    local_site_path = Path(local_site)
    lerf_root_path = Path(lerf_root)
    repo_ready, repo_missing = _check_repo(repo_path, local_site_path)
    scene_summaries = {scene: _check_scene(lerf_root_path, scene) for scene in scenes}
    all_scenes_ready = repo_ready and all(scene["ready_for_training_config"] for scene in scene_summaries.values())
    return {
        "method": "LEGaussians",
        "repo": str(repo_path),
        "local_site": str(local_site_path),
        "lerf_root": str(lerf_root_path),
        "repo_ready": repo_ready,
        "repo_missing": repo_missing,
        "all_scenes_ready": all_scenes_ready,
        "scenes": scene_summaries,
        "next_action": (
            "Run LEGaussians dense feature quantization for each LERF scene before scheduling training/rendering."
            if not all_scenes_ready
            else "Generate scene configs, then schedule LEGaussians training/rendering/evaluation."
        ),
    }


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LEGaussians LERF Readiness Audit",
        "",
        f"- repo ready: {_yes(bool(summary['repo_ready']))}",
        f"- all scenes ready: {_yes(bool(summary['all_scenes_ready']))}",
        f"- next action: {summary['next_action']}",
        "",
        "| Scene | Images | Label frames | Quantized features | Missing |",
        "|---|---:|---:|---:|---|",
    ]
    for scene, item in summary["scenes"].items():
        lines.append(
            "| {scene} | {images} | {labels} | {features} | {missing} |".format(
                scene=scene,
                images=_yes(int(item["image_count"]) > 0),
                labels=item["label_frame_count"],
                features=_yes(bool(item["feature_preprocessing_complete"])),
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
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = build_audit(repo=args.repo, local_site=args.local_site, lerf_root=args.lerf_root, scenes=args.scenes)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
