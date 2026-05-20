#!/usr/bin/env python3
"""Export LaGa LERF prompt masks in the nested layout used by the evaluator."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES


DEFAULT_LAGA_REPO = Path("/root/baselines/LaGa")
DEFAULT_MODEL_ROOT = Path("output/baselines/laga/lerf_compat_20260520")
DEFAULT_LERF_ROOT = Path("/mnt/pool/sqy/3d_understanding/lerf_ovs")
DEFAULT_SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")
DEFAULT_SCENE_ITERATION = 30001
DEFAULT_AFFINITY_ITERATION = 30000
DEFAULT_MASK_THRESH = "0.5"
DEFAULT_CLIP_PRETRAINED = "laion2b_s34b_b88k"
MULTI_LVL_DIM = [16, 8, 8]


def prediction_root(model_root: Path, scene: str, mask_thresh: str | float = DEFAULT_MASK_THRESH) -> Path:
    return model_root / scene / f"predictions_mask_{mask_thresh}" / "renders_silhouette"


def _json_prompts(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts: list[str] = []
    seen: set[str] = set()
    for obj in payload.get("objects", []):
        prompt = str(obj.get("category", "")).strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
    return prompts


def load_scene_frame_prompts(lerf_root: Path, scene: str) -> dict[str, list[str]]:
    if scene not in SCENE_GT_FRAMES:
        choices = ", ".join(sorted(SCENE_GT_FRAMES))
        raise ValueError(f"Unknown scene {scene!r}; expected one of: {choices}")
    label_dir = lerf_root / "label" / scene
    output: dict[str, list[str]] = {}
    for frame in SCENE_GT_FRAMES[scene]:
        path = label_dir / f"{frame}.json"
        if path.is_file():
            output[frame] = _json_prompts(path)
    return output


def camera_lookup(cameras: Iterable[Any]) -> dict[str, Any]:
    lookup: dict[str, Any] = {}
    for camera in cameras:
        name = str(camera.image_name)
        stem = Path(name).stem
        lookup.setdefault(name, camera)
        lookup.setdefault(stem, camera)
        lookup.setdefault(f"{stem}.jpg", camera)
    return lookup


def scene_camera_lookup(scene: Any) -> dict[str, Any]:
    cameras = list(scene.getTrainCameras()) + list(scene.getTestCameras())
    return camera_lookup(cameras)


def resolve_camera(lookup: dict[str, Any], frame: str) -> Any:
    for key in (frame, f"{frame}.jpg", Path(frame).stem):
        if key in lookup:
            return lookup[key]
    choices = ", ".join(sorted(lookup)[:8])
    raise KeyError(f"missing camera for {frame}; sample known camera names: {choices}")


def normalize_scores(scores: np.ndarray, *, clip_quantile: float = 0.25) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).copy()
    if scores.size == 0:
        return scores
    lower = float(np.quantile(scores, clip_quantile))
    scores = np.clip(scores, lower, np.inf)
    scores -= float(scores.min())
    denom = float(scores.max())
    if denom <= 1e-12:
        scores.fill(0.0)
        return scores
    return scores / denom


def _safe_query_filename(query: str) -> str:
    return query.replace("/", "_")


def _repo_import_context(repo: Path) -> None:
    repo_str = str(repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _combined_laga_args(repo: Path, model_path: Path, target_cfg_file: str = "cfg_args") -> Namespace:
    _repo_import_context(repo)
    from arguments import ModelParams, PipelineParams

    parser = ArgumentParser(description="LaGa LERF export argument loader")
    ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument(
        "--target",
        default="contrastive_feature",
        const="contrastive_feature",
        nargs="?",
        choices=["scene", "seg", "feature", "coarse_seg_everything", "contrastive_feature", "xyz"],
    )
    parser.add_argument("--idx", default=0, type=int)
    parser.add_argument("--precomputed_mask", default=None, type=str)
    args_cmdline = parser.parse_args(["--model_path", str(model_path), "--target", "contrastive_feature"])

    cfg_text = "Namespace()"
    cfg_path = model_path / target_cfg_file
    if cfg_path.is_file():
        cfg_text = cfg_path.read_text(encoding="utf-8")
    args_cfgfile = eval(cfg_text, {"Namespace": Namespace})

    merged = vars(args_cfgfile).copy()
    merged.update({key: value for key, value in vars(args_cmdline).items() if value is not None})
    return Namespace(**merged)


def _load_laga_scene(
    *,
    repo: Path,
    model_path: Path,
    scene_iteration: int,
    affinity_iteration: int,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    _repo_import_context(repo)
    import torch
    from arguments import ModelParams, PipelineParams
    from scene import FeatureGaussianModel, Scene

    parser = ArgumentParser(description="LaGa LERF export parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    args = _combined_laga_args(repo, model_path)
    dataset = model.extract(args)
    dataset.need_features = True
    dataset.need_masks = True

    feature_gaussians = FeatureGaussianModel(dataset.feature_dim)

    scene = Scene(
        dataset,
        None,
        feature_gaussians,
        load_iteration=-1,
        feature_load_iteration=affinity_iteration,
        shuffle=False,
        mode="eval",
        target="contrastive_feature",
    )
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    return args, pipeline, scene, None, feature_gaussians, background


def _load_descriptors(model_path: Path, affinity_iteration: int) -> tuple[list[Any], list[Any], list[Any]]:
    import torch

    descriptor_dir = model_path / "point_cloud" / f"iteration_{affinity_iteration}"
    cluster_features = torch.load(descriptor_dir / "multi_lvl_cluster_features.pth")
    cluster_weights = torch.load(descriptor_dir / "multi_lvl_cluster_feature_weights.pth")
    seg_scores = torch.load(descriptor_dir / "multi_lvl_seg_scores.pth")
    return cluster_features, cluster_weights, seg_scores


def _load_clip(repo: Path, clip_pretrained: str) -> Any:
    _repo_import_context(repo)
    from clip_utils.clip_utils import OpenCLIPNetworkConfig, load_clip

    config = OpenCLIPNetworkConfig()
    config.clip_model_pretrained = clip_pretrained
    clip_model = load_clip(config)
    clip_model.eval()
    return clip_model


def _point_colors_for_prompt(
    *,
    clip_model: Any,
    cluster_features: list[Any],
    cluster_weights: list[Any],
    seg_scores: list[Any],
    feature_gaussians: Any,
    prompt: str,
    point_thresh: float,
    cosine_thresh: float,
    clip_quantile: float,
    num_per_cluster_features: int,
) -> Any:
    import torch
    from clip_utils import get_relevancy_cosine

    point_colors: list[np.ndarray] = []
    stack_of_cosine = []
    for lvl in range(len(seg_scores)):
        features = cluster_features[lvl]
        weights = cluster_weights[lvl].clone()
        seg_score = seg_scores[lvl]
        rel, pos, _neg = get_relevancy_cosine(clip_model, torch.nn.functional.normalize(features.cuda(), dim=-1, p=2), prompt)
        cluster_scores = (rel * weights).reshape([-1, num_per_cluster_features])
        cluster_scores, index = cluster_scores.max(dim=1)

        pos = pos.reshape([-1, num_per_cluster_features])
        batch_indices = torch.arange(pos.shape[0], device=pos.device)
        selected_pos = pos[batch_indices, index]
        stack_of_cosine.append(selected_pos[seg_score.argmax(dim=-1).detach().cpu().numpy()])

        cluster_colors = cluster_scores.detach().cpu().numpy()
        cluster_colors[cluster_colors < 0] = 0
        cluster_colors = np.expand_dims(cluster_colors, axis=1)
        point_colors.append(cluster_colors[seg_score.argmax(dim=-1).detach().cpu().numpy()])

    scores = np.stack(point_colors, axis=0).mean(axis=0)
    scores = normalize_scores(scores, clip_quantile=clip_quantile)

    cosine = torch.stack(stack_of_cosine, 0).max(dim=0)[0]
    scores[cosine.detach().cpu().numpy() < cosine_thresh] = 0
    scores[scores < point_thresh] = 0
    scores[scores != 0] = 1

    # The GUI applies a color-aware bilateral filter. For batch export we keep
    # this deterministic binary point mask to avoid requiring pytorch3d KNN at
    # every prompt; the downstream metric is computed on rendered binary masks.
    return torch.from_numpy(scores).float().cuda().repeat(1, 3), torch.from_numpy(scores).float().cuda()[:, 0:1]


def _render_prompt_mask(
    *,
    view: Any,
    feature_gaussians: Any,
    pipeline: Any,
    args: Any,
    background: Any,
    point_color: Any,
    point_mask: Any,
    threshold: int,
) -> np.ndarray:
    import torch
    from gaussian_renderer import render

    with torch.no_grad():
        rendered = render(
            view,
            feature_gaussians,
            pipeline.extract(args),
            background,
            override_color=point_color,
            override_mask=point_mask,
        )["render"]
        score = rendered[0].detach().float().cpu().numpy()
    return (score > (threshold / 255.0)).astype(np.uint8) * 255


def export_scene(
    *,
    repo: Path,
    lerf_root: Path,
    model_root: Path,
    scene: str,
    scene_iteration: int,
    affinity_iteration: int,
    clip_pretrained: str,
    mask_thresh: str,
    point_thresh: float,
    cosine_thresh: float,
    clip_quantile: float,
    num_per_cluster_features: int,
    image_threshold: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    import torch

    model_path = model_root / scene
    pred_root = prediction_root(model_root, scene, mask_thresh)
    pred_root.mkdir(parents=True, exist_ok=True)
    frame_prompts = load_scene_frame_prompts(lerf_root, scene)

    args, pipeline, laga_scene, _scene_gaussians, feature_gaussians, background = _load_laga_scene(
        repo=repo,
        model_path=model_path,
        scene_iteration=scene_iteration,
        affinity_iteration=affinity_iteration,
    )
    cluster_features, cluster_weights, seg_scores = _load_descriptors(model_path, affinity_iteration)
    clip_model = _load_clip(repo, clip_pretrained)
    lookup = scene_camera_lookup(laga_scene)

    written = 0
    skipped = 0
    for frame, prompts in frame_prompts.items():
        view = resolve_camera(lookup, frame)
        frame_dir = pred_root / frame
        frame_dir.mkdir(parents=True, exist_ok=True)
        for prompt in prompts:
            out_path = frame_dir / f"{_safe_query_filename(prompt)}.png"
            if out_path.exists() and not overwrite:
                skipped += 1
                continue
            torch.cuda.empty_cache()
            point_color, point_mask = _point_colors_for_prompt(
                clip_model=clip_model,
                cluster_features=cluster_features,
                cluster_weights=cluster_weights,
                seg_scores=seg_scores,
                feature_gaussians=feature_gaussians,
                prompt=prompt,
                point_thresh=point_thresh,
                cosine_thresh=cosine_thresh,
                clip_quantile=clip_quantile,
                num_per_cluster_features=num_per_cluster_features,
            )
            mask = _render_prompt_mask(
                view=view,
                feature_gaussians=feature_gaussians,
                pipeline=pipeline,
                args=args,
                background=background,
                point_color=point_color,
                point_mask=point_mask,
                threshold=image_threshold,
            )
            Image.fromarray(mask, mode="L").save(out_path)
            written += 1

    return {
        "scene": scene,
        "pred_root": str(pred_root),
        "frames": len(frame_prompts),
        "queries": int(sum(len(prompts) for prompts in frame_prompts.values())),
        "written": written,
        "skipped": skipped,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_LAGA_REPO)
    parser.add_argument("--lerf-root", type=Path, default=DEFAULT_LERF_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES), choices=DEFAULT_SCENES)
    parser.add_argument("--scene-iteration", type=int, default=DEFAULT_SCENE_ITERATION)
    parser.add_argument("--affinity-iteration", type=int, default=DEFAULT_AFFINITY_ITERATION)
    parser.add_argument("--clip-pretrained", default=os.environ.get("LAGA_CLIP_PRETRAINED", DEFAULT_CLIP_PRETRAINED))
    parser.add_argument("--mask-thresh", default=DEFAULT_MASK_THRESH)
    parser.add_argument("--point-thresh", type=float, default=0.5)
    parser.add_argument("--cosine-thresh", type=float, default=0.0)
    parser.add_argument("--clip-quantile", type=float, default=0.25)
    parser.add_argument("--num-per-cluster-features", type=int, default=20)
    parser.add_argument("--image-threshold", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    warnings.filterwarnings("ignore", category=FutureWarning)
    summaries = [
        export_scene(
            repo=args.repo,
            lerf_root=args.lerf_root,
            model_root=args.model_root,
            scene=scene,
            scene_iteration=args.scene_iteration,
            affinity_iteration=args.affinity_iteration,
            clip_pretrained=args.clip_pretrained,
            mask_thresh=str(args.mask_thresh),
            point_thresh=args.point_thresh,
            cosine_thresh=args.cosine_thresh,
            clip_quantile=args.clip_quantile,
            num_per_cluster_features=args.num_per_cluster_features,
            image_threshold=args.image_threshold,
            overwrite=args.overwrite,
        )
        for scene in args.scenes
    ]
    report = {
        "method": "LaGa",
        "model_root": str(args.model_root),
        "lerf_root": str(args.lerf_root),
        "scene_iteration": args.scene_iteration,
        "affinity_iteration": args.affinity_iteration,
        "mask_thresh": str(args.mask_thresh),
        "scenes": summaries,
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
