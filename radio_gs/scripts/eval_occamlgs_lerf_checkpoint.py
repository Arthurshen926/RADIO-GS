#!/usr/bin/env python3
"""Stream OccamLGS checkpoints through the strict LERF-2D paper readout.

The 512-dimensional full-resolution renders are consumed on GPU one frame at
a time. Only relevance maps are retained temporarily in memory; no raw feature
maps are written to disk.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch

from radio_gs.evaluation.openclip_readout import OpenCLIPTextScorer
from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES
from radio_gs.scripts.eval_prerendered_lerf_features import (
    evaluate_relevance_maps,
    load_lerf_objects,
    resolve_protocol_config,
)


class ProtocolError(RuntimeError):
    """Raised before GPU evaluation when checkpoint visibility is ambiguous."""


def _read_namespace_config(path: Path) -> dict[str, object]:
    expression = ast.parse(path.read_text(encoding="utf-8"), mode="eval").body
    if not isinstance(expression, ast.Call):
        raise ProtocolError(f"{path}: expected Namespace(...)")
    if not isinstance(expression.func, ast.Name) or expression.func.id != "Namespace":
        raise ProtocolError(f"{path}: expected Namespace(...)")
    if expression.args:
        raise ProtocolError(f"{path}: positional Namespace arguments are unsupported")
    config: dict[str, object] = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ProtocolError(f"{path}: expanded Namespace arguments are unsupported")
        config[keyword.arg] = ast.literal_eval(keyword.value)
    return config


def validate_label_camera_roles(
    label_frames: Sequence[str],
    train_camera_names: Sequence[str],
    test_camera_names: Sequence[str],
    *,
    require_test_only: bool = False,
) -> dict[str, dict[str, object]]:
    train_by_stem = {Path(name).stem: name for name in train_camera_names}
    test_by_stem = {Path(name).stem: name for name in test_camera_names}
    manifest: dict[str, dict[str, object]] = {}
    for frame in label_frames:
        in_train = frame in train_by_stem
        in_test = frame in test_by_stem
        if in_train and in_test:
            raise ValueError(
                f"{frame}: ambiguous camera name appears in both train and test splits"
            )
        if not in_train and not in_test:
            raise ValueError(f"{frame}: no exact camera-name match in train or test split")
        role = "test" if in_test else "train"
        if require_test_only and role != "test":
            raise ValueError(f"{frame}: --require-test-only rejected resolved role={role}")
        manifest[frame] = {
            "resolved_camera_name": (test_by_stem if in_test else train_by_stem)[frame],
            "camera_role": role,
        }
    return manifest


def _load_occam_modules(occam_root: Path) -> tuple[Any, Any, Any, Any]:
    root = str(occam_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from arguments import OptimizationParams
    from gaussian_renderer import render
    from scene import Scene
    from scene.gaussian_model import GaussianModel

    return OptimizationParams, render, Scene, GaussianModel


def _optimization_defaults(optimization_params: Any) -> Any:
    parser = ArgumentParser(add_help=False)
    group = optimization_params(parser)
    return group.extract(parser.parse_args([]))


def _dataset_args(source: Path, model: Path, feature_level: int) -> SimpleNamespace:
    return SimpleNamespace(
        sh_degree=3,
        source_path=str(source.resolve()),
        model_path=str(model.resolve()),
        images="images",
        depths="",
        resolution=-1,
        white_background=False,
        train_test_exp=False,
        data_device="cuda",
        eval=True,
        language_features_name="language_features",
        feature_level=feature_level,
        lf_path=str((source / "language_features").resolve()),
    )


def _pipeline_args() -> SimpleNamespace:
    return SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )


def _load_language_checkpoint(
    gaussian_model: Any,
    checkpoint: Path,
    optimization: Any,
    device: torch.device,
) -> Any:
    gaussians = gaussian_model(3)
    model_params, _ = torch.load(checkpoint, map_location=device)
    gaussians.restore_language_features(model_params, optimization)
    gaussians.optimizer = None
    return gaussians


def evaluate_scene(args: argparse.Namespace) -> dict[str, object]:
    if args.device != "cuda":
        raise ValueError("OccamLGS checkpoint streaming currently requires --device cuda")
    device = torch.device(args.device)
    checkpoint_config_path = args.model / "cfg_args"
    if not checkpoint_config_path.exists():
        raise FileNotFoundError(checkpoint_config_path)
    checkpoint_config = _read_namespace_config(checkpoint_config_path)
    checkpoint_eval = checkpoint_config.get("eval")
    if checkpoint_eval is not True and not args.allow_training_visible_checkpoint:
        raise ProtocolError(
            f"{checkpoint_config_path}: checkpoint cfg has eval={checkpoint_eval!r}; "
            "the geometry is not a held-out-view reproduction. Pass "
            "--allow-training-visible-checkpoint only for an explicitly labeled diagnostic."
        )
    frames = SCENE_GT_FRAMES[args.scene]
    objects_by_frame = load_lerf_objects(args.label_root, frames=frames)
    if set(objects_by_frame) != set(frames):
        missing = sorted(set(frames) - set(objects_by_frame))
        raise FileNotFoundError(f"missing annotation frames: {missing}")

    optimization_params, render, scene_type, gaussian_model = _load_occam_modules(args.occam_root)
    optimization = _optimization_defaults(optimization_params)
    dataset = _dataset_args(args.source, args.model, feature_level=1)
    bootstrap_gaussians = gaussian_model(3)
    scene = scene_type(
        dataset,
        bootstrap_gaussians,
        load_iteration=args.iteration,
        shuffle=False,
        include_feature=True,
    )
    train_views = list(scene.getTrainCameras())
    test_views = list(scene.getTestCameras())
    camera_manifest = validate_label_camera_roles(
        frames,
        [view.image_name for view in train_views],
        [view.image_name for view in test_views],
        require_test_only=args.require_test_only,
    )
    views_by_stem = {
        Path(view.image_name).stem: view
        for view in [*train_views, *test_views]
    }
    scene.gaussians = None
    del bootstrap_gaussians
    torch.cuda.empty_cache()

    scorer = OpenCLIPTextScorer(
        device,
        model_name=args.openclip_model,
        pretrained=args.openclip_pretrained,
    )
    relevance_by_frame: dict[str, list[np.ndarray]] = {frame: [] for frame in frames}
    render_shapes: dict[str, list[int]] = {}
    background = torch.zeros(3, dtype=torch.float32, device=device)
    pipeline = _pipeline_args()

    with torch.inference_mode():
        for level in (1, 2, 3):
            checkpoint = args.model / f"chkpnt{args.iteration}_langfeat_{level}.pth"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            gaussians = _load_language_checkpoint(
                gaussian_model,
                checkpoint,
                optimization,
                device,
            )
            for frame in frames:
                view = views_by_stem[frame]
                rendered = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    include_feature=True,
                )["render"]
                feature = rendered.permute(1, 2, 0).contiguous()
                objects = objects_by_frame[frame]
                relevance = scorer.relevance(feature.unsqueeze(0), [obj.query for obj in objects])
                relevance_by_frame[frame].append(relevance[0].detach().cpu().float().numpy())
                render_shapes[frame] = [int(feature.shape[0]), int(feature.shape[1])]
                del rendered, feature, relevance
                torch.cuda.empty_cache()
            del gaussians
            torch.cuda.empty_cache()

    stacked_relevance = {
        frame: np.stack(level_maps, axis=0)
        for frame, level_maps in relevance_by_frame.items()
    }
    protocol_args = SimpleNamespace(
        protocol_profile="occam_langsplat_paper",
        mask_thresh=None,
        activation_kernel=None,
        smooth_kernel=None,
        feature_mode=None,
    )
    protocol = resolve_protocol_config(protocol_args)
    result = evaluate_relevance_maps(
        objects_by_frame,
        stacked_relevance,
        mask_thresh=float(protocol["mask_thresh"]),
        activation_kernel=int(protocol["activation_kernel"]),
        smooth_kernel=int(protocol["smooth_kernel"]),
        filter_implementation=str(protocol["filter_implementation"]),
        mask_smoothing_implementation=str(protocol["mask_smoothing_implementation"]),
        resize_policy=str(protocol["resize_policy"]),
    )
    for frame, shape in render_shapes.items():
        camera_manifest[frame]["render_height"] = shape[0]
        camera_manifest[frame]["render_width"] = shape[1]
    result.update(
        {
            "method": "OccamLGS",
            "scene": args.scene,
            "protocol": "strict streamed Occam/LangSplat LERF-2D paper-profile readout",
            "protocol_config": protocol,
            "camera_manifest": camera_manifest,
            "source": str(args.source),
            "model": str(args.model),
            "iteration": args.iteration,
            "checkpoints": [
                str(args.model / f"chkpnt{args.iteration}_langfeat_{level}.pth")
                for level in (1, 2, 3)
            ],
            "raw_feature_cache": "none; frame-local GPU streaming",
            "checkpoint_config": {
                "path": str(checkpoint_config_path),
                "source_path": checkpoint_config.get("source_path"),
                "eval": checkpoint_eval,
                "geometry_visibility": (
                    "held_out_split_configured"
                    if checkpoint_eval is True
                    else "all_registered_views_training_visible"
                ),
            },
            "comparison_status": (
                "strict_checkpoint_visibility"
                if checkpoint_eval is True
                else "diagnostic_only_training_visible_checkpoint"
            ),
        }
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occam-root", type=Path, default=Path("/root/baselines/OccamLGS"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--scene", choices=sorted(SCENE_GT_FRAMES), required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--require-test-only",
        action="store_true",
        help="Optional diagnostic guard; released LERF evaluation resolves exact names across both splits.",
    )
    parser.add_argument(
        "--allow-training-visible-checkpoint",
        action="store_true",
        help=(
            "Explicitly permit a checkpoint whose cfg_args has eval!=True; the result "
            "is labeled diagnostic-only and cannot be called held-out."
        ),
    )
    parser.add_argument("--openclip-model", default="ViT-B-16")
    parser.add_argument("--openclip-pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = evaluate_scene(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = result["query_micro"]
    print(
        f"{args.scene}: LocAcc={metrics['loc_acc']:.4f} "
        f"mIoU={metrics['miou']:.4f} objects={metrics['objects']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
