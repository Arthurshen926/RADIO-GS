#!/usr/bin/env python3
"""Run VALA feature lifting for the ScanNet paper/code cohorts.

The public VALA ScanNet shell script is not directly usable with the prepared
RADIO-GS ScanNet assets: it expects ``transforms_train.json`` and a GraphDECO
checkpoint, while our RGB Gaussian assets are stored as trained PLYs plus
gsplat-style state dicts.  This wrapper keeps the VALA core aggregation code
unchanged, builds a lightweight staging tree with the metadata VALA expects,
loads Gaussians from the trained PLY and aggregates VALA language features.
The historical mesh-kNN evaluator remains available only as a compatibility
diagnostic.  Paper-facing evaluation must use
``eval_vala_scannet_checkpoint_gaussian_protocol`` instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
VALA_ROOT = Path("/root/baselines/VALA-upstream-48902a5")
VALA_PAPER8_SCENES = (
    "scene0000_00",
    "scene0062_00",
    "scene0070_00",
    "scene0097_00",
    "scene0140_00",
    "scene0347_00",
    "scene0400_00",
    "scene0590_00",
)
VALA_CURRENT9_SCENES = VALA_PAPER8_SCENES + ("scene0645_00",)
# Backward-compatible import used by older report builders.
VALA8_SCENES = VALA_PAPER8_SCENES

SPLIT_IDS = {
    "19": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36),
    "15": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 33, 34),
    "10": (1, 2, 4, 5, 6, 7, 8, 9, 10, 33),
}

NYU40_ID_TO_NAME = {
    0: "unlabeled",
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    14: "desk",
    16: "curtain",
    24: "refrigerator",
    28: "showercurtain",
    33: "toilet",
    34: "sink",
    36: "bathtub",
}


@dataclass(frozen=True)
class ScenePaths:
    scene: str
    source: Path
    staged: Path
    model: Path
    label_ply: Path


def _ensure_vala_imports(vala_root: Path) -> None:
    sys.path.insert(0, str(vala_root))
    os.chdir(str(vala_root))


def _symlink_or_replace(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            return
        if dst.is_dir() and not dst.is_symlink():
            raise FileExistsError(f"Refusing to replace directory: {dst}")
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)


def _camera_angle_x(meta: dict) -> float:
    if "camera_angle_x" in meta:
        return float(meta["camera_angle_x"])
    fl_x = float(meta["fl_x"])
    width = float(meta["w"])
    return float(2.0 * math.atan(width / (2.0 * fl_x)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _enrich_vala_transforms(source_path: Path) -> tuple[dict, list[str]]:
    """Add fields required by VALA without changing the source frame cohort."""
    with source_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    frames = []
    frame_ids = []
    for frame in raw["frames"]:
        stem = Path(str(frame["file_path"])).stem
        frame_ids.append(stem)
        frames.append(
            {
                "file_path": f"color/{stem}.jpg",
                "transform_matrix": frame["transform_matrix"],
                "language_features_path": f"langsplat/language_features/{stem}",
                "instance_masks_path": "",
                "correlation_path": "",
            }
        )
    return {"camera_angle_x": _camera_angle_x(raw), "frames": frames}, frame_ids


def _stage_scene(
    scene: str,
    *,
    data_root: Path,
    model_root: Path,
    staging_root: Path,
    test_loader_limit: int = 0,
) -> ScenePaths:
    source = data_root / scene
    model = model_root / scene / "og_rgb_3dgs"
    label_ply = source / f"{scene}_vh_clean_2.labels.ply"
    point_ply = model / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    fallback_transforms = source / "transforms.json"
    train_transforms = source / "transforms_train.json"
    test_transforms = source / "transforms_test.json"
    if not train_transforms.is_file():
        train_transforms = fallback_transforms
    if not test_transforms.is_file():
        test_transforms = fallback_transforms
    for required in (train_transforms, test_transforms, source / "color", source / "language_features", label_ply, point_ply):
        if not required.exists():
            raise FileNotFoundError(required)

    staged = staging_root / scene
    staged.mkdir(parents=True, exist_ok=True)
    _symlink_or_replace(source / "color", staged / "color")
    _symlink_or_replace(source / "language_features", staged / "language_features")
    _symlink_or_replace(source / "language_features", staged / "langsplat" / "language_features")
    _symlink_or_replace(point_ply, staged / "points3d.ply")

    staged_splits = {}
    for split, source_meta in (("train", train_transforms), ("test", test_transforms)):
        staged_meta, frame_ids = _enrich_vala_transforms(source_meta)
        source_frame_ids = list(frame_ids)
        if split == "test" and test_loader_limit > 0:
            staged_meta["frames"] = staged_meta["frames"][: int(test_loader_limit)]
            frame_ids = frame_ids[: int(test_loader_limit)]
        staged_path = staged / f"transforms_{split}.json"
        staged_path.write_text(json.dumps(staged_meta, indent=2) + "\n", encoding="utf-8")
        staged_splits[split] = {
            "source": str(source_meta.resolve()),
            "source_sha256": _sha256(source_meta),
            "source_num_frames": len(source_frame_ids),
            "source_frame_ids": source_frame_ids,
            "staged_num_frames": len(frame_ids),
            "staged_frame_ids": frame_ids,
            "staging_change": "reader-required paths only; frame IDs and poses preserved",
        }
        if split == "test" and test_loader_limit > 0:
            staged_splits[split]["staging_change"] = (
                "reader-required paths plus loader-only test-camera truncation; "
                "feature extraction consumes train cameras exclusively"
            )
    (staged / "staging_manifest.json").write_text(
        json.dumps({"scene": scene, "splits": staged_splits}, indent=2) + "\n",
        encoding="utf-8",
    )

    return ScenePaths(scene=scene, source=source, staged=staged, model=model, label_ply=label_ply)


def _make_vala_params(paths: ScenePaths, *, resolution: int) -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    dataset = SimpleNamespace(
        sh_degree=3,
        source_path=str(paths.staged.resolve()),
        model_path=str(paths.model.resolve()),
        images="color",
        depths="",
        resolution=int(resolution),
        white_background=False,
        train_test_exp=False,
        data_device="cuda",
        eval=True,
        language_features_name="language_features",
        feature_level=0,
    )
    pipeline = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )
    opt = SimpleNamespace(
        iterations=30000,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30000,
        feature_lr=0.0025,
        opacity_lr=0.025,
        scaling_lr=0.005,
        rotation_lr=0.001,
        percent_dense=0.01,
        lambda_dssim=0.2,
        densification_interval=100,
        opacity_reset_interval=3000,
        densify_from_iter=500,
        densify_until_iter=15000,
        densify_grad_threshold=0.0004,
        depth_l1_weight_init=1.0,
        depth_l1_weight_final=0.01,
        random_background=False,
        optimizer_type="default",
    )
    return dataset, pipeline, opt


def _extract_vala_features(
    paths: ScenePaths,
    *,
    vala_root: Path,
    checkpoint: Path,
    resolution: int,
    feature_level: int,
    batch_size: int,
    weight_threshold: float,
    max_views: int,
    force: bool,
    allow_proxy_significance: bool,
) -> Path:
    _ensure_vala_imports(vala_root)
    from scene import Scene
    from gaussian_renderer import GaussianModel
    from gaussian_renderer import render

    if checkpoint.exists() and not force:
        return checkpoint

    dataset, pipeline, opt = _make_vala_params(paths, resolution=resolution)
    dataset.feature_level = int(feature_level)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False, include_feature=True)
        num_gaussians = int(gaussians.get_xyz.shape[0])
        gaussians.max_radii2D = torch.zeros(num_gaussians, device="cuda")
        gaussians.xyz_gradient_accum = torch.zeros((num_gaussians, 1), device="cuda")
        gaussians.denom = torch.zeros((num_gaussians, 1), device="cuda")
        gaussians.training_setup(opt)
        bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        views = scene.getTrainCameras()
        if max_views > 0:
            views = views[: int(max_views)]
        language_dir = paths.staged / "langsplat" / "language_features"
        print(
            f"Aggregating VALA features for {paths.scene}: "
            f"{len(views)} views, feature_level={feature_level}, batch_size={batch_size}",
            flush=True,
        )
        for view_idx, view in enumerate(tqdm(views, desc=f"VALA {paths.scene}")):
            render_pkg = render(view, gaussians, pipeline, bg)
            gt_language_feature, gt_mask = view.get_language_feature(
                language_feature_dir=str(language_dir),
                feature_level=int(feature_level),
            )
            info = render_pkg["info"]
            means2d_all = info["means2d"][0]
            activated = info.get("activated")
            exact_significance = info.get("significance")
            if activated is not None and exact_significance is not None:
                mask = activated[0] > 0
                significance = exact_significance[0, mask].float()
            elif allow_proxy_significance:
                radii = info["radii"][0]
                height, width = int(gt_language_feature.shape[1]), int(gt_language_feature.shape[2])
                in_frame = (
                    (means2d_all[:, 0] >= 0)
                    & (means2d_all[:, 0] < width)
                    & (means2d_all[:, 1] >= 0)
                    & (means2d_all[:, 1] < height)
                )
                mask = (radii > 0) & in_frame
                opacity = info.get("opacities")
                if opacity is None:
                    significance = torch.ones_like(radii[mask], dtype=torch.float32)
                else:
                    significance = opacity[0].float()[mask] * torch.sqrt(
                        radii[mask].float().clamp_min(1.0)
                    )
            else:
                raise RuntimeError(
                    "Pinned VALA gsplat did not return info['activated'] and "
                    "info['significance']; refusing to substitute a visibility proxy"
                )
            if not bool(mask.any()):
                continue
            gaussians.accumulate_gaussian_feature_per_view_robust(
                gt_language_feature.permute(1, 2, 0),
                gt_mask.squeeze(0),
                mask,
                significance,
                means2d_all[mask],
                tau_mass=0.9,
                tau_abs=0.01,
            )
            if view_idx % 10 == 0:
                torch.cuda.empty_cache()
        gaussians.finalize_gaussian_features_robust(w_thr=float(weight_threshold))
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save((gaussians.capture_language_feature(), 0), str(checkpoint))
        print(f"VALA checkpoint saved to: {checkpoint}", flush=True)
    return checkpoint


def _load_vala_language_checkpoint(path: Path) -> tuple[np.ndarray, np.ndarray]:
    model_args, _ = torch.load(str(path), map_location="cpu")
    xyz = model_args[1].detach().float().cpu().numpy()
    lang = model_args[7].detach().float().cpu()
    lang = F.normalize(lang, dim=-1).numpy()
    return xyz.astype(np.float32, copy=False), lang.astype(np.float32, copy=False)


def _load_label_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float32)
    rgb = np.stack([np.asarray(v["red"]), np.asarray(v["green"]), np.asarray(v["blue"])], axis=1).astype(np.uint8)
    labels = np.asarray(v["label"], dtype=np.int32)
    return xyz, rgb, labels


def _load_text_features(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {str(k): np.asarray(v, dtype=np.float32) for k, v in payload.items()}


def _text_matrix(split: str, text_features: dict[str, np.ndarray]) -> tuple[list[int], torch.Tensor]:
    ids = list(SPLIT_IDS[split])
    vectors = []
    missing = []
    for class_id in ids:
        name = NYU40_ID_TO_NAME[class_id]
        if name not in text_features:
            missing.append(name)
            continue
        vectors.append(text_features[name])
    if missing:
        raise KeyError(f"Missing VALA text features for split {split}: {missing}")
    text = torch.from_numpy(np.stack(vectors, axis=0)).float()
    return ids, F.normalize(text, dim=-1)


def _classify_points(
    point_xyz: np.ndarray,
    gaussian_xyz: np.ndarray,
    gaussian_features: np.ndarray,
    text: torch.Tensor,
    *,
    k: int,
    chunk: int,
) -> np.ndarray:
    tree = cKDTree(gaussian_xyz)
    pred_chunks: list[np.ndarray] = []
    text_cpu = text.float().cpu()
    for start in range(0, point_xyz.shape[0], chunk):
        end = min(start + chunk, point_xyz.shape[0])
        dist, idx = tree.query(point_xyz[start:end], k=int(k), workers=-1)
        if int(k) == 1:
            idx = idx[:, None]
            dist = dist[:, None]
        feats = torch.from_numpy(gaussian_features[idx]).float()
        weights = torch.from_numpy(1.0 / np.maximum(dist, 1e-4)).float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        point_feat = F.normalize((feats * weights.unsqueeze(-1)).sum(dim=1), dim=-1)
        logits = point_feat @ text_cpu.T
        pred_chunks.append(logits.argmax(dim=-1).cpu().numpy().astype(np.int32))
    return np.concatenate(pred_chunks, axis=0)


def _compute_metrics(gt: np.ndarray, pred: np.ndarray, class_ids: Iterable[int]) -> dict[str, object]:
    per_class = {}
    ious = []
    accs = []
    valid_ids = [int(x) for x in class_ids]
    for class_id in valid_ids:
        gt_mask = gt == class_id
        pred_mask = pred == class_id
        if not gt_mask.any():
            continue
        inter = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        iou = float(inter / union) if union > 0 else 0.0
        acc = float(inter / gt_mask.sum())
        per_class[str(class_id)] = {
            "name": NYU40_ID_TO_NAME.get(class_id, str(class_id)),
            "iou": iou,
            "acc": acc,
            "gt_points": int(gt_mask.sum()),
            "pred_points": int(pred_mask.sum()),
        }
        ious.append(iou)
        accs.append(acc)
    valid = np.isin(gt, valid_ids)
    overall_acc = float((gt[valid] == pred[valid]).mean()) if valid.any() else 0.0
    return {
        "mIoU": float(np.mean(ious)) if ious else 0.0,
        "mAcc": float(np.mean(accs)) if accs else 0.0,
        "overall_acc": overall_acc,
        "per_class": per_class,
    }


def _write_pred_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, labels: np.ndarray, pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    elements = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("label", "i4"),
            ("pred_label", "i4"),
        ],
    )
    elements["x"] = xyz[:, 0]
    elements["y"] = xyz[:, 1]
    elements["z"] = xyz[:, 2]
    elements["red"] = rgb[:, 0]
    elements["green"] = rgb[:, 1]
    elements["blue"] = rgb[:, 2]
    elements["label"] = labels.astype(np.int32, copy=False)
    elements["pred_label"] = pred.astype(np.int32, copy=False)
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))


def _evaluate_scene(
    paths: ScenePaths,
    checkpoint: Path,
    *,
    output_root: Path,
    text_feature_path: Path,
    knn: int,
    point_chunk: int,
) -> dict[str, object]:
    gaussian_xyz, gaussian_features = _load_vala_language_checkpoint(checkpoint)
    point_xyz, point_rgb, point_labels = _load_label_ply(paths.label_ply)
    text_features = _load_text_features(text_feature_path)
    scene_result = {
        "scene": paths.scene,
        "checkpoint": str(checkpoint),
        "num_gaussians": int(gaussian_xyz.shape[0]),
        "num_points": int(point_xyz.shape[0]),
        "splits": {},
    }
    vis_dir = output_root / "visualizations" / paths.scene
    for split in ("19", "15", "10"):
        class_ids, text = _text_matrix(split, text_features)
        pred_relative = _classify_points(
            point_xyz,
            gaussian_xyz,
            gaussian_features,
            text,
            k=knn,
            chunk=point_chunk,
        )
        pred_labels = np.asarray([class_ids[int(i)] for i in pred_relative], dtype=np.int32)
        invalid = ~np.isin(point_labels, class_ids)
        pred_labels[invalid] = 0
        split_labels = point_labels.copy()
        split_labels[invalid] = 0
        metrics = _compute_metrics(split_labels, pred_labels, class_ids)
        scene_result["splits"][split] = metrics
        _write_pred_ply(vis_dir / f"pred_split_{split}.ply", point_xyz, point_rgb, split_labels, pred_labels)
        _write_pred_ply(vis_dir / f"gt_split_{split}.ply", point_xyz, point_rgb, split_labels, split_labels)
    return scene_result


def _write_reports(output_root: Path, payload: dict[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "vala_scannet_vala8_results.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# VALA ScanNet VALA8 Local Reproduction",
        "",
        "Protocol: VALA-aligned ScanNet-8 scenes, point-level open-vocabulary querying.",
        "",
        "| Split | mIoU | mAcc | Overall Acc |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split, metrics in payload["macro"].items():
        lines.append(
            f"| {split} | {metrics['mIoU']:.4f} | {metrics['mAcc']:.4f} | {metrics['overall_acc']:.4f} |"
        )
    lines.extend(["", "| Scene | Split | mIoU | mAcc |", "| --- | --- | ---: | ---: |"])
    for scene in payload["scenes"]:
        for split, metrics in scene["splits"].items():
            lines.append(f"| {scene['scene']} | {split} | {metrics['mIoU']:.4f} | {metrics['mAcc']:.4f} |")
    (output_root / "vala_scannet_vala8_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _macro(results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    out = {}
    for split in ("19", "15", "10"):
        vals = {"mIoU": [], "mAcc": [], "overall_acc": []}
        for item in results:
            metrics = item["splits"][split]
            for key in vals:
                vals[key].append(float(metrics[key]))
        out[split] = {key: float(np.mean(value)) for key, value in vals.items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "dataset/scannet_og")
    parser.add_argument("--model-root", type=Path, default=REPO_ROOT / "output/3dgs_models/scannet_og")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Output root. paper8 defaults to the frozen exact-protocol root; "
            "code9 and custom scene lists require an explicit root."
        ),
    )
    parser.add_argument("--vala-root", type=Path, default=VALA_ROOT)
    parser.add_argument("--text-features", type=Path, default=None)
    parser.add_argument("--cohort", choices=("paper8", "code9"), default="paper8")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--feature-level", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--weight-threshold", type=float, default=1e-5)
    parser.add_argument("--max-views", type=int, default=0, help="Debug only; 0 uses all views.")
    parser.add_argument(
        "--test-loader-limit",
        type=int,
        default=0,
        help=(
            "Limit eagerly loaded test cameras to reduce memory. Safe for extract-only "
            "runs because VALA aggregates scene.getTrainCameras() exclusively; 0 keeps all."
        ),
    )
    parser.add_argument("--knn", type=int, default=4)
    parser.add_argument("--point-chunk", type=int, default=8192)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--extract-only",
        dest="extract_only",
        action="store_true",
        default=True,
        help="Canonical default: stop after VALA semantic extraction.",
    )
    mode.add_argument(
        "--legacy-mesh-eval",
        dest="extract_only",
        action="store_false",
        help="Diagnostic only: run the superseded mesh-kNN compatibility readout.",
    )
    parser.add_argument(
        "--allow-proxy-significance",
        action="store_true",
        help="Legacy diagnostic only; strict runs fail if official significance is unavailable.",
    )
    args = parser.parse_args()
    if args.test_loader_limit < 0:
        parser.error("--test-loader-limit must be non-negative")

    vala_root = args.vala_root.resolve()
    scenes = args.scenes
    if scenes is None:
        scenes = list(VALA_PAPER8_SCENES if args.cohort == "paper8" else VALA_CURRENT9_SCENES)
    else:
        canonical_scenes = list(
            VALA_PAPER8_SCENES if args.cohort == "paper8" else VALA_CURRENT9_SCENES
        )
        if list(scenes) != canonical_scenes:
            parser.error(
                f"--cohort {args.cohort} rejects a mismatched --scenes list; "
                "use the Gaussian evaluator's --cohort custom for subset diagnostics"
            )
    if args.output_root is None:
        if args.cohort != "paper8":
            parser.error("--output-root is required for the code9 sensitivity")
        output_root = (
            REPO_ROOT
            / "output/protocol_audit_20260801/vala/"
            "scannet_official_significance_paper8_v2"
        ).resolve()
    else:
        output_root = args.output_root.resolve()
    text_feature_path = (
        args.text_features.resolve()
        if args.text_features is not None
        else vala_root / "autolabel" / "text_features.json"
    )
    staging_root = output_root / "staged"
    results = []
    extracted = []
    for scene in scenes:
        paths = _stage_scene(
            scene,
            data_root=args.data_root.resolve(),
            model_root=args.model_root.resolve(),
            staging_root=staging_root,
            test_loader_limit=args.test_loader_limit,
        )
        checkpoint = output_root / "checkpoints" / scene / f"chkpnt30000_langfeat_{args.feature_level}_stochastic_gate.pth"
        if not args.skip_extract:
            checkpoint = _extract_vala_features(
                paths,
                vala_root=vala_root,
                checkpoint=checkpoint,
                resolution=args.resolution,
                feature_level=args.feature_level,
                batch_size=args.batch_size,
                weight_threshold=args.weight_threshold,
                max_views=args.max_views,
                force=args.force_extract,
                allow_proxy_significance=args.allow_proxy_significance,
            )
        if not checkpoint.exists():
            raise FileNotFoundError(f"VALA feature checkpoint missing for {scene}: {checkpoint}")
        extracted.append({"scene": scene, "checkpoint": str(checkpoint)})
        if args.extract_only:
            continue
        results.append(
            _evaluate_scene(
                paths,
                checkpoint,
                output_root=output_root,
                text_feature_path=text_feature_path,
                knn=args.knn,
                point_chunk=args.point_chunk,
            )
        )
    payload = {
        "method": (
            "VALA official-significance feature extraction"
            if args.extract_only
            else "VALA legacy mesh-kNN compatibility reproduction"
        ),
        "cohort": args.cohort,
        "cohort_scenes": scenes,
        "vala_root": str(vala_root),
        "extracted": extracted,
        "scenes": results,
        "macro": _macro(results) if results else {},
        "settings": {
            "feature_level": args.feature_level,
            "resolution": args.resolution,
            "batch_size": args.batch_size,
            "weight_threshold": args.weight_threshold,
            "max_views": args.max_views,
            "test_loader_limit": args.test_loader_limit,
            "knn": args.knn,
            "allow_proxy_significance": args.allow_proxy_significance,
            "paper_facing_evaluator": "radio_gs.scripts.eval_vala_scannet_checkpoint_gaussian_protocol",
        },
    }
    _write_reports(output_root, payload)
    print(json.dumps(payload["macro"], indent=2))
    print(f"Wrote {output_root / 'vala_scannet_vala8_results.json'}")


if __name__ == "__main__":
    main()
