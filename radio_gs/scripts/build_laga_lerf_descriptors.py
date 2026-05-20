#!/usr/bin/env python3
"""Build LaGa descriptor files from trained contrastive affinity Gaussians."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


DEFAULT_LAGA_REPO = Path("/root/baselines/LaGa")
DEFAULT_MODEL_ROOT = Path("output/baselines/laga/lerf_compat_20260520")
DEFAULT_SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")
DEFAULT_SCENE_ITERATION = 30001
DEFAULT_AFFINITY_ITERATION = 30000
DEFAULT_NUM_LEVELS = 3
DEFAULT_FEATURE_DIM = 32
DEFAULT_NUM_PER_CLUSTER_FEATURES = 20


def multi_level_dimensions(*, feature_dim: int, num_levels: int) -> list[int]:
    if num_levels == 1:
        return [feature_dim]
    if feature_dim == 32 and num_levels == 3:
        return [16, 8, 8]
    base = feature_dim // num_levels
    dims = [base] * num_levels
    dims[0] += feature_dim - sum(dims)
    return dims


def contiguous_cluster_labels(labels: np.ndarray) -> tuple[np.ndarray, list[int]]:
    unique = sorted(int(label) for label in np.unique(labels) if int(label) != -1)
    mapping = {label: idx for idx, label in enumerate(unique)}
    remapped = np.full(labels.shape, -1, dtype=np.int64)
    for original, new in mapping.items():
        remapped[labels == original] = new
    return remapped, unique


def cluster_to_masks(labels: np.ndarray, *, device: torch.device) -> list[torch.Tensor]:
    remapped, unique = contiguous_cluster_labels(labels)
    return [torch.as_tensor(remapped == idx, device=device) for idx, _original in enumerate(unique)]


def filter_features_with_mask(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return features[mask.to(device=features.device)]


def _repo_import_context(repo: Path) -> None:
    repo_str = str(repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _combined_laga_args(repo: Path, model_path: Path, target_cfg_file: str = "cfg_args") -> Namespace:
    _repo_import_context(repo)
    from arguments import ModelParams, PipelineParams

    parser = ArgumentParser(description="LaGa descriptor argument loader")
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
    affinity_iteration: int,
) -> tuple[Any, Any, Any, Any, Any]:
    _repo_import_context(repo)
    from arguments import ModelParams, PipelineParams
    from scene import FeatureGaussianModel, Scene

    parser = ArgumentParser(description="LaGa descriptor parameters")
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
    return args, pipeline, scene, feature_gaussians, dataset


def _level_masks(sam_masks: torch.Tensor, *, num_levels: int, mask_size_thresh: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    multi_lvl_masks_before_filt: list[torch.Tensor] = []
    multi_lvl_masks: list[torch.Tensor] = []
    mask_filters: list[torch.Tensor] = []

    sam_masks = sam_masks.clone().float()
    sam_masks[sam_masks == -1] = -1000
    for lvl in range(1, num_levels + 1) if num_levels > 1 else [0]:
        tmp_masks = sam_masks[lvl].clone()
        smallest_index = sam_masks[lvl - 1].int().max() + 1 if lvl > 0 else 0
        tmp_masks -= smallest_index - 1
        tmp_masks[tmp_masks < 0] = 0
        one_hot = torch.nn.functional.one_hot(tmp_masks.long(), num_classes=tmp_masks.max().int().item() + 1)[:, :, 1:].float()
        multi_lvl_masks_before_filt.append(one_hot)

    for lvl in range(num_levels):
        tmp_mask = multi_lvl_masks_before_filt[lvl]
        mask_non_zero_count = tmp_mask.sum(dim=(0, 1))
        mask_filter = mask_non_zero_count > mask_size_thresh

        for sub_lvl in range(0, lvl):
            tmp_mask2 = multi_lvl_masks_before_filt[sub_lvl]
            intersection = torch.einsum("hwc,hwf->cf", tmp_mask, tmp_mask2)
            union = tmp_mask.sum(dim=(0, 1)).unsqueeze(-1) + tmp_mask2.sum(dim=(0, 1)) - intersection
            inter_over_union = intersection / torch.clamp(union, min=1e-6)
            refined = torch.logical_and(mask_filter, inter_over_union.max(dim=1)[0] < 0.8)
            mask_filter = mask_filter if refined.count_nonzero() == 0 else refined

        multi_lvl_masks.append(tmp_mask[:, :, mask_filter])
        mask_filters.append(mask_filter)
    return multi_lvl_masks, mask_filters


def _split_rendered_features(rendered_features: torch.Tensor, dims: list[int]) -> list[torch.Tensor]:
    rendered_features = rendered_features.permute(1, 2, 0)
    normed: list[torch.Tensor] = []
    offset = 0
    for dim in dims:
        normed.append(torch.nn.functional.normalize(rendered_features[:, :, offset : offset + dim], dim=-1))
        offset += dim
    concatenated = torch.cat(normed, dim=-1)

    features: list[torch.Tensor] = []
    offset = 0
    for dim in dims:
        features.insert(0, torch.nn.functional.normalize(concatenated[:, :, : offset + dim], dim=-1))
        offset += dim
    return features


def dynamic_k_selection(
    clus_feats: torch.Tensor,
    *,
    min_clusters: int = 1,
    max_clusters: int = DEFAULT_NUM_PER_CLUSTER_FEATURES,
    n_init: int = 10,
) -> torch.Tensor:
    from sklearn.cluster import KMeans
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import silhouette_score

    clus_feats_np = clus_feats.detach().cpu().numpy()
    best_score = -1.0
    best_centers = clus_feats.mean(dim=0, keepdim=True)

    for k in range(min_clusters, min(max_clusters, clus_feats.shape[0]) + 1):
        if k == 1:
            centers = clus_feats.mean(dim=0, keepdim=True)
            score = -1.0
        else:
            model = KMeans(n_clusters=k, n_init=n_init, random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                labels = model.fit_predict(clus_feats_np)
            if len(set(labels)) < k:
                score = -1.0
            else:
                try:
                    score = float(silhouette_score(clus_feats_np, labels))
                except Exception:
                    score = -1.0
            centers = torch.as_tensor(model.cluster_centers_, device=clus_feats.device, dtype=clus_feats.dtype)
        if score > best_score:
            best_score = score
            best_centers = centers
    return best_centers


def build_scene_descriptors(
    *,
    repo: Path,
    model_path: Path,
    affinity_iteration: int,
    num_levels: int,
    mask_size_thresh: int,
    num_per_cluster_features: int,
    max_views: int | None = None,
) -> dict[str, Any]:
    _repo_import_context(repo)
    import hdbscan
    from gaussian_renderer import render_contrastive_feature

    args, pipeline, scene, feature_gaussians, dataset = _load_laga_scene(
        repo=repo,
        model_path=model_path,
        affinity_iteration=affinity_iteration,
    )
    dims = multi_level_dimensions(feature_dim=dataset.feature_dim, num_levels=num_levels)
    background = torch.zeros([dataset.feature_dim], dtype=torch.float32, device="cuda")
    cameras = scene.getTrainCameras()
    if max_views is not None:
        cameras = cameras[:max_views]

    multi_lvl_prototypes: list[list[torch.Tensor]] = [[] for _ in range(num_levels)]
    multi_lvl_mask_filter: list[list[torch.Tensor]] = [[] for _ in range(num_levels)]
    with torch.no_grad():
        for view in cameras:
            torch.cuda.empty_cache()
            view.feature_height, view.feature_width = view.image_height, view.image_width
            rendered_feature = render_contrastive_feature(
                view,
                feature_gaussians,
                pipeline.extract(args),
                background,
                norm_point_features=True,
                multi_lvl_norm=True,
                multi_lvl_dim=dims,
                smooth_type=None,
            )["render"]
            rendered_feature = torch.nn.functional.interpolate(
                rendered_feature.unsqueeze(0),
                view.original_masks.shape[-2:],
                mode="bilinear",
            ).squeeze(0)
            masks, filters = _level_masks(
                view.original_masks.cuda(),
                num_levels=num_levels,
                mask_size_thresh=mask_size_thresh,
            )
            level_features = _split_rendered_features(rendered_feature, dims)
            for lvl in range(num_levels):
                if level_features[lvl].shape[:2] != masks[lvl].shape[:2]:
                    level_features[lvl] = torch.nn.functional.interpolate(
                        level_features[lvl].permute(2, 0, 1).unsqueeze(0),
                        masks[lvl].shape[:2],
                        mode="bilinear",
                    ).squeeze(0).permute(1, 2, 0)
                if masks[lvl].shape[-1] > 0:
                    prototypes = torch.nn.functional.normalize(
                        torch.einsum("hwc,hwf->fc", level_features[lvl], masks[lvl]),
                        dim=-1,
                        p=2,
                    )
                    multi_lvl_prototypes[lvl].append(prototypes)
                multi_lvl_mask_filter[lvl].append(filters[lvl])

        merged_prototypes = [torch.cat(items, 0) if items else torch.empty(0, dims[min(lvl, len(dims)-1)], device="cuda") for lvl, items in enumerate(multi_lvl_prototypes)]
        merged_filters = [torch.cat(items, 0) for items in multi_lvl_mask_filter]

        multi_lvl_cluster_labels: list[np.ndarray] = []
        multi_lvl_cluster_centers: list[torch.Tensor] = []
        for lvl, prototypes in enumerate(merged_prototypes):
            if prototypes.shape[0] == 0:
                labels = np.asarray([], dtype=np.int64)
                centers = torch.empty(0, prototypes.shape[-1], device="cuda")
            elif prototypes.shape[0] < 5:
                labels = np.arange(prototypes.shape[0], dtype=np.int64)
                centers = prototypes
            else:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=5, cluster_selection_epsilon=0.1 * (lvl + 1))
                raw_labels = clusterer.fit_predict(prototypes.detach().cpu().numpy())
                labels, _unique = contiguous_cluster_labels(raw_labels)
                centers = torch.zeros(max(labels.max() + 1, 0), prototypes.shape[-1], device="cuda")
                for cluster_id in range(centers.shape[0]):
                    centers[cluster_id] = torch.nn.functional.normalize(prototypes[labels == cluster_id].mean(dim=0), dim=-1)
            multi_lvl_cluster_labels.append(labels)
            multi_lvl_cluster_centers.append(centers)

        point_features = feature_gaussians.get_point_features
        normed_point_features: list[torch.Tensor] = []
        offset = 0
        for dim in dims:
            normed_point_features.append(torch.nn.functional.normalize(point_features[:, offset : offset + dim], dim=-1))
            offset += dim
        concatenated_point_features = torch.cat(normed_point_features, dim=-1)

        multi_lvl_point_features: list[torch.Tensor] = []
        offset = 0
        for dim in dims:
            multi_lvl_point_features.insert(0, torch.nn.functional.normalize(concatenated_point_features[:, : offset + dim], dim=-1))
            offset += dim

        multi_lvl_seg_scores = [
            torch.einsum("nc,bc->bn", centers.cuda(), selected.cuda())
            if centers.shape[0] > 0
            else torch.empty(selected.shape[0], 0, device="cuda")
            for centers, selected in zip(multi_lvl_cluster_centers, multi_lvl_point_features)
        ]

        multi_lvl_flatten_features: list[torch.Tensor] = []
        for lvl in range(num_levels):
            flatten_features = []
            for view in cameras:
                feature = view.original_features.view(-1, 512)
                lvl_to_mask_id = []
                last_one = -1
                for idx in range(0, 4) if num_levels == 3 else range(0, 1):
                    current_mask = view.original_masks[idx]
                    current_one = current_mask.max().int().item()
                    lvl_to_mask_id.append([last_one + 1, current_one])
                    last_one = current_one
                if num_levels > 1:
                    interval = lvl_to_mask_id[lvl + 1]
                    feature = feature[interval[0] : interval[1] + 1, :]
                flatten_features.append(feature)
            multi_lvl_flatten_features.append(torch.cat(flatten_features, dim=0))

        multi_lvl_cluster_features: list[torch.Tensor] = []
        multi_lvl_cluster_feature_weights: list[torch.Tensor] = []
        multi_lvl_cluster_feature_weights_only_direction: list[torch.Tensor] = []
        for lvl in range(num_levels):
            masks = cluster_to_masks(multi_lvl_cluster_labels[lvl], device=torch.device("cuda"))
            flatten_features = filter_features_with_mask(multi_lvl_flatten_features[lvl], merged_filters[lvl])
            cluster_features = []
            cluster_weights = []
            cluster_weights_only_direction = []
            for mask in masks:
                if mask.count_nonzero() == 0:
                    cluster_features.append(torch.rand(num_per_cluster_features, flatten_features.shape[-1], device="cuda"))
                    cluster_weights.append(torch.zeros(num_per_cluster_features, device="cuda"))
                    cluster_weights_only_direction.append(torch.zeros(num_per_cluster_features, device="cuda"))
                    continue
                clus_feats = torch.nn.functional.normalize(
                    filter_features_with_mask(flatten_features, mask).squeeze(1),
                    dim=-1,
                    p=2,
                ).cuda()
                avg_clus_feats = torch.nn.functional.normalize(clus_feats.mean(dim=0), dim=-1, p=2)
                centers = dynamic_k_selection(
                    clus_feats,
                    max_clusters=min(clus_feats.shape[0], num_per_cluster_features),
                    n_init=10,
                )
                num_centers = len(centers)
                tmp_features = torch.rand(num_per_cluster_features, flatten_features.shape[-1], device="cuda")
                tmp_features[:num_centers] = centers[:num_centers]
                tmp_weights = torch.zeros(num_per_cluster_features, device="cuda")
                tmp_weights_only_direction = torch.zeros(num_per_cluster_features, device="cuda")
                tmp_weights[:num_centers] = torch.einsum("c,nc->n", avg_clus_feats.cuda(), tmp_features[:num_centers])
                tmp_weights_only_direction[:num_centers] = torch.einsum(
                    "c,nc->n",
                    avg_clus_feats.cuda(),
                    torch.nn.functional.normalize(tmp_features[:num_centers], dim=-1, p=2),
                )
                cluster_features.append(tmp_features)
                cluster_weights.append(tmp_weights)
                cluster_weights_only_direction.append(tmp_weights_only_direction)

            feature_dim = multi_lvl_flatten_features[lvl].shape[-1]
            multi_lvl_cluster_features.append(
                torch.cat(cluster_features, 0) if cluster_features else torch.empty(0, feature_dim, device="cuda")
            )
            multi_lvl_cluster_feature_weights.append(
                torch.cat(cluster_weights, 0) if cluster_weights else torch.empty(0, device="cuda")
            )
            multi_lvl_cluster_feature_weights_only_direction.append(
                torch.cat(cluster_weights_only_direction, 0) if cluster_weights_only_direction else torch.empty(0, device="cuda")
            )

    out_dir = model_path / "point_cloud" / f"iteration_{affinity_iteration}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(multi_lvl_cluster_features, out_dir / "multi_lvl_cluster_features.pth")
    torch.save(multi_lvl_cluster_feature_weights, out_dir / "multi_lvl_cluster_feature_weights.pth")
    torch.save(multi_lvl_seg_scores, out_dir / "multi_lvl_seg_scores.pth")
    torch.save(
        multi_lvl_cluster_feature_weights_only_direction,
        out_dir / "multi_lvl_cluster_feature_weights_only_direction.pth",
    )
    return {
        "model_path": str(model_path),
        "affinity_iteration": affinity_iteration,
        "num_levels": num_levels,
        "views": len(cameras),
        "clusters_per_level": [int(scores.shape[1]) for scores in multi_lvl_seg_scores],
        "output_dir": str(out_dir),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_LAGA_REPO)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES), choices=DEFAULT_SCENES)
    parser.add_argument("--affinity-iteration", type=int, default=DEFAULT_AFFINITY_ITERATION)
    parser.add_argument("--num-levels", type=int, default=DEFAULT_NUM_LEVELS)
    parser.add_argument("--mask-size-thresh", type=int, default=400)
    parser.add_argument("--num-per-cluster-features", type=int, default=DEFAULT_NUM_PER_CLUSTER_FEATURES)
    parser.add_argument("--max-views", type=int, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def _descriptor_files(out_dir: Path) -> list[Path]:
    return [
        out_dir / "multi_lvl_cluster_features.pth",
        out_dir / "multi_lvl_cluster_feature_weights.pth",
        out_dir / "multi_lvl_seg_scores.pth",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    summaries: list[dict[str, Any]] = []
    for scene in args.scenes:
        model_path = args.model_root / scene
        out_dir = model_path / "point_cloud" / f"iteration_{args.affinity_iteration}"
        if args.skip_existing and all(path.is_file() for path in _descriptor_files(out_dir)):
            summaries.append({"scene": scene, "model_path": str(model_path), "skipped": True})
            continue
        summary = build_scene_descriptors(
            repo=args.repo,
            model_path=model_path,
            affinity_iteration=args.affinity_iteration,
            num_levels=args.num_levels,
            mask_size_thresh=args.mask_size_thresh,
            num_per_cluster_features=args.num_per_cluster_features,
            max_views=args.max_views,
        )
        summary["scene"] = scene
        summary["skipped"] = False
        summaries.append(summary)
    report = {"method": "LaGa", "scenes": summaries}
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
