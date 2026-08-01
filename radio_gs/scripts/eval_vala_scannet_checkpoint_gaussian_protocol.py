#!/usr/bin/env python3
"""Evaluate released-code VALA checkpoints in the ScanNet Gaussian domain.

This evaluator is deliberately separate from the historical mesh-kNN
compatibility readout in :mod:`run_vala_scannet_baseline`.  It performs the
paper-facing task directly on the optimized Gaussians:

* classify every Gaussian by cosine similarity to the ScanNet class text;
* attach pseudo ground truth to Gaussian means with VALA's anisotropic
  Mahalanobis-density vote; and
* compute per-scene, opacity-times-volume weighted mIoU and mAcc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.scannet_constants import (
    NYU40_ID_TO_NAME,
    OPENGAUSSIAN_NYU40_CLASS_SPLITS,
)
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _read_label_ply
from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import (
    assign_vala_pseudo_labels,
    volume_weighted_split_metrics,
)


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
VALA_CODE9_SCENES = VALA_PAPER8_SCENES + (
    "scene0645_00",
)
# Compatibility alias for downstream imports. It is not the default cohort.
VALA_CURRENT_SCENES = VALA_CODE9_SCENES

VALA_TEXT_KEYS = {
    "shower curtain": "showercurtain",
    "floor mat": "floormat",
    "night stand": "nightstand",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_scenes(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_cohort_scenes(cohort: str, scenes_value: str | None) -> tuple[list[str], str]:
    canonical = {
        "paper8": list(VALA_PAPER8_SCENES),
        "code9": list(VALA_CODE9_SCENES),
    }
    if cohort == "custom":
        if scenes_value is None:
            raise ValueError("--cohort custom requires an explicit --scenes list")
        scenes = _parse_scenes(scenes_value)
        if not scenes:
            raise ValueError("--scenes is empty")
        return scenes, "custom_explicit_diagnostic"
    scenes = canonical[cohort]
    if scenes_value is not None and _parse_scenes(scenes_value) != scenes:
        raise ValueError(
            f"--cohort {cohort} requires its exact canonical scene list; "
            "use --cohort custom for a diagnostic subset or alternate cohort"
        )
    status = (
        "paper8_canonical_paper_facing"
        if cohort == "paper8"
        else "code9_post_paper_sensitivity_only"
    )
    return scenes, status


def _format_path(pattern: str, scene: str) -> Path:
    return Path(pattern.format(scene=scene)).expanduser().resolve()


def _load_text_features(path: Path) -> dict[str, torch.Tensor]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(name): F.normalize(torch.as_tensor(value, dtype=torch.float32), dim=0)
        for name, value in payload.items()
    }


def _checkpoint_arrays(path: Path) -> dict[str, np.ndarray]:
    payload = torch.load(str(path), map_location="cpu")
    model_args = payload[0] if isinstance(payload, tuple) and len(payload) == 2 else payload
    if not isinstance(model_args, (tuple, list)) or len(model_args) != 13:
        raise ValueError(f"Unexpected VALA checkpoint structure in {path}")
    xyz = torch.as_tensor(model_args[1]).detach().float()
    raw_scale = torch.as_tensor(model_args[4]).detach().float()
    raw_rotation = torch.as_tensor(model_args[5]).detach().float()
    raw_opacity = torch.as_tensor(model_args[6]).detach().float()
    language = torch.as_tensor(model_args[7]).detach().float()
    count = int(xyz.shape[0])
    for name, value in {
        "raw_scale": raw_scale,
        "raw_rotation": raw_rotation,
        "raw_opacity": raw_opacity,
        "language": language,
    }.items():
        if int(value.shape[0]) != count:
            raise ValueError(f"{name} is not aligned with xyz in {path}")
    return {
        "xyz": xyz.numpy(),
        "scale": raw_scale.exp().numpy(),
        "rotation": F.normalize(raw_rotation, dim=-1).numpy(),
        "opacity": raw_opacity.sigmoid().reshape(-1).numpy(),
        "language": F.normalize(language, dim=-1).numpy(),
    }


def _predict(
    language: np.ndarray,
    text_features: dict[str, torch.Tensor],
    split: str,
    *,
    chunk_size: int,
) -> np.ndarray:
    class_ids = np.asarray(OPENGAUSSIAN_NYU40_CLASS_SPLITS[split], dtype=np.int32)
    names = [
        VALA_TEXT_KEYS.get(NYU40_ID_TO_NAME[int(class_id)], NYU40_ID_TO_NAME[int(class_id)])
        for class_id in class_ids
    ]
    missing = [name for name in names if name not in text_features]
    if missing:
        raise KeyError(f"Missing text features for split {split}: {missing}")
    text = torch.stack([text_features[name] for name in names], dim=0)
    if int(language.shape[1]) != int(text.shape[1]):
        raise ValueError(
            f"Language/text dimension mismatch: {language.shape[1]} vs {text.shape[1]}"
        )
    pred = np.empty(language.shape[0], dtype=np.int32)
    for start in range(0, language.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), language.shape[0])
        visual = torch.from_numpy(language[start:end]).float()
        pred[start:end] = class_ids[(visual @ text.T).argmax(dim=-1).numpy()]
    return pred


def _scene_macro(scene_results: dict[str, dict[str, object]]) -> dict[str, dict[str, float]]:
    return {
        split: {
            metric: float(
                np.mean(
                    [
                        float(scene_result["splits"][split][metric])
                        for scene_result in scene_results.values()
                    ]
                )
            )
            for metric in ("miou", "macc")
        }
        for split in ("19", "15", "10")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("paper8", "code9", "custom"), default="paper8")
    parser.add_argument(
        "--scenes",
        default=None,
        help="Explicit comma-separated list. Required for custom; canonical cohorts reject mismatches.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path pattern containing {scene}")
    parser.add_argument("--label-ply", required=True, help="Path pattern containing {scene}")
    parser.add_argument("--text-features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-chunk-size", type=int, default=16384)
    parser.add_argument("--pseudo-chunk-size", type=int, default=512)
    parser.add_argument("--radius-factor", type=float, default=5.0)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--fallback-k", type=int, default=1)
    parser.add_argument("--no-class-balance", action="store_true")
    parser.add_argument("--force-pseudo-gt", action="store_true")
    parser.add_argument("--upstream-commit", default="48902a541333d65aeb0aebf64ad664777a27c3fc")
    args = parser.parse_args()

    scenes, cohort_status = _resolve_cohort_scenes(args.cohort, args.scenes)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = Path(args.text_features).expanduser().resolve()
    text_features = _load_text_features(text_path)
    scene_results: dict[str, dict[str, object]] = {}
    for scene in scenes:
        print(f"\n=== {scene} ===", flush=True)
        checkpoint = _format_path(args.checkpoint, scene)
        label_ply = _format_path(args.label_ply, scene)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if not label_ply.is_file():
            raise FileNotFoundError(label_ply)
        arrays = _checkpoint_arrays(checkpoint)
        point_xyz, point_labels = _read_label_ply(label_ply)
        pseudo_path = output_dir / "pseudo_gt" / f"{scene}.npz"
        pseudo_path.parent.mkdir(parents=True, exist_ok=True)
        pseudo_settings = {
            "checkpoint_sha256": _sha256(checkpoint),
            "label_ply_sha256": _sha256(label_ply),
            "radius_factor": float(args.radius_factor),
            "candidate_k": int(args.candidate_k),
            "fallback_k": int(args.fallback_k),
            "class_balance": not args.no_class_balance,
        }
        pseudo_labels: np.ndarray
        pseudo_stats: dict[str, object]
        if pseudo_path.is_file() and not args.force_pseudo_gt:
            cached = np.load(pseudo_path, allow_pickle=False)
            cached_settings = json.loads(str(cached["settings_json"].item()))
            if cached_settings == pseudo_settings:
                pseudo_labels = np.asarray(cached["pseudo_labels"], dtype=np.int32)
                pseudo_stats = json.loads(str(cached["stats_json"].item()))
            else:
                pseudo_labels, pseudo_stats = assign_vala_pseudo_labels(
                    arrays["xyz"], arrays["scale"], arrays["rotation"], point_xyz, point_labels,
                    radius_factor=args.radius_factor, candidate_k=args.candidate_k,
                    fallback_k=args.fallback_k, class_balance=not args.no_class_balance,
                    chunk_size=args.pseudo_chunk_size,
                )
        else:
            pseudo_labels, pseudo_stats = assign_vala_pseudo_labels(
                arrays["xyz"], arrays["scale"], arrays["rotation"], point_xyz, point_labels,
                radius_factor=args.radius_factor, candidate_k=args.candidate_k,
                fallback_k=args.fallback_k, class_balance=not args.no_class_balance,
                chunk_size=args.pseudo_chunk_size,
            )
        np.savez_compressed(
            pseudo_path,
            pseudo_labels=pseudo_labels,
            settings_json=np.asarray(json.dumps(pseudo_settings, sort_keys=True)),
            stats_json=np.asarray(json.dumps(pseudo_stats, sort_keys=True)),
        )
        significance = arrays["opacity"] * arrays["scale"].prod(axis=1)
        split_results = {}
        prediction_payload: dict[str, np.ndarray] = {
            "xyz": arrays["xyz"],
            "pseudo_labels": pseudo_labels,
            "significance": significance,
        }
        for split in ("19", "15", "10"):
            pred = _predict(
                arrays["language"], text_features, split,
                chunk_size=args.feature_chunk_size,
            )
            metrics = volume_weighted_split_metrics(
                pseudo_labels,
                pred,
                significance,
                OPENGAUSSIAN_NYU40_CLASS_SPLITS[split],
            )
            split_results[split] = metrics
            prediction_payload[f"pred_split_{split}"] = pred
            print(
                f"split{split}: {100.0 * metrics['miou']:.2f}/"
                f"{100.0 * metrics['macc']:.2f}",
                flush=True,
            )
        prediction_path = output_dir / "predictions" / f"{scene}.npz"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prediction_path, **prediction_payload)
        scene_results[scene] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": pseudo_settings["checkpoint_sha256"],
            "label_ply": str(label_ply),
            "label_ply_sha256": pseudo_settings["label_ply_sha256"],
            "num_gaussians": int(arrays["xyz"].shape[0]),
            "pseudo_gt": pseudo_stats,
            "splits": split_results,
            "prediction_npz": str(prediction_path),
        }

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "VALA released-code checkpoint reproduction",
        "protocol": {
            "upstream_commit": args.upstream_commit,
            "task": "text-query open-vocabulary 3D semantic segmentation",
            "prediction_domain": "optimized Gaussian means",
            "pseudo_gt": "VALA anisotropic Mahalanobis-density vote",
            "metric_weights": "sigmoid(opacity_logit) * exp(scale_log).prod()",
            "class_splits": ["19", "15", "10"],
            "class_aggregation": "present classes within each scene",
            "scene_aggregation": "unweighted scene macro",
            "cohort": args.cohort,
            "cohort_scenes": scenes,
            "cohort_status": cohort_status,
            "text_features": str(text_path),
            "text_features_sha256": _sha256(text_path),
        },
        "args": vars(args),
        "macro": _scene_macro(scene_results),
        "scenes": scene_results,
    }
    report["args"] = {key: str(value) for key, value in report["args"].items()}
    report_path = output_dir / "vala_scannet_gaussian_protocol_results.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nMacro:")
    print(json.dumps(report["macro"], indent=2))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
