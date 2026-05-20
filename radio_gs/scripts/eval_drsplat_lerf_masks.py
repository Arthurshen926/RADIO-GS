#!/usr/bin/env python3
"""Evaluate Dr. Splat/VALA-style LERF nested mask renders.

Expected prediction layout:
`predictions_mask_<mask_thresh>/renders_silhouette/<frame>/<query>.png`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from radio_gs.scripts.eval_opengaussian_lerf_baseline import (
    DEFAULT_SCENES,
    SCENE_GT_FRAMES,
    ObjectResult,
    SceneResult,
    _load_frame_gt_masks,
    _mean,
    _rate_above,
    calculate_iou,
    load_binary_mask,
)


def _mask_thresh_name(mask_thresh: str | float) -> str:
    if isinstance(mask_thresh, float):
        return f"{mask_thresh:g}"
    return str(mask_thresh)


def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    output: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        output.append(path)
    return output


def _resize_mask_to_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    resampling_nearest = getattr(Image, "Resampling", Image).NEAREST
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    image = image.resize((shape[1], shape[0]), resampling_nearest)
    return np.asarray(image) > 0


def resolve_scene_pred_root(
    pred_root: Path,
    scene: str,
    *,
    mask_thresh: str | float = "0.4",
    ablation_type: str = "none",
    prediction_dir: str = "renders_silhouette",
    direct_pred_root: bool = False,
) -> Path:
    """Resolve the nested mask root for one scene.

    Dr. Splat training appends run metadata to the scene directory in some
    configurations, so exact `<root>/<scene>/...` and `<root>/<scene>*` layouts
    are both accepted. If no candidate exists yet, return the exact-scene
    candidate so the evaluator records missing masks deterministically.
    """

    if direct_pred_root:
        return pred_root

    mask_dir = f"predictions_mask_{_mask_thresh_name(mask_thresh)}"
    candidates = [
        pred_root / scene / ablation_type / mask_dir / prediction_dir,
        pred_root / scene / mask_dir / prediction_dir,
    ]
    for scene_dir in sorted(path for path in pred_root.glob(f"{scene}*") if path.is_dir()):
        candidates.extend(
            [
                scene_dir / ablation_type / mask_dir / prediction_dir,
                scene_dir / mask_dir / prediction_dir,
            ]
        )
    candidates = _unique_paths(candidates)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def evaluate_scene(gt_root: Path, pred_root: Path, scene: str, *, threshold: int = 10) -> SceneResult:
    if scene not in SCENE_GT_FRAMES:
        choices = ", ".join(sorted(SCENE_GT_FRAMES))
        raise ValueError(f"Unknown scene {scene!r}; expected one of: {choices}")

    objects: list[ObjectResult] = []
    ious: list[float] = []
    missing = 0
    for frame in SCENE_GT_FRAMES[scene]:
        for gt in _load_frame_gt_masks(gt_root, frame, threshold=threshold):
            pred_path = pred_root / frame / f"{gt.query}.png"
            if not pred_path.exists():
                missing += 1
                ious.append(0.0)
                objects.append(
                    ObjectResult(
                        frame=frame,
                        query=gt.query,
                        gt_path=gt.path,
                        pred_path=str(pred_path),
                        iou=0.0,
                        missing=True,
                    )
                )
                continue
            mask_pred = load_binary_mask(pred_path, threshold=threshold, grayscale=True)
            mask_pred = _resize_mask_to_shape(mask_pred, gt.mask.shape)
            iou = calculate_iou(gt.mask, mask_pred)
            ious.append(iou)
            objects.append(
                ObjectResult(
                    frame=frame,
                    query=gt.query,
                    gt_path=gt.path,
                    pred_path=str(pred_path),
                    iou=iou,
                )
            )

    return SceneResult(
        scene=scene,
        miou=_mean(ious),
        acc025=_rate_above(ious, 0.25),
        acc05=_rate_above(ious, 0.5),
        count=len(ious),
        missing=missing,
        objects=objects,
    )


def evaluate_run(
    lerf_root: Path,
    pred_root: Path,
    scenes: Sequence[str],
    *,
    mask_thresh: str | float = "0.4",
    ablation_type: str = "none",
    prediction_dir: str = "renders_silhouette",
    direct_pred_root: bool = False,
    threshold: int = 10,
) -> dict[str, object]:
    scene_results: dict[str, dict[str, object]] = {}
    macro_source: list[SceneResult] = []
    for scene in scenes:
        scene_pred_root = resolve_scene_pred_root(
            pred_root,
            scene,
            mask_thresh=mask_thresh,
            ablation_type=ablation_type,
            prediction_dir=prediction_dir,
            direct_pred_root=direct_pred_root,
        )
        result = evaluate_scene(
            lerf_root / "label" / scene / "gt",
            scene_pred_root,
            scene,
            threshold=threshold,
        )
        result_dict = asdict(result)
        result_dict["pred_root"] = str(scene_pred_root)
        scene_results[scene] = result_dict
        macro_source.append(result)

    return {
        "protocol": "Dr. Splat/VALA LERF nested mask IoU",
        "mask_thresh": _mask_thresh_name(mask_thresh),
        "threshold": threshold,
        "ablation_type": ablation_type,
        "prediction_dir": prediction_dir,
        "scenes": scene_results,
        "macro": {
            "miou": _mean([result.miou for result in macro_source]),
            "acc025": _mean([result.acc025 for result in macro_source]),
            "acc05": _mean([result.acc05 for result in macro_source]),
            "count": int(sum(result.count for result in macro_source)),
            "missing": int(sum(result.missing for result in macro_source)),
        },
    }


def render_markdown(report: dict[str, object]) -> str:
    scenes = report["scenes"]
    macro = report["macro"]
    assert isinstance(scenes, dict)
    assert isinstance(macro, dict)

    lines = [
        "# Dr. Splat LERF Mask Summary",
        "",
        f"Protocol: {report['protocol']}",
        "",
        "| Split | mIoU | Acc@0.25 | Acc@0.5 | Objects | Missing |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scene, row in scenes.items():
        assert isinstance(row, dict)
        lines.append(
            "| {scene} | {miou:.4f} | {acc025:.4f} | {acc05:.4f} | {count} | {missing} |".format(
                scene=scene,
                miou=float(row["miou"]),
                acc025=float(row["acc025"]),
                acc05=float(row["acc05"]),
                count=int(row["count"]),
                missing=int(row["missing"]),
            )
        )
    lines.append(
        "| Macro | {miou:.4f} | {acc025:.4f} | {acc05:.4f} | {count} | {missing} |".format(
            miou=float(macro["miou"]),
            acc025=float(macro["acc025"]),
            acc05=float(macro["acc05"]),
            count=int(macro["count"]),
            missing=int(macro["missing"]),
        )
    )
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerf-root", type=Path, required=True, help="LERF-OVS root containing label/<scene> annotations.")
    parser.add_argument("--pred-root", type=Path, required=True, help="Dr. Splat run root or direct nested prediction root.")
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES), choices=DEFAULT_SCENES)
    parser.add_argument("--mask-thresh", default="0.4", help="Prediction folder suffix, e.g. 0.4 for predictions_mask_0.4.")
    parser.add_argument("--ablation-type", default="none")
    parser.add_argument("--prediction-dir", default="renders_silhouette")
    parser.add_argument("--direct-pred-root", action="store_true", help="Treat --pred-root as predictions_mask_*/renders_silhouette.")
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.direct_pred_root and len(args.scenes) != 1:
        raise SystemExit("--direct-pred-root requires exactly one scene")
    report = evaluate_run(
        args.lerf_root,
        args.pred_root,
        args.scenes,
        mask_thresh=args.mask_thresh,
        ablation_type=args.ablation_type,
        prediction_dir=args.prediction_dir,
        direct_pred_root=args.direct_pred_root,
        threshold=args.threshold,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(report), encoding="utf-8")

    macro = report["macro"]
    print(
        "Macro "
        f"mIoU={macro['miou']:.4f} "
        f"Acc@0.25={macro['acc025']:.4f} "
        f"Acc@0.5={macro['acc05']:.4f} "
        f"objects={macro['count']} missing={macro['missing']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
