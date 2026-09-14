"""Score exported masks using OpenGaussian's released LERF metric semantics.

Reference: https://github.com/yanmin-wu/OpenGaussian/blob/main/scripts/compute_lerf_iou.py
This adapter verifies mask dimensions and reports provenance; it does not certify
the upstream 3D selection, rendering, source split, or ground-truth generation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from radio_gs.utils.immutable_artifacts import sha256_file

FRAMES = {
    "figurines": (41, 105, 152, 195),
    "ramen": (6, 24, 60, 65, 81, 119, 128),
    "teatime": (2, 25, 43, 107, 129, 140),
    "waldo_kitchen": (53, 66, 89, 140, 154),
}


def score_masks(gt_root: Path, prediction_root: Path, scene: str, prediction_layout: str = "flat"):
    if prediction_layout not in ("flat", "nested"):
        raise ValueError("unsupported prediction directory layout")
    observations = []
    for frame_id in FRAMES[scene]:
        frame = f"frame_{frame_id:05d}"
        folder = gt_root / frame
        if not folder.is_dir():
            raise FileNotFoundError(f"required public GT frame directory missing: {folder}")
        targets = sorted(folder.glob("*.jpg"))
        if not targets:
            raise ValueError(f"public GT frame contains no object masks: {folder}")
        for target in targets:
            prediction = prediction_root / f"{frame}_{target.stem}.png"
            if prediction_layout == "nested":
                prediction = prediction_root / frame / (target.stem + ".png")
            record = {"frame_id": frame_id, "object": target.stem,
                      "gt_sha256": sha256_file(target), "prediction_missing": not prediction.exists()}
            if prediction.exists():
                with Image.open(target) as image:
                    gt = np.asarray(image) > 10
                with Image.open(prediction) as image:
                    pred = np.asarray(image.convert("L")) > 10
                # Official GT is a 2D mask. Do not silently broadcast RGB GT.
                if gt.ndim != 2 or gt.shape != pred.shape:
                    raise ValueError(f"public mask shape mismatch: {target}: {gt.shape}, {pred.shape}")
                union = int(np.logical_or(gt, pred).sum())
                record["iou"] = float(np.logical_and(gt, pred).sum() / union) if union else 0.0
                record["prediction_sha256"] = sha256_file(prediction)
            else:
                record["iou"] = 0.0
            observations.append(record)
    values = np.asarray([r["iou"] for r in observations])
    return {"scene": scene, "observation_count": len(observations),
            "mean_iou": float(values.mean()), "acc_at_025": float((values > .25).mean()),
            "acc_at_05": float((values > .5).mean()), "observations": observations,
            "metric_contract": "OpenGaussian compute_lerf_iou: >10 binary masks, missing=0, observation mean, strict accuracy cutoffs",
            "prediction_layout": prediction_layout, "upstream_protocol_validated": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--scene", choices=tuple(FRAMES), required=True)
    parser.add_argument("--prediction-layout", choices=("flat", "nested"), default="flat")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = score_masks(args.gt_root, args.prediction_root, args.scene, args.prediction_layout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
