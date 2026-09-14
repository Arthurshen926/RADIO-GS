"""Geometry-only ceiling: no query can recover pixels outside carrier support."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from radio_gs.data.lerf_dataset import _read_cameras_binary
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_images
from radio_gs.v4.evaluation.lerf_common import camera
from radio_gs.v4.evaluation.lerf_object_ceiling import _load_state, _build_carrier
from radio_gs.scripts.evaluate_opengaussian_lerf_masks import FRAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-report", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = json.loads(args.text_report.read_text())
    scene = report["scene_label"]
    state = _load_state(Path(report["scene_state"]), expected_sha256=report["scene_state_sha256"])
    carrier = _build_carrier(state)
    sparse = args.scene_root / "sparse/0"
    views = _read_images(sparse / "images.bin", _read_cameras_binary(sparse / "cameras.bin"))
    frames, observations = [], []
    for frame in FRAMES[scene]:
        paths = sorted((args.gt_root / f"frame_{frame:05d}").glob("*.jpg"))
        if not paths:
            raise FileNotFoundError("missing public GT frame")
        with Image.open(paths[0]) as image:
            width, height = image.size
        support = carrier.render_posterior(torch.ones(carrier.num_elements), camera(views[frame], frame, height, width)).numpy() > 0
        frames.append({"frame_id": frame, "shape": [height, width], "surface_pixel_coverage": float(support.mean())})
        for path in paths:
            with Image.open(path) as image:
                target = np.asarray(image) > 10
            if target.shape != support.shape:
                raise ValueError("inconsistent public GT shapes")
            observations.append({"frame_id": frame, "object": path.stem, "target_pixels": int(target.sum()),
                                 "support_iou_upper_bound": float((support & target).sum() / target.sum()) if target.any() else 0.0})
        print(scene, frame, frames[-1]["surface_pixel_coverage"], flush=True)
    result = {"scene": scene, "scene_state_sha256": report["scene_state_sha256"], "carrier_configuration": state["method_configuration"]["carrier"],
              "surface_element_count": carrier.num_elements, "frames": frames, "observations": observations,
              "mean_support_iou_upper_bound": float(np.mean([x["support_iou_upper_bound"] for x in observations])),
              "diagnostic_uses_gt": True, "query_selection_performed": False,
              "scope": "unchanged carrier rendered directly at public GT resolution; upper bound, not query result"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)


if __name__ == "__main__":
    main()
