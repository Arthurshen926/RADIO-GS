"""Label-only diagnostic: exhaustive native-raster extent ceilings, never deployment."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from radio_gs.data.lerf_dataset import _read_cameras_binary
from radio_gs.scripts.evaluate_opengaussian_lerf_masks import FRAMES
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.evaluation.lerf_common import camera
from radio_gs.v4.evaluation.lerf_object_ceiling import _load_state, _build_carrier
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_images


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-report", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = json.loads(args.text_report.read_text())
    state = _load_state(Path(ref["scene_state"]), expected_sha256=ref["scene_state_sha256"])
    memory_path = Path(ref["hypothesis_memory"])
    if sha256_file(memory_path) != ref["hypothesis_memory_sha256"]:
        raise ValueError("hypothesis hash differs")
    memory = torch.load(memory_path, weights_only=False, map_location="cpu")
    fragment_path = Path(memory["fragment_memory"])
    if sha256_file(fragment_path) != memory["fragment_memory_sha256"]:
        raise ValueError("fragment hash differs")
    fragment = torch.load(fragment_path, weights_only=False, map_location="cpu")
    if fragment["scene_state_sha256"] != ref["scene_state_sha256"]:
        raise ValueError("fragment scene differs")
    state["method_configuration"]["carrier"]["reference_raster_shape"] = fragment["raster_shape"]
    carrier = _build_carrier(state)
    sparse = args.scene_root / "sparse/0"
    views = _read_images(sparse / "images.bin", _read_cameras_binary(sparse / "cameras.bin"))
    matrices = {"hypothesis": memory["observed_membership"].float().cuda(),
                "source_fragment": fragment["fragment_positive"].T.float().cuda()}
    scores = {key: {} for key in matrices}
    max_error = 0.
    reference_mask_disagreements = 0
    for frame in FRAMES[ref["scene_label"]]:
        paths = sorted((args.gt_root / f"frame_{frame:05d}").glob("*.jpg"))
        if not paths:
            raise ValueError("missing GT inventory")
        targets = []
        for path in paths:
            with Image.open(path) as image:
                targets.append(torch.from_numpy((np.asarray(image) > 10).copy()))
        height, width = targets[0].shape
        gt = torch.stack(targets).reshape(len(paths), -1).float().cuda()
        cam = camera(views[frame], frame, height, width)
        projection = carrier.project(cam)
        pixels = projection.pixel_ids.cuda()
        elements = projection.element_ids.cuda()
        weights = projection.weights.cuda()
        denominator = torch.zeros(height * width, device="cuda").scatter_add_(0, pixels, weights)
        operator = torch.sparse_coo_tensor(torch.stack((pixels, elements)),
            weights / denominator[pixels].clamp_min(1e-12),
            (height * width, carrier.num_elements)).coalesce().to_sparse_csr()
        cpu_check = carrier.render_posterior(memory["observed_membership"][:, 0], cam).flatten().cuda()
        gpu_check = torch.sparse.mm(operator, matrices["hypothesis"][:, :1].contiguous())[:, 0]
        error = float((cpu_check - gpu_check).abs().max())
        max_error = max(max_error, error)
        disagreement = int(((cpu_check >= ref["pixel_threshold"]) != (gpu_check >= ref["pixel_threshold"])).sum())
        reference_mask_disagreements += disagreement
        if error > 2e-6:
            raise ValueError(f"GPU diagnostic disagrees with CPU reference: max error {error}, mask disagreement {disagreement}")
        for kind, matrix in matrices.items():
            ious = []
            for start in range(0, matrix.shape[1], 16):
                pred = (torch.sparse.mm(operator, matrix[:, start:start + 16].contiguous()) >= ref["pixel_threshold"]).float()
                intersection = gt @ pred
                union = gt.sum(-1)[:, None] + pred.sum(0)[None] - intersection
                ious.append((intersection / union.clamp_min(1)).cpu())
            iou = torch.cat(ious, -1)
            for row, path in enumerate(paths):
                scores[kind].setdefault(path.stem, []).append(iou[row])
        carrier._projection_cache.clear()
        del operator, gt, pred
        print(ref["scene_label"], frame, "native exhaustive ceiling done", flush=True)
    report = {"oracle_identity_used": True, "deployment_result": False,
              "hypothesis_memory_sha256": ref["hypothesis_memory_sha256"],
              "fragment_memory_sha256": memory["fragment_memory_sha256"],
              "gt_selected_ids_never_written_to_scene_state": True,
              "pixel_threshold": ref["pixel_threshold"], "gpu_cpu_max_error": max_error,
              "reference_channel_threshold_disagreements": reference_mask_disagreements,
              "floating_point_tolerance_diagnostic_not_exact_cpu_benchmark": True,
              "scene": ref["scene_label"], "variants": {}}
    for kind, categories in scores.items():
        observations, stable, per_query = [], [], {}
        for category, values in categories.items():
            iou = torch.stack(values)
            chosen = int(iou.mean(0).argmax())
            observations.extend(iou.max(-1).values.tolist())
            stable.extend(iou[:, chosen].tolist())
            per_query[category] = {"stable_id": chosen, "stable_iou": float(iou[:, chosen].mean()),
                                   "per_view_oracle_iou": float(iou.max(-1).values.mean())}
        report["variants"][kind] = {"observation_count": len(observations),
            "per_view_oracle_miou": float(np.mean(observations)),
            "stable_oracle_miou": float(np.mean(stable)), "per_query": per_query}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()
