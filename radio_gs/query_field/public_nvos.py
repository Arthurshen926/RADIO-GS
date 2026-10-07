"""Separate source-free signed prompt prediction from target-mask scoring."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .adapters import scribble_packet, raster_extent_scale
from .geometry import projection, selected_alpha
from .storage import load_field, sha256


def load_strokes(positive_path, negative_path, output_hw):
    positive = np.asarray(Image.open(positive_path))
    negative = np.asarray(Image.open(negative_path))
    if positive.ndim != 2 or negative.shape != positive.shape:
        raise ValueError("scribbles must be aligned scalar masks")
    positive, negative = positive != 0, negative != 0
    if not positive.any() or not negative.any() or (positive & negative).any():
        raise ValueError("scribbles must be nonempty and disjoint at native resolution")
    masks = torch.from_numpy(np.stack([positive, negative])).float()[:, None]
    resized = F.interpolate(masks, size=output_hw, mode="area")[:, 0]
    return resized, dict(native_hw=list(positive.shape), positive_pixels=int(positive.sum()),
                         negative_pixels=int(negative.sum()), resampling="fractional area evidence")


def deny_original_inputs(roots):
    """A Python file-open audit, not an OS sandbox; hooks live until process exit."""
    roots = [os.path.realpath(root) for root in roots]
    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = os.path.realpath(os.fsdecode(args[0]))
        if any(path == root or path.startswith(root + os.sep) for root in roots):
            raise PermissionError("source input denied in deployed prompt replay: " + path)
    sys.addaudithook(audit)


def retained_response(field, membership, camera, height, width, world_scale):
    """Same full-scene observed-support normalization as the retained v10 scorer."""
    from gsplat import rasterization
    intrinsic = camera.intrinsic.to(field.xyz).clone()
    intrinsic[0] *= width/camera.width
    intrinsic[1] *= height/camera.height
    image, alpha, _ = rasterization(means=field.xyz, quats=field.quaternions,
        scales=field.scales, opacities=field.opacities.flatten(),
        colors=torch.stack([membership, field.observed.float()], 1),
        viewmats=camera.world_to_camera.to(field.xyz)[None], Ks=intrinsic[None],
        width=width, height=height, near_plane=.001*world_scale, far_plane=100*world_scale, packed=False)
    mass = image[0, ..., 1]
    valid = (mass/alpha[0, ..., 0].clamp_min(1e-8) > .5) & (alpha[0, ..., 0] > 1e-8)
    return torch.where(valid, image[0, ..., 0]/mass.clamp_min(1e-8), 0)


@torch.no_grad()
def predict(config, job, output, device="cuda"):
    if job["mode"] not in {"positive", "signed"}:
        raise ValueError("unknown prompt sign policy")
    out = Path(output) / job["name"]
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    field, manifest = load_field(job["field"], device)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    loaded = time.monotonic()
    fixture = torch.load(job["camera_fixture"], map_location="cpu", weights_only=True)
    world_scale = fixture["world_units_per_metre"]
    if world_scale != field.world_units_per_metre:
        raise ValueError("registered camera/field world units differ")
    height, width = config["projection_hw"]
    masks, stroke_receipt = load_strokes(job["positive"], job["negative"], (height, width))
    reference = SimpleNamespace(**fixture["reference"])
    if stroke_receipt["native_hw"] != [reference.height, reference.width]:
        raise ValueError("scribble raster and registered reference camera differ")
    geometry = SimpleNamespace(xyz=field.xyz, get_xyz=lambda: field.xyz,
        get_rotation=lambda: field.quaternions, get_scaling=lambda: field.scales,
        get_opacity=lambda: field.opacities)
    _, operator, _, _ = projection(geometry, reference, height, width, world_scale)
    scope_policy = job.get("scope", "default")
    if scope_policy not in {"default", "positive_extent"}:
        raise ValueError("unknown registered extent policy")
    scale = None if scope_policy == "default" else raster_extent_scale(
        operator, masks[0], field.xyz, world_scale, field.default_scale)
    packet, registration = scribble_packet(operator, masks[0], masks[1] if job["mode"] == "signed" else None,
                                           scale=scale, preserve_rays=field.support_aggregation == 'expected_energy')
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    registered = time.monotonic()
    result = field.query(packet)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    decoded = time.monotonic()
    target = SimpleNamespace(**fixture["target"])
    alpha = selected_alpha(field, result.selected, target, height, width, world_scale)
    # Preserve continuous alpha through resizing; no PNG quantization in scoring.
    response = F.interpolate(alpha.reshape(1, 1, height, width), config["output_hw"],
                             mode="bilinear", align_corners=False)[0, 0].cpu()
    rh, rw = config["output_hw"]
    normalized = retained_response(field, result.membership_scores, target, rh, rw, world_scale).cpu()
    projected = time.monotonic()
    # Read-only agreement with the authorized prompt, never target labels.
    reference_alpha = selected_alpha(field, result.selected, reference, height, width, world_scale)
    reference_set = (reference_alpha > config["alpha_threshold"]).float().cpu().reshape(height, width)
    prompt_agreement = dict(positive_covered_fraction=float((reference_set*masks[0]).sum()/masks[0].sum()),
                            negative_selected_fraction=float((reference_set*masks[1]).sum()/masks[1].sum()))
    negative_observed = None
    if packet.negative is not None:
        p = packet.negative
        negative_observed = float(p.weights[field.observed[p.indices]].sum() / p.weights.sum())
    torch.save(dict(selected=result.selected.cpu(), membership=result.membership_scores.cpu(),
                    response=response, retained_response=normalized), out / "predictions.pt")
    report = dict(schema="radio_gs.query_field_nvos_prediction.v1", configuration=config, job=job,
                  field_sha256=manifest["files"]["field.pt"]["sha256"], field_spec=manifest["spec"],
                  fixture_sha256=sha256(job["camera_fixture"]), positive_sha256=sha256(job["positive"]),
                  negative_sha256=sha256(job["negative"]), predictions_sha256=sha256(out / "predictions.pt"),
                  selected=int(result.selected.sum()), scale_m=result.scale,
                  diagnostics=result.diagnostics, registration=registration, strokes=stroke_receipt,
                  prompt_agreement=prompt_agreement,
                  negative_observed_mass_fraction=negative_observed,
                  runtime_source_access=False, reference_rgb_opened=False, target_labels_opened=False,
                  negative_used_for_membership=job["mode"] == "signed",
                  deployment_bytes=manifest["files"]["field.pt"]["bytes"],
                  timing_seconds=dict(load=loaded-started, registration=registered-loaded,
                                      shared_decode=decoded-registered, target_projection=projected-decoded),
                  timing_condition="one run, shared GPU, excludes serialization and prompt-agreement audit",
                  peak_cuda_allocated_bytes=None if not str(device).startswith("cuda") else torch.cuda.max_memory_allocated(),
                  prompt_policy="official raw scribbles, pooled uncertain registered rays")
    (out / "prediction.json").write_text(json.dumps(report, indent=2)+"\n")
    return dict(job=job["name"], selected=report["selected"], scale_m=result.scale,
                registration=registration)


def score(scoring, output):
    import cv2
    from radio_gs.evaluation.promptable_segmentation import compute_binary_metrics
    from radio_gs.training.retrieval_metrics import binary_ranking_metrics
    config = json.loads(Path(scoring["prediction_config"]).read_text())
    root = Path(output)
    # Bind every arm before any label load, including arms not being scored first.
    for job in config["jobs"]:
        directory = root / job["name"]
        record = json.loads((directory / "prediction.json").read_text())
        if record["configuration"] != config or record["job"] != job:
            raise ValueError("prediction configuration changed before scoring")
        if sha256(directory / "predictions.pt") != record["predictions_sha256"]:
            raise ValueError("prediction payload changed before scoring")
    # Retained NVOS annotations may be at the original RGB resolution even
    # though the configured prediction raster is 1008x756. Resize continuous
    # responses to annotation size exactly as the historical scorer does.
    raw_target = np.asarray(Image.open(scoring["gt_mask"]))
    target = np.any(raw_target > 127, axis=2) if raw_target.ndim == 3 else raw_target > 127
    if target.ndim != 2:
        raise ValueError("invalid target mask")
    rows = []
    for job in config["jobs"]:
        data = torch.load(root / job["name"] / "predictions.pt", map_location="cpu", weights_only=True)
        response = cv2.resize(data["response"].numpy(), (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
        prediction = response > config["alpha_threshold"]
        metrics = compute_binary_metrics(prediction, target)
        intersection = int((prediction & target).sum())
        retained = torch.from_numpy(cv2.resize(data["retained_response"].numpy(),
                                    (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR))
        normalized_metrics = compute_binary_metrics(retained.numpy() >= config["alpha_threshold"], target)
        normalized_metrics.update(binary_ranking_metrics(retained.flatten(), torch.from_numpy(target.copy()).flatten()))
        rows.append(dict(job=job["name"], **metrics,
                         precision=intersection/max(int(prediction.sum()), 1),
                         recall=intersection/max(int(target.sum()), 1),
                         selected=int(data["selected"].sum()), retained_response=normalized_metrics))
    report = dict(schema="radio_gs.query_field_nvos_evaluation.v1", results=rows,
                  gt_sha256=sha256(scoring["gt_mask"]), prediction_config_sha256=sha256(scoring["prediction_config"]),
                  target_hw=list(target.shape), prediction_hw=config["output_hw"],
                  target_resize="retained cv2.INTER_LINEAR continuous response before threshold",
                  cases=1, full_cohort_cases=8, independent_final_test=False,
                  comparison_to_v10="different raw prompt, source-access and construction budgets",
                  threshold=config["alpha_threshold"], model_promotion=False)
    path = root / "evaluation.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["predict", "score"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--job")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deny-root", action="append", default=[])
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.deny_root:
        if args.stage != "predict":
            raise ValueError("source-denied mode is for prediction")
        deny_original_inputs(args.deny_root)
    config = json.loads(Path(args.config).read_text())
    if args.stage == "predict":
        job = next(j for j in config["jobs"] if j["name"] == args.job)
        result = predict(config, job, args.output, args.device)
    else:
        result = score(config, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
