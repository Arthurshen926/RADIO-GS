"""Fixed development diagnostics, explicitly distinct from benchmark results."""
import gc
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from .adapters import pixel_support
from .contracts import QueryPacket, Support
from .geometry import selected_alpha
from .observations import erode, partial_mask_target
from .storage import load_field, sha256
from .train import partial_membership_loss


def mask_iou(prediction, target):
    union = (prediction | target).sum()
    return float((prediction & target).sum()/union) if int(union) else 1.


def choose_pixel(mask):
    """Interior nearest the mask's centroid; no model/visibility-based selection."""
    pixels = torch.nonzero(erode(mask))
    if not len(pixels):
        pixels = torch.nonzero(mask)
    if not len(pixels):
        return None
    centre = torch.nonzero(mask).float().mean(0)
    pixel = pixels[((pixels-centre)**2).sum(1).argmin()]
    return int(pixel[0]*mask.shape[1]+pixel[1])


def _summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["mode"]+"/"+row["scale_condition"]].append(row)
    result = {}
    for name, group in groups.items():
        # Failures remain zero-IoU entries in the complete teacher-region roster.
        item = dict(cases=len(group), failed=sum(r["status"] != "ok" for r in group),
            mean_projected_membership_iou=sum(r["projected_membership_iou"] for r in group)/len(group))
        rendered = [r for r in group if "actual_subset_iou" in r]
        if rendered:
            item["mean_actual_subset_iou"] = sum(r["actual_subset_iou"] for r in rendered)/len(rendered)
            item["actual_subset_cases"] = len(rendered)
        losses = [r["partial_loss"] for r in group if r.get("partial_loss") is not None]
        item["mean_partial_loss_on_supported_cases"] = sum(losses)/len(losses) if losses else None
        result[name] = item
    return result


@torch.no_grad()
def evaluate(output, device="cuda", stages=("untrained", "trained")):
    root = Path(output)
    target = root / "development.json"
    if target.exists():
        raise FileExistsError("development report already exists; preserve the fixed run")
    observation_root = root / "observations"
    manifest = json.loads((observation_root / "observations.json").read_text())
    height, width = manifest["raster_hw"]
    views = []
    for entry in manifest["frames"]:
        if entry["split"] != "development":
            continue
        path = observation_root / f"view_{entry['frame']:05d}.pt"
        if sha256(path) != entry["compiled"]["sha256"]:
            raise ValueError("development observation digest differs")
        views.append(torch.load(path, map_location="cpu", weights_only=True))
    report = dict(schema="radio_gs.query_field_development.v1", benchmark_result=False,
        independent_final_test=False, feature_training_frames=manifest["splits"]["train_frames"],
        development_frames=manifest["splits"]["development_frames"],
        geometry_condition=manifest["geometry_condition"], observations_sha256=sha256(observation_root / "observations.json"),
        query_contract="All development SAM regions, centroid-interior pixel fixed without model scores; failures retained at zero IoU.",
        image_contract="Teacher-region diagnostic: native masked-crop SigLIP plus up to 16 uniformly sampled interior DINO pixels, not external reference-image evaluation.",
        point_3d_contract="Dominant Gaussian at the registered 2D prompt, not human annotated 3D instance identity.",
        scale_conditions={"default": "Training-mask median physical scale, no development scope access.",
                          "teacher_scope": "Privileged development mask physical extent, diagnostic only."},
        metrics={"projected_membership_iou": "Threshold normalized full-scene alpha-weighted member scores at 0.5 against a complete source SAM mask.",
                 "actual_subset_iou": "Select Gaussian rows at fixed 0.5, re-rasterize only this subset, threshold accumulated alpha at 0.5. Default-scale cases only.",
                 "partial_loss": "Teacher interior versus disjoint comparable-scale masks; unknown areas omitted. Not verified whole-instance truth."},
        text_benchmark_evaluated=False, source_rgb_opened_by_query=False, stages={})
    if not stages or any(stage not in {'untrained','trained'} for stage in stages):
        raise ValueError('unknown evaluation stage')
    for stage in stages:
        started = time.monotonic()
        field, export = load_field(root / "models" / stage, device)
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        rows = []
        for view in views:
            operator = view["operator"].to(device)
            camera = SimpleNamespace(intrinsic=view["intrinsic"], world_to_camera=view["world_to_camera"],
                                     height=view["original_hw"][0], width=view["original_hw"][1])
            for index, mask in enumerate(view["masks"]):
                pixel = choose_pixel(mask)
                target_mask = mask.flatten().to(device)
                positive, negative = partial_mask_target(view["masks"], index, view["scales_m"])
                visible = view["raw_mass"].reshape(mask.shape) > 0
                positive = (positive & visible).flatten().to(device)
                negative = (negative & visible).flatten().to(device)
                packets = {}
                errors = {}
                if pixel is None:
                    errors = {name:"mask vanished at the fixed raster resolution" for name in ["region_image", "point_2d", "point_3d"]}
                else:
                    try:
                        packets["point_2d"] = QueryPacket("point_2d", positive=pixel_support(operator, pixel))
                    except ValueError as exc:
                        errors["point_2d"] = str(exc)
                    anchor = int(view["best_element"][pixel])
                    if anchor >= 0:
                        packets["point_3d"] = QueryPacket("point_3d", exact_positive=True,
                            positive=Support(torch.tensor([anchor], device=device), torch.ones(1, device=device)))
                    else:
                        errors["point_3d"] = "prompt has no projected Gaussian"
                    inside = torch.where(erode(mask).flatten())[0]
                    if not len(inside):
                        inside = torch.where(mask.flatten())[0]
                    positions = torch.linspace(0, len(inside)-1, min(16,len(inside))).round().long()
                    packets["region_image"] = QueryPacket("image", semantic=view["semantic_descriptors"][index].to(device),
                                                          local=view["local_pixels"][inside[positions]].to(device))
                for mode in ["point_2d", "point_3d", "region_image"]:
                    for condition in ["default", "teacher_scope"]:
                        row = dict(frame=view["frame"], region=index, mode=mode, scale_condition=condition,
                                   prompt_pixel=pixel, status="ok", projected_membership_iou=0., partial_loss=None)
                        if condition == "default":
                            row["actual_subset_iou"] = 0.
                        try:
                            if mode in errors:
                                raise ValueError(errors[mode])
                            packet = replace(packets[mode], scale=None if condition == "default" else float(view["scales_m"][index]))
                            result = field.query(packet)
                            probability = torch.sparse.mm(operator, result.membership_scores[:, None])[:, 0]
                            row["projected_membership_iou"] = mask_iou(probability > field.threshold, target_mask)
                            row["selected_gaussians"] = int(result.selected.sum())
                            row["diagnostics"] = result.diagnostics
                            if bool(positive.any()) or bool(negative.any()):
                                row["partial_loss"] = float(partial_membership_loss(probability, positive, negative))
                            if condition == "default":
                                alpha = selected_alpha(field, result.selected, camera, height, width, manifest["world_units_per_metre"])
                                row["actual_subset_iou"] = mask_iou(alpha > .5, target_mask)
                        except ValueError as exc:
                            row["status"] = "unsupported"
                            row["reason"] = str(exc)
                        rows.append(row)
                if index % 20 == 0:
                    print("evaluate", stage, view["frame"], index, "/", len(view["masks"]), flush=True)
            del operator
        payload = dict(summary=_summarize(rows), rows=rows, seconds=time.monotonic()-started,
            field_sha256=export["files"]["field.pt"]["sha256"], deployment_bytes=export["files"]["field.pt"]["bytes"],
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else None)
        (root / f"development_{stage}.json").write_text(json.dumps(payload, indent=2)+"\n")
        report["stages"][stage] = {k:v for k,v in payload.items() if k != "rows"}
        del field
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    report["promotion"] = False
    report["next_gate"] = "Diagnose conditional grouping and identity separately; this pilot is not comparable to saved v10 benchmark accuracy."
    target.write_text(json.dumps(report, indent=2)+"\n")
    return report
