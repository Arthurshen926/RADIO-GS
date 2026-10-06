"""Privileged teacher-mask projection controls, never deployed predictions."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .evaluate import mask_iou
from .geometry import selected_alpha
from .storage import load_field, sha256


@torch.no_grad()
def diagnose(root, device="cuda"):
    root = Path(root)
    target = root / "projection_diagnostic.json"
    if target.exists():
        raise FileExistsError("preserve the existing diagnostic")
    observations = root / "observations"
    manifest = json.loads((observations / "observations.json").read_text())
    field, export = load_field(root / "models" / "trained", device)
    height, width = manifest["raster_hw"]
    rows, frames = [], []
    for entry in manifest["frames"]:
        if entry["split"] != "development":
            continue
        path = observations / f"view_{entry['frame']:05d}.pt"
        if sha256(path) != entry["compiled"]["sha256"]:
            raise ValueError("development observation changed")
        view = torch.load(path, map_location="cpu", weights_only=True)
        operator = view["operator"].to(device).coalesce()
        raw = torch.sparse_coo_tensor(operator.indices(),
            operator.values()*view["raw_mass"].to(device)[operator.indices()[0]], operator.shape).coalesce()
        adjoint = raw.transpose(0,1).coalesce()
        total = torch.sparse.mm(adjoint, torch.ones(height*width,1,device=device))[:,0]
        known_mass = torch.sparse.mm(operator, field.observed.float()[:,None])[:,0]
        camera = SimpleNamespace(intrinsic=view["intrinsic"], world_to_camera=view["world_to_camera"],
                                 height=view["original_hw"][0], width=view["original_hw"][1])
        frames.append(dict(frame=entry["frame"], pixels=height*width,
            original_alpha_above_half=float((view["raw_mass"]>.5).float().mean()),
            mean_trained_observation_ray_mass=float(known_mass.mean()),
            rays_with_majority_observed=float((known_mass>.5).float().mean())))
        for index, mask in enumerate(view["masks"]):
            mask = mask.flatten().to(device)
            positive = torch.sparse.mm(adjoint, mask.float()[:,None])[:,0]
            # A per-view target-mask backprojection is a privileged diagnostic.
            # It is not a mathematically tight best-possible segmentation bound.
            score = positive/total.clamp_min(1e-8)
            row = dict(frame=entry["frame"], region=index,
                       positive_ray_observation_mass=float(known_mass[mask].mean()) if mask.any() else 0.)
            for condition, selected in [("all_geometry",score>.5),("observed_geometry",(score>.5)&field.observed)]:
                alpha = selected_alpha(field, selected, camera, height, width, manifest["world_units_per_metre"])
                row[condition+"_actual_subset_iou"] = mask_iou(alpha>.5,mask)
            rows.append(row)
        print("diagnosed projection",entry["frame"],len(view["masks"]),flush=True)
    report = dict(schema="radio_gs.query_field_projection_diagnostic.v1", benchmark_result=False,
        privileged_development_masks_used=True, deployed_weights_modified=False,
        diagnostic="Per-view native teacher mask backprojected through exact Gaussian contributions at fixed 0.5; not a deployable method or tight oracle upper bound.",
        field_sha256=export["files"]["field.pt"]["sha256"], frames=frames, rows=rows,
        mean_all_geometry_actual_subset_iou=sum(r["all_geometry_actual_subset_iou"] for r in rows)/len(rows),
        mean_observed_geometry_actual_subset_iou=sum(r["observed_geometry_actual_subset_iou"] for r in rows)/len(rows),
        all_cases=len(rows))
    target.write_text(json.dumps(report,indent=2)+"\n")
    return {k:v for k,v in report.items() if k!="rows"}
