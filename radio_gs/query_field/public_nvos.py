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
    def scalar_mask(path):
        array = np.asarray(Image.open(path))
        # Official trex strokes store the same scalar mask in all RGB channels.
        # Collapse only exact replicas; coloured annotations need an explicit contract.
        if array.ndim == 3 and array.shape[2] == 3 and np.all(array == array[..., :1]):
            array = array[..., 0]
        return array
    positive = scalar_mask(positive_path)
    negative = scalar_mask(negative_path)
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
    input_contract=job.get('prompt_input','raw_scribbles')
    if input_contract not in {'raw_scribbles','reference_mask'}:raise ValueError('unknown prompt input contract')
    reference_receipt=None
    if input_contract=='reference_mask':
        if job['mode']!='positive':raise ValueError('reference mask intervention is positive-only')
        if job['reference_frame']!=reference.frame:raise ValueError('reference mask/camera frame mismatch')
        raw=np.asarray(Image.open(job['reference_mask']))
        foreground=np.any(raw>127,axis=2) if raw.ndim==3 else raw>127
        if foreground.shape!=tuple(reversed(job['reference_raster_wh'])) or not foreground.any():
            raise ValueError('reference mask raster differs from declared query')
        reference_receipt=dict(path=job['reference_mask'],sha256=sha256(job['reference_mask']),native_hw=list(foreground.shape),
            input_condition='retained official-positive-derived SAM query mask; not raw scribbles',
            generating_rgb_sam_cost_inherited=True,query_mask_inferred_this_run=False)
        # Keep official strokes separately for audit; only positive inference evidence changes.
        inference_masks=masks.clone()
        inference_masks[0]=F.interpolate(torch.from_numpy(foreground.copy()).float()[None,None],size=(height,width),mode='area')[0,0]
    else:inference_masks=masks
    geometry = SimpleNamespace(xyz=field.xyz, get_xyz=lambda: field.xyz,
        get_rotation=lambda: field.quaternions, get_scaling=lambda: field.scales,
        get_opacity=lambda: field.opacities)
    _, operator, _, _ = projection(geometry, reference, height, width, world_scale)
    scope_policy = job.get("scope", "default")
    if scope_policy not in {"default", "positive_extent"}:
        raise ValueError("unknown registered extent policy")
    scale = None if scope_policy == "default" else raster_extent_scale(
        operator, inference_masks[0], field.xyz, world_scale, field.default_scale)
    packet, registration = scribble_packet(operator, inference_masks[0], inference_masks[1] if job["mode"] == "signed" else None,
                                           scale=scale, preserve_rays=field.support_aggregation == 'expected_energy',
                                           negative_rays=field.negative_rule == 'competitive_rays')
    if job.get('query_program','').startswith('positive_ray'):
        from dataclasses import replace
        from .adapters import raster_support
        positive_rays,_=raster_support(operator,inference_masks[0],preserve_rays=True)
        packet=replace(packet,positive=positive_rays)
        if job.get('query_program') in {'positive_ray_expected_signed','positive_ray_expected_competitive','positive_ray_positive_scope','positive_ray_signed_scope'} and job['mode']=='signed':
            negative_rays,_=raster_support(operator,masks[1],preserve_rays=True)
            packet=replace(packet,negative=negative_rays)
            registration['negative_ray_hypotheses_retained']=True
        registration['positive_ray_hypotheses_retained']=True
        registration['positive_ray_hypotheses']=int(positive_rays.components.unique().numel())
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    registered = time.monotonic()
    program=job.get('query_program','parent')
    extent_state=None
    if program in {'reference_extent_prior','reference_extent_lifted','reference_extent_literal','reference_extent_coupled','reference_extent_observation','reference_extent_parts','reference_extent_shuffled','reference_extent_none','reference_extent_edit'}:
        if input_contract!='reference_mask':raise ValueError('reference extent needs an explicit full query mask')
        from .runtime import QueryFieldRuntime
        extent_state={}
        result=QueryFieldRuntime(field).reference_extent(packet,reference,inference_masks[0],
            initializer='lifted' if program=='reference_extent_lifted' else 'prior',
            decision_cutoff=10.5/255 if program=='reference_extent_literal' else config['alpha_threshold'],
            observation_weight={'reference_extent_coupled':.5,'reference_extent_observation':1.,'reference_extent_parts':.5,'reference_extent_shuffled':.5,'reference_extent_none':.5,'reference_extent_edit':.5}.get(program,0.),
            grouping={'reference_extent_parts':'incidence','reference_extent_shuffled':'shuffled'}.get(program),
            regularizer={'reference_extent_none':'none','reference_extent_edit':'edit'}.get(program,'footprint'),
            audit_state=extent_state)
    elif program in {'reference_scope','reference_scope_visible','reference_scope_conditioned','reference_scope_admitted'}:
        if input_contract!='reference_mask':raise ValueError('composition needs a declared reference mask')
        from .runtime import QueryFieldRuntime
        result=QueryFieldRuntime(field).reference_scope(packet,reference,inference_masks[0],world_scale=world_scale,
            observation="subset" if program=="reference_scope" else "scene",
            member_conditioning=program in {'reference_scope_conditioned','reference_scope_admitted'},
            member_readout='admission' if program=='reference_scope_admitted' else 'fraction')
    elif program=='parent':result=field.query(packet)
    elif program in {'joint_unsigned','joint_signed'}:
        from .runtime import QueryFieldRuntime
        result=QueryFieldRuntime(field).signed_regions(packet,joint=program=='joint_signed')
    elif program in {'positive_ray_shared','positive_ray_local','positive_ray_union','positive_ray_confidence_union','positive_ray_coherent_union','positive_ray_expected_union','positive_ray_expected_signed','positive_ray_expected_competitive','positive_ray_positive_scope','positive_ray_signed_scope'}:
        from .runtime import QueryFieldRuntime
        result=QueryFieldRuntime(field).positive_rays(packet,shared_centre=program=='positive_ray_shared',
            aggregation={'positive_ray_union':'union','positive_ray_confidence_union':'confidence_union','positive_ray_coherent_union':'coherent_union','positive_ray_expected_union':'expected_union','positive_ray_expected_signed':'expected_union','positive_ray_expected_competitive':'expected_union','positive_ray_positive_scope':'expected_union','positive_ray_signed_scope':'expected_union'}.get(program,'mixture'),
            negative_aggregation={'positive_ray_expected_signed':'expected_union','positive_ray_expected_competitive':'expected_competition','positive_ray_positive_scope':'expected_competition','positive_ray_signed_scope':'expected_competition'}.get(program,'legacy'),
            region_scope={'positive_ray_positive_scope':'positive_all','positive_ray_signed_scope':'signed_all'}.get(program,'local'))
    else:raise ValueError('unknown signed query program')
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
                  prompt_input=input_contract,reference_mask_receipt=reference_receipt,
                  prompt_policy=("official raw scribbles; uncertain registered rays; signed policy declared in field spec" if input_contract=='raw_scribbles' else
                                 "explicit retained positive-derived SAM reference mask; generating RGB/SAM cost inherited; no tracks"))
    if extent_state is not None:
        torch.save(extent_state,out/'extent_state.pt')
        report['extent_state']=dict(path=str(out/'extent_state.pt'),sha256=sha256(out/'extent_state.pt'))
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
