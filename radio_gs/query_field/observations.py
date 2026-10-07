"""Compile native teachers through frozen Gaussian projections, before fitting.

Only this construction module can read source images/masks. Development views
are compiled for evaluation, never accumulated into the native scene channels.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image
import torch
from torch.nn import functional as F

from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.models.native_region_appearance import NativeRegionAppearance, CHECKPOINT
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from .storage import sha256
from .geometry import projection


def erode(mask, radius=1):
    shape = mask.shape
    values = mask.float().reshape(-1, 1, *shape[-2:])
    # Outside the raster is unknown, so boundary pixels are not trusted interiors.
    padded = F.pad(values, (radius, radius, radius, radius), value=0)
    return (-F.max_pool2d(-padded, 2*radius+1, stride=1)).reshape(shape) > 0.5


def partial_mask_target(masks, index, scales):
    """Nested/missing masks are unknown; only disjoint comparable scopes exclude."""
    positive = erode(masks[index])
    flat = masks.flatten(1)
    overlaps = (flat & flat[index]).any(dim=1)
    ratio = scales / scales[index].clamp_min(1e-6)
    eligible = (~overlaps) & (ratio >= 0.5) & (ratio <= 2)
    negative = erode(masks[eligible]).any(dim=0) if bool(eligible.any()) else torch.zeros_like(positive)
    negative &= ~masks[index]
    return positive, negative


def training_episode(view, index, observed):
    """One shared definition of known pixels and reliable input supports."""
    positive, negative = partial_mask_target(view['masks'], index, view['scales_m'])
    visible = view['raw_mass'].reshape(positive.shape) > 0
    positive &= visible
    negative &= visible
    pixels = torch.where(positive.flatten() & (view['best_element'] >= 0))[0]
    if len(pixels):
        pixels = pixels[observed[view['best_element'][pixels]]]
    if not len(pixels) or not bool(negative.any()):
        return None
    return positive.flatten(), negative.flatten(), pixels


def _file_receipt(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)


def source_sam_metadata(metadata):
    """Normalize two recorded query-free cache schemas without weakening checks."""
    if "generation_contract" in metadata:
        contract = metadata["generation_contract"]
        source_hash = metadata["source_image"]["sha256"]
        checkpoint = metadata["checkpoint"]
    elif metadata.get("schema_version") in {2, 3} and metadata.get("source") == "official_sam3_interactive_grid_multimask_hierarchy":
        contract = metadata
        source_hash = metadata["source_image_sha256"]
        checkpoint = dict(sha256=metadata["checkpoint_sha256"])
    else:
        raise ValueError("unrecognized source SAM provenance schema")
    if contract.get("query_free") is not True:
        raise ValueError("query-conditioned source masks are not construction supervision")
    return source_hash, checkpoint


@torch.no_grad()
def compile_observations(config_path, output, device="cuda"):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    scene_path = Path(config["scene"])
    scene = json.loads(scene_path.read_text())
    splits = {k: list(map(int, config[k])) for k in ["train_frames", "development_frames"]}
    if not splits["train_frames"] or not splits["development_frames"]:
        raise ValueError("construction requires explicit training and development frames")
    if set(splits["train_frames"]) & set(splits["development_frames"]):
        raise ValueError("feature training and development views overlap")
    frames = splits["train_frames"] + splits["development_frames"]
    if len(set(frames)) != len(frames) or not set(frames) <= set(scene["source_frames"]):
        raise ValueError("frames must be unique members of the declared source authority")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    geometry = FrozenGaussianGeometry.load(scene["geometry"]["path"],
                    expected_sha256=scene["geometry"]["sha256"], device=device)
    if "registered_cameras" in scene:
        from types import SimpleNamespace
        from radio_gs.geometry.registered_pinhole import load_registered_camera_table
        camera_path = scene["registered_cameras"]
        if sha256(camera_path) != scene["registered_cameras_sha256"]:
            raise ValueError("registered camera table digest differs")
        world_scale = float(scene["world_units_per_metre"])
        if not math.isfinite(world_scale) or world_scale <= 0:
            raise ValueError("registered camera world scale must be positive")
        cameras = SimpleNamespace(cameras=load_registered_camera_table(camera_path), scale=world_scale)
    else:
        camera_path = scene["camera_report"]
        cameras = MogeGeometryPrior(camera_path, scene["camera_report_sha256"])
    authority = json.loads(Path(scene["source_rgb_authority"]).read_text())
    images = {int(r["image_id"].split("_")[-1]): r["path"] for r in authority["images"]}
    height, width = config["raster_hw"]
    count = len(geometry.xyz)
    semantic_sum = torch.zeros(count, scene["native_dimension"])
    local_sum = torch.zeros(count, 768)
    semantic_mass = torch.zeros(count, 1)
    local_mass = torch.zeros(count, 1)
    encoder = NativeRegionAppearance(device=device)
    receipt = dict(schema="radio_gs.query_field_observations.v1", scene=scene["scene"],
                   configuration=_file_receipt(config_path), scene_configuration=_file_receipt(scene_path),
                   geometry=_file_receipt(scene["geometry"]["path"]),
                   camera_report=_file_receipt(camera_path),
                   source_authority=_file_receipt(scene["source_rgb_authority"]),
                   dino_checkpoint=_file_receipt(CHECKPOINT), frames=[], splits=splits,
                   native_semantic_space=scene["text_encoder"], world_units_per_metre=cameras.scale,
                   raster_hw=[height, width], semantic_labels_used=False,
                   benchmark_queries_used=False, feature_development_excluded=True,
                   geometry_development_excluded=False,
                   geometry_condition=scene.get("geometry_condition", "same frozen v10 carrier; feature-view holdout only"))
    for frame in frames:
        region_path = Path(scene["native_region_directories"][0]) / f"frame_{frame:05d}.pt"
        region = torch.load(region_path, map_location="cpu", weights_only=False)
        if region.get("query_independent") is not True or region["encoder"]["path"] != scene["text_encoder"]:
            raise ValueError("native region provider is not query-independent or its encoder differs")
        sam_path = Path(region["sam_path"])
        sam = torch.load(sam_path, map_location="cpu", weights_only=False)
        source = _file_receipt(images[frame])
        source_hash, sam_checkpoint = source_sam_metadata(sam["metadata"])
        if source_hash != source["sha256"]:
            raise ValueError("SAM cache does not match source image")
        raw_masks = unpack_masks(sam["packed_masks"], sam["mask_shape"][1])
        indices = region["native_proposal_indices"].long()
        raw_masks = torch.from_numpy(raw_masks.copy())[indices]
        masks = F.interpolate(raw_masks[:, None].float(), size=(height, width), mode="nearest-exact")[:, 0].bool()
        descriptors = F.normalize(region["descriptors"].float(), dim=-1)
        cover = masks.flatten(1).float()
        pixel_semantic = F.normalize(cover.T @ descriptors, dim=-1)
        known = cover.sum(0) > 0
        image = Image.open(images[frame]).convert("RGB")
        image = image.resize((config["dino_width"], round(image.height*config["dino_width"]/image.width)),
                             Image.Resampling.LANCZOS)
        patches, _ = encoder.patches(image)
        ph, pw = math.ceil(image.height/14), math.ceil(image.width/14)
        local_map = patches.T.reshape(1, 768, ph, pw)
        local_map = F.interpolate(local_map, (ph*14, pw*14), mode="bilinear", align_corners=False)
        local_map = local_map[:, :, :image.height, :image.width]
        local_pixels = F.normalize(F.interpolate(local_map, (height, width), mode="bilinear",
                                               align_corners=False)[0].flatten(1).T, dim=-1)
        raw, operator, mass, best = projection(geometry, cameras.cameras[frame], height, width, cameras.scale)
        mask_scales = []
        xyz = geometry.xyz.cpu()
        best_cpu = best.cpu()
        for mask in masks:
            rows = best_cpu[mask.flatten() & (best_cpu >= 0)]
            physical = 2*xyz[rows].std(dim=0, unbiased=False).norm()/cameras.scale if len(rows) else torch.tensor(0.)
            mask_scales.append(max(float(physical), 0.01))
        mask_scales = torch.tensor(mask_scales)
        if frame in splits["train_frames"]:
            adjoint = raw.transpose(0, 1).coalesce()
            for start in range(0, semantic_sum.shape[1], 128):
                semantic_sum[:, start:start+128] += torch.sparse.mm(adjoint,
                    pixel_semantic[:, start:start+128].to(device)).cpu()
            for start in range(0, local_sum.shape[1], 128):
                local_sum[:, start:start+128] += torch.sparse.mm(adjoint, local_pixels[:, start:start+128]).cpu()
            semantic_mass += torch.sparse.mm(adjoint, known.float().to(device)[:, None]).cpu()
            local_mass += torch.sparse.mm(adjoint, torch.ones(height*width, 1, device=device)).cpu()
        payload = dict(frame=frame, operator=operator.cpu(), raw_mass=mass.cpu(),
                       best_element=best_cpu, masks=masks, scales_m=mask_scales,
                       semantic_descriptors=descriptors.half(), semantic_pixels=pixel_semantic.half(),
                       local_pixels=local_pixels.cpu().half(),
                       intrinsic=cameras.cameras[frame].intrinsic,
                       world_to_camera=cameras.cameras[frame].world_to_camera,
                       original_hw=[cameras.cameras[frame].height, cameras.cameras[frame].width])
        target = output / f"view_{frame:05d}.pt"
        torch.save(payload, target)
        receipt["frames"].append(dict(frame=frame, split="train" if frame in splits["train_frames"] else "development",
                       rgb=source, region_cache=_file_receipt(region_path), sam_cache=_file_receipt(sam_path),
                       sam_checkpoint=sam_checkpoint, compiled=_file_receipt(target),
                       masks=len(masks), visible_pixels=int((mass > 0).sum()), nnz=operator._nnz()))
        print("compiled", scene["scene"], frame, "regions", len(masks), flush=True)
    semantic = F.normalize(semantic_sum/semantic_mass.clamp_min(1e-8), dim=-1).half()
    local = F.normalize(local_sum/local_mass.clamp_min(1e-8), dim=-1).half()
    observed = (semantic_mass[:, 0] > 0) & (local_mass[:, 0] > 0)
    geometry_state = {k: v.cpu() for k, v in geometry.state_dict().items()}
    torch.save(dict(geometry=geometry_state, semantic=semantic, local=local, observed=observed), output / "initial.pt")
    receipt["initial"] = _file_receipt(output / "initial.pt")
    receipt["observed_elements"] = int(observed.sum())
    receipt["elements"] = count
    (output / "observations.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return receipt
