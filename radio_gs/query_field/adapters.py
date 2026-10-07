"""Input adapters construct evidence; they never produce an independent mask."""
from types import SimpleNamespace

import torch

from .contracts import QueryPacket, Support


def point_3d(field, xyz, *, negative_xyz=None, scale=None, maximum_distance=0.1):
    """A world point is snapped only when the declared geometric tolerance holds."""
    def locate(point):
        point = torch.as_tensor(point, device=field.xyz.device).float()
        if point.shape != (3,) or not bool(torch.isfinite(point).all()):
            raise ValueError("3D query must be a finite world coordinate")
        distance = (field.xyz-point).norm(dim=1)
        index = distance.argmin()
        if float(distance[index]) > maximum_distance:
            raise ValueError("3D query has no Gaussian within registration tolerance")
        return Support(index.reshape(1), torch.ones(1, device=field.xyz.device))
    return QueryPacket("point_3d", positive=locate(xyz),
                       negative=None if negative_xyz is None else locate(negative_xyz),
                       scale=scale, exact_positive=True)


def pixel_support(operator, pixel):
    """All Gaussian contributions to a pixel remain weighted uncertain support."""
    operator = operator.coalesce()
    if pixel < 0 or pixel >= operator.shape[0]:
        raise ValueError("pixel outside camera raster")
    chosen = operator.indices()[0] == pixel
    indices = operator.indices()[1, chosen]
    weights = operator.values()[chosen]
    support = Support(indices, weights)
    support.validate(operator.shape[1])
    return support


def raster_support(operator, evidence):
    """Pool registered pixel evidence without choosing a dominant Gaussian.

    Each nonempty row of ``operator`` is a normalized uncertain camera ray.
    Evidence may contain fractional area after resampling thin scribbles.
    Unsupported evidence is reported, never replaced by a nearby pixel.
    """
    operator = operator.coalesce()
    evidence = torch.as_tensor(evidence, device=operator.device).float().flatten()
    if evidence.shape != (operator.shape[0],) or not torch.isfinite(evidence).all() or (evidence < 0).any():
        raise ValueError("raster evidence must be aligned finite nonnegative weights")
    if not bool(evidence.sum() > 0):
        raise ValueError("raster evidence is empty")
    pixels, rows = operator.indices().cpu()
    values = operator.values().double().cpu()
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("invalid camera contribution weights")
    # Ordered CPU float64 accumulation avoids CUDA atomic-order differences in
    # large scribbles. This is transient registration, not a new member decoder.
    mass = torch.bincount(pixels, weights=values, minlength=operator.shape[0])
    evidence_cpu = evidence.double().cpu()
    # Normalize here as well, so callers may supply either raw or normalized rays.
    pooled = torch.bincount(rows, weights=values / mass[pixels].clamp_min(1e-12) * evidence_cpu[pixels],
                            minlength=operator.shape[1])
    indices = torch.where(pooled > 0)[0]
    support = Support(indices.to(operator.device), pooled[indices].float().to(operator.device))
    support.validate(operator.shape[1])
    return support, float(evidence_cpu[mass > 0].sum() / evidence_cpu.sum())


def scribble_packet(operator, positive, negative=None, *, scale=None):
    """Registered raster evidence uses the same point packet and member core.

    Positive/negative rays can share contributors. They are uncertain supports,
    not exact element constraints. Native-raster sign overlap must be checked
    before resampling, where two disjoint strokes can share a coarse pixel.
    """
    support, positive_coverage = raster_support(operator, positive)
    exclusion, negative_coverage = (None, None) if negative is None else raster_support(operator, negative)
    packet = QueryPacket("point_2d", positive=support, negative=exclusion, scale=scale)
    return packet, dict(positive_projected_mass_fraction=positive_coverage,
                       negative_projected_mass_fraction=negative_coverage,
                       positive_gaussians=len(support.indices),
                       negative_gaussians=0 if exclusion is None else len(exclusion.indices),
                       exact_element_constraints=False)


def raster_extent_scale(operator, positive, xyz, world_units_per_metre, default_scale):
    """Infer a physical scope floor from distinct positive ray centres.

    The convention 2 * spatial standard deviation matches source mask scopes.
    Within-ray depth uncertainty is not counted as object extent. One pixel
    carries no extent information and retains the scene's default scale.
    """
    import math
    if any(not math.isfinite(float(v)) or v <= 0 for v in [world_units_per_metre, default_scale]):
        raise ValueError("extent inference requires positive physical units and default scope")
    operator = operator.coalesce()
    raster_support(operator, positive)  # Same alignment/coverage validation.
    # The small geometric scope statistic also has a deterministic CPU path.
    operator = operator.cpu().double()
    weights = torch.as_tensor(positive).double().cpu().flatten()
    mass = torch.sparse.sum(operator, dim=1).to_dense()
    active = (weights > 0) & (mass > 0)
    if int(active.sum()) <= 1:
        return float(default_scale)
    if xyz.shape != (operator.shape[1], 3) or not torch.isfinite(xyz).all():
        raise ValueError("ray centres require finite aligned Gaussian coordinates")
    centres = (torch.sparse.mm(operator, xyz.double().cpu()) /
               mass.clamp_min(1e-12)[:, None])[active]
    weights = weights[active] / weights[active].sum()
    centre = (centres * weights[:, None]).sum(0)
    variance = ((centres-centre).square().sum(1) * weights).sum()
    inferred = 2*float(variance.sqrt()) / world_units_per_metre
    return max(float(default_scale), inferred)


def point_2d(field, pixel_xy, intrinsic, world_to_camera, image_hw, *,
             scale=None, negative_xy=None, world_units_per_metre=1.):
    """Camera and pixels only: no RGB, depth, teacher, or source cache access."""
    from .geometry import projection
    # The numerical projector consumes only these Gaussian properties.
    geometry = SimpleNamespace(xyz=field.xyz, get_xyz=lambda:field.xyz,
        get_rotation=lambda:field.quaternions, get_scaling=lambda:field.scales,
        get_opacity=lambda:field.opacities)
    height, width = map(int, image_hw)
    k = torch.as_tensor(intrinsic).float()
    pose = torch.as_tensor(world_to_camera).float()
    if k.shape != (3,3) or pose.shape != (4,4) or not torch.isfinite(k).all() or not torch.isfinite(pose).all():
        raise ValueError("invalid registered camera")
    if min(height,width) <= 0 or k[0,0] <= 0 or k[1,1] <= 0:
        raise ValueError("invalid image dimensions or camera focal lengths")
    camera = SimpleNamespace(intrinsic=k, world_to_camera=pose, height=height, width=width)
    _, operator, _, _ = projection(geometry, camera, height, width, world_units_per_metre)
    def support(xy):
        x,y = map(int,xy)
        if x < 0 or x >= width or y < 0 or y >= height:
            raise ValueError("prompt pixel outside camera image")
        return pixel_support(operator,y*width+x)
    return QueryPacket("point_2d", positive=support(pixel_xy),
                       negative=None if negative_xy is None else support(negative_xy), scale=scale)


class NativeQueryEncoder:
    """Frozen native encoders used only on the current authorized query input."""

    def __init__(self, siglip_path, device="cuda"):
        self.siglip_path = siglip_path
        self.device = device
        self._text = self._vision = self._dino = None

    def text(self, expression, *, task="instance", scale=None):
        from radio_gs.models.native_query_encoders import load_siglip_text
        if self._text is None:
            # Reuse the strict checkpoint head restoration: the native giant
            # tower projects 1152 hidden channels to 1536 semantic channels.
            model, tokenizer = load_siglip_text(self.siglip_path, self.device)
            if torch.device(self.device).type == "cpu":
                model = model.float()
            self._text = (tokenizer,model)
        tokenizer,model = self._text
        with torch.no_grad():
            values = tokenizer([expression], padding="max_length", max_length=64,
                               truncation=True, return_tensors="pt").to(self.device)
            semantic = model(**values).pooler_output[0]
        return QueryPacket("text", task=task, semantic=semantic, scale=scale)

    def image(self, image, *, scale=None):
        import numpy as np
        from pathlib import Path
        from PIL import Image
        from radio_gs.models.native_region_appearance import NativeRegionAppearance
        from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
        if self._vision is None:
            self._vision = NativeSiglip2Runtime.load(Path(self.siglip_path), device=torch.device(self.device))
        if self._dino is None:
            self._dino = NativeRegionAppearance(device=self.device)
        pixels = torch.from_numpy(np.array(image.convert("RGB").resize((384,384),Image.Resampling.BICUBIC),copy=True))
        semantic = self._vision.encode(pixels.permute(2,0,1).float()[None]/255)[0]
        resized = image.resize((448,round(image.height*448/image.width)),Image.Resampling.LANCZOS)
        local,_ = self._dino.patches(resized)
        return QueryPacket("image", semantic=semantic, local=local, scale=scale)
