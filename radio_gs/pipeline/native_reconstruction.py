"""Shared query-independent native-feature writing on frozen geometry.

Region descriptors keep their declared native encoder coordinates. The default
mixture averages overlapping directions. The explicit finest-region experiment
uses one direction per pixel; neither policy certifies physical object identity.
The frozen alpha compositor assigns the resulting direction to visible elements.
This is not a learned compact student or proof of whole/part disentanglement.
"""
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions


class NativeFieldAccumulator:
    def __init__(self, geometry, dimension, device='cuda', region_policy='mixture'):
        self.region_policy = region_policy
        self.geometry = geometry
        self.dimension = dimension
        self.device = torch.device(device)
        self.sums = torch.zeros(len(geometry.xyz), dimension)
        self.mass = torch.zeros(len(geometry.xyz), 1, device=self.device)

    @torch.inference_mode()
    def add_view(self, masks, descriptors, intrinsic, world_to_camera, *, height, width,
                 image_height, image_width, near_plane=.001, far_plane=100.):
        if len(descriptors) == 0:
            return dict(regions=0, covered_pixels=0, touched_elements=0, raster_hw=[height, width])
        masks = torch.as_tensor(masks).float()
        pixel_masks = F.interpolate(masks[:, None], size=(height, width), mode='nearest-exact')[:, 0].flatten(1).to(self.device)
        count = pixel_masks.sum(0)
        # The serialized half-precision teacher cache is the common producer
        # contract across datasets; normalize its direction after loading.
        descriptors = F.normalize(descriptors.half().float().to(self.device), dim=-1)
        if self.region_policy == 'finest':
            # Preserve one observed region direction per pixel instead of
            # blending the object, its parts and surrounding context. Original
            # image areas determine ownership independently of raster size.
            area = masks.flatten(1).sum(1).to(self.device)
            owner = area[:, None].expand_as(pixel_masks).masked_fill(pixel_masks == 0, torch.inf).argmin(0)
            target = descriptors[owner] * (count > 0)[:, None]
        else:
            target = F.normalize(pixel_masks.T @ descriptors / count[:, None].clamp_min(1), dim=-1)
        k = torch.as_tensor(intrinsic, device=self.device, dtype=torch.float32).clone()
        k[0] *= width / image_width
        k[1] *= height / image_height
        renderer = SimpleNamespace(scaled_intrinsics=lambda w, h: k, near_plane=near_plane, far_plane=far_plane)
        hit = rasterize_single_view_contributions(self.geometry, renderer,
            torch.as_tensor(world_to_camera, device=self.device, dtype=torch.float32),
            height=height, width=width, include_depths=False)
        pixels, elements = hit['pixel_ids'].long(), hit['gaussian_ids'].long()
        weights = hit['weights'].float() * (count[pixels] > 0)
        active = weights > 0
        touched, inverse = torch.unique(elements[active], sorted=True, return_inverse=True)
        adjoint = torch.sparse_coo_tensor(torch.stack([inverse, pixels[active]]), weights[active],
                                        (len(touched), height * width)).coalesce()
        for start in range(0, self.dimension, 128):
            self.sums[touched.cpu(), start:start + 128] += torch.sparse.mm(
                adjoint, target[:, start:start + 128].contiguous()).cpu()
        self.mass.index_add_(0, touched, torch.sparse.mm(adjoint, torch.ones(height * width, 1, device=self.device)))
        return dict(regions=len(descriptors), covered_pixels=int((count > 0).sum()),
                    touched_elements=len(touched), raster_hw=[height, width])

    @torch.inference_mode()
    def field(self, geometry_binding, native_space, calibration=None):
        calibration = calibration or dict(log_scale=0., bias=0., kind='uncalibrated_native_ranking')
        return dict(features=F.normalize(self.sums, dim=-1).half(), supported=(self.mass > 0).cpu(),
                    geometry=geometry_binding, native_space=native_space,
                    log_scale=calibration['log_scale'], bias=calibration['bias'],
                    calibration=calibration, region_policy=self.region_policy, feature_write=f'native {self.region_policy} region ownership, alpha adjoint',
                    descriptor_contract='normalized serialized fp16 native coordinates')
