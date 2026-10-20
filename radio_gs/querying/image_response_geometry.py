"""Native-ray image response fitting helpers, preserving the N14 arithmetic."""
from types import SimpleNamespace
import torch


def camera_args(geo, gp, frame, wh):
    width, height = wh
    cam = gp.cameras[frame]
    k = cam.intrinsic.cuda().clone()
    k[0] *= width / cam.width
    k[1] *= height / cam.height
    return dict(means=geo.xyz, quats=geo.quaternions, scales=geo.scales,
                opacities=geo.opacities.flatten(), viewmats=cam.world_to_camera.cuda()[None],
                Ks=k[None], width=width, height=height,
                near_plane=.001 * gp.scale, far_plane=100 * gp.scale, packed=False)


def render_responses(args, values, observed):
    from gsplat import rasterization
    colors = torch.cat([v.cuda() for v in values] + [observed.cuda().float()[:, None]], dim=1)
    image, alpha, _ = rasterization(**args, colors=colors)
    mass = image[0, ..., -1]
    valid = (mass / alpha[0, ..., 0].clamp_min(1e-8) > .5) & (alpha[0, ..., 0] > 1e-8)
    return torch.where(valid[..., None], image[0, ..., :-1] / mass.clamp_min(1e-8)[..., None], 0).cpu()


def contributions(geo, args, backend):
    from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
    renderer = SimpleNamespace(scaled_intrinsics=lambda width, height: args['Ks'][0],
        near_plane=args['near_plane'], far_plane=args['far_plane'])
    hit = rasterize_single_view_contributions(geo, renderer, args['viewmats'][0],
        height=args['height'], width=args['width'], include_depths=False, footprint_backend=backend)
    error = float((hit['accumulated_alpha'] - hit['rendered_alpha']).abs().max())
    if error >= 1e-4:
        raise ValueError('contribution alpha differs from renderer')
    return hit, error


def normalized_image_operator(hit, observed, pixels):
    """Observed-support normalization; zero/unseen pixels carry no authority."""
    pid, gid = hit['pixel_ids'].long(), hit['gaussian_ids'].long()
    weight = hit['weights'].float() * observed.to(pid.device)[gid]
    mass = torch.zeros(pixels, device=pid.device).scatter_add_(0, pid, weight)
    valid = (mass / hit['rendered_alpha'].flatten().clamp_min(1e-8) > .5) & (mass > 1e-8)
    op = torch.sparse_coo_tensor(torch.stack([pid, gid]), weight / mass[pid].clamp_min(1e-8),
                                 (pixels, len(observed))).coalesce().cpu()
    return op, valid.cpu()[:, None]
