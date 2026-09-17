"""Source-only multiplicative depth calibration without changing depth shape."""
import torch


def fit_source_depth_scale(predicted, reference, *, minimum_samples=20, maximum_median_relative_error=.1):
    predicted=torch.as_tensor(predicted,dtype=torch.float64).reshape(-1)
    reference=torch.as_tensor(reference,dtype=torch.float64).reshape(-1)
    if predicted.shape!=reference.shape:raise ValueError('depth sample axes differ')
    valid=torch.isfinite(predicted)&torch.isfinite(reference)&(predicted>0)&(reference>0)
    p,r=predicted[valid],reference[valid]
    if len(p)<minimum_samples:return dict(accepted=False,scale=1.,sample_count=len(p),reason='insufficient_source_tracks',training_median_relative_error=None)
    scale=float((r/p).median());error=float(((p*scale-r).abs()/r).median())
    accepted=error<=maximum_median_relative_error
    return dict(accepted=accepted,scale=scale,sample_count=len(p),reason=None if accepted else 'inconsistent_source_depth_shape',training_median_relative_error=error)


def fit_reprojection_scale(ray_depth_world, camera_origin, target_poses, target_intrinsics,
                           target_pixels, *, initial_scale, minimum_samples=20, steps=200):
    """Experimental positive scalar fit to fixed source-track reprojections.

    ray_depth_world is the unscaled source camera ray times predicted depth,
    rotated into world axes, one row per directed correspondence. Target pixels
    must be training tracks only. No geometry adoption decision is made here.
    """
    ray = torch.as_tensor(ray_depth_world, dtype=torch.float64)
    origin = torch.as_tensor(camera_origin, dtype=torch.float64)
    poses = torch.as_tensor(target_poses, dtype=torch.float64)
    intrinsics = torch.as_tensor(target_intrinsics, dtype=torch.float64)
    pixels = torch.as_tensor(target_pixels, dtype=torch.float64)
    n = len(ray)
    if ray.shape != (n, 3) or origin.shape != (3,) or poses.shape != (n, 4, 4) or intrinsics.shape != (n, 3, 3) or pixels.shape != (n, 2):
        raise ValueError('reprojection sample axes differ')
    if not all(bool(torch.isfinite(x).all()) for x in (ray, origin, poses, intrinsics, pixels)):
        raise ValueError('nonfinite correspondence')
    if not 0 < initial_scale < float('inf'):
        raise ValueError('positive finite initial scale required')
    if n < minimum_samples:
        return dict(scale=float(initial_scale), fitted=False, sample_count=n, reason='insufficient_correspondences')
    camera_ray = torch.einsum('nij,nj->ni', poses[:, :3, :3], ray)
    translation = torch.einsum('nij,j->ni', poses[:, :3, :3], origin) + poses[:, :3, 3]
    parallax = torch.linalg.cross(camera_ray, translation, dim=1).norm(dim=1)
    if not bool((parallax > 1e-8 * camera_ray.norm(dim=1).clamp_min(1e-8)).any()):
        return dict(scale=float(initial_scale), fitted=False, sample_count=n, reason='unobservable_scale_without_parallax')
    log_ratio = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([log_ratio], lr=.02)
    def loss_at(value):
        xyz = ray * (initial_scale * value.exp()) + origin
        camera = torch.einsum('nij,nj->ni', poses[:, :3, :3], xyz) + poses[:, :3, 3]
        projected = torch.einsum('nij,nj->ni', intrinsics, camera)
        uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-6)
        residual = (uv - pixels).norm(dim=1)
        # Pixel Huber loss plus cheirality penalty. No dropping behind-camera rows.
        loss = torch.where(residual < 1., .5 * residual.square(), residual - .5).mean()
        return loss + 1000 * (-camera[:, 2]).clamp_min(0).mean()
    initial_loss = float(loss_at(log_ratio).detach())
    best_loss, best_ratio = initial_loss, 0.
    for _ in range(steps):
        optimizer.zero_grad(); loss = loss_at(log_ratio); loss.backward(); optimizer.step()
        with torch.no_grad():
            log_ratio.clamp_(-.69314718056, .69314718056)
            value = float(loss_at(log_ratio))
            if value < best_loss: best_loss, best_ratio = value, float(log_ratio)
    return dict(scale=float(initial_scale * torch.tensor(best_ratio).exp()), fitted=True,
                sample_count=n, initial_training_loss=initial_loss, training_loss=best_loss,
                reason=None, requires_independent_validation=True)
