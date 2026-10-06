"""Source-depth distribution loss for isolated opacity preprocessing."""
import torch


def depth_distribution_loss(moments, alpha, target, valid, *, absolute_tolerance=.03,
                            relative_tolerance=.02, coverage_weight=.1):
    """Penalize E[(Z-target)^2], not merely (E[Z]-target)^2.

    Inputs use metres. Moments contain alpha-weighted Z and Z^2 sums.
    Invalid reference pixels provide no positive/negative observation.
    """
    if moments.shape != (*target.shape, 2) or alpha.shape != target.shape or valid.shape != target.shape:
        raise ValueError('depth moment raster shape differs')
    valid = valid.bool() & torch.isfinite(target) & (target > 0)
    if not valid.any():
        raise ValueError('no valid source depth evidence')
    a = alpha[valid]
    values = moments[valid]/a.clamp_min(1e-6)[:, None]
    z = target[valid]
    expected_error = (values[:, 1]-2*z*values[:, 0]+z.square()).clamp_min(0)
    tolerance = absolute_tolerance+relative_tolerance*z
    # Smooth robust absolute residual of the whole contribution distribution.
    depth = torch.sqrt(expected_error/tolerance.square()+1e-4).mean()
    coverage = torch.relu(.8-a).square().mean()
    return depth+coverage_weight*coverage, dict(depth=depth.detach(), coverage=coverage.detach())
