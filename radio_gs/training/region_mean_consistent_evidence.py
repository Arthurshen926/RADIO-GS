"""Query-free dense evidence with regularized region-mean consistency.

Global crop descriptors are region evidence, not uniform pixel labels. This
bounded source-only candidate corrects equal overlap painting to fit region
means while preserving a mean-squared proximity penalty to the original map.
It does not certify physical identity or expose extra deployment features.
"""
import torch


def region_mean_consistent_evidence(masks, descriptors):
    """Return corrected P x D evidence and the source objective diagnostics.

    Minimize mean_r ||M X-D||^2 + mean_p ||X-X0||^2 over covered pixels.
    M is row-normalized membership; X0 is the existing equal-overlap mean.
    Both losses have coefficient one; hence lambda=R/Pcovered. There is no
    query-conditioned choice, feature normalization, or tunable scalar here.
    The nonnegative proximity term is bounded by the initial region-fit loss.
    """
    if masks.ndim != 2 or descriptors.ndim != 2 or len(masks) != len(descriptors):
        raise ValueError('Expected aligned R x P masks and R x D descriptors')
    if len(masks) == 0 or masks.shape[1] == 0 or descriptors.shape[1] == 0:
        raise ValueError('Nonempty evidence required')
    if not torch.isfinite(descriptors).all() or not torch.isfinite(masks).all():
        raise ValueError('Finite source evidence required')
    if not ((masks == 0) | (masks == 1)).all():
        raise ValueError('Binary source membership required')
    binary = masks.to(device=descriptors.device, dtype=torch.float64)
    teacher = descriptors.to(torch.float64)
    area = binary.sum(1)
    if (area == 0).any():
        raise ValueError('Empty source masks have no region-mean target')
    counts = binary.sum(0)
    covered = counts > 0
    partition = binary[:,covered]/counts[covered][None]
    initial = partition.T @ teacher
    mean_operator = binary[:,covered]/area[:,None]
    discrepancy = teacher-mean_operator @ initial
    regularizer = len(masks)/int(covered.sum())
    system = mean_operator @ mean_operator.T
    system.diagonal().add_(regularizer)
    coefficients = torch.linalg.solve(system,discrepancy)
    correction = mean_operator.T @ coefficients
    fitted = initial+correction
    baseline = discrepancy.square().mean()
    fit_loss = (mean_operator @ fitted-teacher).square().mean()
    movement = correction.square().mean()
    if fit_loss+movement > baseline+max(1e-12,float(baseline)*1e-9):
        raise RuntimeError('Region mean correction violated its convex objective')
    dense = descriptors.new_zeros((masks.shape[1],descriptors.shape[1]))
    dense[covered] = fitted.to(descriptors.dtype)
    return dense, dict(regions=len(masks),covered_pixels=int(covered.sum()),
        regularizer=float(regularizer),initial_region_mse=float(baseline),
        corrected_region_mse=float(fit_loss),mean_pixel_change=float(movement),
        final_objective=float(fit_loss+movement),source_only=True,
        regional_teacher_means_not_pixel_truth=True)
