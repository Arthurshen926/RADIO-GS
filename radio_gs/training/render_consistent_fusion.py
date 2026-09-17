"""Regularized inversion of a fixed source feature rendering operator."""
import torch


@torch.no_grad()
def fit_render_consistent_features(operator, targets, initial, *, prior_weight=.1,
                                   iterations=80, tolerance=1e-5):
    """Fit A x to source pixels, with a diagonal prior around initial features.

    Each feature coordinate solves the same positive definite normal system.
    Unobserved columns stay at their initial value. No query labels are needed.
    The caller owns visibility, target validity, and descriptor-space contracts.
    """
    if operator.layout != torch.sparse_coo:
        raise ValueError('expected sparse COO render operator')
    operator = operator.coalesce()
    if operator.ndim != 2 or targets.ndim != 2 or initial.ndim != 2 or operator.shape != (len(targets), len(initial)) or targets.shape[1] != initial.shape[1]:
        raise ValueError('render feature axes differ')
    if not 0 < prior_weight or iterations < 1 or not 0 < tolerance:
        raise ValueError('invalid solver controls')
    if not all(torch.isfinite(t).all() for t in [operator.values(), targets, initial]) or (operator.values() < 0).any():
        raise ValueError('invalid render evidence')
    at = operator.transpose(0, 1).coalesce()
    diagonal = initial.new_zeros(len(initial)).scatter_add_(0, operator.indices()[1], operator.values().square())
    regularizer = prior_weight * diagonal
    preconditioner = (diagonal + regularizer).clamp_min(1e-12)[:, None]
    def apply(x):
        return torch.sparse.mm(at, torch.sparse.mm(operator, x)) + regularizer[:, None] * x
    rhs = torch.sparse.mm(at, targets - torch.sparse.mm(operator, initial))
    delta = torch.zeros_like(initial)
    residual = rhs.clone()
    z = residual / preconditioner
    direction = z.clone()
    rz = (residual * z).sum(0)
    initial_norm = rhs.norm(dim=0)
    history = []
    for step in range(iterations):
        active = residual.norm(dim=0) > tolerance * initial_norm.clamp_min(1e-12)
        if not active.any():
            break
        product = apply(direction)
        denom = (direction * product).sum(0)
        alpha = torch.where(active, rz / denom.clamp_min(1e-30), 0)
        delta += direction * alpha
        residual -= product * alpha
        new_z = residual / preconditioner
        new_rz = (residual * new_z).sum(0)
        beta = torch.where(active, new_rz / rz.clamp_min(1e-30), 0)
        direction = new_z + direction * beta
        rz = new_rz
        if step % 10 == 0:
            history.append(float((residual.norm(dim=0) / initial_norm.clamp_min(1e-12)).max()))
    result = initial + delta
    result[diagonal == 0] = initial[diagonal == 0]
    before = float((torch.sparse.mm(operator, initial) - targets).square().mean())
    after = float((torch.sparse.mm(operator, result) - targets).square().mean())
    return result, dict(source_mse_before=before, source_mse_after=after,
                        relative_residual_max=float((residual.norm(dim=0) / initial_norm.clamp_min(1e-12)).max()),
                        iterations=step + 1, residual_history=history,
                        updated_elements=int((diagonal > 0).sum()))
