"""Reconstruct responses from already-admitted observations on frozen geometry.

This module does not infer identity or negative permissions. The diagonal
consensus and rendered objectives receive exactly the same targets and weights.
Optional element priors preserve the two-teacher category fusion contract.
"""
import torch


class AdmittedResponseSystem:
    def __init__(self, initial, observations, *, prior=None, strength=None):
        self.initial = initial
        self.prior = initial if prior is None else prior
        self.strength = strength
        if strength is not None:
            if strength.shape != initial.shape or self.prior.shape != initial.shape:
                raise ValueError('element prior and strength must match response shape')
            if not bool(torch.isfinite(strength).all()) or bool((strength < 0).any()):
                raise ValueError('prior strength must be finite and nonnegative')
        self.terms = []
        self.mass = torch.zeros_like(initial)
        for operator, target, known, weight in observations:
            adjoint = operator.T.coalesce()
            reliability = known.to(dtype=initial.dtype) * weight
            self.terms.append((operator, adjoint, target, reliability))
            self.mass += torch.sparse.mm(adjoint, reliability)

    @torch.no_grad()
    def diagonal(self):
        numerator = torch.zeros_like(self.initial)
        for _, adjoint, target, reliability in self.terms:
            numerator += torch.sparse.mm(adjoint, reliability * target)
        denominator = self.mass
        if self.strength is not None:
            numerator = numerator + self.strength * self.prior
            denominator = denominator + self.strength
        # A prior does not turn an unsupported source location into a new output.
        return torch.where(self.mass > 1e-8,
                           numerator / denominator.clamp_min(1e-8), self.initial)

    @torch.no_grad()
    def rendered(self, steps=120, residual_clip=.25, progress=None):
        if steps < 0 or (residual_clip is not None and residual_clip <= 0):
            raise ValueError('invalid reconstruction iteration parameters')
        value = self.initial.clone()
        denominator = self.mass if self.strength is None else self.mass + self.strength
        for iteration in range(steps):
            gradient = torch.zeros_like(value)
            for operator, adjoint, target, reliability in self.terms:
                residual = target - torch.sparse.mm(operator, value)
                if residual_clip is not None:
                    residual = residual.clamp(-residual_clip, residual_clip)
                gradient += torch.sparse.mm(adjoint, residual * reliability)
            if self.strength is not None:
                gradient += self.strength * (self.prior - value)
                gradient = torch.where(self.mass > 1e-8, gradient, 0.)
            value.add_(.95 * gradient / denominator.clamp_min(1e-8)).clamp_(0, 1)
            if progress is not None and (iteration + 1) % 30 == 0:
                progress(iteration + 1)
        return value

    @torch.no_grad()
    def objective(self, value, residual_clip=None):
        """Unnormalized objective on the same admitted source evidence, not GT."""
        total = torch.zeros((), device=value.device, dtype=torch.float64)
        for operator, _, target, reliability in self.terms:
            error = (target - torch.sparse.mm(operator, value)).abs()
            if residual_clip is None:
                loss = .5 * error.square()
            else:
                loss = torch.where(error <= residual_clip, .5 * error.square(),
                                   residual_clip * (error - .5 * residual_clip))
            total += (loss * reliability).sum(dtype=torch.float64)
        if self.strength is not None:
            total += (.5 * self.strength * (value - self.prior).square()
                      * (self.mass > 1e-8)).sum(dtype=torch.float64)
        return float(total)


@torch.no_grad()
def solve_rendered_evidence(initial, observations, steps=120):
    """Compatibility entry: original Huber update, exact order and defaults."""
    system = AdmittedResponseSystem(initial, observations)
    return system.rendered(steps=steps), system.mass
