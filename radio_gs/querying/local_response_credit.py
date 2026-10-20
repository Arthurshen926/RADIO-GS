"""Attribute source Huber-loss changes to proposed 3D response edits.

The path integral is exact for a linear observation operator. A positive local
credit is evidence for an edit, not a guarantee about target accuracy. No target
labels, geometry fitting or feature representation learning occur here.
"""
import torch


def huber(error, clip):
    absolute = error.abs()
    return torch.where(absolute <= clip, .5 * error.square(),
                       clip * (absolute - .5 * clip))


class LocalResponseCredit:
    def __init__(self, prior, candidate, clip=.25):
        if (prior.shape != candidate.shape or prior.ndim != 2 or clip <= 0
                or not bool(torch.isfinite(prior).all())
                or not bool(torch.isfinite(candidate).all())
                or bool(((prior < 0) | (prior > 1) | (candidate < 0) | (candidate > 1)).any())):
            raise ValueError('bounded finite element-by-query responses required')
        self.prior, self.delta, self.clip = prior, candidate-prior, clip
        self.net = torch.zeros_like(prior, dtype=torch.float64)
        self.gross = torch.zeros_like(self.net)
        self.source_gain = 0.
        self.observations = 0

    @torch.no_grad()
    def add(self, operator, target, weight):
        if (not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(weight).all())
                or bool(((target < 0) | (target > 1) | (weight < 0)).any())):
            raise ValueError('bounded source targets and nonnegative weights required')
        if not bool(torch.isfinite(operator.values()).all()) or bool((operator.values() < 0).any()):
            raise ValueError('finite nonnegative observation operator required')
        # Double precision stabilizes tiny background-score changes. The loss
        # secant integrates the residual derivative along the entire edit path.
        op = operator.to(dtype=torch.float64)
        old = torch.sparse.mm(op, self.prior.double())
        delta = torch.sparse.mm(op, self.delta.double())
        error = target.double()-old
        gain = huber(error, self.clip)-huber(error-delta, self.clip)
        nonzero = delta.abs() > 1e-12
        secant = torch.where(nonzero, gain / torch.where(nonzero, delta, 1.),
                             error.clamp(-self.clip, self.clip))
        adjoint = op.transpose(0, 1).to_sparse_csr()
        weighted = weight.double()*secant
        self.net += self.delta.double()*torch.sparse.mm(adjoint, weighted)
        self.gross += self.delta.double().abs()*torch.sparse.mm(adjoint, weighted.abs())
        self.source_gain += float((weight.double()*gain).sum())
        self.observations += 1

    @torch.no_grad()
    def gate(self, final_candidate):
        if final_candidate.shape != self.prior.shape or not bool(torch.isfinite(final_candidate).all()):
            raise ValueError('final candidate must match finite response shape')
        final_delta = final_candidate-self.prior
        # A split-field permission does not authorize an opposite-direction edit.
        same_direction = final_delta*self.delta > 0
        gate = torch.where((self.gross > 1e-12) & same_direction,
                           self.net.clamp_min(0)/self.gross.clamp_min(1e-12), 0.).clamp(0, 1)
        return gate.to(self.prior.dtype)

    def report(self):
        return dict(observations=self.observations, source_huber_gain=self.source_gain,
                    allocated_gain=float(self.net.sum()),
                    conservation_error=float(self.net.sum())-self.source_gain,
                    gross_credit=float(self.gross.sum()),
                    positive_credit_elements=int((self.net > 0).sum()),
                    negative_credit_elements=int((self.net < 0).sum()),
                    unobserved_credit_elements=int((self.gross <= 1e-12).sum()))


@torch.no_grad()
def matched_uniform_edit(prior, candidate, gate):
    """Same positive/negative element-space L1 edit mass, separately per query."""
    if (prior.shape != candidate.shape or gate.shape != prior.shape
            or not bool(torch.isfinite(gate).all()) or bool(((gate < 0) | (gate > 1)).any())):
        raise ValueError('matching response/gate shapes and bounded gates required')
    delta = candidate-prior
    positive, negative = delta.clamp_min(0), (-delta).clamp_min(0)
    rp = (positive.double()*gate).sum(0)/positive.double().sum(0).clamp_min(1e-30)
    rn = (negative.double()*gate).sum(0)/negative.double().sum(0).clamp_min(1e-30)
    value = prior+positive*rp.to(prior.dtype)-negative*rn.to(prior.dtype)
    return value, dict(positive_retention=rp.tolist(), negative_retention=rn.tolist(),
                       positive_edit_mass=(positive.double()*gate).sum(0).tolist(),
                       negative_edit_mass=(negative.double()*gate).sum(0).tolist())
