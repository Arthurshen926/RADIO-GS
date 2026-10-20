"""Test observation support agreement with a frozen complete-extent provider.

The guide is fallible and cannot supply missing negative evidence. It weights
whole observations, so compatible partial evidence may extend beyond the guide.
A matched control spends the same positive/negative correction mass uniformly.
"""
import torch
from .response_refinement import _validate_prior


class SupportAgreementRefiner:
    def __init__(self, prior, guide):
        _validate_prior(prior)
        if guide.shape != prior.shape or guide.dtype != torch.bool or guide.device != prior.device:
            raise ValueError('guide must be a boolean element-by-query frozen extent')
        self.prior = prior
        self.guide = guide.to(prior.dtype)
        self.capacity = torch.zeros_like(prior)
        self.raw_positive = torch.zeros_like(prior)
        self.raw_negative = torch.zeros_like(prior)
        self.scoped_positive = torch.zeros_like(prior)
        self.scoped_negative = torch.zeros_like(prior)
        self.observations = 0

    @torch.no_grad()
    def add(self, item):
        positive, negative, confidence = item.weights(self.prior)
        if bool((confidence > 1).any()):
            raise ValueError('bounded evidence confidence required')
        domain = item.normalization_domain
        if domain is None or domain.dtype != torch.bool or domain.device != self.prior.device:
            raise ValueError('explicit boolean normalization domain required')
        domain = torch.broadcast_to(domain, item.target.shape)
        if bool(((positive | negative) & ~domain).any()):
            raise ValueError('permission outside normalization domain')
        op = item.operator
        projection = torch.sparse.mm(op, self.guide).clamp(0, 1)
        # All permitted foreground contributes to scope agreement, including
        # observations already consistent with the prior response.
        positive_agreement = (positive * projection).sum(0) / positive.sum(0).clamp_min(1)
        negative_agreement = (negative * (1-projection)).sum(0) / negative.sum(0).clamp_min(1)
        residual = item.target - torch.sparse.mm(op, self.prior)
        plus = positive * confidence * residual.clamp_min(0)
        minus = negative * confidence * (-residual).clamp_min(0)
        back_plus = torch.sparse.mm(op.T, plus)
        back_minus = torch.sparse.mm(op.T, minus)
        self.raw_positive += back_plus
        self.raw_negative += back_minus
        self.scoped_positive += back_plus * positive_agreement
        self.scoped_negative += back_minus * negative_agreement
        available = domain & torch.broadcast_to(item.query_compatible, item.target.shape)
        self.capacity += torch.sparse.mm(op.T, available.to(self.prior.dtype))
        self.observations += 1

    @torch.no_grad()
    def finish(self):
        denominator = self.capacity.clamp_min(1e-8)
        raw_plus, raw_minus = self.raw_positive/denominator, self.raw_negative/denominator
        plus, minus = self.scoped_positive/denominator, self.scoped_negative/denominator
        ratio_plus = plus.double().sum(0) / raw_plus.double().sum(0).clamp_min(1e-30)
        ratio_minus = minus.double().sum(0) / raw_minus.double().sum(0).clamp_min(1e-30)
        uniform_plus = raw_plus * ratio_plus.to(raw_plus.dtype)
        uniform_minus = raw_minus * ratio_minus.to(raw_minus.dtype)
        outputs = {}
        for name, dp, dn in [('scope',plus,minus),('matched',uniform_plus,uniform_minus)]:
            value = (self.prior + dp - dn).clamp(0,1)
            outputs[name] = (value,dict(observations=self.observations,
                updated_entries=int((value!=self.prior).sum()),
                positive_step_sum=dp.double().sum(0).cpu().tolist(),
                negative_step_sum=dn.double().sum(0).cpu().tolist(),
                positive_retention=ratio_plus.cpu().tolist(),negative_retention=ratio_minus.cpu().tolist(),
                guide_used_as_ground_truth=False,unknown_gradient_zero=True,
                matched_control='per-query pre-clipping positive/negative L1 step mass',
                no_accuracy_guarantee=True))
        return outputs
