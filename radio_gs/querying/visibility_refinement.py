"""Keep the same signed residual; normalize by observable assignment capacity.

Unknown observations have zero gradient, but do not disappear from geometric
normalization. A weak or isolated correction no longer normalizes away its own
reliability or coverage. This is a fixed step, not a tuned per-task coefficient.
"""
import torch
from .response_refinement import ResponseRefiner


class VisibilityNormalizedRefiner(ResponseRefiner):
    def __init__(self, prior):
        super().__init__(prior)
        self.capacity = torch.zeros_like(prior)

    @torch.no_grad()
    def add(self, item):
        domain = item.normalization_domain
        if domain is None or domain.dtype != torch.bool or domain.device != self.prior.device:
            raise ValueError('explicit boolean normalization domain is required')
        domain = torch.broadcast_to(domain,item.target.shape)
        positive,negative,confidence = item.weights(self.prior)
        if bool((confidence > 1).any()):
            raise ValueError('visibility normalization requires bounded reliability weights')
        if bool(((positive | negative) & ~domain).any()):
            raise ValueError('evidence outside normalization domain')
        super().add(item)
        available = domain & torch.broadcast_to(item.query_compatible,item.target.shape)
        self.capacity += torch.sparse.mm(item.operator.T,available.to(self.prior.dtype))

    @torch.no_grad()
    def finish(self):
        result = torch.where(self.mass > 1e-8,
            (self.prior+self.numerator/self.capacity.clamp_min(1e-8)).clamp(0,1),self.prior)
        return result,dict(observations=self.observations,positive_entries=self.positive_entries,
            negative_entries=self.negative_entries,updated_entries=int((result!=self.prior).sum()),
            supported_entries=int((self.mass>1e-8).sum()),normalization='observable_assignment_capacity',
            normalization_includes_zero_residual_and_unknown=True,unknown_gradient_zero=True,
            bounded_reliability=True,no_feature_update=True,no_evidence_retains_prior=True,
            accuracy_non_degradation_guaranteed=False)
