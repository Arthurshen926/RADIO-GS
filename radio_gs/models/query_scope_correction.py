"""Actual-query-conditioned, element-wise correction toward the native answer.

This bounded experiment can repair takeover damage where the native answer has
information. It cannot invent an extent absent from both parent and native.
No element state or native/query encoder is learned by this readout.
"""
import torch
from torch import nn
from torch.nn import functional as F


class QueryScopeCorrection(nn.Module):
    def __init__(self, query_dim, latent_dim, hidden_dim=32, mode='clamped_v1'):
        super().__init__()
        if mode not in ('clamped_v1', 'sigmoid_v1'):
            raise ValueError('Unknown scope correction parameterization')
        self.mode = mode
        self.latent_dim = latent_dim
        self.query_coefficients = nn.Sequential(
            nn.Linear(query_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, latent_dim + 3))
        nn.init.zeros_(self.query_coefficients[-1].weight)
        nn.init.zeros_(self.query_coefficients[-1].bias)
        if mode == 'sigmoid_v1':
            self.register_buffer('enabled', torch.tensor(False))
            with torch.no_grad():
                self.query_coefficients[-1].bias[-1] = -4.

    def forward(self, parent, native, latent, query, active):
        if parent.shape != native.shape or parent.shape != (len(latent), len(query)):
            raise ValueError('Aligned element/query logits required')
        if active.shape != (len(query),) or active.dtype != torch.bool:
            raise ValueError('One active-scope flag per query required')
        if self.mode == 'sigmoid_v1' and not self.enabled:
            return parent
        coeff = self.query_coefficients(F.normalize(query, dim=-1))
        parent_probability, native_probability = parent.sigmoid(), native.sigmoid()
        weight = F.normalize(latent, dim=-1) @ coeff[:, :self.latent_dim].T
        weight = (weight + parent_probability * coeff[:, -3]
                  + native_probability * coeff[:, -2] + coeff[:, -1])
        weight = weight.sigmoid() if self.mode == 'sigmoid_v1' else weight.clamp(0, 1)
        probability = parent_probability + weight * (native_probability - parent_probability)
        corrected = torch.logit(probability.clamp(1e-6, 1-1e-6))
        # Preserve exact zero-correction numerics while retaining the derivative
        # of the probability interpolation at its zero-initialized boundary.
        exact_parent = parent + (corrected - corrected.detach())
        value = torch.where(weight == 0, exact_parent, corrected)
        return torch.where(active[None], value, parent)
