"""Source-fitted query-conditional decision boundary; native ranks unchanged."""
import torch
from torch import nn
from torch.nn import functional as F


class NativeQueryThreshold(nn.Module):
    def __init__(self, dimension, lower, upper):
        super().__init__()
        if not -1 <= lower <= upper <= 1:
            raise ValueError('native threshold interval must lie in [-1,1]')
        self.linear = nn.Linear(dimension, 1)
        self.register_buffer('bounds', torch.tensor([lower, upper], dtype=torch.float32))

    def forward(self, query):
        return self.linear(F.normalize(query, dim=-1)).flatten().clamp(*self.bounds)


def fit_threshold_ridge(queries, thresholds, penalty):
    """Dual ridge with unpenalized intercept, avoiding a feature-sized inverse."""
    if len(queries) < 2 or thresholds.shape != (len(queries),) or penalty <= 0:
        raise ValueError('aligned multiple source queries and positive ridge required')
    q = F.normalize(queries.double(), dim=-1)
    target = thresholds.double()
    x = q - q.mean(0)
    weights = x.T @ torch.linalg.solve(x @ x.T + penalty * torch.eye(len(x), device=x.device), target - target.mean())
    intercept = target.mean() - q.mean(0) @ weights
    return weights.float(), intercept.float()
