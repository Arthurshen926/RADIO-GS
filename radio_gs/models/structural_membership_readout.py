"""Query-independent weights for extent inference from automatic native seeds.

Native scores remain an immutable input, not a trainable semantic embedding.
This is an experimental membership readout, not scale-conditioned grouping.
"""
import torch
from torch import nn
from torch.nn import functional as F

class StructuralMembershipReadout(nn.Module):
    def __init__(self, structural_dim, use_structure=True, seeds=4):
        super().__init__()
        self.use_structure = use_structure
        self.seeds = seeds
        self.network = nn.Sequential(nn.Linear(2*structural_dim+4, 64), nn.ReLU(), nn.Linear(64, 1))
        self.register_buffer('xyz', None, persistent=False)

    def bind_geometry(self, xyz, observed):
        if xyz.shape != (len(observed), 3) or not torch.isfinite(xyz).all() or not observed.any():
            raise ValueError('finite bound geometry and observed elements required')
        xyz = xyz.detach()
        lower = xyz[observed].amin(0)
        self.xyz = (xyz-lower)/(xyz[observed].amax(0)-lower).norm().clamp_min(1e-6)

    def seed_indices(self, scores, observed):
        if self.xyz is None:
            raise ValueError('membership requires bound frozen geometry')
        remaining = scores.detach().masked_fill(~observed, -torch.inf).clone()
        selected = []
        for _ in range(self.seeds):
            index = remaining.argmax()
            if not torch.isfinite(remaining[index]):
                break
            selected.append(index)
            remaining[(self.xyz-self.xyz[index]).norm(dim=-1)<.02] = -torch.inf
        return torch.stack(selected)

    def forward(self, scores, structure, observed, modality, rows=None):
        structure = F.normalize(structure, dim=-1)
        ids = self.seed_indices(scores, observed)
        base = scores if rows is None else scores[rows]
        z = structure if rows is None else structure[rows]
        xyz = self.xyz if rows is None else self.xyz[rows]
        outputs = []
        # Looping seeds avoids allocating N*K*feature_dim simultaneously.
        for seed in ids:
            difference = (z-structure[seed]).abs()
            product = z*structure[seed]
            distance = (xyz-self.xyz[seed]).norm(dim=-1)
            if not self.use_structure:
                difference = torch.zeros_like(difference)
                product = torch.zeros_like(product)
                distance = torch.zeros_like(distance)
            features = torch.cat([difference, product, base[:,None],
                (base-scores[seed])[:,None], distance[:,None],
                torch.full_like(base[:,None], float(modality=='image'))], dim=-1)
            outputs.append(self.network(features)[:,0])
        return torch.stack(outputs).amax(0)
