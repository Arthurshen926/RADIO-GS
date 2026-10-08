"""Scene-specific native/geometry conditioning inside the shared member core.

Only exported Gaussian attributes are read. Source observations train these
parameters offline; neither an observation bank nor a labelled v10 head is used.
"""
import torch
from torch import nn
from torch.nn import functional as F


FEATURES = ('group_cosine', 'semantic_cosine', 'local_cosine', 'log_scaled_distance',
            'normal_alignment', 'log_element_scale', 'log_support_scale_ratio',
            'opacity', 'element_observation_confidence', 'support_observation_confidence')


class NativeGeometryRelation(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(len(FEATURES), 32), nn.SiLU(),
                                    nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features):
        # A correction to the existing logit, not an unconstrained new solver.
        return 6 * torch.tanh(self.layers(features).flatten() / 6)


def enable(field, confidence, seed=20261008):
    if field.relation_decoder != 'none':
        raise ValueError('field already has a conditional relation decoder')
    if field.spatial_metric != 'scale_normalized' or field.support_aggregation != 'normalized_prototype':
        raise ValueError('conditional relation requires the declared metric/prototype core')
    confidence = confidence.to(field.xyz).detach()
    if confidence.shape != field.observed.shape or not bool(torch.isfinite(confidence).all()):
        raise ValueError('invalid per-Gaussian observation confidence')
    if bool(((confidence < 0) | (confidence > 1)).any()):
        raise ValueError('observation confidence must lie in [0,1]')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        field.relation = NativeGeometryRelation().to(field.xyz.device)
    field.relation_confidence = confidence
    field.relation_decoder = 'native_geometry_v1'


def residual(field, support, scale, cosine, rows=None, chunk_size=8192):
    ids = support.indices.to(field.xyz.device)
    weight = support.weights.to(field.xyz); weight = weight / weight.sum()
    # All descriptors and geometry are protected buffers. Only group cosine
    # and the small scene decoder receive gradients from this branch.
    with torch.no_grad():
        prototypes = {name: F.normalize((F.normalize(getattr(field, name).decode(ids), dim=-1)
                                        * weight[:, None]).sum(0), dim=0)
                      for name in ['semantic', 'local']}
        centre = (field.xyz[ids] * weight[:, None]).sum(0)
        normal = F.normalize((field.normals[ids] * weight[:, None]).sum(0), dim=0)
        physical = float(scale) * field.world_units_per_metre
        support_size = (field.scales[ids].clamp_min(1e-8).log().mean(-1) * weight).sum()
        confidence = (field.relation_confidence[ids] * weight).sum()
    indices = torch.arange(len(field.xyz), device=field.xyz.device) if rows is None else rows.to(field.xyz.device)
    values = []
    for start in range(0, len(indices), chunk_size):
        selected = indices[start:start + chunk_size]
        with torch.no_grad():
            native = [(F.normalize(getattr(field, name).decode(selected), dim=-1) * prototypes[name]).sum(-1)
                      for name in ['semantic', 'local']]
            distance = ((field.xyz[selected] - centre) / physical).square().sum(-1)
            sizes = field.scales[selected].clamp_min(1e-8).log().mean(-1)
            others = [*native, torch.log1p(distance),
                (F.normalize(field.normals[selected], dim=-1) * normal).sum(-1).abs(),
                (sizes - torch.tensor(physical, device=sizes.device).log()).clamp(-8, 8),
                (sizes - support_size).abs().clamp_max(8), field.opacities[selected].flatten(),
                field.relation_confidence[selected], confidence.expand(len(selected))]
        features = torch.stack([cosine[start:start + len(selected)], *others], dim=-1)
        values.append(field.relation(features))
    return torch.cat(values) if values else cosine.new_empty(0)
