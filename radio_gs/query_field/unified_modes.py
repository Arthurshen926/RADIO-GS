"""Transient native modes decoded from ONE Gaussian latent.

The scene stores neither teacher descriptors nor per-Gaussian mode codes.
Complete-menu competition can precede integration, as in SemanticMeasure.
Other input paths deliberately retain the v1 member model during this control.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .unified_compact import GEOMETRY, UnifiedCompactField

ARCHITECTURE = "unified_compact_modes_v2"


class UnifiedModesField(UnifiedCompactField):
    def __init__(self, *args, mode_count=4, integration="reliable", temperature=100., **kwargs):
        super().__init__(*args, **kwargs)
        if type(mode_count) is not int or mode_count < 1:
            raise ValueError("invalid number of decoded modes")
        if integration not in {"mean", "measure", "reliable"}:
            raise ValueError("invalid integration order")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("invalid competition temperature")
        self.mode_count, self.integration, self.temperature = mode_count, integration, float(temperature)
        self.mode_heads = nn.ModuleList([nn.Linear(self.latent.shape[1], self.semantic.native_dim)
                                        for _ in range(mode_count)])
        self.mode_mass = nn.Linear(self.latent.shape[1], mode_count)

    def decoded_modes(self, rows):
        code = self.latent[rows].float()
        return (F.normalize(torch.stack([head(code) for head in self.mode_heads], dim=1), dim=-1),
                self.mode_mass(code).softmax(-1))

    def category_probabilities(self, queries, rows=None, *, chunk_size=1024):
        if (queries.ndim != 2 or len(queries) < 2 or queries.shape[1] != self.semantic.native_dim
                or not bool(torch.isfinite(queries).all()) or chunk_size < 1):
            raise ValueError("finite complete native query menu required")
        if rows is None:
            return torch.cat([self.category_probabilities(queries, slice(i, i + chunk_size))
                              for i in range(0, len(self.latent), chunk_size)])
        modes, mass = self.decoded_modes(rows)
        identity = modes @ F.normalize(queries.to(modes), dim=-1).T
        if self.integration == "mean":
            return (self.temperature * (identity * mass[..., None]).sum(1)).softmax(-1)
        response = (self.temperature * identity).softmax(-1)
        ordinary = (response * mass[..., None]).sum(1)
        if self.integration == "measure":
            return ordinary
        entropy = -(response * response.clamp_min(torch.finfo(response.dtype).tiny).log()).sum(-1)
        reliability = (1 - entropy / math.log(len(queries))).clamp(0, 1)
        weighted = mass * reliability
        total = weighted.sum(1, keepdim=True)
        numerator = (response * weighted[..., None]).sum(1)
        return torch.where(total > 0, numerator / total.clamp_min(torch.finfo(total.dtype).tiny), ordinary)

    def category_partition(self, queries):
        probabilities = self.category_probabilities(queries)
        return probabilities, probabilities.argmax(1).masked_fill(~self.observed, -1)

    @torch.no_grad()
    def query(self, packet):
        result = super().query(packet)
        result.diagnostics.update(architecture=ARCHITECTURE, category_integration=self.integration,
                                  other_modalities_use_v1_readout=True)
        return result

    def export_spec(self):
        spec = super().export_spec()
        spec.update(architecture=ARCHITECTURE, mode_count=self.mode_count,
                    integration=self.integration, temperature=self.temperature)
        return spec


def from_state(state, spec, device="cpu"):
    arguments = {k: v for k, v in spec.items() if k not in {"architecture", "latent_dim"}}
    if state["latent"].shape[1] != spec["latent_dim"]:
        raise ValueError("latent dimension differs from specification")
    field = UnifiedModesField({k: state[k] for k in GEOMETRY}, state["observed"], state["latent"], **arguments)
    field.load_state_dict(state, strict=True)
    return field.to(device).eval()
