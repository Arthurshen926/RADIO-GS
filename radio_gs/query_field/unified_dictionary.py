"""A compact native basis shared by all rows, addressed by the single latent.

The fixed decoder basis constrains every query direction to native teacher
coordinates. Only transient mixture masses are decoded per Gaussian; no
per-row teacher features, distributions, indices or source proposals are saved.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .unified_compact import GEOMETRY, UnifiedCompactField

ARCHITECTURE = "unified_compact_dictionary_v3"


class UnifiedDictionaryField(UnifiedCompactField):
    def __init__(self, *args, semantic_atoms, integration="reliable", temperature=100., **kwargs):
        super().__init__(*args, **kwargs)
        if (semantic_atoms.ndim != 2 or len(semantic_atoms) < 2
                or semantic_atoms.shape[1] != self.semantic.native_dim
                or not bool(torch.isfinite(semantic_atoms).all())):
            raise ValueError("invalid shared native decoder basis")
        if integration not in {"mean", "measure", "reliable"} or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("invalid menu integration")
        if not torch.allclose(semantic_atoms.norm(dim=-1), torch.ones_like(semantic_atoms[:, 0]), atol=1e-5):
            raise ValueError("decoder basis must already have unit native directions")
        self.register_buffer("semantic_atoms", semantic_atoms.detach().clone())
        self.mode_mass = nn.Linear(self.latent.shape[1], len(semantic_atoms))
        self.integration, self.temperature = integration, float(temperature)

    def mode_probabilities(self, rows):
        return self.mode_mass(self.latent[rows].float()).softmax(-1)

    def category_probabilities(self, queries, rows=None, *, chunk_size=4096):
        if (queries.ndim != 2 or len(queries) < 2 or queries.shape[1] != self.semantic.native_dim
                or not bool(torch.isfinite(queries).all()) or chunk_size < 1):
            raise ValueError("finite complete native query menu required")
        if rows is None:
            return torch.cat([self.category_probabilities(queries, slice(i, i + chunk_size))
                              for i in range(0, len(self.latent), chunk_size)])
        return integrate(self.semantic_atoms, self.mode_probabilities(rows), queries,
                         self.integration, self.temperature)

    def category_partition(self, queries):
        p = self.category_probabilities(queries)
        return p, p.argmax(-1).masked_fill(~self.observed, -1)

    @torch.no_grad()
    def query(self, packet):
        result = super().query(packet)
        result.diagnostics.update(architecture=ARCHITECTURE, category_integration=self.integration,
                                  other_modalities_use_v1_readout=True)
        return result

    def export_spec(self):
        spec = super().export_spec()
        spec.update(architecture=ARCHITECTURE, atom_count=len(self.semantic_atoms),
                    integration=self.integration, temperature=self.temperature)
        return spec


def integrate(atoms, mass, queries, rule="reliable", temperature=100.):
    identity = atoms @ F.normalize(queries.to(atoms), dim=-1).T
    if rule == "mean":
        return (temperature * (mass @ identity)).softmax(-1)
    response = (temperature * identity).softmax(-1)
    ordinary = mass @ response
    if rule == "measure":
        return ordinary
    if rule != "reliable":
        raise ValueError("unknown integration rule")
    entropy = -(response * response.clamp_min(torch.finfo(response.dtype).tiny).log()).sum(-1)
    reliability = (1 - entropy / math.log(len(queries))).clamp(0, 1)
    total = mass @ reliability[:, None]
    result = (mass @ (response * reliability[:, None])) / total.clamp_min(torch.finfo(total.dtype).tiny)
    return torch.where(total > 0, result, ordinary)


@torch.no_grad()
def compress_teacher(teacher, count, seed):
    """Construction-only spherical codebook and source-measure projection.

    The N x K projected mass is an ephemeral training target, never an export.
    K is declared before benchmark predictions; no class labels enter clustering.
    """
    from sklearn.cluster import KMeans
    values = teacher.atoms.cpu().numpy()
    if count > len(values):
        raise ValueError("basis exceeds source atom count")
    cluster = KMeans(n_clusters=count, n_init=1, max_iter=100, random_state=seed).fit(values)
    atoms = F.normalize(torch.from_numpy(cluster.cluster_centers_).to(teacher.device), dim=-1)
    return atoms, project_teacher(teacher, atoms)


@torch.no_grad()
def project_teacher(teacher, atoms):
    """Project construction-only source mass onto an already frozen basis."""
    count = len(atoms)
    assignment = (teacher.atoms @ atoms.T).argmax(-1).cpu().numpy()
    from scipy import sparse
    mapping = sparse.csr_matrix((np.ones(len(teacher.atoms), dtype=np.float32),
                                 (np.arange(len(teacher.atoms)), assignment)), shape=(len(teacher.atoms), count))
    mass = torch.from_numpy((teacher.transfer @ mapping).toarray()).float()
    # Unsupported source rows are explicitly a single nearest fallback atom.
    unsupported = torch.where(~teacher.supported)[0]
    for start in range(0, len(unsupported), 4096):
        rows = unsupported[start:start + 4096]
        ids = (teacher.fallback[rows].to(atoms) @ atoms.T).argmax(-1).cpu()
        mass[rows, ids] = 1.
    if not torch.allclose(mass.sum(-1), torch.ones(len(mass)), atol=2e-5, rtol=0):
        raise ValueError("compressed source measure lost mass")
    return mass


@torch.no_grad()
def initialize(base, config, integration, atoms, mass):
    torch.manual_seed(config["seed"])
    spec = {k: v for k, v in base.export_spec().items() if k not in {"architecture", "latent_dim"}}
    field = UnifiedDictionaryField({k: getattr(base, k) for k in GEOMETRY}, base.observed, base.latent,
                **spec, semantic_atoms=atoms, integration=integration,
                temperature=config["temperature"]).to(base.latent.device)
    state = field.state_dict()
    state.update(base.state_dict())
    field.load_state_dict(state, strict=True)
    generator = torch.Generator().manual_seed(config["seed"])
    available = torch.where(base.observed.cpu())[0]
    rows = available[torch.randperm(len(available), generator=generator)[:config["basis_initial_rows"]]]
    x = base.latent[rows.to(base.latent.device)]
    x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
    y = mass[rows].to(x).clamp_min(1e-4).log()
    y -= y.mean(-1, keepdim=True)
    penalty = torch.eye(x.shape[1], device=x.device) * config["basis_initial_ridge"]
    coefficients = torch.linalg.solve(x.T @ x + penalty, x.T @ y)
    field.mode_mass.weight.copy_(coefficients[:-1].T)
    field.mode_mass.bias.copy_(coefficients[-1])
    return field


def from_state(state, spec, device="cpu"):
    arguments = {k: v for k, v in spec.items() if k not in {"architecture", "latent_dim", "atom_count"}}
    if state["latent"].shape[1] != spec["latent_dim"] or len(state["semantic_atoms"]) != spec["atom_count"]:
        raise ValueError("decoder dimensions differ from specification")
    field = UnifiedDictionaryField({k: state[k] for k in GEOMETRY}, state["observed"], state["latent"],
                                    semantic_atoms=state["semantic_atoms"], **arguments)
    field.load_state_dict(state, strict=True)
    return field.to(device).eval()
