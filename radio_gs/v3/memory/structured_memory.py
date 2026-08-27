"""Global functional projections over the sole persistent D512 latent."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class StructuredMemoryHeads(nn.Module):
    """Visual, scale-conditioned instance, and boundary views of one latent."""

    def __init__(
        self,
        latent_dim: int = 512,
        visual_dim: int = 256,
        instance_dim: int = 32,
        boundary_dim: int = 16,
        scale_frequencies: int = 4,
    ) -> None:
        super().__init__()
        if min(latent_dim, visual_dim, instance_dim, boundary_dim, scale_frequencies) <= 0:
            raise ValueError("structured memory dimensions must be positive")
        self.visual = nn.Linear(latent_dim, visual_dim, bias=False)
        self.instance = nn.Linear(latent_dim, instance_dim, bias=False)
        self.boundary = nn.Linear(latent_dim, boundary_dim, bias=False)
        self.scale_adapter = nn.Sequential(
            nn.Linear(2 * scale_frequencies, 2 * instance_dim),
            nn.GELU(),
            nn.Linear(2 * instance_dim, 2 * instance_dim),
        )
        self.scale_frequencies = int(scale_frequencies)

    def visual_view(self, latent: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.visual(_validate_latent(latent)), dim=-1, eps=1e-8)

    def instance_view(self, latent: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
        z = _validate_latent(latent)
        scale_tensor = torch.as_tensor(scale, device=z.device, dtype=z.dtype).reshape(-1)
        if scale_tensor.numel() == 1:
            scale_tensor = scale_tensor.expand(z.shape[0])
        if scale_tensor.shape != (z.shape[0],) or not bool(torch.isfinite(scale_tensor).all()):
            raise ValueError("mask scale must be finite and scalar or per Gaussian")
        frequency = torch.arange(1, self.scale_frequencies + 1, device=z.device, dtype=z.dtype)
        phase = math.pi * scale_tensor[:, None].clamp(0, 1) * frequency[None]
        gamma, beta = self.scale_adapter(torch.cat((phase.sin(), phase.cos()), dim=-1)).chunk(2, dim=-1)
        value = self.instance(z) * (1 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
        return F.normalize(value, dim=-1, eps=1e-8)

    def boundary_view(self, latent: torch.Tensor) -> torch.Tensor:
        return self.boundary(_validate_latent(latent))

    def orthogonality_loss(self) -> torch.Tensor:
        visual = F.normalize(self.visual.weight, dim=1)
        instance = F.normalize(self.instance.weight, dim=1)
        boundary = F.normalize(self.boundary.weight, dim=1)
        return (visual @ instance.T).square().mean() + (visual @ boundary.T).square().mean()


@dataclass(frozen=True)
class SharedPrivateLayout:
    """Column ownership inside the one physically persistent D512 table."""

    shared: int = 320
    semantic: int = 128
    instance: int = 48
    boundary: int = 16

    def __post_init__(self) -> None:
        if min(self.shared, self.instance, self.boundary) <= 0 or self.semantic < 0:
            raise ValueError("shared/private dimensions must be positive and semantic nonnegative")
        if self.shared + self.semantic + self.instance + self.boundary != 512:
            raise ValueError("shared-private dimensions must sum to 512")

    @property
    def slices(self) -> dict[str, slice]:
        shared_stop = self.shared
        semantic_stop = shared_stop + self.semantic
        instance_stop = semantic_stop + self.instance
        return {
            "shared": slice(0, shared_stop),
            "semantic": slice(shared_stop, semantic_stop),
            "instance": slice(semantic_stop, instance_stop),
            "boundary": slice(instance_stop, 512),
        }


class StructuredSharedPrivateMemory(nn.Module):
    """One D512 table with shared reads and parameter-level write ownership."""

    deployment_eligible = True
    architecture = "hard_block_shared_private"

    def __init__(
        self,
        initial_memory: torch.Tensor,
        *,
        layout: SharedPrivateLayout = SharedPrivateLayout(),
        bridges_enabled: bool = False,
    ) -> None:
        super().__init__()
        value = _validate_latent(initial_memory).detach().clone()
        self.memory = nn.Parameter(value)
        self.layout = layout
        self.bridges_enabled = bool(bridges_enabled)
        # Communication is global, one-way, zero initialized, and reads only
        # stop-gradient context. It adds no Gaussian-indexed sidecar.
        visual_dim = layout.shared + layout.semantic
        self.visual_to_instance = nn.Linear(visual_dim, layout.instance, bias=False)
        self.context_to_boundary = nn.Linear(
            visual_dim + layout.instance, layout.boundary, bias=False
        )
        nn.init.zeros_(self.visual_to_instance.weight)
        nn.init.zeros_(self.context_to_boundary.weight)
        self.scale_adapter = nn.Linear(2, 2 * layout.instance)

    def block(self, name: str, rows: torch.Tensor | None = None) -> torch.Tensor:
        if name not in self.layout.slices:
            raise ValueError("unknown shared-private block")
        # Slice owned columns first. Advanced row indexing the D512 table and
        # slicing afterwards would materialize an unnecessary hits-by-512
        # tensor (over 1 GiB for dense exact-MPR views).
        value = self.memory[:, self.layout.slices[name]]
        if rows is not None:
            value = value[
                torch.as_tensor(rows, device=self.memory.device, dtype=torch.long)
            ]
        return value

    def visual_view(self, rows: torch.Tensor | None = None) -> torch.Tensor:
        value = torch.cat((self.block("shared", rows), self.block("semantic", rows)), dim=-1)
        return F.normalize(value, dim=-1, eps=1e-8)

    def shared_capability_view(self, value: torch.Tensor) -> torch.Tensor:
        """Map rendered shared coefficients into the registered image space."""

        return value

    def shared_target_view(self, value: torch.Tensor) -> torch.Tensor:
        """Embed a fixed shared teacher in the same registered image space."""

        return value

    def visual_auxiliary_parameters(self) -> tuple[nn.Parameter, ...]:
        return ()

    def instance_auxiliary_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.scale_adapter.parameters())

    def boundary_auxiliary_parameters(self) -> tuple[nn.Parameter, ...]:
        return ()

    def semantic_view(self, rows: torch.Tensor | None = None) -> torch.Tensor:
        if self.layout.semantic == 0:
            return self.block("semantic", rows)
        return F.normalize(self.block("semantic", rows), dim=-1, eps=1e-8)

    def instance_view(
        self, scale: float = 0.5, rows: torch.Tensor | None = None
    ) -> torch.Tensor:
        private = self.block("instance", rows)
        phase = private.new_tensor([float(scale)]).clamp(0, 1) * torch.pi
        gamma, beta = self.scale_adapter(
            torch.cat((phase.sin(), phase.cos()))
        ).chunk(2)
        value = private
        if self.bridges_enabled:
            value = value + self.visual_to_instance(self.visual_view(rows).detach())
        value = value * (1 + 0.1 * gamma.tanh()) + 0.1 * beta
        return F.normalize(value, dim=-1, eps=1e-8)

    def boundary_view(self, rows: torch.Tensor | None = None) -> torch.Tensor:
        value = self.block("boundary", rows)
        if self.bridges_enabled:
            context = torch.cat(
                (self.visual_view(rows), self.block("instance", rows)), dim=-1
            ).detach()
            value = value + self.context_to_boundary(context)
        return value

    def forward(
        self, scale: float = 0.5, rows: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.instance_view(scale, rows)

    @torch.no_grad()
    def deployment_memory(self) -> torch.Tensor:
        return self.memory.detach().clone()


class OrthogonalProductMemory(StructuredSharedPrivateMemory):
    """Learned global orthogonal bases over the same persistent D512 codes.

    A bank of disjoint Givens rotations mixes every private coordinate with a
    distinct shared coordinate.  The transform is exactly orthogonal for every
    angle, starts as the identity, and adds only constant-size global state.
    Private dot-product objectives can use their isometric coefficient views;
    source visual authority alone owns the basis angles.
    """

    architecture = "learned_orthogonal_product"

    def __init__(
        self,
        initial_memory: torch.Tensor,
        *,
        layout: SharedPrivateLayout = SharedPrivateLayout(),
        bridges_enabled: bool = False,
    ) -> None:
        super().__init__(
            initial_memory, layout=layout, bridges_enabled=bridges_enabled
        )
        private = 512 - layout.shared
        if private > layout.shared:
            raise ValueError("orthogonal product requires shared >= private dimensions")
        self.basis_angles = nn.Parameter(torch.zeros(private))
        self.register_buffer(
            "basis_left", torch.arange(private, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "basis_right",
            torch.arange(layout.shared, 512, dtype=torch.long),
            persistent=True,
        )

    def _embed_shared(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.layout.shared:
            raise ValueError("orthogonal shared value has the wrong dimension")
        return F.pad(value, (0, 512 - self.layout.shared))

    def rotate(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != 512:
            raise ValueError("orthogonal product transform requires D512")
        left = value[..., self.basis_left]
        right = value[..., self.basis_right]
        cosine = self.basis_angles.cos()
        sine = self.basis_angles.sin()
        output = value.clone()
        output[..., self.basis_left] = cosine * left - sine * right
        output[..., self.basis_right] = sine * left + cosine * right
        return output

    def shared_capability_view(self, value: torch.Tensor) -> torch.Tensor:
        return self.rotate(self._embed_shared(value))

    def shared_target_view(self, value: torch.Tensor) -> torch.Tensor:
        return self._embed_shared(value)

    def visual_auxiliary_parameters(self) -> tuple[nn.Parameter, ...]:
        return (self.basis_angles,)


class LowRankPrivateBranchMemory(StructuredSharedPrivateMemory):
    """Shared core with zero-output low-rank one-way private branches."""

    architecture = "shared_core_low_rank_private_branches"

    def __init__(
        self,
        initial_memory: torch.Tensor,
        *,
        layout: SharedPrivateLayout = SharedPrivateLayout(),
        instance_rank: int = 8,
        boundary_rank: int = 4,
    ) -> None:
        super().__init__(initial_memory, layout=layout, bridges_enabled=False)
        visual_dim = layout.shared + layout.semantic
        if not 0 < instance_rank <= min(visual_dim, layout.instance):
            raise ValueError("invalid instance private-branch rank")
        if not 0 < boundary_rank <= min(visual_dim + layout.instance, layout.boundary):
            raise ValueError("invalid boundary private-branch rank")
        self.instance_down = nn.Linear(visual_dim, instance_rank, bias=False)
        self.instance_up = nn.Linear(instance_rank, layout.instance, bias=False)
        self.boundary_down = nn.Linear(
            visual_dim + layout.instance, boundary_rank, bias=False
        )
        self.boundary_up = nn.Linear(boundary_rank, layout.boundary, bias=False)
        nn.init.zeros_(self.instance_up.weight)
        nn.init.zeros_(self.boundary_up.weight)

    def _normalized_visual_parts(
        self, rows: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.block("shared", rows).detach()
        semantic = self.block("semantic", rows).detach()
        squared_norm = shared.square().sum(-1, keepdim=True)
        if semantic.shape[-1]:
            squared_norm = squared_norm + semantic.square().sum(-1, keepdim=True)
        inverse_norm = squared_norm.clamp_min(1e-16).rsqrt()
        return shared * inverse_norm, semantic * inverse_norm

    def _instance_branch(self, rows: torch.Tensor | None) -> torch.Tensor:
        shared, semantic = self._normalized_visual_parts(rows)
        split = self.layout.shared
        code = F.linear(shared, self.instance_down.weight[:, :split])
        if semantic.shape[-1]:
            code = code + F.linear(semantic, self.instance_down.weight[:, split:])
        return self.instance_up(code)

    def _boundary_branch(self, rows: torch.Tensor | None) -> torch.Tensor:
        shared, semantic = self._normalized_visual_parts(rows)
        private = self.block("instance", rows).detach()
        shared_stop = self.layout.shared
        semantic_stop = shared_stop + self.layout.semantic
        code = F.linear(shared, self.boundary_down.weight[:, :shared_stop])
        if semantic.shape[-1]:
            code = code + F.linear(
                semantic, self.boundary_down.weight[:, shared_stop:semantic_stop]
            )
        code = code + F.linear(private, self.boundary_down.weight[:, semantic_stop:])
        return self.boundary_up(code)

    def instance_view(
        self, scale: float = 0.5, rows: torch.Tensor | None = None
    ) -> torch.Tensor:
        private = self.block("instance", rows)
        private = private + self._instance_branch(rows)
        phase = private.new_tensor([float(scale)]).clamp(0, 1) * torch.pi
        gamma, beta = self.scale_adapter(
            torch.cat((phase.sin(), phase.cos()))
        ).chunk(2)
        value = private * (1 + 0.1 * gamma.tanh()) + 0.1 * beta
        return F.normalize(value, dim=-1, eps=1e-8)

    def boundary_view(self, rows: torch.Tensor | None = None) -> torch.Tensor:
        branch = self._boundary_branch(rows)
        return self.block("boundary", rows) + branch

    def instance_auxiliary_parameters(self) -> tuple[nn.Parameter, ...]:
        return (
            *self.scale_adapter.parameters(),
            *self.instance_down.parameters(),
            *self.instance_up.parameters(),
        )

    def boundary_auxiliary_parameters(self) -> tuple[nn.Parameter, ...]:
        return (
            *self.boundary_down.parameters(),
            *self.boundary_up.parameters(),
        )


class ExtraInstanceCodeOracle(nn.Module):
    """Temporary Gaussian-indexed D16 upper bound; forbidden at deployment."""

    deployment_eligible = False

    def __init__(self, num_gaussians: int, instance_dim: int = 16, seed: int = 20260826) -> None:
        super().__init__()
        if num_gaussians <= 0 or instance_dim <= 0:
            raise ValueError("oracle dimensions must be positive")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        initial = torch.randn(num_gaussians, instance_dim, generator=generator) / instance_dim**0.5
        self.code = nn.Parameter(initial)

    def forward(self) -> torch.Tensor:
        return F.normalize(self.code, dim=-1, eps=1e-8)


def _validate_latent(latent: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(latent)
    if value.ndim != 2 or value.shape[1] != 512 or not bool(torch.isfinite(value).all()):
        raise ValueError("structured memory input must be finite [N,512]")
    return value.float()


__all__ = [
    "ExtraInstanceCodeOracle",
    "LowRankPrivateBranchMemory",
    "OrthogonalProductMemory",
    "SharedPrivateLayout",
    "StructuredMemoryHeads",
    "StructuredSharedPrivateMemory",
]
