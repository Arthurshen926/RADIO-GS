"""Shared SigLIP2 projection utilities for grounding-aware supervision."""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from radio_gs.utils.immutable_artifacts import (
    load_fixed_radio_checkpoint_payload,
    load_torch_payload,
)


OFFICIAL_C_RADIO_V4_H_HALF_SHA256 = (
    "bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9"
)


def _xformers_memory_efficient_attention():
    try:
        from xformers.ops import memory_efficient_attention
    except ImportError as exc:
        raise RuntimeError(
            "xFormers is required for memory-efficient SigLIP2 training "
            "attention; install the wheel matching the active Torch/CUDA ABI"
        ) from exc
    return memory_efficient_attention


def _xformers_attention_forward(
    attention: nn.Module,
    x: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact global timm attention using xFormers' bounded-memory kernel."""

    if attn_mask is not None:
        raise ValueError(
            "xFormers SigLIP2 projection does not accept an attention mask"
        )
    batch, tokens, _channels = x.shape
    qkv = attention.qkv(x).reshape(
        batch,
        tokens,
        3,
        attention.num_heads,
        attention.head_dim,
    )
    query, key, value = qkv.unbind(2)
    query = attention.q_norm(query)
    key = attention.k_norm(key)
    projected = _xformers_memory_efficient_attention()(
        query,
        key,
        value,
        p=attention.attn_drop.p if attention.training else 0.0,
        scale=attention.scale,
    )
    projected = projected.reshape(batch, tokens, attention.attn_dim)
    projected = attention.norm(projected)
    projected = attention.proj(projected)
    return attention.proj_drop(projected)


def _chunked_token_mlp_forward(mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Evaluate a timm token-wise MLP in exact bounded token blocks."""

    chunk_size = int(getattr(mlp, "_radio_gs_token_chunk_size"))
    if x.requires_grad and not any(
        parameter.requires_grad for parameter in mlp.parameters()
    ):
        return _FrozenChunkedTokenMLP.apply(x, mlp, chunk_size)
    return _run_token_mlp_chunks(mlp, x, chunk_size)


def _run_token_mlp(mlp: nn.Module, token_chunk: torch.Tensor) -> torch.Tensor:
    """Run one exact timm MLP block without dispatching its patched forward."""

    token_chunk = mlp.fc1(token_chunk)
    token_chunk = mlp.act(token_chunk)
    token_chunk = mlp.drop1(token_chunk)
    token_chunk = mlp.norm(token_chunk)
    token_chunk = mlp.fc2(token_chunk)
    return mlp.drop2(token_chunk)


def _run_token_mlp_chunks(
    mlp: nn.Module,
    x: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for token_chunk in x.split(chunk_size, dim=1):
        outputs.append(_run_token_mlp(mlp, token_chunk))
    return torch.cat(outputs, dim=1)


class _FrozenChunkedTokenMLP(torch.autograd.Function):
    """Exact frozen-MLP VJP with activation memory bounded by one chunk.

    The official adaptor is frozen, so backward only needs ``dL/dx``.  A
    conventional chunked forward still retains every chunk's hidden
    activation until its enclosing transformer backward.  This function
    stores only the MLP input, then recomputes and differentiates one chunk at
    a time.  It changes execution order and residency, not the function or
    first-order gradient.
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        mlp: nn.Module,
        chunk_size: int,
    ) -> torch.Tensor:
        ctx.mlp = mlp
        ctx.chunk_size = int(chunk_size)
        ctx.save_for_backward(x)
        return _run_token_mlp_chunks(mlp, x, int(chunk_size))

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        input_gradients: list[torch.Tensor] = []
        with torch.enable_grad():
            for input_chunk, output_gradient_chunk in zip(
                x.split(ctx.chunk_size, dim=1),
                grad_output.split(ctx.chunk_size, dim=1),
            ):
                differentiable_input = input_chunk.detach().requires_grad_(True)
                output_chunk = _run_token_mlp(ctx.mlp, differentiable_input)
                (input_gradient,) = torch.autograd.grad(
                    output_chunk,
                    differentiable_input,
                    output_gradient_chunk,
                    retain_graph=False,
                    create_graph=False,
                )
                input_gradients.append(input_gradient)
        return torch.cat(input_gradients, dim=1), None, None


def _run_output_mlp(
    projection: "SigLIP2FeatureProjection",
    token_chunk: torch.Tensor,
) -> torch.Tensor:
    return projection.mlp_final(projection.mlp_fc1(token_chunk))


class _FrozenChunkedOutputMLP(torch.autograd.Function):
    """Exact bounded VJP for the frozen final SigLIP2 projection MLP."""

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        projection: "SigLIP2FeatureProjection",
        chunk_size: int,
    ) -> torch.Tensor:
        ctx.projection = projection
        ctx.chunk_size = int(chunk_size)
        ctx.save_for_backward(x)
        return torch.cat(
            [
                _run_output_mlp(projection, chunk)
                for chunk in x.split(int(chunk_size), dim=1)
            ],
            dim=1,
        )

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        input_gradients: list[torch.Tensor] = []
        with torch.enable_grad():
            for input_chunk, output_gradient_chunk in zip(
                x.split(ctx.chunk_size, dim=1),
                grad_output.split(ctx.chunk_size, dim=1),
            ):
                differentiable_input = input_chunk.detach().requires_grad_(True)
                output_chunk = _run_output_mlp(
                    ctx.projection,
                    differentiable_input,
                )
                (input_gradient,) = torch.autograd.grad(
                    output_chunk,
                    differentiable_input,
                    output_gradient_chunk,
                    retain_graph=False,
                    create_graph=False,
                )
                input_gradients.append(input_gradient)
        return torch.cat(input_gradients, dim=1), None, None


def _load_weights_only(path: str) -> object:
    payload, _, _ = load_torch_payload(
        path,
        map_location="cpu",
        label="SigLIP2 projection checkpoint",
    )
    return payload


def _load_fixed_radio_checkpoint(
    path: str,
    *,
    expected_sha256: str,
) -> object:
    payload, _, _ = load_fixed_radio_checkpoint_payload(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="official C-RADIOv4-H checkpoint",
    )
    return payload


class SigLIP2FeatureProjection(nn.Module):
    """Project RADIO 1280d features into SigLIP2 visual embedding space.

    Uses the spatial feature projection from RADIO: 2 attention blocks + MLP.
    Maps to SigLIP2's *spatial* vision space (1536d).
    NOTE: For text grounding, use SigLIP2SummaryHead instead — it maps to
    the text-aligned summary space.
    """

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[
            Block(1280, num_heads=16, init_values=1e-5)
            for _ in range(2)
        ])
        self.mlp_fc1 = nn.Linear(1280, 1520)
        self.mlp_final = nn.Sequential(
            nn.LayerNorm(1520),
            nn.GELU(),
            nn.Linear(1520, 1536),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, 1280] -> [B, N, 1536]."""
        x = self.blocks(x)
        chunk_size = int(getattr(self, "token_mlp_chunk_size", 0))
        if chunk_size <= 0 or x.shape[1] <= chunk_size:
            return self.mlp_final(self.mlp_fc1(x))
        if x.requires_grad and not any(
            parameter.requires_grad for parameter in self.parameters()
        ):
            return _FrozenChunkedOutputMLP.apply(x, self, chunk_size)
        return torch.cat(
            [self.mlp_final(self.mlp_fc1(chunk)) for chunk in x.split(chunk_size, dim=1)],
            dim=1,
        )

    def enable_chunked_token_mlp(
        self, chunk_size: int
    ) -> "SigLIP2FeatureProjection":
        """Bound token-wise MLP activations without changing token scope."""

        if int(chunk_size) <= 0:
            raise ValueError("SigLIP2 token MLP chunk size must be positive")
        self.token_mlp_chunk_size = int(chunk_size)
        for block in self.blocks:
            block.mlp._radio_gs_token_chunk_size = int(chunk_size)
            block.mlp.forward = MethodType(_chunked_token_mlp_forward, block.mlp)
        return self

    def enable_xformers_memory_efficient_attention(
        self,
    ) -> "SigLIP2FeatureProjection":
        """Retain complete-grid attention with a training-capable sm86 kernel."""

        # Resolve the optional dependency before mutating any module method.
        _xformers_memory_efficient_attention()
        for block in self.blocks:
            block.attn.forward = MethodType(
                _xformers_attention_forward,
                block.attn,
            )
        self.attention_runtime = "xformers_memory_efficient_exact_global"
        return self

    @classmethod
    def from_extracted_weights(cls, ckpt_path: str) -> "SigLIP2FeatureProjection":
        """Load from already-extracted projection state dict (e.g. siglip2_feat_projection.pth)."""
        sd = _load_weights_only(ckpt_path)
        proj = cls()
        proj.load_state_dict(sd, strict=True)
        return proj

    @classmethod
    def from_radio_checkpoint(
        cls,
        ckpt_path: str,
        *,
        expected_sha256: str = OFFICIAL_C_RADIO_V4_H_HALF_SHA256,
    ) -> "SigLIP2FeatureProjection":
        chk = _load_fixed_radio_checkpoint(
            ckpt_path,
            expected_sha256=expected_sha256,
        )
        sd = chk["state_dict"]
        proj = cls()
        proj_sd = {}
        prefix = "_feature_projections.siglip2-g."
        for k, v in sd.items():
            if not k.startswith(prefix):
                continue
            new_k = k[len(prefix):]
            if new_k.startswith("mlp.fc1"):
                new_k = new_k.replace("mlp.fc1", "mlp_fc1")
            elif new_k.startswith("mlp.final"):
                new_k = new_k.replace("mlp.final", "mlp_final")
            proj_sd[new_k] = v.float()
        proj.load_state_dict(proj_sd, strict=True)
        return proj


class SigLIP2SummaryHead(nn.Module):
    """RADIO's SigLIP2 summary head — maps 1280d tokens to the text-aligned 1536d space.

    Architecture: Linear(1280→1520) + 2 residual blocks(LN+GELU+Linear) + final(LN+GELU+Linear→1536).
    This is ``_heads.siglip2-g`` from the RADIO checkpoint. Unlike the spatial feature projection,
    the summary head produces embeddings in the same space as SigLIP2 text embeddings, making it
    suitable for text grounding / open-vocabulary tasks when its inputs are
    genuine or explicitly predicted summary tokens.  Applying this head to
    every spatial pixel or Gaussian is not an official SigLIP2 projection;
    use :class:`SigLIP2FeatureProjection` for full-grid spatial preservation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1280, 1520)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(1520), nn.GELU(), nn.Linear(1520, 1520)),
            nn.Sequential(nn.LayerNorm(1520), nn.GELU(), nn.Linear(1520, 1520)),
        ])
        self.final = nn.Sequential(
            nn.LayerNorm(1520),
            nn.GELU(),
            nn.Linear(1520, 1536),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, 1280] -> [B, N, 1536]."""
        x = self.fc1(x)
        for blk in self.blocks:
            x = x + blk(x)
        return self.final(x)

    @classmethod
    def from_extracted_weights(cls, ckpt_path: str) -> "SigLIP2SummaryHead":
        """Load from extracted state dict (e.g. siglip2_summary_head.pth)."""
        sd = _load_weights_only(ckpt_path)
        head = cls()
        head.load_state_dict(sd, strict=True)
        return head

    @classmethod
    def from_radio_checkpoint(
        cls,
        ckpt_path: str,
        *,
        expected_sha256: str = OFFICIAL_C_RADIO_V4_H_HALF_SHA256,
    ) -> "SigLIP2SummaryHead":
        """Extract ``_heads.siglip2-g`` from a full RADIO checkpoint."""
        chk = _load_fixed_radio_checkpoint(
            ckpt_path,
            expected_sha256=expected_sha256,
        )
        sd = chk["state_dict"]
        head = cls()
        head_sd = {}
        prefix = "_heads.siglip2-g."
        for k, v in sd.items():
            if k.startswith(prefix):
                head_sd[k[len(prefix):]] = v.float()
        head.load_state_dict(head_sd, strict=True)
        return head
