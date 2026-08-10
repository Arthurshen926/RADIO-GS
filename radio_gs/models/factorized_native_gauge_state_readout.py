"""Gauge-explicit, query-free readout for factorized RADIO region tokens.

The model never reconstructs a raw RADIO vector.  Semantic direction is the
only vector-valued carrier; log amplitude and observation state can condition
attention and the learned residual only through separate scalar encoders.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


DIRECTION_ONLY = "direction_only"
DIRECTION_PLUS_LOG_AMPLITUDE = "direction_plus_log_amplitude"
DIRECTION_PLUS_LOG_AMPLITUDE_PLUS_FULL_STATE = (
    "direction_plus_log_amplitude_plus_full_state"
)
FACTORIZED_NATIVE_READOUT_ARMS = (
    DIRECTION_ONLY,
    DIRECTION_PLUS_LOG_AMPLITUDE,
    DIRECTION_PLUS_LOG_AMPLITUDE_PLUS_FULL_STATE,
)


@dataclass(frozen=True)
class FactorizedNativeReadoutOutput:
    """Summary-token output plus query-independent diagnostics."""

    summary_token: torch.Tensor
    pooled_hidden: torch.Tensor
    attention_weights: torch.Tensor


class FactorizedNativeGaugeStateReadout(nn.Module):
    """Map explicit factorized token state to an official RADIO summary token.

    ``state[..., 0]`` is the canonical predicted-log-amplitude state column.
    It must equal the independent ``log_amplitude`` carrier on every active
    token.  The duplicate state column is intentionally excluded from the
    full-state encoder; amplitude therefore has one and only one learned path.
    """

    FEATURE_DIM = 1280
    STATE_DIM = 6
    NON_GAUGE_STATE_DIM = 5

    def __init__(
        self,
        *,
        arm: str,
        log_amplitude_median: torch.Tensor,
        log_amplitude_robust_scale: torch.Tensor,
        state_median: torch.Tensor,
        state_robust_scale: torch.Tensor,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.arm = str(arm)
        self.hidden_dim = int(hidden_dim)
        if self.arm not in FACTORIZED_NATIVE_READOUT_ARMS:
            raise ValueError("unsupported factorized-native readout arm")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.register_buffer(
            "log_amplitude_median",
            self._buffer(
                log_amplitude_median,
                shape=(1,),
                positive=False,
                label="log-amplitude median",
            ),
        )
        self.register_buffer(
            "log_amplitude_robust_scale",
            self._buffer(
                log_amplitude_robust_scale,
                shape=(1,),
                positive=True,
                label="log-amplitude robust scale",
            ),
        )
        self.register_buffer(
            "state_median",
            self._buffer(
                state_median,
                shape=(self.STATE_DIM,),
                positive=False,
                label="state median",
            ),
        )
        self.register_buffer(
            "state_robust_scale",
            self._buffer(
                state_robust_scale,
                shape=(self.STATE_DIM,),
                positive=True,
                label="state robust scale",
            ),
        )

        self.direction_encoder = nn.Sequential(
            nn.LayerNorm(self.FEATURE_DIM),
            nn.Linear(self.FEATURE_DIM, self.hidden_dim),
        )
        self.log_amplitude_encoder: nn.Module | None = None
        if self.uses_log_amplitude:
            self.log_amplitude_encoder = nn.Sequential(
                nn.LayerNorm(1),
                nn.Linear(1, self.hidden_dim),
            )
        self.state_encoder: nn.Module | None = None
        if self.uses_full_state:
            # Five non-gauge state values plus five availability bits.  State
            # column zero remains an audited duplicate of the separate gauge
            # carrier and is deliberately not encoded a second time.
            self.state_encoder = nn.Sequential(
                nn.LayerNorm(2 * self.NON_GAUGE_STATE_DIM),
                nn.Linear(2 * self.NON_GAUGE_STATE_DIM, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        self.query = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.key = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.FEATURE_DIM),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    @staticmethod
    def _buffer(
        value: torch.Tensor,
        *,
        shape: tuple[int, ...],
        positive: bool,
        label: str,
    ) -> torch.Tensor:
        result = torch.as_tensor(value).detach().float().cpu().clone()
        if (
            result.shape != shape
            or not bool(torch.isfinite(result).all())
            or (positive and not bool((result > 0).all()))
        ):
            qualifier = "positive finite" if positive else "finite"
            raise ValueError(f"{label} must be {qualifier} with shape {list(shape)}")
        return result

    @property
    def uses_log_amplitude(self) -> bool:
        return self.arm != DIRECTION_ONLY

    @property
    def uses_full_state(self) -> bool:
        return self.arm == DIRECTION_PLUS_LOG_AMPLITUDE_PLUS_FULL_STATE

    def architecture(self, interface_contract_sha256: str) -> dict[str, Any]:
        return {
            "name": "factorized_native_gauge_state_readout_v1",
            "arm": self.arm,
            "feature_dim": self.FEATURE_DIM,
            "state_dim": self.STATE_DIM,
            "hidden_dim": self.hidden_dim,
            "output": "official_radio_summary_token_1280",
            "direction_gauge": "unit_l2_only",
            "amplitude_path": (
                "separate_scalar_conditioning_only"
                if self.uses_log_amplitude
                else "validated_but_structurally_unused"
            ),
            "state_path": (
                "five_non_gauge_values_plus_five_known_bits"
                if self.uses_full_state
                else "validated_but_structurally_unused"
            ),
            "raw_vector_reconstruction": False,
            "query_conditioning": False,
            "interface_contract_sha256": str(interface_contract_sha256),
        }

    def _validated_inputs(
        self,
        unit_direction: torch.Tensor,
        log_amplitude: torch.Tensor,
        state: torch.Tensor,
        state_known_mask: torch.Tensor,
        token_mask: torch.Tensor,
        anchor_index: torch.Tensor | int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        source_direction = torch.as_tensor(unit_direction)
        if not source_direction.is_floating_point():
            raise ValueError("unit_direction must be floating point")
        direction = source_direction.float()
        squeeze = direction.ndim == 2
        if squeeze:
            direction = direction[None]
            log_amplitude = torch.as_tensor(log_amplitude)[None]
            state = torch.as_tensor(state)[None]
            state_known_mask = torch.as_tensor(state_known_mask)[None]
            token_mask = torch.as_tensor(token_mask)[None]
        amplitude = torch.as_tensor(log_amplitude, device=direction.device).float()
        if amplitude.ndim == 3 and amplitude.shape[-1] == 1:
            amplitude = amplitude[..., 0]
        values = torch.as_tensor(state, device=direction.device).float()
        known = torch.as_tensor(state_known_mask, device=direction.device)
        mask = torch.as_tensor(token_mask, device=direction.device)
        if (
            direction.ndim != 3
            or direction.shape[-1] != self.FEATURE_DIM
            or amplitude.shape != direction.shape[:2]
            or values.shape != (*direction.shape[:2], self.STATE_DIM)
            or known.dtype != torch.bool
            or known.shape != values.shape
            or mask.dtype != torch.bool
            or mask.shape != direction.shape[:2]
            or not bool(mask.any(dim=1).all())
        ):
            raise ValueError("factorized-native token carriers differ")
        active_direction = direction[mask]
        active_amplitude = amplitude[mask]
        if (
            not bool(torch.isfinite(active_direction).all())
            or not bool(torch.isfinite(active_amplitude).all())
            or not bool(torch.isfinite(values[mask]).all())
        ):
            raise ValueError("active factorized-native carriers must be finite")
        tolerance = (
            5e-4
            if source_direction.dtype in {torch.float16, torch.bfloat16}
            else 2e-4
        )
        norms = torch.linalg.vector_norm(active_direction, dim=-1)
        if not torch.allclose(
            norms, torch.ones_like(norms), rtol=0.0, atol=tolerance
        ):
            raise ValueError("unit_direction must use the unit L2 gauge")
        if (
            bool(direction[~mask].count_nonzero())
            or bool(amplitude[~mask].count_nonzero())
            or bool(values[~mask].count_nonzero())
            or bool(known[~mask].any())
        ):
            raise ValueError("padding carriers must be exact zero/missing")
        if bool(values[~known].count_nonzero()):
            raise ValueError("unknown state values must be exact zero")
        if not bool(known[..., 0][mask].all()):
            raise ValueError("active log-amplitude state must be known")
        if not torch.allclose(
            values[..., 0][mask], active_amplitude, rtol=0.0, atol=1e-6
        ):
            raise ValueError("separate log amplitude differs from state column zero")
        anchor = torch.as_tensor(anchor_index, device=direction.device).long().reshape(-1)
        if anchor.numel() == 1:
            anchor = anchor.expand(direction.shape[0])
        batch = torch.arange(direction.shape[0], device=direction.device)
        if (
            anchor.shape != (direction.shape[0],)
            or bool((anchor < 0).any())
            or bool((anchor >= direction.shape[1]).any())
            or not bool(mask[batch, anchor].all())
        ):
            raise ValueError("anchor_index must select one active exact-state token")
        return direction, amplitude, values, known, mask, anchor

    def forward_with_diagnostics(
        self,
        unit_direction: torch.Tensor,
        log_amplitude: torch.Tensor,
        state: torch.Tensor,
        state_known_mask: torch.Tensor,
        *,
        token_mask: torch.Tensor,
        anchor_index: torch.Tensor | int,
    ) -> FactorizedNativeReadoutOutput:
        squeeze = torch.as_tensor(unit_direction).ndim == 2
        direction, amplitude, values, known, mask, anchor = self._validated_inputs(
            unit_direction,
            log_amplitude,
            state,
            state_known_mask,
            token_mask,
            anchor_index,
        )
        hidden = self.direction_encoder(direction)
        if self.log_amplitude_encoder is not None:
            normalized_amplitude = (
                amplitude[..., None] - self.log_amplitude_median
            ) / self.log_amplitude_robust_scale
            hidden = hidden + self.log_amplitude_encoder(normalized_amplitude)
        if self.state_encoder is not None:
            non_gauge_known = known[..., 1:]
            normalized_state = (
                values[..., 1:] - self.state_median[1:]
            ) / self.state_robust_scale[1:]
            normalized_state = normalized_state.masked_fill(~non_gauge_known, 0.0)
            state_input = torch.cat(
                (normalized_state, non_gauge_known.to(normalized_state.dtype)),
                dim=-1,
            )
            hidden = hidden + self.state_encoder(state_input)
        hidden = hidden.masked_fill(~mask[..., None], 0.0)
        batch = torch.arange(direction.shape[0], device=direction.device)
        query = self.query(hidden[batch, anchor])
        logits = torch.einsum("bh,bth->bt", query, self.key(hidden))
        logits = logits / math.sqrt(float(self.hidden_dim))
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=1)

        # The base is a spherical direction pool.  Amplitude never multiplies
        # direction, so this cannot silently recreate the legacy raw gauge.
        mean_direction = torch.einsum("bt,btc->bc", attention, direction)
        anchor_direction = direction[batch, anchor]
        base = F.normalize(
            mean_direction + 0.25 * (anchor_direction - mean_direction),
            dim=-1,
        )
        pooled = torch.einsum("bt,bth->bh", attention, hidden) + query
        summary = base + self.residual(pooled)
        output = FactorizedNativeReadoutOutput(summary, pooled, attention)
        if not squeeze:
            return output
        return FactorizedNativeReadoutOutput(
            output.summary_token[0],
            output.pooled_hidden[0],
            output.attention_weights[0],
        )

    def forward(
        self,
        unit_direction: torch.Tensor,
        log_amplitude: torch.Tensor,
        state: torch.Tensor,
        state_known_mask: torch.Tensor,
        *,
        token_mask: torch.Tensor,
        anchor_index: torch.Tensor | int,
    ) -> torch.Tensor:
        return self.forward_with_diagnostics(
            unit_direction,
            log_amplitude,
            state,
            state_known_mask,
            token_mask=token_mask,
            anchor_index=anchor_index,
        ).summary_token


__all__ = [
    "DIRECTION_ONLY",
    "DIRECTION_PLUS_LOG_AMPLITUDE",
    "DIRECTION_PLUS_LOG_AMPLITUDE_PLUS_FULL_STATE",
    "FACTORIZED_NATIVE_READOUT_ARMS",
    "FactorizedNativeGaugeStateReadout",
    "FactorizedNativeReadoutOutput",
]
