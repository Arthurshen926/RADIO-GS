"""Deterministic sparse-support augmentation for surface-region training.

The surface-region caches store tokens in their declared geodesic order.  This
module only removes a suffix of that ordered valid set (while always retaining
the anchor); it never reorders or synthesizes tokens.  Sampling is keyed by a
stable region identifier rather than by batch position, so dataloader order
and cache sharding cannot change an augmented example.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from collections.abc import Mapping, Sequence

import torch


_HASH_DOMAIN = "radio_gs.surface_region_sparse_support.v1"


def _region_digest(*, seed: int, epoch: int, region_id: str) -> bytes:
    """Return an unambiguous, versioned digest for one augmentation decision."""

    payload = json.dumps(
        [_HASH_DOMAIN, int(seed), int(epoch), str(region_id)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _log_uniform_token_count(
    *,
    minimum_tokens: int,
    available_tokens: int,
    digest: bytes,
) -> int:
    """Map a SHA256 digest to an integer log-uniform token count.

    We draw continuously and quantize by flooring.  Using ``n + 1`` as the
    upper edge gives every integer in ``[minimum_tokens, available_tokens]`` a
    non-empty log interval.  The digest is interpreted as a uniform value in
    ``[0, 1)`` and no process-global random-number state is consulted.
    """

    minimum = int(minimum_tokens)
    available = int(available_tokens)
    if minimum <= 0:
        raise ValueError("minimum_tokens must be positive")
    if available < minimum:
        raise ValueError("available_tokens cannot be below minimum_tokens")
    if len(digest) != hashlib.sha256().digest_size:
        raise ValueError("digest must be one SHA256 value")
    if available == minimum:
        return available
    unit = int.from_bytes(digest, byteorder="big") / float(1 << 256)
    log_value = math.log(minimum) + unit * (
        math.log(available + 1) - math.log(minimum)
    )
    return min(available, max(minimum, int(math.floor(math.exp(log_value)))))


@dataclass(frozen=True)
class SparseTokenSupport:
    """Shape-preserving sparse selection and its per-region token counts."""

    token_mask: torch.Tensor
    kept_counts: torch.Tensor
    available_counts: torch.Tensor

    def zero_tensor(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``values`` with every dropped/padded token set exactly to zero.

        ``values`` must start with the same ``[batch, token]`` axes as the
        selection.  Trailing feature axes are broadcast without being copied
        into the selection object.  The operation preserves dtype, device, and
        autograd connectivity of retained values.
        """

        tensor = torch.as_tensor(values)
        if tensor.ndim < 2 or tuple(tensor.shape[:2]) != tuple(self.token_mask.shape):
            raise ValueError(
                "token tensor must begin with the selection's [batch, token] axes"
            )
        mask = self.token_mask.to(device=tensor.device)
        mask = mask.reshape(*mask.shape, *((1,) * (tensor.ndim - 2)))
        return tensor.masked_fill(~mask, 0)

    def zero_tensors(
        self,
        tensors: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply :meth:`zero_tensor` to a named collection of token tensors."""

        return {str(name): self.zero_tensor(value) for name, value in tensors.items()}


def deterministic_sparse_token_support(
    token_mask: torch.Tensor,
    *,
    anchor_index: torch.Tensor | Sequence[int],
    region_ids: Sequence[str],
    minimum_tokens: int,
    seed: int,
    epoch: int,
) -> SparseTokenSupport:
    """Select deterministic sparse prefixes of ordered surface-region tokens.

    For a region with ``n`` valid tokens, the kept count ``k`` is sampled
    log-uniformly from ``[minimum_tokens, n]`` using
    ``SHA256(seed, epoch, region_id)``.  Valid tokens are kept in their existing
    cache order, which is the surface contract's token/geodesic order.  If an
    input ever places its anchor outside the nearest ``k`` valid positions, the
    farthest selected non-anchor is replaced by the anchor without changing
    tensor layout or ``anchor_index``.

    Regions below ``minimum_tokens`` fail closed.  Such a row cannot satisfy
    the declared sampling range and should be repaired at cache construction,
    rather than silently changing the augmentation distribution.
    """

    mask = torch.as_tensor(token_mask).bool()
    if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[1] == 0:
        raise ValueError("token_mask must be a non-empty [batch, token] tensor")
    batch_size, token_count = mask.shape
    if len(region_ids) != batch_size:
        raise ValueError("region_ids must contain one stable ID per batch row")
    normalized_ids = [str(value) for value in region_ids]
    if any(not value for value in normalized_ids):
        raise ValueError("region_ids cannot contain empty identifiers")
    minimum = int(minimum_tokens)
    if minimum <= 0:
        raise ValueError("minimum_tokens must be positive")
    seed_value = int(seed)
    epoch_value = int(epoch)
    if seed_value < 0 or epoch_value < 0:
        raise ValueError("seed and epoch must be non-negative")

    anchors = torch.as_tensor(anchor_index, device=mask.device).long().reshape(-1)
    if anchors.shape != (batch_size,):
        raise ValueError("anchor_index must contain one position per batch row")
    if bool(((anchors < 0) | (anchors >= token_count)).any()):
        raise ValueError("anchor_index is outside the token axis")
    batch = torch.arange(batch_size, device=mask.device)
    if not bool(mask[batch, anchors].all()):
        raise ValueError("every anchor must be valid in the input token_mask")

    available = mask.sum(dim=1, dtype=torch.long)
    if bool((available < minimum).any()):
        raise ValueError(
            "input region has fewer valid tokens than minimum_tokens"
        )
    selected = torch.zeros_like(mask)
    kept: list[int] = []
    for row, region_id in enumerate(normalized_ids):
        count = _log_uniform_token_count(
            minimum_tokens=minimum,
            available_tokens=int(available[row]),
            digest=_region_digest(
                seed=seed_value,
                epoch=epoch_value,
                region_id=region_id,
            ),
        )
        valid_positions = torch.nonzero(mask[row], as_tuple=False).flatten()
        chosen = valid_positions[:count]
        anchor = anchors[row]
        if not bool((chosen == anchor).any()):
            chosen = torch.cat((chosen[:-1], anchor.reshape(1)))
        selected[row, chosen] = True
        kept.append(count)

    return SparseTokenSupport(
        token_mask=selected,
        kept_counts=torch.tensor(kept, dtype=torch.long, device=mask.device),
        available_counts=available,
    )


__all__ = [
    "SparseTokenSupport",
    "deterministic_sparse_token_support",
]
