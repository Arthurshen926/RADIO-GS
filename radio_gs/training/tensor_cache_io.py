"""Tensor-cache loading helpers used by RADIO-GS training code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_training_tensor_cache(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    purpose: str = "tensor_cache",
) -> Any:
    """Load local feature/text/cache tensors; model checkpoints use trusted IO."""
    cache_path = Path(path).expanduser()
    if not cache_path.exists():
        raise FileNotFoundError(f"{purpose} not found: {cache_path}")
    return torch.load(cache_path, map_location=map_location)
