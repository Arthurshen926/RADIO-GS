from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_trusted_checkpoint(path: str | Path, **kwargs: Any) -> Any:
    """Load a project-produced checkpoint with PyTorch's full unpickler.

    PyTorch 2.6+ defaults ``torch.load`` to ``weights_only=True``. The training
    checkpoints in this project are trusted local artifacts and may contain
    numpy scalar metadata, optimizer state, and other non-tensor entries.
    """

    checkpoint_path = Path(path)
    try:
        return torch.load(checkpoint_path, weights_only=False, **kwargs)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(checkpoint_path, **kwargs)

