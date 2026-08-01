from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from radio_gs.utils.immutable_artifacts import (
    load_sha_bound_project_checkpoint_payload,
)


def load_trusted_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    **kwargs: Any,
) -> Any:
    """Load a project checkpoint, with a restricted path for formal callers.

    PyTorch 2.6+ defaults ``torch.load`` to ``weights_only=True``. The training
    checkpoints in this project are trusted local artifacts and may contain
    numpy scalar metadata, optimizer state, and other non-tensor entries.
    Supplying ``expected_sha256`` changes the trust boundary: the checkpoint
    is read through one stable descriptor and a minimal legacy-project pickle
    allowlist, with no unrestricted fallback.
    """

    checkpoint_path = Path(path)
    if expected_sha256 is not None:
        unsupported = set(kwargs) - {"map_location"}
        if unsupported:
            raise TypeError(
                "SHA-bound project checkpoint loading does not support: "
                + ", ".join(sorted(unsupported))
            )
        payload, _digest, _source = (
            load_sha_bound_project_checkpoint_payload(
                checkpoint_path,
                expected_sha256=expected_sha256,
                map_location=kwargs.get("map_location", "cpu"),
                label="externally SHA-bound RADIO-GS checkpoint",
            )
        )
        return payload
    try:
        return torch.load(checkpoint_path, weights_only=False, **kwargs)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(checkpoint_path, **kwargs)
