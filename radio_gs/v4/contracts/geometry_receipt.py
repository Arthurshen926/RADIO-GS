"""Hash-bound geometry authority records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class HashedInput:
    role: str
    path: str
    sha256: str

    @classmethod
    def seal(cls, role: str, path: str | Path) -> "HashedInput":
        resolved = Path(path).resolve(strict=True)
        return cls(role=role, path=str(resolved), sha256=sha256_file(resolved))


@dataclass(frozen=True)
class GeometryReceipt:
    carrier: str
    coordinate_convention: str
    inputs: tuple[HashedInput, ...]
    source_rgb_opened: bool
    target_rgb_opened: bool
    benchmark_images_opened: bool
    benchmark_masks_opened: bool
    benchmark_labels_opened: bool
    model_family: str | None = None
    model_checkpoint_sha256: str | None = None
    calibration: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema: str = "radio_gs.surface_object_memory_v4.geometry_receipt.v1"

    def __post_init__(self) -> None:
        if self.target_rgb_opened and not self.metadata.get("assisted_diagnostic", False):
            raise ValueError("target RGB is forbidden outside an assisted diagnostic")
        if self.model_family and not self.model_checkpoint_sha256:
            raise ValueError("model-backed geometry requires a checkpoint hash")
        if not self.inputs:
            raise ValueError("geometry receipt requires at least one sealed input")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
