"""Cryptographic authority for canonical primitive row identity.

The canonical geometry row, its validity bit, and the compact global-row order
jointly define the meaning of every feature, graph, and query-cache row.  A
checkpoint hash alone cannot detect a stale or reordered derived artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import torch


ROW_AUTHORITY_SCHEMA = "radio_gs.primitive_row_authority.v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        _canonical_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
    )
    digest.update(b"\0")
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PrimitiveRowAuthority:
    """Immutable identity of one full primitive domain and active row subset."""

    num_global_rows: int
    num_active_rows: int
    xyz_sha256: str
    valid_sha256: str
    global_rows_sha256: str
    schema: str = ROW_AUTHORITY_SCHEMA
    global_row_order: str = "torch_where_valid_ascending"

    def __post_init__(self) -> None:
        if self.schema != ROW_AUTHORITY_SCHEMA:
            raise ValueError("unsupported primitive row authority schema")
        if self.global_row_order != "torch_where_valid_ascending":
            raise ValueError("unsupported primitive global-row order")
        if int(self.num_global_rows) <= 0:
            raise ValueError("primitive row authority requires a non-empty domain")
        if not 0 <= int(self.num_active_rows) <= int(self.num_global_rows):
            raise ValueError("primitive row authority active-row count is invalid")
        for name in ("xyz_sha256", "valid_sha256", "global_rows_sha256"):
            digest = str(getattr(self, name))
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"primitive row authority {name} is not SHA-256")

    @classmethod
    def from_tensors(
        cls, xyz: torch.Tensor, valid: torch.Tensor
    ) -> "PrimitiveRowAuthority":
        points = torch.as_tensor(xyz).detach().cpu().float().contiguous()
        mask = torch.as_tensor(valid).detach().cpu().bool().contiguous()
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("primitive authority xyz must be [N,3]")
        if mask.shape != (points.shape[0],):
            raise ValueError("primitive authority valid mask must align with xyz")
        if not bool(torch.isfinite(points).all()):
            raise ValueError("primitive authority xyz must be finite")
        global_rows = torch.where(mask)[0].long().contiguous()
        return cls(
            num_global_rows=int(points.shape[0]),
            num_active_rows=int(global_rows.numel()),
            xyz_sha256=_tensor_sha256(points),
            valid_sha256=_tensor_sha256(mask),
            global_rows_sha256=_tensor_sha256(global_rows),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "num_global_rows": int(self.num_global_rows),
            "num_active_rows": int(self.num_active_rows),
            "xyz_sha256": self.xyz_sha256,
            "valid_sha256": self.valid_sha256,
            "global_rows_sha256": self.global_rows_sha256,
            "global_row_order": self.global_row_order,
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()

    @classmethod
    def from_mapping(cls, value: object) -> "PrimitiveRowAuthority":
        if not isinstance(value, Mapping):
            raise ValueError("primitive row authority must be a mapping")
        expected = {
            "schema",
            "num_global_rows",
            "num_active_rows",
            "xyz_sha256",
            "valid_sha256",
            "global_rows_sha256",
            "global_row_order",
        }
        if set(value) != expected:
            raise ValueError("primitive row authority keys differ from schema")
        return cls(**dict(value))

    def validate(self, xyz: torch.Tensor, valid: torch.Tensor) -> None:
        actual = type(self).from_tensors(xyz, valid)
        if self.to_dict() != actual.to_dict() or self.digest != actual.digest:
            raise ValueError("primitive row authority does not match tensor rows")

