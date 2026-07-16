"""Fail-closed signatures for canonical and derived RADIO feature spaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping


@dataclass(frozen=True)
class FeatureSpaceSignature:
    """Complete compatibility contract for a query/field feature space.

    Equal dimensions are intentionally insufficient.  A query can only be
    compared with field descriptors when every representation-defining field
    below agrees.
    """

    radio_version: str
    radio_checkpoint_sha256: str
    raw_feature_dim: int
    adaptor_name: str = "backbone"
    adaptor_sha256: str = ""
    adaptor_output_dim: int = 0
    token_type: str = "spatial"
    normalization: str = "l2"
    extraction_resolution: tuple[int, int] | None = None
    crop_policy: str = "full_frame"
    field_checkpoint_sha256: str = ""
    semantic_alignment: str = "none"
    semantic_alignment_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.radio_version:
            raise ValueError("radio_version is required")
        if not self.radio_checkpoint_sha256:
            raise ValueError("radio_checkpoint_sha256 is required")
        if int(self.raw_feature_dim) <= 0:
            raise ValueError("raw_feature_dim must be positive")
        if int(self.adaptor_output_dim) < 0:
            raise ValueError("adaptor_output_dim cannot be negative")
        if self.token_type not in {"spatial", "summary", "region", "primitive"}:
            raise ValueError("unsupported token_type")
        if self.normalization not in {
            "none",
            "l2",
            "whiten_l2",
            "radio_raw_full",
            "radio_direction_unit",
        }:
            raise ValueError("unsupported normalization")
        if self.extraction_resolution is not None:
            resolution = tuple(int(v) for v in self.extraction_resolution)
            if len(resolution) != 2 or min(resolution) <= 0:
                raise ValueError("extraction_resolution must be (height,width)")
            object.__setattr__(self, "extraction_resolution", resolution)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def assert_compatible(
        self,
        other: "FeatureSpaceSignature",
        *,
        allow_field_checkpoint_difference: bool = False,
    ) -> None:
        if not isinstance(other, FeatureSpaceSignature):
            raise TypeError("other must be a FeatureSpaceSignature")
        left = self.to_dict()
        right = other.to_dict()
        if allow_field_checkpoint_difference:
            left.pop("field_checkpoint_sha256", None)
            right.pop("field_checkpoint_sha256", None)
        mismatches = {
            key: (left.get(key), right.get(key))
            for key in sorted(set(left) | set(right))
            if left.get(key) != right.get(key)
        }
        if mismatches:
            details = ", ".join(
                f"{key}={old!r}!={new!r}" for key, (old, new) in mismatches.items()
            )
            raise ValueError(f"Incompatible RADIO feature spaces: {details}")

    def assert_comparable(self, other: "FeatureSpaceSignature") -> None:
        """Require the same output space without erasing token provenance.

        Image spatial tokens and canonical primitive descriptors are not
        identical artifacts, so exact compatibility should reject them.  They
        are valid cosine operands when their frozen official adaptor,
        checkpoint, output dimension, and normalization agree.  Token type,
        crop policy, field hash, and bridge remain auditable provenance.
        """

        if not isinstance(other, FeatureSpaceSignature):
            raise TypeError("other must be a FeatureSpaceSignature")
        keys = (
            "radio_version",
            "radio_checkpoint_sha256",
            "raw_feature_dim",
            "adaptor_name",
            "adaptor_sha256",
            "adaptor_output_dim",
            "normalization",
        )
        mismatches = {
            key: (getattr(self, key), getattr(other, key))
            for key in keys
            if getattr(self, key) != getattr(other, key)
        }
        if mismatches:
            details = ", ".join(
                f"{key}={old!r}!={new!r}"
                for key, (old, new) in mismatches.items()
            )
            raise ValueError(f"Incomparable RADIO output spaces: {details}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FeatureSpaceSignature":
        return cls(**dict(value))
