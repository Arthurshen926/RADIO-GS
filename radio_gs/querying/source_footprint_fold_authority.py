"""CPU authority for source-raster dominant-footprint OOF groups.

The grouping in this module is deliberately query-free.  It consumes only an
already verified sparse raster-responsibility matrix and the immutable leaf
row authority of a surface hierarchy.  Source masks, labels, probabilities,
and benchmark targets are not inputs to authority construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


METHOD_CONTRACT = "source_raster_dominant_footprint_blocks_v1"
LONG_AXIS_BLOCKS = 8
OOF_FOLDS = 3
MINIMUM_CLASS_ROWS = 32
FIELD_BASE_ACTION = "field_base"
RUN_SOURCE_OOF_ACTION = "run_source_oof"
AUTHORITY_ARTIFACT_SCHEMA = "radio_gs.source_footprint_fold_authority.v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}
_AUTHORITY_TENSOR_NAMES = (
    "primitive_rows",
    "group_ids",
    "visible_mass",
    "dominant_mass",
    "purity",
)
_EVIDENCE_NAMES = (
    "positive_weight",
    "negative_weight",
    "raw_positive_mass",
    "raw_negative_mass",
)


def _require_sha256(value: str, name: str) -> str:
    digest = str(value)
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _cpu_tensor(value: object, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must reside on CPU")
    return tensor.detach().contiguous()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_named_tensors(named_tensors: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in named_tensors:
        tensor = _cpu_tensor(value, name).contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _grid_shape(height: int, width: int) -> tuple[int, int]:
    h = int(height)
    w = int(width)
    if h <= 0 or w <= 0:
        raise ValueError("source raster height and width must be positive")
    if h > np.iinfo(np.int32).max or w > np.iinfo(np.int32).max:
        raise ValueError("source raster dimensions exceed the registered integer domain")
    if h >= w:
        block_rows = LONG_AXIS_BLOCKS
        block_cols = max(1, (LONG_AXIS_BLOCKS * w * 2 + h) // (2 * h))
    else:
        block_cols = LONG_AXIS_BLOCKS
        block_rows = max(1, (LONG_AXIS_BLOCKS * h * 2 + w) // (2 * w))
    return int(block_rows), int(block_cols)


def _authority_contract(
    *,
    height: int,
    width: int,
    block_rows: int,
    block_cols: int,
    source_triplet_authority_sha256: str,
    canonical_triplet_sha256: str,
    tensor_sha256: Mapping[str, str],
    tensor_bundle_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "method_contract": METHOD_CONTRACT,
        "height": int(height),
        "width": int(width),
        "long_axis_blocks": LONG_AXIS_BLOCKS,
        "block_rows": int(block_rows),
        "block_cols": int(block_cols),
        "block_count": int(block_rows) * int(block_cols),
        "invisible_group_id": int(block_rows) * int(block_cols),
        "pixel_assignment": "row_major_pixel_center",
        "primitive_assignment": "responsibility_mass_argmax_exact_tie_smallest_block",
        "source_triplet_authority_sha256": source_triplet_authority_sha256,
        "canonical_triplet_sha256": canonical_triplet_sha256,
        "tensor_sha256": dict(tensor_sha256),
        "tensor_bundle_sha256": tensor_bundle_sha256,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }


@dataclass(frozen=True)
class SourceFootprintFoldAuthority:
    """Immutable grouping aligned exactly to hierarchy leaf rows."""

    primitive_rows: torch.Tensor
    group_ids: torch.Tensor
    visible_mass: torch.Tensor
    dominant_mass: torch.Tensor
    purity: torch.Tensor
    height: int
    width: int
    block_rows: int
    block_cols: int
    source_triplet_authority_sha256: str
    canonical_triplet_sha256: str
    tensor_sha256: dict[str, str]
    tensor_bundle_sha256: str
    authority_sha256: str

    def __post_init__(self) -> None:
        rows = _cpu_tensor(self.primitive_rows, "primitive_rows").long().reshape(-1)
        groups = _cpu_tensor(self.group_ids, "group_ids").long().reshape(-1)
        visible = _cpu_tensor(self.visible_mass, "visible_mass").double().reshape(-1)
        dominant = _cpu_tensor(self.dominant_mass, "dominant_mass").double().reshape(-1)
        purity = _cpu_tensor(self.purity, "purity").double().reshape(-1)
        h = int(self.height)
        w = int(self.width)
        expected_grid = _grid_shape(h, w)
        grid = (int(self.block_rows), int(self.block_cols))
        if grid != expected_grid:
            raise ValueError("footprint block grid differs from the fixed raster grid")
        if rows.numel() == 0 or bool((rows < 0).any()) or (
            rows.numel() > 1 and not bool((rows[1:] > rows[:-1]).all())
        ):
            raise ValueError("primitive_rows must be non-empty, unique, and sorted")
        if any(value.shape != rows.shape for value in (groups, visible, dominant, purity)):
            raise ValueError("footprint authority tensors do not align to primitive_rows")
        if any(
            not bool(torch.isfinite(value).all()) or bool((value < 0).any())
            for value in (visible, dominant, purity)
        ):
            raise ValueError("footprint authority masses and purity must be finite and non-negative")
        block_count = grid[0] * grid[1]
        if bool((groups < 0).any()) or bool((groups > block_count).any()):
            raise ValueError("footprint group id is outside the registered block domain")
        observed = visible > 0
        if bool((groups[observed] >= block_count).any()) or not bool(
            (groups[~observed] == block_count).all()
        ):
            raise ValueError("visible/invisible footprint group semantics differ")
        tolerance = 1e-12 * torch.maximum(torch.ones_like(visible), visible)
        if bool((dominant > visible + tolerance).any()) or bool((purity > 1).any()):
            raise ValueError("dominant footprint mass or purity exceeds visible mass")
        expected_purity = torch.zeros_like(visible)
        expected_purity[observed] = dominant[observed] / visible[observed]
        if not torch.equal(purity, expected_purity):
            raise ValueError("footprint purity differs from dominant/visible mass")
        triplet_authority = _require_sha256(
            self.source_triplet_authority_sha256,
            "source_triplet_authority_sha256",
        )
        triplet_content = _require_sha256(
            self.canonical_triplet_sha256, "canonical_triplet_sha256"
        )
        provided_tensor_sha = dict(self.tensor_sha256)
        if set(provided_tensor_sha) != set(_AUTHORITY_TENSOR_NAMES):
            raise ValueError("footprint authority tensor hash schema differs")
        named = list(
            zip(
                _AUTHORITY_TENSOR_NAMES,
                (rows, groups, visible, dominant, purity),
            )
        )
        expected_tensor_sha = {
            name: _hash_named_tensors([(name, value)]) for name, value in named
        }
        if provided_tensor_sha != expected_tensor_sha:
            raise ValueError("footprint authority tensor hash differs")
        bundle_sha = _require_sha256(
            self.tensor_bundle_sha256, "tensor_bundle_sha256"
        )
        expected_bundle_sha = _hash_named_tensors(named)
        if bundle_sha != expected_bundle_sha:
            raise ValueError("footprint authority tensor bundle hash differs")
        authority = _require_sha256(self.authority_sha256, "authority_sha256")
        contract = _authority_contract(
            height=h,
            width=w,
            block_rows=grid[0],
            block_cols=grid[1],
            source_triplet_authority_sha256=triplet_authority,
            canonical_triplet_sha256=triplet_content,
            tensor_sha256=expected_tensor_sha,
            tensor_bundle_sha256=expected_bundle_sha,
        )
        if authority != _json_sha256(contract):
            raise ValueError("footprint authority digest differs")
        object.__setattr__(self, "primitive_rows", rows)
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "visible_mass", visible)
        object.__setattr__(self, "dominant_mass", dominant)
        object.__setattr__(self, "purity", purity)
        object.__setattr__(self, "height", h)
        object.__setattr__(self, "width", w)
        object.__setattr__(self, "block_rows", grid[0])
        object.__setattr__(self, "block_cols", grid[1])
        object.__setattr__(self, "source_triplet_authority_sha256", triplet_authority)
        object.__setattr__(self, "canonical_triplet_sha256", triplet_content)
        object.__setattr__(self, "tensor_sha256", expected_tensor_sha)
        object.__setattr__(self, "tensor_bundle_sha256", expected_bundle_sha)
        object.__setattr__(self, "authority_sha256", authority)

    @property
    def block_count(self) -> int:
        return int(self.block_rows) * int(self.block_cols)

    @property
    def invisible_group_id(self) -> int:
        return self.block_count

    @property
    def authority_digest(self) -> str:
        return self.authority_sha256

    def validate(self, *, expected_authority_sha256: str | None = None) -> None:
        """Recompute all digests, detecting mutation of contained tensors."""

        replay = SourceFootprintFoldAuthority(
            primitive_rows=self.primitive_rows,
            group_ids=self.group_ids,
            visible_mass=self.visible_mass,
            dominant_mass=self.dominant_mass,
            purity=self.purity,
            height=self.height,
            width=self.width,
            block_rows=self.block_rows,
            block_cols=self.block_cols,
            source_triplet_authority_sha256=self.source_triplet_authority_sha256,
            canonical_triplet_sha256=self.canonical_triplet_sha256,
            tensor_sha256=self.tensor_sha256,
            tensor_bundle_sha256=self.tensor_bundle_sha256,
            authority_sha256=self.authority_sha256,
        )
        if expected_authority_sha256 is not None and replay.authority_sha256 != _require_sha256(
            expected_authority_sha256, "expected_authority_sha256"
        ):
            raise ValueError("unknown source-footprint fold authority")

    def to_payload(self) -> dict[str, object]:
        """Return the exact, tensor-bound persistence schema."""

        self.validate(expected_authority_sha256=self.authority_sha256)
        return {
            "schema": AUTHORITY_ARTIFACT_SCHEMA,
            "schema_version": 1,
            "method_contract": METHOD_CONTRACT,
            "authority": {
                "height": self.height,
                "width": self.width,
                "block_rows": self.block_rows,
                "block_cols": self.block_cols,
                "source_triplet_authority_sha256": (
                    self.source_triplet_authority_sha256
                ),
                "canonical_triplet_sha256": self.canonical_triplet_sha256,
                "tensor_sha256": dict(self.tensor_sha256),
                "tensor_bundle_sha256": self.tensor_bundle_sha256,
                "authority_sha256": self.authority_sha256,
            },
            "tensors": {
                "primitive_rows": self.primitive_rows,
                "group_ids": self.group_ids,
                "visible_mass": self.visible_mass,
                "dominant_mass": self.dominant_mass,
                "purity": self.purity,
            },
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_computed": False,
        }


def save_source_footprint_fold_authority(
    authority: SourceFootprintFoldAuthority,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, str]:
    """Atomically persist an authority after replaying all tensor digests."""

    if not isinstance(authority, SourceFootprintFoldAuthority):
        raise TypeError("authority must be a SourceFootprintFoldAuthority")
    payload = authority.to_payload()
    output = Path(path).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(output),
        "file_sha256": _sha256_file(output),
        "authority_sha256": authority.authority_sha256,
        "tensor_bundle_sha256": authority.tensor_bundle_sha256,
    }


def load_source_footprint_fold_authority(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_authority_sha256: str,
) -> SourceFootprintFoldAuthority:
    """Load only the registered schema and replay its full authority."""

    source = Path(path).expanduser().absolute()
    if _sha256_file(source) != _require_sha256(
        expected_file_sha256, "expected_file_sha256"
    ):
        raise ValueError("source-footprint artifact file SHA-256 differs")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("source-footprint loading requires weights_only=True") from error
    required = {
        "schema",
        "schema_version",
        "method_contract",
        "authority",
        "tensors",
        "target_rgb_opened",
        "target_mask_opened",
        "target_metric_computed",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("source-footprint artifact fields differ")
    if (
        payload.get("schema") != AUTHORITY_ARTIFACT_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("method_contract") != METHOD_CONTRACT
    ):
        raise ValueError("source-footprint artifact schema differs")
    for flag in ("target_rgb_opened", "target_mask_opened", "target_metric_computed"):
        if payload.get(flag) is not False:
            raise ValueError("source-footprint artifact target-access flag differs")
    metadata = payload.get("authority")
    tensors = payload.get("tensors")
    metadata_keys = {
        "height",
        "width",
        "block_rows",
        "block_cols",
        "source_triplet_authority_sha256",
        "canonical_triplet_sha256",
        "tensor_sha256",
        "tensor_bundle_sha256",
        "authority_sha256",
    }
    if not isinstance(metadata, dict) or set(metadata) != metadata_keys:
        raise ValueError("source-footprint authority metadata fields differ")
    if not isinstance(tensors, dict) or set(tensors) != set(_AUTHORITY_TENSOR_NAMES):
        raise ValueError("source-footprint authority tensor fields differ")
    authority = SourceFootprintFoldAuthority(**metadata, **tensors)
    authority.validate(expected_authority_sha256=expected_authority_sha256)
    return authority


def build_source_raster_dominant_footprint_authority(
    pixel_ids: torch.Tensor,
    primitive_ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    height: int,
    width: int,
    hierarchy_primitive_rows: torch.Tensor,
    primitive_id_domain: str,
    source_triplet_authority_sha256: str,
    expected_source_triplet_authority_sha256: str,
) -> SourceFootprintFoldAuthority:
    """Build fixed source-raster groups without accepting query supervision."""

    source_authority = _require_sha256(
        source_triplet_authority_sha256, "source_triplet_authority_sha256"
    )
    expected_source = _require_sha256(
        expected_source_triplet_authority_sha256,
        "expected_source_triplet_authority_sha256",
    )
    if source_authority != expected_source:
        raise ValueError("unknown exact raster-responsibility authority")
    pids = _cpu_tensor(pixel_ids, "pixel_ids")
    ids = _cpu_tensor(primitive_ids, "primitive_ids")
    raw_weights = _cpu_tensor(weights, "weights")
    rows_raw = _cpu_tensor(hierarchy_primitive_rows, "hierarchy_primitive_rows")
    if pids.ndim != 1 or ids.ndim != 1 or raw_weights.ndim != 1:
        raise ValueError("exact responsibility triplets must be one-dimensional")
    if pids.shape != ids.shape or pids.shape != raw_weights.shape:
        raise ValueError("exact responsibility triplets must have matching shapes")
    if pids.dtype not in _INTEGER_DTYPES or ids.dtype not in _INTEGER_DTYPES:
        raise ValueError("pixel_ids and primitive_ids must have integer dtype")
    if rows_raw.ndim != 1 or rows_raw.dtype not in _INTEGER_DTYPES:
        raise ValueError("hierarchy_primitive_rows must be a one-dimensional integer tensor")
    if not raw_weights.dtype.is_floating_point:
        raise ValueError("exact responsibility weights must have floating-point dtype")
    if (
        not bool(torch.isfinite(raw_weights).all())
        or bool((raw_weights <= 0).any())
        or bool((raw_weights > 1).any())
    ):
        raise ValueError(
            "exact responsibility weights must be finite and in the interval (0,1]"
        )
    rows = rows_raw.long().reshape(-1)
    if rows.numel() == 0 or bool((rows < 0).any()) or (
        rows.numel() > 1 and not bool((rows[1:] > rows[:-1]).all())
    ):
        raise ValueError("hierarchy primitive rows must be non-empty, unique, and sorted")
    h = int(height)
    w = int(width)
    block_rows, block_cols = _grid_shape(h, w)
    pixel_count = h * w
    pids64 = pids.long().reshape(-1)
    if pids64.numel() and (
        int(pids64.min()) < 0 or int(pids64.max()) >= pixel_count
    ):
        raise ValueError("pixel id is outside the source raster")
    ids64 = ids.long().reshape(-1)
    domain = str(primitive_id_domain)
    if domain == "local_indices":
        if ids64.numel() and (
            int(ids64.min()) < 0 or int(ids64.max()) >= int(rows.numel())
        ):
            raise ValueError("local primitive id is outside hierarchy primitive_rows")
        local = ids64
    elif domain == "global_rows":
        if ids64.numel() and bool((ids64 < 0).any()):
            raise ValueError("global primitive row must be non-negative")
        local = torch.searchsorted(rows, ids64)
        if ids64.numel() and (
            bool((local >= rows.numel()).any())
            or not torch.equal(rows[local.clamp_max(rows.numel() - 1)], ids64)
        ):
            raise ValueError("global primitive row is absent from hierarchy primitive_rows")
    else:
        raise ValueError("primitive_id_domain must be 'local_indices' or 'global_rows'")

    pids_numpy = pids64.numpy()
    local_numpy = local.numpy()
    weight_numpy = raw_weights.double().reshape(-1).numpy()
    if pids_numpy.size:
        order = np.lexsort((pids_numpy, local_numpy))
        canonical_pids = pids_numpy[order]
        canonical_local = local_numpy[order]
        canonical_weights = weight_numpy[order]
        duplicate = (canonical_pids[1:] == canonical_pids[:-1]) & (
            canonical_local[1:] == canonical_local[:-1]
        )
        if bool(np.any(duplicate)):
            raise ValueError("exact responsibility triplets repeat pixel/primitive pairs")
    else:
        canonical_pids = pids_numpy.astype(np.int64, copy=False)
        canonical_local = local_numpy.astype(np.int64, copy=False)
        canonical_weights = weight_numpy.astype(np.float64, copy=False)
    if canonical_pids.size:
        pixel_order = np.argsort(canonical_pids, kind="stable")
        pixel_sorted = canonical_pids[pixel_order]
        pixel_weights = canonical_weights[pixel_order]
        pixel_start = np.r_[True, pixel_sorted[1:] != pixel_sorted[:-1]]
        pixel_indices = np.flatnonzero(pixel_start)
        pixel_mass = np.add.reduceat(pixel_weights, pixel_indices)
        pixel_tolerance = 1e-5 * np.maximum(1.0, pixel_mass)
        if bool(np.any(pixel_mass > 1.0 + pixel_tolerance)):
            raise ValueError(
                "exact front-to-back responsibility mass must not exceed one per pixel"
            )
    canonical_triplet_sha = _hash_named_tensors(
        [
            ("pixel_ids", torch.from_numpy(canonical_pids)),
            ("local_primitive_indices", torch.from_numpy(canonical_local)),
            ("weights", torch.from_numpy(canonical_weights)),
        ]
    )

    primitive_count = int(rows.numel())
    block_count = block_rows * block_cols
    visible_numpy = np.zeros(primitive_count, dtype=np.float64)
    dominant_numpy = np.zeros(primitive_count, dtype=np.float64)
    group_numpy = np.full(primitive_count, block_count, dtype=np.int64)
    if canonical_pids.size:
        y = canonical_pids // np.int64(w)
        x = canonical_pids - y * np.int64(w)
        block_y = ((2 * y + 1) * np.int64(block_rows)) // np.int64(2 * h)
        block_x = ((2 * x + 1) * np.int64(block_cols)) // np.int64(2 * w)
        block_ids = block_y * np.int64(block_cols) + block_x

        primitive_start = np.r_[True, canonical_local[1:] != canonical_local[:-1]]
        primitive_indices = np.flatnonzero(primitive_start)
        primitive_values = canonical_local[primitive_indices]
        primitive_mass = np.add.reduceat(canonical_weights, primitive_indices)
        visible_numpy[primitive_values] = primitive_mass

        pair_order = np.lexsort((canonical_pids, block_ids, canonical_local))
        pair_sorted_local = canonical_local[pair_order]
        pair_sorted_block = block_ids[pair_order]
        pair_sorted_weights = canonical_weights[pair_order]
        pair_start = np.r_[
            True,
            (pair_sorted_local[1:] != pair_sorted_local[:-1])
            | (pair_sorted_block[1:] != pair_sorted_block[:-1]),
        ]
        pair_indices = np.flatnonzero(pair_start)
        pair_local = pair_sorted_local[pair_indices]
        pair_block = pair_sorted_block[pair_indices]
        pair_mass = np.add.reduceat(pair_sorted_weights, pair_indices)
        np.maximum.at(dominant_numpy, pair_local, pair_mass)
        tied = pair_mass == dominant_numpy[pair_local]
        np.minimum.at(group_numpy, pair_local[tied], pair_block[tied])

    purity_numpy = np.zeros(primitive_count, dtype=np.float64)
    observed_numpy = visible_numpy > 0
    purity_numpy[observed_numpy] = (
        dominant_numpy[observed_numpy] / visible_numpy[observed_numpy]
    )
    tensors = {
        "primitive_rows": rows,
        "group_ids": torch.from_numpy(group_numpy),
        "visible_mass": torch.from_numpy(visible_numpy),
        "dominant_mass": torch.from_numpy(dominant_numpy),
        "purity": torch.from_numpy(purity_numpy),
    }
    tensor_sha = {
        name: _hash_named_tensors([(name, tensors[name])])
        for name in _AUTHORITY_TENSOR_NAMES
    }
    tensor_bundle_sha = _hash_named_tensors(
        [(name, tensors[name]) for name in _AUTHORITY_TENSOR_NAMES]
    )
    contract = _authority_contract(
        height=h,
        width=w,
        block_rows=block_rows,
        block_cols=block_cols,
        source_triplet_authority_sha256=source_authority,
        canonical_triplet_sha256=canonical_triplet_sha,
        tensor_sha256=tensor_sha,
        tensor_bundle_sha256=tensor_bundle_sha,
    )
    return SourceFootprintFoldAuthority(
        **tensors,
        height=h,
        width=w,
        block_rows=block_rows,
        block_cols=block_cols,
        source_triplet_authority_sha256=source_authority,
        canonical_triplet_sha256=canonical_triplet_sha,
        tensor_sha256=tensor_sha,
        tensor_bundle_sha256=tensor_bundle_sha,
        authority_sha256=_json_sha256(contract),
    )


def splitmix64_source_group_folds(
    group_ids: torch.Tensor,
    *,
    num_folds: int = OOF_FOLDS,
) -> torch.Tensor:
    """Assign every complete footprint group to one of exactly three folds."""

    if int(num_folds) != OOF_FOLDS:
        raise ValueError("source footprint authority requires exactly three folds")
    groups_raw = _cpu_tensor(group_ids, "group_ids")
    if groups_raw.ndim != 1 or groups_raw.dtype not in _INTEGER_DTYPES:
        raise ValueError("group_ids must be a one-dimensional integer tensor")
    groups = groups_raw.long().reshape(-1)
    if groups.numel() == 0 or bool((groups < 0).any()):
        raise ValueError("group_ids must be non-empty and non-negative")
    values = groups.numpy().astype(np.uint64, copy=True)
    values += np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    values = (values ^ (values >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    values ^= values >> np.uint64(31)
    return torch.from_numpy((values % np.uint64(OOF_FOLDS)).astype(np.int64))


def _evidence_vectors(
    authority: SourceFootprintFoldAuthority,
    values: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    expected_shape = authority.primitive_rows.shape
    result: dict[str, torch.Tensor] = {}
    for name in _EVIDENCE_NAMES:
        raw = _cpu_tensor(values[name], name)
        if not raw.dtype.is_floating_point:
            raise ValueError(f"{name} must have floating-point dtype")
        vector = raw.double().reshape(-1)
        if vector.shape != expected_shape:
            raise ValueError(f"{name} must align with hierarchy primitive_rows")
        if not bool(torch.isfinite(vector).all()) or bool((vector < 0).any()):
            raise ValueError(f"{name} must be finite and non-negative")
        result[name] = vector
    return result


@dataclass(frozen=True)
class ClearedSourceEvidenceFold:
    heldout_fold: int
    fold_ids: torch.Tensor
    heldout_rows: torch.Tensor
    training_positive_weight: torch.Tensor
    training_negative_weight: torch.Tensor
    training_raw_positive_mass: torch.Tensor
    training_raw_negative_mass: torch.Tensor
    authority_sha256: str


def clear_four_source_evidence_for_fold(
    authority: SourceFootprintFoldAuthority,
    positive_weight: torch.Tensor,
    negative_weight: torch.Tensor,
    raw_positive_mass: torch.Tensor,
    raw_negative_mass: torch.Tensor,
    *,
    heldout_fold: int,
    expected_authority_sha256: str,
) -> ClearedSourceEvidenceFold:
    """Zero all four evidence tensors on every row of a heldout group."""

    authority.validate(expected_authority_sha256=expected_authority_sha256)
    fold = int(heldout_fold)
    if fold not in range(OOF_FOLDS):
        raise ValueError("heldout_fold must be 0, 1, or 2")
    evidence = _evidence_vectors(
        authority,
        {
            "positive_weight": positive_weight,
            "negative_weight": negative_weight,
            "raw_positive_mass": raw_positive_mass,
            "raw_negative_mass": raw_negative_mass,
        },
    )
    fold_ids = splitmix64_source_group_folds(authority.group_ids)
    heldout = fold_ids == fold
    training = {name: value.clone() for name, value in evidence.items()}
    for value in training.values():
        value[heldout] = 0.0
        if bool((value[heldout] != 0).any()):
            raise RuntimeError("heldout source evidence survived whole-group clearing")
    return ClearedSourceEvidenceFold(
        heldout_fold=fold,
        fold_ids=fold_ids,
        heldout_rows=heldout,
        training_positive_weight=training["positive_weight"],
        training_negative_weight=training["negative_weight"],
        training_raw_positive_mass=training["raw_positive_mass"],
        training_raw_negative_mass=training["raw_negative_mass"],
        authority_sha256=authority.authority_sha256,
    )


@dataclass(frozen=True)
class SourceFoldBaseDecision:
    selected_action: str
    run_source_oof: bool
    reason: str
    fold_reports: tuple[dict[str, object], ...]
    minimum_class_rows: int
    authority_sha256: str


def source_fold_population_base_decision(
    authority: SourceFootprintFoldAuthority,
    positive_weight: torch.Tensor,
    negative_weight: torch.Tensor,
    *,
    expected_authority_sha256: str,
) -> SourceFoldBaseDecision:
    """Return the field base when any structured fold lacks either class."""

    authority.validate(expected_authority_sha256=expected_authority_sha256)
    evidence = _evidence_vectors(
        authority,
        {
            "positive_weight": positive_weight,
            "negative_weight": negative_weight,
            "raw_positive_mass": positive_weight,
            "raw_negative_mass": negative_weight,
        },
    )
    positive = evidence["positive_weight"]
    negative = evidence["negative_weight"]
    signed = positive - negative
    reference_weight = positive + negative
    observed = (reference_weight > 0) & (signed != 0)
    labels = signed > 0
    fold_ids = splitmix64_source_group_folds(authority.group_ids)
    reports: list[dict[str, object]] = []
    reasons: list[str] = []
    for fold in range(OOF_FOLDS):
        heldout_group = fold_ids == fold
        report: dict[str, object] = {"fold": fold}
        for population_name, mask in (
            ("heldout", observed & heldout_group),
            ("training", observed & ~heldout_group),
        ):
            positive_rows = mask & labels
            negative_rows = mask & ~labels
            positive_count = int(positive_rows.sum())
            negative_count = int(negative_rows.sum())
            positive_mass = float(reference_weight[positive_rows].sum())
            negative_mass = float(reference_weight[negative_rows].sum())
            report[f"{population_name}_positive_rows"] = positive_count
            report[f"{population_name}_negative_rows"] = negative_count
            report[f"{population_name}_positive_weight"] = positive_mass
            report[f"{population_name}_negative_weight"] = negative_mass
            if (
                positive_count < MINIMUM_CLASS_ROWS
                or negative_count < MINIMUM_CLASS_ROWS
                or not math.isfinite(positive_mass)
                or not math.isfinite(negative_mass)
                or positive_mass <= 0
                or negative_mass <= 0
            ):
                reasons.append(
                    f"fold_{fold}_{population_name}_signed_population_below_{MINIMUM_CLASS_ROWS}"
                )
        reports.append(report)
    run_oof = not reasons
    return SourceFoldBaseDecision(
        selected_action=RUN_SOURCE_OOF_ACTION if run_oof else FIELD_BASE_ACTION,
        run_source_oof=run_oof,
        reason="eligible" if run_oof else ";".join(reasons),
        fold_reports=tuple(reports),
        minimum_class_rows=MINIMUM_CLASS_ROWS,
        authority_sha256=authority.authority_sha256,
    )
