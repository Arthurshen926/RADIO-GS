"""Immutable sparse authority for exact 3DGS marginal responsibilities.

The authority is deliberately feature independent.  It stores only the
accepted front-to-back compositor hits and their exact base contribution
weights.  Raw RADIO, DINO and SAM consumers derive the same continuous
marginal weight from those bytes, so observation support cannot drift between
feature spaces.
"""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import torch

from radio_gs.rendering.contribution_compositor import (
    MARGINAL_RESPONSIBILITY_CONTRACT,
    marginal_responsibility_statistics,
)
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    load_torch_mapping,
    sha256_file,
)


SPARSE_EXACT_MARGINAL_AUTHORITY_SCHEMA = (
    "radio_gs.sparse_exact_marginal_responsibility_authority.v1"
)
SPARSE_EXACT_MARGINAL_VIEW_SCHEMA = (
    "radio_gs.sparse_exact_marginal_responsibility_view.v1"
)
SPARSE_EXACT_MARGINAL_PROGRESS_SCHEMA = (
    "radio_gs.sparse_exact_marginal_responsibility_progress.v1"
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def sparse_exact_marginal_formula_contract() -> dict[str, object]:
    """Return the parameter-free visibility and target-weight declaration."""

    return {
        "name": "sparse_exact_marginal_responsibility_authority_v1",
        "schema_version": 1,
        "base_weight": "exact_front_to_back_transmittance_times_footprint_alpha",
        "pixel_mass": "sum_base_weight_over_all_accepted_hits_at_pixel",
        "responsibility": "base_weight_divided_by_pixel_mass",
        "target_weight": "base_weight_times_responsibility",
        "target_weight_contract": MARGINAL_RESPONSIBILITY_CONTRACT,
        "accepted_hit_policy": "gsplat_rasterize_to_indices_in_range_unmodified",
        "post_compositor_threshold": 0.0,
        "canonical_order": "pixel_id_ascending_then_original_front_to_back_order",
        "feature_independent": True,
        "query_independent": True,
    }


SPARSE_EXACT_MARGINAL_FORMULA_SHA256 = _canonical_sha256(
    sparse_exact_marginal_formula_contract()
)


def sparse_exact_marginal_implementation_sha256() -> str:
    """Digest the implementation that defines the persisted authority."""

    return sha256_file(Path(__file__).resolve())


def canonicalize_sparse_marginal_view(
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    base_weights: torch.Tensor,
    *,
    num_gaussians: int,
    num_pixels: int,
) -> dict[str, torch.Tensor]:
    """Validate and canonically order one exact sparse compositor view."""

    gids = torch.as_tensor(gaussian_ids).detach().long().cpu().reshape(-1)
    pids = torch.as_tensor(pixel_ids).detach().long().cpu().reshape(-1)
    weights = torch.as_tensor(base_weights).detach().float().cpu().reshape(-1)
    if gids.shape != pids.shape or gids.shape != weights.shape:
        raise ValueError("sparse marginal view tensors must align")
    if int(num_gaussians) <= 0 or int(num_pixels) <= 0:
        raise ValueError("sparse marginal authority dimensions must be positive")
    if gids.numel() and (
        int(gids.min()) < 0
        or int(gids.max()) >= int(num_gaussians)
        or int(pids.min()) < 0
        or int(pids.max()) >= int(num_pixels)
    ):
        raise ValueError("sparse marginal view id outside declared domain")
    if (
        not bool(torch.isfinite(weights).all())
        or bool((weights <= 0).any())
        or bool((weights > 1).any())
    ):
        raise ValueError("sparse marginal base weights must lie in (0,1]")

    # rasterize_single_view_contributions already groups hits by pixel while
    # preserving the rasterizer's front-to-back order.  The stable sort makes
    # that order explicit without inventing a depth-dependent tie breaker.
    if pids.numel():
        order = torch.argsort(pids, stable=True)
        gids = gids[order]
        pids = pids[order]
        weights = weights[order]
        pair_key = pids * int(num_gaussians) + gids
        if torch.unique(pair_key).numel() != pair_key.numel():
            raise ValueError("sparse marginal view repeats a Gaussian/pixel pair")

    statistics = marginal_responsibility_statistics(
        pids,
        weights,
        num_pixels=int(num_pixels),
    )
    return {
        "gaussian_ids": gids.to(torch.int32),
        "pixel_ids": pids.to(torch.int32),
        "base_weights": weights.contiguous(),
        "marginal_weights": statistics.target_weight.float().cpu().contiguous(),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _commit_new_file(temporary: Path, path: Path) -> None:
    """Publish one complete file without ever replacing an existing inode."""

    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"immutable artifact already exists: {path}") from error
    _fsync_directory(path.parent)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    """Replace the writer-owned mutable progress receipt atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8"
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8"
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        _commit_new_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        _commit_new_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _AuthorityBuildLock:
    """Advisory single-writer lock whose ownership is released after a crash."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self._descriptor = os.open(path, flags, 0o600)
        try:
            fcntl.flock(self._descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(self._descriptor)
            self._descriptor = -1
            raise RuntimeError(
                "sparse marginal authority is already being written"
            ) from error

    def close(self) -> None:
        if self._descriptor >= 0:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            os.close(self._descriptor)
            self._descriptor = -1


def _load_view_shard(
    manifest_path: Path,
    record: Mapping[str, object],
    *,
    num_gaussians: int,
    num_pixels: int,
) -> dict[str, torch.Tensor]:
    relative = str(record.get("relative_path", ""))
    shard_path = (manifest_path.parent / relative).resolve()
    try:
        shard_path.relative_to(manifest_path.parent.resolve())
    except ValueError as error:
        raise ValueError("sparse marginal shard escapes authority directory") from error
    payload, digest, _source = load_torch_mapping(
        shard_path,
        expected_sha256=str(record.get("sha256", "")),
        map_location="cpu",
        label="sparse exact marginal responsibility view",
    )
    required = {
        "schema",
        "schema_version",
        "formula_sha256",
        "view_index",
        "frame_index",
        "num_gaussians",
        "num_pixels",
        "gaussian_ids",
        "pixel_ids",
        "base_weights",
    }
    if set(payload) != required or (
        payload.get("schema") != SPARSE_EXACT_MARGINAL_VIEW_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("formula_sha256")
        != SPARSE_EXACT_MARGINAL_FORMULA_SHA256
        or int(payload.get("view_index", -1)) != int(record.get("view_index", -2))
        or int(payload.get("frame_index", -1)) != int(record.get("frame_index", -2))
        or int(payload.get("num_gaussians", -1)) != int(num_gaussians)
        or int(payload.get("num_pixels", -1)) != int(num_pixels)
        or digest != str(record.get("sha256", ""))
    ):
        raise ValueError("sparse marginal view shard contract differs")
    if (
        not torch.is_tensor(payload["gaussian_ids"])
        or payload["gaussian_ids"].dtype != torch.int32
        or not torch.is_tensor(payload["pixel_ids"])
        or payload["pixel_ids"].dtype != torch.int32
        or not torch.is_tensor(payload["base_weights"])
        or payload["base_weights"].dtype != torch.float32
    ):
        raise ValueError("sparse marginal persisted tensor dtypes differ")
    assignment = canonicalize_sparse_marginal_view(
        payload["gaussian_ids"],
        payload["pixel_ids"],
        payload["base_weights"],
        num_gaussians=int(num_gaussians),
        num_pixels=int(num_pixels),
    )
    if not torch.equal(assignment["gaussian_ids"], payload["gaussian_ids"].int()) or (
        not torch.equal(assignment["pixel_ids"], payload["pixel_ids"].int())
        or not torch.equal(assignment["base_weights"], payload["base_weights"].float())
    ):
        raise ValueError("sparse marginal view shard ordering differs")
    if int(record.get("num_hits", -1)) != int(assignment["gaussian_ids"].numel()):
        raise ValueError("sparse marginal view shard hit count differs")
    return assignment


def _recover_unbound_view_shard(
    manifest_path: Path,
    *,
    view_index: int,
    frame_index: int,
    num_gaussians: int,
    num_pixels: int,
) -> dict[str, object]:
    """Validate and bind a shard committed just before a power interruption."""

    relative = f"{manifest_path.name}.views/view_{int(view_index):05d}.pt"
    shard_path = manifest_path.parent / relative
    payload, digest, _source = load_torch_mapping(
        shard_path,
        map_location="cpu",
        label="unbound sparse exact marginal responsibility view",
    )
    gaussian_ids = payload.get("gaussian_ids")
    if not torch.is_tensor(gaussian_ids):
        raise ValueError("unbound sparse marginal view shard is malformed")
    record: dict[str, object] = {
        "view_index": int(view_index),
        "frame_index": int(frame_index),
        "relative_path": relative,
        "sha256": digest,
        "num_hits": int(gaussian_ids.numel()),
    }
    _load_view_shard(
        manifest_path,
        record,
        num_gaussians=int(num_gaussians),
        num_pixels=int(num_pixels),
    )
    return record


class SparseExactMarginalAuthorityWriter:
    """Atomic, resumable writer for a manifest plus immutable view shards."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        metadata: Mapping[str, object],
        frame_indices: Sequence[int],
        num_gaussians: int,
        num_pixels: int,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.partial_path = self.manifest_path.with_suffix(
            self.manifest_path.suffix + ".partial.json"
        )
        self.shard_directory = self.manifest_path.parent / (
            self.manifest_path.name + ".views"
        )
        self.metadata = dict(metadata)
        self.frame_indices = [int(value) for value in frame_indices]
        self.num_gaussians = int(num_gaussians)
        self.num_pixels = int(num_pixels)
        self._lock: _AuthorityBuildLock | None = None
        try:
            self._lock = _AuthorityBuildLock(
                self.manifest_path.with_suffix(self.manifest_path.suffix + ".lock")
            )
            if self.manifest_path.exists():
                raise FileExistsError(
                    "sparse marginal authority already exists; use it as an immutable input"
                )
            if (
                self.num_gaussians <= 0
                or self.num_pixels <= 0
                or not self.frame_indices
                or len(set(self.frame_indices)) != len(self.frame_indices)
            ):
                raise ValueError("sparse marginal authority dimensions/views differ")
            self.records: dict[int, dict[str, object]] = {}
            if self.partial_path.exists():
                progress, _digest, _source = load_json_object(
                    self.partial_path, label="sparse marginal authority progress"
                )
                if (
                    progress.get("schema") != SPARSE_EXACT_MARGINAL_PROGRESS_SCHEMA
                    or progress.get("schema_version") != 1
                    or progress.get("formula_sha256")
                    != SPARSE_EXACT_MARGINAL_FORMULA_SHA256
                    or progress.get("metadata") != self.metadata
                    or progress.get("frame_indices") != self.frame_indices
                    or int(progress.get("num_gaussians", -1)) != self.num_gaussians
                    or int(progress.get("num_pixels", -1)) != self.num_pixels
                    or not isinstance(progress.get("views"), list)
                ):
                    raise ValueError("sparse marginal authority resume contract differs")
                for record in progress["views"]:
                    if not isinstance(record, Mapping):
                        raise ValueError("sparse marginal progress view is malformed")
                    view_index = int(record.get("view_index", -1))
                    if view_index in self.records or not 0 <= view_index < len(
                        self.frame_indices
                    ):
                        raise ValueError("sparse marginal progress repeats a view")
                    if (
                        int(record.get("frame_index", -1))
                        != self.frame_indices[view_index]
                    ):
                        raise ValueError("sparse marginal progress frame differs")
                    _load_view_shard(
                        self.manifest_path,
                        record,
                        num_gaussians=self.num_gaussians,
                        num_pixels=self.num_pixels,
                    )
                    self.records[view_index] = dict(record)

            recovered = False
            for view_index, frame_index in enumerate(self.frame_indices):
                if view_index in self.records:
                    continue
                shard_path = self.shard_directory / f"view_{view_index:05d}.pt"
                if shard_path.exists():
                    self.records[view_index] = _recover_unbound_view_shard(
                        self.manifest_path,
                        view_index=view_index,
                        frame_index=frame_index,
                        num_gaussians=self.num_gaussians,
                        num_pixels=self.num_pixels,
                    )
                    recovered = True
            if not self.partial_path.exists() or recovered:
                self._write_progress()
        except Exception:
            self._release_lock()
            raise

    def _release_lock(self) -> None:
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def __del__(self) -> None:
        self._release_lock()

    @property
    def completed_view_indices(self) -> frozenset[int]:
        return frozenset(self.records)

    def _progress_payload(self) -> dict[str, object]:
        return {
            "schema": SPARSE_EXACT_MARGINAL_PROGRESS_SCHEMA,
            "schema_version": 1,
            "formula_sha256": SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
            "metadata": self.metadata,
            "frame_indices": self.frame_indices,
            "num_gaussians": self.num_gaussians,
            "num_pixels": self.num_pixels,
            "views": [self.records[index] for index in sorted(self.records)],
        }

    def _write_progress(self) -> None:
        _atomic_json(self.partial_path, self._progress_payload())

    def add_view(
        self,
        view_index: int,
        gaussian_ids: torch.Tensor,
        pixel_ids: torch.Tensor,
        base_weights: torch.Tensor,
    ) -> None:
        index = int(view_index)
        if not 0 <= index < len(self.frame_indices):
            raise ValueError("sparse marginal authority view index outside contract")
        if index in self.records:
            raise ValueError("sparse marginal authority view is already complete")
        assignment = canonicalize_sparse_marginal_view(
            gaussian_ids,
            pixel_ids,
            base_weights,
            num_gaussians=self.num_gaussians,
            num_pixels=self.num_pixels,
        )
        relative = f"{self.manifest_path.name}.views/view_{index:05d}.pt"
        shard_path = self.manifest_path.parent / relative
        if shard_path.exists():
            raise FileExistsError("unbound sparse marginal view shard already exists")
        _write_new_torch(
            shard_path,
            {
                "schema": SPARSE_EXACT_MARGINAL_VIEW_SCHEMA,
                "schema_version": 1,
                "formula_sha256": SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
                "view_index": index,
                "frame_index": self.frame_indices[index],
                "num_gaussians": self.num_gaussians,
                "num_pixels": self.num_pixels,
                "gaussian_ids": assignment["gaussian_ids"],
                "pixel_ids": assignment["pixel_ids"],
                "base_weights": assignment["base_weights"],
            },
        )
        self.records[index] = {
            "view_index": index,
            "frame_index": self.frame_indices[index],
            "relative_path": relative,
            "sha256": sha256_file(shard_path),
            "num_hits": int(assignment["gaussian_ids"].numel()),
        }
        self._write_progress()

    def finalize(self) -> tuple[Path, str]:
        expected = set(range(len(self.frame_indices)))
        if set(self.records) != expected:
            missing = sorted(expected - set(self.records))
            raise RuntimeError(f"sparse marginal authority is incomplete: {missing}")
        manifest = {
            "schema": SPARSE_EXACT_MARGINAL_AUTHORITY_SCHEMA,
            "schema_version": 1,
            "formula_contract": sparse_exact_marginal_formula_contract(),
            "formula_sha256": SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
            "metadata": self.metadata,
            "frame_indices": self.frame_indices,
            "num_gaussians": self.num_gaussians,
            "num_pixels": self.num_pixels,
            "views": [self.records[index] for index in sorted(self.records)],
            "total_hits": sum(
                int(self.records[index]["num_hits"]) for index in sorted(self.records)
            ),
        }
        try:
            _write_new_json(self.manifest_path, manifest)
            self.partial_path.unlink(missing_ok=True)
            _fsync_directory(self.manifest_path.parent)
            return self.manifest_path, sha256_file(self.manifest_path)
        finally:
            self._release_lock()


def load_sparse_exact_marginal_authority(
    manifest_path: str | Path,
    *,
    expected_metadata: Mapping[str, object],
    expected_frame_indices: Sequence[int],
    num_gaussians: int,
    num_pixels: int,
    expected_sha256: str,
) -> tuple[list[dict[str, torch.Tensor]], str, Path]:
    """Load every bound shard and derive exact marginal target weights."""

    path = Path(manifest_path).expanduser().resolve()
    manifest, digest, source = load_json_object(
        path,
        expected_sha256=str(expected_sha256),
        label="sparse exact marginal responsibility authority",
    )
    required = {
        "schema",
        "schema_version",
        "formula_contract",
        "formula_sha256",
        "metadata",
        "frame_indices",
        "num_gaussians",
        "num_pixels",
        "views",
        "total_hits",
    }
    frames = [int(value) for value in expected_frame_indices]
    if set(manifest) != required or (
        manifest.get("schema") != SPARSE_EXACT_MARGINAL_AUTHORITY_SCHEMA
        or manifest.get("schema_version") != 1
        or manifest.get("formula_contract")
        != sparse_exact_marginal_formula_contract()
        or manifest.get("formula_sha256")
        != SPARSE_EXACT_MARGINAL_FORMULA_SHA256
        or manifest.get("metadata") != dict(expected_metadata)
        or manifest.get("frame_indices") != frames
        or int(manifest.get("num_gaussians", -1)) != int(num_gaussians)
        or int(manifest.get("num_pixels", -1)) != int(num_pixels)
        or not isinstance(manifest.get("views"), list)
        or len(manifest["views"]) != len(frames)
    ):
        raise ValueError("sparse marginal authority contract differs")
    assignments: list[dict[str, torch.Tensor]] = []
    total_hits = 0
    for expected_view, record in enumerate(manifest["views"]):
        if not isinstance(record, Mapping) or (
            int(record.get("view_index", -1)) != expected_view
            or int(record.get("frame_index", -1)) != frames[expected_view]
        ):
            raise ValueError("sparse marginal authority view order differs")
        assignment = _load_view_shard(
            path,
            record,
            num_gaussians=int(num_gaussians),
            num_pixels=int(num_pixels),
        )
        assignments.append(assignment)
        total_hits += int(assignment["gaussian_ids"].numel())
    if int(manifest.get("total_hits", -1)) != total_hits:
        raise ValueError("sparse marginal authority total hit count differs")
    return assignments, digest, source
