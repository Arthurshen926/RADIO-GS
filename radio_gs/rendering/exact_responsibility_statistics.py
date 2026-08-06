"""Source-only sufficient statistics for an exact marginal authority.

This sidecar is descriptive evidence only.  It streams immutable authority
view shards, never opens benchmark data or text queries, and deliberately
contains no threshold or admission decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import torch

from radio_gs.rendering.sparse_marginal_authority import (
    SPARSE_EXACT_MARGINAL_AUTHORITY_SCHEMA,
    SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
    SPARSE_EXACT_MARGINAL_VIEW_SCHEMA,
    canonicalize_sparse_marginal_view,
    sparse_exact_marginal_formula_contract,
)
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    load_torch_mapping,
    sha256_file,
    write_torch_noclobber,
)


EXACT_RESPONSIBILITY_STATISTICS_SCHEMA = (
    "radio_gs.exact_responsibility_statistics.v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TENSOR_NAMES = (
    "visible_mass",
    "semantic_mass",
    "semantic_mass_sq",
    "nonzero_hit_count",
    "nonzero_view_count",
    "kish_effective_sample_size",
)


def exact_responsibility_statistics_contract() -> dict[str, object]:
    return {
        "name": "source_only_exact_responsibility_statistics_v1",
        "schema_version": 1,
        "parent_authority_schema": SPARSE_EXACT_MARGINAL_AUTHORITY_SCHEMA,
        "parent_formula_sha256": SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
        "accumulation_dtype": "float64",
        "visible_mass": "sum_base_weight_per_primitive",
        "semantic_mass": (
            "sum_base_weight_squared_divided_by_full_view_pixel_mass_per_primitive"
        ),
        "semantic_mass_sq": "sum_semantic_hit_weight_squared_per_primitive",
        "nonzero_hit_count": "count_persisted_strictly_positive_base_weight_hits",
        "nonzero_view_count": (
            "count_source_views_with_at_least_one_strictly_positive_base_weight_hit"
        ),
        "kish_effective_sample_size": (
            "semantic_mass_squared_divided_by_semantic_mass_sq_or_zero"
        ),
        "streaming_unit": "one_exact_authority_view_shard",
        "query_independent": True,
        "source_only": True,
        "contains_admission_decision": False,
    }


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


EXACT_RESPONSIBILITY_STATISTICS_CONTRACT_SHA256 = _canonical_json_sha256(
    exact_responsibility_statistics_contract()
)


def _require_sha256(value: str, label: str) -> str:
    digest = str(value)
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _tensor_sha256(name: str, value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    array = tensor.numpy()
    digest = hashlib.sha256()
    digest.update(str(name).encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _tensor_bundle_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in _TENSOR_NAMES:
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(_tensor_sha256(name, tensors[name])))
    return digest.hexdigest()


def _validate_authority_header(
    authority_path: str | Path,
    *,
    expected_authority_sha256: str,
) -> tuple[dict[str, object], str, Path]:
    expected = _require_sha256(
        expected_authority_sha256, "exact authority expected SHA-256"
    )
    manifest, observed, source = load_json_object(
        authority_path,
        expected_sha256=expected,
        label="exact marginal responsibility authority",
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
    metadata = manifest.get("metadata")
    frames = manifest.get("frame_indices")
    views = manifest.get("views")
    if set(manifest) != required or (
        manifest.get("schema") != SPARSE_EXACT_MARGINAL_AUTHORITY_SCHEMA
        or manifest.get("schema_version") != 1
        or manifest.get("formula_contract")
        != sparse_exact_marginal_formula_contract()
        or manifest.get("formula_sha256")
        != SPARSE_EXACT_MARGINAL_FORMULA_SHA256
        or not isinstance(metadata, Mapping)
        or not isinstance(frames, list)
        or not frames
        or len({int(value) for value in frames}) != len(frames)
        or not isinstance(views, list)
        or len(views) != len(frames)
        or int(manifest.get("num_gaussians", 0)) <= 0
        or int(manifest.get("num_pixels", 0)) <= 0
        or int(manifest.get("total_hits", -1)) < 0
    ):
        raise ValueError("exact marginal authority header contract differs")
    if any(
        metadata.get(name) is not False
        for name in (
            "benchmark_images_opened",
            "benchmark_masks_opened",
            "text_queries_opened",
        )
    ) or metadata.get("query_independent") is not True:
        raise ValueError("exact marginal authority is not source-only/query-free")
    return manifest, observed, source


def _load_authority_view(
    authority_path: Path,
    record: object,
    *,
    expected_view_index: int,
    expected_frame_index: int,
    num_gaussians: int,
    num_pixels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    required_record = {
        "view_index",
        "frame_index",
        "relative_path",
        "sha256",
        "num_hits",
    }
    if not isinstance(record, Mapping) or set(record) != required_record:
        raise ValueError("exact marginal authority view record differs")
    if (
        int(record.get("view_index", -1)) != int(expected_view_index)
        or int(record.get("frame_index", -1)) != int(expected_frame_index)
        or _SHA256.fullmatch(str(record.get("sha256", ""))) is None
        or int(record.get("num_hits", -1)) < 0
    ):
        raise ValueError("exact marginal authority view identity differs")
    relative = str(record.get("relative_path", ""))
    shard_path = (authority_path.parent / relative).resolve()
    try:
        shard_path.relative_to(authority_path.parent.resolve())
    except ValueError as error:
        raise ValueError("exact marginal authority shard escapes its directory") from error
    payload, digest, _source = load_torch_mapping(
        shard_path,
        expected_sha256=str(record["sha256"]),
        map_location="cpu",
        label="exact marginal authority view shard",
    )
    required_payload = {
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
    gaussian_ids = payload.get("gaussian_ids")
    pixel_ids = payload.get("pixel_ids")
    base_weights = payload.get("base_weights")
    if set(payload) != required_payload or (
        payload.get("schema") != SPARSE_EXACT_MARGINAL_VIEW_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("formula_sha256")
        != SPARSE_EXACT_MARGINAL_FORMULA_SHA256
        or int(payload.get("view_index", -1)) != int(expected_view_index)
        or int(payload.get("frame_index", -1)) != int(expected_frame_index)
        or int(payload.get("num_gaussians", -1)) != int(num_gaussians)
        or int(payload.get("num_pixels", -1)) != int(num_pixels)
        or not torch.is_tensor(gaussian_ids)
        or gaussian_ids.dtype != torch.int32
        or not torch.is_tensor(pixel_ids)
        or pixel_ids.dtype != torch.int32
        or not torch.is_tensor(base_weights)
        or base_weights.dtype != torch.float32
    ):
        raise ValueError("exact marginal authority view shard contract differs")
    assignment = canonicalize_sparse_marginal_view(
        gaussian_ids,
        pixel_ids,
        base_weights,
        num_gaussians=int(num_gaussians),
        num_pixels=int(num_pixels),
    )
    if (
        not torch.equal(assignment["gaussian_ids"], gaussian_ids)
        or not torch.equal(assignment["pixel_ids"], pixel_ids)
        or not torch.equal(assignment["base_weights"], base_weights)
        or int(gaussian_ids.numel()) != int(record["num_hits"])
    ):
        raise ValueError("exact marginal authority view shard values differ")
    return gaussian_ids.long(), pixel_ids.long(), base_weights.double(), digest


def validate_exact_responsibility_statistics_payload(
    payload: object,
    *,
    expected_authority_sha256: str | None = None,
) -> dict[str, object]:
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "authority",
        "tensors",
        "tensor_sha256",
        "tensor_bundle_sha256",
        "metadata",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("exact responsibility statistics fields differ")
    if (
        payload.get("schema") != EXACT_RESPONSIBILITY_STATISTICS_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("contract") != exact_responsibility_statistics_contract()
        or payload.get("contract_sha256")
        != EXACT_RESPONSIBILITY_STATISTICS_CONTRACT_SHA256
    ):
        raise ValueError("exact responsibility statistics contract differs")
    authority = payload.get("authority")
    tensors = payload.get("tensors")
    tensor_hashes = payload.get("tensor_sha256")
    metadata = payload.get("metadata")
    if (
        not isinstance(authority, Mapping)
        or set(authority)
        != {
            "path",
            "sha256",
            "schema",
            "formula_sha256",
            "metadata_sha256",
            "frame_indices_sha256",
            "num_views",
            "num_gaussians",
            "num_pixels",
            "total_hits",
        }
        or authority.get("schema") != SPARSE_EXACT_MARGINAL_AUTHORITY_SCHEMA
        or authority.get("formula_sha256")
        != SPARSE_EXACT_MARGINAL_FORMULA_SHA256
        or _SHA256.fullmatch(str(authority.get("sha256", ""))) is None
        or _SHA256.fullmatch(str(authority.get("metadata_sha256", ""))) is None
        or _SHA256.fullmatch(str(authority.get("frame_indices_sha256", ""))) is None
        or int(authority.get("num_views", 0)) <= 0
        or int(authority.get("num_gaussians", 0)) <= 0
        or int(authority.get("num_pixels", 0)) <= 0
        or int(authority.get("total_hits", -1)) < 0
    ):
        raise ValueError("exact responsibility statistics authority differs")
    if expected_authority_sha256 is not None and authority.get(
        "sha256"
    ) != _require_sha256(expected_authority_sha256, "expected authority SHA-256"):
        raise ValueError("exact responsibility statistics authority SHA-256 differs")
    if (
        not isinstance(tensors, Mapping)
        or set(tensors) != set(_TENSOR_NAMES)
        or not isinstance(tensor_hashes, Mapping)
        or set(tensor_hashes) != set(_TENSOR_NAMES)
        or not isinstance(metadata, Mapping)
    ):
        raise ValueError("exact responsibility statistics tensor declaration differs")
    num_gaussians = int(authority["num_gaussians"])
    checked: dict[str, torch.Tensor] = {}
    for name in _TENSOR_NAMES:
        value = tensors[name]
        expected_dtype = (
            torch.int64
            if name in {"nonzero_hit_count", "nonzero_view_count"}
            else torch.float64
        )
        if (
            not torch.is_tensor(value)
            or value.device.type != "cpu"
            or value.dtype != expected_dtype
            or value.shape != (num_gaussians,)
            or not bool(torch.isfinite(value).all())
            or bool((value < 0).any())
            or tensor_hashes.get(name) != _tensor_sha256(name, value)
        ):
            raise ValueError(f"exact responsibility statistic {name} differs")
        checked[name] = value.detach().contiguous()
    if payload.get("tensor_bundle_sha256") != _tensor_bundle_sha256(checked):
        raise ValueError("exact responsibility statistics tensor bundle differs")
    visible = checked["visible_mass"]
    semantic = checked["semantic_mass"]
    semantic_sq = checked["semantic_mass_sq"]
    hits = checked["nonzero_hit_count"]
    views = checked["nonzero_view_count"]
    ess = checked["kish_effective_sample_size"]
    supported = visible > 0
    expected_ess = torch.where(
        semantic_sq > 0,
        semantic.square() / semantic_sq,
        torch.zeros_like(semantic),
    )
    tolerance = 1e-12
    if (
        bool((semantic > visible + tolerance).any())
        or not torch.equal(supported, hits > 0)
        or not torch.equal(supported, views > 0)
        or bool((views > int(authority["num_views"])).any())
        or bool((views > hits).any())
        or not torch.equal(ess, expected_ess)
        or bool((ess > hits.double() + tolerance).any())
        or bool((semantic_sq[~supported] != 0).any())
    ):
        raise ValueError("exact responsibility statistics numeric relations differ")
    if (
        set(metadata)
        != {
            "builder_implementation_sha256",
            "source_only",
            "query_independent",
            "benchmark_images_opened",
            "benchmark_masks_opened",
            "text_queries_opened",
            "contains_admission_decision",
        }
        or _SHA256.fullmatch(
            str(metadata.get("builder_implementation_sha256", ""))
        )
        is None
        or metadata.get("source_only") is not True
        or metadata.get("query_independent") is not True
        or metadata.get("benchmark_images_opened") is not False
        or metadata.get("benchmark_masks_opened") is not False
        or metadata.get("text_queries_opened") is not False
        or metadata.get("contains_admission_decision") is not False
    ):
        raise ValueError("exact responsibility statistics safety metadata differs")
    return dict(payload)


def build_exact_responsibility_statistics(
    *,
    authority_path: str | Path,
    expected_authority_sha256: str,
    output_path: str | Path,
) -> tuple[Path, str]:
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable statistics sidecar already exists: {output}")
    manifest, authority_sha256, authority_source = _validate_authority_header(
        authority_path,
        expected_authority_sha256=expected_authority_sha256,
    )
    num_gaussians = int(manifest["num_gaussians"])
    num_pixels = int(manifest["num_pixels"])
    frame_indices = [int(value) for value in manifest["frame_indices"]]
    tensors: dict[str, torch.Tensor] = {
        "visible_mass": torch.zeros(num_gaussians, dtype=torch.float64),
        "semantic_mass": torch.zeros(num_gaussians, dtype=torch.float64),
        "semantic_mass_sq": torch.zeros(num_gaussians, dtype=torch.float64),
        "nonzero_hit_count": torch.zeros(num_gaussians, dtype=torch.int64),
        "nonzero_view_count": torch.zeros(num_gaussians, dtype=torch.int64),
        "kish_effective_sample_size": torch.zeros(
            num_gaussians, dtype=torch.float64
        ),
    }
    observed_hits = 0
    for view_index, (frame_index, record) in enumerate(
        zip(frame_indices, manifest["views"])
    ):
        gaussian_ids, pixel_ids, base_weights, _view_sha256 = _load_authority_view(
            authority_source,
            record,
            expected_view_index=view_index,
            expected_frame_index=frame_index,
            num_gaussians=num_gaussians,
            num_pixels=num_pixels,
        )
        observed_hits += int(gaussian_ids.numel())
        if gaussian_ids.numel() == 0:
            continue
        pixel_mass = torch.zeros(num_pixels, dtype=torch.float64)
        pixel_mass.index_add_(0, pixel_ids, base_weights)
        if not bool(torch.isfinite(pixel_mass).all()) or bool(
            (pixel_mass[pixel_ids] <= 0).any()
        ):
            raise ValueError("exact authority view produced invalid pixel mass")
        semantic_hit_weight = base_weights.square() / pixel_mass[pixel_ids]
        if (
            not bool(torch.isfinite(semantic_hit_weight).all())
            or bool((semantic_hit_weight <= 0).any())
            or bool((semantic_hit_weight > base_weights + 1e-12).any())
        ):
            raise ValueError("exact authority view produced invalid semantic mass")
        tensors["visible_mass"].index_add_(0, gaussian_ids, base_weights)
        tensors["semantic_mass"].index_add_(
            0, gaussian_ids, semantic_hit_weight
        )
        tensors["semantic_mass_sq"].index_add_(
            0, gaussian_ids, semantic_hit_weight.square()
        )
        tensors["nonzero_hit_count"].index_add_(
            0, gaussian_ids, torch.ones_like(gaussian_ids, dtype=torch.int64)
        )
        frame_support = torch.zeros(num_gaussians, dtype=torch.bool)
        frame_support[gaussian_ids] = True
        tensors["nonzero_view_count"][frame_support] += 1
    if observed_hits != int(manifest["total_hits"]):
        raise ValueError("exact authority total hit count differs while streaming")
    semantic_sq = tensors["semantic_mass_sq"]
    semantic = tensors["semantic_mass"]
    tensors["kish_effective_sample_size"] = torch.where(
        semantic_sq > 0,
        semantic.square() / semantic_sq,
        torch.zeros_like(semantic),
    )
    tensor_hashes = {
        name: _tensor_sha256(name, tensors[name]) for name in _TENSOR_NAMES
    }
    payload: dict[str, object] = {
        "schema": EXACT_RESPONSIBILITY_STATISTICS_SCHEMA,
        "schema_version": 1,
        "contract": exact_responsibility_statistics_contract(),
        "contract_sha256": EXACT_RESPONSIBILITY_STATISTICS_CONTRACT_SHA256,
        "authority": {
            "path": str(authority_source),
            "sha256": authority_sha256,
            "schema": SPARSE_EXACT_MARGINAL_AUTHORITY_SCHEMA,
            "formula_sha256": SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
            "metadata_sha256": _canonical_json_sha256(manifest["metadata"]),
            "frame_indices_sha256": _canonical_json_sha256(frame_indices),
            "num_views": len(frame_indices),
            "num_gaussians": num_gaussians,
            "num_pixels": num_pixels,
            "total_hits": int(manifest["total_hits"]),
        },
        "tensors": tensors,
        "tensor_sha256": tensor_hashes,
        "tensor_bundle_sha256": _tensor_bundle_sha256(tensors),
        "metadata": {
            "builder_implementation_sha256": sha256_file(Path(__file__).resolve()),
            "source_only": True,
            "query_independent": True,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "contains_admission_decision": False,
        },
    }
    validate_exact_responsibility_statistics_payload(
        payload,
        expected_authority_sha256=authority_sha256,
    )
    written = write_torch_noclobber(output, payload)
    output_sha256 = sha256_file(written)
    persisted, observed_sha256, _source = load_torch_mapping(
        written,
        expected_sha256=output_sha256,
        map_location="cpu",
        label="exact responsibility statistics sidecar",
    )
    validate_exact_responsibility_statistics_payload(
        persisted,
        expected_authority_sha256=authority_sha256,
    )
    if observed_sha256 != output_sha256:
        raise RuntimeError("statistics sidecar digest changed after publication")
    return written, output_sha256


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output, digest = build_exact_responsibility_statistics(
        authority_path=args.authority,
        expected_authority_sha256=args.expected_authority_sha256,
        output_path=args.output,
    )
    print(json.dumps({"output": str(output), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
