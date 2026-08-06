#!/usr/bin/env python3
"""Pack exact pre-projection DINOv3/SAM3 MPR teachers for query compilation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from radio_gs.interfaces.capability_cache import load_canonical_capability_bank
from radio_gs.interfaces.capability_projection_contract import (
    FORMAL_PROJECTION_CONTRACT,
)
from radio_gs.interfaces.primitive_row_authority import PrimitiveRowAuthority
from radio_gs.training.tensor_cache_io import ShardedMPRCache, load_mpr_cache


EXACT_CAPABILITY_MPR_SOURCE = (
    "exact_capability_mpr_official_frozen_capability_views"
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair_sha256(dino_sha256: str, sam3_sha256: str) -> str:
    value = f"dino_v3:{dino_sha256}\nsam3:{sam3_sha256}\n"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fetch_rows(
    mpr: Mapping[str, object] | ShardedMPRCache,
    rows: torch.Tensor,
) -> torch.Tensor:
    if isinstance(mpr, ShardedMPRCache):
        return mpr.fetch_rows(rows)
    features = mpr.get("features")
    if not torch.is_tensor(features):
        raise ValueError("capability MPR lacks features")
    return features[rows]


def _check_query_independent(metadata: object, *, space: str) -> Mapping[str, object]:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{space} MPR metadata must be a mapping")
    if any(
        metadata.get(key) is not False
        for key in (
            "benchmark_images_opened",
            "benchmark_masks_opened",
            "text_queries_opened",
        )
    ):
        raise ValueError(f"{space} MPR is not query independent")
    if metadata.get("feature_space") != space:
        raise ValueError(f"{space} MPR feature-space declaration differs")
    if metadata.get("shared_registration_responsibility") is not True:
        raise ValueError(f"{space} MPR lacks shared registration authority")
    if metadata.get("capability_projection_before_mpr") is not True:
        raise ValueError(f"{space} MPR did not project official views before MPR")
    lifting_contract = metadata.get("observation_lifting_contract")
    if not isinstance(lifting_contract, Mapping) or (
        lifting_contract.get("name") != "canonical-mpr-v1"
        or lifting_contract.get("feature_projection_order")
        != "per_view_before_mpr"
        or lifting_contract.get("query_independent") is not True
    ):
        raise ValueError(f"{space} MPR projection-order contract differs")
    return metadata


def _normalized_compact_rows(
    mpr: Mapping[str, object] | ShardedMPRCache,
    rows: torch.Tensor,
    *,
    feature_dim: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    output = torch.empty((rows.numel(), int(feature_dim)), dtype=torch.float16)
    for start in range(0, rows.numel(), int(batch_size)):
        selected = rows[start : start + int(batch_size)]
        values = _fetch_rows(mpr, selected).to(device=device, dtype=torch.float32)
        values = F.normalize(values, dim=-1, eps=1e-8)
        output[start : start + selected.numel()].copy_(values.half().cpu())
    return output


@torch.inference_mode()
def build(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    dino_path = Path(args.dino_mpr).expanduser().resolve()
    sam3_path = Path(args.sam3_mpr).expanduser().resolve()
    dino_sha256 = _sha256_file(dino_path)
    sam3_sha256 = _sha256_file(sam3_path)
    pair_sha256 = _pair_sha256(dino_sha256, sam3_sha256)

    reference_path = Path(args.reference_compact_capability).expanduser().resolve()
    reference_sha256 = _sha256_file(reference_path)
    reference = load_canonical_capability_bank(
        reference_path,
        require_formal_projection_order=True,
        legacy_projection_authority=str(
            getattr(args, "reference_projection_authority", "")
        ),
    )

    dino, _, _ = load_mpr_cache(
        dino_path,
        expected_sha256=dino_sha256,
        expected_feature_space="dino_v3",
        require_reliability=True,
        require_formal_safety=False,
    )
    dino_metadata = _check_query_independent(dino.get("metadata"), space="dino_v3")
    xyz = torch.as_tensor(dino["xyz"]).float().cpu()
    valid = torch.as_tensor(dino["valid"]).bool().cpu()
    if not torch.equal(reference.xyz, xyz) or not torch.equal(reference.valid, valid):
        raise ValueError("DINO MPR and compact capability row authority differ")
    rows = torch.where(valid)[0]
    appearance = _normalized_compact_rows(
        dino,
        rows,
        feature_dim=int(reference.appearance.shape[1]),
        batch_size=batch_size,
        device=device,
    )
    dino_responsibility = str(
        dino_metadata.get("registration_responsibility_cache_sha256", "")
    )
    del dino
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    sam3, _, _ = load_mpr_cache(
        sam3_path,
        expected_sha256=sam3_sha256,
        expected_feature_space="sam3",
        require_reliability=True,
        require_formal_safety=False,
    )
    sam3_metadata = _check_query_independent(sam3.get("metadata"), space="sam3")
    if not torch.equal(torch.as_tensor(sam3["xyz"]).float().cpu(), xyz) or not torch.equal(
        torch.as_tensor(sam3["valid"]).bool().cpu(), valid
    ):
        raise ValueError("DINO and SAM3 MPR row authorities differ")
    if str(sam3_metadata.get("registration_responsibility_cache_sha256", "")) != (
        dino_responsibility
    ):
        raise ValueError("DINO and SAM3 MPR registration authorities differ")
    boundary = _normalized_compact_rows(
        sam3,
        rows,
        feature_dim=int(reference.boundary.shape[1]),
        batch_size=batch_size,
        device=device,
    )
    del sam3
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    signatures = {
        name: replace(signature, field_checkpoint_sha256=pair_sha256)
        for name, signature in reference.signatures.items()
    }
    metadata: dict[str, object] = {
        "schema_version": 1,
        "source": EXACT_CAPABILITY_MPR_SOURCE,
        "dino_v3_mpr": str(dino_path),
        "dino_v3_mpr_sha256": dino_sha256,
        "sam3_mpr": str(sam3_path),
        "sam3_mpr_sha256": sam3_sha256,
        "exact_capability_mpr_pair_sha256": pair_sha256,
        "field_checkpoint_sha256": pair_sha256,
        "reference_compact_capability": str(reference_path),
        "reference_compact_capability_sha256": reference_sha256,
        "radio_checkpoint": str(reference.metadata.get("radio_checkpoint", "")),
        "radio_checkpoint_sha256": str(
            reference.metadata.get("radio_checkpoint_sha256", "")
        ),
        "custom_adaptor_head": False,
        "query_independent": True,
        "feature_storage": "valid_rows_compact_v1",
        "feature_row_order": "torch_where_valid_ascending",
        "feature_row_count": int(rows.numel()),
        "primitive_row_authority": PrimitiveRowAuthority.from_tensors(
            xyz, valid
        ).to_dict(),
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "capability_signatures": {
            name: signature.to_dict() for name, signature in signatures.items()
        },
        "projection_contract": {
            "contract": FORMAL_PROJECTION_CONTRACT,
            "eligibility": "formal_exact_teacher",
            "artifact_role": "exact_capability_mpr_teacher",
            "operator": "per-view official adaptor, shared MPR, fp32 row L2",
            "projection_order": "official_adaptor_before_mpr",
            "nonlinear_adaptor_after_raw_mpr": False,
            "query_dependent": False,
        },
        "registration_responsibility_cache_sha256": dino_responsibility,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "xyz": xyz,
            "valid": valid,
            "appearance_dino_v3": appearance,
            "boundary_sam3": boundary,
            "metadata": metadata,
        },
        output,
    )
    report = {
        **metadata,
        "output": str(output),
        "output_sha256": _sha256_file(output),
        "num_gaussians": int(xyz.shape[0]),
        "valid_gaussians": int(valid.sum()),
        "appearance_dim": int(appearance.shape[1]),
        "boundary_dim": int(boundary.shape[1]),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-mpr", required=True)
    parser.add_argument("--sam3-mpr", required=True)
    parser.add_argument("--reference-compact-capability", required=True)
    parser.add_argument(
        "--reference-projection-authority",
        default=(
            "paper/artifacts/"
            "formal_capability_projection_lineage_closure_20260805.json"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    print(json.dumps(build(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
