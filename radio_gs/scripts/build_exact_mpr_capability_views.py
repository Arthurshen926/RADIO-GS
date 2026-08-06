#!/usr/bin/env python3
"""Project exact raw-MPR rows into the frozen official capability spaces.

This is a diagnostic teacher-side counterpart to
``build_canonical_capability_views.py``.  The latter decodes a compact field;
this script changes only that source tensor and applies the same frozen
DINOv3/SAM3 adaptors and normalization.  It never opens a prompt, query,
benchmark image, or benchmark mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from radio_gs.field import FeatureSpaceSignature
from radio_gs.interfaces.capability_cache import load_canonical_capability_bank
from radio_gs.interfaces.capability_projection_contract import (
    RAW_MPR_DIAGNOSTIC_CONTRACT,
)
from radio_gs.interfaces.primitive_row_authority import PrimitiveRowAuthority
from radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint
from radio_gs.training.tensor_cache_io import ShardedMPRCache, load_mpr_cache


EXACT_MPR_CAPABILITY_SOURCE = "exact_radio_mpr_official_frozen_capability_views"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_rows(
    mpr: Mapping[str, object] | ShardedMPRCache,
    rows: torch.Tensor,
) -> torch.Tensor:
    if isinstance(mpr, ShardedMPRCache):
        return mpr.fetch_rows(rows)
    values = mpr.get("features")
    if not torch.is_tensor(values):
        raise ValueError("exact raw MPR lacks a feature matrix")
    return values[rows]


def _exact_signature(
    reference: FeatureSpaceSignature,
    *,
    exact_mpr_sha256: str,
) -> FeatureSpaceSignature:
    # The output-space identity (official checkpoint/adaptor/dimension) is
    # inherited from the frozen compact cache.  ``field_checkpoint_sha256``
    # is deliberately replaced by the exact source digest rather than falsely
    # claiming that these rows were decoded from that field.
    return replace(reference, field_checkpoint_sha256=exact_mpr_sha256)


@torch.inference_mode()
def build(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    raw_mpr_path = Path(args.raw_mpr).expanduser().resolve()
    exact_mpr_sha256 = _sha256_file(raw_mpr_path)
    if (
        str(args.expected_raw_mpr_sha256).strip()
        and exact_mpr_sha256 != str(args.expected_raw_mpr_sha256).strip()
    ):
        raise ValueError("exact raw MPR SHA-256 differs")
    mpr, loaded_mpr_sha256, loaded_mpr_path = load_mpr_cache(
        raw_mpr_path,
        expected_sha256=exact_mpr_sha256,
        expected_feature_space="radio",
        require_reliability=True,
        # The frozen July NVOS assets predate the later feature-output-bundle
        # receipt.  They are still SHA-bound below to the accepted compact
        # cache and are independently checked for the original canonical-MPR
        # query/benchmark safety declarations.
        require_formal_safety=False,
    )
    if loaded_mpr_sha256 != exact_mpr_sha256:
        raise ValueError("validated exact raw MPR digest differs")
    xyz = torch.as_tensor(mpr["xyz"]).float().cpu()
    valid = torch.as_tensor(mpr["valid"]).bool().cpu()
    rows = torch.where(valid)[0]

    reference_path = Path(args.reference_compact_capability).expanduser().resolve()
    reference_sha256 = _sha256_file(reference_path)
    if (
        str(args.expected_reference_compact_capability_sha256).strip()
        and reference_sha256
        != str(args.expected_reference_compact_capability_sha256).strip()
    ):
        raise ValueError("reference compact capability SHA-256 differs")
    reference = load_canonical_capability_bank(
        reference_path,
        # July NVOS compact caches predate the explicit row-authority field.
        # Exact tensor equality of xyz and valid below is the legacy authority
        # bridge; the newly written diagnostic cache publishes the modern one.
        require_row_authority=False,
        require_formal_projection_order=True,
        legacy_projection_authority=str(
            getattr(args, "reference_projection_authority", "")
        ),
    )
    if not torch.equal(reference.valid, valid) or not torch.equal(reference.xyz, xyz):
        raise ValueError("exact MPR and compact capability row authority differ")

    radio_checkpoint = Path(args.radio_checkpoint).expanduser().resolve()
    radio_checkpoint_sha256 = _sha256_file(radio_checkpoint)
    if str(reference.metadata.get("radio_checkpoint_sha256", "")) != (
        radio_checkpoint_sha256
    ):
        raise ValueError("reference capability uses another RADIO checkpoint")
    adaptors = {
        "appearance_dino_v3": load_radio_adaptor_from_checkpoint(
            str(radio_checkpoint), "dino_v3_7b", kind="feature_projection"
        ).to(device).eval(),
        "boundary_sam3": load_radio_adaptor_from_checkpoint(
            str(radio_checkpoint), "sam3", kind="feature_projection"
        ).to(device).eval(),
    }
    for module in adaptors.values():
        module.requires_grad_(False)

    outputs = {
        name: torch.empty((rows.numel(), module.output_dim), dtype=torch.float16)
        for name, module in adaptors.items()
    }
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, rows.numel(), batch_size):
        selected_cpu = rows[start : start + batch_size]
        exact_radio = _feature_rows(mpr, selected_cpu).to(
            device=device, dtype=torch.float32
        )
        for name, adaptor in adaptors.items():
            projected = F.normalize(adaptor(exact_radio).float(), dim=-1, eps=1e-8)
            outputs[name][start : start + selected_cpu.numel()].copy_(
                projected.half().cpu()
            )

    reference_signatures = reference.signatures
    if set(reference_signatures) != {"appearance", "boundary"}:
        raise ValueError("reference compact capability signatures are incomplete")
    signatures = {
        name: _exact_signature(signature, exact_mpr_sha256=exact_mpr_sha256)
        for name, signature in reference_signatures.items()
    }
    raw_metadata = mpr.get("metadata", {})
    if not isinstance(raw_metadata, Mapping):
        raise ValueError("exact raw MPR metadata must be a mapping")
    forbidden_true = [
        key
        for key in (
            "benchmark_images_opened",
            "benchmark_masks_opened",
            "text_queries_opened",
        )
        if raw_metadata.get(key) is not False
    ]
    if forbidden_true:
        raise ValueError(
            "exact raw MPR is not query independent: " + ", ".join(forbidden_true)
        )
    lifting_contract = raw_metadata.get("observation_lifting_contract", {})
    if not isinstance(lifting_contract, Mapping) or (
        lifting_contract.get("name") != "canonical-mpr-v1"
        or lifting_contract.get("query_independent") is not True
    ):
        raise ValueError("exact raw MPR lacks the frozen query-independent contract")

    metadata: dict[str, object] = {
        "schema_version": 1,
        "source": EXACT_MPR_CAPABILITY_SOURCE,
        "exact_raw_mpr": str(loaded_mpr_path),
        "exact_raw_mpr_sha256": exact_mpr_sha256,
        "reference_compact_capability": str(reference_path),
        "reference_compact_capability_sha256": reference_sha256,
        "reference_compact_field_checkpoint": str(
            reference.metadata.get("field_checkpoint", "")
        ),
        "reference_compact_field_checkpoint_sha256": str(
            reference.metadata.get("field_checkpoint_sha256", "")
        ),
        "field_checkpoint_sha256": exact_mpr_sha256,
        "radio_checkpoint": str(radio_checkpoint),
        "radio_checkpoint_sha256": radio_checkpoint_sha256,
        "appearance_view": "official C-RADIOv4 dino_v3_7b feature_projection",
        "boundary_view": "official C-RADIOv4 sam3 feature_projection",
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
            "contract": RAW_MPR_DIAGNOSTIC_CONTRACT,
            "eligibility": "diagnostic_only",
            "source": "exact valid rows of raw RADIO MPR",
            "operator": "frozen official adaptor then fp32 L2 normalization",
            "projection_order": "raw_radio_mpr_then_official_adaptor",
            "nonlinear_adaptor_after_mpr": True,
            "comparison": "compact field decode then identical adaptor and normalization",
            "query_dependent": False,
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "xyz": xyz,
            "valid": valid,
            **outputs,
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
        "appearance_dim": int(outputs["appearance_dino_v3"].shape[1]),
        "boundary_dim": int(outputs["boundary_sam3"].shape[1]),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-mpr", required=True)
    parser.add_argument("--expected-raw-mpr-sha256", default="")
    parser.add_argument("--reference-compact-capability", required=True)
    parser.add_argument(
        "--expected-reference-compact-capability-sha256", default=""
    )
    parser.add_argument(
        "--reference-projection-authority",
        default=(
            "paper/artifacts/"
            "formal_capability_projection_lineage_closure_20260805.json"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    print(json.dumps(build(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
