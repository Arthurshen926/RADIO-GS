#!/usr/bin/env python3
"""Derive frozen official DINOv3/SAM3 primitive views from one canonical field."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from radio_gs.field import (
    FeatureSpaceSignature,
    load_canonical_field_checkpoint,
)
from radio_gs.field.factorized_radio_contract import (
    CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME,
)
from radio_gs.interfaces.factorized_primitive_state import (
    FACTORIZED_FIELD_CHECKPOINT_SCHEMA_V2,
    load_factorized_field_support,
)
from radio_gs.interfaces.capability_projection_contract import (
    FORMAL_PROJECTION_CONTRACT,
    FORMAL_TARGET_MODE_TO_CONTRACT,
    FORMAL_TARGET_MODES,
)
from radio_gs.interfaces.primitive_row_authority import PrimitiveRowAuthority
from radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint
from radio_gs.training.tensor_cache_io import load_mpr_cache
from radio_gs.utils.immutable_artifacts import (
    write_frozen_json,
    write_torch_noclobber,
)


CANONICAL_FIELD_CHECKPOINT_SCHEMA_V1 = "canonical-v1"


def _require_sha256(value: object, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} is not SHA-256")
    return digest


def _verified_file_record(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} authority must be a mapping")
    path = Path(str(value.get("path", ""))).expanduser().resolve()
    expected = _require_sha256(value.get("sha256"), label=f"{label} SHA-256")
    if not path.is_file() or _sha256_file(path) != expected:
        raise ValueError(f"{label} file is missing or differs")
    return {"path": str(path), "sha256": expected, "size_bytes": path.stat().st_size}


def _load_verified_json_record(
    record: Mapping[str, object], *, label: str
) -> dict[str, object]:
    """Load the JSON content behind an already hash-verified file record."""

    path = Path(str(record["path"]))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not a valid JSON authority") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} content must be a mapping")
    return dict(value)


def _validate_formal_feature_bundle_cohort(
    *,
    cohort_record: Mapping[str, object],
    expected_cache_records: Mapping[str, Mapping[str, object]],
    expected_feature_bundle_sha256: str,
    expected_responsibility_sha256: str,
    expected_radio_checkpoint_sha256: str,
    payload: Mapping[str, object],
) -> None:
    """Validate the older formal bundle receipt without weakening its seal.

    This is the producer schema emitted by the matched-top1 formal cohort.  It
    stores the receipt under ``legacy_receipt_correction`` for CLI backwards
    compatibility, but the receipt itself is a formal source-only authority.
    Every cache record was re-hashed before this function and must agree with
    the corresponding record inside the sealed JSON authority.
    """

    cohort = _load_verified_json_record(
        cohort_record, label="formal feature-bundle cohort authority"
    )
    target_access = cohort.get("target_access")
    storage = cohort.get("storage_authority")
    lineage = cohort.get("shared_lineage")
    geometry = payload.get("geometry_fingerprint")
    if (
        cohort.get("schema_version") != 1
        or cohort.get("artifact_type") != "factorized_capability_cohort_authority"
        or cohort.get("experiment")
        != "canonical-factorized-radio-v1-formal-capability-cohort"
        or cohort.get("feature_output_bundle_sha256")
        != expected_feature_bundle_sha256
        or not isinstance(target_access, Mapping)
        or any(
            target_access.get(name) is not False
            for name in (
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "text_queries_opened",
                "target_metrics_used_for_selection",
            )
        )
        or not isinstance(storage, Mapping)
        or storage.get("schema") != "radio_gs.channel_sharded_mpr.v1"
        or storage.get("formal_loader_all_hash_gates") != "passed"
        or int(storage.get("shard_channels", 0)) <= 0
        or not isinstance(lineage, Mapping)
        or lineage.get("registration_responsibility_cache_sha256")
        != expected_responsibility_sha256
        or lineage.get("official_radio_checkpoint_sha256")
        != expected_radio_checkpoint_sha256
        or not isinstance(geometry, Mapping)
        or int(lineage.get("num_gaussians", -1))
        != int(geometry.get("num_gaussians", -2))
        or not 0 < int(lineage.get("valid_count", 0)) <= int(
            geometry.get("num_gaussians", 0)
        )
        or len(str(lineage.get("geometry_checkpoint_sha256", ""))) != 64
    ):
        raise ValueError("formal feature-bundle cohort authority differs")

    authorities = cohort.get("frozen_cache_authorities")
    if not isinstance(authorities, Mapping) or set(authorities) != set(
        expected_cache_records
    ):
        raise ValueError("formal feature-bundle cache authorities differ")
    for name, expected in expected_cache_records.items():
        authority = authorities.get(name)
        if not isinstance(authority, Mapping) or set(authority) != {
            "path",
            "sha256",
        }:
            raise ValueError(f"formal feature-bundle {name} authority differs")
        actual_path = Path(str(authority.get("path", ""))).expanduser().resolve()
        if (
            actual_path != Path(str(expected["path"])).resolve()
            or authority.get("sha256") != expected["sha256"]
        ):
            raise ValueError(f"formal feature-bundle {name} authority differs")


def _formal_capability_training_authority(
    payload: Mapping[str, object],
    *,
    expected_radio_checkpoint_sha256: str,
    expected_factorized_radio_cache_sha256: str,
) -> dict[str, object]:
    """Reopen and preserve every exact source behind a schema-v2 bank.

    A field checkpoint cryptographically commits to its training sources, but
    older capability-bank metadata retained only their projection-order labels.
    Preserve the exact DINO/SAM MPR files and capability cohort here so a graph
    or downstream authority can validate the complete source-only chain without
    trusting an out-of-band training receipt.
    """

    targets = payload.get("capability_mpr_targets")
    reference = payload.get("capability_observation_reference")
    if not isinstance(targets, Mapping) or not isinstance(reference, Mapping):
        raise ValueError("factorized field lacks formal capability source authority")
    target_mode = str(payload.get("capability_target_mode", ""))
    target_contract = str(payload.get("capability_target_contract", ""))
    cohort_mode = str(reference.get("capability_cohort_authority_mode", ""))
    if cohort_mode == "formal_exact_marginal_v1":
        if (
            not isinstance(reference.get("capability_cohort_authority"), Mapping)
            or "legacy_receipt_correction" in reference
        ):
            raise ValueError("exact-marginal capability cohort authority is mixed")
        cohort_key = "capability_cohort_authority"
    elif cohort_mode == "formal_feature_bundle_v1":
        if (
            not isinstance(reference.get("legacy_receipt_correction"), Mapping)
            or "capability_cohort_authority" in reference
        ):
            raise ValueError("feature-bundle capability cohort authority is mixed")
        cohort_key = "legacy_receipt_correction"
    else:
        raise ValueError("unsupported capability cohort authority mode")
    responsibility_sha256 = _require_sha256(
        reference.get("registration_responsibility_cache_sha256"),
        label="capability responsibility SHA-256",
    )
    feature_bundle_sha256 = _require_sha256(
        reference.get("feature_output_bundle_sha256"),
        label="capability feature bundle SHA-256",
    )
    exact_sources: dict[str, object] = {}
    for target_name, role in (("dino_v3", "appearance"), ("sam3", "boundary")):
        raw = targets.get(target_name)
        record = _verified_file_record(raw, label=f"exact {target_name} MPR")
        assert isinstance(raw, Mapping)
        if (
            raw.get("feature_space") != target_name
            or raw.get("projection_order") != target_mode
            or raw.get("target_contract") != target_contract
            or raw.get("uses_query_or_benchmark_supervision") is not False
            or raw.get("official_adaptor_checkpoint_sha256")
            != expected_radio_checkpoint_sha256
            or raw.get("registration_responsibility_cache_sha256")
            != responsibility_sha256
            or raw.get("feature_output_bundle_sha256") != feature_bundle_sha256
            or (
                cohort_mode == "formal_feature_bundle_v1"
                and (
                    raw.get("formal_feature_bundle_authority") is not True
                    or raw.get("historical_feature_bundle_receipt_compatibility")
                    is not False
                )
            )
        ):
            raise ValueError(f"exact {target_name} capability authority differs")
        exact_sources[role] = {
            **record,
            "feature_space": target_name,
            "projection_order": target_mode,
            "target_contract": target_contract,
        }
    reference_record = _verified_file_record(
        reference, label="exact raw capability observation reference"
    )
    cohort_record = _verified_file_record(
        reference.get(cohort_key),
        label="capability cohort authority",
    )
    if reference.get("sha256") == expected_factorized_radio_cache_sha256:
        # The reference is normally a separate matched exact-marginal cache;
        # this branch merely rejects accidentally relabelling the raw support
        # cache as that authority.
        raise ValueError("capability observation reference aliases raw support")
    if (
        reference.get("uses_query_or_benchmark_supervision") is not False
        or reference.get("feature_output_bundle_sha256") != feature_bundle_sha256
        or reference.get("registration_responsibility_cache_sha256")
        != responsibility_sha256
        or (
            cohort_mode == "formal_feature_bundle_v1"
            and (
                reference.get("formal_feature_bundle_authority") is not True
                or reference.get("historical_feature_bundle_receipt_compatibility")
                is not False
            )
        )
    ):
        raise ValueError("capability observation reference authority differs")
    if cohort_mode == "formal_feature_bundle_v1":
        _validate_formal_feature_bundle_cohort(
            cohort_record=cohort_record,
            expected_cache_records={
                "radio": reference_record,
                "dino_v3": exact_sources["appearance"],
                "sam3": exact_sources["boundary"],
            },
            expected_feature_bundle_sha256=feature_bundle_sha256,
            expected_responsibility_sha256=responsibility_sha256,
            expected_radio_checkpoint_sha256=expected_radio_checkpoint_sha256,
            payload=payload,
        )
    return {
        "schema_version": 1,
        "source": (
            "formal_exact_marginal_capability_training_authority_v1"
            if cohort_mode == "formal_exact_marginal_v1"
            else "formal_feature_bundle_capability_training_authority_v1"
        ),
        "capability_cohort_authority_mode": cohort_mode,
        "target_mode": target_mode,
        "target_contract": target_contract,
        "exact_source_capabilities": exact_sources,
        "exact_raw_observation_reference": reference_record,
        "capability_cohort_authority": cohort_record,
        "registration_responsibility_cache_sha256": responsibility_sha256,
        "feature_output_bundle_sha256": feature_bundle_sha256,
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_compatible_legacy_observation(metadata: dict) -> None:
    """Accept only the audited query-free completed-MPR legacy contract."""
    if metadata.get("construction") != (
        "dominant_primary_with_query_free_support_completion"
    ):
        raise ValueError("compatible-legacy capability MPR construction differs")
    contaminated = [
        key
        for key in (
            "benchmark_images_opened",
            "benchmark_masks_opened",
            "text_queries_opened",
        )
        if metadata.get(key) is not False
    ]
    if contaminated:
        raise ValueError(
            f"compatible-legacy capability MPR is not query-independent: {contaminated}"
        )


def _formal_projection_contract(payload: dict) -> dict[str, object]:
    """Return the formal compact-field lineage or fail before GPU projection."""

    target_mode = str(payload.get("capability_target_mode", ""))
    target_contract = str(payload.get("capability_target_contract", ""))
    if target_mode not in FORMAL_TARGET_MODES:
        raise ValueError(
            "canonical field is not supervised by an adaptor-before-MPR teacher"
        )
    if target_contract != FORMAL_TARGET_MODE_TO_CONTRACT[target_mode]:
        raise ValueError("canonical field capability target mode and contract differ")
    raw_targets = payload.get("capability_mpr_targets")
    if not isinstance(raw_targets, dict):
        raise ValueError("canonical field capability MPR provenance must be a mapping")
    teacher_projection_orders: dict[str, str] = {}
    for target_name, output_name in (
        ("dino_v3", "appearance"),
        ("sam3", "boundary"),
    ):
        target = raw_targets.get(target_name)
        if not isinstance(target, dict):
            raise ValueError(
                f"canonical field {target_name} capability provenance is absent"
            )
        projection_order = str(target.get("projection_order", ""))
        if projection_order != target_mode:
            raise ValueError(
                f"canonical field {target_name} projection order differs from "
                "the declared target mode"
            )
        if target.get("uses_query_or_benchmark_supervision") is not False:
            raise ValueError(
                f"canonical field {target_name} target is not query independent"
            )
        teacher_projection_orders[output_name] = projection_order
    return {
        "schema_version": 1,
        "contract": FORMAL_PROJECTION_CONTRACT,
        "eligibility": "formal_one_field",
        "artifact_role": "capability_supervised_compact_field",
        "field_output_projection_order": ("compact_radio_field_then_official_adaptor"),
        "capability_target_mode": target_mode,
        "capability_target_contract": target_contract,
        "teacher_projection_orders": teacher_projection_orders,
        "nonlinear_adaptor_after_raw_mpr": False,
        "query_independent": True,
    }


def _load_field_and_support(
    args: argparse.Namespace,
    *,
    field_checkpoint_sha256: str,
) -> tuple[
    torch.nn.Module,
    dict,
    torch.Tensor,
    torch.Tensor,
    str,
    Path,
    str,
    dict[str, object],
]:
    """Load one explicitly selected field schema and its exact source support.

    The two checkpoint schemas deliberately have no automatic fallback.  This
    keeps a malformed factorized checkpoint from being reinterpreted as the
    historical canonical field, and keeps legacy schema-v1 behavior unchanged.
    """

    field_schema = str(
        getattr(
            args,
            "field_checkpoint_schema",
            CANONICAL_FIELD_CHECKPOINT_SCHEMA_V1,
        )
    )
    observation_contract = str(getattr(args, "observation_contract", "canonical"))
    if field_schema == CANONICAL_FIELD_CHECKPOINT_SCHEMA_V1:
        field, raw_payload = load_canonical_field_checkpoint(
            args.field_checkpoint,
            map_location="cpu",
            expected_sha256=field_checkpoint_sha256,
        )
        payload = dict(raw_payload)
        mpr_path = Path(args.mpr_cache or payload["mpr_cache"])
        expected_mpr_sha256 = str(
            getattr(args, "expected_mpr_cache_sha256", "")
        ) or str(payload.get("mpr_cache_sha256", ""))
        payload_mpr_sha256 = str(payload.get("mpr_cache_sha256", ""))
        if not expected_mpr_sha256 or (
            payload_mpr_sha256 and expected_mpr_sha256 != payload_mpr_sha256
        ):
            raise ValueError("capability MPR SHA-256 is absent or differs from field")
        mpr, actual_mpr_sha256, resolved_mpr_path = load_mpr_cache(
            mpr_path,
            expected_sha256=expected_mpr_sha256,
            expected_feature_space="radio",
            require_reliability=True,
            require_formal_safety=observation_contract == "canonical",
        )
        if observation_contract == "compatible-legacy":
            _validate_compatible_legacy_observation(dict(mpr.get("metadata", {})))
        xyz = torch.as_tensor(mpr["xyz"]).float().cpu()
        valid = torch.as_tensor(mpr["valid"]).bool().cpu()
        return (
            field,
            payload,
            xyz,
            valid,
            actual_mpr_sha256,
            resolved_mpr_path,
            observation_contract,
            {},
        )

    if field_schema != FACTORIZED_FIELD_CHECKPOINT_SCHEMA_V2:
        raise ValueError("unsupported canonical field checkpoint schema")
    if observation_contract != "canonical":
        raise ValueError(
            "factorized schema-v2 requires the strict canonical observation mode"
        )
    support = load_factorized_field_support(
        args.field_checkpoint,
        expected_field_checkpoint_sha256=field_checkpoint_sha256,
        mpr_cache=(args.mpr_cache or None),
        expected_mpr_cache_sha256=str(
            getattr(args, "expected_mpr_cache_sha256", "")
        ),
    )
    field = support.field
    payload = dict(support.field_payload)
    factorized = support.cache
    return (
        field,
        payload,
        factorized.xyz.float().cpu(),
        factorized.valid.bool().cpu(),
        factorized.sha256,
        factorized.source,
        CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME,
        support.lineage,
    )


@torch.no_grad()
def build(args: argparse.Namespace) -> dict:
    output = Path(args.output).expanduser().resolve()
    report_output = output.with_suffix(output.suffix + ".json")
    if output.exists() or output.is_symlink() or report_output.exists() or report_output.is_symlink():
        raise FileExistsError(
            f"refuses to clobber canonical capability output: {output}"
        )
    device = torch.device(args.device)
    field_checkpoint_sha256 = _sha256_file(args.field_checkpoint)
    expected_field_sha256 = str(getattr(args, "expected_field_checkpoint_sha256", ""))
    requested_field_schema = str(
        getattr(
            args,
            "field_checkpoint_schema",
            CANONICAL_FIELD_CHECKPOINT_SCHEMA_V1,
        )
    )
    if (
        requested_field_schema == FACTORIZED_FIELD_CHECKPOINT_SCHEMA_V2
        and not expected_field_sha256
    ):
        raise ValueError(
            "factorized capability materialization requires a caller-trusted field SHA-256"
        )
    if expected_field_sha256 and field_checkpoint_sha256 != expected_field_sha256:
        raise ValueError("canonical field checkpoint SHA-256 differs")
    (
        field,
        payload,
        xyz,
        valid,
        _mpr_sha256,
        mpr_path,
        observation_contract,
        field_representation_lineage,
    ) = _load_field_and_support(
        args,
        field_checkpoint_sha256=field_checkpoint_sha256,
    )
    if field.num_gaussians != xyz.shape[0] or valid.shape != (xyz.shape[0],):
        raise ValueError("canonical field and MPR rows do not align")
    capability_projection_contract = (
        _formal_projection_contract(payload) if field_representation_lineage else None
    )
    radio_checkpoint_sha256 = _sha256_file(args.radio_checkpoint)
    expected_radio_sha256 = str(
        getattr(args, "expected_radio_checkpoint_sha256", "")
    )
    if expected_radio_sha256 and radio_checkpoint_sha256 != expected_radio_sha256:
        raise ValueError("official RADIO adaptor checkpoint SHA-256 differs")
    if (
        field_representation_lineage
        and field.signature.radio_checkpoint_sha256 != radio_checkpoint_sha256
    ):
        raise ValueError(
            "factorized field and official RADIO adaptor checkpoint differ"
        )
    if field_representation_lineage and not expected_radio_sha256:
        raise ValueError(
            "factorized capability materialization requires a caller-trusted RADIO SHA-256"
        )
    capability_training_authority = (
        _formal_capability_training_authority(
            payload,
            expected_radio_checkpoint_sha256=radio_checkpoint_sha256,
            expected_factorized_radio_cache_sha256=_mpr_sha256,
        )
        if field_representation_lineage
        else None
    )

    adaptors = {
        "appearance_dino_v3": load_radio_adaptor_from_checkpoint(
            args.radio_checkpoint,
            "dino_v3_7b",
            kind="feature_projection",
            expected_sha256=radio_checkpoint_sha256,
        )
        .to(device)
        .eval(),
        "boundary_sam3": load_radio_adaptor_from_checkpoint(
            args.radio_checkpoint,
            "sam3",
            kind="feature_projection",
            expected_sha256=radio_checkpoint_sha256,
        )
        .to(device)
        .eval(),
    }
    field = field.to(device).eval()
    for module in (field, *adaptors.values()):
        module.requires_grad_(False)

    rows = torch.where(valid)[0]
    # Capability consumers operate only on ``valid`` primitive rows.  The
    # historical dense layout allocated N x (4096 + 1024) fp16 values and
    # filled invalid rows with zero.  That is needlessly close to host OOM for
    # multi-million-Gaussian scenes (for example SPIn-NeRF truck).  Store rows
    # in the deterministic ``torch.where(valid)`` order instead; the aligned
    # xyz/valid tensors retain the global primitive domain.
    outputs = {
        name: torch.empty(rows.numel(), adaptor.output_dim, dtype=torch.float16)
        for name, adaptor in adaptors.items()
    }
    for start in range(0, rows.numel(), int(args.batch_size)):
        selected_cpu = rows[start : start + int(args.batch_size)]
        selected = selected_cpu.to(device)
        radio = field.radio_features(selected).float()
        for name, adaptor in adaptors.items():
            projected = F.normalize(adaptor(radio).float(), dim=-1, eps=1e-8)
            outputs[name][start : start + selected_cpu.numel()] = projected.half().cpu()

    base_signature = field.signature.to_dict()
    raw_capability_targets = payload.get("capability_mpr_targets", {})
    if not isinstance(raw_capability_targets, dict):
        raise ValueError("canonical field capability MPR provenance must be a mapping")
    capability_teacher_sources: dict[str, dict[str, object]] = {}
    for target_name, output_name in (
        ("dino_v3", "appearance"),
        ("sam3", "boundary"),
    ):
        target = raw_capability_targets.get(target_name, {})
        if not isinstance(target, dict):
            raise ValueError(
                f"canonical field {target_name} capability provenance must be a mapping"
            )
        native_grid = target.get("capability_native_map_grid", [])
        if not isinstance(native_grid, (list, tuple)):
            raise ValueError(
                f"canonical field {target_name} native-map grid must be a sequence"
            )
        capability_teacher_sources[output_name] = {
            "capability_map_source": str(
                target.get("capability_map_source", "project_raw")
            ),
            "capability_native_map_manifest": str(
                target.get("capability_native_map_manifest", "")
            ),
            "capability_native_map_manifest_sha256": str(
                target.get("capability_native_map_manifest_sha256", "")
            ),
            "capability_native_map_grid": list(native_grid),
            "capability_adaptor_execution": str(
                target.get("capability_adaptor_execution", "")
            ),
            **(
                {
                    "target_contract": str(
                        target.get(
                            "target_contract",
                            payload.get("capability_target_contract", ""),
                        )
                    ),
                    "projection_order": str(target.get("projection_order", "")),
                }
                if field_representation_lineage
                else {}
            ),
        }
    render_optimization = payload.get("render_optimization", {})
    if not isinstance(render_optimization, dict):
        raise ValueError(
            "canonical field render optimization provenance must be a mapping"
        )
    render_capability = render_optimization.get("official_render_capability", {})
    if not isinstance(render_capability, dict):
        raise ValueError(
            "canonical field render capability provenance must be a mapping"
        )
    render_teacher_provenance = render_capability.get("teacher_map_provenance", {})
    if not isinstance(render_teacher_provenance, dict):
        raise ValueError("canonical field render teacher provenance must be a mapping")

    def capability_signature(name: str, output_dim: int) -> dict:
        return FeatureSpaceSignature(
            **{
                **base_signature,
                "adaptor_name": f"{name}.feature_projection",
                "adaptor_sha256": radio_checkpoint_sha256,
                "adaptor_output_dim": int(output_dim),
                "token_type": "primitive",
                "normalization": "l2",
                "field_checkpoint_sha256": field_checkpoint_sha256,
                "semantic_alignment": "none",
                "semantic_alignment_sha256": "",
            }
        ).to_dict()

    metadata = {
        "schema_version": 1,
        "source": "canonical_radio_field_official_frozen_capability_views",
        "field_checkpoint": str(Path(args.field_checkpoint).resolve()),
        "field_checkpoint_sha256": field_checkpoint_sha256,
        "mpr_cache": str(mpr_path.resolve()),
        "mpr_cache_sha256": _mpr_sha256,
        "observation_contract": observation_contract,
        "radio_checkpoint": str(Path(args.radio_checkpoint).resolve()),
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
        "capability_training_mpr_sources": capability_teacher_sources,
        **(
            {"capability_training_authority": capability_training_authority}
            if capability_training_authority is not None
            else {}
        ),
        **(
            {"capability_projection_contract": capability_projection_contract}
            if capability_projection_contract is not None
            else {}
        ),
        "render_capability_teacher_source": str(
            render_capability.get("teacher_map_source", "project_raw")
        ),
        "render_capability_teacher_provenance": dict(render_teacher_provenance),
        "capability_signatures": {
            "appearance": capability_signature(
                "dino_v3_7b", outputs["appearance_dino_v3"].shape[1]
            ),
            "boundary": capability_signature("sam3", outputs["boundary_sam3"].shape[1]),
        },
        **field_representation_lineage,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_torch_noclobber(
        output,
        {
            "schema_version": 1,
            "xyz": xyz,
            "valid": valid,
            **outputs,
            "metadata": metadata,
        },
    )
    report = {
        **metadata,
        "output": str(output),
        "num_gaussians": int(xyz.shape[0]),
        "valid_gaussians": int(valid.sum()),
        "appearance_dim": int(outputs["appearance_dino_v3"].shape[1]),
        "boundary_dim": int(outputs["boundary_sam3"].shape[1]),
    }
    write_frozen_json(report_output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument(
        "--field-checkpoint-schema",
        choices=(
            CANONICAL_FIELD_CHECKPOINT_SCHEMA_V1,
            FACTORIZED_FIELD_CHECKPOINT_SCHEMA_V2,
        ),
        default=CANONICAL_FIELD_CHECKPOINT_SCHEMA_V1,
    )
    parser.add_argument("--expected-field-checkpoint-sha256", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mpr-cache", default="")
    parser.add_argument("--expected-mpr-cache-sha256", default="")
    parser.add_argument(
        "--observation-contract",
        choices=("canonical", "compatible-legacy"),
        default="canonical",
    )
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--expected-radio-checkpoint-sha256", default="")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
