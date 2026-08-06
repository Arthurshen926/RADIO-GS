"""Fail-closed projection-order contracts for formal capability assets.

Official RADIO adaptors are nonlinear, so applying an adaptor after a raw
multi-view primitive reduction is not interchangeable with reducing official
per-view capability features.  This module keeps that distinction explicit at
the artifact boundary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CANONICAL_FIELD_CAPABILITY_SOURCE = (
    "canonical_radio_field_official_frozen_capability_views"
)
EXACT_RAW_MPR_CAPABILITY_SOURCE = (
    "exact_radio_mpr_official_frozen_capability_views"
)
EXACT_CAPABILITY_MPR_SOURCE = (
    "exact_capability_mpr_official_frozen_capability_views"
)

FORMAL_PROJECTION_CONTRACT = "radio_gs.formal_capability_projection.v1"
RAW_MPR_DIAGNOSTIC_CONTRACT = (
    "radio_gs.raw_mpr_then_nonlinear_adaptor_diagnostic.v1"
)
LEGACY_PROJECTION_AUTHORITY_CONTRACT = (
    "radio_gs.legacy_capability_projection_authority.v1"
)

FORMAL_TARGET_MODE_TO_CONTRACT = {
    "official_adaptor_then_geometry_matched_mpr": "matched_top1",
    "official_adaptor_then_exact_raster_adjoint_contribution_mpr": (
        "field_a_exact_adjoint"
    ),
    "official_adaptor_then_exact_center_plus_uncertainty_mpr": (
        "field_c_exact_center_uncertainty"
    ),
    "official_adaptor_then_shared_exact_marginal_mpr": "matched_exact_marginal",
}
FORMAL_TARGET_MODES = frozenset(FORMAL_TARGET_MODE_TO_CONTRACT)
FORMAL_TARGET_CONTRACTS = frozenset(FORMAL_TARGET_MODE_TO_CONTRACT.values())
LEGACY_MATCHED_TOP1_CONTRACT = "legacy_pre_contract_matched_top1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_authority(
    value: str | Path | Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, Mapping):
        authority: Mapping[str, Any] = value
    else:
        path = Path(value).expanduser().resolve()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError("capability projection authority must be an object")
        authority = loaded
    nested = authority.get("legacy_compatibility_authority")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ValueError("legacy compatibility authority must be an object")
        authority = nested
    if authority.get("contract") != LEGACY_PROJECTION_AUTHORITY_CONTRACT:
        raise ValueError("unsupported legacy capability projection authority")
    if int(authority.get("schema_version", -1)) != 1:
        raise ValueError("unsupported legacy capability projection authority schema")
    return authority


def _validate_target_lineage(
    *,
    target_mode: object,
    target_contract: object,
    teacher_projection_orders: object,
    allow_legacy_target_contract: bool,
) -> None:
    mode = str(target_mode)
    expected_contract = FORMAL_TARGET_MODE_TO_CONTRACT.get(mode)
    if expected_contract is None:
        raise ValueError("formal capability target mode is not adaptor-before-MPR")
    contract = str(target_contract)
    if contract == LEGACY_MATCHED_TOP1_CONTRACT and allow_legacy_target_contract:
        if expected_contract != "matched_top1":
            raise ValueError(
                "legacy matched-top1 contract requires geometry-matched target mode"
            )
    elif contract != expected_contract:
        raise ValueError("formal capability target mode and contract differ")
    if not isinstance(teacher_projection_orders, Mapping):
        raise ValueError("formal capability teacher projection orders are absent")
    if set(teacher_projection_orders) != {"appearance", "boundary"}:
        raise ValueError("formal capability teacher roles must be appearance and boundary")
    invalid = {
        name: str(order)
        for name, order in teacher_projection_orders.items()
        if str(order) != mode
    }
    if invalid:
        raise ValueError(
            "formal capability teacher order differs from the declared target mode: "
            f"{invalid}"
        )


def _validate_inline_formal_contract(
    metadata: Mapping[str, Any],
    *,
    source: str,
) -> Mapping[str, Any]:
    raw = metadata.get("capability_projection_contract")
    if not isinstance(raw, Mapping):
        # The exact capability-MPR builder originally used the shorter key.
        raw = metadata.get("projection_contract")
    if not isinstance(raw, Mapping):
        raise ValueError("formal capability projection contract is absent")

    if source == EXACT_CAPABILITY_MPR_SOURCE:
        declared_contract = str(raw.get("contract", ""))
        if declared_contract and declared_contract != FORMAL_PROJECTION_CONTRACT:
            raise ValueError("unsupported exact capability MPR formal contract")
        if str(raw.get("projection_order", "")) != "official_adaptor_before_mpr":
            raise ValueError(
                "exact capability MPR must project each view before MPR"
            )
        eligibility = str(raw.get("eligibility", "formal_exact_teacher"))
        if eligibility != "formal_exact_teacher":
            raise ValueError("exact capability MPR is not a formal exact teacher")
        if raw.get("query_dependent") is not False:
            raise ValueError("exact capability MPR must be query independent")
        if declared_contract and raw.get("nonlinear_adaptor_after_raw_mpr") is not False:
            raise ValueError("exact capability MPR admits nonlinear adaptor after MPR")
        return raw

    if raw.get("contract") != FORMAL_PROJECTION_CONTRACT:
        raise ValueError("unsupported formal capability projection contract")
    if raw.get("eligibility") != "formal_one_field":
        raise ValueError("compact capability is not eligible as a formal one-field asset")
    if raw.get("artifact_role") != "capability_supervised_compact_field":
        raise ValueError("formal compact capability artifact role differs")
    if (
        raw.get("field_output_projection_order")
        != "compact_radio_field_then_official_adaptor"
    ):
        raise ValueError("formal compact field output projection order differs")
    if raw.get("nonlinear_adaptor_after_raw_mpr") is not False:
        raise ValueError("raw-MPR then nonlinear adaptor cannot be a formal capability")
    _validate_target_lineage(
        target_mode=raw.get("capability_target_mode"),
        target_contract=raw.get("capability_target_contract"),
        teacher_projection_orders=raw.get("teacher_projection_orders"),
        allow_legacy_target_contract=False,
    )
    return raw


def _validate_legacy_authority(
    metadata: Mapping[str, Any],
    *,
    cache_path: Path,
    authority_value: str | Path | Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    authority = _load_authority(authority_value)
    if authority is None:
        raise ValueError(
            "legacy capability cache lacks an inline formal projection contract; "
            "an exact compatibility authority is required"
        )
    entries = authority.get("entries")
    if not isinstance(entries, list):
        raise ValueError("legacy capability projection authority entries are absent")
    resolved = str(cache_path.expanduser().resolve())
    matches = [
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and str(entry.get("capability_cache", "")) == resolved
    ]
    if len(matches) != 1:
        raise ValueError("capability cache is not uniquely bound by legacy authority")
    entry = matches[0]
    if entry.get("formal_one_field_eligible") is not True:
        raise ValueError("legacy authority does not mark this field formally eligible")
    if entry.get("source") != CANONICAL_FIELD_CAPABILITY_SOURCE:
        raise ValueError("legacy authority capability source differs")
    if str(entry.get("field_checkpoint_sha256", "")) != str(
        metadata.get("field_checkpoint_sha256", "")
    ):
        raise ValueError("legacy authority field checkpoint digest differs")
    if str(entry.get("field_checkpoint", "")) != str(
        metadata.get("field_checkpoint", "")
    ):
        raise ValueError("legacy authority field checkpoint path differs")
    sidecar = cache_path.with_suffix(cache_path.suffix + ".json")
    if not sidecar.is_file() or sha256_file(sidecar) != str(
        entry.get("capability_cache_sidecar_sha256", "")
    ):
        raise ValueError("legacy authority capability sidecar digest differs")
    _validate_target_lineage(
        target_mode=entry.get("capability_target_mode"),
        target_contract=entry.get("capability_target_contract"),
        teacher_projection_orders=entry.get("teacher_projection_orders"),
        allow_legacy_target_contract=True,
    )
    if entry.get("nonlinear_adaptor_after_raw_mpr") is not False:
        raise ValueError("legacy authority admits raw-MPR then nonlinear adaptor")
    return entry


def validate_capability_projection_order(
    metadata: Mapping[str, Any],
    *,
    cache_path: str | Path,
    expected_source: str,
    require_formal: bool,
    allow_raw_mpr_diagnostic: bool,
    legacy_authority: str | Path | Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    """Validate formal or explicitly diagnostic capability projection lineage."""

    source = str(metadata.get("source", ""))
    if source != expected_source:
        raise ValueError("capability cache source contract differs")
    if require_formal and allow_raw_mpr_diagnostic:
        raise ValueError("formal and raw-MPR diagnostic capability modes conflict")

    if source == EXACT_RAW_MPR_CAPABILITY_SOURCE:
        if not allow_raw_mpr_diagnostic:
            raise ValueError(
                "raw-MPR then nonlinear adaptor is diagnostic-only; explicit "
                "diagnostic authority is required"
            )
        raw = metadata.get("projection_contract")
        if not isinstance(raw, Mapping):
            raise ValueError("raw-MPR projection diagnostic contract is absent")
        declared = str(raw.get("projection_order", ""))
        declared_contract = str(raw.get("contract", ""))
        if declared_contract and declared_contract != RAW_MPR_DIAGNOSTIC_CONTRACT:
            raise ValueError("unsupported raw-MPR projection diagnostic contract")
        legacy_operator = str(raw.get("operator", ""))
        if declared != "raw_radio_mpr_then_official_adaptor" and not (
            legacy_operator == "frozen official adaptor then fp32 L2 normalization"
            and str(raw.get("source", ""))
            == "exact valid rows of raw RADIO MPR"
        ):
            raise ValueError("raw-MPR diagnostic projection order differs")
        if raw.get("query_dependent") is not False:
            raise ValueError("raw-MPR projection diagnostic must be query independent")
        return raw

    if source == EXACT_CAPABILITY_MPR_SOURCE:
        return _validate_inline_formal_contract(metadata, source=source)

    if source != CANONICAL_FIELD_CAPABILITY_SOURCE:
        raise ValueError("unsupported capability projection source")
    if not require_formal:
        return None
    if isinstance(metadata.get("capability_projection_contract"), Mapping):
        return _validate_inline_formal_contract(metadata, source=source)
    return _validate_legacy_authority(
        metadata,
        cache_path=Path(cache_path),
        authority_value=legacy_authority,
    )
