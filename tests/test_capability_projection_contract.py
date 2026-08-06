from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from radio_gs.field import FeatureSpaceSignature
from radio_gs.interfaces.capability_cache import load_canonical_capability_bank
from radio_gs.interfaces.capability_projection_contract import (
    CANONICAL_FIELD_CAPABILITY_SOURCE,
    EXACT_CAPABILITY_MPR_SOURCE,
    EXACT_RAW_MPR_CAPABILITY_SOURCE,
    FORMAL_PROJECTION_CONTRACT,
    LEGACY_MATCHED_TOP1_CONTRACT,
    LEGACY_PROJECTION_AUTHORITY_CONTRACT,
)
from radio_gs.scripts.build_canonical_capability_views import (
    _formal_projection_contract,
)
from radio_gs.scripts.build_exact_capability_mpr_views import (
    _check_query_independent,
)


def _signature(name: str, dim: int, field_hash: str) -> dict:
    return FeatureSpaceSignature(
        radio_version="c-radio_v4-h",
        radio_checkpoint_sha256="a" * 64,
        raw_feature_dim=4,
        adaptor_name=name,
        adaptor_sha256="a" * 64,
        adaptor_output_dim=dim,
        token_type="primitive",
        normalization="l2",
        field_checkpoint_sha256=field_hash,
    ).to_dict()


def _write_bank(
    path: Path,
    *,
    source: str = CANONICAL_FIELD_CAPABILITY_SOURCE,
    projection_contract: dict | None = None,
) -> dict:
    field_hash = "b" * 64
    metadata = {
        "schema_version": 1,
        "source": source,
        "field_checkpoint": str((path.parent / "field.pth").resolve()),
        "field_checkpoint_sha256": field_hash,
        "radio_checkpoint_sha256": "a" * 64,
        "custom_adaptor_head": False,
        "query_independent": True,
        "capability_signatures": {
            "appearance": _signature("dino", 3, field_hash),
            "boundary": _signature("sam3", 2, field_hash),
        },
    }
    if source == EXACT_RAW_MPR_CAPABILITY_SOURCE:
        metadata["exact_raw_mpr_sha256"] = field_hash
    elif source == EXACT_CAPABILITY_MPR_SOURCE:
        metadata["exact_capability_mpr_pair_sha256"] = field_hash
    if projection_contract is not None:
        key = (
            "capability_projection_contract"
            if source == CANONICAL_FIELD_CAPABILITY_SOURCE
            else "projection_contract"
        )
        metadata[key] = projection_contract
    torch.save(
        {
            "schema_version": 1,
            "xyz": torch.zeros(2, 3),
            "valid": torch.tensor([True, True]),
            "appearance_dino_v3": torch.zeros(2, 3),
            "boundary_sam3": torch.zeros(2, 2),
            "metadata": metadata,
        },
        path,
    )
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps({**metadata, "output": str(path.resolve())}), encoding="utf-8"
    )
    return metadata


def _formal_contract(
    *,
    target_mode: str = "official_adaptor_then_geometry_matched_mpr",
    target_contract: str = "matched_top1",
    boundary_projection_order: str | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "contract": FORMAL_PROJECTION_CONTRACT,
        "eligibility": "formal_one_field",
        "artifact_role": "capability_supervised_compact_field",
        "field_output_projection_order": (
            "compact_radio_field_then_official_adaptor"
        ),
        "capability_target_mode": target_mode,
        "capability_target_contract": target_contract,
        "teacher_projection_orders": {
            "appearance": target_mode,
            "boundary": boundary_projection_order or target_mode,
        },
        "nonlinear_adaptor_after_raw_mpr": False,
        "query_independent": True,
    }


def test_formal_compact_loader_requires_capability_first_lineage(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "compact.pt"
    _write_bank(cache)
    with pytest.raises(ValueError, match="compatibility authority"):
        load_canonical_capability_bank(
            cache, require_formal_projection_order=True
        )

    _write_bank(cache, projection_contract=_formal_contract())
    bank = load_canonical_capability_bank(
        cache, require_formal_projection_order=True
    )
    assert bank.metadata["capability_projection_contract"]["eligibility"] == (
        "formal_one_field"
    )


def test_legacy_authority_is_exact_path_sidecar_and_field_bound(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "legacy.pt"
    metadata = _write_bank(cache)
    sidecar = cache.with_suffix(cache.suffix + ".json")
    authority = {
        "schema_version": 1,
        "contract": LEGACY_PROJECTION_AUTHORITY_CONTRACT,
        "entries": [
            {
                "capability_cache": str(cache.resolve()),
                "capability_cache_sidecar_sha256": hashlib.sha256(
                    sidecar.read_bytes()
                ).hexdigest(),
                "source": CANONICAL_FIELD_CAPABILITY_SOURCE,
                "field_checkpoint": metadata["field_checkpoint"],
                "field_checkpoint_sha256": metadata["field_checkpoint_sha256"],
                "capability_target_mode": (
                    "official_adaptor_then_geometry_matched_mpr"
                ),
                "capability_target_contract": LEGACY_MATCHED_TOP1_CONTRACT,
                "teacher_projection_orders": {
                    "appearance": "official_adaptor_then_geometry_matched_mpr",
                    "boundary": "official_adaptor_then_geometry_matched_mpr",
                },
                "nonlinear_adaptor_after_raw_mpr": False,
                "formal_one_field_eligible": True,
            }
        ],
    }
    load_canonical_capability_bank(
        cache,
        require_formal_projection_order=True,
        legacy_projection_authority=authority,
    )
    sidecar.write_text(sidecar.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar digest"):
        load_canonical_capability_bank(
            cache,
            require_formal_projection_order=True,
            legacy_projection_authority=authority,
        )


def test_raw_mpr_then_adaptor_requires_explicit_diagnostic_mode(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "raw-after.pt"
    _write_bank(
        cache,
        source=EXACT_RAW_MPR_CAPABILITY_SOURCE,
        projection_contract={
            "contract": "radio_gs.raw_mpr_then_nonlinear_adaptor_diagnostic.v1",
            "eligibility": "diagnostic_only",
            "projection_order": "raw_radio_mpr_then_official_adaptor",
            "query_dependent": False,
        },
    )
    with pytest.raises(ValueError, match="diagnostic-only"):
        load_canonical_capability_bank(
            cache, expected_source=EXACT_RAW_MPR_CAPABILITY_SOURCE
        )
    load_canonical_capability_bank(
        cache,
        expected_source=EXACT_RAW_MPR_CAPABILITY_SOURCE,
        allow_raw_mpr_projection_diagnostic=True,
    )


def test_exact_capability_mpr_is_formal_only_when_projection_precedes_mpr(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "exact-teacher.pt"
    contract = {
        "operator": "per-view official adaptor, shared MPR, fp32 row L2",
        "projection_order": "official_adaptor_before_mpr",
        "eligibility": "formal_exact_teacher",
        "query_dependent": False,
    }
    _write_bank(
        cache, source=EXACT_CAPABILITY_MPR_SOURCE, projection_contract=contract
    )
    load_canonical_capability_bank(
        cache, expected_source=EXACT_CAPABILITY_MPR_SOURCE
    )
    contract["projection_order"] = "raw_radio_mpr_then_official_adaptor"
    _write_bank(
        cache, source=EXACT_CAPABILITY_MPR_SOURCE, projection_contract=contract
    )
    with pytest.raises(ValueError, match="project each view before MPR"):
        load_canonical_capability_bank(
            cache, expected_source=EXACT_CAPABILITY_MPR_SOURCE
        )


def test_builder_rejects_raw_mpr_supervision_as_formal_capability() -> None:
    payload = {
        "capability_target_mode": "official_adaptor_then_geometry_matched_mpr",
        "capability_target_contract": "matched_top1",
        "capability_mpr_targets": {
            name: {
                "projection_order": "official_adaptor_then_geometry_matched_mpr",
                "uses_query_or_benchmark_supervision": False,
            }
            for name in ("dino_v3", "sam3")
        },
    }
    assert _formal_projection_contract(payload)["eligibility"] == "formal_one_field"
    payload["capability_target_mode"] = "adaptor_of_raw_mpr_target"
    with pytest.raises(ValueError, match="not supervised"):
        _formal_projection_contract(payload)


def test_exact_marginal_target_lineage_is_formal_and_loadable(tmp_path: Path) -> None:
    target_mode = "official_adaptor_then_shared_exact_marginal_mpr"
    target_contract = "matched_exact_marginal"
    payload = {
        "capability_target_mode": target_mode,
        "capability_target_contract": target_contract,
        "capability_mpr_targets": {
            name: {
                "projection_order": target_mode,
                "uses_query_or_benchmark_supervision": False,
            }
            for name in ("dino_v3", "sam3")
        },
    }
    built = _formal_projection_contract(payload)
    assert built["capability_target_mode"] == target_mode
    assert built["capability_target_contract"] == target_contract
    assert built["teacher_projection_orders"] == {
        "appearance": target_mode,
        "boundary": target_mode,
    }

    cache = tmp_path / "exact-marginal-compact.pt"
    _write_bank(
        cache,
        projection_contract=_formal_contract(
            target_mode=target_mode,
            target_contract=target_contract,
        ),
    )
    bank = load_canonical_capability_bank(
        cache, require_formal_projection_order=True
    )
    assert bank.metadata["capability_projection_contract"][
        "capability_target_contract"
    ] == target_contract


@pytest.mark.parametrize(
    ("target_mode", "target_contract"),
    (
        (
            "official_adaptor_then_geometry_matched_mpr",
            "matched_top1",
        ),
        (
            "official_adaptor_then_exact_raster_adjoint_contribution_mpr",
            "field_a_exact_adjoint",
        ),
        (
            "official_adaptor_then_exact_center_plus_uncertainty_mpr",
            "field_c_exact_center_uncertainty",
        ),
        (
            "official_adaptor_then_shared_exact_marginal_mpr",
            "matched_exact_marginal",
        ),
    ),
)
def test_formal_builder_preserves_every_registered_mode_contract_pair(
    target_mode: str,
    target_contract: str,
) -> None:
    payload = {
        "capability_target_mode": target_mode,
        "capability_target_contract": target_contract,
        "capability_mpr_targets": {
            name: {
                "projection_order": target_mode,
                "uses_query_or_benchmark_supervision": False,
            }
            for name in ("dino_v3", "sam3")
        },
    }
    assert _formal_projection_contract(payload)[
        "capability_target_contract"
    ] == target_contract


@pytest.mark.parametrize(
    ("target_mode", "target_contract"),
    (
        (
            "official_adaptor_then_shared_exact_marginal_mpr",
            "matched_top1",
        ),
        (
            "official_adaptor_then_geometry_matched_mpr",
            "matched_exact_marginal",
        ),
    ),
)
def test_formal_target_mode_and_contract_must_match(
    tmp_path: Path,
    target_mode: str,
    target_contract: str,
) -> None:
    payload = {
        "capability_target_mode": target_mode,
        "capability_target_contract": target_contract,
        "capability_mpr_targets": {
            name: {
                "projection_order": target_mode,
                "uses_query_or_benchmark_supervision": False,
            }
            for name in ("dino_v3", "sam3")
        },
    }
    with pytest.raises(ValueError, match="mode and contract differ"):
        _formal_projection_contract(payload)

    cache = tmp_path / f"mismatch-{target_contract}.pt"
    _write_bank(
        cache,
        projection_contract=_formal_contract(
            target_mode=target_mode,
            target_contract=target_contract,
        ),
    )
    with pytest.raises(ValueError, match="mode and contract differ"):
        load_canonical_capability_bank(
            cache, require_formal_projection_order=True
        )


def test_formal_teacher_orders_must_match_declared_target_mode(
    tmp_path: Path,
) -> None:
    target_mode = "official_adaptor_then_shared_exact_marginal_mpr"
    other_mode = "official_adaptor_then_geometry_matched_mpr"
    payload = {
        "capability_target_mode": target_mode,
        "capability_target_contract": "matched_exact_marginal",
        "capability_mpr_targets": {
            "dino_v3": {
                "projection_order": target_mode,
                "uses_query_or_benchmark_supervision": False,
            },
            "sam3": {
                "projection_order": other_mode,
                "uses_query_or_benchmark_supervision": False,
            },
        },
    }
    with pytest.raises(ValueError, match="projection order differs"):
        _formal_projection_contract(payload)

    cache = tmp_path / "mixed-teacher-order.pt"
    _write_bank(
        cache,
        projection_contract=_formal_contract(
            target_mode=target_mode,
            target_contract="matched_exact_marginal",
            boundary_projection_order=other_mode,
        ),
    )
    with pytest.raises(ValueError, match="teacher order differs"):
        load_canonical_capability_bank(
            cache, require_formal_projection_order=True
        )


def test_exact_teacher_builder_requires_per_view_projection_before_mpr() -> None:
    metadata = {
        "feature_space": "dino_v3",
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "shared_registration_responsibility": True,
        "capability_projection_before_mpr": True,
        "observation_lifting_contract": {
            "name": "canonical-mpr-v1",
            "feature_projection_order": "per_view_before_mpr",
            "query_independent": True,
        },
    }
    assert _check_query_independent(metadata, space="dino_v3") is metadata
    metadata["observation_lifting_contract"]["feature_projection_order"] = (
        "after_mpr"
    )
    with pytest.raises(ValueError, match="projection-order"):
        _check_query_independent(metadata, space="dino_v3")
