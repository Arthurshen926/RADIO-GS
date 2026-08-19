from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from radio_gs.field import (
    AffineBasisDecoder,
    CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME,
    CanonicalGaussianField,
    FeatureSpaceSignature,
    canonical_observation_contract,
    load_canonical_field_checkpoint,
    load_factorized_canonical_field_checkpoint,
    observation_contract_sha256,
)
from radio_gs.field.factorized_radio_contract import (
    CANONICAL_FACTORIZED_RADIO_CHECKPOINT_CONTRACT,
    CANONICAL_FACTORIZED_RADIO_CHECKPOINT_SCHEMA_VERSION,
    CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256,
    FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES,
    FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256,
    FactorizedRadioFieldSignature,
    canonical_factorized_radio_contract,
    factorized_radio_checkpoint_metadata,
)
from radio_gs.scripts.train_canonical_radio_field import (
    CAPABILITY_TARGET_CONTRACT_MATCHED_EXACT_MARGINAL,
    _load_factorized_exact_marginal_capability_targets,
    _load_factorized_matched_capability_targets,
)
from radio_gs.rendering.contribution_compositor import (
    MARGINAL_RESPONSIBILITY_CONTRACT,
)
import radio_gs.scripts.train_canonical_radio_field as trainer
from radio_gs.training.canonical_field_losses import (
    CanonicalFieldLossConfig,
    canonical_primitive_loss,
)
from radio_gs.training.factorized_radio_cache import (
    CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA,
    CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA_V2,
    FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY,
    FACTORIZED_RADIO_TOP1_PURITY_AUTHORITY,
    FactorizedRadioTrainingCache,
    canonical_factorized_radio_builder_contract,
    canonical_factorized_radio_builder_contract_v2,
    factorized_radio_builder_contract_sha256,
    factorized_radio_builder_contract_v2_sha256,
    load_factorized_radio_training_cache,
    validate_factorized_radio_training_payload,
)
from radio_gs.training.factorized_radio_loss import (
    FACTORIZED_RADIO_RELIABILITY_POLICY_LEGACY,
    FACTORIZED_RADIO_RELIABILITY_POLICY_MATCHED_EXACT_MARGINAL_VISIBILITY_SAFE,
)
from radio_gs.utils.immutable_artifacts import sha256_file


SHA = "a" * 64
BUNDLE_SHA = "b" * 64
RESPONSIBILITY_SHA = "c" * 64


def _xyz_sha(values: torch.Tensor) -> str:
    array = values.float().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _factorized_payload(rows: int = 3) -> dict:
    xyz = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
    valid = torch.tensor([True] * (rows - 1) + [False])
    counts = torch.tensor([1] * (rows - 1) + [0], dtype=torch.long)
    direction = torch.zeros(rows, 1280, dtype=torch.float32)
    direction[:-1, 0] = 1.0
    log_amplitude = torch.zeros(rows, dtype=torch.float32)
    log_amplitude[:-1] = torch.log(torch.tensor(2.0))
    canonical = (torch.exp(log_amplitude)[:, None] * direction).half()
    reliability = torch.zeros(rows, 5, dtype=torch.float32)
    reliability[:-1] = torch.tensor([0.8, 0.2, 0.1, 0.5, 0.0])
    builder = canonical_factorized_radio_builder_contract()
    metadata = {
        "builder_contract": builder,
        "builder_contract_sha256": factorized_radio_builder_contract_sha256(),
        "construction": "canonical-factorized-radio-v1",
        "feature_space": "radio",
        "input_feature_space": "radio_raw_full",
        "feature_dim": 1280,
        "config": "/tmp/config.yaml",
        "checkpoint": "/tmp/geometry.pt",
        "geometry_checkpoint_sha256": SHA,
        "feature_frame_manifest_sha256": "d" * 64,
        "feature_output_bundle_sha256": BUNDLE_SHA,
        "selected_dataset_indices": [0],
        "selected_frame_indices": [0],
        "num_declared_views": 1,
        "max_views_authority": 120,
        "aggregation_mode": "raster_gaussian_top1",
        "raster_view_fusion": "contribution_mean",
        "registration_weight_mode": "alpha_depth",
        "registration_responsibility_cache_sha256": RESPONSIBILITY_SHA,
        "semantic_direction_storage": "derived_from_canonical_feature_not_persisted",
        "canonical_feature_dtype": "float16",
        "log_amplitude_dtype": "float32",
        "reliability_dtype": "float32",
        "robust_mpr": False,
        "visibility_purity_authority": {
            **FACTORIZED_RADIO_TOP1_PURITY_AUTHORITY,
            "registration_responsibility_cache_sha256": RESPONSIBILITY_SHA,
        },
        "benchmark_masks_opened": False,
        "benchmark_images_opened": False,
        "text_queries_opened": False,
        "query_independent": True,
    }
    return {
        "schema": CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA,
        "schema_version": 1,
        "xyz": xyz,
        "geometry_fingerprint": {
            "num_gaussians": rows,
            "xyz_sha256": _xyz_sha(xyz),
        },
        "factorized_radio": {
            "schema": "radio_gs.canonical_factorized_radio.v1",
            "schema_version": 1,
            "contract": canonical_factorized_radio_contract(),
            "contract_sha256": CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256,
            "reliability_scalar_names": list(FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES),
            "reliability_scalar_names_sha256": (
                FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256
            ),
            "log_amplitude": log_amplitude,
            "canonical_feature": canonical,
            "valid": valid,
            "reliability": reliability,
        },
        "view_counts": counts,
        "metadata": metadata,
    }


def _training_cache(payload: dict) -> FactorizedRadioTrainingCache:
    core = payload["factorized_radio"]
    return FactorizedRadioTrainingCache(
        source=Path("/tmp/factorized.pt"),
        sha256=SHA,
        xyz=payload["xyz"],
        geometry_fingerprint=dict(payload["geometry_fingerprint"]),
        canonical_feature=core["canonical_feature"],
        log_amplitude=core["log_amplitude"],
        valid=core["valid"],
        view_counts=payload["view_counts"],
        reliability=core["reliability"],
        reliability_scalar_names=tuple(core["reliability_scalar_names"]),
        reliability_scalar_names_sha256=core["reliability_scalar_names_sha256"],
        metadata=dict(payload["metadata"]),
    )


def _exact_factorized_payload(rows: int = 2) -> dict:
    payload = _factorized_payload(rows=rows)
    payload["schema"] = CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA_V2
    payload["schema_version"] = 2
    payload["factorized_radio"]["valid"][:] = True
    payload["view_counts"][:] = 1
    payload["factorized_radio"]["canonical_feature"].zero_()
    payload["factorized_radio"]["canonical_feature"][:, 0] = 2.0
    payload["factorized_radio"]["log_amplitude"][:] = torch.log(torch.tensor(2.0))
    payload["factorized_radio"]["reliability"][:] = torch.tensor(
        [0.8, 0.2, 0.1, 0.5, 0.75]
    )
    metadata = payload["metadata"]
    responsibility_contract = {
        "schema_version": 1,
        "assignment_mode": "exact_front_to_back_sparse_marginal",
        "registration_weight_mode": ("exact_front_to_back_marginal_responsibility"),
        "post_compositor_alpha_threshold": 0.0,
        "formula_sha256": FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY[
            "formula_sha256"
        ],
        "builder_implementation_sha256": "1" * 64,
        "authority_implementation_sha256": "2" * 64,
        "query_independent": True,
    }
    metadata.update(
        {
            "builder_contract": canonical_factorized_radio_builder_contract_v2(),
            "builder_contract_sha256": (factorized_radio_builder_contract_v2_sha256()),
            "aggregation_mode": "raster_marginal_responsibility",
            "registration_weight_mode": ("exact_front_to_back_marginal_responsibility"),
            "shared_registration_responsibility": True,
            "registration_responsibility_contract": responsibility_contract,
            "observation_lifting_contract": canonical_factorized_radio_contract(),
            "observation_lifting_contract_sha256": (
                CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256
            ),
            "visibility_purity_authority": {
                **FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY,
                "registration_responsibility_cache_sha256": RESPONSIBILITY_SHA,
            },
            "semantic_assignment_gate": (
                "pre_adaptor_raw_radio_l2_norm_strictly_positive"
            ),
            "valid_semantics": (
                "positive_raw_radio_amplitude_responsibility_mass_and_"
                "nonzero_direction_resultant"
            ),
            "view_count_semantics": (
                "views_with_pre_adaptor_raw_radio_l2_norm_strictly_positive"
            ),
            "geometric_visibility_semantics": (
                "independent_exact_base_weight_authority_includes_"
                "zero_amplitude_hits"
            ),
            "geometric_view_counts_sha256": "e" * 64,
            "geometric_visible_gaussian_count": rows,
            "semantic_valid_gaussian_count": rows,
            "geometric_visible_semantic_invalid_gaussian_count": 0,
            "invalid_row_purity_policy": (
                "core_v1_requires_zero_for_semantically_invalid_rows"
            ),
        }
    )
    return payload


def test_factorized_loss_policy_is_bound_to_exact_source_only_purity() -> None:
    top1 = _training_cache(_factorized_payload())
    assert trainer._factorized_loss_reliability_policy(
        top1,
        capability_target_contract="matched_top1",
    ) == FACTORIZED_RADIO_RELIABILITY_POLICY_LEGACY

    exact = _training_cache(_exact_factorized_payload())
    assert trainer._factorized_loss_reliability_policy(
        exact,
        capability_target_contract=CAPABILITY_TARGET_CONTRACT_MATCHED_EXACT_MARGINAL,
    ) == (
        FACTORIZED_RADIO_RELIABILITY_POLICY_MATCHED_EXACT_MARGINAL_VISIBILITY_SAFE
    )


def test_factorized_exact_loss_policy_fails_closed_on_purity_authority() -> None:
    exact = _training_cache(_exact_factorized_payload())
    exact.metadata["visibility_purity_authority"] = {
        **exact.metadata["visibility_purity_authority"],
        "measurement_available": False,
    }
    with pytest.raises(ValueError, match="measured-purity authority"):
        trainer._factorized_loss_reliability_policy(
            exact,
            capability_target_contract=(
                CAPABILITY_TARGET_CONTRACT_MATCHED_EXACT_MARGINAL
            ),
        )


def test_factorized_exact_loss_policy_fails_closed_on_target_access() -> None:
    exact = _training_cache(_exact_factorized_payload())
    exact.metadata["benchmark_images_opened"] = True
    with pytest.raises(ValueError, match="source-only"):
        trainer._factorized_loss_reliability_policy(
            exact,
            capability_target_contract=(
                CAPABILITY_TARGET_CONTRACT_MATCHED_EXACT_MARGINAL
            ),
        )


def test_factorized_training_loader_is_explicit_and_fail_closed(tmp_path: Path) -> None:
    payload = _factorized_payload()
    path = tmp_path / "factorized.pt"
    torch.save(payload, path)
    cache = load_factorized_radio_training_cache(
        path,
        expected_sha256=sha256_file(path),
        expected_feature_output_bundle_sha256=BUNDLE_SHA,
    )
    assert cache.shape == (3, 1280)
    assert cache.as_consensus().reliability.shape == (3, 5)

    renamed = copy.deepcopy(payload)
    renamed["factorized_radio"]["reliability_scalar_names"][0] = "agreement"
    with pytest.raises(ValueError, match="core contract"):
        validate_factorized_radio_training_payload(
            renamed, expected_feature_output_bundle_sha256=BUNDLE_SHA
        )
    wrong_evidence = copy.deepcopy(payload)
    wrong_evidence["factorized_radio"]["reliability"][0, 3] = 0.75
    with pytest.raises(ValueError, match="evidence"):
        validate_factorized_radio_training_payload(
            wrong_evidence, expected_feature_output_bundle_sha256=BUNDLE_SHA
        )
    with pytest.raises(ValueError, match="bundle"):
        validate_factorized_radio_training_payload(
            payload, expected_feature_output_bundle_sha256="e" * 64
        )


class _RawNormOfficialViews:
    seen_norms: list[torch.Tensor] = []

    @classmethod
    def _project(cls, values: torch.Tensor) -> torch.Tensor:
        cls.seen_norms.append(torch.linalg.vector_norm(values.detach(), dim=-1))
        return torch.nn.functional.normalize(values, dim=-1)

    project_dino_primitives = _project
    project_sam3_primitives = _project


def test_factorized_loss_keeps_five_columns_out_of_field_and_projects_raw() -> None:
    payload = _factorized_payload()
    target = _training_cache(payload)
    consensus = target.as_consensus()
    decoder = AffineBasisDecoder(
        feature_dim=1280,
        coefficient_dim=2,
        basis=torch.eye(1280, 2),
        trainable_basis=False,
    )
    field = CanonicalGaussianField(
        3,
        decoder,
        FeatureSpaceSignature(
            radio_version="test",
            radio_checkpoint_sha256=SHA,
            raw_feature_dim=1280,
            token_type="primitive",
            normalization="radio_raw_full",
        ),
        reliability=None,
        fusion_reliability=False,
        use_fusion=False,
    )
    with torch.no_grad():
        field.local_codes.zero_()
        field.local_codes[:2, 0] = 2.0
    capability = copy.deepcopy(consensus)
    config = CanonicalFieldLossConfig(
        mpr_weight=1.0,
        dino_weight=0.2,
        sam3_weight=0.2,
        relation_weight=0.0,
        coefficient_weight=0.0,
        basis_orthogonality_weight=0.0,
    )
    _RawNormOfficialViews.seen_norms.clear()
    loss, stats = canonical_primitive_loss(
        field,
        consensus,
        torch.tensor([0, 1]),
        official_views=_RawNormOfficialViews(),
        capability_targets={"dino_v3": capability, "sam3": capability},
        factorized_target=target,
        config=config,
    )
    assert torch.isfinite(loss)
    assert "factorized_direction" in stats
    assert field.reliability.shape == (3, 0)
    assert field.primitive_confidence() is None
    assert len(_RawNormOfficialViews.seen_norms) == 2
    for norms in _RawNormOfficialViews.seen_norms:
        torch.testing.assert_close(norms, torch.full_like(norms, 2.0))


def _field_checkpoint_payload() -> tuple[dict, FactorizedRadioFieldSignature]:
    decoder = AffineBasisDecoder(feature_dim=4, coefficient_dim=2)
    base = FeatureSpaceSignature(
        radio_version="test",
        radio_checkpoint_sha256=SHA,
        raw_feature_dim=4,
        token_type="primitive",
        normalization="radio_raw_full",
        crop_policy="training_views_canonical_factorized_radio_v1",
    )
    signature = FactorizedRadioFieldSignature.create(base)
    field = CanonicalGaussianField(
        2,
        decoder,
        base,
        reliability=None,
        fusion_reliability=False,
        use_fusion=False,
    )
    architecture = {
        "num_gaussians": 2,
        "feature_dim": 4,
        "coefficient_dim": 2,
        "local_dim": 2,
        "coarse_dim": 0,
        "spatial_hash": None,
        "position_storage": "none",
        "fusion_reliability": False,
        "hidden_dim": 8,
        "fusion_residual_blocks": 0,
        "use_fusion": False,
        "trainable_basis": True,
        "trainable_statistics": False,
    }
    return {
        "schema_version": CANONICAL_FACTORIZED_RADIO_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_contract": CANONICAL_FACTORIZED_RADIO_CHECKPOINT_CONTRACT,
        "factorized_radio_metadata": factorized_radio_checkpoint_metadata(signature),
        "architecture": architecture,
        "state_dict": field.state_dict(),
        "reliability": torch.empty(2, 0),
        "geometry_fingerprint": {"num_gaussians": 2, "xyz_sha256": "d" * 64},
        "factorized_cache_sha256": "e" * 64,
        "feature_output_bundle_sha256": BUNDLE_SHA,
    }, signature


def test_factorized_checkpoint_v2_round_trip_and_bidirectional_rejection(
    tmp_path: Path,
) -> None:
    payload, signature = _field_checkpoint_payload()
    path = tmp_path / "field-v2.pt"
    torch.save(payload, path)
    field, restored, restored_signature = load_factorized_canonical_field_checkpoint(
        path,
        expected_sha256=sha256_file(path),
        expected_signature=signature,
    )
    assert restored["schema_version"] == 2
    assert restored_signature == signature
    assert field.reliability.shape == (2, 0)
    with pytest.raises(ValueError, match="schema-v1"):
        load_canonical_field_checkpoint(path)

    legacy = copy.deepcopy(payload)
    legacy["schema_version"] = 1
    legacy["feature_signature"] = signature.base_feature_signature.to_dict()
    legacy.pop("checkpoint_contract")
    legacy.pop("factorized_radio_metadata")
    legacy.pop("factorized_cache_sha256")
    legacy_path = tmp_path / "field-v1.pt"
    torch.save(legacy, legacy_path)
    with pytest.raises(ValueError, match="schema-v2"):
        load_factorized_canonical_field_checkpoint(legacy_path)

    leaked = copy.deepcopy(payload)
    leaked["reliability"] = torch.ones(2, 5)
    leaked_path = tmp_path / "field-leaked.pt"
    torch.save(leaked, leaked_path)
    with pytest.raises(ValueError, match="must not persist"):
        load_factorized_canonical_field_checkpoint(leaked_path)


def _mpr_payload(
    *,
    feature_space: str,
    features: torch.Tensor,
    xyz: torch.Tensor,
    official_checkpoint_sha256: str = "",
    feature_output_bundle_sha256: str = "",
) -> dict:
    rows = int(xyz.shape[0])
    valid = torch.ones(rows, dtype=torch.bool)
    counts = torch.ones(rows, dtype=torch.long)
    geometry = {"num_gaussians": rows, "xyz_sha256": _xyz_sha(xyz)}
    metadata = {
        "schema_version": 1,
        "feature_space": feature_space,
        "num_declared_views": 1,
        "selected_dataset_indices": [0],
        "selected_frame_indices": [0],
        "excluded_frame_ids": [],
        "config": "/tmp/config.yaml",
        "checkpoint": "/tmp/geometry.pt",
        "aggregation_mode": "raster_gaussian_top1",
        "registration_weight_mode": "alpha_depth",
        "raster_view_fusion": "contribution_mean",
        "raster_topk": 3,
        "depth_tolerance": 0.08,
        "relative_depth_tolerance": 0.02,
        "alpha_threshold": 0.02,
        "normalize_each_view": True,
        "per_view_normalization_applied": True,
        "per_view_normalization_stage": "pixel_feature_before_raster_lifting",
        "raster_reliability_mode": "legacy_valid",
        "registration_responsibility_cache_sha256": RESPONSIBILITY_SHA,
        "shared_registration_responsibility": True,
        "xyz_sha256": geometry["xyz_sha256"],
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "observation_lifting_contract": {
            "name": "canonical-mpr-v1",
            "feature_projection_order": "per_view_before_mpr",
            "responsibility_sharing": "exact_sidecar_across_feature_spaces",
            "query_independent": True,
        },
    }
    if feature_output_bundle_sha256:
        metadata["feature_output_bundle_sha256"] = feature_output_bundle_sha256
    if feature_space != "radio":
        metadata.update(
            {
                "capability_projection_before_mpr": True,
                "custom_adaptor_head": False,
                "capability_map_source": "project_raw",
                "official_adaptor_checkpoint_sha256": official_checkpoint_sha256,
            }
        )
    return {
        "xyz": xyz,
        "features": features.half(),
        "valid": valid,
        "view_counts": counts,
        "reliability": torch.ones(rows, 3).half(),
        "geometry_fingerprint": geometry,
        "metadata": metadata,
    }


def _exact_mpr_payload(
    *,
    feature_space: str,
    features: torch.Tensor,
    xyz: torch.Tensor,
    official_checkpoint_sha256: str = "",
) -> dict:
    payload = _mpr_payload(
        feature_space=feature_space,
        features=features,
        xyz=xyz,
        official_checkpoint_sha256=official_checkpoint_sha256,
        feature_output_bundle_sha256=BUNDLE_SHA,
    )
    lifting = canonical_observation_contract(
        CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME
    )
    metadata = payload["metadata"]
    metadata.update(
        {
            "aggregation_mode": "raster_marginal_responsibility",
            "registration_weight_mode": ("exact_front_to_back_marginal_responsibility"),
            "depth_tolerance": 0.0,
            "relative_depth_tolerance": 0.0,
            "alpha_threshold": 0.0,
            "shared_registration_responsibility": True,
            "registration_responsibility_contract": {
                "schema_version": 1,
                "assignment_mode": "exact_front_to_back_sparse_marginal",
                "registration_weight_mode": (
                    "exact_front_to_back_marginal_responsibility"
                ),
                "post_compositor_alpha_threshold": 0.0,
                "formula_sha256": (
                    FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY["formula_sha256"]
                ),
                "builder_implementation_sha256": "1" * 64,
                "authority_implementation_sha256": "2" * 64,
                "query_independent": True,
            },
            "marginal_responsibility_contract": MARGINAL_RESPONSIBILITY_CONTRACT,
            "visibility_uncertainty_semantics": (
                "per_primitive_sum_weight_times_responsibility_over_sum_weight"
            ),
            "observation_lifting_contract": lifting,
            "observation_lifting_contract_sha256": observation_contract_sha256(lifting),
            "semantic_assignment_gate": (
                "pre_adaptor_raw_radio_l2_norm_strictly_positive"
            ),
            "valid_semantics": (
                "positive_pre_adaptor_raw_radio_amplitude_responsibility_mass"
            ),
            "view_count_semantics": (
                "views_with_pre_adaptor_raw_radio_l2_norm_strictly_positive"
            ),
            "geometric_visibility_semantics": (
                "independent_exact_base_weight_authority_includes_"
                "zero_amplitude_hits"
            ),
            "geometric_view_counts_sha256": "e" * 64,
            "geometric_visible_gaussian_count": int(xyz.shape[0]),
            "semantic_valid_gaussian_count": int(xyz.shape[0]),
            "geometric_visible_semantic_invalid_gaussian_count": 0,
            "invalid_row_purity_policy": (
                "mpr_schema_v1_requires_zero_for_semantically_invalid_rows"
            ),
        }
    )
    payload["visibility_purity"] = torch.ones(int(xyz.shape[0]), dtype=torch.float16)
    if feature_space == "radio":
        metadata.update(
            {
                "capability_projection_before_mpr": False,
                "capability_map_source": "not_applicable",
                "custom_adaptor_head": False,
            }
        )
    else:
        metadata["capability_map_source"] = "project_raw"
    return payload


def test_factorized_exact_marginal_cohort_is_four_cache_fail_closed(
    tmp_path: Path,
) -> None:
    factorized_payload = _exact_factorized_payload(rows=2)
    validate_factorized_radio_training_payload(
        factorized_payload,
        expected_feature_output_bundle_sha256=BUNDLE_SHA,
    )
    factorized = _training_cache(factorized_payload)
    xyz = factorized.xyz
    paths = {
        "radio": tmp_path / "exact-raw.pt",
        "dino_v3": tmp_path / "exact-dino.pt",
        "sam3": tmp_path / "exact-sam.pt",
    }
    for name, dimension in (("radio", 3), ("dino_v3", 4), ("sam3", 5)):
        torch.save(
            _exact_mpr_payload(
                feature_space=name,
                features=torch.ones(2, dimension),
                xyz=xyz,
                official_checkpoint_sha256=(SHA if name != "radio" else ""),
            ),
            paths[name],
        )
    digests = {name: sha256_file(path) for name, path in paths.items()}
    authority = {
        "schema_version": 1,
        "artifact_type": "factorized_capability_cohort_authority",
        "experiment": "canonical-factorized-radio-v1-formal-capability-cohort",
        "feature_output_bundle_sha256": BUNDLE_SHA,
        "frozen_cache_authorities": {
            name: {"path": str(path), "sha256": digests[name]}
            for name, path in paths.items()
        },
        "target_access": {
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "target_metrics_used_for_selection": False,
        },
    }
    authority_path = tmp_path / "exact-authority.json"
    authority_path.write_text(json.dumps(authority), encoding="utf-8")

    targets, provenance, reference = _load_factorized_exact_marginal_capability_targets(
        factorized=factorized,
        reference_path=paths["radio"],
        reference_expected_sha256=digests["radio"],
        dino_path=paths["dino_v3"],
        dino_expected_sha256=digests["dino_v3"],
        sam3_path=paths["sam3"],
        sam3_expected_sha256=digests["sam3"],
        correction_path=authority_path,
        correction_expected_sha256=sha256_file(authority_path),
        radio_checkpoint_sha256=SHA,
    )
    assert set(targets) == {"dino_v3", "sam3"}
    assert provenance["dino_v3"]["target_contract"] == (
        CAPABILITY_TARGET_CONTRACT_MATCHED_EXACT_MARGINAL
    )
    assert reference["capability_cohort_authority_mode"] == ("formal_exact_marginal_v1")
    assert reference["registration_responsibility_cache_sha256"] == (RESPONSIBILITY_SHA)

    lineage_only = _exact_mpr_payload(
        feature_space="sam3",
        features=torch.ones(2, 5),
        xyz=xyz,
        official_checkpoint_sha256=SHA,
    )
    lineage_only["metadata"]["registration_responsibility_contract"].update(
        {
            "builder_implementation_sha256": "3" * 64,
            "authority_implementation_sha256": "4" * 64,
        }
    )
    lineage_only_path = tmp_path / "exact-sam-lineage-only.pt"
    torch.save(lineage_only, lineage_only_path)
    lineage_authority = copy.deepcopy(authority)
    lineage_authority["frozen_cache_authorities"]["sam3"] = {
        "path": str(lineage_only_path),
        "sha256": sha256_file(lineage_only_path),
    }
    lineage_authority_path = tmp_path / "exact-authority-lineage-only.json"
    lineage_authority_path.write_text(
        json.dumps(lineage_authority), encoding="utf-8"
    )
    lineage_targets, _, _ = _load_factorized_exact_marginal_capability_targets(
        factorized=factorized,
        reference_path=paths["radio"],
        reference_expected_sha256=digests["radio"],
        dino_path=paths["dino_v3"],
        dino_expected_sha256=digests["dino_v3"],
        sam3_path=lineage_only_path,
        sam3_expected_sha256=sha256_file(lineage_only_path),
        correction_path=lineage_authority_path,
        correction_expected_sha256=sha256_file(lineage_authority_path),
        radio_checkpoint_sha256=SHA,
    )
    assert set(lineage_targets) == {"dino_v3", "sam3"}

    material_drift = copy.deepcopy(lineage_only)
    material_drift["metadata"]["registration_responsibility_contract"][
        "post_compositor_alpha_threshold"
    ] = 0.01
    material_drift_path = tmp_path / "exact-sam-material-drift.pt"
    torch.save(material_drift, material_drift_path)
    material_authority = copy.deepcopy(authority)
    material_authority["frozen_cache_authorities"]["sam3"] = {
        "path": str(material_drift_path),
        "sha256": sha256_file(material_drift_path),
    }
    material_authority_path = tmp_path / "exact-authority-material-drift.json"
    material_authority_path.write_text(
        json.dumps(material_authority), encoding="utf-8"
    )
    with pytest.raises(
        ValueError,
        match="registration_responsibility_contract|lifting",
    ):
        _load_factorized_exact_marginal_capability_targets(
            factorized=factorized,
            reference_path=paths["radio"],
            reference_expected_sha256=digests["radio"],
            dino_path=paths["dino_v3"],
            dino_expected_sha256=digests["dino_v3"],
            sam3_path=material_drift_path,
            sam3_expected_sha256=sha256_file(material_drift_path),
            correction_path=material_authority_path,
            correction_expected_sha256=sha256_file(material_authority_path),
            radio_checkpoint_sha256=SHA,
        )

    fake_normalized_factorized_payload = _exact_factorized_payload(rows=2)
    fake_lifting = canonical_observation_contract(
        CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME
    )
    fake_normalized_factorized_payload["metadata"].update(
        {
            "observation_lifting_contract": fake_lifting,
            "observation_lifting_contract_sha256": observation_contract_sha256(
                fake_lifting
            ),
        }
    )
    with pytest.raises(ValueError, match="raw-amplitude factorized core contract"):
        _load_factorized_exact_marginal_capability_targets(
            factorized=_training_cache(fake_normalized_factorized_payload),
            reference_path=paths["radio"],
            reference_expected_sha256=digests["radio"],
            dino_path=paths["dino_v3"],
            dino_expected_sha256=digests["dino_v3"],
            sam3_path=paths["sam3"],
            sam3_expected_sha256=digests["sam3"],
            correction_path=authority_path,
            correction_expected_sha256=sha256_file(authority_path),
            radio_checkpoint_sha256=SHA,
        )

    semantic_drift = _exact_mpr_payload(
        feature_space="sam3",
        features=torch.ones(2, 5),
        xyz=xyz,
        official_checkpoint_sha256=SHA,
    )
    semantic_drift["metadata"]["semantic_assignment_gate"] = "post_adaptor_nonzero"
    semantic_drift_path = tmp_path / "exact-sam-semantic-drift.pt"
    torch.save(semantic_drift, semantic_drift_path)
    semantic_authority = copy.deepcopy(authority)
    semantic_authority["frozen_cache_authorities"]["sam3"] = {
        "path": str(semantic_drift_path),
        "sha256": sha256_file(semantic_drift_path),
    }
    semantic_authority_path = tmp_path / "exact-authority-semantic-drift.json"
    semantic_authority_path.write_text(json.dumps(semantic_authority), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic authority"):
        _load_factorized_exact_marginal_capability_targets(
            factorized=factorized,
            reference_path=paths["radio"],
            reference_expected_sha256=digests["radio"],
            dino_path=paths["dino_v3"],
            dino_expected_sha256=digests["dino_v3"],
            sam3_path=semantic_drift_path,
            sam3_expected_sha256=sha256_file(semantic_drift_path),
            correction_path=semantic_authority_path,
            correction_expected_sha256=sha256_file(semantic_authority_path),
            radio_checkpoint_sha256=SHA,
        )

    old_factorized = _training_cache(_factorized_payload(rows=2))
    old_factorized.valid[:] = True
    old_factorized.view_counts[:] = 1
    with pytest.raises(ValueError, match="builder-v2"):
        _load_factorized_exact_marginal_capability_targets(
            factorized=old_factorized,
            reference_path=paths["radio"],
            reference_expected_sha256=digests["radio"],
            dino_path=paths["dino_v3"],
            dino_expected_sha256=digests["dino_v3"],
            sam3_path=paths["sam3"],
            sam3_expected_sha256=digests["sam3"],
            correction_path=authority_path,
            correction_expected_sha256=sha256_file(authority_path),
            radio_checkpoint_sha256=SHA,
        )

    drifted = _exact_mpr_payload(
        feature_space="sam3",
        features=torch.ones(2, 5),
        xyz=xyz,
        official_checkpoint_sha256=SHA,
    )
    drifted["metadata"]["registration_responsibility_cache_sha256"] = "e" * 64
    torch.save(drifted, paths["sam3"])
    authority["frozen_cache_authorities"]["sam3"]["sha256"] = sha256_file(paths["sam3"])
    authority_path = tmp_path / "exact-authority-drifted.json"
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    with pytest.raises(ValueError, match="lifting"):
        _load_factorized_exact_marginal_capability_targets(
            factorized=factorized,
            reference_path=paths["radio"],
            reference_expected_sha256=digests["radio"],
            dino_path=paths["dino_v3"],
            dino_expected_sha256=digests["dino_v3"],
            sam3_path=paths["sam3"],
            sam3_expected_sha256=sha256_file(paths["sam3"]),
            correction_path=authority_path,
            correction_expected_sha256=sha256_file(authority_path),
            radio_checkpoint_sha256=SHA,
        )


def test_factorized_capability_receipt_bridges_only_frozen_matched_top1(
    tmp_path: Path,
) -> None:
    factorized_payload = _factorized_payload(rows=2)
    factorized_payload["factorized_radio"]["valid"][:] = True
    factorized_payload["view_counts"][:] = 1
    factorized_payload["factorized_radio"]["reliability"][:] = torch.tensor(
        [0.8, 0.2, 0.1, 0.5, 0.0]
    )
    factorized = _training_cache(factorized_payload)
    xyz = factorized.xyz
    paths = {
        "radio": tmp_path / "raw.pt",
        "dino_v3": tmp_path / "dino.pt",
        "sam3": tmp_path / "sam.pt",
    }
    torch.save(
        _mpr_payload(feature_space="radio", features=torch.ones(2, 3), xyz=xyz),
        paths["radio"],
    )
    torch.save(
        _mpr_payload(
            feature_space="dino_v3",
            features=torch.ones(2, 4),
            xyz=xyz,
            official_checkpoint_sha256=SHA,
        ),
        paths["dino_v3"],
    )
    torch.save(
        _mpr_payload(
            feature_space="sam3",
            features=torch.ones(2, 5),
            xyz=xyz,
            official_checkpoint_sha256=SHA,
        ),
        paths["sam3"],
    )
    digests = {name: sha256_file(path) for name, path in paths.items()}
    receipt = {
        "experiment": "canonical-factorized-radio-v1-label-free-source-gate",
        "frozen_cache_authorities": {
            name: {"path": str(path), "sha256": digests[name]}
            for name, path in paths.items()
        },
        "scientific_contract_unchanged": {"target_access": False},
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    targets, provenance, reference = _load_factorized_matched_capability_targets(
        factorized=factorized,
        reference_path=paths["radio"],
        reference_expected_sha256=digests["radio"],
        dino_path=paths["dino_v3"],
        dino_expected_sha256=digests["dino_v3"],
        sam3_path=paths["sam3"],
        sam3_expected_sha256=digests["sam3"],
        correction_path=receipt_path,
        correction_expected_sha256=sha256_file(receipt_path),
        radio_checkpoint_sha256=SHA,
    )
    assert set(targets) == {"dino_v3", "sam3"}
    assert provenance["dino_v3"]["historical_feature_bundle_receipt_compatibility"]
    assert reference["registration_responsibility_cache_sha256"] == RESPONSIBILITY_SHA

    wrong = copy.deepcopy(receipt)
    wrong["frozen_cache_authorities"]["sam3"]["sha256"] = "f" * 64
    wrong_path = tmp_path / "wrong-receipt.json"
    wrong_path.write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt authority"):
        _load_factorized_matched_capability_targets(
            factorized=factorized,
            reference_path=paths["radio"],
            reference_expected_sha256=digests["radio"],
            dino_path=paths["dino_v3"],
            dino_expected_sha256=digests["dino_v3"],
            sam3_path=paths["sam3"],
            sam3_expected_sha256=digests["sam3"],
            correction_path=wrong_path,
            correction_expected_sha256=sha256_file(wrong_path),
            radio_checkpoint_sha256=SHA,
        )


def test_factorized_formal_capability_cohort_binds_feature_bundle(
    tmp_path: Path,
) -> None:
    factorized_payload = _factorized_payload(rows=2)
    factorized_payload["factorized_radio"]["valid"][:] = True
    factorized_payload["view_counts"][:] = 1
    factorized_payload["factorized_radio"]["reliability"][:] = torch.tensor(
        [0.8, 0.2, 0.1, 0.5, 0.0]
    )
    factorized = _training_cache(factorized_payload)
    xyz = factorized.xyz
    paths = {
        "radio": tmp_path / "formal-raw.pt",
        "dino_v3": tmp_path / "formal-dino.pt",
        "sam3": tmp_path / "formal-sam.pt",
    }
    for name, dimension in (("radio", 3), ("dino_v3", 4), ("sam3", 5)):
        torch.save(
            _mpr_payload(
                feature_space=name,
                features=torch.ones(2, dimension),
                xyz=xyz,
                official_checkpoint_sha256=(SHA if name != "radio" else ""),
                feature_output_bundle_sha256=BUNDLE_SHA,
            ),
            paths[name],
        )
    digests = {name: sha256_file(path) for name, path in paths.items()}
    receipt = {
        "schema_version": 1,
        "artifact_type": "factorized_capability_cohort_authority",
        "experiment": "canonical-factorized-radio-v1-formal-capability-cohort",
        "feature_output_bundle_sha256": BUNDLE_SHA,
        "frozen_cache_authorities": {
            name: {"path": str(path), "sha256": digests[name]}
            for name, path in paths.items()
        },
        "target_access": {
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "target_metrics_used_for_selection": False,
        },
    }
    receipt_path = tmp_path / "formal-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    targets, provenance, reference = _load_factorized_matched_capability_targets(
        factorized=factorized,
        reference_path=paths["radio"],
        reference_expected_sha256=digests["radio"],
        dino_path=paths["dino_v3"],
        dino_expected_sha256=digests["dino_v3"],
        sam3_path=paths["sam3"],
        sam3_expected_sha256=digests["sam3"],
        correction_path=receipt_path,
        correction_expected_sha256=sha256_file(receipt_path),
        radio_checkpoint_sha256=SHA,
    )
    assert set(targets) == {"dino_v3", "sam3"}
    assert provenance["dino_v3"]["formal_feature_bundle_authority"] is True
    assert (
        provenance["dino_v3"]["historical_feature_bundle_receipt_compatibility"]
        is False
    )
    assert reference["capability_cohort_authority_mode"] == "formal_feature_bundle_v1"
    assert reference["feature_output_bundle_sha256"] == BUNDLE_SHA

    mismatched = copy.deepcopy(receipt)
    mismatched["feature_output_bundle_sha256"] = "e" * 64
    mismatch_path = tmp_path / "formal-receipt-mismatch.json"
    mismatch_path.write_text(json.dumps(mismatched), encoding="utf-8")
    with pytest.raises(ValueError, match="formal factorized capability receipt"):
        _load_factorized_matched_capability_targets(
            factorized=factorized,
            reference_path=paths["radio"],
            reference_expected_sha256=digests["radio"],
            dino_path=paths["dino_v3"],
            dino_expected_sha256=digests["dino_v3"],
            sam3_path=paths["sam3"],
            sam3_expected_sha256=digests["sam3"],
            correction_path=mismatch_path,
            correction_expected_sha256=sha256_file(mismatch_path),
            radio_checkpoint_sha256=SHA,
        )

    wrong_sam_path = tmp_path / "formal-sam-wrong-bundle.pt"
    torch.save(
        _mpr_payload(
            feature_space="sam3",
            features=torch.ones(2, 5),
            xyz=xyz,
            official_checkpoint_sha256=SHA,
            feature_output_bundle_sha256="e" * 64,
        ),
        wrong_sam_path,
    )
    wrong_sam_receipt = copy.deepcopy(receipt)
    wrong_sam_receipt["frozen_cache_authorities"]["sam3"] = {
        "path": str(wrong_sam_path),
        "sha256": sha256_file(wrong_sam_path),
    }
    wrong_sam_receipt_path = tmp_path / "formal-receipt-wrong-sam.json"
    wrong_sam_receipt_path.write_text(json.dumps(wrong_sam_receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="formal factorized sam3 capability bundle"):
        _load_factorized_matched_capability_targets(
            factorized=factorized,
            reference_path=paths["radio"],
            reference_expected_sha256=digests["radio"],
            dino_path=paths["dino_v3"],
            dino_expected_sha256=digests["dino_v3"],
            sam3_path=wrong_sam_path,
            sam3_expected_sha256=sha256_file(wrong_sam_path),
            correction_path=wrong_sam_receipt_path,
            correction_expected_sha256=sha256_file(wrong_sam_receipt_path),
            radio_checkpoint_sha256=SHA,
        )


def test_factorized_trainer_cpu_smoke_writes_v2_without_target_reliability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _factorized_payload(rows=3)
    payload["factorized_radio"]["valid"][:] = True
    payload["view_counts"][:] = 1
    payload["factorized_radio"]["canonical_feature"].zero_()
    payload["factorized_radio"]["canonical_feature"][0, 0] = 2.0
    payload["factorized_radio"]["canonical_feature"][1, 1] = 3.0
    payload["factorized_radio"]["canonical_feature"][2, 0] = 2.5
    payload["factorized_radio"]["log_amplitude"][:] = torch.log(
        torch.tensor([2.0, 3.0, 2.5])
    )
    payload["factorized_radio"]["reliability"][:] = torch.tensor(
        [0.8, 0.2, 0.1, 0.5, 0.0]
    )
    target = _training_cache(payload)
    capability = target.as_consensus()
    monkeypatch.setattr(
        trainer, "load_factorized_radio_training_cache", lambda *args, **kwargs: target
    )
    monkeypatch.setattr(
        trainer,
        "_load_factorized_matched_capability_targets",
        lambda **kwargs: (
            {"dino_v3": capability, "sam3": capability},
            {
                "dino_v3": {"sha256": "d" * 64},
                "sam3": {"sha256": "e" * 64},
            },
            {"sha256": "f" * 64},
        ),
    )

    class IdentityViews:
        @classmethod
        def from_radio_checkpoint(cls, *args, **kwargs):
            return cls()

        def to(self, device):
            return self

        def project_dino_primitives(self, values):
            return values

        def project_sam3_primitives(self, values):
            return values

    monkeypatch.setattr(trainer, "FrozenRadioViews", IdentityViews)
    radio_checkpoint = tmp_path / "radio.pt"
    radio_checkpoint.write_bytes(b"frozen-radio")
    output = tmp_path / "field.pt"
    args = SimpleNamespace(
        seed=0,
        device="cpu",
        observation_contract="canonical-factorized-radio-v1",
        capability_target_contract="matched_top1",
        fusion_reliability=False,
        relation_objective="disabled",
        relation_weight=0.0,
        relation_triplet_cache="",
        expected_relation_triplet_cache_sha256="",
        field_b_experiment_registration="",
        expected_field_b_experiment_registration_sha256="",
        mpr_cache="factorized.pt",
        expected_mpr_cache_sha256=SHA,
        expected_feature_output_bundle_sha256=BUNDLE_SHA,
        radio_checkpoint=str(radio_checkpoint),
        expected_radio_checkpoint_sha256=sha256_file(radio_checkpoint),
        official_capability_loss=True,
        dino_mpr_cache="dino.pt",
        expected_dino_v3_mpr_cache_sha256="d" * 64,
        sam3_mpr_cache="sam.pt",
        expected_sam3_mpr_cache_sha256="e" * 64,
        factorized_capability_reference_mpr_cache="raw.pt",
        expected_factorized_capability_reference_mpr_cache_sha256="f" * 64,
        factorized_capability_legacy_receipt_correction="receipt.json",
        expected_factorized_capability_legacy_receipt_correction_sha256="1" * 64,
        capability_observation_reference_mpr_cache="",
        expected_capability_observation_reference_mpr_cache_sha256="",
        radio_version="test",
        initial_field_checkpoint="",
        expected_initial_field_checkpoint_sha256="",
        coefficient_dim=1,
        pca_samples=3,
        no_standardize=True,
        freeze_basis=True,
        local_dim=0,
        primitive_fusion=False,
        spatial_coarse_dim=0,
        hash_levels=2,
        hash_features_per_level=2,
        hash_log2_size=4,
        hash_base_resolution=2,
        hash_max_resolution=4,
        hash_hidden_dim=4,
        hidden_dim=8,
        fusion_residual_blocks=0,
        mpr_weight=1.0,
        dino_weight=0.2,
        sam3_weight=0.2,
        coefficient_weight=0.0,
        basis_orthogonality_weight=0.0,
        epochs=1,
        min_epochs=1,
        batch_size=2,
        eval_batch_size=3,
        learning_rate=1e-3,
        weight_decay=0.0,
        validation_fraction=0.34,
        target_cosine=-1.0,
        output=str(output),
    )
    report = trainer.train(args)
    saved = torch.load(output, map_location="cpu", weights_only=False)
    assert saved["schema_version"] == 2
    assert saved["architecture"]["fusion_reliability"] is False
    assert saved["reliability"].shape == (3, 0)
    assert saved["state_dict"]["reliability"].shape == (3, 0)
    assert (
        saved["factorized_loss_contract"]["target_reliability_columns_entered_field"]
        is False
    )
    for name in (
        "mean_abs_log_amplitude_error",
        "p95_abs_log_amplitude_error",
        "predicted_norm_median",
        "target_norm_median",
    ):
        assert name in saved["final_metrics"]
        assert torch.isfinite(torch.tensor(saved["final_metrics"][name]))
    assert report["factorized_field_signature"]["schema_version"] == 1


@pytest.mark.parametrize("existing_suffix", ["", ".json"])
def test_trainer_refuses_existing_checkpoint_or_report_before_loading(
    tmp_path: Path, existing_suffix: str
) -> None:
    output = tmp_path / "field.pth"
    existing = Path(f"{output}{existing_suffix}")
    existing.write_bytes(b"immutable-existing-artifact")
    with pytest.raises(FileExistsError, match="refuse to overwrite"):
        trainer.train(SimpleNamespace(output=str(output)))
