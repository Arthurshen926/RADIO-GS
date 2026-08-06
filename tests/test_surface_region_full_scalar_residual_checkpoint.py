from __future__ import annotations

import copy

import pytest
import torch

from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    sha256_file,
    write_frozen_json,
)

from radio_gs.interfaces.surface_region_full_scalar_contract import (
    SURFACE_REGION_FULL_SCALAR_DIM,
    build_full_scalar_normalization_authority,
)
from radio_gs.interfaces.full_scalar_sparse_teacher_selection import (
    REGION_CAP_PER_SCENE,
    SAMPLING_CONTRACT_SHA256,
    SPARSE_V2_PREREG_FILE_SHA256,
    VIEW_CAP_PER_REGION,
)
from radio_gs.interfaces.surface_region_full_scalar_residual_checkpoint import (
    SURFACE_REGION_FULL_SCALAR_RESIDUAL_CHECKPOINT_CONTRACT_SHA256,
    build_surface_region_full_scalar_residual_checkpoint_payload,
    load_surface_region_full_scalar_residual_checkpoint,
    surface_region_full_scalar_residual_checkpoint_contract,
    validate_surface_region_full_scalar_residual_checkpoint_payload,
    write_surface_region_full_scalar_residual_checkpoint,
)
from radio_gs.interfaces.surface_region_full_scalar_training_certificate import (
    build_training_certificate_payload,
    load_training_certificate,
    training_certificate_content_sha256,
    validate_training_certificate_payload,
)
from radio_gs.interfaces.surface_region_summary import (
    ACCEPTED_SURFACE_REGION_V2_ARCHITECTURE_SHA256,
    ACCEPTED_SURFACE_REGION_V2_CHECKPOINT_SHA256,
    ACCEPTED_SURFACE_REGION_V2_CONTRACT_SHA256,
    ACCEPTED_SURFACE_REGION_V2_PROVENANCE_SHA256,
    ACCEPTED_SURFACE_REGION_V2_STATE_DICT_SHA256,
    surface_region_state_dict_sha256,
)
from radio_gs.models.surface_region_dual_descriptor import (
    SurfaceRegionAcceptedV2FullScalarResidualV1,
)
from radio_gs.scripts import train_surface_region_full_scalar_residual as trainer


COHORT_SHA = "a" * 64
NORMALIZATION_SHA = "b" * 64
CERTIFICATE_SHA = "c" * 64


def _unit_training_contract() -> dict:
    return trainer.training_contract()


def _unit_scene_coverage() -> dict:
    scenes = [f"validation_{index}" for index in range(8)]
    return {
        "expected_scenes": scenes,
        "scene_count": 8,
        "minimum_rows_per_scene": 2,
        "per_scene_rows": {scene: 2 for scene in scenes},
        "covered_scene_count": 8,
        "missing_or_insufficient_scenes": [],
        "passed": True,
    }


def _unit_selected_validation() -> dict:
    return {
        "aggregation": "scene_macro",
        "non_regression_passed": True,
        "eligible_scene_coverage": _unit_scene_coverage(),
        "in_domain_scene_coverage": _unit_scene_coverage(),
        "in_domain": {
            "non_regression_passed": True,
            "vacuous_fallback_only": False,
            "row_count": 16,
        },
        "eligible_rows": 16,
        "trained_or_residual_eligible_rows": 16,
    }


def _unit_sampling_authority() -> dict:
    def record(scene: str) -> dict:
        return {
            "scene_id": scene,
            "canonical_candidate_region_count": 4,
            "exact_overlap_candidate_count": 4,
            "teacher_visible_candidate_count": 2,
            "selected_region_count": 2,
            "selected_count_by_scale": [2],
            "teacher_pair_count": 2,
            "maximum_views_per_region": 1,
        }

    return {
        "sampling_contract_sha256": SAMPLING_CONTRACT_SHA256,
        "preregistration_file_sha256": SPARSE_V2_PREREG_FILE_SHA256,
        "per_scene_region_cap": REGION_CAP_PER_SCENE,
        "per_region_view_cap": VIEW_CAP_PER_REGION,
        "storage": "sparse_coo_pairs_plus_merged_csr_offsets",
        "global_or_cohort_teacher_densification": False,
        "batch_local_gather_only": True,
        "scene_records_by_split": {
            "source_train": [record(f"train_{index:02d}") for index in range(24)],
            "source_validation": [
                record(f"validation_{index}") for index in range(8)
            ],
        },
    }


def _normalization_authority() -> dict:
    values = torch.stack(
        (
            torch.zeros(SURFACE_REGION_FULL_SCALAR_DIM),
            torch.ones(SURFACE_REGION_FULL_SCALAR_DIM),
            torch.full((SURFACE_REGION_FULL_SCALAR_DIM,), 2.0),
        )
    )
    return build_full_scalar_normalization_authority(
        values,
        torch.ones(3, dtype=torch.bool),
        source_state_cohort_sha256=COHORT_SHA,
    )


def _model(authority: dict, *, descriptor_dim: int = 8):
    return SurfaceRegionAcceptedV2FullScalarResidualV1(
        descriptor_dim=descriptor_dim,
        scalar_median=authority["median"],
        scalar_robust_scale=authority["robust_scale"],
    )


def _certificate(model) -> dict:
    architecture = model.architecture()
    state_sha = surface_region_state_dict_sha256(model.state_dict())
    return build_training_certificate_payload(
        training_contract=_unit_training_contract(),
        model_authority={
            "class": type(model).__name__,
            "architecture": architecture,
            "architecture_sha256": canonical_json_sha256(architecture),
            "state_dict_sha256": state_sha,
        },
        normalization_authority={"path": "/unit/normalization.pt", "sha256": NORMALIZATION_SHA},
        cohort_authority={
            "file": {"path": "/unit/cohort.json", "sha256": "d" * 64},
            "authority_sha256": "e" * 64,
        },
        source_state_manifest={
            "file": {"path": "/unit/source.json", "sha256": "f" * 64},
            "authority_sha256": COHORT_SHA,
        },
        teacher_manifest={
            "file": {"path": "/unit/teacher.json", "sha256": "1" * 64},
            "authority_sha256": "2" * 64,
        },
        benchmark_exclusion_manifest={
            "file": {"path": "/unit/exclusion.json", "sha256": "3" * 64},
            "authority_sha256": "4" * 64,
        },
        input_shards={
            "source_train": [{"path": "/unit/train.pt", "sha256": "5" * 64}],
            "source_validation": [{"path": "/unit/val.pt", "sha256": "6" * 64}],
        },
        sampling_authority=_unit_sampling_authority(),
        selected_epoch=0,
        selected_validation=_unit_selected_validation(),
    )


def _payload(model=None, authority=None):
    authority = authority or _normalization_authority()
    model = model or _model(authority)
    certificate = (
        _certificate(model)
        if type(model) is SurfaceRegionAcceptedV2FullScalarResidualV1
        else {}
    )
    return build_surface_region_full_scalar_residual_checkpoint_payload(
        model,
        normalization_authority=authority,
        normalization_authority_sha256=NORMALIZATION_SHA,
        source_state_cohort_authority_sha256=COHORT_SHA,
        training_certificate=certificate,
        training_certificate_sha256=CERTIFICATE_SHA,
    )


def _certificate_for_payload(payload):
    return build_training_certificate_payload(
        training_contract=_unit_training_contract(),
        model_authority={
            "class": payload["model_class"],
            "architecture": payload["model_architecture"],
            "architecture_sha256": payload["model_architecture_sha256"],
            "state_dict_sha256": payload["model_state_dict_sha256"],
        },
        normalization_authority={"path": "/unit/normalization.pt", "sha256": NORMALIZATION_SHA},
        cohort_authority={"file": {"path": "/unit/cohort.json", "sha256": "d" * 64}, "authority_sha256": "e" * 64},
        source_state_manifest={"file": {"path": "/unit/source.json", "sha256": "f" * 64}, "authority_sha256": COHORT_SHA},
        teacher_manifest={"file": {"path": "/unit/teacher.json", "sha256": "1" * 64}, "authority_sha256": "2" * 64},
        benchmark_exclusion_manifest={"file": {"path": "/unit/exclusion.json", "sha256": "3" * 64}, "authority_sha256": "4" * 64},
        input_shards={"source_train": [{"path": "/unit/train.pt", "sha256": "5" * 64}], "source_validation": [{"path": "/unit/val.pt", "sha256": "6" * 64}]},
        sampling_authority=_unit_sampling_authority(),
        selected_epoch=0,
        selected_validation=_unit_selected_validation(),
    )


def _validate(payload, authority):
    return validate_surface_region_full_scalar_residual_checkpoint_payload(
        payload,
        normalization_authority=authority,
        expected_normalization_authority_sha256=NORMALIZATION_SHA,
        expected_source_state_cohort_authority_sha256=COHORT_SHA,
        training_certificate=_certificate_for_payload(payload),
        expected_training_certificate_sha256=CERTIFICATE_SHA,
    )


def test_checkpoint_contract_freezes_all_accepted_v2_authorities() -> None:
    contract = surface_region_full_scalar_residual_checkpoint_contract()
    assert contract["immutable_accepted_v2_authority"] == {
        "checkpoint_sha256": ACCEPTED_SURFACE_REGION_V2_CHECKPOINT_SHA256,
        "architecture_sha256": ACCEPTED_SURFACE_REGION_V2_ARCHITECTURE_SHA256,
        "state_dict_sha256": ACCEPTED_SURFACE_REGION_V2_STATE_DICT_SHA256,
        "provenance_sha256": ACCEPTED_SURFACE_REGION_V2_PROVENANCE_SHA256,
        "contract_sha256": ACCEPTED_SURFACE_REGION_V2_CONTRACT_SHA256,
    }
    assert len(SURFACE_REGION_FULL_SCALAR_RESIDUAL_CHECKPOINT_CONTRACT_SHA256) == 64


def test_payload_reconstructs_only_allowed_model_and_binds_buffers() -> None:
    authority = _normalization_authority()
    payload = _payload(authority=authority)
    validated, model = _validate(payload, authority)
    assert validated["model_class"] == (
        "SurfaceRegionAcceptedV2FullScalarResidualV1"
    )
    assert torch.equal(model.scalar_median, authority["median"])
    assert torch.equal(model.scalar_robust_scale, authority["robust_scale"])
    assert payload["checkpoint_state_assertions"] == {
        "residual_projection_exact_zero": True,
        "exact_base_identity_at_checkpoint": True,
        "identity_proof": "zero_residual_projection_structural",
    }


def test_checkpoint_write_is_no_clobber_and_load_requires_expected_sha(
    tmp_path,
) -> None:
    authority = _normalization_authority()
    path = tmp_path / "residual.pt"
    model = _model(authority)
    certificate = _certificate(model)
    written, digest = write_surface_region_full_scalar_residual_checkpoint(
        path,
        model,
        normalization_authority=authority,
        normalization_authority_sha256=NORMALIZATION_SHA,
        source_state_cohort_authority_sha256=COHORT_SHA,
        training_certificate=certificate,
        training_certificate_sha256=CERTIFICATE_SHA,
    )
    assert written == path
    loaded, payload = load_surface_region_full_scalar_residual_checkpoint(
        path,
        expected_checkpoint_sha256=digest,
        normalization_authority=authority,
        expected_normalization_authority_sha256=NORMALIZATION_SHA,
        expected_source_state_cohort_authority_sha256=COHORT_SHA,
        training_certificate=certificate,
        expected_training_certificate_sha256=CERTIFICATE_SHA,
        map_location="cpu",
    )
    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    assert payload["source_authority"]["source_state_cohort_authority_sha256"] == (
        COHORT_SHA
    )
    with pytest.raises(ValueError, match="SHA-256 differs"):
        load_surface_region_full_scalar_residual_checkpoint(
            path,
            expected_checkpoint_sha256="0" * 64,
            normalization_authority=authority,
            expected_normalization_authority_sha256=NORMALIZATION_SHA,
            expected_source_state_cohort_authority_sha256=COHORT_SHA,
            training_certificate=certificate,
            expected_training_certificate_sha256=CERTIFICATE_SHA,
        )
    with pytest.raises(FileExistsError, match="already exists"):
        write_surface_region_full_scalar_residual_checkpoint(
            path,
                model,
            normalization_authority=authority,
            normalization_authority_sha256=NORMALIZATION_SHA,
            source_state_cohort_authority_sha256=COHORT_SHA,
            training_certificate=certificate,
            training_certificate_sha256=CERTIFICATE_SHA,
        )


def test_exact_keys_and_accepted_v2_authority_fail_closed() -> None:
    authority = _normalization_authority()
    payload = _payload(authority=authority)
    extra = copy.deepcopy(payload)
    extra["legacy_fallback"] = True
    with pytest.raises(ValueError, match="fields differ"):
        _validate(extra, authority)

    changed = copy.deepcopy(payload)
    changed["accepted_v2_authority"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="contract differs"):
        _validate(changed, authority)


def test_architecture_and_state_dict_digests_are_recomputed() -> None:
    authority = _normalization_authority()
    payload = _payload(authority=authority)
    changed_architecture = copy.deepcopy(payload)
    changed_architecture["model_architecture"]["max_alpha"] = 0.5
    with pytest.raises(ValueError, match="model architecture|architecture authority"):
        _validate(changed_architecture, authority)

    changed_state = copy.deepcopy(payload)
    changed_state["model_state_dict"]["scalar_projection.bias"][0] = 1.0
    with pytest.raises(ValueError, match="model state authority"):
        _validate(changed_state, authority)


def test_normalization_buffers_are_bitwise_bound_even_with_rehashed_state() -> None:
    authority = _normalization_authority()
    payload = _payload(authority=authority)
    changed = copy.deepcopy(payload)
    changed["model_state_dict"]["scalar_median"][0] += 1.0
    changed["model_state_dict_sha256"] = surface_region_state_dict_sha256(
        changed["model_state_dict"]
    )
    with pytest.raises(ValueError, match="scalar_median"):
        _validate(changed, authority)

    wrong_model = _model(authority)
    wrong_model.scalar_robust_scale[0] += 1.0
    with pytest.raises(ValueError, match="scalar_robust_scale"):
        _payload(wrong_model, authority)


def test_identity_markers_are_recomputed_from_final_projection() -> None:
    authority = _normalization_authority()
    zero_payload = _payload(authority=authority)
    forged_zero = copy.deepcopy(zero_payload)
    forged_zero["checkpoint_state_assertions"][
        "exact_base_identity_at_checkpoint"
    ] = False
    with pytest.raises(ValueError, match="identity assertion"):
        _validate(forged_zero, authority)

    trained = _model(authority)
    trained.residual_projection.bias.data[0] = 1.0
    trained_payload = _payload(trained, authority)
    assert not trained_payload["checkpoint_state_assertions"][
        "exact_base_identity_at_checkpoint"
    ]
    forged_trained = copy.deepcopy(trained_payload)
    forged_trained["checkpoint_state_assertions"] = copy.deepcopy(
        zero_payload["checkpoint_state_assertions"]
    )
    with pytest.raises(ValueError, match="identity assertion"):
        _validate(forged_trained, authority)


def test_source_cohort_and_normalization_sha_are_independent_authorities() -> None:
    authority = _normalization_authority()
    payload = _payload(authority=authority)
    with pytest.raises(ValueError, match="source authority"):
        validate_surface_region_full_scalar_residual_checkpoint_payload(
            payload,
            normalization_authority=authority,
            expected_normalization_authority_sha256="c" * 64,
            expected_source_state_cohort_authority_sha256=COHORT_SHA,
            training_certificate=_certificate_for_payload(payload),
            expected_training_certificate_sha256=CERTIFICATE_SHA,
        )
    changed_authority = copy.deepcopy(authority)
    changed_authority["source_state_cohort_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="cohort authority"):
        _validate(payload, changed_authority)


def test_subclass_is_not_an_alternate_checkpoint_model() -> None:
    authority = _normalization_authority()

    class Alternate(SurfaceRegionAcceptedV2FullScalarResidualV1):
        pass

    alternate = Alternate(
        descriptor_dim=8,
        scalar_median=authority["median"],
        scalar_robust_scale=authority["robust_scale"],
    )
    with pytest.raises(TypeError, match="only"):
        _payload(alternate, authority)


def test_checkpoint_rejects_valid_certificate_for_a_different_model_state() -> None:
    authority = _normalization_authority()
    model = _model(authority)
    payload = _payload(model, authority)
    certificate = _certificate_for_payload(payload)
    certificate["model_authority"]["state_dict_sha256"] = "9" * 64
    certificate["content_sha256"] = training_certificate_content_sha256(
        certificate
    )
    with pytest.raises(ValueError, match="training certificate model state"):
        validate_surface_region_full_scalar_residual_checkpoint_payload(
            payload,
            normalization_authority=authority,
            expected_normalization_authority_sha256=NORMALIZATION_SHA,
            expected_source_state_cohort_authority_sha256=COHORT_SHA,
            training_certificate=certificate,
            expected_training_certificate_sha256=CERTIFICATE_SHA,
        )


def test_training_certificate_recomputes_nonvacuous_scene_coverage() -> None:
    certificate = _certificate(_model(_normalization_authority()))
    forged = copy.deepcopy(certificate)
    coverage = forged["selected_validation"]["in_domain_scene_coverage"]
    coverage["per_scene_rows"]["validation_0"] = 1
    # Retain the optimistic booleans to prove the validator derives coverage
    # from the signed counts instead of trusting the declaration.
    forged["content_sha256"] = training_certificate_content_sha256(forged)

    with pytest.raises(ValueError, match="in-domain-scene coverage"):
        validate_training_certificate_payload(forged)


def test_training_certificate_rejects_self_hashed_contract_or_sampling_drift() -> None:
    certificate = _certificate(_model(_normalization_authority()))
    forged_contract = copy.deepcopy(certificate)
    forged_contract["training_contract"]["schema_version"] = 99
    forged_contract["training_contract_sha256"] = canonical_json_sha256(
        forged_contract["training_contract"]
    )
    forged_contract["content_sha256"] = training_certificate_content_sha256(
        forged_contract
    )
    with pytest.raises(ValueError, match="training contract differs"):
        validate_training_certificate_payload(forged_contract)

    forged_sampling = copy.deepcopy(certificate)
    forged_sampling["sampling_authority"]["batch_local_gather_only"] = False
    forged_sampling["content_sha256"] = training_certificate_content_sha256(
        forged_sampling
    )
    with pytest.raises(ValueError, match="sampling authority differs"):
        validate_training_certificate_payload(forged_sampling)


def test_training_certificate_file_requires_caller_sha(tmp_path) -> None:
    certificate = _certificate(_model(_normalization_authority()))
    path = tmp_path / "certificate.json"
    write_frozen_json(path, certificate)
    digest = sha256_file(path)
    loaded, record = load_training_certificate(path, expected_sha256=digest)
    assert loaded["content_sha256"] == certificate["content_sha256"]
    assert record == {"path": str(path.resolve()), "sha256": digest}
    with pytest.raises(ValueError, match="SHA-256 differs"):
        load_training_certificate(path, expected_sha256="0" * 64)
