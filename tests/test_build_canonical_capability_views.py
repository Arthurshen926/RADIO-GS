from argparse import Namespace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from radio_gs.field import FeatureSpaceSignature
from radio_gs.field.factorized_radio_contract import FactorizedRadioFieldSignature
import radio_gs.scripts.build_canonical_capability_views as capability_builder
from radio_gs.scripts.build_canonical_capability_views import (
    FACTORIZED_FIELD_CHECKPOINT_SCHEMA_V2,
    _formal_capability_training_authority,
    _load_field_and_support,
    _validate_compatible_legacy_observation,
)


def test_compatible_legacy_capability_observation_is_narrow_and_query_free() -> None:
    metadata = {
        "construction": "dominant_primary_with_query_free_support_completion",
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    _validate_compatible_legacy_observation(metadata)
    with pytest.raises(ValueError, match="construction"):
        _validate_compatible_legacy_observation({**metadata, "construction": "other"})
    with pytest.raises(ValueError, match="query-independent"):
        _validate_compatible_legacy_observation(
            {**metadata, "text_queries_opened": True}
        )


def test_factorized_capability_source_uses_only_explicit_schema_v2_loaders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = FeatureSpaceSignature(
        radio_version="test",
        radio_checkpoint_sha256="a" * 64,
        raw_feature_dim=1280,
        token_type="primitive",
        normalization="radio_raw_full",
        crop_policy="training_views_canonical_factorized_radio_v1",
    )
    signature = FactorizedRadioFieldSignature.create(base)
    field = SimpleNamespace(num_gaussians=3, signature=base)
    geometry = {"num_gaussians": 3, "xyz_sha256": "b" * 64}
    cache_sha = "c" * 64
    bundle_sha = "d" * 64
    payload = {
        "mpr_cache": str(tmp_path / "factorized.pt"),
        "mpr_cache_sha256": cache_sha,
        "factorized_cache_sha256": cache_sha,
        "feature_output_bundle_sha256": bundle_sha,
        "geometry_fingerprint": geometry,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    factorized = SimpleNamespace(
        xyz=torch.zeros(3, 3),
        valid=torch.tensor([True, False, True]),
        geometry_fingerprint=geometry,
        sha256=cache_sha,
        source=tmp_path / "factorized.pt",
        metadata={"registration_responsibility_cache_sha256": "f" * 64},
    )

    def reject_legacy(*_args, **_kwargs):
        raise AssertionError("schema-v2 must not fall back to the schema-v1 loader")

    monkeypatch.setattr(
        capability_builder, "load_canonical_field_checkpoint", reject_legacy
    )
    support = SimpleNamespace(
        field=field,
        field_payload=payload,
        field_signature=signature,
        cache=factorized,
        lineage={
            "field_checkpoint_schema_version": 2,
            "factorized_radio_field_signature_sha256": signature.digest,
            "factorized_radio_cache_sha256": cache_sha,
            "mpr_geometry_fingerprint": geometry,
            "registration_responsibility_cache_sha256": "f" * 64,
        },
    )
    monkeypatch.setattr(
        capability_builder,
        "load_factorized_field_support",
        lambda *_args, **_kwargs: support,
    )
    args = Namespace(
        field_checkpoint=str(tmp_path / "field.pt"),
        field_checkpoint_schema=FACTORIZED_FIELD_CHECKPOINT_SCHEMA_V2,
        observation_contract="canonical",
        mpr_cache="",
        expected_mpr_cache_sha256="",
    )
    loaded = _load_field_and_support(
        args,
        field_checkpoint_sha256="e" * 64,
    )
    assert loaded[4] == cache_sha
    assert loaded[6] == "canonical-factorized-radio-v1"
    assert loaded[7]["field_checkpoint_schema_version"] == 2
    assert loaded[7]["mpr_geometry_fingerprint"] == geometry
    assert loaded[7]["registration_responsibility_cache_sha256"] == "f" * 64


def test_factorized_capability_source_rejects_legacy_observation_mode() -> None:
    args = Namespace(
        field_checkpoint="field.pt",
        field_checkpoint_schema=FACTORIZED_FIELD_CHECKPOINT_SCHEMA_V2,
        observation_contract="compatible-legacy",
        mpr_cache="",
        expected_mpr_cache_sha256="",
    )
    with pytest.raises(ValueError, match="strict canonical"):
        _load_field_and_support(args, field_checkpoint_sha256="e" * 64)


def test_formal_capability_training_authority_preserves_exact_sources(
    tmp_path: Path,
) -> None:
    paths = {}
    for name in ("dino", "sam", "raw", "cohort"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        paths[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    radio_sha = "a" * 64
    responsibility_sha = "b" * 64
    bundle_sha = "c" * 64
    mode = "official_adaptor_then_shared_exact_marginal_mpr"
    contract = "matched_exact_marginal"
    targets = {}
    for name, record in (("dino_v3", paths["dino"]), ("sam3", paths["sam"])):
        targets[name] = {
            **record,
            "feature_space": name,
            "projection_order": mode,
            "target_contract": contract,
            "uses_query_or_benchmark_supervision": False,
            "official_adaptor_checkpoint_sha256": radio_sha,
            "registration_responsibility_cache_sha256": responsibility_sha,
            "feature_output_bundle_sha256": bundle_sha,
        }
    payload = {
        "mpr_cache_sha256": "d" * 64,
        "capability_target_mode": mode,
        "capability_target_contract": contract,
        "capability_mpr_targets": targets,
        "capability_observation_reference": {
            **paths["raw"],
            "capability_cohort_authority": paths["cohort"],
            "capability_cohort_authority_mode": "formal_exact_marginal_v1",
            "registration_responsibility_cache_sha256": responsibility_sha,
            "feature_output_bundle_sha256": bundle_sha,
            "uses_query_or_benchmark_supervision": False,
        },
    }
    authority = _formal_capability_training_authority(
        payload,
        expected_radio_checkpoint_sha256=radio_sha,
        expected_factorized_radio_cache_sha256="d" * 64,
    )
    assert authority["exact_source_capabilities"]["appearance"]["sha256"] == paths[
        "dino"
    ]["sha256"]
    assert authority["exact_source_capabilities"]["boundary"]["sha256"] == paths[
        "sam"
    ]["sha256"]
    assert authority["capability_cohort_authority"]["sha256"] == paths["cohort"][
        "sha256"
    ]


def _formal_feature_bundle_payload(tmp_path: Path) -> tuple[dict, dict[str, dict]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, dict] = {}
    for name in ("dino", "sam", "raw"):
        path = tmp_path / f"{name}.pt"
        path.write_bytes(name.encode("ascii"))
        paths[name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    radio_sha = "a" * 64
    responsibility_sha = "b" * 64
    bundle_sha = "c" * 64
    geometry_checkpoint_sha = "d" * 64
    mode = "official_adaptor_then_geometry_matched_mpr"
    contract = "matched_top1"
    targets = {}
    for name, record in (("dino_v3", paths["dino"]), ("sam3", paths["sam"])):
        targets[name] = {
            **record,
            "feature_space": name,
            "projection_order": mode,
            "target_contract": contract,
            "uses_query_or_benchmark_supervision": False,
            "official_adaptor_checkpoint_sha256": radio_sha,
            "registration_responsibility_cache_sha256": responsibility_sha,
            "feature_output_bundle_sha256": bundle_sha,
            "formal_feature_bundle_authority": True,
            "historical_feature_bundle_receipt_compatibility": False,
        }
    cohort_path = tmp_path / "cohort.json"
    cohort = {
        "schema_version": 1,
        "artifact_type": "factorized_capability_cohort_authority",
        "experiment": "canonical-factorized-radio-v1-formal-capability-cohort",
        "feature_output_bundle_sha256": bundle_sha,
        "frozen_cache_authorities": {
            "radio": paths["raw"],
            "dino_v3": paths["dino"],
            "sam3": paths["sam"],
        },
        "storage_authority": {
            "schema": "radio_gs.channel_sharded_mpr.v1",
            "shard_channels": 256,
            "formal_loader_all_hash_gates": "passed",
        },
        "shared_lineage": {
            "num_gaussians": 3,
            "valid_count": 2,
            "registration_responsibility_cache_sha256": responsibility_sha,
            "geometry_checkpoint_sha256": geometry_checkpoint_sha,
            "official_radio_checkpoint_sha256": radio_sha,
        },
        "target_access": {
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "target_metrics_used_for_selection": False,
        },
    }
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
    paths["cohort"] = {
        "path": str(cohort_path.resolve()),
        "sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
    }
    payload = {
        "mpr_cache_sha256": "e" * 64,
        "geometry_fingerprint": {"num_gaussians": 3, "xyz_sha256": "f" * 64},
        "capability_target_mode": mode,
        "capability_target_contract": contract,
        "capability_mpr_targets": targets,
        "capability_observation_reference": {
            **paths["raw"],
            "legacy_receipt_correction": paths["cohort"],
            "capability_cohort_authority_mode": "formal_feature_bundle_v1",
            "registration_responsibility_cache_sha256": responsibility_sha,
            "feature_output_bundle_sha256": bundle_sha,
            "uses_query_or_benchmark_supervision": False,
            "formal_feature_bundle_authority": True,
            "historical_feature_bundle_receipt_compatibility": False,
        },
    }
    return payload, paths


def test_formal_feature_bundle_authority_is_hash_bound_and_source_only(
    tmp_path: Path,
) -> None:
    payload, paths = _formal_feature_bundle_payload(tmp_path)
    authority = _formal_capability_training_authority(
        payload,
        expected_radio_checkpoint_sha256="a" * 64,
        expected_factorized_radio_cache_sha256="e" * 64,
    )
    assert authority["capability_cohort_authority_mode"] == (
        "formal_feature_bundle_v1"
    )
    assert authority["source"] == (
        "formal_feature_bundle_capability_training_authority_v1"
    )
    assert authority["capability_cohort_authority"]["sha256"] == paths["cohort"][
        "sha256"
    ]


def test_formal_feature_bundle_authority_rejects_mixing_and_content_drift(
    tmp_path: Path,
) -> None:
    payload, paths = _formal_feature_bundle_payload(tmp_path)
    reference = payload["capability_observation_reference"]
    reference["capability_cohort_authority"] = paths["cohort"]
    with pytest.raises(ValueError, match="mixed"):
        _formal_capability_training_authority(
            payload,
            expected_radio_checkpoint_sha256="a" * 64,
            expected_factorized_radio_cache_sha256="e" * 64,
        )

    payload, _ = _formal_feature_bundle_payload(tmp_path / "drift")
    cohort_record = payload["capability_observation_reference"][
        "legacy_receipt_correction"
    ]
    cohort = json.loads(Path(cohort_record["path"]).read_text(encoding="utf-8"))
    cohort["target_access"]["text_queries_opened"] = True
    Path(cohort_record["path"]).write_text(json.dumps(cohort), encoding="utf-8")
    cohort_record["sha256"] = hashlib.sha256(
        Path(cohort_record["path"]).read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="cohort authority differs"):
        _formal_capability_training_authority(
            payload,
            expected_radio_checkpoint_sha256="a" * 64,
            expected_factorized_radio_cache_sha256="e" * 64,
        )


def test_formal_capability_training_authority_rejects_unknown_mode(
    tmp_path: Path,
) -> None:
    payload, _ = _formal_feature_bundle_payload(tmp_path)
    payload["capability_observation_reference"][
        "capability_cohort_authority_mode"
    ] = "unknown"
    with pytest.raises(ValueError, match="unsupported"):
        _formal_capability_training_authority(
            payload,
            expected_radio_checkpoint_sha256="a" * 64,
            expected_factorized_radio_cache_sha256="e" * 64,
        )


def test_capability_builder_no_clobber_precedes_loading(tmp_path: Path) -> None:
    output = tmp_path / "capability.pt"
    output.write_bytes(b"occupied")
    with pytest.raises(FileExistsError, match="refuses to clobber"):
        capability_builder.build(Namespace(output=str(output)))
