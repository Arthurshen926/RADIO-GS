from pathlib import Path
import hashlib

import pytest
import torch

from radio_gs.field import (
    AffineBasisDecoder,
    CanonicalGaussianField,
    FeatureSpaceSignature,
)
from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
    project_official_capability_maps,
)
from radio_gs.scripts.train_canonical_radio_field import (
    _load_capability_mpr_target,
)
from radio_gs.training.canonical_field_losses import (
    CanonicalFieldLossConfig,
    canonical_primitive_loss,
)
from radio_gs.training.primitive_consensus import PrimitiveConsensus


def _consensus(features: torch.Tensor) -> PrimitiveConsensus:
    count = features.shape[0]
    return PrimitiveConsensus(
        targets=features,
        valid=torch.ones(count, dtype=torch.bool),
        observation_count=torch.ones(count, dtype=torch.long),
        reliability=torch.ones(count, 3),
        per_view_agreement=torch.empty(0, count),
    )


class _IdentityOfficialViews:
    @staticmethod
    def project_dino_primitives(features: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(features, dim=-1)

    @staticmethod
    def project_sam3_primitives(features: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(features, dim=-1)


def _identity_field(rows: torch.Tensor) -> CanonicalGaussianField:
    decoder = AffineBasisDecoder(
        feature_dim=2,
        coefficient_dim=2,
        basis=torch.eye(2),
        trainable_basis=False,
    )
    field = CanonicalGaussianField(
        rows.shape[0],
        decoder,
        FeatureSpaceSignature(
            radio_version="test",
            radio_checkpoint_sha256="radio",
            raw_feature_dim=2,
            token_type="primitive",
        ),
        use_fusion=False,
    )
    with torch.no_grad():
        field.local_codes.copy_(rows)
    return field


def test_auxiliary_capability_target_is_not_adaptor_of_raw_mpr() -> None:
    raw = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    field = _identity_field(raw)
    config = CanonicalFieldLossConfig(
        mpr_weight=0.0,
        dino_weight=1.0,
        sam3_weight=0.0,
        relation_weight=0.0,
        coefficient_weight=0.0,
        basis_orthogonality_weight=0.0,
    )
    legacy_loss, legacy = canonical_primitive_loss(
        field,
        _consensus(raw),
        torch.arange(2),
        official_views=_IdentityOfficialViews(),
        config=config,
    )
    auxiliary_loss, auxiliary = canonical_primitive_loss(
        field,
        _consensus(raw),
        torch.arange(2),
        official_views=_IdentityOfficialViews(),
        capability_targets={"dino_v3": _consensus(-raw)},
        config=config,
    )

    torch.testing.assert_close(legacy_loss, torch.tensor(0.0))
    torch.testing.assert_close(legacy["dino"], torch.tensor(0.0))
    torch.testing.assert_close(auxiliary_loss, torch.tensor(2.0))
    torch.testing.assert_close(auxiliary["dino"], torch.tensor(2.0))


def test_official_capability_projection_happens_on_complete_2d_maps() -> None:
    maps = torch.zeros(1, 1280, 1, 2)
    maps[0, 0, 0, 0] = 3.0
    maps[0, 1, 0, 1] = 4.0

    projected = project_official_capability_maps(
        maps,
        torch.nn.Identity(),
        device=torch.device("cpu"),
        batch_size=1,
    )

    assert projected.shape == maps.shape
    torch.testing.assert_close(projected[0, 0, 0, 0].float(), torch.tensor(1.0))
    torch.testing.assert_close(projected[0, 1, 0, 1].float(), torch.tensor(1.0))


_FEATURE_BUNDLE_SHA256 = "b" * 64
_RESPONSIBILITY_SHA256 = "c" * 64


def _xyz_sha256(xyz: torch.Tensor) -> str:
    array = xyz.float().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _cache_metadata(space: str, xyz: torch.Tensor) -> dict:
    return {
        "schema_version": 1,
        "feature_space": space,
        "config": "scene.yaml",
        "checkpoint": "geometry.pth",
        "selected_frame_indices": [1, 2],
        "num_declared_views": 2,
        "xyz_sha256": _xyz_sha256(xyz),
        "excluded_frame_ids": [3],
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
        "shared_registration_responsibility": True,
        "registration_responsibility_cache_sha256": _RESPONSIBILITY_SHA256,
        "feature_output_bundle_sha256": _FEATURE_BUNDLE_SHA256,
        "benchmark_masks_opened": False,
        "benchmark_images_opened": False,
        "text_queries_opened": False,
        "capability_projection_before_mpr": space != "radio",
        "custom_adaptor_head": False,
        "official_adaptor_name": "sam3",
        "official_adaptor_checkpoint_sha256": "radio",
    }


def _mpr_payload(
    space: str,
    xyz: torch.Tensor,
    features: torch.Tensor,
    counts: torch.Tensor,
    **metadata_updates,
) -> dict:
    valid = counts > 0
    metadata = {**_cache_metadata(space, xyz), **metadata_updates}
    reliability = torch.stack(
        [counts.float() / 2.0, valid.float(), valid.float()], dim=-1
    )
    return {
        "xyz": xyz.float(),
        "features": features,
        "valid": valid,
        "view_counts": counts,
        "reliability": reliability,
        "geometry_fingerprint": {
            "num_gaussians": int(xyz.shape[0]),
            "xyz_sha256": _xyz_sha256(xyz),
        },
        "metadata": metadata,
    }


def test_capability_mpr_loader_requires_exact_observation_contract(
    tmp_path: Path,
) -> None:
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    counts = torch.tensor([2, 1])
    raw = _mpr_payload("radio", xyz, torch.randn(2, 3), counts)
    target = _mpr_payload("sam3", xyz, torch.randn(2, 4).half(), counts)
    path = tmp_path / "sam3.pt"
    torch.save(target, path)

    consensus, provenance = _load_capability_mpr_target(
        path,
        expected_space="sam3",
        raw_cache=raw,
        raw_metadata=raw["metadata"],
        radio_checkpoint_sha256="radio",
        expected_feature_output_bundle_sha256=_FEATURE_BUNDLE_SHA256,
    )
    assert consensus.targets.dtype == torch.float16
    assert provenance["projection_order"] == (
        "official_adaptor_then_geometry_matched_mpr"
    )
    assert provenance["capability_map_source"] == "project_raw"

    contaminated = dict(target)
    contaminated["metadata"] = {
        **target["metadata"],
        "benchmark_images_opened": True,
    }
    torch.save(contaminated, path)
    with pytest.raises(
        ValueError, match="contaminated|safety contract|safety declaration"
    ):
        _load_capability_mpr_target(
            path,
            expected_space="sam3",
            raw_cache=raw,
            raw_metadata=raw["metadata"],
            radio_checkpoint_sha256="radio",
            expected_feature_output_bundle_sha256=_FEATURE_BUNDLE_SHA256,
        )


def test_direct_capability_mpr_requires_native_runtime_provenance(
    tmp_path: Path,
) -> None:
    xyz = torch.tensor([[0.0, 0.0, 0.0]])
    counts = torch.tensor([1])
    raw = _mpr_payload("radio", xyz, torch.randn(1, 3), counts)
    metadata = {
        **_cache_metadata("sam3", xyz),
        "capability_map_source": "official_extracted",
        "official_adaptor_checkpoint_provenance": "explicit_file_sha256",
        "capability_native_map_manifest": "/tmp/frame_manifest.json",
        "capability_native_map_manifest_sha256": "manifest",
        "capability_native_map_output_bundle_sha256": _FEATURE_BUNDLE_SHA256,
        "capability_native_map_radio_checkpoint_load_contract": (
            "external_sha256_same_fd_restricted_pickle_hub_injection_v1"
        ),
        "capability_adaptor_execution": "official_c_radio_runtime_adaptor_output",
    }
    path = tmp_path / "sam3_direct.pt"
    torch.save(
        _mpr_payload(
            "sam3",
            xyz,
            torch.randn(1, 4).half(),
            counts,
            **metadata,
        ),
        path,
    )

    _consensus_value, provenance = _load_capability_mpr_target(
        path,
        expected_space="sam3",
        raw_cache=raw,
        raw_metadata=raw["metadata"],
        radio_checkpoint_sha256="radio",
        expected_feature_output_bundle_sha256=_FEATURE_BUNDLE_SHA256,
    )
    assert provenance["capability_map_source"] == "official_extracted"

    unbound_metadata = dict(metadata)
    unbound_metadata.pop("official_adaptor_checkpoint_provenance")
    torch.save(
        _mpr_payload(
            "sam3",
            xyz,
            torch.randn(1, 4).half(),
            counts,
            **unbound_metadata,
        ),
        path,
    )
    with pytest.raises(ValueError, match="extraction checkpoint SHA256"):
        _load_capability_mpr_target(
            path,
            expected_space="sam3",
            raw_cache=raw,
            raw_metadata=raw["metadata"],
            radio_checkpoint_sha256="radio",
            expected_feature_output_bundle_sha256=_FEATURE_BUNDLE_SHA256,
        )

    metadata.pop("capability_native_map_manifest_sha256")
    torch.save(
        _mpr_payload(
            "sam3",
            xyz,
            torch.randn(1, 4).half(),
            counts,
            **metadata,
        ),
        path,
    )
    with pytest.raises(ValueError, match="native-map provenance"):
        _load_capability_mpr_target(
            path,
            expected_space="sam3",
            raw_cache=raw,
            raw_metadata=raw["metadata"],
            radio_checkpoint_sha256="radio",
            expected_feature_output_bundle_sha256=_FEATURE_BUNDLE_SHA256,
        )
