import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256
from radio_gs.interfaces.query_diffusion_relation_cache import (
    ARTIFACT_TYPE,
    PAYLOAD_KEYS,
    canonical_json_sha256,
    validate_query_diffusion_relation_payload,
)
from radio_gs.scripts.build_query_diffusion_pca40_relation_cache import (
    REGISTRATION_SHA256,
    fit_ludvig_inspired_pca_relation,
    float_rows_sha256,
)


SHA = "a" * 64


def _payload() -> dict:
    generator = torch.Generator().manual_seed(7)
    tensors = {
        "global_rows": torch.tensor([0, 2, 4, 5], dtype=torch.int64),
        "relation_features": torch.randn(4, 40, generator=generator),
        "feature_mean": torch.randn(48, generator=generator),
        "feature_std": torch.rand(48, generator=generator) + 0.2,
        "pca_mean": torch.randn(48, generator=generator),
        "pca_components": torch.randn(40, 48, generator=generator),
        "pca_singular_values": torch.rand(40, generator=generator) + 0.2,
    }
    tensors = {name: value.contiguous() for name, value in tensors.items()}
    digests = {
        name: tensor_sha256(value) for name, value in sorted(tensors.items())
    }
    payload = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "scene_id": "fern",
        "num_global_rows": 6,
        "source_feature_sha256": "b" * 64,
        "source_xyz_sha256": "c" * 64,
        "tensors": tensors,
        "tensor_sha256": digests,
        "tensor_bundle_sha256": canonical_json_sha256(digests),
        "metadata": {
            "relation_source": "official_C_RADIOv4_dino_v3_7b_primitive_rows",
            "source_dimension": 48,
            "transform": "standardize_PCA40_singular_value_weighted",
            "pca_components": 40,
            "experiment_registration_sha256": REGISTRATION_SHA256,
            "capability_sidecar_sha256": "d" * 64,
            "field_checkpoint_sha256": "e" * 64,
            "query_independent": True,
            "labels_opened": False,
            "target_rgb_opened": False,
            "target_masks_opened": False,
            "target_metrics_opened": False,
            "native_ludvig_dinov2_pca40_exact": False,
        },
    }
    assert set(payload) == PAYLOAD_KEYS
    return payload


def test_registered_pca_sequence_matches_release_operations_on_small_matrix():
    from sklearn.decomposition import PCA

    source = torch.randn(60, 8, generator=torch.Generator().manual_seed(11))
    actual, diagnostics = fit_ludvig_inspired_pca_relation(
        source,
        n_components=4,
        pca_subsample=500_000,
        seed=0,
        projection_chunk_size=13,
    )
    expected_mean = source.mean(0)
    expected_std = source.std(0, correction=1)
    standardized = (source - expected_mean) / expected_std
    np.random.seed(0)
    pca = PCA(n_components=4)
    pca.fit(standardized.numpy())
    expected = torch.from_numpy(
        ((standardized.numpy() - pca.mean_) @ pca.components_.T)
        * pca.singular_values_
    ).float()
    torch.testing.assert_close(actual["feature_mean"], expected_mean)
    torch.testing.assert_close(actual["feature_std"], expected_std)
    torch.testing.assert_close(actual["relation_features"], expected, rtol=2e-5, atol=2e-5)
    assert diagnostics["standardization_std_correction"] == 1
    assert diagnostics["singular_value_weighting"] is True


def test_pca_fit_is_query_independent_and_fails_on_zero_variance():
    source = torch.randn(50, 6, generator=torch.Generator().manual_seed(3))
    source[:, 2] = 1
    with pytest.raises(ValueError, match="zero-variance"):
        fit_ludvig_inspired_pca_relation(source, n_components=3)


def test_geometry_digest_matches_raw_little_endian_float32_authority():
    rows = torch.tensor([[1.0, 2.5, -3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
    expected = hashlib.sha256(
        rows.float().numpy().astype("<f4", copy=False).tobytes(order="C")
    ).hexdigest()
    assert float_rows_sha256(rows, row_chunk_size=1) == expected


def test_relation_validator_accepts_only_frozen_non_native_cache():
    payload = _payload()
    cache = validate_query_diffusion_relation_payload(
        payload,
        expected_scene_id="fern",
        expected_global_rows=torch.tensor([0, 2, 4, 5]),
        expected_num_global_rows=6,
        expected_source_feature_sha256="b" * 64,
        expected_source_xyz_sha256="c" * 64,
        expected_registration_sha256=REGISTRATION_SHA256,
        expected_capability_sidecar_sha256="d" * 64,
        expected_field_checkpoint_sha256="e" * 64,
    )
    assert cache.relation_dimension == 40
    assert cache.num_nodes == 4
    assert cache.metadata["native_ludvig_dinov2_pca40_exact"] is False


@pytest.mark.parametrize(
    "tensor_name",
    [
        "global_rows",
        "relation_features",
        "feature_mean",
        "feature_std",
        "pca_mean",
        "pca_components",
        "pca_singular_values",
    ],
)
def test_relation_validator_rejects_every_tensor_tamper(tensor_name):
    payload = _payload()
    value = payload["tensors"][tensor_name]
    value.reshape(-1)[0] += 1
    with pytest.raises(ValueError):
        validate_query_diffusion_relation_payload(payload)


@pytest.mark.parametrize(
    "tamper",
    ["schema", "registration", "target", "native_claim", "source", "rows", "bundle"],
)
def test_relation_validator_fails_closed(tamper):
    payload = _payload()
    if tamper == "schema":
        payload["extra"] = False
    elif tamper == "registration":
        payload["metadata"]["experiment_registration_sha256"] = SHA
    elif tamper == "target":
        payload["metadata"]["target_masks_opened"] = True
    elif tamper == "native_claim":
        payload["metadata"]["native_ludvig_dinov2_pca40_exact"] = True
    elif tamper == "source":
        payload["metadata"]["relation_source"] = "native_DINOv2"
    elif tamper == "rows":
        payload["tensors"]["global_rows"][1] = 0
    elif tamper == "bundle":
        payload["tensor_bundle_sha256"] = SHA
    with pytest.raises(ValueError):
        validate_query_diffusion_relation_payload(
            payload,
            expected_registration_sha256=REGISTRATION_SHA256,
        )


def test_repository_registration_hash_matches_pca40_binding():
    root = Path(__file__).resolve().parents[1]
    registration = (
        root
        / "paper"
        / "artifacts"
        / "cradio_dino_pca40_relation_cache_registration_20260803.json"
    )
    assert hashlib.sha256(registration.read_bytes()).hexdigest() == REGISTRATION_SHA256
