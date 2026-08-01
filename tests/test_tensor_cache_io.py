from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch

from radio_gs.training.tensor_cache_io import (
    load_mpr_cache,
    load_training_tensor_cache,
    validate_mpr_cache_payload,
)


def _write_sentinel(path: str) -> None:
    Path(path).write_text("pickle executed", encoding="utf-8")


class _MaliciousPayload:
    def __init__(self, sentinel: Path) -> None:
        self.sentinel = sentinel

    def __reduce__(self):
        return _write_sentinel, (str(self.sentinel),)


def _xyz_sha256(xyz: torch.Tensor) -> str:
    array = xyz.float().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _mpr_payload() -> dict:
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    counts = torch.tensor([2, 1, 0], dtype=torch.int64)
    valid = counts > 0
    geometry_sha256 = _xyz_sha256(xyz)
    return {
        "xyz": xyz,
        "features": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float16
        ),
        "valid": valid,
        "view_counts": counts,
        "reliability": torch.tensor(
            [[1.0, 1.0, 1.0], [0.5, 1.0, 1.0], [0.0, 0.0, 0.0]],
            dtype=torch.float16,
        ),
        "geometry_fingerprint": {
            "num_gaussians": 3,
            "xyz_sha256": geometry_sha256,
        },
        "metadata": {
            "schema_version": 1,
            "feature_space": "radio",
            "num_declared_views": 2,
            "selected_frame_indices": [4, 9],
            "xyz_sha256": geometry_sha256,
            "raster_reliability_mode": "legacy_valid",
            "aggregation_mode": "raster_gaussian_top1",
            "shared_registration_responsibility": True,
            "registration_responsibility_cache_sha256": "c" * 64,
            "feature_output_bundle_sha256": "b" * 64,
            "benchmark_masks_opened": False,
            "benchmark_images_opened": False,
            "text_queries_opened": False,
        },
    }


def test_mpr_validator_accepts_one_fully_bound_cache(tmp_path: Path) -> None:
    path = tmp_path / "mpr.pt"
    torch.save(_mpr_payload(), path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    payload, digest, source = load_mpr_cache(
        path,
        expected_sha256=expected,
        expected_feature_space="radio",
        require_formal_safety=True,
    )

    assert payload["features"].dtype == torch.float16
    assert digest == expected
    assert source == path.resolve()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["features"].__setitem__((0, 0), float("nan")),
            "non-finite",
        ),
        (
            lambda value: value["valid"].__setitem__(2, True),
            "valid mask",
        ),
        (
            lambda value: value["features"].__setitem__((2, 0), 1.0),
            "unsupported feature",
        ),
        (
            lambda value: value["reliability"].__setitem__((1, 0), 0.75),
            "coverage",
        ),
        (
            lambda value: value["metadata"].__setitem__(
                "registration_responsibility_cache_sha256", "not-a-digest"
            ),
            "responsibility provenance",
        ),
        (
            lambda value: value["metadata"].__setitem__(
                "feature_output_bundle_sha256", "self-asserted"
            ),
            "feature output bundle",
        ),
    ],
)
def test_mpr_validator_rejects_deep_contract_corruption(mutate, message) -> None:
    payload = copy.deepcopy(_mpr_payload())
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        validate_mpr_cache_payload(payload, require_formal_safety=True)


def test_tensor_cache_loader_rejects_pickle_code_execution(tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel"
    path = tmp_path / "malicious.pt"
    torch.save(_MaliciousPayload(sentinel), path)

    with pytest.raises(Exception):
        load_training_tensor_cache(path)

    assert not sentinel.exists()


def test_tensor_cache_loader_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.pt"
    alias = tmp_path / "alias.pt"
    torch.save(torch.ones(1), target)
    alias.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        load_training_tensor_cache(alias)


def test_tensor_cache_loader_rejects_wrong_external_digest(tmp_path: Path) -> None:
    path = tmp_path / "mpr.pt"
    torch.save(_mpr_payload(), path)

    with pytest.raises(ValueError, match="SHA-256"):
        load_mpr_cache(path, expected_sha256="0" * 64)
