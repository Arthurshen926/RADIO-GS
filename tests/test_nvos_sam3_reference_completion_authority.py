from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pytest

from radio_gs.querying.sam3_reference_completion import (
    aggregate_completed_positive as canonical_aggregate_completed_positive,
    deterministic_positive_points as canonical_deterministic_positive_points,
)
from radio_gs.scripts import build_nvos_sam3_reference_completion as builder


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(
    scene_ids=builder.FULL8_EXPANSION_ORDER,
    *,
    manifest_sha256=builder.FROZEN_MANIFEST_SHA256,
    registration_sha256=builder.FULL8_EXPANSION_REGISTRATION_SHA256,
    manifest_cohort=builder.FROZEN_FULL_COHORT,
):
    return builder.validate_registered_execution_authority(
        scene_ids=scene_ids,
        manifest_sha256=manifest_sha256,
        registration_sha256=registration_sha256,
        manifest_cohort=manifest_cohort,
    )


def test_full8_expansion_authority_accepts_only_fixed_remaining_order() -> None:
    authority = _validate()

    assert authority == {
        "name": "remaining_six_full8_expansion_v1",
        "registration_sha256": builder.FULL8_EXPANSION_REGISTRATION_SHA256,
    }


def test_legacy_sentinel_authority_remains_exactly_replayable() -> None:
    authority = _validate(
        builder.LEGACY_SENTINEL_ORDER,
        registration_sha256=builder.LEGACY_SENTINEL_REGISTRATION_SHA256,
    )

    assert authority["name"] == "legacy_two_task_sentinel_v1"


def test_isolated_runtime_loader_executes_canonical_helper_implementation() -> None:
    positive = np.zeros((8, 10), dtype=bool)
    positive.reshape(-1)[np.arange(32)] = True
    negative = np.zeros_like(positive)
    negative[7, 9] = True
    trials = np.repeat(positive[None], 10, axis=0)

    np.testing.assert_array_equal(
        builder.deterministic_positive_points(positive, count=30),
        canonical_deterministic_positive_points(positive, count=30),
    )
    actual = builder.aggregate_completed_positive(
        trials, positive, negative, threshold=0.5
    )
    expected = canonical_aggregate_completed_positive(
        trials, positive, negative, threshold=0.5
    )
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize(
    "scene_ids",
    [
        tuple(reversed(builder.FULL8_EXPANSION_ORDER)),
        builder.FULL8_EXPANSION_ORDER[:-1],
        builder.FULL8_EXPANSION_ORDER + ("fern",),
        ("flower", "flower", *builder.FULL8_EXPANSION_ORDER[1:]),
        ("unknown", *builder.FULL8_EXPANSION_ORDER[1:]),
    ],
)
def test_full8_expansion_authority_rejects_order_subset_duplicate_or_unknown(
    scene_ids: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="not a registered execution authority"):
        _validate(scene_ids)


def test_full8_expansion_authority_rejects_wrong_manifest_hash() -> None:
    with pytest.raises(ValueError, match="manifest SHA256 differs"):
        _validate(manifest_sha256="0" * 64)


def test_full8_expansion_authority_rejects_wrong_manifest_cohort() -> None:
    with pytest.raises(ValueError, match="manifest cohort differs"):
        _validate(manifest_cohort=builder.FROZEN_FULL_COHORT[:-1])


def test_full8_expansion_authority_rejects_wrong_registration_hash() -> None:
    with pytest.raises(ValueError, match="registration SHA256 differs"):
        _validate(registration_sha256="0" * 64)


def test_full8_registration_file_is_the_bound_immutable_authority() -> None:
    repository = Path(__file__).resolve().parents[1]
    registration = repository / (
        "paper/artifacts/"
        "nvos_source_completion_loo_abstention_full8_expansion_"
        "preregistration_20260805.json"
    )

    assert _sha256(registration) == builder.FULL8_EXPANSION_REGISTRATION_SHA256


def test_run_rejects_unregistered_order_before_cuda_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    manifest = Path(
        "/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/"
        "manifests/nvos_strict_unseen_v1.json"
    )
    if not manifest.is_file():
        pytest.skip("frozen NVOS manifest is not mounted")
    registration = repository / (
        "paper/artifacts/"
        "nvos_source_completion_loo_abstention_full8_expansion_"
        "preregistration_20260805.json"
    )
    cuda_initialized = False

    def _forbid_cuda(_device: str) -> None:
        nonlocal cuda_initialized
        cuda_initialized = True
        raise AssertionError("CUDA must not initialize for invalid authority")

    monkeypatch.setattr(builder, "set_requested_cuda_device", _forbid_cuda)
    args = argparse.Namespace(
        manifest=str(manifest),
        registration=str(registration),
        checkpoint=str(tmp_path / "unused.pt"),
        output_root=str(tmp_path / "output"),
        scene_ids="flower fortress horns_center leaves trex orchids",
        device="cuda:0",
    )

    with pytest.raises(ValueError, match="not a registered execution authority"):
        builder.run(args)
    assert cuda_initialized is False
