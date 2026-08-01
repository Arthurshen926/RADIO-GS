import pytest
import torch

from radio_gs.config import RadioGSConfig
from radio_gs.scripts.audit_canonical_capability_fidelity import (
    _capability_fidelity_maps,
    _resolved_config_sha256,
)


def test_capability_audit_can_use_native_official_targets() -> None:
    predicted = torch.tensor([[[1.0]], [[0.0]]])
    raw_teacher = torch.tensor([[[0.0]], [[1.0]]])
    official_dino = torch.tensor([[[2.0]], [[0.0]]])
    official_sam = torch.tensor([[[0.0]], [[3.0]]])

    maps = _capability_fidelity_maps(
        predicted,
        raw_teacher,
        torch.nn.Identity(),
        torch.nn.Identity(),
        official_targets={
            "dino_v3": official_dino,
            "sam3": official_sam,
        },
    )

    torch.testing.assert_close(
        maps["official_dino_v3"][1],
        torch.tensor([[[1.0]], [[0.0]]]),
    )
    torch.testing.assert_close(
        maps["official_sam3"][1],
        torch.tensor([[[0.0]], [[1.0]]]),
    )
    assert maps["official_dino_v3"][0] is not maps["official_dino_v3"][1]


def test_capability_audit_project_raw_proxy_remains_available() -> None:
    predicted = torch.tensor([[[1.0]], [[0.0]]])
    raw_teacher = torch.tensor([[[0.0]], [[2.0]]])

    maps = _capability_fidelity_maps(
        predicted,
        raw_teacher,
        torch.nn.Identity(),
        torch.nn.Identity(),
    )

    torch.testing.assert_close(
        maps["official_dino_v3"][1],
        torch.tensor([[[0.0]], [[1.0]]]),
    )


def test_capability_audit_rejects_incomplete_or_misaligned_official_targets() -> None:
    raw = torch.ones(2, 2, 2)
    with pytest.raises(ValueError, match="must contain"):
        _capability_fidelity_maps(
            raw,
            raw,
            torch.nn.Identity(),
            torch.nn.Identity(),
            official_targets={"dino_v3": raw},
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        _capability_fidelity_maps(
            raw,
            raw,
            torch.nn.Identity(),
            torch.nn.Identity(),
            official_targets={
                "dino_v3": torch.ones(3, 2, 2),
                "sam3": raw,
            },
        )


def test_capability_audit_hashes_the_resolved_config() -> None:
    first = RadioGSConfig(scene="scene_a", feature_dir="/features")
    same = RadioGSConfig(scene="scene_a", feature_dir="/features")
    changed = RadioGSConfig(scene="scene_a", feature_dir="/other")

    assert _resolved_config_sha256(first) == _resolved_config_sha256(same)
    assert _resolved_config_sha256(first) != _resolved_config_sha256(changed)
