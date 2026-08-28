from pathlib import Path

import pytest
import torch

from radio_gs.v3.memory.structured_memory import (
    LowRankPrivateBranchMemory,
    SharedPrivateLayout,
)
from radio_gs.v3.training.instance_upper_bound import sha256_file
from radio_gs.v3.training.run_structured_source_mapping import (
    load_protected_initialization,
)


def _checkpoint(path: Path, membership: Path, **metadata_override) -> None:
    metadata = {
        "source_only": True,
        "historical_field_opened": False,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "layout": dict(SharedPrivateLayout().__dict__),
        "membership": {"sha256": sha256_file(membership)},
        "phase_order": "masked_gate_authorized_unknown_structural_initialization",
        **metadata_override,
    }
    torch.save(
        {
            "schema": "radio_gs.sugm_v3.structured_source_mapping.v1",
            "metadata": metadata,
            "state_dict": {
                "memory": torch.randn(7, 512),
                "scale_adapter.weight": torch.randn(96, 2),
                "visual_codec.gate.weight": torch.randn(1, 640),
                "codec.siglip_basis": torch.randn(5, 128),
            },
        },
        path,
    )


def test_protected_continuation_loads_memory_and_carries_only_codecs(tmp_path):
    membership = tmp_path / "membership.pt"
    membership.write_bytes(b"source authority")
    checkpoint = tmp_path / "candidate.pt"
    _checkpoint(checkpoint, membership)

    memory, initialization, parent, carried = load_protected_initialization(
        checkpoint,
        membership_path=membership,
        layout=SharedPrivateLayout(),
    )

    assert memory.shape == (7, 512)
    assert initialization["shared_and_semantic_frozen"] is True
    assert "scale_adapter.weight" in parent
    assert set(carried) == {"visual_codec.gate.weight", "codec.siglip_basis"}


@pytest.mark.parametrize(
    "override",
    (
        {"source_only": False},
        {"historical_field_opened": True},
        {"target_rgb_opened": True},
        {"benchmark_metrics_opened": True},
    ),
)
def test_protected_continuation_rejects_forbidden_authority(tmp_path, override):
    membership = tmp_path / "membership.pt"
    membership.write_bytes(b"source authority")
    checkpoint = tmp_path / "candidate.pt"
    _checkpoint(checkpoint, membership, **override)

    with pytest.raises(ValueError):
        load_protected_initialization(
            checkpoint,
            membership_path=membership,
            layout=SharedPrivateLayout(),
        )


def test_owned_training_blocks_merge_into_one_deployment_memory():
    initial = torch.randn(9, 512)
    model = LowRankPrivateBranchMemory(initial)
    model.enable_owned_training_blocks("instance", "boundary")

    assert not model.memory.requires_grad
    with torch.no_grad():
        model.owned_training_parameter("instance").add_(2)
        model.owned_training_parameter("boundary").sub_(3)
    deployed = model.deployment_memory()
    slices = SharedPrivateLayout().slices

    torch.testing.assert_close(deployed[:, slices["shared"]], initial[:, slices["shared"]])
    torch.testing.assert_close(deployed[:, slices["semantic"]], initial[:, slices["semantic"]])
    torch.testing.assert_close(
        deployed[:, slices["instance"]], initial[:, slices["instance"]] + 2
    )
    torch.testing.assert_close(
        deployed[:, slices["boundary"]], initial[:, slices["boundary"]] - 3
    )


def test_owned_training_blocks_avoid_dense_d512_gradient():
    model = LowRankPrivateBranchMemory(torch.randn(11, 512))
    model.enable_owned_training_blocks("instance", "boundary")

    (model(0.5).square().sum() + model.boundary_view().square().sum()).backward()

    assert model.memory.grad is None
    assert model.owned_training_parameter("instance").grad.shape == (11, 48)
    assert model.owned_training_parameter("boundary").grad.shape == (11, 16)
