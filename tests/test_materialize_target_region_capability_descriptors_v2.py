from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch

from radio_gs.scripts import materialize_region_capability_descriptors_v2 as source
from radio_gs.scripts import (
    materialize_target_region_capability_descriptors_v2 as target,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


class _Bank:
    def __init__(self, row_authority: dict) -> None:
        self.global_rows = torch.tensor([0, 1])
        self.num_gaussians = 2
        self.metadata = {
            "primitive_row_authority": row_authority,
            "mpr_geometry_fingerprint": {"xyz_sha256": "b" * 64},
        }

    def valid_feature_banks(self) -> dict[str, torch.Tensor]:
        return {
            "appearance": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "boundary": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        }


def test_target_wrapper_uses_target_validator_and_preserves_v2_schema(
    monkeypatch, tmp_path: Path
) -> None:
    row_authority = {"axis": "synthetic"}
    accepted = {
        "scene_id": "figurines",
        "physical_space_id": "lerf:figurines:geometry-checkpoint-sha256:" + "9" * 64,
        "region_fingerprints": ["1" * 64],
        "canonical_region_indices": torch.tensor([0]),
        "region_rows": torch.tensor([[0, 1]]),
        "token_mask": torch.tensor([[True, True]]),
        "input_authority": {
            "geometry_authority": {
                "factorized_field_checkpoint_file_sha256": "a" * 64,
                "primitive_row_authority_sha256": canonical_json_sha256(row_authority),
                "geometry_fingerprint": {
                    "num_gaussians": 2,
                    "xyz_sha256": "b" * 64,
                },
            }
        },
    }
    called = {"target": 0}
    captured = {}

    def validate_target(value):
        called["target"] += 1
        return accepted

    monkeypatch.setattr(
        target,
        "load_torch_mapping",
        lambda *args, **kwargs: (accepted, "c" * 64, tmp_path / "accepted.pt"),
    )
    monkeypatch.setattr(
        target, "validate_target_accepted_v2_authority", validate_target
    )
    monkeypatch.setattr(target, "sha256_file", lambda path: "d" * 64)
    monkeypatch.setattr(
        target,
        "load_canonical_capability_bank",
        lambda *args, **kwargs: _Bank(row_authority),
    )
    monkeypatch.setattr(
        target,
        "file_record",
        lambda path: {"path": str(Path(path).resolve()), "sha256": "e" * 64},
    )

    def write(output, payload):
        captured.update(payload)
        return Path(output)

    monkeypatch.setattr(target, "write_torch_noclobber", write)
    result = target.materialize(
        Namespace(
            target_accepted_v2=str(tmp_path / "accepted.pt"),
            expected_target_accepted_v2_sha256="c" * 64,
            capability_bank=str(tmp_path / "capability.pt"),
            expected_capability_bank_sha256="d" * 64,
            batch_size=1,
            output=str(tmp_path / "output.pt"),
        )
    )
    assert called["target"] == 1
    assert captured["schema"] == source.SCHEMA
    assert captured["schema_version"] == 2
    source.validate_region_capability_descriptor_authority(captured)
    assert result["target_metric_computed"] is False
