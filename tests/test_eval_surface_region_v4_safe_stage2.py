from argparse import Namespace
import copy

import pytest
import torch

from radio_gs.models.siglip_projection import OFFICIAL_C_RADIO_V4_H_HALF_SHA256
from radio_gs.scripts import eval_surface_region_v4_safe_stage2 as stage2


def _metadata() -> dict:
    return {
        "region_contract_version": "surface-region-contract-v4",
        "region_contract_sha256": stage2.V4_CONTRACT_SHA256,
        "split_hashes": [stage2.VALIDATION_SPLIT_SHA256],
        "scenes": list(stage2.VALIDATION_SCENES),
        "radio_checkpoint_sha256": OFFICIAL_C_RADIO_V4_H_HALF_SHA256,
        "cache_artifacts": [
            {"path": "/cache/validation_shard0.pt", "sha256": "a" * 64},
            {"path": "/cache/validation_shard1.pt", "sha256": "b" * 64},
        ],
        "eligibility_completion": {
            "enabled": True,
            "variants_per_teacher_region": 1,
            "validation_checkpoint_selection": "full_support_rows_only",
        },
    }


def test_frozen_stage2_registration_is_sha_bound_and_exact() -> None:
    registration = stage2._load_stage2_registration()
    assert registration["single_change"]["digest"] == stage2.V4_CONTRACT_SHA256
    assert (
        registration["staged_validation"]["stage_2"][
            "base_descriptor_cosine_floor"
        ]
        == stage2.BASE_DESCRIPTOR_COSINE_FLOOR
    )


def test_stage2_authority_and_gate_fail_closed() -> None:
    stage2._validate_stage2_validation_authority(_metadata())
    wrong = copy.deepcopy(_metadata())
    wrong["scenes"][-1] = "scene_benchmark"
    with pytest.raises(ValueError, match="exactly the eight frozen scenes"):
        stage2._validate_stage2_validation_authority(wrong)

    passing = stage2._stage2_gate(
        {
            "summary_token_cosine": 0.8,
            "mean_descriptor_cosine": stage2.BASE_DESCRIPTOR_COSINE_FLOOR,
            "all_view_descriptor_cosine": 0.9,
        }
    )
    assert passing["passed"] is True
    assert passing["stage3_residual_experiment_authorized"] is True
    assert passing["benchmark_opening_authorized"] is False
    failing = stage2._stage2_gate(
        {
            "summary_token_cosine": 0.8,
            "mean_descriptor_cosine": stage2.BASE_DESCRIPTOR_COSINE_FLOOR - 1e-8,
            "all_view_descriptor_cosine": 0.9,
        }
    )
    assert failing["passed"] is False


def test_evaluate_wires_zero_parameter_adapter_without_training(
    tmp_path, monkeypatch
) -> None:
    class FakeModel(torch.nn.Module):
        def architecture(self, contract_sha256: str) -> dict:
            return {
                "name": "fake-v4-safe-adapter",
                "contract_sha256": contract_sha256,
                "trainable_parameter_count": 0,
            }

    class FakeAdapterFactory:
        @staticmethod
        def from_accepted_v2_checkpoint(path, *, map_location):
            return FakeModel(), {"accepted": True}

    class FakeHead(torch.nn.Module):
        @classmethod
        def from_radio_checkpoint(cls, path):
            return cls()

    full = {
        "radio_features": torch.empty(8, 1, 1),
        "scene_ids": list(stage2.VALIDATION_SCENES),
    }
    completion = {
        "radio_features": torch.empty(8, 1, 1),
        "scene_ids": list(stage2.VALIDATION_SCENES),
    }
    monkeypatch.setattr(stage2, "_load_stage2_registration", lambda: {
        "staged_validation": {
            "stage_2": {
                "scope": "all eight held-out query-free ScanNet validation scenes"
            }
        }
    })
    monkeypatch.setattr(stage2, "_load", lambda *args, **kwargs: ({}, _metadata()))
    monkeypatch.setattr(
        stage2, "_completion_validation_views", lambda data: (full, completion)
    )
    monkeypatch.setattr(stage2, "SurfaceRegionSummaryReadoutV4", FakeAdapterFactory)
    monkeypatch.setattr(stage2, "SigLIP2SummaryHead", FakeHead)
    metrics = {
        "summary_token_cosine": 0.8,
        "mean_descriptor_cosine": 0.95,
        "all_view_descriptor_cosine": 0.9,
    }
    monkeypatch.setattr(stage2, "_evaluate", lambda *args, **kwargs: dict(metrics))
    output = tmp_path / "stage2.json"
    report = stage2.evaluate(
        Namespace(
            validation_cache=["shard0.pt", "shard1.pt"],
            validation_cache_sha256=["a" * 64, "b" * 64],
            accepted_v2_checkpoint="accepted.pt",
            radio_checkpoint="radio.pt",
            batch_size=4,
            device="cpu",
            output=str(output),
        )
    )

    assert output.is_file()
    assert report["protocol"]["training"] is False
    assert report["protocol"]["query_or_text_opened"] is False
    assert report["protocol"]["benchmark_opened"] is False
    assert report["stage2_gate"]["passed"] is True
    assert report["readout"]["trainable_parameter_count"] == 0
