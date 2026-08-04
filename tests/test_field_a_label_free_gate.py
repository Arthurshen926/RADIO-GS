import csv

import pytest
import torch

from radio_gs.scripts.eval_field_a_label_free_gate import (
    _canonical_architecture,
    _same_state_structure,
    _training_runtime_receipt,
    evaluate_gate,
)


def test_legacy_missing_zero_residual_blocks_is_same_architecture() -> None:
    initial = {"coefficient_dim": 256, "use_fusion": True}
    final = {
        "coefficient_dim": 256,
        "use_fusion": True,
        "fusion_residual_blocks": 0,
    }
    assert _canonical_architecture(initial) == _canonical_architecture(final)
    assert _canonical_architecture(initial) != _canonical_architecture(
        {**final, "fusion_residual_blocks": 1}
    )
    assert _same_state_structure(torch.nn.Linear(2, 3), torch.nn.Linear(2, 3))
    assert not _same_state_structure(torch.nn.Linear(2, 3), torch.nn.Linear(2, 4))


def test_field_a_gate_requires_both_capabilities_and_raw_non_regression() -> None:
    initial = {
        "mean_cosine": 0.90,
        "p05_cosine": 0.80,
        "dino_v3_target_mean_cosine": 0.70,
        "sam3_target_mean_cosine": 0.60,
    }
    passed = evaluate_gate(
        initial,
        {
            "mean_cosine": 0.899,
            "p05_cosine": 0.795,
            "dino_v3_target_mean_cosine": 0.71,
            "sam3_target_mean_cosine": 0.62,
        },
    )
    assert passed["passed"] is True

    failed = evaluate_gate(
        initial,
        {
            "mean_cosine": 0.90,
            "p05_cosine": 0.80,
            "dino_v3_target_mean_cosine": 0.71,
            "sam3_target_mean_cosine": 0.59,
        },
    )
    assert failed["passed"] is False
    assert failed["checks"]["sam3_exact_mean_cosine_improves"] is False


def test_training_runtime_receipt_requires_gpu1_and_clean_release(tmp_path) -> None:
    telemetry = tmp_path / "telemetry.csv"
    with telemetry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "gpu",
                "bus_id",
                "temp_c",
                "power_w",
                "power_limit_w",
                "util_pct",
                "memory_mib",
                "pstate",
                "event",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "now",
                "gpu": "1",
                "bus_id": "0000:82:00.0",
                "temp_c": "55",
                "power_w": "254.48",
                "power_limit_w": "300.00",
                "util_pct": "71",
                "memory_mib": "2512",
                "pstate": "P2",
                "event": "sample",
            }
        )
        writer.writerow(
            {
                "timestamp": "later",
                "gpu": "1",
                "bus_id": "0000:82:00.0",
                "temp_c": "47",
                "power_w": "48.02",
                "power_limit_w": "300.00",
                "util_pct": "0",
                "memory_mib": "1",
                "pstate": "P8",
                "event": "cuda_release_verified_no_compute_owner",
            }
        )
    owner = tmp_path / "owner.csv"
    with owner.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "gpu_uuid",
                "child_pgid",
                "owner_pids",
                "child_owner_pids",
                "foreign_owner_pids",
                "event",
            ],
        )
        writer.writeheader()
        for event in ("prelaunch_owner_clear", "postexit_owner_clear"):
            writer.writerow(
                {
                    "timestamp": "now",
                    "gpu_uuid": "GPU-test",
                    "child_pgid": "1",
                    "owner_pids": "",
                    "child_owner_pids": "",
                    "foreign_owner_pids": "",
                    "event": event,
                }
            )
    receipt = _training_runtime_receipt(telemetry, owner)
    assert receipt["physical_gpu"] == 1
    assert receipt["max_temperature_c"] == 55
    assert receipt["max_power_w"] == pytest.approx(254.48)
    assert receipt["cuda_release_verified"] is True
