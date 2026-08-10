from __future__ import annotations

from pathlib import Path

import torch

from radio_gs.scripts import (
    select_lerf_source_only_reliability_weighted_knn as selector,
)
from radio_gs.utils.immutable_artifacts import file_record


def _base_payload(path: Path) -> dict[str, torch.Tensor]:
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    valid = torch.tensor([True, True, False, True, True, True])
    rows = torch.where(valid)[0]
    features = torch.zeros(rows.numel(), selector.DESCRIPTOR_DIMENSION).half()
    payload = {
        "xyz": xyz,
        "features": features,
        "summary_features": features,
        "global_rows": rows,
        "features_by_scale": torch.zeros(
            rows.numel(), 3, selector.DESCRIPTOR_DIMENSION
        ).half(),
        "valid": valid,
        "metadata": {
            "feature_space": "official_siglip2_summary_descriptor_multiscale"
        },
    }
    torch.save(payload, path)
    return payload


def _teacher_payload(path: Path, rows: torch.Tensor) -> dict[str, torch.Tensor]:
    descriptor = torch.zeros(rows.numel(), selector.DESCRIPTOR_DIMENSION).half()
    descriptor[:, 0] = 1
    descriptor[-1, 0] = 0
    descriptor[-1, 1] = 1
    valid = torch.ones(rows.numel(), dtype=torch.bool)
    payload = {
        "schema": "radio_gs.lerf_source_teacher_mean_siglip.v2",
        "global_rows": rows,
        "teacher_mean": descriptor,
        "teacher_valid": valid,
        "retained_view_count": torch.tensor([4, 3, 2, 4, 1], dtype=torch.uint8),
        "teacher_view_directional_resultant": torch.tensor(
            [1.0, 0.9, 0.8, 1.0, 1.0], dtype=torch.float32
        ),
    }
    torch.save(payload, path)
    return payload


def test_selective_zip_loaders_extract_only_declared_axes(tmp_path: Path) -> None:
    base_path = tmp_path / "base.pt"
    teacher_path = tmp_path / "teacher.pt"
    base_raw = _base_payload(base_path)
    _teacher_payload(teacher_path, base_raw["global_rows"])
    geometry = selector.load_base_geometry_selective(file_record(base_path))
    teacher = selector.load_teacher_selective(file_record(teacher_path))
    assert torch.equal(geometry["xyz"], base_raw["xyz"])
    assert torch.equal(geometry["global_rows"], base_raw["global_rows"])
    assert torch.equal(teacher["global_rows"], base_raw["global_rows"])
    assert teacher["teacher_mean_memmap"].shape == (
        base_raw["global_rows"].numel(),
        selector.DESCRIPTOR_DIMENSION,
    )
    assert teacher["reliability"][-1] == 0


def _scene_rows(
    *,
    uniform_mean: float,
    uniform_p05: float,
    winner_mean: float,
    winner_p05: float,
) -> dict[str, object]:
    rows = []
    for policy in selector.weighted.POLICIES:
        mean = uniform_mean
        p05 = uniform_p05
        if policy.policy_id == "gaussian_reliability_precision":
            mean = winner_mean
            p05 = winner_p05
        rows.append(
            {
                "policy_id": policy.policy_id,
                "observations": 10,
                "cosine_sum": 10 * mean,
                "mean_cosine": mean,
                "p05_cosine": p05,
            }
        )
    return {"candidate_statistics": rows}


def test_selector_requires_every_scene_mean_and_tail_nonregression() -> None:
    first = _scene_rows(
        uniform_mean=0.8,
        uniform_p05=0.5,
        winner_mean=0.82,
        winner_p05=0.51,
    )
    second = _scene_rows(
        uniform_mean=0.7,
        uniform_p05=0.4,
        winner_mean=0.72,
        winner_p05=0.39,
    )
    rejected = selector.select_policy([first, second])
    assert rejected["selected_policy_id"] == "uniform"
    assert rejected["target_metric_execution_authorized"] is False
    precision_row = next(
        row
        for row in second["candidate_statistics"]
        if row["policy_id"] == "gaussian_reliability_precision"
    )
    precision_row["p05_cosine"] = 0.41
    accepted = selector.select_policy([first, second])
    assert accepted["selected_policy_id"] == "gaussian_reliability_precision"
    assert accepted["target_metric_execution_authorized"] is True


def test_deterministic_audit_rows_are_repeatable_and_bounded() -> None:
    rows = torch.arange(100)
    first = selector.deterministic_audit_rows(rows, maximum=7)
    second = selector.deterministic_audit_rows(rows, maximum=7)
    assert torch.equal(first, second)
    assert first.tolist() == [0, 16, 33, 50, 66, 82, 99]
