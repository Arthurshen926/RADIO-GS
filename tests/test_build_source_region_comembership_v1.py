from __future__ import annotations

import pytest
import torch

from radio_gs.scripts import build_source_region_comembership_v1 as builder


def test_exact_hit_mass_uses_base_squared_over_pixel_mass() -> None:
    keys, mass = builder._exact_hit_instance_mass(
        gaussian_ids=torch.tensor([0, 1, 0, 2]),
        pixel_ids=torch.tensor([0, 0, 1, 1]),
        base_weights=torch.tensor([0.6, 0.4, 0.5, 0.5]),
        pixel_instance_ids=torch.tensor([3, 4]),
        num_gaussians=3,
        num_pixels=2,
    )
    observed = {int(key): float(value) for key, value in zip(keys, mass)}
    assert observed[0 * builder.INSTANCE_KEY_STRIDE + 3] == pytest.approx(0.36)
    assert observed[1 * builder.INSTANCE_KEY_STRIDE + 3] == pytest.approx(0.16)
    assert observed[0 * builder.INSTANCE_KEY_STRIDE + 4] == pytest.approx(0.25)
    assert observed[2 * builder.INSTANCE_KEY_STRIDE + 4] == pytest.approx(0.25)


def test_region_dominant_instance_purity_and_coverage_are_mass_based() -> None:
    dense = torch.tensor(
        [
            [0.1, 0.6, 0.3],
            [0.2, 0.1, 0.7],
            [1.0, 0.0, 0.0],
        ]
    )
    result = builder._region_instance_statistics(
        dense,
        torch.tensor([[0, 1], [2, -1]]),
        torch.tensor([[True, True], [True, False]]),
    )
    assert result["dominant_instance_ids"].tolist() == [2, -1]
    assert result["instance_purity"][0] == pytest.approx(1.0 / 1.7)
    assert result["instance_label_coverage"][0] == pytest.approx(1.7 / 2.0)
    assert result["instance_purity"][1] == 0
    assert result["instance_label_coverage"][1] == 0


def test_knn_pairs_are_undirected_sorted_and_deterministic() -> None:
    values = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    first = builder._knn_pairs(values, neighbors=1, cosine=True)
    second = builder._knn_pairs(values, neighbors=1, cosine=True)
    assert first == second
    assert first == {(0, 1), (1, 2)}
    assert all(left < right for left, right in first)


def test_builder_cli_has_no_target_or_benchmark_command() -> None:
    parser = builder.build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) == {"synthetic-dry-run", "build"}
    dry = builder.synthetic_dry_run()
    assert dry["benchmark_opened"] is False


def test_authority_identity_binds_builder_implementation() -> None:
    source = builder.Path(builder.__file__).resolve()
    record = builder.file_record(source)
    assert record["path"] == str(source)
    assert len(record["sha256"]) == 64
