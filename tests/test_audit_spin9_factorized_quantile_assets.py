from __future__ import annotations

import json
from pathlib import Path

import pytest

from radio_gs.scripts.audit_spin9_factorized_quantile_assets import (
    estimate_resources,
    ply_vertex_count,
    sidecar_rows,
)


def test_ply_vertex_count_reads_header_only(tmp_path: Path) -> None:
    path = tmp_path / "carrier.ply"
    path.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 123\n"
        b"property float x\nend_header\n" + bytes(range(32))
    )
    assert ply_vertex_count(path) == 123


def test_sidecar_rows_accepts_all_current_authority_names(tmp_path: Path) -> None:
    path = tmp_path / "authority.json"
    path.write_text(
        '{"num_global_rows": 20, "num_nodes": 7}\n', encoding="utf-8"
    )
    assert sidecar_rows(path) == (20, 7)
    path.write_text(
        '{"num_gaussians": 21, "valid_gaussians": 8}\n', encoding="utf-8"
    )
    assert sidecar_rows(path) == (21, 8)


def test_resource_estimate_is_monotone_and_bounded() -> None:
    small = estimate_resources(100_000, 20_000, 10)
    large = estimate_resources(1_000_000, 500_000, 100)
    assert large["estimated_cpu_peak_gib_factorized_builder"] > small[
        "estimated_cpu_peak_gib_factorized_builder"
    ]
    assert large["estimated_new_disk_gib_before_exact_w_and_predictions"] > small[
        "estimated_new_disk_gib_before_exact_w_and_predictions"
    ]
    assert large["estimated_gpu_vram_gib"][1] <= 22
    assert large["estimated_fullfit_and_target_minutes"][0] < large[
        "estimated_fullfit_and_target_minutes"
    ][1]


def test_invalid_ply_header_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "invalid.ply"
    path.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
    with pytest.raises(ValueError, match="vertex count absent"):
        ply_vertex_count(path)


def test_full9_preregistration_freezes_one_uniform_gauge() -> None:
    repo = Path(__file__).resolve().parents[1]
    path = (
        repo
        / "paper/artifacts/"
        "spin9_factorized_source_quantile_full9_expansion_preregistration_20260805.json"
    )
    prereg = json.loads(path.read_text(encoding="utf-8"))
    assert prereg["source_quantile_gauge"]["completion_quantile"] == 0.96
    assert prereg["source_quantile_gauge"]["tie_semantics"] == "right-continuous"
    assert prereg["source_quantile_gauge"]["quantile_or_threshold_scan"] is False
    assert prereg["uniform_representation_contract"][
        "historical_compact_substitution"
    ] == "forbidden"
    assert prereg["cohort_receipt_gate"]["partial_evaluation"] == "forbidden"
    assert len(prereg["frozen_protocol"]["scene_order"]) == 9
