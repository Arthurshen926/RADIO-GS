from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from radio_gs.querying.all_available_source_view_authority import (
    validate_supplemental_responsibility,
)
from radio_gs.querying.all_available_source_views import audit_source_view_domain
from radio_gs.rendering.sparse_marginal_authority import (
    SPARSE_EXACT_MARGINAL_AUTHORITY_SCHEMA,
    SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
    sparse_exact_marginal_formula_contract,
)
from radio_gs.scripts import materialize_lerf_o1_o2_all_available_streaming as all_view
from radio_gs.scripts import materialize_lerf_o1_o2_streaming as legacy
from radio_gs.utils.immutable_artifacts import sha256_file


def _supplement_fixture(tmp_path: Path) -> tuple[dict, dict, object]:
    audit = audit_source_view_domain(
        feature_frame_ids=[1, 2, 3, 4, 5],
        excluded_frame_ids=[2],
        legacy_frame_ids=[1, 4],
    )
    legacy_record = {"path": "/frozen/legacy.json", "sha256": "a" * 64}
    feature_record = {"path": "/frozen/features.json", "sha256": "b" * 64}
    legacy_metadata = {
        "config": "/frozen/config.yaml",
        "checkpoint": "/frozen/geometry.pt",
        "geometry_checkpoint_sha256": "c" * 64,
        "xyz_sha256": "d" * 64,
        "gaussian_state_sha256": "e" * 64,
        "feature_height": 4,
        "feature_width": 5,
        "excluded_frame_ids": [2],
    }
    reference = {
        "responsibility": {
            "metadata": legacy_metadata,
            "num_gaussians": 7,
            "num_pixels": 20,
        },
        "records": {
            "responsibility_authority": legacy_record,
            "feature_manifest": feature_record,
        },
    }
    root = tmp_path / "supplement"
    root.mkdir()
    views = []
    for index, frame in enumerate(audit.omitted_frames):
        shard = root / f"view_{index}.pt"
        shard.write_bytes(f"frame={frame}".encode("ascii"))
        views.append(
            {
                "view_index": index,
                "frame_index": frame,
                "relative_path": shard.name,
                "sha256": sha256_file(shard),
                "num_hits": index + 1,
            }
        )
    metadata = {
        **legacy_metadata,
        "supplement_contract": (
            "radio_gs.lerf_omitted_source_view_exact_marginal_supplement.v1"
        ),
        "legacy_responsibility_authority": legacy_record,
        "feature_manifest": feature_record,
        "feature_independent": True,
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "target_metrics_opened": False,
    }
    payload = {
        "schema": SPARSE_EXACT_MARGINAL_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "formula_contract": sparse_exact_marginal_formula_contract(),
        "formula_sha256": SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
        "metadata": metadata,
        "frame_indices": list(audit.omitted_frames),
        "num_gaussians": 7,
        "num_pixels": 20,
        "views": views,
        "total_hits": sum(row["num_hits"] for row in views),
    }
    return payload, reference, audit


def test_supplement_must_be_exact_omitted_axis(tmp_path: Path) -> None:
    payload, reference, audit = _supplement_fixture(tmp_path)
    validated = validate_supplemental_responsibility(
        payload,
        source_path=tmp_path / "supplement" / "manifest.json",
        audit=audit,
        reference=reference,
    )
    assert validated["frame_indices"] == [3, 5]

    broken = copy.deepcopy(payload)
    broken["frame_indices"] = [5, 3]
    with pytest.raises(ValueError, match="supplemental responsibility"):
        validate_supplemental_responsibility(
            broken,
            source_path=tmp_path / "supplement" / "manifest.json",
            audit=audit,
            reference=reference,
        )


@pytest.mark.parametrize("corruption", ["overlap", "missing", "sidecar", "lineage"])
def test_supplement_fails_closed(tmp_path: Path, corruption: str) -> None:
    payload, reference, audit = _supplement_fixture(tmp_path)
    broken = copy.deepcopy(payload)
    if corruption == "overlap":
        broken["frame_indices"][0] = 1
        broken["views"][0]["frame_index"] = 1
    elif corruption == "missing":
        broken["frame_indices"].pop()
        broken["views"].pop()
        broken["total_hits"] = broken["views"][0]["num_hits"]
    elif corruption == "sidecar":
        broken["views"][0]["sha256"] = "0" * 64
    else:
        broken["metadata"]["geometry_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_supplemental_responsibility(
            broken,
            source_path=tmp_path / "supplement" / "manifest.json",
            audit=audit,
            reference=reference,
        )


def _retained_top4(order: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    descriptors = torch.zeros(1, legacy.TOP_VIEW_COUNT, 1536, dtype=torch.float16)
    mass = torch.zeros(1, legacy.TOP_VIEW_COUNT)
    frames = torch.full((1, legacy.TOP_VIEW_COUNT), -1, dtype=torch.int32)
    masses = {10: 0.8, 20: 0.9, 30: 0.7, 40: 0.9, 50: 0.6}
    for frame in order:
        observation = torch.full((1, 1536), frame / 100.0, dtype=torch.float16)
        legacy._update_top_views(
            top_descriptors=descriptors,
            top_mass=mass,
            top_frame_ids=frames,
            rows=torch.tensor([0]),
            descriptors=observation,
            mass=torch.tensor([masses[frame]]),
            frame_id=frame,
        )
    return legacy._canonicalize_view_axis(descriptors, mass, frames)


def test_global_top4_merge_is_source_partition_and_order_invariant() -> None:
    forward = _retained_top4([10, 20, 30, 40, 50])
    partitioned = _retained_top4([30, 50, 10, 40, 20])
    for left, right in zip(forward, partitioned):
        assert torch.equal(left, right)
    assert forward[2].tolist() == [[20, 40, 10, 30]]


def test_contract_is_dynamic_source_only_and_legacy_core_is_unchanged() -> None:
    contract = all_view.method_contract()
    assert contract["source_view_count"] == "runtime_exact_all_available_count"
    assert contract["legacy_default_and_contract_modified"] is False
    assert contract["per_scene_or_per_query_hyperparameters"] is False
    assert contract["target_metric_execution_authorized"] is False
    assert all(value is False for key, value in all_view.access_audit().items() if key.startswith("target_"))
    assert sha256_file(Path(legacy.__file__).resolve()) == (
        "f779d025e0754dec583c4565542995c6133e4217d60bcf725090336c81370058"
    )
