from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest
import torch

from radio_gs.scripts import materialize_rank256_o0_primitive_lifting_dryrun as dryrun


SHA = "0" * 64


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _gate(tmp_path: Path) -> dict:
    source = tmp_path / "source.json"
    source.write_text("{}\n")
    return {
        "source_result": _record(source),
        "checkpoint": {"path": str((tmp_path / "model.pt").resolve()), "sha256": SHA},
        "normalization_authority": {
            "path": str((tmp_path / "normalization.pt").resolve()),
            "sha256": SHA,
        },
        "source_promotion_authorized": True,
        "benchmark_opened": False,
    }


def _builder_args(tmp_path: Path, gate: dict) -> argparse.Namespace:
    inputs = {}
    for name in ("o0.pt", "target.pt", "accepted.pt", "renderer.pt"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        inputs[name] = path
    return argparse.Namespace(
        source_variant="v21b",
        source_result=gate["source_result"]["path"],
        expected_source_result_sha256=gate["source_result"]["sha256"],
        scene_id="figurines",
        o0_descriptor=str(inputs["o0.pt"].resolve()),
        rank256_target_descriptor=str(inputs["target.pt"].resolve()),
        target_accepted_v2=str(inputs["accepted.pt"].resolve()),
        renderer_geometry_checkpoint=str(inputs["renderer.pt"].resolve()),
        valid_row_prefix_limit=128,
        max_angle_radians=0.15,
        minimum_region_reliability=0.0,
        output=str((tmp_path / "dryrun.pt").resolve()),
        output_authority=str((tmp_path / "authority.json").resolve()),
    )


def test_builder_validates_source_before_any_target_path(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []

    def reject(*_args, **_kwargs):
        events.append("source")
        raise ValueError("source gate rejected")

    monkeypatch.setattr(dryrun.champion, "validate_champion_source", reject)
    args = argparse.Namespace(
        source_variant="v21b",
        source_result=str((tmp_path / "missing-source.json").resolve()),
        expected_source_result_sha256=SHA,
        scene_id="figurines",
        o0_descriptor=str((tmp_path / "missing-o0.pt").resolve()),
        rank256_target_descriptor=str((tmp_path / "missing-target.pt").resolve()),
        target_accepted_v2=str((tmp_path / "missing-accepted.pt").resolve()),
        renderer_geometry_checkpoint=str((tmp_path / "missing-renderer.pt").resolve()),
        valid_row_prefix_limit=2,
        max_angle_radians=0.15,
        minimum_region_reliability=0.0,
        output=str((tmp_path / "dryrun.pt").resolve()),
        output_authority=str((tmp_path / "authority.json").resolve()),
    )
    with pytest.raises(ValueError, match="source gate rejected"):
        dryrun.build_authority(args)
    assert events == ["source"]
    assert not Path(args.output_authority).exists()


def test_builder_is_no_clobber_and_seals_query_free_scope(
    tmp_path: Path, monkeypatch
) -> None:
    gate = _gate(tmp_path)
    monkeypatch.setattr(
        dryrun.champion, "validate_champion_source", lambda *_args, **_kwargs: gate
    )
    args = _builder_args(tmp_path, gate)
    result = dryrun.build_authority(args)
    assert result["status"] == "rank256_o0_lifting_dryrun_authority_built"
    payload = json.loads(Path(args.output_authority).read_text())
    assert payload["scope"]["dry_run_only"] is True
    assert payload["scope"]["formal_candidate_authorized"] is False
    assert payload["query_execution_authorized"] is False
    assert payload["metric_execution_authorized"] is False
    assert payload["access_audit"] == dryrun.access_audit()
    with pytest.raises(FileExistsError, match="lifting authority"):
        dryrun.build_authority(args)


def test_prefix_reindex_keeps_global_to_storage_alignment() -> None:
    positions, rows, mask = dryrun._reindex_regions_to_prefix(
        region_rows=torch.tensor(
            [[0, 6, 20, -1], [7, 11, -1, -1], [99, 100, -1, -1]]
        ),
        token_mask=torch.tensor(
            [
                [True, True, True, False],
                [True, True, False, False],
                [True, True, False, False],
            ]
        ),
        prefix_global_rows=torch.tensor([0, 6, 7, 11], dtype=torch.long),
    )
    assert positions.tolist() == [0, 1]
    assert rows.tolist() == [[0, 1, -1, -1], [2, 3, -1, -1]]
    assert mask.tolist() == [
        [True, True, False, False],
        [True, True, False, False],
    ]


def test_o0_validator_binds_exact_renderer_geometry_and_sparse_rows() -> None:
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    descriptor = torch.zeros((1, 3, 1536), dtype=torch.float16)
    descriptor[..., 0] = 1.0
    payload = {
        "metadata": {
            "feature_space": "official_siglip2_summary_descriptor_multiscale",
            "query_set_invariant": True,
            "text_queries_opened": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
        },
        "xyz": xyz,
        "valid": torch.tensor([True, False]),
        "global_rows": torch.tensor([0], dtype=torch.long),
        "features_by_scale": descriptor,
    }
    result = dryrun._validate_o0(payload, renderer_xyz=xyz.clone())
    assert result["descriptor"].shape == (1, 3, 1536)
    with pytest.raises(ValueError, match="geometry/order"):
        dryrun._validate_o0(payload, renderer_xyz=xyz + 1.0)

