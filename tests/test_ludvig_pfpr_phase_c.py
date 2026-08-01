import json
from pathlib import Path

import numpy as np
import pytest
import torch

from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_c import (
    LudvigPFPRPhaseCError,
    PhaseCConfig,
    audit_phase_b_attempt,
    reconstruct_ludvig_feature_map,
)
from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_b import (
    PHASE_B_SCHEMA_VERSION,
    PHASE_B_STATUS,
)
from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_a import sha256_file
from radio_gs.benchmarks.scannet_pfpr.protocol import canonical_json_sha256


def test_reconstruct_overlap_mean_and_resize_identity() -> None:
    tokens = torch.zeros(2, 2, 2, 1)
    tokens[0].fill_(2)
    tokens[1].fill_(4)
    plan = {
        "indices_yx": [[0, 0], [0, 2]],
        "effective_crop_size": 4,
        "aligned_height": 4,
        "aligned_width": 6,
    }
    result = reconstruct_ludvig_feature_map(
        tokens, plan, output_height=4, output_width=6
    )
    expected = torch.tensor([[2, 2, 3, 3, 4, 4]], dtype=torch.float32).repeat(4, 1)
    assert tuple(result.shape) == (1, 4, 6)
    assert torch.allclose(result[0], expected)


def test_reconstruct_rejects_uncovered_pixels() -> None:
    with pytest.raises(LudvigPFPRPhaseCError, match="uncovered"):
        reconstruct_ludvig_feature_map(
            torch.ones(1, 1, 1, 1),
            {
                "indices_yx": [[0, 0]],
                "effective_crop_size": 1,
                "aligned_height": 2,
                "aligned_width": 2,
            },
            output_height=2,
            output_width=2,
        )


def test_phase_b_audit_requires_full_ordered_view_binding(tmp_path: Path) -> None:
    views = [{"rank": index} for index in range(120)]
    payload = {
        "schema_version": PHASE_B_SCHEMA_VERSION,
        "status": PHASE_B_STATUS,
        "result_eligible": False,
        "views": views,
        "views_sha256": canonical_json_sha256(views),
    }
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    config = PhaseCConfig(
        phase_b_dir=tmp_path,
        expected_phase_b_manifest_sha256=sha256_file(manifest),
        ludvig_upstream=tmp_path,
        output_dir=tmp_path / "out",
    )
    root, loaded = audit_phase_b_attempt(config)
    assert root == tmp_path.resolve()
    assert loaded["views_sha256"] == canonical_json_sha256(views)


def test_phase_b_audit_rejects_mutated_order_binding(tmp_path: Path) -> None:
    views = [{"rank": index} for index in range(120)]
    payload = {
        "schema_version": PHASE_B_SCHEMA_VERSION,
        "status": PHASE_B_STATUS,
        "result_eligible": False,
        "views": views,
        "views_sha256": "0" * 64,
    }
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    config = PhaseCConfig(
        phase_b_dir=tmp_path,
        expected_phase_b_manifest_sha256=sha256_file(manifest),
        ludvig_upstream=tmp_path,
        output_dir=tmp_path / "out",
    )
    with pytest.raises(LudvigPFPRPhaseCError, match="ordered view"):
        audit_phase_b_attempt(config)
