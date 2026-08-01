import json
from pathlib import Path

import numpy as np
import pytest

from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_a import sha256_file
from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_d import (
    PHASE_D_SCHEMA_VERSION,
    PHASE_D_STATUS,
)
from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_e import (
    LudvigPFPRPhaseEError,
    PhaseEConfig,
    audit_phase_d_predictions,
)
from radio_gs.benchmarks.scannet_pfpr.protocol import canonical_json_sha256


def _attempt(root: Path) -> tuple[Path, str]:
    predictions = root / "predictions"
    predictions.mkdir()
    queries = []
    for index in range(10):
        query_id = f"scene0050_02_pfpr_{index:03d}"
        path = predictions / f"{query_id}.npy"
        np.save(path, np.arange(5, dtype=np.float32), allow_pickle=False)
        queries.append(
            {
                "query_id": query_id,
                "scores": {
                    "relative_path": f"predictions/{query_id}.npy",
                    "sha256": sha256_file(path),
                    "shape": [5],
                    "dtype": "float32",
                },
            }
        )
    manifest = {
        "schema_version": PHASE_D_SCHEMA_VERSION,
        "status": PHASE_D_STATUS,
        "result_eligible": False,
        "scene_id": "scene0050_02",
        "queries": queries,
        "queries_sha256": canonical_json_sha256(queries),
    }
    path = root / "run_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, sha256_file(path)


def test_phase_e_pre_private_audit_accepts_exact_prediction_set(tmp_path: Path) -> None:
    _path, digest = _attempt(tmp_path)
    manifest, predictions = audit_phase_d_predictions(
        PhaseEConfig(tmp_path, digest, tmp_path, tmp_path / "out")
    )
    assert len(predictions) == 10
    assert manifest["scene_id"] == "scene0050_02"


def test_phase_e_pre_private_audit_rejects_extra_prediction(tmp_path: Path) -> None:
    _path, digest = _attempt(tmp_path)
    np.save(tmp_path / "predictions" / "extra.npy", np.ones(5, dtype=np.float32))
    with pytest.raises(LudvigPFPRPhaseEError, match="extra/missing"):
        audit_phase_d_predictions(
            PhaseEConfig(tmp_path, digest, tmp_path, tmp_path / "out")
        )
