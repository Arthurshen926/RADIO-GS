from pathlib import Path

import numpy as np
import pytest

from radio_gs.scripts.bind_nvos_current_field_exact_authority import (
    AuthorityError,
    validate_protocol_semantics,
    validate_score_file,
    validate_target_fence,
)


def _protocol():
    return {
        "benchmark": "NVOS",
        "aggregation": "per_frame_then_per_scene_then_dataset_scene_macro",
        "resize": "nearest",
        "prediction_representation": "continuous_margin",
        "threshold_comparison": "greater_or_equal",
        "score_semantics": "cosine_similarity_foreground_minus_background",
        "score_temperature": "none",
        "target_rgb_during_field_training": "forbidden",
        "target_rgb_at_query": "forbidden",
        "target_mask_use": "scoring_only",
        "within_scene_aggregation": "single_official_target",
        "cohort": [
            "fern", "flower", "fortress", "horns_center", "horns_left",
            "leaves", "orchids", "trex",
        ],
        "threshold": {"mode": "fixed", "value": 0.0},
    }


def test_protocol_requires_zero_margin_nearest_full8():
    validate_protocol_semantics(_protocol())
    changed = _protocol()
    changed["resize"] = "bilinear"
    with pytest.raises(AuthorityError, match="resize"):
        validate_protocol_semantics(changed)


def test_target_fence_rejects_training_leakage():
    scene = {
        "scene_id": "fern",
        "prompt_frame_ids": ["reference"],
        "evaluation_frame_ids": ["target"],
        "training_frames": [{"frame_id": "reference"}],
        "excluded_training_frame_ids": ["target"],
        "target_rgb_policy": "excluded_from_field_training_and_query",
    }
    assert validate_target_fence(scene) == ("reference", "target")
    scene["training_frames"].append({"frame_id": "target"})
    with pytest.raises(AuthorityError, match="leaked"):
        validate_target_fence(scene)


def test_score_binding_requires_finite_float32(tmp_path: Path):
    path = tmp_path / "score.npy"
    np.save(path, np.asarray([[0.25, -0.5]], dtype=np.float32))
    import hashlib

    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    row = validate_score_file(path, expected)
    assert row["dtype"] == "float32"
    np.save(path, np.asarray([[0.25]], dtype=np.float64))
    changed = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(AuthorityError, match="float32"):
        validate_score_file(path, changed)
