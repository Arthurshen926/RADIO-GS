from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from radio_gs.querying.synchronous_multiview_candidate_marginal import (
    QueryAbstention,
)
from radio_gs.scripts import build_nvos_synchronous_multiview_sam3_inventory as module
from radio_gs.scripts.build_nvos_synchronous_multiview_sam3_inventory import (
    INVENTORY_TYPE,
    PLAN_TYPE,
    run,
    validate_plan,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _npy(path: Path, value: np.ndarray) -> dict[str, str]:
    np.save(path, value, allow_pickle=False)
    return {"path": str(path), "sha256": _sha(path)}


class _FakeModel:
    def predict_inst(self, state, *, point_coords, point_labels, multimask_output):
        assert multimask_output is False
        assert point_coords.shape == (6, 2)
        assert point_labels.tolist() == [1, 1, 1, 0, 0, 0]
        mask = np.zeros((1, state[0], state[1]), dtype=np.float32)
        mask[0, :, :2] = 1.0
        return mask, np.asarray([0.9], dtype=np.float32), np.zeros((1, 2, 2))


class _FakeProcessor:
    def __init__(self):
        self.model = _FakeModel()
        self.set_image_calls = 0

    def set_image(self, image):
        self.set_image_calls += 1
        return image.height, image.width


def _plan(tmp_path: Path) -> tuple[dict, Path]:
    rgb = tmp_path / "view.png"
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8), mode="RGB").save(rgb)
    probability = _npy(
        tmp_path / "projected.npy",
        np.asarray(
            [[0.9, 0.8, 0.2, 0.1], [0.9, 0.8, 0.2, 0.1], [0.9, 0.8, 0.2, 0.1]],
            dtype=np.float32,
        ),
    )
    visibility = _npy(tmp_path / "visible.npy", np.ones((3, 4), dtype=np.uint8))
    positive = np.zeros((3, 4), dtype=np.uint8)
    positive[:, :2] = 1
    negative = np.zeros((3, 4), dtype=np.uint8)
    negative[:, 2:] = 1
    positive_record = _npy(tmp_path / "positive.npy", positive)
    negative_record = _npy(tmp_path / "negative.npy", negative)
    assignment = tmp_path / "assignment.pt"
    torch.save(
        {
            "gaussian_ids": torch.tensor([0]),
            "pixel_ids": torch.tensor([0]),
            "weights": torch.tensor([1.0]),
        },
        assignment,
    )
    base_view = {
        "view_digest": "b" * 64,
        "rgb": {"path": str(rgb), "sha256": _sha(rgb)},
        "projected_probability": probability,
        "visibility": visibility,
        "positive_authority": positive_record,
        "negative_authority": negative_record,
        "assignment": {"path": str(assignment), "sha256": _sha(assignment)},
        "log_precision": 0.25,
        "candidate_trial_rank": 0,
    }
    second_view = {**base_view, "view_digest": "a" * 64}
    plan = {
        "schema_version": 1,
        "artifact_type": PLAN_TYPE,
        "scene_id": "fern",
        "num_gaussians": 1,
        "all_candidate_view_inputs_sealed": True,
        "target_mask_opened": False,
        "target_metric_opened": False,
        "candidates": [
            {
                "candidate_digest": f"{rank + 1:064x}",
                "candidate_logit": 0.0,
                "trial_rank": rank,
                "views": [
                    {**base_view, "candidate_trial_rank": rank},
                    {**second_view, "candidate_trial_rank": rank},
                ],
            }
            for rank in range(10)
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return plan, path


def test_producer_seals_complete_cartesian_candidate_view_inventory(tmp_path: Path) -> None:
    _, plan_path = _plan(tmp_path)
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"fake official checkpoint")
    args = SimpleNamespace(
        plan=str(plan_path),
        expected_plan_sha256=_sha(plan_path),
        output_dir=str(tmp_path / "output"),
        expected_candidates=10,
        points_per_sign=3,
        checkpoint=str(checkpoint),
        expected_checkpoint_sha256=_sha(checkpoint),
        resolution=1008,
        device="cpu",
    )
    processor = _FakeProcessor()
    result = run(args, processor=processor)
    assert result["artifact_type"] == INVENTORY_TYPE
    assert result["candidate_count"] == 10
    assert result["view_count"] == 2
    assert result["view_digests"] == ["a" * 64, "b" * 64]
    assert result["all_candidate_view_predictions_sealed"] is True
    assert result["candidate_selection"] is False
    assert result["view_selection"] is False
    assert processor.set_image_calls == 2
    for candidate in result["candidates"]:
        assert [row["view_digest"] for row in candidate["views"]] == [
            "a" * 64,
            "b" * 64,
        ]
        for view in candidate["views"]:
            value = np.load(view["probability"]["path"], allow_pickle=False)
            assert value.shape == (3, 4)
            assert float(value[:, :2].mean()) == 1.0


def test_plan_rejects_candidate_with_missing_registered_view(tmp_path: Path) -> None:
    plan, _ = _plan(tmp_path)
    plan["candidates"][1]["views"] = plan["candidates"][1]["views"][:1]
    with pytest.raises(QueryAbstention, match="cohorts differ"):
        validate_plan(plan, expected_candidates=10)


def test_plan_rejects_non_hex_digest_before_output_path_use(tmp_path: Path) -> None:
    plan, _ = _plan(tmp_path)
    plan["candidates"][0]["candidate_digest"] = "/" * 64
    with pytest.raises(QueryAbstention, match="candidate cohort identity"):
        validate_plan(plan, expected_candidates=10)


def test_plan_rejects_non_systematic_or_mismatched_trial_rank(tmp_path: Path) -> None:
    plan, _ = _plan(tmp_path)
    plan["candidates"][4]["views"][0]["candidate_trial_rank"] = 3
    with pytest.raises(QueryAbstention, match="trial rank differs"):
        validate_plan(plan, expected_candidates=10)


def test_plan_hash_verifies_each_shared_assignment_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _ = _plan(tmp_path)
    original = module._load_bound
    calls = []

    def counted(record, *, label):
        calls.append((dict(record), label))
        return original(record, label=label)

    monkeypatch.setattr(module, "_load_bound", counted)
    validate_plan(plan, expected_candidates=10)
    assert len(calls) == 2


def test_plan_rejects_assignment_lineage_change_across_candidates(
    tmp_path: Path,
) -> None:
    plan, _ = _plan(tmp_path)
    plan["candidates"][1]["views"][0]["assignment"] = {
        **plan["candidates"][1]["views"][0]["assignment"],
        "sha256": "0" * 64,
    }
    with pytest.raises(QueryAbstention, match="lineage differs"):
        validate_plan(plan, expected_candidates=10)
