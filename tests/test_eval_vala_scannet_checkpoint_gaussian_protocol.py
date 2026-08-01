from pathlib import Path

import numpy as np
import pytest
import torch

from radio_gs.scripts.eval_vala_scannet_checkpoint_gaussian_protocol import (
    VALA_CODE9_SCENES,
    VALA_PAPER8_SCENES,
    _checkpoint_arrays,
    _predict,
    _resolve_cohort_scenes,
)


def _write_checkpoint(path: Path) -> None:
    count = 2
    model_args = (
        0,
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        torch.zeros(count, 1, 3),
        torch.zeros(count, 0, 3),
        torch.log(torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.2, 0.2]])),
        torch.tensor([[2.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0], [2.0]]),
        torch.tensor([[3.0, 0.0], [0.0, 4.0]]),
        torch.zeros(count),
        torch.zeros(count, 1),
        torch.zeros(count, 1),
        {},
        1.0,
    )
    torch.save((model_args, 0), path)


def test_checkpoint_arrays_apply_graphdeco_activations(tmp_path: Path):
    path = tmp_path / "checkpoint.pth"
    _write_checkpoint(path)

    arrays = _checkpoint_arrays(path)

    assert np.allclose(arrays["scale"][0], [0.1, 0.2, 0.3])
    assert np.allclose(arrays["rotation"][:, 0], 1.0)
    assert np.allclose(arrays["opacity"], [0.5, torch.sigmoid(torch.tensor(2.0)).item()])
    assert np.allclose(arrays["language"], np.eye(2))


def test_predict_uses_text_conditioned_class_argmax():
    prediction = _predict(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        {
            "wall": torch.tensor([1.0, 0.0]),
            "floor": torch.tensor([0.0, 1.0]),
            "bed": torch.tensor([-1.0, 0.0]),
            "chair": torch.tensor([0.0, -1.0]),
            "sofa": torch.tensor([-1.0, -1.0]),
            "table": torch.tensor([-1.0, -1.0]),
            "door": torch.tensor([-1.0, -1.0]),
            "window": torch.tensor([-1.0, -1.0]),
            "bookshelf": torch.tensor([-1.0, -1.0]),
            "toilet": torch.tensor([-1.0, -1.0]),
        },
        "10",
        chunk_size=1,
    )

    assert prediction.tolist() == [1, 2]


def test_default_paper8_and_code9_cohorts_are_not_interchangeable():
    paper8, status = _resolve_cohort_scenes("paper8", None)
    assert paper8 == list(VALA_PAPER8_SCENES)
    assert status == "paper8_canonical_paper_facing"
    code9, status = _resolve_cohort_scenes("code9", None)
    assert code9 == list(VALA_CODE9_SCENES)
    assert status == "code9_post_paper_sensitivity_only"


def test_named_cohort_rejects_scene_mismatch_and_custom_requires_scenes():
    with pytest.raises(ValueError, match="exact canonical scene list"):
        _resolve_cohort_scenes("paper8", "scene0000_00")
    with pytest.raises(ValueError, match="requires an explicit"):
        _resolve_cohort_scenes("custom", None)
    scenes, status = _resolve_cohort_scenes("custom", "scene0000_00")
    assert scenes == ["scene0000_00"]
    assert status == "custom_explicit_diagnostic"
