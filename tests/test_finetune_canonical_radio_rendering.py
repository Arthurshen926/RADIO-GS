from pathlib import Path

import pytest
import torch

import radio_gs.scripts.finetune_canonical_radio_rendering as finetune


def test_checkpoint_persists_capability_pareto_drop_authority() -> None:
    source = Path(finetune.__file__).read_text(encoding="utf-8")
    assert '"max_capability_drop": float(args.max_capability_drop)' in source


class _IdentityAdaptor(torch.nn.Module):
    def forward(self, values):
        return values


class _OneFrameDataset:
    def __init__(self, features: torch.Tensor) -> None:
        self.frame_indices = [7]
        self._features = features

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        assert index == 0
        return {
            "radio_features": self._features.clone(),
            "pose_w2c": torch.eye(4),
        }


def test_capability_validation_uses_direct_official_maps_without_reprojection(
    monkeypatch,
) -> None:
    """The high-fidelity branch compares predictions to official maps directly."""

    predicted = torch.tensor([[[1.0]], [[0.0]]])
    # A deliberately different raw feature proves that direct maps are not
    # sent through the raw adaptor path again.
    raw_dataset = _OneFrameDataset(torch.tensor([[[0.0]], [[1.0]]]))
    official_dataset = _OneFrameDataset(predicted)

    def _render(*_args, **_kwargs):
        return {
            "feature_map": predicted.clone(),
            "alpha_map": torch.ones(1, 1),
        }

    monkeypatch.setattr(finetune, "render_canonical_radio", _render)
    kwargs = {
        "adaptors": {"sam3": _IdentityAdaptor()},
        "reliability_splat": False,
        "alpha_threshold": 0.02,
    }
    legacy = finetune._mean_multicapability_fidelity(
        torch.nn.Identity(),
        None,
        None,
        raw_dataset,
        {7: 0},
        [7],
        torch.device("cpu"),
        **kwargs,
    )
    direct = finetune._mean_multicapability_fidelity(
        torch.nn.Identity(),
        None,
        None,
        raw_dataset,
        {7: 0},
        [7],
        torch.device("cpu"),
        capability_teacher_datasets={"sam3": official_dataset},
        capability_teacher_frame_to_index={"sam3": {7: 0}},
        **kwargs,
    )

    assert legacy["sam3"] == pytest.approx(0.0, abs=1e-6)
    assert direct["sam3"] == pytest.approx(1.0, abs=1e-6)
