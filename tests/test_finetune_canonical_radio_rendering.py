from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import radio_gs.scripts.finetune_canonical_radio_rendering as finetune


def test_method_v1_lineage_recovers_base_and_parent_stage(tmp_path: Path) -> None:
    base = tmp_path / "base.pth"
    parent = tmp_path / "spatial.pth"
    base.write_bytes(b"base")
    parent.write_bytes(b"spatial")
    parent_payload = {
        "architecture": {"coefficient_dim": 512, "local_dim": 512},
        "training_config": {"output": str(base)},
        "render_optimization": {
            "selection_policy": "capability_pareto",
            "best_step": 64,
            "official_render_capability": {"adaptor_weights": {"siglip2-g": 0.05}},
            "semantic_capability": {"enabled": False},
            "generic_text_response": {"enabled": False},
        },
    }
    args = SimpleNamespace(field_checkpoint=str(parent), construction_prior_field=[])

    lineage = finetune._method_v1_predecessor_lineage(
        args,
        parent_payload,
        current_stage="genuine_source_crop_region_summary",
    )

    assert [record["stage"] for record in lineage] == [
        "factorized_d512_l512",
        "official_siglip2_full_grid",
    ]
    assert all(len(record["sha256"]) == 64 for record in lineage)


def test_checkpoint_persists_capability_pareto_drop_authority() -> None:
    source = Path(finetune.__file__).read_text(encoding="utf-8")
    assert '"max_capability_drop": float(args.max_capability_drop)' in source


def test_render_finetune_exposes_official_siglip_spatial_capability() -> None:
    source = Path(finetune.__file__).read_text(encoding="utf-8")
    assert '"siglip2-g": float(args.siglip_spatial_render_weight)' in source
    assert "SigLIP2FeatureProjection.from_radio_checkpoint" in source
    assert "--siglip-spatial-render-weight" in source


def test_mpr_exclusions_accept_nested_registration_authority() -> None:
    metadata = {
        "registration_responsibility_contract": {
            "excluded_frame_ids": [41, 105, 152, 195]
        }
    }

    assert finetune._excluded_mpr_frame_ids(metadata) == {41, 105, 152, 195}


def test_mpr_exclusions_fail_closed_when_authority_is_missing() -> None:
    with pytest.raises(ValueError, match="does not declare"):
        finetune._excluded_mpr_frame_ids({})


def test_load_consensus_accepts_factorized_radio_cache(tmp_path: Path) -> None:
    path = tmp_path / "factorized.pt"
    torch.save(
        {
            "factorized_radio": {
                "canonical_feature": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "valid": torch.tensor([True, False]),
                "reliability": torch.ones(2, 5),
            },
            "view_counts": torch.tensor([3, 0]),
            "metadata": {"selected_frame_indices": [1]},
        },
        path,
    )

    consensus, payload = finetune._load_consensus(str(path))

    assert consensus.targets.shape == (2, 2)
    assert consensus.reliability.shape == (2, 5)
    assert consensus.observation_count.tolist() == [3, 0]
    assert payload["metadata"]["selected_frame_indices"] == [1]


def test_semantic_fidelity_aligns_one_row_teacher_rounding_difference() -> None:
    """Crop-summary teachers may round the native RADIO height up by one row."""

    predicted = torch.randn(1, 8, 45, 62)
    teacher = torch.randn(8, 46, 62)
    alpha = torch.ones(45, 62)

    absolute, centered, pixels = finetune._semantic_fidelity_losses(
        predicted,
        teacher,
        alpha,
        alpha_threshold=0.02,
    )

    assert pixels == 45 * 62
    assert torch.isfinite(absolute)
    assert torch.isfinite(centered)


def test_semantic_fidelity_rejects_non_rounding_grid_mismatch() -> None:
    predicted = torch.randn(1, 8, 45, 62)
    teacher = torch.randn(8, 47, 62)

    with pytest.raises(ValueError, match="semantic teacher/prediction mismatch"):
        finetune._semantic_fidelity_losses(
            predicted,
            teacher,
            torch.ones(45, 62),
            alpha_threshold=0.02,
        )


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
