from pathlib import Path

import pytest
import torch

from radio_gs.scripts.train_surface_region_summary_readout import (
    _load,
    _paths,
    _seed_training,
    _targets,
    inject_tangent_direction_noise,
)
from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.interfaces.surface_region_summary import (
    SurfaceRegionSummaryReadoutV2,
)


def _cache(path: Path, role: str, scene: str) -> None:
    contract = SurfaceRegionContractV2(minimum_tokens=1, maximum_tokens=4)
    torch.save({
        "radio_features": torch.randn(2, 4, 1280), "geometry": torch.randn(2, 4, 14),
        "token_mask": torch.ones(2, 4, dtype=torch.bool), "reliability": torch.ones(2, 4, 1),
        "anchor_index": torch.tensor([0, 1]),
        "official_summary_tokens": torch.randn(2, 3, 1280),
        "official_crop_summaries": torch.randn(2, 3, 1536),
        "teacher_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        "metadata": {"schema_version": 3, "split_role": role, "scene_names": [scene],
            "split_file_sha256": "abc", "uses_benchmark_scenes": False,
            "uses_benchmark_test_vocabulary": False, "annotations_opened": False,
            "labels_opened": False, "instances_opened": False, "text_opened": False,
            "radio_checkpoint_sha256": "radio",
            "region_contract": contract.to_dict(),
            "region_contract_version": contract.version,
            "region_contract_sha256": contract.digest},
    }, path)


def test_load_and_multiview_targets(tmp_path: Path) -> None:
    path = tmp_path / "train.pt"; _cache(path, "train", "scene1")
    data, meta = _load([path], "train")
    token, descriptor, views, mask = _targets(data, torch.tensor([0, 1]))
    assert token.shape == (2, 1280) and descriptor.shape == (2, 1536)
    assert views.shape == (2, 3, 1536) and mask.shape == (2, 3)
    assert meta["scenes"] == ["scene1"]
    assert data["scene_ids"] == ["scene1", "scene1"]


def test_cache_paths_accept_absolute_globs(tmp_path: Path) -> None:
    first = tmp_path / "train_shard0.pt"
    second = tmp_path / "train_shard1.pt"
    first.touch()
    second.touch()

    assert _paths(str(tmp_path / "train_shard*.pt")) == [
        first,
        second,
    ]


def test_training_seed_covers_model_initialization_and_random_stream() -> None:
    first_generator = _seed_training(17, device="cpu")
    first_model = SurfaceRegionSummaryReadoutV2(
        feature_dim=8,
        hidden_dim=4,
    )
    first_random = torch.randn(5, generator=first_generator)

    second_generator = _seed_training(17, device="cpu")
    second_model = SurfaceRegionSummaryReadoutV2(
        feature_dim=8,
        hidden_dim=4,
    )
    second_random = torch.randn(5, generator=second_generator)

    for name, first_value in first_model.state_dict().items():
        torch.testing.assert_close(
            first_value,
            second_model.state_dict()[name],
        )
    torch.testing.assert_close(first_random, second_random)
    with pytest.raises(ValueError, match="non-negative"):
        _seed_training(-1, device="cpu")


def test_load_rejects_annotation_access(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"; _cache(path, "train", "scene1")
    payload = torch.load(path); payload["metadata"]["labels_opened"] = True; torch.save(payload, path)
    with pytest.raises(ValueError): _load([path], "train")


def test_canonical_noise_stays_on_unit_sphere_and_preserves_padding() -> None:
    values = torch.nn.functional.normalize(torch.randn(2, 4, 16), dim=-1)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    perturbed = inject_tangent_direction_noise(values, mask, angle_degrees=8.0)
    torch.testing.assert_close(
        perturbed[mask].norm(dim=-1), torch.ones(int(mask.sum())), atol=1e-5, rtol=1e-5
    )
    assert torch.equal(perturbed[~mask], torch.zeros_like(perturbed[~mask]))
