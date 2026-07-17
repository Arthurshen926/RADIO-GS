from pathlib import Path

import pytest
import torch

from radio_gs.scripts.train_surface_region_summary_readout import (
    _load,
    _targets,
    inject_tangent_direction_noise,
)
from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2


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
