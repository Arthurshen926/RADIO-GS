from pathlib import Path

import pytest
import torch

from radio_gs.scripts.train_surface_region_summary_readout import _load, _targets


def _cache(path: Path, role: str, scene: str) -> None:
    torch.save({
        "radio_features": torch.randn(2, 4, 1280), "geometry": torch.randn(2, 4, 12),
        "token_mask": torch.ones(2, 4, dtype=torch.bool), "reliability": torch.ones(2, 4, 1),
        "official_summary_tokens": torch.randn(2, 3, 1280),
        "official_crop_summaries": torch.randn(2, 3, 1536),
        "teacher_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        "metadata": {"schema_version": 2, "split_role": role, "scene_names": [scene],
            "split_file_sha256": "abc", "uses_benchmark_scenes": False,
            "uses_benchmark_test_vocabulary": False, "annotations_opened": False,
            "labels_opened": False, "instances_opened": False, "text_opened": False},
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
