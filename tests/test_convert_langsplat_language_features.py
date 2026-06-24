import json

import numpy as np
import pytest
import torch

from radio_gs.scripts import convert_langsplat_language_features as conv


def test_materialize_dense_feature_handles_invalid_pixels():
    features = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    seg = np.array(
        [
            [[0, 1], [2, -1]],
            [[1, 1], [0, 0]],
            [[2, 2], [1, 1]],
            [[0, 0], [2, 2]],
        ],
        dtype=np.int32,
    )

    dense = conv.materialize_dense_feature(
        features,
        seg,
        level=0,
        output_size=None,
        dtype=torch.float32,
    )

    assert dense.shape == (2, 2, 2)
    assert torch.allclose(dense[:, 0, 0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(dense[:, 0, 1], torch.tensor([0.0, 1.0]))
    assert torch.allclose(dense[:, 1, 0], torch.tensor([0.6, 0.8]), atol=1e-6)
    assert torch.allclose(dense[:, 1, 1], torch.zeros(2))


def test_materialize_dense_feature_rejects_out_of_range_ids():
    features = np.eye(2, dtype=np.float32)
    seg = np.zeros((4, 2, 2), dtype=np.int32)
    seg[1, 0, 0] = 2

    with pytest.raises(ValueError, match="out of range"):
        conv.materialize_dense_feature(
            features,
            seg,
            level=1,
            output_size=None,
            dtype=torch.float32,
        )


def test_feature_stem_to_frame_id_lerf_and_scannet():
    assert conv.feature_stem_to_frame_id("frame_00016") == 16
    assert conv.feature_stem_to_frame_id("0") == 0
    assert conv.output_feature_name("frame_00016") == "rgb_16.pt"
    assert conv.output_feature_name("0") == "rgb_0.pt"


def test_discover_feature_pairs_requires_matching_segmentation(tmp_path):
    source = tmp_path / "language_features"
    source.mkdir()
    np.save(source / "0_f.npy", np.eye(2, dtype=np.float32))
    np.save(source / "1_s.npy", np.zeros((4, 2, 2), dtype=np.int32))

    assert conv.discover_feature_pairs(source) == []


def test_convert_scene_writes_manifest_and_tensor(tmp_path):
    source = tmp_path / "language_features"
    source.mkdir()
    np.save(source / "frame_00001_f.npy", np.eye(3, dtype=np.float32))
    np.save(source / "frame_00001_s.npy", np.zeros((4, 2, 2), dtype=np.int32))
    output = tmp_path / "converted"

    manifest = conv.convert_scene(
        source,
        output,
        levels=[1],
        output_size=(1, 1),
        dtype=torch.float32,
        dry_run=False,
    )

    tensor_path = output / "l1" / "backbone" / "rgb_1.pt"
    assert tensor_path.exists()
    tensor = torch.load(tensor_path, map_location="cpu")
    assert tensor.shape == (3, 1, 1)
    assert manifest["levels"] == [1]
    assert manifest["frames_converted"] == 1
    assert (output / "l1" / "samclip_manifest.json").exists()
    saved = json.loads((output / "l1" / "samclip_manifest.json").read_text(encoding="utf-8"))
    assert saved["feature_dim"] == 3
    assert saved["outputs"][0]["tensor"] == "backbone/rgb_1.pt"
