from pathlib import Path

import numpy as np
import torch

from radio_gs.scripts.convert_rendered_pt_features_to_npy import (
    convert_directory,
    convert_tensor,
    output_frame_name,
)


def test_output_frame_name_zero_pads_rgb_id() -> None:
    assert output_frame_name(Path("rgb_7.pt")) == "frame_00007.npy"


def test_convert_tensor_outputs_hwc_and_normalizes_channels() -> None:
    tensor = torch.zeros(2, 2, 1)
    tensor[:, 0, 0] = torch.tensor([3.0, 4.0])
    tensor[:, 1, 0] = torch.tensor([0.0, 2.0])

    array = convert_tensor(tensor, normalize=True, dtype="fp32")

    assert array.shape == (2, 1, 2)
    np.testing.assert_allclose(array[0, 0], np.array([0.6, 0.8]), atol=1e-6)
    np.testing.assert_allclose(array[1, 0], np.array([0.0, 1.0]), atol=1e-6)


def test_convert_directory_accepts_backbone_layout(tmp_path: Path) -> None:
    backbone = tmp_path / "rendered" / "backbone"
    backbone.mkdir(parents=True)
    torch.save(torch.ones(3, 2, 2), backbone / "rgb_12.pt")

    written = convert_directory(tmp_path / "rendered", tmp_path / "npy", dtype="fp16")

    assert [path.name for path in written] == ["frame_00012.npy"]
    array = np.load(written[0])
    assert array.shape == (2, 2, 3)
    assert array.dtype == np.float16
