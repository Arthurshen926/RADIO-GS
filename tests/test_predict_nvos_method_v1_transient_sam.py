from __future__ import annotations

import numpy as np
from PIL import Image

from radio_gs.scripts.predict_nvos_method_v1_transient_sam import run_sam_trials


class _Model:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray, bool]] = []

    def predict_inst(
        self,
        state,
        *,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        multimask_output: bool,
    ):
        self.calls.append((point_coords.copy(), point_labels.copy(), multimask_output))
        height, width = state["shape"]
        mask = np.zeros((1, height, width), dtype=bool)
        mask[:, :, : width // 2] = True
        return mask, np.array([0.75], dtype=np.float32), np.zeros((1, 4, 4))


class _Processor:
    def __init__(self) -> None:
        self.model = _Model()

    def set_image(self, image: Image.Image):
        return {"shape": (image.height, image.width)}


def test_transient_sam_uses_ten_signed_trials_without_candidate_selection() -> None:
    margin = np.ones((10, 10), dtype=np.float32)
    margin[:, 5:] = -1.0
    processor = _Processor()

    result = run_sam_trials(
        processor,
        Image.new("RGB", (40, 30)),
        margin,
        device="cpu",
        amp_dtype=None,
    )

    assert len(processor.model.calls) == 10
    for points, labels, multimask in processor.model.calls:
        assert points.shape == (6, 2)
        np.testing.assert_array_equal(labels, np.array([1, 1, 1, 0, 0, 0]))
        assert multimask is False
    assert result["trial_masks"].shape == (10, 1, 30, 40)
    np.testing.assert_array_equal(result["binary_mask"][:, :20], True)
    np.testing.assert_array_equal(result["binary_mask"][:, 20:], False)
    np.testing.assert_allclose(result["continuous_margin"][:, :20], 0.5)
    np.testing.assert_allclose(result["continuous_margin"][:, 20:], -0.5)
