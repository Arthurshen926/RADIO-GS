from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from radio_gs.scripts.build_nvos_synchronous_multiview_box_sam3_inventory import (
    predict_box_view,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _npy(path: Path, value: np.ndarray) -> dict[str, str]:
    np.save(path, value, allow_pickle=False)
    return {"path": str(path), "sha256": _sha(path)}


class _Processor:
    def set_image(self, image):
        return {"height": image.height, "width": image.width}

    def add_geometric_prompt(self, box, label, state):
        assert label is True
        assert len(box) == 4
        height, width = state["height"], state["width"]
        masks = np.zeros((2, 1, height, width), dtype=bool)
        masks[0, 0, :, : width // 2] = True
        masks[1, 0, :, width // 2 :] = True
        return {"masks": masks, "scores": np.asarray([0.2, 0.9], dtype=np.float32)}


def test_box_extent_is_selected_by_signed_field_not_sam_score(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8), mode="RGB").save(rgb)
    probability = np.full((6, 8), 0.1, dtype=np.float32)
    probability[:, :4] = 0.9
    record = {
        "rgb": {"path": str(rgb), "sha256": _sha(rgb)},
        "projected_probability": _npy(tmp_path / "p.npy", probability),
        "visibility": _npy(tmp_path / "v.npy", np.ones_like(probability, dtype=np.uint8)),
    }
    selected, report = predict_box_view(
        _Processor(), record, device="cpu", box_padding_pixels=0
    )
    assert selected.shape == probability.shape
    assert bool((selected[:, :4] == 1).all())
    assert bool((selected[:, 4:] == 0).all())
    assert report["accepted"] is True
    assert report["selected_index"] == 0
    assert report["candidate_count"] == 2
