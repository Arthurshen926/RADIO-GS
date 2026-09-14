import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from radio_gs.models.sam3_multiscale_hierarchy import pack_masks
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.contracts.build_lerf_fragment_language_memory import (
    build_frame_memory,
    preflight_inputs,
)


def _fixture(tmp_path: Path):
    image = tmp_path / "frame_00001.jpg"
    Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(image)
    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[0, 1:7, 1:7] = True
    masks[1, 2:4, 2:4] = True
    boxes = torch.tensor([[1, 1, 7, 7], [2, 2, 4, 4]], dtype=torch.int32)
    payload = {
        "schema_version": 1,
        "mask_shape": [8, 8],
        "packed_masks": pack_masks(masks),
        "boxes_xyxy": boxes,
        "quality": torch.tensor([0.9, 0.8]),
        "stability": torch.tensor([0.95, 0.9]),
        "parent_index": torch.tensor([-1, 0]),
        "proposal_area_fraction": torch.tensor([36 / 64, 4 / 64]),
    }
    sam = tmp_path / "frame_00001.pt"
    torch.save(payload, sam)
    sam_manifest = tmp_path / "sam.json"
    sam_manifest.write_text(json.dumps({
        "contract": "official-sam3-query-free-multiscale-hierarchy-manifest-v1",
        "generation_contract": {"official_decoder": True, "query_free": True},
        "images": [{
            "image_id": "frame_00001",
            "output": str(sam),
            "output_sha256": sha256_file(sam),
            "proposal_count": 2,
        }],
    }))
    authority = tmp_path / "authority.json"
    authority.write_text(json.dumps({
        "contract": "sam3-query-free-source-rgb-authority-v1",
        "scene": "scene",
        "images": [{
            "image_id": "frame_00001",
            "path": str(image),
            "sha256": sha256_file(image),
        }],
        "information_policy": {
            "benchmark_ground_truth_used": False,
            "query_text_used": False,
            "target_or_evaluation_rgb_used": False,
            "registered_source_rgb_only": True,
        },
    }))
    return authority, sam_manifest


class _Runtime:
    def encode(self, crops):
        output = torch.zeros(crops.shape[0], 1536, device=crops.device)
        output[:, 0] = 1
        return output


def test_preflight_binds_source_rgb_and_multiscale_sam(tmp_path):
    authority, sam = _fixture(tmp_path)
    _payload, records = preflight_inputs(
        scene_label="scene", source_rgb_authority=authority, sam_manifests=[sam]
    )
    assert len(records) == 1
    assert records[0]["masks"].shape == (2, 8, 8)


def test_frame_memory_uses_complete_crop_encoder_and_preserves_fragments(tmp_path):
    authority, sam = _fixture(tmp_path)
    _payload, records = preflight_inputs(
        scene_label="scene", source_rgb_authority=authority, sam_manifests=[sam]
    )
    memory = build_frame_memory(
        records[0],
        _Runtime(),
        device=torch.device("cpu"),
        crop_resolution=16,
        batch_size=1,
        context_expansion=1.5,
        masked_background_rgb=(0.5, 0.5, 0.5),
    )
    assert memory["masked_crop_descriptor"].shape == (2, 1536)
    assert memory["context_crop_descriptor"].shape == (2, 1536)
    assert memory["masked_crop_descriptor"].dtype == torch.float16
    assert memory["parent_index"].tolist() == [-1, 0]
    assert memory["metadata"]["text_queries_opened"] is False


def test_preflight_rejects_mutated_rgb(tmp_path):
    authority, sam = _fixture(tmp_path)
    Path(json.loads(authority.read_text())["images"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="source RGB digest"):
        preflight_inputs(
            scene_label="scene", source_rgb_authority=authority, sam_manifests=[sam]
        )
