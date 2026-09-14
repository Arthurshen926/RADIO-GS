import json
import torch
import pytest
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.evaluation.lerf_text_cold_evaluator import _load_fragment_prototypes, FRAME_SCHEMA, MANIFEST_SCHEMA


def test_nonchronological_mapping_preserves_descriptor_token_identity(tmp_path):
    records = []
    for frame in (1, 9):
        path = tmp_path / f"{frame}.pt"
        torch.save({"schema": FRAME_SCHEMA, "frame_id": frame,
            "metadata": {"sam_fragment_payload_sha256": str(frame)},
            "masked_crop_descriptor": torch.full((1, 1536), float(frame)),
            "context_crop_descriptor": torch.full((1, 1536), float(frame + 10))}, path)
        records.append({"frame_id": frame, "proposal_count": 1, "output": str(path),
                        "output_sha256": sha256_file(path), "sam_fragment_payload_sha256": str(frame)})
    manifest = {"schema": MANIFEST_SCHEMA, "scene_label": "scene", "sam_manifests": [{"sha256": "sam"}],
                "outputs": records, "information_policy": {key: False for key in (
                    "benchmark_labels_opened", "benchmark_masks_opened", "text_queries_opened", "target_rgb_opened")}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    state = {"scene_label": "scene", "source_input_digests": {"sam_manifest_0": "sam"},
             "source_frames": [9, 1], "prototype_token_ids": torch.tensor([4, 7])}
    features, ids, _ = _load_fragment_prototypes(path, state=state)
    assert features[:, 0].tolist() == [9, 1, 19, 11]
    assert ids.tolist() == [4, 7, 4, 7]
    manifest["outputs"][0]["proposal_count"] = 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="per-frame"):
        _load_fragment_prototypes(path, state=state)
