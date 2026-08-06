from __future__ import annotations

from argparse import Namespace
import copy
import hashlib
from pathlib import Path

from PIL import Image
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from radio_gs.scripts import materialize_clean_source_rgb_scene_authority as source_sealer
from radio_gs.scripts import extract_radio_features as feature_extraction
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.scripts import materialize_official_multiview_siglip2_teacher_authority as producer
from radio_gs.utils.immutable_artifacts import sha256_file, write_frozen_json


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _source_inputs(tmp_path: Path) -> Namespace:
    rgb = tmp_path / "color"
    rgb.mkdir(parents=True)
    frames = []
    for frame, color in ((0, (255, 0, 0)), (2, (0, 255, 0))):
        path = rgb / f"{frame:06d}.png"
        Image.new("RGB", (12, 8), color).save(path)
        frames.append(
            {
                "frame_idx": frame,
                "source_file": path.name,
                "source_sha256": sha256_file(path),
            }
        )
    field = {
        "scene_id": "scene0001_00",
        "selected_frame_indices": [0, 2],
        "field_frame_count": 2,
        "field_frame_manifest_sha256": _sha("field-frames"),
        "excluded_query_source_frame_count": 0,
        "uses_private_anchor": False,
        "uses_private_depth_pixel": False,
        "uses_instances_or_semantic_labels": False,
        "contains_instance_or_label_directories": False,
    }
    feature = {
        "scene": "scene0001_00",
        "num_frames": 2,
        "frames": frames,
        "execution": {
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
        "excluded_image_names": [],
        "excluded_image_stems": [],
    }
    field_path = tmp_path / "field.json"
    feature_path = tmp_path / "feature.json"
    write_frozen_json(field_path, field)
    write_frozen_json(feature_path, feature)
    return Namespace(
        field_source_contract=str(field_path),
        expected_field_source_contract_sha256=sha256_file(field_path),
        feature_frame_manifest=str(feature_path),
        expected_feature_frame_manifest_sha256=sha256_file(feature_path),
        source_rgb_root=str(rgb),
        output=str(tmp_path / "source_rgb_authority.json"),
        preflight_only=True,
    )


def test_source_rgb_sealer_is_caller_bound_and_noclobber(tmp_path: Path) -> None:
    args = _source_inputs(tmp_path)
    ready = source_sealer.materialize(args)
    assert ready["status"] == "ready"
    assert ready["frames"] == 2
    assert ready["outputs_written"] is False
    assert not Path(args.output).exists()

    args.preflight_only = False
    result = source_sealer.materialize(args)
    assert result["status"] == "materialized"
    authority = producer.validate_source_rgb_scene_authority(
        __import__("json").loads(Path(args.output).read_text())
    )
    assert [record["frame_id"] for record in authority["frame_records"]] == [
        "000000",
        "000002",
    ]
    assert authority["frame_records"][0]["source_image_height"] == 8
    with pytest.raises(FileExistsError, match="refuses to clobber"):
        source_sealer.materialize(args)

    wrong = _source_inputs(tmp_path / "wrong")
    wrong.expected_feature_frame_manifest_sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 differs"):
        source_sealer.materialize(wrong)
    assert not Path(wrong.output).exists()


def test_source_rgb_sealer_accepts_only_fully_validated_strict_resume_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _source_inputs(tmp_path)
    legacy_feature_path = Path(args.feature_frame_manifest)
    feature = __import__("json").loads(legacy_feature_path.read_text())
    feature["image_dir"] = str(Path(args.source_rgb_root).resolve())
    feature["execution"] = {
        "resume_partial": True,
        "resume_contract": feature_extraction.RESUME_CONTRACT_FILENAME,
        "resume_contract_sha256": _sha("resume"),
        "resume_contract_file_sha256": _sha("resume-file"),
    }
    feature_path = tmp_path / "strict-features" / "frame_manifest.json"
    write_frozen_json(feature_path, feature)
    args.feature_frame_manifest = str(feature_path)
    args.expected_feature_frame_manifest_sha256 = sha256_file(feature_path)
    observed = {}

    def validate(output_root, manifest, **kwargs):
        observed["output_root"] = Path(output_root)
        observed["manifest"] = manifest
        observed.update(kwargs)
        return {"num_frames": 2}

    monkeypatch.setattr(
        feature_extraction,
        "_validate_final_output_bundle",
        validate,
    )
    ready = source_sealer.materialize(args)

    assert ready["status"] == "ready"
    assert observed["output_root"] == feature_path.parent
    assert observed["manifest"] == feature
    assert observed["verify_source_images"] is True
    assert observed["expected_manifest_sha256"] == sha256_file(feature_path)


def test_source_rgb_sealer_rejects_incomplete_or_misbound_strict_resume_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _source_inputs(tmp_path)
    legacy_feature_path = Path(args.feature_frame_manifest)
    feature = __import__("json").loads(legacy_feature_path.read_text())
    feature["image_dir"] = str(Path(args.source_rgb_root).resolve())
    feature["execution"] = {
        "resume_partial": False,
        "resume_contract": feature_extraction.RESUME_CONTRACT_FILENAME,
    }
    feature_path = tmp_path / "incomplete-features" / "frame_manifest.json"
    write_frozen_json(feature_path, feature)
    args.feature_frame_manifest = str(feature_path)
    args.expected_feature_frame_manifest_sha256 = sha256_file(feature_path)
    called = False

    def validate(*_args, **_kwargs):
        nonlocal called
        called = True
        return {"num_frames": 2}

    monkeypatch.setattr(
        feature_extraction,
        "_validate_final_output_bundle",
        validate,
    )
    with pytest.raises(ValueError, match="not complete"):
        source_sealer.materialize(args)
    assert called is False

    feature["execution"] = {
        "resume_partial": True,
        "resume_contract": "unrecognized-resume-contract.json",
    }
    unsupported_path = tmp_path / "unsupported-features" / "frame_manifest.json"
    write_frozen_json(unsupported_path, feature)
    args.feature_frame_manifest = str(unsupported_path)
    args.expected_feature_frame_manifest_sha256 = sha256_file(unsupported_path)
    with pytest.raises(ValueError, match="unsupported resume provenance"):
        source_sealer.materialize(args)
    assert called is False

    feature["execution"] = {
        "resume_partial": True,
        "resume_contract": feature_extraction.RESUME_CONTRACT_FILENAME,
    }
    feature["image_dir"] = str((tmp_path / "different-rgb-root").resolve())
    feature_path = tmp_path / "misbound-features" / "frame_manifest.json"
    write_frozen_json(feature_path, feature)
    args.feature_frame_manifest = str(feature_path)
    args.expected_feature_frame_manifest_sha256 = sha256_file(feature_path)
    with pytest.raises(ValueError, match="image directory differs"):
        source_sealer.materialize(args)
    assert called is True


def _crop_accepted() -> dict:
    return {
        "accepted_base_valid": torch.ones(5, dtype=torch.bool),
        "region_rows": torch.tensor(
            [[0, 1, -1], [2, 3, -1], [4, -1, -1]], dtype=torch.long
        ),
        "token_mask": torch.tensor(
            [[True, True, False], [True, True, False], [True, False, False]]
        ),
        "anchor_index": torch.tensor([0, 1, 0], dtype=torch.long),
    }


def test_crop_evidence_is_exact_marginal_and_anchor_visible() -> None:
    view = {
        "num_pixels": 16,
        "gaussian_ids": torch.tensor([0, 0, 1, 2, 4]),
        "pixel_ids": torch.tensor([1, 5, 10, 15, 3]),
        "base_weights": torch.tensor([0.4, 0.2, 0.3, 0.5, 0.0]),
    }
    boxes, hits, primitives, mask = producer.region_view_crop_evidence(
        _crop_accepted(),
        view,
        feature_height=4,
        feature_width=4,
        image_height=8,
        image_width=12,
        region_batch_size=2,
    )
    assert boxes.tolist() == [
        [0, 3, 6, 9],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
    ]
    assert hits.tolist() == [3, 0, 0]
    assert primitives.tolist() == [2, 0, 0]
    assert mask.tolist() == [True, False, False]


@pytest.mark.parametrize(
    "weights",
    [
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        torch.tensor([0.0]),
    ],
    ids=["nan", "inf", "zero-positive-hit"],
)
def test_responsibility_view_rejects_nonfinite_or_zero_positive_hits(
    weights: torch.Tensor,
) -> None:
    record = {
        "view_index": 3,
        "frame_index": 7,
        "num_hits": 1,
    }
    payload = {
        "schema": producer.RESPONSIBILITY_VIEW_SCHEMA,
        "schema_version": 1,
        "formula_sha256": _sha("formula"),
        "view_index": 3,
        "frame_index": 7,
        "num_gaussians": 2,
        "num_pixels": 4,
        "gaussian_ids": torch.tensor([0]),
        "pixel_ids": torch.tensor([1]),
        "base_weights": weights,
    }
    with pytest.raises(ValueError, match="responsibility view tensor authority"):
        producer.validate_responsibility_view(
            payload,
            record=record,
            formula_sha256=_sha("formula"),
            num_gaussians=2,
        )


class _ParityRuntime:
    def __init__(self, *, corrupt: bool = False) -> None:
        self.summary_head = nn.Linear(1280, 1536, bias=False)
        torch.manual_seed(7)
        nn.init.normal_(self.summary_head.weight, std=0.01)
        self.corrupt = corrupt

    def encode_training_pair(self, crops: torch.Tensor):
        summary = torch.arange(
            crops.shape[0] * 1280, dtype=torch.float32
        ).reshape(crops.shape[0], 1280)
        descriptor = F.normalize(
            self.summary_head(summary[:, None])[:, 0].float(), dim=-1, eps=1e-8
        )
        if self.corrupt:
            descriptor = descriptor.roll(1, dims=-1)
        return torch.empty(0), summary, descriptor


def test_summary_head_parity_proves_same_accepted_v2_projection() -> None:
    crops = torch.zeros(2, 3, 8, 8)
    descriptor = producer.encode_region_crops_with_summary_head_parity(
        _ParityRuntime(), crops
    )
    assert descriptor.shape == (2, 1536)
    assert torch.allclose(torch.linalg.vector_norm(descriptor, dim=-1), torch.ones(2))
    with pytest.raises(RuntimeError, match="differs from AcceptedV2 head"):
        producer.encode_region_crops_with_summary_head_parity(
            _ParityRuntime(corrupt=True), crops
        )


def test_topk_uses_sealed_responsibility_view_index_as_final_tie_break() -> None:
    visible = torch.ones(1, 5, dtype=torch.bool)
    primitives = torch.ones(1, 5, dtype=torch.long)
    hits = torch.ones(1, 5, dtype=torch.long)
    boxes = torch.tensor(
        [[[0, index, 1, index + 1] for index in range(5)]], dtype=torch.long
    )
    result = producer.select_topk_region_views(
        visible,
        primitives,
        hits,
        boxes,
        [90, 70, 50, 30, 10],
    )
    assert result["pair_view_indices"].tolist() == [4, 3, 2, 1]


def _teacher_input(source_content_sha: str) -> dict:
    return {
        "source_rgb_scene_authority_file_sha256": _sha("source-file"),
        "source_rgb_scene_authority_content_sha256": source_content_sha,
        "factorized_primitive_state_file_sha256": _sha("state"),
        "accepted_region_authority_file_sha256": _sha("accepted-file"),
        "accepted_region_channel_sha256": _sha("accepted-channel"),
        "accepted_region_fingerprints_sha256": _sha("accepted-fingerprints"),
        "exact_marginal_responsibility_authority_file_sha256": _sha(
            "responsibility-file"
        ),
        "official_radio_checkpoint_file_sha256": (
            shard.OFFICIAL_RADIO_CHECKPOINT_SHA256
        ),
        "descriptor_definition": shard.official_teacher_descriptor_definition(),
    }


def test_teacher_payload_requires_crop_and_exact_source_lineage() -> None:
    source_content = _sha("source-content")
    fingerprints = [_sha("region-0"), _sha("region-1")]
    views = [
        {
            "frame_id": "000000",
            "source_relative_path": "000000.png",
            "source_image_sha256": _sha("rgb"),
            "field_frame_authority_sha256": _sha("field-frame"),
            "source_image_height": 8,
            "source_image_width": 12,
            "feature_grid_height": 4,
            "feature_grid_width": 4,
            "responsibility_view_index": 0,
            "responsibility_view_file_sha256": _sha("responsibility-view"),
        }
    ]
    descriptors = torch.zeros(2, 1536)
    descriptors[0, 0] = descriptors[1, 1] = 1
    accepted_audit = {
        "sampling_contract_sha256": shard.SAMPLING_CONTRACT_SHA256,
        "canonical_candidate_region_count": 8,
        "exact_overlap_candidate_count": 8,
        "teacher_visible_candidate_count": 2,
        "selected_region_count": 2,
        "selected_count_by_scale": [2],
    }
    teacher = producer.build_teacher_payload(
        scene_id="scene0001_00",
        source_rgb_scene_authority_sha256=source_content,
        canonical_region_indices=torch.tensor([2, 7]),
        region_fingerprints=fingerprints,
        view_records=views,
        pair_region_indices=torch.tensor([0, 1]),
        pair_view_indices=torch.tensor([0, 0]),
        pair_descriptors=descriptors,
        pair_crop_boxes_tlbr=torch.tensor([[0, 3, 6, 9], [1, 2, 7, 10]]),
        pair_support_hit_counts=torch.tensor([3, 2]),
        pair_visible_primitive_counts=torch.tensor([2, 1]),
        selection_audit={
            "accepted_selection_audit": accepted_audit,
            "pair_count": 2,
            "maximum_views_per_region": 1,
        },
        input_authority=_teacher_input(source_content),
    )
    assert teacher["pair_descriptors"].shape == (2, 1536)

    impostor = copy.deepcopy(teacher)
    impostor.pop("pair_crop_boxes_tlbr")
    impostor.pop("pair_support_hit_counts")
    with pytest.raises(ValueError, match="fields differ"):
        shard.validate_teacher_observation_authority(impostor)

    wrong_source = copy.deepcopy(teacher)
    wrong_source["source_rgb_scene_authority_sha256"] = _sha("other-source")
    with pytest.raises(ValueError, match="source RGB content authority differs"):
        shard.validate_teacher_observation_authority(wrong_source)

    duplicate = copy.deepcopy(teacher)
    for key in (
        "pair_region_indices",
        "pair_view_indices",
        "pair_descriptors",
        "pair_crop_boxes_tlbr",
        "pair_support_hit_counts",
        "pair_visible_primitive_counts",
    ):
        duplicate[key] = torch.cat((duplicate[key][:1], duplicate[key]))
    duplicate["selection_audit"]["pair_count"] = 3
    duplicate["selection_audit"]["maximum_views_per_region"] = 2
    duplicate["channel_sha256"] = shard.teacher_observation_channel_sha256(
        duplicate
    )
    with pytest.raises(ValueError, match="repeats a sparse region-view pair"):
        shard.validate_teacher_observation_authority(duplicate)


def test_teacher_rejects_more_than_four_sparse_views_per_region() -> None:
    source_content = _sha("source-content")
    views = [
        {
            "frame_id": f"{index:06d}",
            "source_relative_path": f"{index:06d}.png",
            "source_image_sha256": _sha(f"rgb-{index}"),
            "field_frame_authority_sha256": _sha(f"field-{index}"),
            "source_image_height": 8,
            "source_image_width": 12,
            "feature_grid_height": 4,
            "feature_grid_width": 4,
            "responsibility_view_index": index,
            "responsibility_view_file_sha256": _sha(f"view-{index}"),
        }
        for index in range(5)
    ]
    descriptors = torch.zeros(6, 1536)
    descriptors[:, 0] = 1
    with pytest.raises(ValueError, match="sparse row coverage"):
        producer.build_teacher_payload(
            scene_id="scene0001_00",
            source_rgb_scene_authority_sha256=source_content,
            canonical_region_indices=torch.tensor([2, 7]),
            region_fingerprints=[_sha("region-0"), _sha("region-1")],
            view_records=views,
            pair_region_indices=torch.tensor([0, 0, 0, 0, 0, 1]),
            pair_view_indices=torch.tensor([0, 1, 2, 3, 4, 0]),
            pair_descriptors=descriptors,
            pair_crop_boxes_tlbr=torch.tensor([[0, 0, 2, 2]] * 6),
            pair_support_hit_counts=torch.ones(6, dtype=torch.long),
            pair_visible_primitive_counts=torch.ones(6, dtype=torch.long),
            selection_audit={
                "accepted_selection_audit": {
                    "sampling_contract_sha256": shard.SAMPLING_CONTRACT_SHA256,
                    "canonical_candidate_region_count": 8,
                    "exact_overlap_candidate_count": 8,
                    "teacher_visible_candidate_count": 2,
                    "selected_region_count": 2,
                    "selected_count_by_scale": [2],
                },
                "pair_count": 6,
                "maximum_views_per_region": 5,
            },
            input_authority=_teacher_input(source_content),
        )


def test_teacher_noclobber_precedes_preflight_and_model_load(tmp_path: Path) -> None:
    output = tmp_path / "teacher.pt"
    output.write_bytes(b"occupied")
    args = Namespace(output=str(output), preflight_only=False)
    with pytest.raises(FileExistsError, match="refuses to clobber"):
        producer.materialize(args)
