import numpy as np
from PIL import Image
import pytest
import torch

from radio_gs.scripts.build_lerf_query_free_sam3_exact_mpr_memberships import (
    EXPECTED_P0_GENERATION,
    lift_binary_masks_with_exact_mpr,
    validate_automatic_mask_contract,
    validate_automatic_mask_payload,
    validate_responsibility_authority,
)
from radio_gs.rendering.sparse_marginal_authority import (
    SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
    sparse_exact_marginal_formula_contract,
)
from radio_gs.scripts.build_sam3_automatic_mask_cache import pack_masks
from radio_gs.utils.immutable_artifacts import canonical_json_sha256, sha256_file


def _payload(image, *, source_sha, checkpoint="a" * 64, grid=12):
    return {
        "metadata": {
            **EXPECTED_P0_GENERATION,
            "image": str(image.resolve()),
            "source_image_sha256": source_sha,
            "checkpoint_sha256": checkpoint,
            "grid_size": grid,
        }
    }


def _automatic_payload(mask: np.ndarray, *, grid: int = 2) -> dict:
    masks = np.asarray(mask, dtype=bool)[None]
    y, x = np.where(masks[0])
    box = [[int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]]
    return {
        "packed_masks": pack_masks(masks),
        "mask_shape": list(masks.shape[1:]),
        "scores": np.asarray([0.9], dtype=np.float32),
        "stability": np.asarray([1.0], dtype=np.float32),
        "seed_xy": np.asarray([[masks.shape[2] * 0.25, masks.shape[1] * 0.25]], dtype=np.float32),
        "prompt_index": np.asarray([0], dtype=np.int32),
        "candidate_index": np.asarray([1], dtype=np.int8),
        "boxes_xyxy": np.asarray(box, dtype=np.int32),
        "proposal_area_fraction": np.asarray([masks.mean()], dtype=np.float32),
        "proposal_count_before_deduplication": 1,
        "decoder_logits_available": True,
        "metadata": {},
    }


def test_exact_mpr_p0_requires_manifest_source_sha_and_uniform_grid(tmp_path) -> None:
    image = tmp_path / "frame_00001.png"
    mask = tmp_path / "frame_00001.pt"
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(image)
    mask.touch()
    digest = sha256_file(image)
    manifest = {
        "image": str(image.resolve()),
        "output": str(mask.resolve()),
        "source_image_sha256": digest,
        "output_sha256": sha256_file(mask),
    }
    payload = _payload(image, source_sha=digest)
    assert validate_automatic_mask_contract(
        payload,
        mask_path=mask,
        manifest_record=manifest,
        source_image=image,
        expected_checkpoint_sha256="a" * 64,
        expected_grid_size=12,
    ) == digest

    without_sha = _payload(image, source_sha="")
    with pytest.raises(ValueError, match="source RGB byte binding differ"):
        validate_automatic_mask_contract(
            without_sha,
            mask_path=mask,
            manifest_record=manifest,
            source_image=image,
            expected_checkpoint_sha256="a" * 64,
            expected_grid_size=12,
        )
    mixed_grid = _payload(image, source_sha=digest, grid=4)
    with pytest.raises(ValueError, match="mix producer, checkpoint, grid"):
        validate_automatic_mask_contract(
            mixed_grid,
            mask_path=mask,
            manifest_record=manifest,
            source_image=image,
            expected_checkpoint_sha256="a" * 64,
            expected_grid_size=12,
        )


def test_packed_mask_payload_rejects_padding_unknown_and_area_corruption() -> None:
    mask = np.zeros((4, 5), dtype=bool)
    mask[:2, :2] = True
    payload = _automatic_payload(mask)
    validated = validate_automatic_mask_payload(
        payload, image_height=4, image_width=5, expected_grid_size=2
    )
    assert validated["masks"].shape == (1, 4, 5)

    unknown = {**payload, "unknown": 1}
    with pytest.raises(ValueError, match="payload fields differ"):
        validate_automatic_mask_payload(
            unknown, image_height=4, image_width=5, expected_grid_size=2
        )

    bad_padding = {**payload, "packed_masks": payload["packed_masks"].clone()}
    bad_padding["packed_masks"][0, 0, -1] |= 0b10000000
    with pytest.raises(ValueError, match="padding bits"):
        validate_automatic_mask_payload(
            bad_padding, image_height=4, image_width=5, expected_grid_size=2
        )

    bad_area = {**payload, "proposal_area_fraction": np.asarray([0.9], dtype=np.float32)}
    with pytest.raises(ValueError, match="areas differ"):
        validate_automatic_mask_payload(
            bad_area, image_height=4, image_width=5, expected_grid_size=2
        )


def test_exact_mpr_lift_rejects_zero_threshold_and_invalid_base_weights() -> None:
    masks = np.ones((1, 2, 2), dtype=bool)
    common = dict(
        masks=masks,
        gaussian_ids=np.asarray([0]),
        pixel_ids=np.asarray([0]),
        num_gaussians=1,
        feature_height=2,
        feature_width=2,
    )
    with pytest.raises(ValueError, match=r"\(0,1\]"):
        lift_binary_masks_with_exact_mpr(
            **common, base_weights=np.asarray([0.5]), min_membership=0.0
        )
    with pytest.raises(ValueError, match=r"base weights.*\(0,1\]"):
        lift_binary_masks_with_exact_mpr(
            **common, base_weights=np.asarray([1.1]), min_membership=0.5
        )


def test_responsibility_authority_rejects_self_consistent_invented_formula(tmp_path) -> None:
    authority_path = tmp_path / "authority.json"
    shard_dir = tmp_path / "authority.json.views"
    shard_dir.mkdir()
    shard = shard_dir / "view_00000.pt"
    torch.save({
        "schema": "radio_gs.sparse_exact_marginal_responsibility_view.v1",
        "schema_version": 1,
        "formula_sha256": SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
        "view_index": 0,
        "frame_index": 1,
        "num_gaussians": 1,
        "num_pixels": 1,
        "gaussian_ids": torch.tensor([0]),
        "pixel_ids": torch.tensor([0]),
        "base_weights": torch.tensor([0.5]),
    }, shard)
    metadata = {
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "formula_sha256": SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
        "xyz_sha256": "x" * 64,
        "selected_frame_indices": [1],
        "feature_height": 1,
        "feature_width": 1,
    }
    authority = {
        "schema": "radio_gs.sparse_exact_marginal_responsibility_authority.v1",
        "schema_version": 1,
        "formula_contract": sparse_exact_marginal_formula_contract(),
        "formula_sha256": SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
        "frame_indices": [1],
        "metadata": metadata,
        "num_gaussians": 1,
        "num_pixels": 1,
        "total_hits": 1,
        "views": [{
            "frame_index": 1,
            "num_hits": 1,
            "relative_path": "authority.json.views/view_00000.pt",
            "sha256": sha256_file(shard),
            "view_index": 0,
        }],
    }
    validate_responsibility_authority(
        authority, authority_path=authority_path, num_gaussians=1, xyz_sha256="x" * 64
    )
    invented = {**sparse_exact_marginal_formula_contract(), "target_weight": "invented"}
    invented_sha = canonical_json_sha256(invented)
    authority["formula_contract"] = invented
    authority["formula_sha256"] = invented_sha
    authority["metadata"] = {**metadata, "formula_sha256": invented_sha}
    with pytest.raises(ValueError, match="authority differs"):
        validate_responsibility_authority(
            authority,
            authority_path=authority_path,
            num_gaussians=1,
            xyz_sha256="x" * 64,
        )
