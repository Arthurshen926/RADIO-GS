import json

import torch

from radio_gs.scripts.materialize_lerf_teacher_view_siglip_authority import (
    _canonicalize_view_axis,
    _load_responsibility_view,
    _update_top_views,
    _validate_responsibility_authority,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256, sha256_file


def _descriptor(value: float) -> torch.Tensor:
    return torch.full((1, 1536), value, dtype=torch.float32)


def _synthetic_authority(frames: list[int]) -> dict:
    formula = {"query_independent": True, "feature_independent": True}
    views = [
        {
            "frame_index": frame,
            "view_index": index,
            "num_hits": 1,
            "relative_path": f"authority.json.views/view_{index:05d}.pt",
            "sha256": "a" * 64,
        }
        for index, frame in enumerate(frames)
    ]
    return {
        "schema": "radio_gs.sparse_exact_marginal_responsibility_authority.v1",
        "schema_version": 1,
        "formula_contract": formula,
        "formula_sha256": canonical_json_sha256(formula),
        "frame_indices": frames,
        "num_gaussians": 3,
        "num_pixels": 2,
        "total_hits": len(frames),
        "views": views,
        "metadata": {
            "assignment_mode": "exact_front_to_back_sparse_marginal",
            "registration_weight_mode": (
                "exact_front_to_back_marginal_responsibility"
            ),
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "query_independent": True,
            "xyz_sha256": "b" * 64,
            "excluded_frame_ids": [41, 105, 152, 195],
            "selected_frame_indices": frames,
        },
    }


def test_top4_teacher_views_use_mass_then_frame_id() -> None:
    descriptors = torch.zeros(1, 4, 1536, dtype=torch.float16)
    mass = torch.zeros(1, 4)
    frames = torch.full((1, 4), -1, dtype=torch.int32)
    for value, weight, frame in (
        (1.0, 1.0, 9),
        (2.0, 2.0, 8),
        (3.0, 3.0, 7),
        (4.0, 4.0, 6),
        (5.0, 1.0, 5),
        (6.0, 0.5, 4),
    ):
        _update_top_views(
            top_descriptors=descriptors,
            top_mass=mass,
            top_frame_ids=frames,
            rows=torch.tensor([0]),
            descriptors=_descriptor(value),
            mass=torch.tensor([weight]),
            frame_id=frame,
        )
    descriptors, mass, frames = _canonicalize_view_axis(
        descriptors, mass, frames
    )
    assert frames.tolist() == [[6, 7, 8, 5]]
    assert mass.tolist() == [[4.0, 3.0, 2.0, 1.0]]
    assert descriptors[0, :, 0].tolist() == [4.0, 3.0, 2.0, 5.0]


def test_canonical_view_axis_zeros_invalid_padding() -> None:
    descriptors = torch.ones(2, 4, 1536, dtype=torch.float16)
    mass = torch.tensor([[0.2, 0.4, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    frames = torch.tensor([[9, 3, -1, -1], [-1, -1, -1, -1]], dtype=torch.int32)
    descriptors, mass, frames = _canonicalize_view_axis(
        descriptors, mass, frames
    )
    assert frames[0].tolist() == [3, 9, -1, -1]
    assert torch.allclose(mass[0], torch.tensor([0.4, 0.2, 0.0, 0.0]))
    assert not bool(descriptors[frames < 0].any())


def test_exact_marginal_view_reconstructs_target_weights(tmp_path) -> None:
    views_dir = tmp_path / "authority.json.views"
    views_dir.mkdir()
    view_path = views_dir / "view_00000.pt"
    formula_sha256 = "a" * 64
    torch.save(
        {
            "schema": "radio_gs.sparse_exact_marginal_responsibility_view.v1",
            "schema_version": 1,
            "formula_sha256": formula_sha256,
            "view_index": 0,
            "frame_index": 1,
            "num_gaussians": 3,
            "num_pixels": 2,
            "gaussian_ids": torch.tensor([0, 1, 2]),
            "pixel_ids": torch.tensor([0, 0, 1]),
            "base_weights": torch.tensor([0.2, 0.3, 0.4]),
        },
        view_path,
    )
    authority_path = tmp_path / "authority.json"
    authority_path.write_text("{}", encoding="utf-8")
    authority = {
        "formula_sha256": formula_sha256,
        "num_gaussians": 3,
        "num_pixels": 2,
    }
    record = {
        "relative_path": "authority.json.views/view_00000.pt",
        "sha256": sha256_file(view_path),
        "num_hits": 3,
        "view_index": 0,
        "frame_index": 1,
    }
    gaussian, pixels, marginal = _load_responsibility_view(
        authority, authority_path, record
    )
    assert gaussian.tolist() == [0, 1, 2]
    assert pixels.tolist() == [0, 0, 1]
    assert torch.allclose(marginal, torch.tensor([0.08, 0.18, 0.4]))


def test_responsibility_authority_rejects_train16_source(tmp_path) -> None:
    frames = list(range(1, 17))
    payload = _synthetic_authority(frames)
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        _validate_responsibility_authority(
            authority_path,
            sha256_file(authority_path),
            expected_xyz_sha256="b" * 64,
        )
    except ValueError as exc:
        assert "120-view" in str(exc)
    else:
        raise AssertionError("train16 authority unexpectedly accepted")


def test_responsibility_authority_accepts_exact_120_view_contract(tmp_path) -> None:
    frames = [value for value in range(1, 125) if value not in {41, 105, 121, 122}]
    assert len(frames) == 120
    payload = _synthetic_authority(frames)
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded, loaded_path = _validate_responsibility_authority(
        authority_path,
        sha256_file(authority_path),
        expected_xyz_sha256="b" * 64,
    )
    assert loaded["frame_indices"] == frames
    assert loaded_path == authority_path.resolve()


def test_responsibility_authority_allows_declared_empty_source_view(tmp_path) -> None:
    frames = [value for value in range(1, 125) if value not in {41, 105, 121, 122}]
    payload = _synthetic_authority(frames)
    payload["views"][64]["num_hits"] = 0
    payload["total_hits"] -= 1
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded, _ = _validate_responsibility_authority(
        authority_path,
        sha256_file(authority_path),
        expected_xyz_sha256="b" * 64,
    )
    assert loaded["views"][64]["num_hits"] == 0
