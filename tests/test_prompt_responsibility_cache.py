from dataclasses import replace

import pytest
import torch

from radio_gs.interfaces.prompt_responsibility_cache import (
    COMPOSITOR_CONTRACT,
    PromptResponsibilityAuthority,
    build_prompt_responsibility_cache,
    load_prompt_responsibility_cache,
    save_prompt_responsibility_cache,
    sha256_file,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _authority() -> PromptResponsibilityAuthority:
    return PromptResponsibilityAuthority(
        scene_id="fern",
        frame_id="image001",
        camera_name="IMG_4038",
        colmap_camera_name="IMG_4038",
        geometry_checkpoint_sha256=SHA_A,
        geometry_xyz_sha256=SHA_B,
        pose_sha256=SHA_C,
        intrinsics_sha256=SHA_D,
        height=1,
        width=2,
        num_gaussians=3,
        alpha_threshold=0.0,
        compositor_contract=COMPOSITOR_CONTRACT,
        target_rgb_opened=False,
        target_mask_opened=False,
        source_sha256={"manifest": SHA_A, "config": SHA_B},
    )


def _cache():
    return build_prompt_responsibility_cache(
        authority=_authority(),
        gaussian_ids=torch.tensor([0, 1, 1, 2]),
        pixel_ids=torch.tensor([0, 0, 1, 1]),
        weights=torch.tensor([0.6, 0.3, 0.2, 0.7]),
    )


def _load_payload(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def test_exact_adjoint_forward_cycle_algebra(tmp_path):
    cache = _cache()
    assert cache.visible_mass.dtype == torch.float64
    assert torch.allclose(
        cache.visible_mass, torch.tensor([0.6, 0.5, 0.7], dtype=torch.float64)
    )

    adjoint = cache.adjoint(torch.tensor([[1.0, 0.25]]))
    assert torch.allclose(
        adjoint.weighted_sum, torch.tensor([0.6, 0.35, 0.175], dtype=torch.float64)
    )
    assert torch.allclose(
        adjoint.primitive_probability, torch.tensor([1.0, 0.7, 0.25], dtype=torch.float64)
    )

    forward = cache.forward(adjoint.primitive_probability)
    assert torch.allclose(
        forward.weighted_sum, torch.tensor([[0.81, 0.315]], dtype=torch.float64)
    )
    assert torch.allclose(
        forward.pixel_mass, torch.tensor([[0.9, 0.9]], dtype=torch.float64)
    )
    assert torch.allclose(
        forward.normalized_probability, torch.tensor([[0.9, 0.35]], dtype=torch.float64)
    )
    assert torch.equal(cache.cycle(torch.tensor([1.0, 0.25])).supported, forward.supported)

    artifact = save_prompt_responsibility_cache(cache, tmp_path / "cache.pt")
    restored = load_prompt_responsibility_cache(
        artifact.path,
        expected_authority=_authority(),
        expected_file_sha256=artifact.file_sha256,
    )
    assert restored.tensor_bundle_sha256 == artifact.tensor_bundle_sha256


def test_loader_rejects_wrong_geometry_authority(tmp_path):
    artifact = save_prompt_responsibility_cache(_cache(), tmp_path / "cache.pt")
    wrong = replace(_authority(), geometry_xyz_sha256="e" * 64)
    with pytest.raises(ValueError, match="authority differs"):
        load_prompt_responsibility_cache(artifact.path, expected_authority=wrong)


@pytest.mark.parametrize("tamper", ["weight", "visible_mass", "digest", "shape", "dtype"])
def test_loader_rejects_tensor_or_metadata_tampering(tmp_path, tamper):
    cache = _cache()
    path = tmp_path / "cache.pt"
    save_prompt_responsibility_cache(cache, path)
    payload = _load_payload(path)
    if tamper == "weight":
        payload["tensors"]["weights"][0] += 0.01
    elif tamper == "visible_mass":
        payload["tensors"]["visible_mass"][0] += 0.01
    elif tamper == "digest":
        payload["tensor_sha256"]["weights"] = "f" * 64
    elif tamper == "shape":
        payload["tensors"]["visible_mass"] = payload["tensors"]["visible_mass"][:2]
    elif tamper == "dtype":
        payload["tensors"]["pixel_ids"] = payload["tensors"]["pixel_ids"].to(torch.int32)
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_prompt_responsibility_cache(path, expected_authority=_authority())


def test_builder_rejects_front_to_back_mass_above_one():
    with pytest.raises(ValueError, match="mass exceeds one"):
        build_prompt_responsibility_cache(
            authority=_authority(),
            gaussian_ids=torch.tensor([0, 1]),
            pixel_ids=torch.tensor([0, 0]),
            weights=torch.tensor([0.8, 0.4]),
        )


def test_builder_rejects_duplicate_pairs_and_zero_weights():
    with pytest.raises(ValueError, match="duplicate"):
        build_prompt_responsibility_cache(
            authority=_authority(),
            gaussian_ids=torch.tensor([0, 0]),
            pixel_ids=torch.tensor([0, 0]),
            weights=torch.tensor([0.4, 0.4]),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        build_prompt_responsibility_cache(
            authority=_authority(),
            gaussian_ids=torch.tensor([0, 1]),
            pixel_ids=torch.tensor([0, 1]),
            weights=torch.tensor([0.4, 0.0]),
        )


def test_loader_rejects_whole_file_digest_mismatch(tmp_path):
    path = tmp_path / "cache.pt"
    save_prompt_responsibility_cache(_cache(), path)
    assert sha256_file(path) != "0" * 64
    with pytest.raises(ValueError, match="file SHA-256 differs"):
        load_prompt_responsibility_cache(
            path, expected_authority=_authority(), expected_file_sha256="0" * 64
        )
