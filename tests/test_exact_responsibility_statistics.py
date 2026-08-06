from __future__ import annotations

import copy

import pytest
import torch

from radio_gs.rendering.exact_responsibility_statistics import (
    EXACT_RESPONSIBILITY_STATISTICS_SCHEMA,
    build_exact_responsibility_statistics,
    validate_exact_responsibility_statistics_payload,
)
from radio_gs.rendering.sparse_marginal_authority import (
    SparseExactMarginalAuthorityWriter,
)
from radio_gs.utils.immutable_artifacts import load_torch_mapping, sha256_file


def _source_metadata() -> dict[str, object]:
    return {
        "schema_version": 1,
        "assignment_mode": "exact_front_to_back_sparse_marginal",
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "query_independent": True,
    }


def _authority(tmp_path, *, metadata=None):
    path = tmp_path / "authority.json"
    writer = SparseExactMarginalAuthorityWriter(
        path,
        metadata=_source_metadata() if metadata is None else metadata,
        frame_indices=[3, 5],
        num_gaussians=3,
        num_pixels=2,
    )
    writer.add_view(
        0,
        torch.tensor([0, 1, 0]),
        torch.tensor([0, 0, 1]),
        torch.tensor([0.8, 0.2, 0.6]),
    )
    writer.add_view(
        1,
        torch.tensor([0, 1]),
        torch.tensor([0, 0]),
        torch.tensor([0.5, 0.5]),
    )
    _path, digest = writer.finalize()
    return path, digest


def test_exact_responsibility_statistics_streams_expected_sufficient_statistics(
    tmp_path,
) -> None:
    authority, authority_sha256 = _authority(tmp_path)
    output = tmp_path / "statistics.pt"
    written, digest = build_exact_responsibility_statistics(
        authority_path=authority,
        expected_authority_sha256=authority_sha256,
        output_path=output,
    )

    assert written == output.resolve()
    assert digest == sha256_file(output)
    payload, _observed, _source = load_torch_mapping(
        output,
        expected_sha256=digest,
        map_location="cpu",
        label="statistics fixture",
    )
    validate_exact_responsibility_statistics_payload(
        payload,
        expected_authority_sha256=authority_sha256,
    )
    assert payload["schema"] == EXACT_RESPONSIBILITY_STATISTICS_SCHEMA
    tensors = payload["tensors"]
    torch.testing.assert_close(
        tensors["visible_mass"], torch.tensor([1.9, 0.7, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        tensors["semantic_mass"],
        torch.tensor([1.49, 0.29, 0.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        tensors["semantic_mass_sq"],
        torch.tensor([0.8321, 0.0641, 0.0], dtype=torch.float64),
    )
    assert tensors["nonzero_hit_count"].tolist() == [3, 2, 0]
    assert tensors["nonzero_view_count"].tolist() == [2, 2, 0]
    torch.testing.assert_close(
        tensors["kish_effective_sample_size"],
        torch.tensor(
            [1.49**2 / 0.8321, 0.29**2 / 0.0641, 0.0], dtype=torch.float64
        ),
    )
    assert payload["metadata"]["contains_admission_decision"] is False
    assert "threshold" not in payload["metadata"]


def test_exact_responsibility_statistics_is_no_clobber(tmp_path) -> None:
    authority, authority_sha256 = _authority(tmp_path)
    output = tmp_path / "statistics.pt"
    build_exact_responsibility_statistics(
        authority_path=authority,
        expected_authority_sha256=authority_sha256,
        output_path=output,
    )
    original = sha256_file(output)
    with pytest.raises(FileExistsError, match="already exists"):
        build_exact_responsibility_statistics(
            authority_path=authority,
            expected_authority_sha256=authority_sha256,
            output_path=output,
        )
    assert sha256_file(output) == original


def test_exact_responsibility_statistics_rejects_authority_sha_or_contamination(
    tmp_path,
) -> None:
    authority, authority_sha256 = _authority(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        build_exact_responsibility_statistics(
            authority_path=authority,
            expected_authority_sha256="f" * 64,
            output_path=tmp_path / "wrong-sha.pt",
        )

    contaminated = {
        **_source_metadata(),
        "benchmark_masks_opened": True,
    }
    other_root = tmp_path / "contaminated"
    other_root.mkdir()
    contaminated_authority, contaminated_sha = _authority(
        other_root, metadata=contaminated
    )
    with pytest.raises(ValueError, match="source-only"):
        build_exact_responsibility_statistics(
            authority_path=contaminated_authority,
            expected_authority_sha256=contaminated_sha,
            output_path=tmp_path / "contaminated.pt",
        )
    assert authority_sha256 == sha256_file(authority)


@pytest.mark.parametrize(
    "tamper,match",
    [
        ("nan", "visible_mass"),
        ("ess", "kish_effective_sample_size|tensor bundle|numeric relations"),
        ("authority", "authority SHA-256"),
        ("decision", "safety metadata"),
    ],
)
def test_exact_responsibility_statistics_validator_fails_closed(
    tmp_path, tamper, match
) -> None:
    authority, authority_sha256 = _authority(tmp_path)
    output = tmp_path / "statistics.pt"
    build_exact_responsibility_statistics(
        authority_path=authority,
        expected_authority_sha256=authority_sha256,
        output_path=output,
    )
    payload, _digest, _source = load_torch_mapping(
        output, map_location="cpu", label="statistics fixture"
    )
    malformed = copy.deepcopy(payload)
    if tamper == "nan":
        malformed["tensors"]["visible_mass"][0] = float("nan")
    elif tamper == "ess":
        malformed["tensors"]["kish_effective_sample_size"][0] += 1.0
    elif tamper == "authority":
        malformed["authority"]["sha256"] = "f" * 64
    else:
        malformed["metadata"]["contains_admission_decision"] = True
    with pytest.raises(ValueError, match=match):
        validate_exact_responsibility_statistics_payload(
            malformed,
            expected_authority_sha256=authority_sha256,
        )
