from __future__ import annotations

import hashlib

import pytest
import torch

from radio_gs.scripts.eval_lerf_support_readout_oracle_d0_d5 import (
    _d4_score_variants,
    _require_file,
)


def test_d4_reports_all_fixed_scales_and_target_independent_max() -> None:
    scores = torch.tensor(
        [
            [[0.1, 0.8], [0.6, 0.2], [0.4, 0.7]],
            [[0.9, 0.9], [0.8, 0.8], [0.7, 0.7]],
            [[0.3, 0.1], [0.2, 0.5], [0.7, 0.4]],
        ],
        dtype=torch.float32,
    )
    variants = _d4_score_variants(
        scores,
        torch.tensor([True, False, True]),
    )
    assert set(variants) == {
        "D4_s0",
        "D4_s1",
        "D4_s2",
        "D4_per_primitive_max",
    }
    assert torch.equal(variants["D4_s0"][0], scores[0, 0])
    assert torch.equal(variants["D4_s1"][0], scores[0, 1])
    assert torch.equal(variants["D4_s2"][0], scores[0, 2])
    assert torch.equal(
        variants["D4_per_primitive_max"],
        torch.tensor([[0.6, 0.8], [0.0, 0.0], [0.7, 0.5]]),
    )
    assert all(torch.count_nonzero(value[1]) == 0 for value in variants.values())


def test_d4_rejects_missing_scale_or_availability_axes() -> None:
    with pytest.raises(ValueError, match="three registered scale"):
        _d4_score_variants(torch.ones(2, 2, 1), torch.ones(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="availability axis"):
        _d4_score_variants(torch.ones(2, 3, 1), torch.ones(3, dtype=torch.bool))


def test_registered_result_binding_requires_exact_file_sha256(tmp_path) -> None:
    result = tmp_path / "result.json"
    payload = b'{"metric": 0.5}\n'
    result.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    assert _require_file(result, expected, "registered result") == result.resolve()
    with pytest.raises(ValueError, match="different SHA-256"):
        _require_file(result, "0" * 64, "registered result")
