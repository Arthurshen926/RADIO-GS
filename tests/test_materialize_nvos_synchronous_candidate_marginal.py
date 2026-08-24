from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from radio_gs.scripts.materialize_nvos_synchronous_candidate_marginal import (
    fuse_one_candidate,
    lift_candidate_views,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _view(tmp_path: Path, rank: int, probability: np.ndarray) -> dict[str, object]:
    probability_path = tmp_path / f"probability_{rank}.npy"
    with probability_path.open("wb") as handle:
        np.save(handle, probability.astype(np.float32), allow_pickle=False)
    assignment_path = tmp_path / f"assignment_{rank}.pt"
    torch.save(
        {
            "gaussian_ids": torch.tensor([0, 1]),
            "pixel_ids": torch.tensor([0, 1]),
            "weights": torch.tensor([1.0, 1.0]),
        },
        assignment_path,
    )
    return {
        "probability": {"path": str(probability_path), "sha256": _sha256(probability_path)},
        "assignment": {"path": str(assignment_path), "sha256": _sha256(assignment_path)},
        "log_precision": 0.0,
        "view_digest": f"{rank + 10:064x}",
    }


def test_streamed_exact_adjoint_and_robust_view_fusion(tmp_path: Path):
    records = [
        _view(tmp_path, 0, np.array([[0.8, 0.2]], dtype=np.float32)),
        _view(tmp_path, 1, np.array([[0.8, 0.2]], dtype=np.float32)),
        _view(tmp_path, 2, np.array([[0.001, 0.999]], dtype=np.float32)),
    ]
    views, precision, digests = lift_candidate_views(records, num_gaussians=2)
    assert views.shape == (3, 2)
    result = fuse_one_candidate(
        views,
        precision,
        candidate_digest=f"{1:064x}",
        view_digests=digests,
    )
    assert result.shape == (2,)
    assert result[0] > 0.65
    assert result[1] < 0.35


def test_invisible_rows_are_neutral(tmp_path: Path):
    probability_path = tmp_path / "probability.npy"
    np.save(probability_path, np.array([[0.9]], dtype=np.float32), allow_pickle=False)
    assignment_path = tmp_path / "assignment.pt"
    torch.save(
        {
            "gaussian_ids": torch.tensor([0]),
            "pixel_ids": torch.tensor([0]),
            "weights": torch.tensor([1.0]),
        },
        assignment_path,
    )
    record = {
        "probability": {"path": str(probability_path), "sha256": _sha256(probability_path)},
        "assignment": {"path": str(assignment_path), "sha256": _sha256(assignment_path)},
        "log_precision": 0.0,
        "view_digest": f"{2:064x}",
    }
    views, _precision, _digests = lift_candidate_views([record], num_gaussians=2)
    assert torch.equal(views[0], torch.tensor([0.9, 0.5]))


def test_assignment_cache_reuses_view_lineage(tmp_path: Path):
    record = _view(tmp_path, 0, np.array([[0.7, 0.3]], dtype=np.float32))
    assignment = torch.load(
        record["assignment"]["path"], map_location="cpu", weights_only=False
    )
    views, _precision, _digests = lift_candidate_views(
        [record],
        num_gaussians=2,
        assignment_cache={str(record["view_digest"]): assignment},
        device="cpu",
    )
    assert torch.allclose(views[0], torch.tensor([0.7, 0.3]))


def test_positive_unknown_fusion_unions_complementary_extent_without_negative_vote():
    views = torch.tensor([[0.9, 0.1, 0.2], [0.1, 0.8, 0.3]])
    result = fuse_one_candidate(
        views,
        torch.zeros(2),
        candidate_digest=f"{3:064x}",
        view_digests=(f"{4:064x}", f"{5:064x}"),
        view_fusion="positive_unknown_noisy_or",
    )
    assert torch.allclose(result, torch.tensor([0.9, 0.8, 0.0]))


def test_positive_unknown_fusion_noisy_or_corroborating_detection():
    views = torch.tensor([[0.8], [0.8]])
    result = fuse_one_candidate(
        views,
        torch.zeros(2),
        candidate_digest=f"{6:064x}",
        view_digests=(f"{7:064x}", f"{8:064x}"),
        view_fusion="positive_unknown_noisy_or",
    )
    assert torch.allclose(result, torch.tensor([0.96]))


def test_positive_unknown_fusion_does_not_promote_neutral_invisible_rows():
    views = torch.tensor([[0.9, 0.5], [0.5, 0.5]])
    result = fuse_one_candidate(
        views,
        torch.zeros(2),
        candidate_digest=f"{9:064x}",
        view_digests=(f"{10:064x}", f"{11:064x}"),
        view_fusion="positive_unknown_noisy_or",
    )
    assert torch.allclose(result, torch.tensor([0.9, 0.0]))
