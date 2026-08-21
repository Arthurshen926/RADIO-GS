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
