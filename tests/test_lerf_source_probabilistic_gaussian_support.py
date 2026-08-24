import numpy as np
from scipy import sparse
import torch

from radio_gs.scripts.evaluate_lerf_source_probabilistic_gaussian_support import (
    probabilistic_reciprocal_score,
    sparse_fuzzy_containment,
)


def test_fuzzy_containment_preserves_diffuse_support_below_hard_threshold() -> None:
    incidence = sparse.csr_matrix(np.asarray([
        [0.4, 0.4, 0.0],
        [0.3, 0.4, 0.0],
    ], dtype=np.float32))
    value = sparse_fuzzy_containment(incidence, torch.tensor([[0, 1]]))
    assert torch.allclose(value, torch.tensor([1.0]))


def test_probabilistic_support_still_requires_reciprocal_identity() -> None:
    probability = torch.tensor([
        [1.0, 0.8, 0.9],
        [0.8, 1.0, 0.2],
        [0.9, 0.2, 1.0],
    ])
    views = torch.tensor([0, 1, 1])
    incidence = sparse.csr_matrix(np.asarray([
        [0.4, 0.4], [0.4, 0.4], [0.4, 0.4]
    ], dtype=np.float32))
    score = probabilistic_reciprocal_score(probability, views, incidence)
    assert score[0, 2] == torch.tensor(0.9)
    assert score[0, 1] == 0
