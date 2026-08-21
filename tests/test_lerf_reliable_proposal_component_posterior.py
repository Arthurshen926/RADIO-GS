from __future__ import annotations

import torch

from radio_gs.querying.latent_proposal_posterior import DIFFERENT_RELATION, SAME_RELATION
from radio_gs.scripts.build_lerf_reliable_proposal_component_posterior import select_reliable_components


def test_known_different_competition_selects_consistent_same_component() -> None:
    authority = {
        "proposal_valid": torch.ones(3, 1, dtype=torch.bool),
        "descriptor_score": torch.tensor([[0.8], [0.7], [0.9]]),
        "field_tail": torch.tensor([[0.8], [0.7], [0.4]]),
        "proposal_probability": torch.tensor([[0.4], [0.3], [0.3]]),
        "proposal_view_indices": torch.tensor([0, 1, 2]),
        "edge_left": torch.tensor([0, 0]), "edge_right": torch.tensor([1, 2]),
        "edge_relation": torch.tensor([SAME_RELATION, DIFFERENT_RELATION], dtype=torch.int8),
    }
    selected = select_reliable_components(authority)
    assert selected[:, 0].tolist() == [True, True, False]
