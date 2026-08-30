import torch

from radio_gs.v3.evaluation.explore_text_instance_coupling import _coherent_anchors


def test_coherent_anchor_compiler_rejects_semantically_close_other_instance() -> None:
    identity = torch.tensor([0.9, 0.85, 0.84, 0.1])
    instance = torch.tensor([
        [1.0, 0.0],
        [0.99, 0.01],
        [-1.0, 0.0],
        [0.0, 0.0],
    ])
    instance = torch.nn.functional.normalize(instance, dim=-1)
    rows, weights = _coherent_anchors(
        identity, instance, candidate_k=3, anchor_k=2, affinity_weight=1.0
    )
    assert set(rows.tolist()) == {0, 1}
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
