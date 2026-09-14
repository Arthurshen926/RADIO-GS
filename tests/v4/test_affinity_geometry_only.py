import torch

from radio_gs.v4.object_memory.learned_fragment_affinity import (
    FragmentAffinityMLP, PAIR_FEATURE_LAYOUT, checkpoint_model,
)


def test_geometry_only_checkpoint_ignores_all_appearance_channels():
    model = FragmentAffinityMLP(16, "geometry_only").eval()
    first = torch.rand(20, 8)
    second = first.clone()
    second[:, [1, 4, 5, 7]] = torch.rand(20, 4)
    restored = checkpoint_model({
        "schema": "radio_gs.surface_object_memory_v4.fragment_affinity_checkpoint.v1",
        "pair_feature_layout": list(PAIR_FEATURE_LAYOUT),
        "hidden_dimension": 16, "feature_mode": "geometry_only",
        "model_state_dict": model.state_dict(),
    })
    torch.testing.assert_close(model(first), restored(second), rtol=0, atol=0)
