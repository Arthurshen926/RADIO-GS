import torch

from radio_gs.querying.typed_posteriors import CategoricalPosteriorV2
from radio_gs.scripts.screen_scannet_categorical_posterior_v2_cpu import variant_state


def test_zero_shrinkage_is_exact_primitive_identity() -> None:
    model = CategoricalPosteriorV2(num_classes=3)
    trained = {key: torch.randn_like(value) for key, value in model.state_dict().items()}
    state = variant_state(trained, alpha=0.0, background=True)
    model.load_state_dict(state)
    semantic = torch.tensor([[0.1, 0.4, 0.2], [0.8, 0.3, 0.1]])
    output = model(
        semantic, reliability=torch.zeros(2, 5), valid=torch.ones(2, dtype=torch.bool)
    )
    assert torch.equal(output.prediction, semantic.argmax(dim=-1))


def test_class_only_variant_disables_background_without_changing_class_params() -> None:
    model = CategoricalPosteriorV2(num_classes=3)
    trained = {key: value.clone() for key, value in model.state_dict().items()}
    trained["class_bias"] = torch.tensor([1.0, -2.0, 3.0])
    state = variant_state(trained, alpha=0.5, background=False)
    assert torch.equal(state["class_bias"], torch.tensor([0.5, -1.0, 1.5]))
    assert float(state["background_bias"]) == -80.0
    assert torch.count_nonzero(state["background_reliability.weight"]) == 0
    assert torch.count_nonzero(state["background_ambiguity.weight"]) == 0
