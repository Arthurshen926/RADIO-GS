import torch

from radio_gs.field import (
    AffineBasisDecoder,
    CanonicalGaussianField,
    DirectionalCanonicalField,
    FeatureSpaceSignature,
)


def _base_field() -> CanonicalGaussianField:
    decoder = AffineBasisDecoder(
        feature_dim=2,
        coefficient_dim=2,
        basis=torch.eye(2),
        trainable_basis=False,
    )
    field = CanonicalGaussianField(
        3,
        decoder,
        FeatureSpaceSignature(
            radio_version="test", radio_checkpoint_sha256="test", raw_feature_dim=2
        ),
        local_dim=2,
        use_fusion=False,
    )
    with torch.no_grad():
        field.local_codes.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
    return field


def test_directional_field_uses_modes_only_on_supported_rows():
    base = _base_field()
    directional = DirectionalCanonicalField(
        base,
        AffineBasisDecoder(
            feature_dim=2,
            coefficient_dim=2,
            basis=torch.eye(2),
            trainable_basis=False,
        ),
        global_rows=torch.tensor([1]),
        prototype_coefficients=torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]]),
        mixture_weight=torch.tensor([[0.6, 0.4]]),
    )

    modes, weights = directional.radio_prototypes()

    torch.testing.assert_close(modes[0, 0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(modes[0, 1], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(modes[1], torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
    torch.testing.assert_close(weights[0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(weights[1], torch.tensor([0.6, 0.4]))


def test_directional_query_pooling_preserves_minority_mode():
    directional = DirectionalCanonicalField(
        _base_field(),
        AffineBasisDecoder(
            feature_dim=2,
            coefficient_dim=2,
            basis=torch.eye(2),
            trainable_basis=False,
        ),
        global_rows=torch.tensor([1]),
        prototype_coefficients=torch.tensor([[[1.0, 0.0], [0.0, -1.0]]]),
        mixture_weight=torch.tensor([[0.8, 0.2]]),
    )

    logits = directional.prototype_query_logits(
        torch.tensor([[0.0, -1.0]]), indices=torch.tensor([1])
    )

    torch.testing.assert_close(logits, torch.ones(1, 1))
