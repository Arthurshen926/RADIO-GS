import torch

from radio_gs.field.basis_decoder import AffineBasisDecoder
from radio_gs.field.canonical_gaussian_field import CanonicalGaussianField
from radio_gs.field.field_signature import FeatureSpaceSignature


def test_primitive_confidence_requires_all_reliability_channels():
    decoder = AffineBasisDecoder(feature_dim=2, coefficient_dim=2, trainable_basis=False)
    signature = FeatureSpaceSignature(
        radio_version="test", radio_checkpoint_sha256="a", raw_feature_dim=2,
        normalization="none",
    )
    field = CanonicalGaussianField(
        2, decoder, signature, reliability=torch.tensor([[1.0, 0.25], [1.0, 0.0]]),
        use_fusion=False,
    )
    torch.testing.assert_close(field.primitive_confidence(), torch.tensor([0.5, 0.0]))
