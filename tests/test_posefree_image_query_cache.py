import pytest
import torch
from torch import nn
from types import SimpleNamespace

from radio_gs.field import FeatureSpaceSignature
from radio_gs.interfaces import OfficialRadioRuntime
from radio_gs.querying.query_compilers import compile_image_query
from radio_gs.scripts.build_posefree_image_query_cache import parse_bbox


def test_posefree_bbox_is_clipped_and_validated() -> None:
    assert parse_bbox("-2,3,12,9", 10, 8) == (0, 3, 10, 8)
    assert parse_bbox("", 10, 8) == (0, 0, 10, 8)
    with pytest.raises(ValueError, match="empty"):
        parse_bbox("4,4,3,5", 10, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_posefree_image_query_keeps_default_weights_on_token_device() -> None:
    signature = FeatureSpaceSignature(
        radio_version="test",
        radio_checkpoint_sha256="checkpoint",
        raw_feature_dim=4,
        adaptor_name="test",
        adaptor_sha256="adaptor",
        adaptor_output_dim=4,
        token_type="region",
        normalization="l2",
        field_checkpoint_sha256="field",
    )
    query = compile_image_query(
        torch.randn(1, 4, device="cuda"),
        torch.randn(8, 4, device="cuda"),
        semantic_signature=signature,
        appearance_signature=signature,
        semantic_negatives=torch.randn(1, 4, device="cuda"),
        appearance_negatives=torch.randn(1, 4, device="cuda"),
        prototype_count=2,
    )
    assert query.appearance_evidence.features.device.type == "cuda"
    assert query.semantic_evidence.negatives.device.type == "cuda"


class _FakeOfficialModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_shape = None

    def get_nearest_supported_resolution(self, _height: int, _width: int):
        return SimpleNamespace(height=8, width=12)

    def forward(self, images: torch.Tensor, *, feature_fmt: str):
        self.seen_shape = tuple(images.shape)
        assert feature_fmt == "NCHW"
        return {
            "dino_v3_7b": (
                torch.ones(images.shape[0], 4),
                torch.ones(images.shape[0], 4, 2, 3),
            )
        }


def test_official_runtime_resizes_and_unpacks_released_tuple_output() -> None:
    model = _FakeOfficialModel()
    runtime = OfficialRadioRuntime(
        model=model,
        version="test",
        adaptor_names=("dino_v3_7b",),
    )
    summary, spatial = runtime.encode_adaptor_images(
        torch.zeros(1, 3, 5, 7), "dino_v3_7b"
    )
    assert model.seen_shape == (1, 3, 8, 12)
    assert summary.shape == (1, 4)
    assert spatial.shape == (1, 4, 2, 3)
