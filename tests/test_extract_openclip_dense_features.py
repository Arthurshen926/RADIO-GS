from __future__ import annotations

import torch

from radio_gs.scripts import extract_openclip_dense_features as extract


def test_project_patch_tokens_projects_and_normalizes_channels():
    tokens = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 2.0]], [[2.0, 0.0]]]],
        dtype=torch.float32,
    )
    projection = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    projected = extract.project_patch_tokens(tokens, projection, normalize=True)

    assert projected.shape == (1, 2, 1, 2)
    assert torch.allclose(projected.norm(dim=1), torch.ones(1, 1, 2), atol=1e-6)
    assert torch.allclose(projected[:, :, 0, 0], torch.tensor([[3.0, 2.0]]) / 13.0**0.5)


def test_extract_dense_feature_from_visual_uses_requested_intermediate():
    class FakeVisual:
        def __init__(self) -> None:
            self.proj = torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=torch.float32,
            )
            self.calls = []

        def forward_intermediates(self, x, **kwargs):
            self.calls.append(kwargs)
            return [torch.zeros(x.shape[0], 3, 2, 2, dtype=torch.float32, device=x.device)]

    visual = FakeVisual()
    image = torch.zeros(2, 3, 224, 224)

    dense = extract.extract_dense_feature_from_visual(
        visual,
        image,
        intermediate_index=-2,
        normalize=False,
    )

    assert dense.shape == (2, 2, 2, 2)
    assert torch.allclose(dense, torch.zeros_like(dense))
    assert visual.calls == [
        {
            "indices": [-2],
            "normalize_intermediates": True,
            "intermediates_only": True,
            "output_fmt": "NCHW",
        }
    ]


def test_extract_dense_feature_from_visual_supports_maskclip_value_tokens():
    class FakeOutProj(torch.nn.Module):
        def forward(self, x):
            return x + 1.0

    class FakeAttn:
        embed_dim = 2

        def __init__(self) -> None:
            self.in_proj_weight = torch.cat(
                [
                    torch.zeros(2, 2),
                    torch.zeros(2, 2),
                    torch.eye(2),
                ],
                dim=0,
            )
            self.in_proj_bias = torch.zeros(6)
            self.out_proj = FakeOutProj()

    class FakeBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln_1 = torch.nn.Identity()
            self.attn = FakeAttn()

        def forward(self, x):
            return x

    class FakeTransformer:
        def __init__(self) -> None:
            self.resblocks = [FakeBlock()]

    class FakeVisual:
        def __init__(self) -> None:
            self.transformer = FakeTransformer()
            self.ln_post = torch.nn.Identity()
            self.proj = torch.eye(2)
            self.grid_size = (1, 2)

        def _embeds(self, image):
            return torch.tensor(
                [
                    [
                        [0.0, 0.0],
                        [3.0, 4.0],
                        [5.0, 12.0],
                    ]
                ],
                dtype=image.dtype,
                device=image.device,
            )

    dense = extract.extract_dense_feature_from_visual(
        FakeVisual(),
        torch.zeros(1, 3, 224, 224),
        intermediate_index=-1,
        token_mode="maskclip_value",
        normalize=True,
    )

    expected = torch.tensor([[[[4.0, 6.0]], [[5.0, 13.0]]]])
    expected = torch.nn.functional.normalize(expected, dim=1)
    assert dense.shape == (1, 2, 1, 2)
    assert torch.allclose(dense, expected)


def test_discover_source_frames_sorts_feature_ids(tmp_path):
    backbone = tmp_path / "backbone"
    backbone.mkdir()
    torch.save(torch.zeros(1), backbone / "rgb_10.pt")
    torch.save(torch.zeros(1), backbone / "rgb_2.pt")
    torch.save(torch.zeros(1), backbone / "rgb_7.pt")

    assert extract.discover_source_frames(tmp_path) == [2, 7, 10]
    assert extract.discover_source_frames(tmp_path, frame_ids=[7, 2]) == [2, 7]
