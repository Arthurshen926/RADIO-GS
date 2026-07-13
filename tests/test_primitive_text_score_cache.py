import torch

from radio_gs.scripts.build_primitive_text_score_cache import compile_scores


def test_compile_scores_respects_invalid_rows_and_query_peaks():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    text = torch.eye(2)
    valid = torch.tensor([True, True, False])

    scores = compile_scores(
        features,
        text,
        valid,
        temperature=10.0,
        chunk_size=1,
        peak_normalize=True,
    ).float()

    assert torch.allclose(scores[:2].amax(dim=0), torch.ones(2))
    assert torch.equal(scores[2], torch.zeros(2))
