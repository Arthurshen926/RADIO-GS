import torch

from radio_gs.v4.evaluation.sam_object_evidence_audit import _proposal_association


def test_partial_association_keeps_parts_and_rejects_merged_masks():
    oracle = torch.zeros(2, 4, 2)
    oracle[:, :2, 0] = 1
    oracle[:, 2:, 1] = 1
    masks = torch.tensor([
        [[1, 0, 0, 0], [1, 0, 0, 0]],  # pure part of token 0
        [[1, 1, 1, 1], [1, 1, 1, 1]],  # merged tokens
    ]).float()
    result = _proposal_association(
        masks, oracle, minimum_purity=0.7, minimum_margin=0.2, whole_recall=0.75
    )
    assert result["part"].tolist() == [True, False]
    assert result["associated"].tolist() == [True, False]
