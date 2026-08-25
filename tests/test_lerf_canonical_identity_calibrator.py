import torch

from radio_gs.scripts.train_lerf_canonical_identity_calibrator import _class_balanced_loss


def test_class_balanced_identity_loss_is_invariant_to_sampled_class_ratio() -> None:
    probability = torch.tensor([0.8, 0.7, 0.2, 0.1])
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])
    reference = _class_balanced_loss(probability, target, logits=False)
    duplicated_negatives = _class_balanced_loss(
        torch.cat((probability[:2], probability[2:].repeat(5))),
        torch.cat((target[:2], target[2:].repeat(5))),
        logits=False,
    )
    torch.testing.assert_close(reference, duplicated_negatives)


def test_class_balanced_identity_loss_requires_both_classes() -> None:
    try:
        _class_balanced_loss(torch.tensor([0.9, 0.8]), torch.ones(2), logits=False)
    except ValueError as error:
        assert "both classes" in str(error)
    else:
        raise AssertionError("one-class identity episode was accepted")
