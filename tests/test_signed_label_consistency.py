import torch
import pytest
from radio_gs.training.signed_label_consistency import conflicting_signed_targets


def test_majority_conflict_respects_unknowns_and_observation_support():
    rows=[torch.tensor([0,1])];weights=[torch.tensor([.8,.2])]
    target=torch.tensor([1.,0.]);known=torch.tensor([True,True]);observed=known.clone()
    assert conflicting_signed_targets(rows,weights,[0],target,known,observed)==[True]
    assert conflicting_signed_targets(rows,weights,[1],target,known,observed)==[False]
    assert conflicting_signed_targets(rows,weights,[0],target,torch.tensor([False,True]),observed)==[False]
    assert conflicting_signed_targets(rows,weights,[0],target,known,torch.tensor([False,True]))==[False]


def test_bad_weights_are_not_silently_treated_as_ambiguous():
    with pytest.raises(ValueError,match='weights'):
        conflicting_signed_targets([torch.tensor([0])],[torch.tensor([-1.])],[1],torch.ones(1),torch.ones(1,dtype=torch.bool),torch.ones(1,dtype=torch.bool))
