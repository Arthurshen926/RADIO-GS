import pytest
import torch
from radio_gs.training.selected_output_supervision import selected_mask_supervision


def test_occluded_selected_identity_is_unknown_not_negative():
    # A foreground occluder transmits only 10% of the isolated target alpha.
    valid, fraction = selected_mask_supervision(torch.tensor([0,0,1],dtype=torch.bool),
        torch.tensor([.09,.9,.09]),torch.tensor([.9,.9,.9]),torch.ones(3,dtype=torch.bool))
    assert valid.tolist()==[False,True,True]
    torch.testing.assert_close(fraction,torch.tensor([.1,1.,.1]))


def test_unsupported_background_is_not_a_negative_and_no_visibility_gradient():
    mass=torch.tensor([0.,.5,.8],requires_grad=True)
    alpha=torch.tensor([0.,1.,1.],requires_grad=True)
    valid,fraction=selected_mask_supervision(torch.zeros(3,dtype=torch.bool),mass,alpha,
        torch.tensor([True,True,False]))
    assert valid.tolist()==[False,True,False]
    assert not fraction.requires_grad


def test_bad_axes_and_nonfinite_fail():
    with pytest.raises(ValueError):
        selected_mask_supervision(torch.ones(1),torch.ones(2),torch.ones(2),torch.ones(2))
    with pytest.raises(ValueError):
        selected_mask_supervision(torch.ones(1),torch.tensor([float('nan')]),torch.ones(1),torch.ones(1))
