"""Independent compositor derivation for the selected-visibility loss control."""
import torch
from radio_gs.training.selected_output_supervision import selected_mask_supervision


def compose(alpha,gates):
    transmission=torch.cat([torch.ones_like(alpha[:1]),(1-alpha).cumprod(0)[:-1]],0)
    contribution=transmission*alpha
    return (contribution*gates).sum(0),1-(1-alpha*gates).prod(0),contribution


def test_hard_hidden_subset_fraction_is_original_transmittance():
    full,isolated,_=compose(torch.tensor([[.9],[.8]]),torch.tensor([[0.],[1.]]))
    valid,fraction=selected_mask_supervision(torch.tensor([False]),full,isolated,torch.tensor([True]))
    torch.testing.assert_close(fraction,torch.tensor([.1]))
    assert not valid.item()


def test_normalized_observed_mapping_times_mass_is_exact_attribution():
    alpha=torch.tensor([[.2,.5],[.7,.3],[.9,.8]],dtype=torch.float64)
    observed=torch.tensor([[1.],[0.],[1.]],dtype=torch.float64)
    gates=torch.tensor([[.4,.8],[0.,0.],[.7,.3]],dtype=torch.float64)
    full,isolated,contribution=compose(alpha,gates)
    observed_mass=(contribution*observed).sum(0)
    normalized=contribution*observed/observed_mass[None]
    implementation=(normalized*gates).sum(0)*observed_mass
    torch.testing.assert_close(implementation,full)
    assert (full<=isolated).all()


def test_soft_gate_attribution_fraction_is_bounded_in_multilayer_scene():
    generator=torch.Generator().manual_seed(943)
    alpha=torch.rand((25,100),generator=generator,dtype=torch.float64)*.99
    gates=torch.rand((25,100),generator=generator,dtype=torch.float64)
    full,isolated,_=compose(alpha,gates)
    assert (full>=0).all() and (full<=isolated+1e-14).all()
    valid,fraction=selected_mask_supervision(torch.zeros(100,dtype=torch.bool),full,isolated,torch.ones(100,dtype=torch.bool))
    torch.testing.assert_close(fraction,full/isolated)
    assert torch.equal(valid,fraction>=.5)


def test_visibility_detaches_but_selected_loss_still_backpropagates():
    mass=torch.tensor([.1,.8],requires_grad=True)
    alpha=torch.tensor([.8,.9],requires_grad=True)
    selected,fraction=selected_mask_supervision(torch.tensor([False,False]),mass,alpha,torch.ones(2,dtype=torch.bool))
    loss=(alpha*selected.float()).sum()
    loss.backward()
    assert mass.grad is None and not fraction.requires_grad
    torch.testing.assert_close(alpha.grad,torch.tensor([0.,1.]))
