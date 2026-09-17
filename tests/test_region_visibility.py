import torch
from radio_gs.training.region_visibility import normalize_visible_region_mass


def test_tiny_mass_cannot_create_confident_invisible_labels():
    mass=torch.tensor([1.,1e-7,0.]);region_mass=torch.tensor([[.8,.2],[1e-7,0.],[0.,0.]])
    p,seen=normalize_visible_region_mass(region_mass,mass)
    assert seen.tolist()==[True,False,False]
    torch.testing.assert_close(p[0],torch.tensor([.8,.2]))
    assert not p[~seen].any()
