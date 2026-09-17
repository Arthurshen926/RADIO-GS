import torch
from radio_gs.training.robust_source_fusion import robust_view_fusion


def test_fixed_consensus_reduces_isolated_outlier_and_is_view_permutation_invariant():
    values=torch.tensor([[[1.,0.]],[[1.,0.]],[[0.,1.]]])
    weights=torch.ones(3,1)
    mean,_=robust_view_fusion(values,weights,iterations=0)
    result,valid=robust_view_fusion(values,weights)
    assert valid.all() and result[0,0]>mean[0,0]
    torch.testing.assert_close(result,robust_view_fusion(values[[2,0,1]],weights)[0])


def test_excluded_view_never_affects_solution_and_zero_mass_is_invalid():
    values=torch.tensor([[[1.,2.],[9.,9.]],[[9.,9.],[2.,3.]]])
    mass=torch.tensor([[1.,0.],[1.,0.]])
    x,valid=robust_view_fusion(values,mass,excluded_view=1)
    values[1]=1000
    torch.testing.assert_close(x,robust_view_fusion(values,mass,excluded_view=1)[0])
    assert valid.tolist()==[True,False]
    torch.testing.assert_close(x,torch.tensor([[1.,2.],[0.,0.]]))
