import math
import pytest
import torch
from radio_gs.query_field.support_footprint import projected_integral, SupportFootprint


def test_integral_matches_independent_dense_quadrature():
    # Rotated anisotropic Gaussian; independent direct numerical integration.
    c = torch.tensor([[2., .4, .5]], dtype=torch.float64)
    grid = torch.linspace(-12, 12, 1601, dtype=torch.float64)
    x, y = torch.meshgrid(grid, grid, indexing='ij')
    density = .7 * torch.exp(-.5*(2*x*x+.8*x*y+.5*y*y))
    numeric = torch.trapezoid(torch.trapezoid(density, grid), grid)
    actual = projected_integral(c, torch.tensor([.7], dtype=torch.float64), torch.tensor([True]))[0]
    assert float(actual) == pytest.approx(float(numeric), rel=1e-8)


def test_domain_masks_unavailable_and_ignores_invalid_hidden_conic():
    result = projected_integral(torch.tensor([[1., 0, 1], [float('nan'), 0, 0]]),
                                torch.tensor([.5, .5]), torch.tensor([True, False]))
    assert float(result[0]) == pytest.approx(math.pi)
    assert result[1] == 0


def test_view_selection_preserves_source_active_view_sum():
    m = SupportFootprint({'cost': torch.tensor([[1., 2], [4., 8]])}, 2)
    assert torch.equal(m(torch.tensor([True, False])), torch.tensor([1., 2]))
    assert torch.equal(m(torch.tensor([False, False])), torch.zeros(2))
    with pytest.raises(ValueError): m(torch.tensor([1., 0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='projection requires CUDA')
def test_compiler_matches_source_reference_and_expands_actual_domain():
    from types import SimpleNamespace
    from radio_gs.query_field.support_footprint import compile_support_footprint
    from scripts.research.conditional_subset_reference import unoccluded_footprint
    field = SimpleNamespace(xyz=torch.tensor([[0.,0.,2.], [.1,0.,3.], [0.,0.,-2.]],device='cuda'),
        quaternions=torch.tensor([[1.,0,0,0]],device='cuda').repeat(3,1),
        scales=torch.full((3,3),.1,device='cuda'),opacities=torch.full((3,1),.8,device='cuda'),
        observed=torch.tensor([True,False,True],device='cuda'),world_units_per_metre=1.)
    camera = SimpleNamespace(intrinsic=torch.tensor([[30.,0,16.5],[0,30.,16.5],[0,0,1.]]),
        world_to_camera=torch.eye(4),height=32,width=32)
    before = field.xyz.clone()
    legacy = unoccluded_footprint(field,camera,32,32,1.)
    actual = compile_support_footprint(field,camera,32,32)
    assert torch.equal(actual,legacy)
    expanded = compile_support_footprint(field,camera,32,32,member_domain=torch.ones(3,dtype=torch.bool,device='cuda'))
    assert expanded[0] == legacy[0] and expanded[1] > 0 and expanded[2] == 0
    assert torch.equal(before,field.xyz)
