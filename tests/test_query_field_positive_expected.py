import torch
from radio_gs.query_field.contracts import Support
from radio_gs.query_field.positive_rays import mixture_logits
from test_query_preserving_field import example


def test_expected_ray_union_matches_direct_per_gaussian_energy_integral():
    f=example();f.spatial_metric='scale_normalized';f.world_units_per_metre=1.
    p=Support(torch.tensor([0,1,3]),torch.tensor([.3,.4,.6]),torch.tensor([0,0,1]))
    direct=torch.stack([f.member_logits(Support(torch.tensor([i]),torch.ones(1)),2.,include_evidence=False) for i in [0,1,3]],1)
    energies=torch.stack([(direct[:,0]*.3+direct[:,1]*.4)/.7,direct[:,2]],1)
    expected=(energies.sigmoid()*torch.tensor([.7,.6])).max(1).values.clamp(1e-7,1-1e-7)
    actual=mixture_logits(f,p,2.,aggregation='expected_union')
    torch.testing.assert_close(actual.sigmoid(),expected,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(actual,mixture_logits(f,p,2.,aggregation='expected_union',chunk_size=1))
    coherent=mixture_logits(f,p,2.,aggregation='coherent_union')
    assert bool((actual<=coherent+1e-6).all()) and bool((actual<coherent-1e-4).any())


def test_single_depth_expected_and_normalized_union_agree():
    f=example();p=Support(torch.tensor([0,2]),torch.tensor([1.,.7]),torch.tensor([0,1]))
    torch.testing.assert_close(mixture_logits(f,p,1.,aggregation='expected_union'),mixture_logits(f,p,1.,aggregation='confidence_union'))
