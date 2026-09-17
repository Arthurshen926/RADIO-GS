"""Use one visibility contract for lifted labels and observed element support."""
import torch


def normalize_visible_region_mass(region_mass, element_mass, minimum_mass=1e-6):
    if region_mass.ndim!=2 or element_mass.shape!=(region_mass.shape[0],):
        raise ValueError('region mass axes differ')
    seen=element_mass>minimum_mass
    probability=region_mass/element_mass[:,None].clamp_min(1e-8)
    return torch.where(seen[:,None],probability,torch.zeros_like(probability)),seen
