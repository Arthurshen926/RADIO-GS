import torch
from radio_gs.training.source_region_components import connected_source_regions

def test_disconnected_objects_do_not_share_one_crop_descriptor():
 mask=torch.zeros(10,10,dtype=torch.bool);mask[1:4,1:4]=True;mask[6:9,6:9]=True
 components=connected_source_regions(mask)
 assert len(components)==2 and not (components[0]&components[1]).any()
 assert torch.equal(components[0]|components[1],mask)

def test_tiny_component_does_not_expand_other_region():
 mask=torch.zeros(10,10,dtype=torch.bool);mask[1:4,1:4]=True;mask[8,8]=True
 components=connected_source_regions(mask)
 assert len(components)==1 and components[0].sum()==9
