import torch
from radio_gs.training.native_region_mixing import mix_region_descriptors

def test_specific_region_is_not_diluted_equally_by_broad_region():
 masks=torch.tensor([[1,1,1,1],[1,0,0,0]],dtype=torch.bool);descriptors=torch.eye(2)
 equal,_=mix_region_descriptors(masks,descriptors)
 specific,valid=mix_region_descriptors(masks,descriptors,inverse_area=True)
 assert torch.allclose(equal[0],torch.tensor([.5,.5]))
 assert torch.allclose(specific[0],torch.tensor([.2,.8]))
 assert torch.allclose(specific[1:],torch.tensor([[1.,0.]]).expand(3,-1)) and valid.all()

def test_mixing_does_not_invent_uncovered_evidence():
 value,valid=mix_region_descriptors(torch.tensor([[True,False]]),torch.tensor([[1.,2.]]),inverse_area=True)
 assert valid.tolist()==[True,False] and value[1].count_nonzero()==0
