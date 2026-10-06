import torch
from radio_gs.training.alpha_weighted_source_evidence import alpha_weighted_source_evidence
from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights


def test_transparent_ray_does_not_equal_opaque_ray_across_views():
    mask=torch.ones((1,1),dtype=torch.bool);pid=torch.tensor([0]);gid=torch.tensor([0])
    old=[];new=[]
    for mass in (1.,1e-6):
        value=torch.tensor([mass])
        old.append(source_pixel_evidence_weights(mask,pid,gid,value,1,partition_overlap=True)[0][0,0])
        new.append(alpha_weighted_source_evidence(mask,pid,gid,value,1)[0][0,0])
    old=torch.stack(old);new=torch.stack(new)
    torch.testing.assert_close(old,torch.ones(2))
    assert new[0]/new[1]>=999999
    teacher=torch.eye(2)
    torch.testing.assert_close(old@teacher/old.sum(),torch.tensor([.5,.5]))
    assert (new@teacher/new.sum())[0]>.999


def test_overlap_partition_conserves_raw_covered_evidence_mass():
    masks=torch.tensor([[1,1,0],[0,1,0]],dtype=torch.bool)
    pid=torch.tensor([0,0,1,2]);gid=torch.tensor([0,1,1,2]);value=torch.tensor([.1,.3,.2,.9])
    weights,mass=alpha_weighted_source_evidence(masks,pid,gid,value,3)
    torch.testing.assert_close(weights.sum(0),torch.tensor([.1,.5,0.]))
    torch.testing.assert_close(mass,torch.tensor([.4,.2,.9]))


def test_opaque_pixel_baseline_equivalence_and_same_weak_element_gate():
    masks=torch.ones((1,2),dtype=torch.bool)
    pid=torch.tensor([0,0,1,1]);gid=torch.tensor([0,1,1,2]);value=torch.tensor([.25,.75,1.,1e-8])
    old,_=source_pixel_evidence_weights(masks,pid,gid,value,3,partition_overlap=True)
    new,_=alpha_weighted_source_evidence(masks,pid,gid,value,3)
    torch.testing.assert_close(new,old)
    assert new[0,2]==0
