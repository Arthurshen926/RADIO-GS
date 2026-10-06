import pytest
import torch
from radio_gs.training.fine_pixel_ownership import fine_pixel_ownership
from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights


def masks():
    old=torch.zeros((3,10000),dtype=torch.bool);fine=torch.zeros((1,10000),dtype=torch.bool)
    fine[0,:10]=True;old[0,:21]=True;old[1,:20]=True;old[2,:8]=True;old[2,30:50]=True
    return old,fine


def test_strict_area_ratio_and_small_denominator_containment():
    old,fine=masks();owned,r=fine_pixel_ownership(old,fine)
    assert r['relation'].tolist()==[[True],[False],[False]]
    assert not owned[0,:10].any() and owned[0,10:21].all()
    assert torch.equal(owned[1:3],old[1:3])
    assert torch.equal(owned[3],fine[0])


def test_ninety_percent_intersection_only_removes_real_overlap():
    old,fine=masks();old[0,9]=False;old[0,40]=True
    owned,r=fine_pixel_ownership(old,fine)
    assert r['relation'][0,0]
    assert r['suppressed_region_pixel_votes']==9
    assert torch.equal(owned[:,10:],torch.cat([old,fine])[:,10:])


def test_source_ownership_mass_and_lift_conserved():
    old,fine=masks();owned,r=fine_pixel_ownership(old,fine);plain=torch.cat([old,fine])
    p=plain.float()/plain.sum(0).clamp_min(1);o=owned.float()/owned.sum(0).clamp_min(1)
    assert torch.equal(p[:,~r['affected_pixels_mask']],o[:,~r['affected_pixels_mask']])
    assert torch.allclose(p.sum(0),o.sum(0),atol=1e-7,rtol=0)
    ids=torch.arange(10000);elements=ids%5;contributions=torch.ones(10000)
    a,_=source_pixel_evidence_weights(plain,ids,elements,contributions,5,partition_overlap=True)
    b,_=source_pixel_evidence_weights(owned,ids,elements,contributions,5,partition_overlap=True)
    assert torch.allclose(a.sum(0),b.sum(0),atol=2e-6,rtol=0)


def test_overlapping_fine_masks_partition_equally_and_unrelated_evidence_retained():
    old,fine=masks();fine=torch.cat([fine,fine]);owned,_=fine_pixel_ownership(old,fine)
    ownership=owned.float()/owned.sum(0).clamp_min(1)
    assert torch.equal(ownership[-1],ownership[-2])
    assert torch.equal(owned[1:3],old[1:3])


def test_no_small_and_no_parent_are_exact_noops():
    old,fine=masks();owned,_=fine_pixel_ownership(old,fine[:0]);assert torch.equal(owned,old)
    old=old[:1].clone();old[:,:10]=False
    owned,r=fine_pixel_ownership(old,fine)
    assert not r['relation'].any() and torch.equal(owned,torch.cat([old,fine]))


def test_invalid_masks_rejected_and_inputs_unchanged():
    old,fine=masks();backup=old.clone();fine_pixel_ownership(old,fine);assert torch.equal(backup,old)
    with pytest.raises(ValueError):fine_pixel_ownership(old.float(),fine)
    with pytest.raises(ValueError):fine_pixel_ownership(old,torch.zeros_like(fine))
    with pytest.raises(ValueError):fine_pixel_ownership(old,torch.ones_like(fine))
