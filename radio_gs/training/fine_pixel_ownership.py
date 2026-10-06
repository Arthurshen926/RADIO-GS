"""Fixed source-writing ownership for restored small masks, without identity labels."""
import torch


def fine_pixel_ownership(existing_masks,restored_small_masks):
    old=torch.as_tensor(existing_masks);fine=torch.as_tensor(restored_small_masks)
    if old.dtype!=torch.bool or fine.dtype!=torch.bool or old.ndim!=2 or fine.ndim!=2 or old.shape[1]!=fine.shape[1] or old.shape[1]<1:
        raise ValueError('Aligned Boolean region by pixel arrays required')
    if old.device!=fine.device:raise ValueError('Mask devices differ')
    areas=fine.sum(1)
    if len(fine) and (bool((areas==0).any()) or bool((fine.float().mean(1)>=.002).any())):
        raise ValueError('Restored masks must be nonempty and below the original0.002area cutoff')
    intersection=old.float()@fine.float().T
    # Parent is strictly >2x area; >=90% refers to the SMALL region's area.
    relation=(old.sum(1)[:,None]>2*areas[None])&(intersection>=.9*areas[None])
    suppressed=old&((relation.float()@fine.float())>0)
    owned=torch.cat([old&~suppressed,fine],dim=0)
    plain=torch.cat([old,fine],dim=0)
    affected=suppressed.any(0)
    assert torch.equal(owned.any(0),plain.any(0)),'Source pixel union changed'
    assert torch.equal(owned[:,~affected],plain[:,~affected]),'Outside ownership pixels changed'
    assert torch.equal(owned[len(old):],fine),'Fine evidence changed'
    return owned,dict(containment_pairs=int(relation.sum()),suppressed_region_pixel_votes=int(suppressed.sum()),
        affected_pixels=int(affected.sum()),coverage_exact=True,outside_affected_masks_exact=True,
        untouched_original_regions=int((~relation.any(1)).sum()),small_regions=len(fine),
        relation=relation,affected_pixels_mask=affected)
