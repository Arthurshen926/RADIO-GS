"""Augment query-free source evidence without replacing it with grounded labels."""
import torch


def augment_region_inventory(parent_masks,parent_views,parent_descriptors,masks,views,descriptors):
    if parent_masks.shape[1:]!=masks.shape[1:] or parent_descriptors.shape[1:]!=descriptors.shape[1:]:
        raise ValueError('source inventory feature or raster contracts differ')
    if any(len(m)!=len(v) or len(m)!=len(d) for m,v,d in [(parent_masks,parent_views,parent_descriptors),(masks,views,descriptors)]):
        raise ValueError('source inventory axes differ')
    if parent_masks.dtype!=torch.bool or masks.dtype!=torch.bool:
        raise ValueError('source region masks must be boolean')
    # Keep all original rows/identities unchanged; skip only exact same-view
    # duplicate additions. Nested parts and wholes are never deduplicated.
    keys={(int(v),m.cpu().numpy().tobytes()) for m,v in zip(parent_masks,parent_views)}
    added=[]
    for i,(mask,view) in enumerate(zip(masks,views)):
        key=(int(view),mask.cpu().numpy().tobytes())
        if key not in keys:added.append(i);keys.add(key)
    ids=torch.tensor(added,dtype=torch.long,device=masks.device)
    return (torch.cat([parent_masks,masks[ids]]),torch.cat([parent_views,views[ids]]),
            torch.cat([parent_descriptors,descriptors[ids]]),ids)
