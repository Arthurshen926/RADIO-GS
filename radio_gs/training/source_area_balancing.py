"""A bounded control giving equal evidence to occupied mask-area octaves."""
import torch


def area_balanced_pixel_votes(masks):
    if masks.ndim != 2:
        raise ValueError('Masks must be region by pixel')
    masks=masks.bool()
    area=masks.sum(1)
    octave=area.clamp_min(1).float().log2().floor().long()
    result=torch.zeros_like(masks,dtype=torch.float32)
    occupied=torch.zeros(masks.shape[1],device=masks.device)
    for group in octave[area>0].unique():
        selected=(octave==group)&(area>0)
        count=masks[selected].sum(0)
        result[selected]=masks[selected].float()/count.clamp_min(1)[None]
        occupied+=(count>0)
    return result/occupied.clamp_min(1)[None]
