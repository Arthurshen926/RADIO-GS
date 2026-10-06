"""Keep reliability attached to the instance that supplied a mask pixel.

The union is a task output, not an observation confidence. A strong detection
must not lend its score to a disjoint, weak detection. The complement policy is
separate: retain the declared view strength or explicitly request the bounded
minimum-confidence experiment. Neither is verified negative evidence; an absent
detection supplies no exclusion evidence.
"""
import torch
from torch.nn import functional as F


def instance_reliability(record, size, device, scores=None, complement='view'):
    scores=record['scores'] if scores is None else scores
    scores=scores.float().to(device)
    if not len(scores):
        return torch.zeros(size,device=device),torch.zeros((),device=device)
    masks=record['masks'].float().to(device)
    owner=masks.argmax(0)
    positive=scores[owner]
    positive=F.interpolate(positive[None,None],size=size,mode='bilinear',align_corners=False)[0,0]
    negative=(scores.max() if complement=='view' else scores.min())*float(record.get('negative_reliability',1.))
    return positive,negative
