"""Keep renderer membership strength when a view only supports foreground.

The foreground mass is an evidence weight, not the normalizing visibility.
Only the same high-purity foreground already required by the downstream
consensus can vote. Uncertain / outside elements supply neither sign.
"""
import torch


def positive_scope_evidence(foreground, visibility, *, minimum_purity=.75):
    if foreground.shape != visibility.shape:
        raise ValueError('foreground and visibility must align')
    posterior = (foreground / visibility.clamp_min(1e-8)).clamp(0, 1)
    valid = (visibility > 1e-6) & (foreground > 1e-6) & (posterior >= minimum_purity)
    weight = torch.where(valid, foreground, 0)
    # n/n incorrectly treats every small positive tail as posterior one.
    return weight * posterior, weight
