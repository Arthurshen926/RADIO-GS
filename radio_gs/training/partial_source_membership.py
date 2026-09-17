"""Conservative source pseudo-labels; contradictory views remain unknown."""
import torch


def corroborated_membership_masks(probability, visible_mass):
    """Require two positive/negative observations and no opposing observation.

    Uses the existing source positive >.7 / negative <.1 conventions. Agreement
    does not establish ground truth or prove completeness of teacher detections.
    """
    if probability.ndim != 2 or probability.shape != visible_mass.shape:
        raise ValueError('source view/element axes differ')
    seen = visible_mass > 1e-6
    if not torch.isfinite(visible_mass).all() or (visible_mass < 0).any():
        raise ValueError('invalid visible mass')
    if not torch.isfinite(probability[seen]).all() or ((probability[seen] < 0) | (probability[seen] > 1)).any():
        raise ValueError('invalid observed membership')
    pos = ((probability > .7) & seen).sum(0)
    neg = ((probability < .1) & seen).sum(0)
    positive = (pos >= 2) & (neg == 0)
    negative = (neg >= 2) & (pos == 0)
    conflict = (pos > 0) & (neg > 0)
    return positive, negative, conflict
