"""Visibility-qualified negatives for isolated selected-element silhouettes.

A visible source mask cannot label occluded target identity. Removing the
occluders during selected-only rendering must not turn hidden identity into
negative supervision. The unchanged full-scene objective supplies visible
background negatives separately.
"""
import torch


def selected_mask_supervision(target, full_selected_mass, isolated_alpha, source_valid,
                              minimum_visible_fraction=.5):
    if not (target.shape == full_selected_mass.shape == isolated_alpha.shape == source_valid.shape):
        raise ValueError('Pixel axes differ')
    if not 0 < minimum_visible_fraction <= 1:
        raise ValueError('Invalid visibility fraction')
    if not torch.isfinite(full_selected_mass).all() or not torch.isfinite(isolated_alpha).all():
        raise ValueError('Nonfinite compositor evidence')
    if (full_selected_mass < 0).any() or (isolated_alpha < 0).any():
        raise ValueError('Negative compositor evidence')
    positive = target.bool()
    visible_fraction = (full_selected_mass.detach()/isolated_alpha.detach().clamp_min(1e-8)).clamp(0,1)
    reliable_negative = source_valid.bool() & (isolated_alpha.detach()>1e-8) & (visible_fraction>=minimum_visible_fraction)
    return positive | reliable_negative, visible_fraction
