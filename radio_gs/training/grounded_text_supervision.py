"""Cross-view corroboration for text-grounded source masks."""
import torch


def independent_positive_carriers(positive_counts, current_positive_vote, negative_counts):
    """One OTHER view supports a current positive pixel; negatives veto rescue.

    This is a pixel-supervision support, not a new hard 3D positive label.
    """
    if positive_counts.shape!=current_positive_vote.shape or positive_counts.shape!=negative_counts.shape:
        raise ValueError('independent view vote axes differ')
    if current_positive_vote.dtype!=torch.bool or (positive_counts<current_positive_vote).any() or (negative_counts<0).any():
        raise ValueError('invalid independent view votes')
    return (positive_counts-current_positive_vote.to(positive_counts.dtype)>=1)&(negative_counts==0)


def corroborated_labels(positive_views, negative_views, minimum_views=2):
    if positive_views.shape!=negative_views.shape or minimum_views<2:
        raise ValueError('invalid cross-view label counts')
    positive=positive_views>=minimum_views;negative=negative_views>=minimum_views
    conflict=positive&negative
    return positive&~conflict,negative&~conflict


def grounded_pixel_evidence(probability, other_positive, positive_support, negative_support, geometry_support):
    if any(x.shape!=probability.shape for x in (other_positive,positive_support,negative_support,geometry_support)):
        raise ValueError('grounded pixel axes differ')
    positive=(probability>.8)&(positive_support>.5)&(geometry_support>.5)
    # No detection is an abstention, even if other categories were detected.
    negative=(probability<.05)&bool((probability>.8).any())&other_positive.bool()&(negative_support>.5)&(geometry_support>.5)
    return positive,negative
