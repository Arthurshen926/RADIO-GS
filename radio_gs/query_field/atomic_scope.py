"""Set-valued native evidence on query-free segmentation atoms.

Overlapping proposals are alternative semantic scopes, not independent votes.
Their response interval is not a calibrated probability interval. Unanimous
signs can supply evidence; disagreement and uncovered atoms remain unknown.
This is a fixed research candidate, not an established identity mechanism.
"""
import torch


@torch.no_grad()
def atomic_response_interval(view, probability):
    if (probability.shape != view.retained.shape or not probability.is_floating_point()
            or probability.device != view.incidence.device
            or not torch.isfinite(probability).all()
            or bool(((probability < 0) | (probability > 1)).any())):
        raise ValueError('invalid native proposal probabilities')
    membership = view.incidence
    covered = membership.any(1)
    lower = torch.where(membership, probability[None], 1.).amin(1)
    upper = torch.where(membership, probability[None], 0.).amax(1)
    positive = covered & (lower > .5)
    negative = covered & (upper < .5)
    return positive, negative, dict(covered=covered, lower=lower, upper=upper)


@torch.no_grad()
def assemble_atomic_scope(view, preparation_view, prior, probability, confidence):
    """Only sign-consistent atoms grant evidence; never infer complement negatives.

    The scalar confidence, native-positive negative protection, view activation,
    initializer, footprint and member engine keep the preceding contract. The
    member cost remains a regularizer even on unknown rows; unknown means no
    data label, not a guarantee that every prior member will be retained.
    """
    positive_atoms, negative_atoms, interval = atomic_response_interval(view, probability)
    neighbors = view.neighborhood
    positive = positive_atoms[neighbors].all(1)
    negative = negative_atoms[neighbors].all(1)
    valid = preparation_view.valid
    native_positive = torch.sparse.mm(preparation_view.operator(), prior[:, None])[:, 0] > .5
    positive &= valid
    negative &= valid & ~native_positive
    target = positive_atoms[neighbors[:, 4]].float()
    scope = dict(target=target, positive=positive, negative=negative, confidence=confidence.float())
    center = neighbors[:, 4]
    diagnostic = dict(positive_atoms=int(positive_atoms.sum()), negative_atoms=int(negative_atoms.sum()),
        unknown_atoms=int((~positive_atoms & ~negative_atoms).sum()),
        conflicting_valid_rays=int((valid & interval['covered'][center]
            & ~positive_atoms[center] & ~negative_atoms[center]).sum()),
        uncovered_valid_rays=int((valid & ~interval['covered'][center]).sum()),
        positive_rays=int(positive.sum()), negative_rays=int(negative.sum()))
    return scope, diagnostic
