"""One-pass spatial completion from other views' admitted signed evidence.

All inputs belong to the same field/current query. This is a functional research
candidate, not an independent observation or a calibrated localization model.
Target-view evidence never contributes to its own witness. No completed labels
are recycled. Current-view native positive interiors limit the completion domain.
"""
import torch
from .scope_activation import admitted_negative


EPS = 1e-8


def crossed_scope(positive, negative, confidence):
    """Replace functions after materializing the originating negative permission."""
    n = admitted_negative(negative)
    conflict = n & (positive['target'] > .5)
    return dict(target=positive['target'], positive=positive['positive'],
                negative=n & ~conflict, confidence=confidence), int(conflict.sum())


@torch.no_grad()
def complete_other_view_positive(preparation, atomic, native):
    """Lift donor labels, exclude self, then project a majority visible witness.

    A donor Gaussian is positive only when its positive mass exceeds negative
    mass by EPS. Unknown donor support remains unknown. More than half of the
    target ray's TOTAL available observation mass must have this positive
    witness: a tiny observed fragment cannot normalize itself into certainty.
    Only previously unknown, native-positive interior rays can be completed.
    Original atomic negatives retain their original view permission even when
    new positives activate a previously inactive target view.
    """
    views = preparation.views
    if len(views) != len(atomic) or len(native) != len(atomic):
        raise ValueError('unaligned scope views')
    masses = []
    active = []
    for view, scope in zip(views, atomic):
        enabled = bool(scope['confidence'] > 0 and scope['positive'].any())
        active.append(enabled)
        labels = torch.stack([scope['positive'], scope['negative']], 1).double()
        labels *= scope['confidence'].double() if enabled else 0.
        op = view.operator().double()
        masses.append(torch.sparse.mm(op.T, labels))
    result, diagnostics = [], []
    for vi, (view, base, hypothesis) in enumerate(zip(views, atomic, native)):
        # Explicit exclusion avoids total-minus-self floating-point remnants.
        donor = torch.zeros_like(masses[vi])
        for other in range(len(views)):
            if other != vi:
                donor += masses[other]
        witness = (donor[:, 0] - donor[:, 1]) > EPS
        inputs = torch.stack([witness.double(), torch.ones_like(witness, dtype=torch.float64)], 1)
        projected = torch.sparse.mm(view.operator().double(), inputs)
        supported = projected[:, 0] > .5 * projected[:, 1] + EPS
        original_n = admitted_negative(base)
        unknown = ~(base['positive'] | original_n)
        added = unknown & hypothesis['positive'] & supported & view.valid
        target = base['target'].clone()
        target[added] = 1.
        scope = dict(target=target, positive=base['positive'] | added,
                     negative=original_n & ~(target > .5), confidence=base['confidence'])
        result.append(scope)
        diagnostics.append(dict(added_positive_rays=int(added.sum()),
            donor_views=sum(active) - int(active[vi]), donor_positive_elements=int(witness.sum()),
            witness_rays=int((supported & view.valid).sum()),
            originally_active=active[vi],
            newly_active=bool(not active[vi] and scope['confidence'] > 0 and scope['positive'].any())))
    return result, diagnostics
