"""Partial, view-balanced relation lookup before mixing uncertain supports.

No observation is an abstention, never a negative vote. Each component's
association is normalized within a view; its confidence and its input weight
remain outside that normalization. The target evidence/readout is unchanged.
"""
import math

import torch

from .soft_paired_evidence import SoftPairedEvidence

KIND = 'mixture_partial_view_evidence_v1'


class MixturePairedEvidence(SoftPairedEvidence):
    def preference(self, positive, known):
        return positive / known.clamp_min(1e-12)

    def coefficients(self, support, scale):
        support.validate(self.count)
        if not math.isfinite(float(scale)) or float(scale) <= 0:
            raise ValueError('invalid physical scale')
        weights = support.weights.to(self.scales)
        weights = weights / weights.sum()
        ratio = self.scales / float(scale)
        compatible = (ratio >= .5) & (ratio <= 2)
        coefficient = torch.zeros_like(self.scales)
        # Chunking bounds temporary support x region state for full scribbles.
        for offset in range(0, len(support.indices), 512):
            rows = support.indices[offset:offset + 512]
            positive = self.matrix('positive', rows).to_dense()
            negative = self.matrix('negative', rows).to_dense()
            known = positive + negative
            admitted = (positive > negative) & compatible[None, :]
            preference = torch.where(admitted, self.preference(positive, known), 0.)
            views = self.views.expand(len(rows), -1)
            mass = positive.new_zeros((len(rows), self.view_count))
            mass.scatter_add_(1, views, preference)
            confidence = torch.zeros_like(mass)
            confidence.scatter_reduce_(1, views, torch.where(admitted, known, 0.),
                                       reduce='amax', include_self=True)
            association = preference / mass[:, self.views].clamp_min(1e-12)
            association *= confidence[:, self.views].clamp_max(1.)
            coefficient += (association * weights[offset:offset + len(rows), None]).sum(0)
        return coefficient

    def joint_features(self, support, scale, rows=None):
        """Six partial-evidence analogues of v10's joint observation statistics.

        Exact v10 parity holds for binary membership/complementary visible
        exclusion and a single anchor. Soft partial evidence weights known
        mass explicitly. This is diagnostic state, not a transferred labelled
        decision head, and does not claim matching v10's observation budget.
        """
        coefficient = self.coefficients(support, scale)
        active = coefficient > 0
        count = self.count if rows is None else len(rows)
        if not bool(active.any()):
            return self.scales.new_zeros((count, 6))
        index = torch.where(active)[0]
        views = self.views[index]
        association = torch.sparse_coo_tensor(torch.stack([index, views]), coefficient[index],
            (len(coefficient), self.view_count)).to_dense()
        p, n = [torch.sparse.mm(self.matrix(prefix, rows), association)
                for prefix in ['positive', 'negative']]
        known = p + n
        denominator = known.sum(1).clamp_min(1e-12)
        any_mask = (known * (p > 0)).sum(1) / denominator
        all_masks = (known * ((p > 0) & (n <= 1e-6))).sum(1) / denominator
        return torch.stack([p.sum(1) / denominator, any_mask, all_masks,
            any_mask - all_masks, known.sum(1) / coefficient.sum().clamp_min(1e-12),
            (known.sum(1) > 0).to(p)], 1)


def enable(field, state):
    if field.evidence is not None or field.support_aggregation != 'normalized_prototype':
        raise ValueError('mixture evidence requires an unaugmented prototype core')
    field.evidence = MixturePairedEvidence(state, len(field.xyz)).to(field.xyz.device)
    field.evidence_kind = KIND
