"""Keep hard support associations while retaining weak target evidence softly."""
import math

import torch

from .paired_evidence import PairedEvidence

KIND = 'soft_partial_view_evidence_v1'


class SoftPairedEvidence(PairedEvidence):
    def __init__(self, state, count):
        extra = {k: v for k, v in state.items() if k.startswith('association_')}
        if set(extra) != {'association_' + s for s in ['pointer', 'index', 'value']}:
            raise ValueError('missing fixed support association coordinates')
        base = {k: v for k, v in state.items() if k not in extra}
        super().__init__(base, count)
        # Reuse the same CSR validation for the fixed association function.
        PairedEvidence(dict(base, **{k.replace('association_', 'positive_'): v for k, v in extra.items()}), count)
        for k, v in extra.items(): self.register_buffer(k, v.detach())

    def coefficients(self, support, scale):
        support.validate(self.count)
        if not math.isfinite(float(scale)) or float(scale) <= 0:
            raise ValueError('invalid physical scale')
        matrix = self.matrix('association', support.indices)
        weights = support.weights.to(self.scales); weights = weights / weights.sum()
        association = torch.sparse.mm(matrix.transpose(0, 1), weights[:, None]).flatten()
        ratio = self.scales / float(scale)
        association = association * ((association >= .5) & (ratio >= .5) & (ratio <= 2))
        mass = self.scales.new_zeros(self.view_count).scatter_add_(0, self.views, association)
        return association / mass[self.views].clamp_min(1e-12)


def enable(field, state):
    if field.evidence is not None or field.support_aggregation != 'normalized_prototype':
        raise ValueError('soft paired evidence requires an unaugmented prototype core')
    field.evidence = SoftPairedEvidence(state, len(field.xyz)).to(field.xyz.device)
    field.evidence_kind = KIND
