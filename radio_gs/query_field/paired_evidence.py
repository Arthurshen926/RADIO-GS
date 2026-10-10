"""Uncompressed partial-relation functions on the shared Gaussian carrier.

This is a capacity reference, not a learned compact embedding. Sparse positive
and reliable-exclusion coordinates preserve the source relation functions;
view identifiers only balance dependent nested coordinates. No images, masks,
cameras, source loaders or labelled member model are used
by this module. Unknown relations contribute neither positive nor negative mass.
"""
import math

import torch
from torch import nn


KIND = 'partial_view_evidence_v1'


class PairedEvidence(nn.Module):
    def __init__(self, state, count):
        super().__init__()
        state = dict(state)
        semantic = state.pop('region_semantic', None)
        expected = {'views', 'scales'} | {f'{p}_{k}' for p in ['positive', 'negative']
                                           for k in ['pointer', 'index', 'value']}
        if set(state) != expected:
            raise ValueError('incomplete paired-evidence state')
        for name, value in state.items():
            self.register_buffer(name, value.detach())
        if semantic is not None and (semantic.ndim != 2 or semantic.shape[0] != len(self.views) or
                                     semantic.shape[1] == 0 or not bool(torch.isfinite(semantic).all())):
            raise ValueError('unaligned native region semantics')
        self.register_buffer('region_semantic', None if semantic is None else semantic.detach())
        self.count = count
        if self.views.ndim != 1 or self.scales.shape != self.views.shape or not len(self.views):
            raise ValueError('unaligned relation coordinates')
        if self.views.dtype != torch.int64 or bool((self.views < 0).any()):
            raise ValueError('invalid relation view groups')
        if not torch.equal(torch.unique(self.views), torch.arange(int(self.views.max()) + 1, device=self.views.device)):
            raise ValueError('relation view groups must be contiguous')
        if not bool(torch.isfinite(self.scales).all()) or bool((self.scales <= 0).any()):
            raise ValueError('invalid relation physical scales')
        self.view_count = int(self.views.max()) + 1
        for prefix in ['positive', 'negative']:
            pointer, index, value = self.parts(prefix)
            if pointer.shape != (count + 1,) or pointer.dtype != torch.int64 or index.dtype != torch.int64:
                raise ValueError('invalid Gaussian CSR rows')
            if int(pointer[0]) != 0 or int(pointer[-1]) != len(value) or index.shape != value.shape:
                raise ValueError('unaligned Gaussian CSR payload')
            if bool((pointer[1:] < pointer[:-1]).any()) or bool(((index < 0) | (index >= len(self.views))).any()):
                raise ValueError('invalid Gaussian CSR indices')
            if not bool(torch.isfinite(value).all()) or bool(((value <= 0) | (value > 1 + 1e-5)).any()):
                raise ValueError('invalid partial evidence mass')

    def parts(self, prefix):
        return tuple(getattr(self, prefix + '_' + k) for k in ['pointer', 'index', 'value'])

    def matrix(self, prefix, rows=None):
        pointer, index, value = self.parts(prefix)
        if rows is not None:
            rows = rows.to(pointer.device)
            lengths = pointer[rows + 1] - pointer[rows]
            offset = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
            positions = (torch.arange(int(offset[-1]), device=rows.device)
                         + torch.repeat_interleave(pointer[rows] - offset[:-1], lengths))
            pointer, index, value = offset, index[positions], value[positions]
        return torch.sparse_csr_tensor(pointer, index, value,
            size=(len(pointer) - 1, len(self.views)), device=value.device)

    def coefficients(self, support, scale):
        support.validate(self.count)
        if not math.isfinite(float(scale)) or float(scale) <= 0:
            raise ValueError('invalid physical scale')
        # Retain the whole uncertain support, including repeated weighted rows.
        matrix = self.matrix('positive', support.indices)
        weights = support.weights.to(self.scales); weights = weights / weights.sum()
        association = torch.sparse.mm(matrix.transpose(0, 1), weights[:, None]).flatten()
        ratio = self.scales / float(scale)
        association = association * ((association >= .5) & (ratio >= .5) & (ratio <= 2))
        mass = self.scales.new_zeros(self.view_count).scatter_add_(0, self.views, association)
        return association / mass[self.views].clamp_min(1e-12)

    def masses(self, support, scale, rows=None):
        coefficient = self.coefficients(support, scale)
        return tuple(torch.sparse.mm(self.matrix(p, rows), coefficient[:, None]).flatten()
                     for p in ['positive', 'negative'])

    def refine(self, logits, support, scale, rows=None):
        positive, negative = self.masses(support, scale, rows)
        # One prior unit and at most one unit per supported view. This fixed
        # finite-evidence rule is not a calibrated Bayesian posterior.
        probability = (logits.sigmoid() + positive) / (1 + positive + negative)
        known = (positive + negative) > 0
        updated = torch.logit(probability.clamp(1e-6, 1 - 1e-6))
        return torch.where(known, updated, logits)


def enable(field, state):
    if field.evidence is not None or field.support_aggregation != 'normalized_prototype':
        raise ValueError('paired evidence requires an unaugmented prototype core')
    field.evidence = PairedEvidence(state, len(field.xyz)).to(field.xyz.device)
    field.evidence_kind = KIND
