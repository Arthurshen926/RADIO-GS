"""Measure observable utility evidence for intact responses, without labels.

This is an attribution control, not a trained replacement policy. Source masks
were also consumed by the candidates; their agreement is not held-out accuracy.
"""
from dataclasses import replace

import numpy as np
import torch


class CompleteResponseAgreement:
    """Equal-view authorized Brier agreement, preserving the decision unit.

    A query response may be chosen independently; an exclusive category menu
    must be chosen intact. Unknown observations never contribute to the loss.
    """
    def __init__(self, responses, *, whole_menu=False):
        if (responses.ndim != 3 or len(responses) < 2 or not torch.isfinite(responses).all()
                or (responses < 0).any() or (responses > 1).any()):
            raise ValueError('bounded candidate-by-element-by-query responses required')
        self.responses = responses
        self.whole_menu = whole_menu
        self.rows = []

    @torch.no_grad()
    def add(self, item, *, permutation=None):
        positive, negative, confidence = item.weights(self.responses[0])
        target = item.target
        if permutation is not None:
            permutation = torch.as_tensor(permutation, device=target.device, dtype=torch.long)
            if (permutation.shape != (len(target),) or
                    not torch.equal(permutation.sort().values, torch.arange(len(target), device=target.device))):
                raise ValueError('control must jointly permute all spatial evidence rows')
            target = target[permutation]
            positive, negative, confidence = positive[permutation], negative[permutation], confidence[permutation]
        weight = (positive | negative) * confidence
        mass = weight.sum(0)
        losses = []
        for response in self.responses:
            rendered = torch.sparse.mm(item.operator, response)
            loss = (weight * (rendered-target).square()).sum(0) / mass.clamp_min(1e-8)
            losses.append(loss)
        losses = torch.stack(losses)
        valid = mass > 0
        if self.whole_menu:
            # Pool the COMPLETE menu before making any candidate choice.
            losses = (losses[:, valid].mean(1, keepdim=True) if valid.any()
                      else losses.new_zeros((len(losses), 1)))
            valid = valid.any().reshape(1)
        self.rows.append((losses.cpu().numpy(), valid.cpu().numpy()))

    def finish(self):
        units = 1 if self.whole_menu else self.responses.shape[2]
        if not self.rows:
            return dict(gains=np.zeros((len(self.responses), units)).tolist(),
                        choices=[0]*units, views=[0]*units, positive_view_fraction=[0.]*units)
        losses = np.stack([r[0] for r in self.rows])
        valid = np.stack([r[1] for r in self.rows])
        delta = losses[:, :1]-losses
        count = valid.sum(0)
        gains = (delta*valid[:, None]).sum(0)/np.maximum(count, 1)
        gains[0] = 0
        choices = gains.argmax(0)  # exact ties retain incumbent zero
        selected = delta[:, choices, np.arange(units)]
        return dict(gains=gains.tolist(), choices=choices.tolist(), views=count.tolist(),
                    positive_view_fraction=((selected > 0)*valid).sum(0).__truediv__(np.maximum(count, 1)).tolist())


def whole_response_metrics(candidate_metrics, choices, records):
    """Select cached metrics by query identity; never select by frame or metric.

    records is the baseline list of frame/query observations. Every candidate
    must have the same exact record keys. The choice mapping contains one index
    per query, so AP and IoU always belong to the same intact answer.
    """
    keys = [(r['frame'], r['query']) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate frame/query scoring record')
    if set(choices) != {r['query'] for r in records}:
        raise ValueError('choices must cover exactly the scored query identities')
    indexed = []
    for rows in candidate_metrics:
        lookup = {(r['frame'], r['query']): r for r in rows}
        if len(lookup) != len(rows) or set(lookup) != set(keys):
            raise ValueError('candidate scoring record sets differ')
        indexed.append(lookup)
    selected = [indexed[choices[q]][(f, q)] for f, q in keys]
    return {k: float(np.mean([r[k] for r in selected])) for k in ('iou', 'ap')}
