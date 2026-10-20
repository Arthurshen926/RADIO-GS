"""Estimate a response readout from source evidence, without target labels.

The criterion is a ratio of soft expected confusion counts, not expected IoU
and not a guarantee that teacher probabilities are calibrated. Histogram bins
give an explicitly quantized, deterministic threshold rule shared by queries.
"""
import torch


class SourceReadoutHistogram:
    def __init__(self, bins=512):
        if type(bins) is not int or bins < 2 or bins % 2:
            raise ValueError('an even number of at least two bins is required')
        self.bins = bins
        self.positive = torch.zeros(bins, dtype=torch.float64)
        self.negative = torch.zeros_like(self.positive)
        self.observations = 0

    def add(self, response, target, weight):
        response, target, weight = [x.detach().cpu().flatten().double()
                                    for x in (response, target, weight)]
        if response.shape != target.shape or target.shape != weight.shape:
            raise ValueError('response, target and weight shapes differ')
        if not all(bool(torch.isfinite(x).all()) for x in (response, target, weight)):
            raise ValueError('source evidence must be finite')
        if bool(((response < 0) | (response > 1) | (target < 0) | (target > 1) | (weight < 0)).any()):
            raise ValueError('probabilities must be in [0,1] and weights nonnegative')
        index = (response*self.bins).long().clamp_max(self.bins-1)
        self.positive.scatter_add_(0, index, weight*target)
        self.negative.scatter_add_(0, index, weight*(1-target))
        self.observations += int((weight > 0).sum())

    def result(self):
        tp = self.positive.flip(0).cumsum(0).flip(0)
        fp = self.negative.flip(0).cumsum(0).flip(0)
        total = float(self.positive.sum())
        scores = tp/(total+fp).clamp_min(1e-12)
        # Exclude zero: unsupported responses are zero and must stay negative.
        candidates = torch.arange(1, self.bins)
        if total <= 1e-12:
            index = self.bins//2
            reason = 'no_positive_source_mass'
        else:
            maximum = scores[candidates].max()
            tied = candidates[(scores[candidates]-maximum).abs() <= 1e-12]
            # Prefer the original readout for a tie; then the smaller bin index.
            index = int(tied[(tied-self.bins//2).abs().argmin()])
            reason = 'maximum_soft_source_iou'
        return dict(threshold=index/self.bins, bin_index=index, bins=self.bins,
                    criterion=reason, soft_iou=float(scores[index]),
                    soft_iou_at_half=float(scores[self.bins//2]),
                    positive_mass=total, negative_mass=float(self.negative.sum()),
                    observations=self.observations, positive_histogram=self.positive.tolist(),
                    negative_histogram=self.negative.tolist())
