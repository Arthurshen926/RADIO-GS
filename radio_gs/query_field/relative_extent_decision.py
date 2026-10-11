"""Use uncalibrated evidence strength relatively, with matched cost units."""
import numpy as np
from .extent_decision_linked import LinkedExtentDecision


class RelativeEvidenceExtentDecision(LinkedExtentDecision):
    def solve(self, initial, target, positive, negative, confidence, cost, *, max_passes=4):
        confidence = np.asarray(confidence, dtype=np.float64)
        if not np.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
            raise ValueError('invalid view confidence')
        active = confidence > 0
        if not active.any():
            return super().solve(initial,target,positive,negative,confidence,cost,max_passes=max_passes)
        maximum = float(confidence.max())
        relative = confidence / maximum
        factor = float(relative[active].mean())
        # The inherited kernel minimizes factor * [sum(c*loss)/sum(c) + .1*cost].
        # Its data and cost coefficients remain within the established ranges.
        # Global rescaling of confidence cancels before bounded binary decisions.
        selected, info = super().solve(initial,target,positive,negative,relative,
                                       np.asarray(cost,dtype=np.float64)*factor,max_passes=max_passes)
        info.update(relative_evidence=True,confidence_maximum=maximum,cost_factor=factor,
                    weighted_mean_energy_trace=(np.asarray(info['energy_trace'])/factor).tolist())
        return selected,info
