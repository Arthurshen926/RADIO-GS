from itertools import product
import numpy as np
from scripts.research.audit_query_field_monotone_capacity import bounds


def test_oracle_bounds_equal_exhaustive_monotone_mask_optima():
    masks = [np.array(v, dtype=bool) for v in product([False, True], repeat=4)]
    for prediction in masks:
        for target in masks:
            result = bounds(prediction, target)
            additive = [p for p in masks if np.all(~prediction | p)]
            subtractive = [p for p in masks if np.all(~p | prediction)]
            def iou(p): return np.count_nonzero(p & target) / max(np.count_nonzero(p | target), 1)
            assert result['addition_only_pixel_ceiling'] == max(map(iou, additive))
            assert result['deletion_only_pixel_ceiling'] == max(map(iou, subtractive))
