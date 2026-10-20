"""Choose a reference response using query evidence available at inference."""
import math


def reference_choices(records):
    """No scene identifier or target score enters either fixed policy.

    The primary repairs only a semantic response contradicting original signed
    annotations. The secondary uses SAM's source-image predicted mask quality.
    Both controls choose whole responses, with a deterministic semantic tie.
    """
    a, b = [records[k] for k in ['semantic_memory', 'constrained_memory']]
    p, n = a['positive_coverage'], a['negative_violation']
    if not (0 <= p <= 1 and 0 <= n <= 1):
        raise ValueError('invalid signed-query agreement')
    feasible = p == 1 and n == 0
    qa, qb = a['predicted_quality'], b['predicted_quality']
    if qa is None or qb is None:
        quality = 'constrained_memory' if not feasible else 'semantic_memory'
    else:
        if not all(math.isfinite(x) for x in [qa, qb]):
            raise ValueError('nonfinite source quality')
        quality = 'constrained_memory' if qb > qa else 'semantic_memory'
    return dict(feasible='semantic_memory' if feasible else 'constrained_memory', quality=quality)
