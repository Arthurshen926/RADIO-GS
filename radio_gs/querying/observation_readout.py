"""Native identity followed by task-specific element readout.

The same numerical identity/projection primitives accept NumPy/SciPy and Torch
carriers. Encoders retain their own coordinates. Category competition, single
instance scope and rendered set inference remain explicit task choices.
"""


def native_identity(descriptors, queries):
    """Inputs are already normalized in the same native teacher space."""
    return descriptors @ queries.T


def project_observations(operator, values):
    """Project evidence with the declared mesh or rendering-derived operator."""
    return operator @ values


def positive_support_rank(native_rank, support):
    """Bounded support in [0,1], retaining the native floor when support is absent.

    This preserves coordinates, not every pairwise ranking; it is not a
    calibrated foreground or identity probability.
    """
    return native_rank + native_rank * (1 - native_rank) * support


def corroborate_identity(native, member, coverage):
    """Conservative native-space identity corroboration, never a mask weight.

    The lower native cosine applies where other views observe the region's
    members. No independent observation retains the original identity exactly.
    Confidence is not calibrated and this cannot guarantee the right instance.
    Works on NumPy and Torch arrays without changing encoder coordinates.
    """
    difference=native-member
    return native-coverage[:,None]*(difference+abs(difference))*.5


def mean_native_identity_ranks(first, second, available):
    """Equal ordinal evidence from two declared native descriptor channels.

    Coordinates and cosine scales are not mixed. Only supported candidates
    compete, with average ranks for ties. This is an identity-selection rule,
    not a calibrated foreground probability or a change to the semantic field.
    """
    import numpy as np
    from scipy.stats import rankdata
    result=np.zeros_like(first)
    result[available]=(rankdata(first[available],axis=0)+rankdata(second[available],axis=0))/(2*len(available))
    return result
