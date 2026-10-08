"""Typed category evidence over the same native and conditional-member field."""
import torch

from radio_gs.querying.observation_readout import positive_support_rank


def corroborated_category_scores(native_cosines, member_support):
    """Complete-menu native competition plus bounded positive member evidence.

    The native temperature100 is retained from v10's declared ScanNet prior,
    not fitted to this cohort. The v10 positive-support primitive preserves a
    native floor. Applying it to common-field category members is a new typed
    readout hypothesis, not a claim of reproducing the old category solver.

    Scores are rankings, not calibrated class probabilities. In particular,
    absence of member support does not erase native semantic evidence. Compute
    over the entire declared menu before selecting any reported class subset.
    """
    if (native_cosines.ndim != 2 or native_cosines.shape != member_support.shape
            or native_cosines.shape[1] < 2):
        raise ValueError('category evidence requires aligned complete-menu matrices')
    if not bool(torch.isfinite(native_cosines).all()) or not bool(torch.isfinite(member_support).all()):
        raise ValueError('category evidence must be finite')
    if bool(((member_support < 0) | (member_support > 1)).any()):
        raise ValueError('member support must lie in [0,1]')
    prior = (100 * native_cosines.float()).softmax(dim=1)
    return positive_support_rank(prior, member_support.float())
