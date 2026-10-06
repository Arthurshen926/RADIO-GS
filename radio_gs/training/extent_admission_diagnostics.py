"""Diagnostic counts for scope takeover; does not change selection or routing."""
import torch


def admission_diagnostics(native, extent, allowed, tolerance=1e-6):
    if native.ndim != 1 or extent.shape != native.shape or allowed.shape != native.shape:
        raise ValueError('One native/extent utility and admission per query required')
    if allowed.dtype != torch.bool or not torch.isfinite(native).all() or not torch.isfinite(extent).all():
        raise ValueError('Finite utilities and boolean admissions required')
    if tolerance < 0:
        raise ValueError('Tolerance must be nonnegative')
    gain = extent - native
    harmful, useful = gain < -tolerance, gain > tolerance
    harmful_takeover, missed = harmful & allowed, useful & ~allowed
    deployed = torch.where(allowed, extent, native)
    oracle = torch.maximum(native, extent)
    count = len(native)
    return dict(
        queries=count, allowed=int(allowed.sum()),
        policy='empty' if not count else 'always_extent' if bool(allowed.all())
        else 'always_native' if not bool(allowed.any()) else 'selective',
        harmful_candidates=int(harmful.sum()), useful_candidates=int(useful.sum()),
        harmful_takeovers=int(harmful_takeover.sum()), missed_improvements=int(missed.sum()),
        harmful_takeover_rate=float(harmful_takeover.sum()/harmful.sum()) if harmful.any() else None,
        missed_improvement_rate=float(missed.sum()/useful.sum()) if useful.any() else None,
        native_regret_sum=float((-gain[harmful_takeover]).sum()),
        missed_gain_sum=float(gain[missed].sum()),
        oracle_regret_mean=float((oracle-deployed).mean()) if count else None,
        independent_validation=False,
    )
