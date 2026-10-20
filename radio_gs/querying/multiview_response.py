"""Equal-view response fusion on frozen geometry, with matched batched controls.

Columns are independent experiments. Source observations are averaged within
each column; a declared target observation has equal total weight to that source
group. Zero-confidence views and unsupported elements contribute no evidence.
"""
import torch

from .response_reconstruction import AdmittedResponseSystem


def normalize_view_confidence(confidence):
    if (confidence.ndim != 2 or not bool(torch.isfinite(confidence).all()) or
            bool(((confidence < 0) | (confidence > 1)).any())):
        raise ValueError('finite [0,1] confidence matrix required')
    mass = confidence.sum(0, keepdim=True)
    return confidence / mass.clamp_min(1e-8)


def fit_multiview_response(initial, source_terms, target_term=None, steps=120):
    """Solve independent source-only / source+target columns in one CSR pass."""
    if initial.ndim != 2 or not source_terms:
        raise ValueError('initial matrix and nonempty source observations required')
    count = torch.zeros((1, initial.shape[1]), dtype=initial.dtype)
    terms = []
    for op, y, c in source_terms:
        if y.shape != c.shape or y.shape != (op.shape[0], initial.shape[1]) or op.shape[1] != len(initial):
            raise ValueError('observation dimensions differ')
        if not bool(torch.isfinite(y).all()) or bool(((y < 0) | (y > 1)).any()):
            raise ValueError('finite [0,1] response required')
        known = normalize_view_confidence(c)
        count += (c.sum(0, keepdim=True) > 1e-8).to(initial.dtype)
        terms.append((op, y, known))
    # Keep the inherited solver's absolute numerical guards in pixel-mass units.
    # This common scale changes neither relative weights nor the update equation.
    scale = max([a.shape[0] for a, _, _ in terms] +
                ([] if target_term is None else [target_term[0].shape[0]]))
    observations = [(a, y, scale * c / count.clamp_min(1), 1.) for a, y, c in terms]
    if target_term is not None:
        a, y, c = target_term
        if y.shape != c.shape or y.shape != (a.shape[0], initial.shape[1]) or a.shape[1] != len(initial):
            raise ValueError('target observation dimensions differ')
        if not bool(torch.isfinite(y).all()) or bool(((y < 0) | (y > 1)).any()):
            raise ValueError('finite [0,1] target response required')
        observations.append((a, y, scale * normalize_view_confidence(c), 1.))
    system = AdmittedResponseSystem(initial, observations)
    system.terms = [(a.to_sparse_csr(), at.to_sparse_csr(), y, c) for a, at, y, c in system.terms]
    before = system.objective(initial, .25)
    value = system.rendered(steps=steps, residual_clip=.25)
    unsupported = system.mass <= 1e-8
    value[unsupported] = initial[unsupported]
    return value, dict(active_source_views=count.flatten().int().tolist(),
                      objective_before=before, objective_after=system.objective(value, .25),
                      unsupported_changed=int((value[unsupported] != initial[unsupported]).sum()),
                      target_confidence_mass=([0.] * initial.shape[1] if target_term is None else
                                              target_term[2].sum(0).tolist()))
