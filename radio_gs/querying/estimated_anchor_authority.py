"""Isolate an estimated anchor's forced-positive assignment from its identity.

The complete frozen classifier and its conditioning anchor are unchanged. This
control changes only the explicit override at one image-estimated vertex; it
does not claim to solve identity uncertainty for the remaining member response.
"""
import numpy as np


def estimated_anchor_patch(reader, detail, retained_response, retained_selection):
    """Return a sparse intervention, without modifying either retained array."""
    if detail['modality'] != 'image':
        raise ValueError('only estimated image anchors qualify; actual points are authoritative')
    context = detail['inferred_anchor_membership']
    if context['anchor_identity_guaranteed'] is not False:
        raise ValueError('estimated-anchor uncertainty must be explicit')
    anchor = int(context['anchor_vertex'])
    threshold = float(reader.saved['threshold'])
    if not np.isclose(threshold, context['threshold'], rtol=0, atol=1e-12):
        raise ValueError('frozen classifier threshold changed')
    if retained_response.ndim != 1 or retained_selection.shape != retained_response.shape:
        raise ValueError('expected aligned one-dimensional retained outputs')
    if not 0 <= anchor < len(retained_response):
        raise ValueError('anchor outside the complete carrier')
    if retained_response[anchor] != 1. or not retained_selection[anchor]:
        raise ValueError('retained output does not have the declared positive override')
    features = reader.evidence.encode(anchor, np.array([anchor], dtype=np.int64))
    # Match infer's float32 conversion, but omit its probability[anchor] = 1.
    probability = float(reader.saved['estimator'].predict_proba(features)[:, 1].astype(np.float32)[0])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('classifier returned an invalid probability')
    return dict(query_id=detail['query_id'], anchor_vertex=anchor, threshold=threshold,
                baseline_probability=1., unforced_probability=probability,
                baseline_selected=True, unforced_selected=bool(probability > threshold),
                rank_entries_changed=int(probability != 1.),
                selection_entries_changed=int(probability <= threshold),
                other_vertices_unchanged=True, conditioning_anchor_unchanged=True)


def apply_patch(response, selection, patch):
    """Reconstruct the tiny controlled change from a sealed baseline and patch."""
    response, selection = response.copy(), selection.copy()
    anchor = patch['anchor_vertex']
    if response[anchor] != patch['baseline_probability'] or not selection[anchor]:
        raise ValueError('baseline no longer matches the sealed sparse intervention')
    response[anchor] = patch['unforced_probability']
    selection[anchor] = patch['unforced_selected']
    return response, selection
