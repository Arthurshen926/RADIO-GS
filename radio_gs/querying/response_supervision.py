"""Independent multimodal supervision contracts for intact response decisions.

Neither query modality nor provider name is a learner input. They only define
balanced training/calibration groups and the frozen incumbent to preserve.
"""
import numpy as np


def heldout_views(sensor, source_frames, quantiles, minimum_fraction):
    frames = np.asarray(source_frames, dtype=int)
    references = []; retained = np.ones(len(frames), bool)
    for q in quantiles:
        frame = int(frames[min(int(len(frames)*q), len(frames)-1)])
        if frame in references:
            continue
        _, excluded = sensor.exclude(references+[frame])
        legal = ~np.isin(frames, excluded)
        if legal.sum() >= max(5, int(np.ceil(len(frames)*minimum_fraction))):
            references.append(frame); retained = legal
    if not references:
        raise ValueError('no independent reference views while preserving mapping observations')
    _, excluded = sensor.exclude(references)
    if np.isin(frames[retained], excluded).any():
        raise AssertionError('reference leakage into retained observations')
    return references, np.flatnonzero(retained), excluded


def incumbent_first(row, provider):
    """Reindex every candidate-valued field, preserving the original identity."""
    first = next(i for i, (s, p) in enumerate(zip(row['seeds'], row['providers']))
                 if s == row['seeds'][0] and p == provider)
    order = [first] + [i for i in range(len(row['ious'])) if i != first]
    result = dict(row)
    for name in ('features', 'permuted_features', 'ious', 'seeds', 'providers', 'anchors'):
        if name in result:
            result[name] = [row[name][i] for i in order]
    return result


def coverage_reference_views(sensor, source_frames, visibility, labels, target_ids, config):
    """Use independent constructor labels, never teacher scores or predictions.

    Choose a small common held-out reference set covering annotated training
    targets, retaining at least half of original mapping views. Candidate views
    must leave other observations of the target; no held-out view is reinstated.
    """
    frames = np.asarray(source_frames, int)
    counts = np.array([np.asarray(visibility[labels == t].sum(0)).ravel() for t in target_ids])
    sizes = np.array([np.count_nonzero(labels == t) for t in target_ids])
    visible = counts >= np.maximum(20, .05*sizes[:, None])
    excluded_by_frame = {int(f): sensor.exclude([int(f)])[1] for f in frames}
    refs = []; kept = np.ones(len(frames), bool); covered = np.zeros(len(target_ids), bool)
    minimum = max(5, int(np.ceil(len(frames)*config['minimum_retained_view_fraction'])))
    for _ in range(config['reference_views']):
        best = None
        for i, frame in enumerate(frames):
            if frame in refs:
                continue
            legal = kept & ~np.isin(frames, excluded_by_frame[int(frame)])
            if legal.sum() < minimum:
                continue
            supported = (counts[:, legal] > 0).sum(1) >= config['minimum_target_mapping_views']
            admitted = visible[:, i] & supported
            new = admitted & ~covered
            # Prefer newly covered targets; view indices are a stable tie break.
            key = (int(new.sum()), float((counts[new, i]/np.maximum(sizes[new], 1)).sum()), -i)
            if new.any() and (best is None or key > best[0]):
                best = key, int(frame), legal, admitted
        if best is None:
            break
        _, frame, kept, admitted = best
        refs.append(frame); covered |= admitted
    if not refs:
        raise ValueError('no supported held-out reference views')
    _, excluded = sensor.exclude(refs)
    if np.isin(frames[kept], excluded).any():
        raise AssertionError('reference leakage into retained observations')
    return refs, np.flatnonzero(kept), excluded


def calibration_groups(rows):
    """A single shared threshold must protect every room AND input modality."""
    if any('modality' not in r or 'scene' not in r for r in rows):
        raise ValueError('explicit scene/modality calibration groups required')
    return [dict(r, scene=r['scene']+'|'+r['modality']) for r in rows]
