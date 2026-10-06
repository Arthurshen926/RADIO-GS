"""One-hop scope evidence with explicit source provenance and heldout exclusion.

Geometric links support correspondence hypotheses, not verified instance IDs.
Missing detections are unknown. Existing phrase detections are never overwritten
or enlarged by propagation; conflicting destinations are left for review.
"""


def linked_phrase_regions(records, links, query_index, heldout_frames=()):
    heldout = set(heldout_frames)
    originals = {
        i for i, record in enumerate(records)
        if record['frame'] not in heldout
        and query_index in record['phrase_indices']
        and record['confidence'] >= .7
    }
    detected_frames = {records[i]['frame'] for i in originals}
    propagated = {}
    for link_index, link in enumerate(links):
        for side in ('left', 'right'):
            for endpoint in link[side + '_regions']:
                if records[endpoint]['frame'] != link[side + '_frame']:
                    raise ValueError('Link endpoint does not match evidence frame')
        if link['relation'] != 'mutual_support_candidate':
            continue
        if {link['left_frame'], link['right_frame']} & heldout:
            continue
        for source_key, target_key in [('left', 'right'), ('right', 'left')]:
            source = set(link[source_key + '_regions']) & originals
            if not source or link[target_key + '_frame'] in detected_frames:
                continue
            for target in link[target_key + '_regions']:
                propagated.setdefault(target, []).append({
                    'source_regions': sorted(source), 'link_index': link_index,
                    'relation': 'mutual_support_candidate',
                    'physical_identity_verified': False,
                })
    return sorted(originals), propagated
