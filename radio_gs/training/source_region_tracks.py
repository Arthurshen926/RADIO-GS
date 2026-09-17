"""Conservative global consistency for pairwise source-region positives.

An accepted graph cannot identify two different scopes in the same view.
This is a necessary condition, not proof of physical instance correspondence.
Rejected links are unknown, never automatically negatives. No transitive links
are emitted as new supervision.
"""
import math
import torch


def consistent_region_matches(matches, masks, views, *, duplicate_iou=0.9):
    """Keep strongest compatible links, permitting near-duplicate same-view masks.

    Call separately per train/validation split. ``matches`` may contain both
    directions; their maximum confidence determines deterministic edge ordering.
    All same-view pairs in a proposed component must pass the duplicate test,
    so duplicate equivalence is not incorrectly assumed to be transitive.
    """
    if masks.ndim != 3 or masks.dtype != torch.bool or views.shape != (len(masks),):
        raise ValueError('expected R,H,W boolean masks and R view identities')
    if not 0 < duplicate_iou <= 1:
        raise ValueError('invalid duplicate IoU')
    views = views.cpu().tolist()
    flat = masks.flatten(1).float()
    intersection = flat @ flat.T
    area = flat.sum(1)
    ious = (intersection / (area[:, None] + area[None, :] - intersection).clamp_min(1)).cpu()
    parent = list(range(len(masks)))
    members = {i: [i] for i in parent}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = {}
    for a, b, score in matches:
        if not (0 <= a < len(masks) and 0 <= b < len(masks)) or a == b or views[a] == views[b] or not math.isfinite(score):
            raise ValueError('invalid cross-view match')
        key = tuple(sorted((a, b)))
        edges[key] = max(edges.get(key, -math.inf), score)
    accepted, rejected = set(), []
    for (a, b), score in sorted(edges.items(), key=lambda item: (-item[1], item[0])):
        ra, rb = find(a), find(b)
        if ra == rb:
            accepted.add((a, b))
            continue
        conflict = next(((x, y) for x in members[ra] for y in members[rb]
                         if views[x] == views[y] and float(ious[x, y]) < duplicate_iou), None)
        if conflict is not None:
            rejected.append(dict(regions=[a, b], confidence=score,
                                 conflicting_regions=list(conflict),
                                 same_view_iou=float(ious[conflict]), reason='incompatible_same_view_scope'))
            continue
        parent[rb] = ra
        members[ra].extend(members.pop(rb))
        accepted.add((a, b))
    retained = [(a, b, score) for a, b, score in matches if tuple(sorted((a, b))) in accepted]
    groups = sorted([sorted(group) for group in members.values() if len(group) > 1])
    return retained, dict(candidate_edges=len(edges), accepted_edges=len(accepted), rejected=rejected,
                          groups=groups, duplicate_iou=duplicate_iou,
                          physical_identity_verified=False, transitive_supervision_added=False,
                          rejected_links_are_unknown=True)
