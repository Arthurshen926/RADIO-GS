"""Partition same-name source scopes without assuming name implies equal extent."""
from itertools import combinations


def scope_pair_relation(forward, reverse):
    """Forward source -> target; absent or asymmetric evidence stays unknown."""
    if forward is None or reverse is None:
        return 'unknown'
    if min(forward['visible_source_pixels'], forward['visible_target_pixels'],
           reverse['visible_source_pixels'], reverse['visible_target_pixels']) < 20:
        return 'unknown'
    if forward['iou'] >= .5 and reverse['iou'] >= .5:
        return 'support'
    def contains(row):
        return row['recall'] >= .8 and row['precision'] < .5
    def part(row):
        return row['precision'] >= .8 and row['recall'] < .5
    if contains(forward) and part(reverse):
        return 'source_contains_target'
    if part(forward) and contains(reverse):
        return 'source_part_of_target'
    if forward['iou'] <= .1 and reverse['iou'] <= .1:
        return 'incompatible'
    return 'unknown'


def partition_source_scopes(group, training_frames, records, *, allow_unknown_bridges=False):
    """Complete-link clusters; never bridge an unsupported pair transitively.

    With allow_unknown_bridges, supported paths may join unknown pairs, but
    explicit containment/incompatibility always blocks merging. Unknown edges
    retain their label; this is a scope hypothesis, not verified identity.

    Only rows with both endpoints in the explicitly allowed source split enter
    decisions. Containment is retained as a relation, never equality.
    """
    frames=sorted(set(training_frames));allowed=set(frames)
    rows={(r['source_frame'],r['target_frame']):r for r in records
          if r['group']==group and r['source_frame'] in allowed and r['target_frame'] in allowed}
    relations={};quality={}
    for left,right in combinations(frames,2):
        relation=scope_pair_relation(rows.get((left,right)),rows.get((right,left)))
        relations[left,right]=relation
        if relation=='support':quality[left,right]=min(rows[left,right]['iou'],rows[right,left]['iou'])
    clusters=[(f,) for f in frames]
    while True:
        candidates=[]
        for i,j in combinations(range(len(clusters)),2):
            pairs=[tuple(sorted((a,b))) for a in clusters[i] for b in clusters[j]]
            eligible=(all(relations[p] in ('support','unknown') for p in pairs) and any(relations[p]=='support' for p in pairs) if allow_unknown_bridges else all(relations[p]=='support' for p in pairs))
            if eligible:
                merged=tuple(sorted(clusters[i]+clusters[j]))
                candidates.append((min(quality[p] for p in pairs if p in quality),len(merged),tuple(-v for v in merged),i,j,merged))
        if not candidates:break
        _,_,_,i,j,merged=max(candidates)
        clusters=[v for k,v in enumerate(clusters) if k not in (i,j)]+[merged]
        clusters.sort()
    return dict(clusters=[list(c) for c in clusters],relations=[dict(source=a,target=b,relation=r) for (a,b),r in relations.items()],training_frames=frames)
