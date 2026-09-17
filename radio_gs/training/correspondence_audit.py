"""Source-only diagnostics for pairwise identity and text pseudo supervision."""
from collections import defaultdict


def audit_correspondence_graph(episodes, proposal_views):
    """Report contradictory transitive tracks without inventing object labels.

    Reciprocal pair matches do not guarantee that their connected component
    denotes one physical instance. Two regions from the same source view in a
    component violate the one-region-per-view track assumption.
    """
    result={}
    for split in ('train','validation'):
        edges=sorted({tuple(sorted((int(e['query_region']),int(e['target_region']))))
                      for e in episodes if e['split']==split and e['modality']=='image'})
        neighbors=defaultdict(set)
        for left,right in edges:
            if left<0 or right<0 or max(left,right)>=len(proposal_views):
                raise ValueError('correspondence region outside source evidence')
            neighbors[left].add(right);neighbors[right].add(left)
        visited=set();components=[]
        for start in sorted(neighbors):
            if start in visited:continue
            stack=[start];members=[]
            while stack:
                region=stack.pop()
                if region in visited:continue
                visited.add(region);members.append(region);stack.extend(neighbors[region]-visited)
            by_view=defaultdict(list)
            for region in sorted(members):by_view[int(proposal_views[region])].append(region)
            conflicts={str(view):regions for view,regions in by_view.items() if len(regions)>1}
            components.append({'regions':sorted(members),'views':sorted(by_view),
                               'same_view_conflicts':conflicts,'track_consistent':not conflicts})
        result[split]={'unique_pairs':len(edges),'matched_regions':len(neighbors),
                       'components':components,'inconsistent_components':sum(not c['track_consistent'] for c in components)}
    return result


def training_query_conflicts(episodes, region_probability):
    """Find opposite hard labels for the same query/element across train views."""
    import torch
    grouped=defaultdict(list)
    for episode in episodes:
        if episode['split']=='train':
            key=(episode['modality'],episode.get('text_index',episode['query_region']))
            grouped[key].append(episode)
    conflicts={}
    for key,group in grouped.items():
        positive=torch.zeros(region_probability.shape[1],device=region_probability.device,dtype=torch.bool)
        negative=torch.zeros_like(positive)
        for episode in group:
            pos=region_probability[episode.get('positive_regions',[episode['target_region']])].amax(0)>.7
            neg=(region_probability[episode['negative_regions']].amax(0)>.7
                 if episode['negative_regions'] else torch.zeros_like(pos))
            known=pos^neg;positive|=pos&known;negative|=neg&known
        ambiguous=positive&negative
        if ambiguous.any():conflicts[key]=ambiguous
    return conflicts
