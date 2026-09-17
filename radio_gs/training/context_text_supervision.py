"""Abstaining context text supervision on existing source correspondence edges."""
from radio_gs.training.text_task_aggregation import canonical_text_tasks


def context_text_tasks(episodes, proposal_views, labels, accepted):
    """Only agreement at BOTH endpoints licenses a text-positive region.

    Never invent cross-view identities or turn unaccepted regions into negatives.
    Validation episodes are ignored and the input episodes are not mutated.
    """
    if len(labels)!=len(proposal_views) or len(accepted)!=len(proposal_views):
        raise ValueError('context evidence region axes differ')
    records=[]
    for index,e in enumerate(episodes):
        if e['split']!='train' or e['modality']!='image':continue
        q,t=e['query_region'],e['target_region']
        if not accepted[q] or not accepted[t] or labels[q]!=labels[t]:continue
        if labels[q]<0:raise ValueError('accepted evidence has no text label')
        negatives=[r for r in e['negative_regions'] if accepted[r] and labels[r]>=0 and labels[r]!=labels[q]]
        records.append((index,{**e,'modality':'text','text_index':int(labels[q]),'positive_regions':[t],'negative_regions':negatives}))
    return canonical_text_tasks(records)
