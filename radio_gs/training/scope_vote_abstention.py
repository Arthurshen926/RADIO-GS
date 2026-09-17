"""Local supervision abstention from source-only, bidirectional scope relations."""
import torch


def scope_vote_abstention(probability, mass, frames, relations):
    if probability.ndim!=2 or mass.shape!=probability.shape or len(frames)!=len(probability):
        raise ValueError('Source views and element evidence axes differ')
    if len(set(frames))!=len(frames):raise ValueError('Duplicate source frame')
    if not torch.isfinite(probability).all() or not torch.isfinite(mass).all() or (mass<0).any():
        raise ValueError('Finite probabilities and nonnegative mass required')
    lookup={f:i for i,f in enumerate(frames)}
    strong=(mass>=.1*mass.amax(0,keepdim=True))&(mass>1e-6)
    positive=probability>.8;negative=probability<.2
    containment=torch.zeros_like(strong);incompatible=torch.zeros_like(strong)
    for row in relations:
        if row['source'] not in lookup or row['target'] not in lookup:
            raise ValueError('Relation endpoint is outside the allowed source split')
        a,b=lookup[row['source']],lookup[row['target']];both=strong[a]&strong[b];kind=row['relation']
        if kind=='source_contains_target':containment[b]|=both&positive[a]&negative[b]
        elif kind=='source_part_of_target':containment[a]|=both&negative[a]&positive[b]
        elif kind=='incompatible':
            conflict=both&((positive[a]&negative[b])|(negative[a]&positive[b]))
            incompatible[a]|=conflict;incompatible[b]|=conflict
        elif kind not in ('support','unknown'):raise ValueError('Unknown scope relation')
    result={}
    for name,remove in [('containment_abstention',containment),('incompatible_abstention',incompatible),('combined_abstention',containment|incompatible)]:
        weights=mass.masked_fill(remove,0);total=weights.sum(0)
        result[name]=dict(probability=(probability*weights).sum(0)/total.clamp_min(1e-8),mass=total,removed_votes=int(remove.sum()),removed_mass=float(mass[remove].sum()))
    return result
