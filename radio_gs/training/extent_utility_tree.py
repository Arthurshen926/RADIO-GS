"""Bounded source-only decisions maximizing measured extent-over-native utility."""
import torch


def fit_extent_utility_tree(features, gain, weight, query_groups, *, minimum_groups=5):
    if features.ndim!=2 or gain.shape!=features.shape[:1] or weight.shape!=gain.shape or query_groups.shape!=gain.shape:
        raise ValueError('Training axes differ')
    if not torch.isfinite(features).all() or not torch.isfinite(gain).all() or not torch.isfinite(weight).all() or (weight<0).any():
        raise ValueError('Finite training values and nonnegative weights required')
    feature=torch.full((7,),-1,dtype=torch.long);cutoff=torch.zeros(7);decision=torch.zeros(7,dtype=torch.bool)
    def grow(ids,node,depth):
        values=gain[ids]*weight[ids];total=values.sum();decision[node]=total>0
        if depth==2 or len(query_groups[ids].unique())<2*minimum_groups:return
        best=max(0.,float(total));chosen=None
        for col in range(features.shape[1]):
            order=features[ids,col].argsort(stable=True);rows=ids[order];x=features[rows,col];prefix=(gain[rows]*weight[rows]).cumsum(0)
            utility=prefix.clamp_min(0)+(total-prefix).clamp_min(0)
            for at in torch.where(x[:-1]<x[1:])[0].tolist():
                if float(utility[at])<=best+1e-8:continue
                left,right=rows[:at+1],rows[at+1:]
                if min(len(query_groups[left].unique()),len(query_groups[right].unique()))<minimum_groups:continue
                best=float(utility[at]);chosen=(col,(x[at]+x[at+1])/2,left,right)
        if chosen is None:return
        col,value,left,right=chosen;feature[node]=col;cutoff[node]=value
        grow(left,2*node+1,depth+1);grow(right,2*node+2,depth+1)
    grow(torch.arange(len(gain)),0,0)
    return dict(feature=feature,cutoff=cutoff,decision=decision)
