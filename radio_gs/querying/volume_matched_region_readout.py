"""Use the existing field's query extent as a count prior for a new ranking.

This only shifts logits, so ranking and AP of the proposed evidence are
unchanged. Ties in the proposed ranking remain ties; callers record that case.
"""
import torch


def match_element_count(proposed, parent, observed):
    if proposed.shape != parent.shape or proposed.ndim != 2 or observed.shape != (len(proposed),):
        raise ValueError('proposed/parent/observed domains must align')
    if observed.dtype != torch.bool:
        raise ValueError('observed must be Boolean')
    result=parent.clone()
    diagnostics=[]
    idx=torch.where(observed)[0]
    n=len(idx)
    for j in range(proposed.shape[1]):
        k=int(((parent[:,j]>.5)&observed).sum())
        if not 0<k<n:
            diagnostics.append(dict(query=j,parent_count=k,new_count=k,threshold=None,abstained=True,tie=False))
            continue
        values=torch.logit(proposed[idx,j].clamp(1e-6,1-1e-6))
        top=torch.topk(values,k+1,sorted=True).values
        hi=top[k-1];lo=top[k]
        threshold=(hi+lo)/2
        changed=torch.sigmoid(values-threshold)
        result[idx,j]=changed
        new_count=int((changed>.5).sum())
        diagnostics.append(dict(query=j,parent_count=k,new_count=new_count,threshold=float(threshold),abstained=False,tie=bool(hi==lo)))
    return result,diagnostics
