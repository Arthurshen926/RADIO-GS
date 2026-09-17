"""Categorical native text queries for point-domain OVS; no GT-dependent query set."""
import torch


def classify_elements(field,queries,class_ids):
    ids=torch.as_tensor(class_ids,device=queries.device)
    if ids.ndim!=1 or ids.dtype not in (torch.int32,torch.int64) or len(ids)!=len(queries) or not len(ids) or (ids<=0).any() or ids.unique().numel()!=len(ids):raise ValueError('unique positive raw semantic class IDs required')
    logits=field.logits(queries,'text')
    labels=ids[logits.argmax(1)]
    return {'raw_class_ids':torch.where(field.observed,labels,torch.zeros_like(labels)),'observed':field.observed,'logits':logits}
