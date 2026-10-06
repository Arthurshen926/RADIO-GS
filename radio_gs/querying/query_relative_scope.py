"""Experimental frozen-policy intervention, not the retained deployment recipe.

An admitted concept scope receives the difference between actual-query and
concept-prototype native logits. It tests information loss after routing without
fitting to evaluation labels. Parent admission is deliberately held fixed; a
positive result would still require source-family validation and recalibration.
"""
import torch


def add_query_difference(scope_logits, query_logits, prototype_logits, apply):
    if scope_logits.shape != query_logits.shape or scope_logits.shape != prototype_logits.shape:
        raise ValueError('Scope, query and prototype axes differ')
    if apply.shape != scope_logits.shape[1:] or apply.dtype != torch.bool:
        raise ValueError('Expected one boolean decision per query')
    return torch.where(apply[None],scope_logits+(query_logits-prototype_logits),scope_logits)


def install_query_relative_scope_control(field):
    if field.grounded_transfer is None or field.native_decoder is None:
        raise ValueError('Native decoder and grounded scope required')
    if field.bounded_extent is not None or field.source_membership is not None:
        raise ValueError('The controlled grounded branch must be active')
    original=field.logits
    def logits(query,modality,rows=None):
        if modality!='text':return original(query,modality,rows)
        capture={}
        def head_hook(module,inputs,output):capture['eligible']=output[1]
        hooks=[field.grounded_transfer.register_forward_hook(head_hook)]
        if field.extent_benefit_admission is not None:
            def gate_hook(module,inputs,output):capture['allowed']=output
            hooks.append(field.extent_benefit_admission.register_forward_hook(gate_hook))
        try:parent=original(query,modality,rows)
        finally:
            for hook in hooks:hook.remove()
        index,concept,_=field.grounded_transfer.route(query)
        apply=concept & capture['eligible'] & capture.get('allowed',torch.ones_like(concept))
        if not apply.any():return parent
        prototype=field.grounded_transfer.prototypes[index]
        qcos=field.native_cosine(query,rows,modality='text')
        pcos=field.native_cosine(prototype,rows,modality='text')
        if 'text' in field.native_query_thresholds:
            scale=field.readout_log_scale['text'].clamp(-4.,6.).exp() if 'text' in field.readout_log_scale else 10.
            qlog=scale*(qcos-field.native_query_thresholds['text'](query)[None])
            plog=scale*(pcos-field.native_query_thresholds['text'](prototype)[None])
        else:qlog=field.cosine_logits(qcos,'text');plog=field.cosine_logits(pcos,'text')
        return add_query_difference(parent,qlog,plog,apply)
    field.logits=logits
    return lambda:setattr(field,'logits',original)
