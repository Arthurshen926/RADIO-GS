"""Inherit a source category gate only through an explicitly validated head refit."""
import math
from radio_gs.querying.source_accepted_category import source_accepted_category_logits


def source_refitted_category_logits(native,scope,*,scope_receipt,parent_admission,
                                    refit_receipt,field_sha256,max_correction=1.0):
    r=refit_receipt
    if r.get('schema')!='registered_visibility_qualified_selected_scope.v1':
        raise ValueError('Registered source scope-refit receipt required')
    if r.get('mode') not in ('selected','visible_selected'):
        raise ValueError('Unknown source refit mode')
    if r.get('source_only') is not True or r.get('benchmark_data_used') is not False:
        raise ValueError('Source-only refit required')
    if any(not isinstance(v,str) or len(v)!=64 for v in
           (field_sha256,r.get('field_sha256'),r.get('parent_field_sha256'),parent_admission.get('field_sha256'))):
        raise ValueError('Explicit field digests required')
    if r.get('field_sha256')!=field_sha256 or r.get('parent_field_sha256')!=parent_admission.get('field_sha256'):
        raise ValueError('Refit field or parent ancestry differs')
    if any(r.get(k) is not True for k in ('geometry_unchanged','native_field_unchanged','query_encoder_unchanged')):
        raise ValueError('Refit cannot change geometry, native state or encoder')
    chosen=r.get('selected_step')
    if type(chosen) is not int or chosen<0 or chosen>r['steps']:
        raise ValueError('Invalid source-selected step')
    history=r['history']
    if not history or history[0]['step']!=0:
        raise ValueError('Missing step-zero comparison')
    selected=[x['validation'] for x in history if x['step']==chosen]
    if len(selected)!=1:
        raise ValueError('Selected checkpoint is not unique in source history')
    initial=history[0]['validation'];metric=selected[0]
    if type(initial.get('observations')) is not int or initial['observations']<1 or metric.get('observations')!=initial['observations']:
        raise ValueError('Refit validation cohort differs')
    if any(not math.isfinite(float(m[k])) for m in (initial,metric) for k in ('full_iou','selected_iou')):
        raise ValueError('Nonfinite source validation')
    if metric['full_iou']<initial['full_iou']-1e-6 or (chosen and metric['selected_iou']<=initial['selected_iou']+1e-6):
        raise ValueError('Refit violates joint source acceptance')
    result,decision=source_accepted_category_logits(native,scope,scope_receipt=scope_receipt,
        admission_receipt=parent_admission,max_correction=max_correction)
    return result,dict(decision,refit_step=chosen,refit_mode=r['mode'],parent_scope_policy_retained=True)
