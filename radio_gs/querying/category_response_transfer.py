"""Intact category responses on the original benchmark Gaussian carrier.

The incumbent is replay-checked then retained byte for byte. Native region
support comes from the stored nonzero alpha transfer; its per-view union is a
region-covered visibility domain, not all geometrically visible Gaussians.
This evidence-domain shift from independent mesh supervision is explicit.
"""
import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F

from .native_observation_prior import compiled_observation_prior
from .local_element_fusion import compile_local_evidence
from .response_replacement import RegionResponseEvidence
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS as SPLITS


def assert_shared_carrier(first, second):
    if first['shape'] != second['shape'] or first['geometry'] != second['geometry']:
        raise ValueError('native teachers do not share the original response carrier')
    for key in ('crow','columns','weights','supported'):
        if not torch.equal(first[key], second[key]):
            raise ValueError('native teacher region transfer changed: '+key)


def replay_menu(original, reconstructed, class_ids, observed):
    delta=(original-reconstructed).abs()
    changed={}
    for name, ids in SPLITS.items():
        columns=[class_ids.index(c) for c in ids]
        changed[name]=int(((original[:,columns].argmax(1)!=reconstructed[:,columns].argmax(1))&observed).sum())
    report=dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()),changed_subset_labels=changed)
    if report['max_abs']>2e-5 or report['mean_abs']>1e-6 or any(changed.values()):
        raise ValueError('original category replay changed response or output labels: '+str(report))
    return report


@torch.inference_mode()
def complete_menu_candidates(data, scene, auxiliary_scene, auxiliary_readout, field, auxiliary_field):
    assert_shared_carrier(field, auxiliary_field)
    if (data['texts'] != auxiliary_readout['texts'] or len(data['texts']) != 19
            or scene.get('local_element_checkpoint') or scene['local_element_control'] != 'average_prior'
            or field['geometry'] != data['semantic_geometry']):
        raise ValueError('unexpected original complete-menu provider contract')
    class_ids=[q['class_id'] for q in data['queries']]
    if class_ids != list(SPLITS['19']):
        raise ValueError('complete 19-class order changed')
    initial=data['native_logits']; observed=data['semantic_observed']
    # Keep original CUDA normalization/matmul/softmax. CPU accumulation changed
    # a tied 10-class prior argmax in the unscored v1 preparation.
    query=data['encodings'].float().cuda(); auxiliary_query=auxiliary_readout['encodings'].float().cuda()
    primary, _=compiled_observation_prior(scene,query,observed.cuda(),initial.cuda(),field=field)
    primary_check=replay_menu(initial.sigmoid(),primary.cpu().sigmoid(),class_ids,observed)
    auxiliary_scene=dict(auxiliary_scene,native_observation_reliability=scene['native_observation_reliability'])
    auxiliary_scene.pop('observation_weighting',None)
    auxiliary, _=compiled_observation_prior(auxiliary_scene,auxiliary_query,observed.cuda(),initial.cuda(),field=auxiliary_field)
    auxiliary=auxiliary.cpu()
    original=compile_local_evidence(initial,observed,scene,data['texts'],auxiliary)
    fusion_check=replay_menu(data['category_logits'].sigmoid(),original['average_logits'].sigmoid(),class_ids,observed)
    candidates=torch.stack([data['category_logits'].sigmoid(),
        .5*(initial.sigmoid()+auxiliary.sigmoid()), original['features'][:,:,4]]).numpy()
    scores=torch.stack([F.normalize(field['descriptors'].float().cuda(),dim=-1)@query.T,
        F.normalize(auxiliary_field['descriptors'].float().cuda(),dim=-1)@auxiliary_query.T],-1).cpu().numpy()
    return candidates,scores,dict(primary=primary_check,fusion=fusion_check,
        incumbent_probability_exact=True,full_original_source_fusion=True)


def covered_region_carrier(field, offsets):
    offsets=np.asarray(offsets,dtype=np.int64)
    n,r=field['shape']
    if offsets[0]!=0 or offsets[-1]!=r or np.any(np.diff(offsets)<0):
        raise ValueError('unaligned original region/view offsets')
    transfer=sparse.csr_matrix((field['weights'].numpy(),field['columns'].numpy(),field['crow'].numpy()),shape=(n,r))
    if np.any(transfer.data<=0):
        raise ValueError('compiled transfer must contain only positive region support')
    membership=transfer.copy(); membership.data=np.ones_like(membership.data)
    frames=np.repeat(np.arange(len(offsets)-1),np.diff(offsets))
    view_map=sparse.csr_matrix((np.ones(r,np.float32),(np.arange(r),frames)),shape=(r,len(offsets)-1))
    visibility=(membership@view_map).tocsr(); visibility.data[:]=1
    return membership,visibility,frames


def menu_signature(candidates, observed, field, offsets, teacher_scores):
    membership,visibility,frames=covered_region_carrier(field,offsets)
    encoder=RegionResponseEvidence(membership,visibility,frames)
    # Observation-excluded entries remain intact in returned predictions. Only
    # the signature follows the independent corpus's zero-on-unobserved rule.
    value=candidates.copy(); value[:,~observed]=0
    chosen=value.argmax(2); full=[]; shuffled=[]
    for c in range(value.shape[2]):
        selected=(chosen==c)&observed[None]
        x,info=encoder.encode(value[:,:,c],selected,teacher_scores[:,c])
        xp,_=encoder.encode(value[:,:,c],selected,teacher_scores[:,c],permuted=True)
        full.append(x); shuffled.append(xp)
    return np.mean(full,0),np.mean(shuffled,0),dict(info,
        evidence_domain='positive native alpha-transfer support; visibility is its per-view union',
        geometry_visibility_equivalence=False,source_evidence_reused=True)
