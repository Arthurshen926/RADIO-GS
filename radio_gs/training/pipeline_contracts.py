"""Fail-closed contracts for the frozen-geometry retrieval training pipeline."""
import torch


def validate_grounded_supervision(bundle, *, frames, train_frames, proposal_views,
                                   num_elements, num_pixels, queries, allow_empty=False):
    if bundle.get('schema') != 'grounded_text_supervision.v2':
        raise ValueError('obsolete grounded supervision: rebuild with the current compiler')
    if bundle.get('negative_policy') != 'same_view_positive_detection_required':
        raise ValueError('unknown grounded negative policy')
    if bundle.get('queries') != queries or bundle.get('train_frames') != train_frames:
        raise ValueError('grounded vocabulary or training frame axes differ')
    tasks=bundle.get('tasks',[])
    if not tasks and not (allow_empty and bundle.get('empty_text_supervision_explicitly_allowed') is True
                          and bundle.get('direct_text_supervision_available') is False):
        raise ValueError('grounded supervision has no explicitly admitted tasks')
    for positive,negative,shape in [('positive_pixels','negative_pixels',(len(tasks),num_pixels)),
                                    ('positive_elements','negative_elements',(len(queries),num_elements))]:
        left,right=bundle[positive],bundle[negative]
        if left.dtype!=torch.bool or right.dtype!=torch.bool or tuple(left.shape)!=shape or tuple(right.shape)!=shape:
            raise ValueError('grounded label axes or boolean dtype differ')
        if (left&right).any():raise ValueError('contradictory grounded signs')
    seen=set()
    for index,e in enumerate(tasks):
        vi,qi=e['target_view'],e['text_index']
        if e['split']!='train' or e['modality']!='text' or e['grounded_text_task']!=index:
            raise ValueError('grounded task indexing or modality differs')
        if not 0<=vi<len(frames) or frames[vi] not in train_frames or not 0<=qi<len(queries):
            raise ValueError('grounded task outside training axes')
        if (vi,qi) in seen:raise ValueError('duplicate grounded query/view task')
        seen.add((vi,qi))
        for key in ('target_region','query_region'):
            ri=e[key]
            if not 0<=ri<len(proposal_views) or int(proposal_views[ri])!=vi:
                raise ValueError('grounded auxiliary region belongs to another view')
        for key in ('positive_pixels','negative_pixels'):
            if not bundle[key][index].any():raise ValueError('grounded task lacks signed pixel evidence')
    return {'schema':bundle['schema'],'tasks':len(tasks),'direct_text_supervision_available':bool(tasks),'disjoint_signs':True,'train_only':True}


def validate_contributions(pixel_ids, element_ids, weights, *, num_pixels, num_elements):
    if pixel_ids.ndim!=1 or element_ids.shape!=pixel_ids.shape or weights.shape!=pixel_ids.shape:
        raise ValueError('contribution axes differ')
    if pixel_ids.dtype not in (torch.int32,torch.int64) or element_ids.dtype not in (torch.int32,torch.int64):
        raise ValueError('contribution indices must be integer')
    if not torch.isfinite(weights).all() or (weights<0).any():raise ValueError('invalid contribution weights')
    if len(weights) and ((pixel_ids<0).any() or (pixel_ids>=num_pixels).any() or (element_ids<0).any() or (element_ids>=num_elements).any()):
        raise ValueError('contribution index outside frozen geometry/raster')


def validate_training_axes(data, authority, info):
    n=authority['num_gaussians'];h=authority['metadata']['feature_height'];w=authority['metadata']['feature_width']
    frames=info['frames'];regions=len(data['proposal_views'])
    if authority['frame_indices']!=frames:raise ValueError('teacher and geometry frame order differs')
    if len(set(frames))!=len(frames):raise ValueError('duplicate source frames')
    if data['seen'].dtype!=torch.bool or tuple(data['seen'].shape)!=(len(frames),n):raise ValueError('seen axes differ')
    if data['masks'].dtype!=torch.bool or tuple(data['masks'].shape)!=(regions,h,w):raise ValueError('region mask axes differ')
    probability=data['region_probability']
    if tuple(probability.shape)!=(regions,n) or not torch.isfinite(probability).all() or ((probability<0)|(probability>1)).any():raise ValueError('region membership axes or values differ')
    views=data['proposal_views']
    if views.dtype not in (torch.int32,torch.int64) or ((views<0)|(views>=len(frames))).any():raise ValueError('region view indices differ')
    for key in ('image','prompt'):
        value=data['queries'][key]
        if value.ndim!=2 or len(value)!=regions or not torch.isfinite(value).all():raise ValueError('query region axis differs')
    text=data['queries']['text']
    if text.ndim!=2 or len(text)!=len(data['queries']['queries']) or not torch.isfinite(text).all():raise ValueError('text vocabulary axis differs')
    seeds=data['queries']['seed_pixels']
    valid=data['queries'].get('prompt_valid',torch.ones(regions,dtype=torch.bool))
    if valid.dtype!=torch.bool or tuple(valid.shape)!=(regions,):raise ValueError('invalid prompt admission axis')
    if seeds.dtype not in (torch.int32,torch.int64) or tuple(seeds.shape)!=(regions,2):raise ValueError('invalid registered prompt seeds')
    active=seeds[valid]
    if ((active<0)|(active>=h*w)).any() or (active[:,0]==active[:,1]).any():raise ValueError('invalid registered prompt seeds')
    if (seeds[~valid]!=-1).any() or data['queries']['prompt'][~valid].count_nonzero():raise ValueError('invalid inactive prompt sentinel')
    rejected=data.get('prompt_admission',{}).get('rejected_regions',[])
    if len({r['region'] for r in rejected})!=len(rejected) or {r['region'] for r in rejected}!=set(torch.where(~valid)[0].tolist()):raise ValueError('prompt rejection evidence differs')
    if any(r['reason'] not in ('missing_positive_support','missing_negative_support') for r in rejected):raise ValueError('unknown prompt rejection reason')
    if any(e['modality']=='prompt' and not bool(valid[e['query_region']]) for e in data.get('episodes',[])):raise ValueError('inactive prompt used by an episode')
    if 'seed_rows' in data['queries']:
        rows=data['queries']['seed_rows']
        if rows.dtype not in (torch.int32,torch.int64) or tuple(rows.shape)!=(regions,2) or ((rows[valid]<0)|(rows[valid]>=n)).any() or (rows[~valid]!=-1).any():raise ValueError('invalid registered prompt element rows')
    return {'frames':len(frames),'regions':regions,'elements':n,'pixels':h*w,'axes_valid':True}


def validate_native_element_teacher(bundle, *, num_elements, dimensions, query_dimension, observed):
    if bundle.get('schema')!='consistent_native_element_teacher.v1' or bundle.get('benchmark_data_used') is not False:
        raise ValueError('invalid native element teacher contract')
    target,basis,valid=bundle['directions'],bundle['basis'],bundle['valid']
    if tuple(target.shape)!=(num_elements,dimensions) or tuple(basis.shape)!=(dimensions,query_dimension) or tuple(valid.shape)!=(num_elements,) or valid.dtype!=torch.bool:
        raise ValueError('native element teacher axes differ')
    if not torch.isfinite(target).all() or not torch.isfinite(basis).all() or not valid.any():raise ValueError('invalid native teacher values')
    if (valid&~observed.cpu()).any() or target[~valid].count_nonzero():raise ValueError('native teacher writes outside observed support')
    if not torch.allclose(target[valid].float().norm(dim=-1),torch.ones(int(valid.sum())),atol=.002,rtol=0):raise ValueError('native teacher directions are not unit')
    if not torch.allclose(basis.float()@basis.float().T,torch.eye(dimensions),atol=1e-4,rtol=0):raise ValueError('native basis is not orthonormal')
    return {'teacher_elements':int(valid.sum()),'directions_finite_unit':True,'source_observed_only':True}
