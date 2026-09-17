"""Conservative source-only admission of region correspondence supervision."""
import torch


def reliable_correspondence_tasks(indexed, probability, views, seen, *, minimum_coverage=.5):
    """Keep bidirectionally supported edges in view-unique region tracks.

    Tracks are an admission constraint, not ground-truth instance identities.
    No new matches or negatives are invented. Validation cannot select edges.
    """
    if not 0 < minimum_coverage <= 1:
        raise ValueError('invalid correspondence coverage')
    if any(e['split']!='train' for _,e in indexed):
        raise ValueError('correspondence admission is training-only')
    images=[e for _,e in indexed if e['modality']=='image']
    pairs=sorted({tuple(sorted((e['query_region'],e['target_region']))) for e in images})
    hard=probability>.7; candidates=[];rejected={};quality={}
    for q,t in pairs:
        v,w=int(views[q]),int(views[t])
        if v==w:raise ValueError('cross-view correspondence has identical views')
        shared=seen[v]&seen[w];a=hard[q]&shared;b=hard[t]&shared
        intersection=int((a&b).sum());coverage=min(intersection/max(1,int(a.sum())),intersection/max(1,int(b.sum())))
        quality[(q,t)]=coverage
        if coverage<minimum_coverage:rejected[(q,t)]='insufficient_bidirectional_coverage'
        else:candidates.append((coverage,q,t))
    parent=list(range(len(views)));members={i:{i} for i in range(len(views))}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    accepted=set()
    for score,q,t in sorted(candidates,key=lambda x:(-x[0],x[1],x[2])):
        l,r=find(q),find(t)
        if l!=r:
            left_views={int(views[x]) for x in members[l]};right_views={int(views[x]) for x in members[r]}
            if left_views&right_views:
                rejected[(q,t)]='same_view_track_collision';continue
            # Existing source pair evidence must not contradict a proposed merge.
            contradiction=False
            for x in members[l]:
                for y in members[r]:
                    pair=tuple(sorted((x,y)))
                    if pair in quality and quality[pair]<minimum_coverage:contradiction=True
            if contradiction:
                rejected[(q,t)]='inconsistent_observed_pair';continue
            parent[r]=l;members[l]|=members.pop(r)
        accepted.add((q,t))
    directed={(q,t) for q,t in accepted}|{(t,q) for q,t in accepted}
    counterparts={(int(views[e['query_region']]),int(views[e['target_region']]),e['target_region']):e['query_region'] for e in images}
    result=[];removed_negatives=0;positive_only=0;dropped=0
    for i,e in indexed:
        if e['modality'] not in ('image','prompt'):
            result.append((i,e));continue
        q,t=e['query_region'],e['target_region']
        if (q,t) not in directed:dropped+=1;continue
        negative=[]
        for nt in e['negative_regions']:
            nq=counterparts.get((int(views[q]),int(views[t]),nt))
            if nq is not None and (nq,nt) in directed and find(q)!=find(nt):negative.append(nt)
        removed_negatives+=len(e['negative_regions'])-len(negative)
        # Retain positive-only evidence, explicitly report it; absence isn't negative.
        positive_only+=not bool(negative)
        result.append((i,dict(e,negative_regions=negative)))
    if not all(any(e['modality']==m for _,e in result) for m in ('image','prompt')):
        raise ValueError('reliable correspondence has no remaining query modality')
    audit={'schema':'source_correspondence_admission.v1','minimum_bidirectional_coverage':minimum_coverage,
           'candidate_pairs':len(pairs),'accepted_pairs':len(accepted),'dropped_image_prompt_tasks':dropped,
           'removed_negative_region_entries':removed_negatives,'retained_positive_only_tasks':positive_only,
           'accepted_edges':[list(x) for x in sorted(accepted)],
           'rejected_edges':[{'pair':list(k),'reason':v} for k,v in sorted(rejected.items())],
           'tracks':[sorted(m) for m in members.values() if len(m)>1],
           'benchmark_data_used':False,'physical_instance_identity_verified':False}
    return result,audit


def source_region_query_tasks(indexed, data, train_views):
    """Add exact same-view source-mask tasks independent of cross-view matching.

    Explicit negatives are disjoint, appearance-distinct source proposals;
    uncovered pixels remain unknown. Text keeps its separate grounded contract.
    """
    from torch.nn import functional as F
    views=data['proposal_views'];masks=data['masks'].flatten(1);image=F.normalize(data['queries']['image'].float(),dim=-1);dino=F.normalize(data['queries']['region_dino'].float(),dim=-1)
    prompt_valid=data['queries'].get('prompt_valid',torch.ones(len(views),dtype=torch.bool))
    if prompt_valid.shape!=(len(views),) or prompt_valid.dtype!=torch.bool:raise ValueError('invalid prompt admission mask')
    for _,e in indexed:
        if e['modality']=='prompt' and not bool(prompt_valid[e['query_region']]):raise ValueError('inadmissible prompt task')
    image_valid=torch.ones(len(views),dtype=torch.bool)
    for rejection in data.get('prompt_admission',{}).get('rejected_regions',[]):
        if rejection['reason']=='missing_positive_support':image_valid[rejection['region']]=False
    source=[i for i,v in enumerate(views.tolist()) if v in set(train_views) and bool(image_valid[i])]
    if not source:raise ValueError('no source query regions')
    next_index=max([i for i,_ in indexed]+[len(data['episodes'])])+1;out=list(indexed);negative_count=0
    for q in source:
        negatives=[r for r in source if r!=q and int(views[r])==int(views[q]) and not bool((masks[q]&masks[r]).any()) and float(image[q]@image[r])<=.8 and float(dino[q]@dino[r])<=.8]
        negative_count+=bool(negatives)
        for modality in ('image','prompt'):
            if modality=='prompt' and not bool(prompt_valid[q]):continue
            out.append((next_index,{'split':'train','modality':modality,'query_region':q,'target_region':q,'target_view':int(views[q]),'negative_regions':negatives,'source_region_task':True}));next_index+=1
    return out,{'schema':'source_region_query_tasks.v1','regions':len(source),'added_tasks_per_modality':{'image':len(source),'prompt':sum(bool(prompt_valid[q]) for q in source)},'rejected_prompt_regions':[q for q in source if not bool(prompt_valid[q])],'rejected_no_positive_geometry_regions':torch.where(~image_valid)[0].tolist(),'regions_with_explicit_negatives':negative_count,'source_views':list(train_views),'benchmark_data_used':False,'cross_view_identity_claim':False}


def cross_view_image_tasks_only(indexed, views):
    """Remove same-view image membership tasks; retain other supervision intact."""
    result=[];removed=[]
    for index,episode in indexed:
        if episode['split']!='train':raise ValueError('query admission is training-only')
        if episode['modality']=='image' and int(views[episode['query_region']])==int(views[episode['target_region']]):
            removed.append(index)
        else:result.append((index,episode))
    if not any(e['modality']=='image' for _,e in result):raise ValueError('no cross-view image tasks remain')
    return result,{'removed_same_view_image_tasks':len(removed),'removed_task_indices':removed,'remaining_image_tasks':sum(e['modality']=='image' for _,e in result),'teacher_region_coverage_modified':False}
