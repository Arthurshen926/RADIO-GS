"""Source-only actual-query scope correction, with exact frozen-parent control."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file
from scripts.audits.train_rendered_extent import balanced_mask_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',required=True);p.add_argument('--steps',type=int,default=600);p.add_argument('--name',default='query_conditioned_scope_correction');p.add_argument('--mode',choices=['clamped_v1','sigmoid_v1'],default='clamped_v1');p.add_argument('--epochs',type=int,default=0);p.add_argument('--conflict-replay',type=Path);a=p.parse_args()
    if a.conflict_replay and (a.epochs or a.mode!='sigmoid_v1'):p.error('Conflict replay requires sigmoid_v1 and a fixed step budget')
    torch.set_num_threads(4);torch.manual_seed(42)
    out=a.root/a.name;out.mkdir(exist_ok=False)
    fp=a.root/'counterfactual_extent_benefit_admission/field.pt';pr=json.loads(fp.with_name('report.json').read_text());assert sha256_file(fp)==pr['field_sha256'];payload=torch.load(fp,weights_only=False)
    arch=dict(payload['architecture'],query_scope_correction=True if a.mode=='clamped_v1' else {'kind':a.mode});model=ElementRetrievalField(**arch).to(a.device);missing=model.load_state_dict(payload['state_dict'],strict=False)
    assert not missing.unexpected_keys and all(k.startswith('query_scope_correction.') for k in missing.missing_keys)
    model.requires_grad_(False);model.query_scope_correction.requires_grad_(True)
    ep=a.root/'caption_region_evidence.pt';assert sha256_file(ep)==json.loads(ep.with_suffix('.json').read_text())['sha256'];e=torch.load(ep,weights_only=False)
    assert e['source_only'] and not e['benchmark_data_used'] and e['geometry_binding']==payload['geometry_binding']
    info=json.loads((a.root/'teachers.json').read_text());auth=json.loads((a.root/'responsibility.json').read_text());assert sha256_file(a.root/'responsibility.json')==info['authority_sha256']
    masks=e['masks'].bool();pixels=masks[0].numel();n=len(model.latent);by_view={}
    for frame in info['train_frames']:
        targets={}
        for i,r in enumerate(e['records']):
            if r['frame']!=frame or r['confidence']<.7 or int(masks[i].sum())<20:continue
            for qi in r['phrase_indices']:targets[qi]=targets.get(qi,torch.zeros_like(masks[i]))|masks[i]
        if targets:by_view[frame]=targets
    query_ids=sorted({qi for t in by_view.values() for qi in t});positions={qi:i for i,qi in enumerate(query_ids)}
    queries=e['text_embeddings'][query_ids].float().to(a.device)
    del e['region_probability']
    # Cache frozen full-field responses once; training is through the renderer,
    # without changing routing, query encoders, native coordinates or geometry.
    parent=torch.empty(n,len(queries),device=a.device);native=torch.empty_like(parent)
    with torch.no_grad():
        for start in range(0,len(queries),8):
            q=queries[start:start+8];cos=model.native_cosine(q);nl=model.readout_log_scale['text'].clamp(-4,6).exp()*(cos-model.native_query_thresholds['text'](q)[None]);native[:,start:start+len(q)]=nl
            parent[:,start:start+len(q)]=model.logits(q,'text')
        active=((parent-native).abs()>0).any(0)
    views=[]
    for frame,targets in by_view.items():
        rec=auth['views'][info['frames'].index(frame)];hp=a.root/rec['relative_path'];assert sha256_file(hp)==rec['sha256'];hit=torch.load(hp,map_location=a.device,weights_only=False)
        pid,gid,weight=hit['pixel_ids'].long(),hit['gaussian_ids'].long(),hit['base_weights'].float();full=torch.zeros(pixels,device=a.device).scatter_add_(0,pid,weight);keep=model.observed[gid];pid,gid,weight=pid[keep],gid[keep],weight[keep];mass=torch.zeros_like(full).scatter_add_(0,pid,weight);valid=(full>1e-8)&(mass/full.clamp_min(1e-8)>.5)
        rows,inverse=gid.unique(return_inverse=True);mapping=torch.sparse_coo_tensor(torch.stack([pid,inverse]),weight/mass[pid].clamp_min(1e-8),(pixels,len(rows))).coalesce()
        unique={}
        for qi,mask in targets.items():
            key=hashlib.sha256(mask.numpy().tobytes()).hexdigest()
            task=unique.setdefault(key,dict(target=mask.flatten().float().to(a.device),queries=[]))
            task['queries'].append(positions[qi])
        views.append(dict(frame=frame,rows=rows,mapping=mapping,valid=valid,tasks=list(unique.values())))
    assert len(views)>=8
    train,validation=views[:-4],views[-4:]
    def prediction(view,ids):
        rows=view['rows'];idx=torch.tensor(ids,device=a.device)
        logits=model.query_scope_correction(parent[rows][:,idx],native[rows][:,idx],model.latent[rows],queries[idx],active[idx])
        return torch.sparse.mm(view['mapping'],logits.sigmoid())
    @torch.no_grad()
    def evaluate():
        values=[];harmful=0;missed=0
        for v in validation:
            for task in v['tasks']:
                target=task['target'].bool();per_alias=[]
                for ids in torch.tensor(task['queries']).split(8):
                    ids=ids.tolist();pred=(prediction(v,ids)>.5)&v['valid'][:,None]
                    old=(torch.sparse.mm(v['mapping'],parent[v['rows']][:,ids].sigmoid())>.5)&v['valid'][:,None]
                    nat=(torch.sparse.mm(v['mapping'],native[v['rows']][:,ids].sigmoid())>.5)&v['valid'][:,None]
                    def iou(x):return (x&target[:,None]).sum(0)/(x|target[:,None]).sum(0).clamp_min(1)
                    val,oldval,natval=iou(pred),iou(old),iou(nat);per_alias.extend(val.tolist());harmful+=int((val<natval-1e-6).sum());missed+=int((val<oldval-1e-6).sum())
                values.append(sum(per_alias)/len(per_alias))
        return dict(scope_macro_iou=sum(values)/len(values),unique_view_scopes=len(values),below_native_aliases=harmful,below_parent_aliases=missed)
    initial=evaluate();best=initial['scope_macro_iou'];chosen=0
    def snapshot():return {k:v.detach().cpu().clone() for k,v in model.query_scope_correction.state_dict().items()}
    state=snapshot();history=[dict(step=0,validation=initial)]
    if a.mode=='sigmoid_v1':model.query_scope_correction.enabled.fill_(True)
    scope_tasks=[(v,t) for v in train for t in v['tasks']]
    conflict_tasks=[]
    if a.conflict_replay:
        conflicts=json.loads(a.conflict_replay.read_text());assert conflicts['source_only'] and not conflicts['benchmark_data_used']
        assert conflicts['field_sha256']==sha256_file(fp) and conflicts['evidence_sha256']==sha256_file(ep)
        train_by_frame={v['frame']:v for v in train}
        for row in conflicts['observations']:
            if row['frame'] not in train_by_frame:continue
            v=train_by_frame[row['frame']];ids=[positions[qi] for qi in row['query_indices']]
            tasks=[next(t for t in v['tasks'] if qi in t['queries']) for qi in ids]
            conflict_tasks.append((v,ids,tasks))
        if not conflict_tasks:raise ValueError('No incompatible pair confined to correction-training views')
    schedule=[]
    for epoch in range(a.epochs):
        schedule.extend(scope_tasks[i] for i in torch.randperm(len(scope_tasks)).tolist())
    steps=len(schedule) if a.epochs else a.steps
    exposures={qi:0 for qi in range(len(queries))}
    optimizer=torch.optim.Adam(model.query_scope_correction.parameters(),lr=.001)
    print('initial',initial,flush=True)
    for step in range(steps):
        if conflict_tasks and step%2==0:
            v,ids,tasks=conflict_tasks[(step//2)%len(conflict_tasks)]
        elif a.epochs:
            v,task=schedule[step];ids=task['queries'];tasks=[task]*len(ids)
        else:
            v=train[step%len(train)];selected=torch.randperm(len(v['tasks']))[:4].tolist();tasks=[v['tasks'][j] for j in selected];ids=[t['queries'][int(torch.randint(len(t['queries']),(1,)))] for t in tasks]
        for qi in ids:exposures[qi]+=1
        pred=prediction(v,ids);loss=balanced_mask_loss(pred,torch.stack([t['target'] for t in tasks],1),v['valid']);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.query_scope_correction.parameters(),1.);optimizer.step()
        if (step+1)%(len(scope_tasks) if a.epochs else 100)==0 or step+1==steps:
            metric=evaluate();history.append(dict(step=step+1,validation=metric,train_loss=float(loss.detach())));print(history[-1],flush=True)
            if metric['scope_macro_iou']>best+1e-6:best=metric['scope_macro_iou'];chosen=step+1;state=snapshot()
    model.query_scope_correction.load_state_dict(state);saved={k:v.detach().cpu() for k,v in model.state_dict().items()}
    for k,v in payload['state_dict'].items():assert torch.equal(v,saved[k]),k
    torch.save(dict(payload,architecture=arch,state_dict=saved),out/'field.pt')
    report=dict(schema='query_conditioned_scope_correction.v1',source_only=True,benchmark_data_used=False,manual_review_used_for_training=False,field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=sha256_file(fp),evidence_sha256=sha256_file(ep),geometry_unchanged=True,all_parent_tensors_exact=True,training_frames=[v['frame'] for v in train],validation_frames=[v['frame'] for v in validation],selected_step=chosen,steps=steps,epochs=a.epochs,mode=a.mode,conflict_replay_sha256=sha256_file(a.conflict_replay) if a.conflict_replay else None,training_conflict_pairs=len(conflict_tasks),sampling='Alternate automatic incompatible query pairs with unchanged broad source sampling' if conflict_tasks else 'Original source sampling',training_query_exposures={e['phrases'][query_ids[qi]]:count for qi,count in exposures.items()},history=history,unique_scope_counts={str(v['frame']):len(v['tasks']) for v in views},selection='Best full-pixel-denominator source validation unique-scope mean IoU; step zero is eligible; aliases average within each physical pixel-scope observation',validation_scope='Four source views held out only from correction fitting. Parent field/heads saw source data; NOT independent family or benchmark validation.',model_limit='Actual-query-conditioned convex interpolation of existing parent and native probabilities. Cannot recover target scope absent from both.',implementations={f:sha256_file(f) for f in [__file__,'radio_gs/models/query_scope_correction.py','radio_gs/models/element_retrieval_field.py']})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('selected',chosen,best,flush=True)

if __name__=='__main__':main()
