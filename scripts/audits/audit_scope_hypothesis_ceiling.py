"""Four source-view folds: decide scope relations without validation-view masks."""
import argparse,json,shutil
from pathlib import Path
import torch
import torch.nn.functional as F
from radio_gs.training.scope_correspondence import partition_source_scopes
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--allow-unknown-bridges',action='store_true');a=p.parse_args();torch.set_num_threads(4)
root=Path('output/pixel_evidence_rebuild_20260917')/a.scene/'field';out=root.parent/'scope_hypothesis_ceiling_20260918';out.mkdir(exist_ok=False);snap=out/'sources';snap.mkdir();implementations={}
for path in [Path(__file__),Path('radio_gs/training/scope_correspondence.py')]:shutil.copy2(path,snap/path.name);implementations[str(path)]=sha256_file(path)
inputs={}
def load(name,digest):
 path=root/name;assert sha256_file(path)==digest;inputs[str(path)]=digest;return torch.load(path,weights_only=False)
info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256'];e=load('caption_region_evidence.pt',json.loads((root/'caption_region_evidence.json').read_text())['sha256']);assert e['teacher_manifest_sha256']==sha256_file(root/'teachers.json') and e['geometry_binding']==auth['metadata']['frozen_geometry_binding'];parent=root/'joint_supervised_image_residual';payload=load('joint_supervised_image_residual/field.pt',json.loads((parent/'report.json').read_text())['field_sha256']);assert payload['geometry_binding']==e['geometry_binding'];observed=payload['state_dict']['observed'].to(a.device);n=len(observed);P=e['masks'][0].numel();gp=root/'grounded_semantic_transfer96/report.json';groups=json.loads(gp.read_text())['groups'];inputs[str(gp)]=sha256_file(gp);G=len(groups);pairpath=root.parent/'scope_depth_scope_pairs.json';pairreport=json.loads(pairpath.read_text());inputs[str(pairpath)]=sha256_file(pairpath);records=pairreport['scope_pairs'];assert pairreport['source_only'] and not pairreport['benchmark_data_used'];views={};frames=info['train_frames'];native=F.normalize(e['native_descriptors'].float(),dim=-1);queries=F.normalize(e['text_embeddings'].float(),dim=-1)
with torch.no_grad():
 for frame in frames:
  vi=info['frames'].index(frame);rec=auth['views'][vi];hit=load(rec['relative_path'],rec['sha256']);gid=hit['gaussian_ids'].long().to(a.device);pid=hit['pixel_ids'].long().to(a.device);w=hit['base_weights'].float().to(a.device);pm=torch.zeros(P,device=a.device).scatter_add_(0,pid,w);mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/pm[pid].clamp_min(1e-8),(P,n)).coalesce();mass=torch.sparse.sum(mat,dim=0).to_dense();mass*=mass>1e-6;ids=[j for j,g in enumerate(groups) if vi in g['views']];members=[[i for i in groups[j]['regions'] if int(e['proposal_views'][i])==vi] for j in ids];masks=torch.stack([e['masks'][rr].bool().any(0).flatten() for rr in members],1);num=torch.sparse.mm(mat.T,masks.float().to(a.device));num*=mass[:,None]>0
  views[frame]=dict(ids=ids,masks=masks,mat=mat,mass=mass.cpu(),num=num.cpu(),emb=torch.stack([native[rr].mean(0) for rr in members]));print(a.scene,'load',frame,flush=True)
 planpath=root.parent/'bridged_scope_validation_20260918/report.json';planreport=json.loads(planpath.read_text());inputs[str(planpath)]=sha256_file(planpath);plans={(r['fold'],r['group']):r for r in planreport['plans']};rows=[]
 for fold in range(4):
  validation=set(frames[fold::4]);training=set(frames)-validation
  for j,g in enumerate(groups):
   targetframes=[f for f in sorted(validation) if j in views[f]['ids']]
   if not targetframes:continue
   plan=plans.get((fold,j))
   if plan is None:
    for f in targetframes:rows.append(dict(fold=fold,frame=f,group=j,phrase=g['phrase'],baseline=0.,per_observation_oracle=0.,group_fold_oracle=0.,no_training_evidence=True))
    continue
   assert not set(plan['training_frames'])&validation
   values=[];known=[];total=torch.zeros(n,device=a.device);mass=torch.zeros_like(total)
   for cluster in plan['clusters']:
    assert set(cluster)<=training
    cn=torch.zeros(n,device=a.device);cm=torch.zeros_like(cn)
    for f in cluster:
     v=views[f];col=v['ids'].index(j);cn+=v['num'][:,col].to(a.device);cm+=v['mass'].to(a.device)
    values.append(cn/cm.clamp_min(1e-8));known.append(cm>1e-6);total+=cn;mass+=cm
   # Include the merged response as candidate zero. These GT selections are
   # evaluation-only upper bounds, never the source-selected policy outputs.
   values=torch.stack([total/mass.clamp_min(1e-8),*values],1);known=torch.stack([mass>1e-6,*known],1)&observed[:,None];scores=[]
   for f in targetframes:
    v=views[f];truth=v['masks'][:,v['ids'].index(j)].to(a.device);support=torch.sparse.mm(v['mat'],known.float());pred=(torch.sparse.mm(v['mat'],values*known)/support.clamp_min(1e-8)>.5)&(support>.5);ious=(pred&truth[:,None]).sum(0)/(pred|truth[:,None]).sum(0).clamp_min(1);scores.append(ious)
   scores=torch.stack(scores);chosen=int(scores.mean(0).argmax())
   for i,f in enumerate(targetframes):rows.append(dict(fold=fold,frame=f,group=j,phrase=g['phrase'],baseline=float(scores[i,0]),per_observation_oracle=float(scores[i].max()),group_fold_oracle=float(scores[i,chosen]),group_fold_candidate=chosen,candidates=scores.shape[1],no_training_evidence=False))
  print(a.scene,'ceiling fold',fold,len(rows),flush=True)
 macro={k:sum(r[k] for r in rows)/len(rows) for k in ['baseline','per_observation_oracle','group_fold_oracle']};result=dict(schema='source_scope_hypothesis_ceiling.v1',source_only=True,benchmark_data_used=False,diagnostic_only=True,deployable=False,validation_masks_used_for_oracle_selection=True,validation_masks_used_for_correspondence=False,macro=macro,rows=rows,inputs=inputs,implementations=implementations,limitation='Oracle chooses among fixed source-only scope hypotheses and merged baseline. Group/fold oracle holds one candidate fixed across that fold target views. Teacher pseudo-mask upper bound is not a real-query metric and cannot select training policy.')
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(a.scene,macro,flush=True)
