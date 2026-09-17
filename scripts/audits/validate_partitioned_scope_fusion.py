"""Four source-view folds: decide scope relations without validation-view masks."""
import argparse,json,shutil
from pathlib import Path
import torch
import torch.nn.functional as F
from radio_gs.training.scope_correspondence import partition_source_scopes
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--allow-unknown-bridges',action='store_true');a=p.parse_args();torch.set_num_threads(4)
root=Path('output/pixel_evidence_rebuild_20260917')/a.scene/'field';out=root.parent/('bridged_scope_validation_20260918' if a.allow_unknown_bridges else 'local_partitioned_scope_validation_20260918');out.mkdir(exist_ok=False);snap=out/'sources';snap.mkdir();implementations={}
for path in [Path(__file__),Path('radio_gs/training/scope_correspondence.py')]:shutil.copy2(path,snap/path.name);implementations[str(path)]=sha256_file(path)
inputs={}
def load(name,digest):
 path=root/name;assert sha256_file(path)==digest;inputs[str(path)]=digest;return torch.load(path,weights_only=False)
info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256'];e=load('caption_region_evidence.pt',json.loads((root/'caption_region_evidence.json').read_text())['sha256']);assert e['teacher_manifest_sha256']==sha256_file(root/'teachers.json') and e['geometry_binding']==auth['metadata']['frozen_geometry_binding'];parent=root/'joint_supervised_image_residual';payload=load('joint_supervised_image_residual/field.pt',json.loads((parent/'report.json').read_text())['field_sha256']);assert payload['geometry_binding']==e['geometry_binding'];observed=payload['state_dict']['observed'].to(a.device);n=len(observed);P=e['masks'][0].numel();gp=root/'grounded_semantic_transfer96/report.json';groups=json.loads(gp.read_text())['groups'];inputs[str(gp)]=sha256_file(gp);G=len(groups);pairpath=root.parent/'scope_depth_scope_pairs.json';pairreport=json.loads(pairpath.read_text());inputs[str(pairpath)]=sha256_file(pairpath);records=pairreport['scope_pairs'];assert pairreport['source_only'] and not pairreport['benchmark_data_used'];views={};frames=info['train_frames'];native=F.normalize(e['native_descriptors'].float(),dim=-1);queries=F.normalize(e['text_embeddings'].float(),dim=-1)
with torch.no_grad():
 for frame in frames:
  vi=info['frames'].index(frame);rec=auth['views'][vi];hit=load(rec['relative_path'],rec['sha256']);gid=hit['gaussian_ids'].long().to(a.device);pid=hit['pixel_ids'].long().to(a.device);w=hit['base_weights'].float().to(a.device);pm=torch.zeros(P,device=a.device).scatter_add_(0,pid,w);mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/pm[pid].clamp_min(1e-8),(P,n)).coalesce();mass=torch.sparse.sum(mat,dim=0).to_dense();mass*=mass>1e-6;ids=[j for j,g in enumerate(groups) if vi in g['views']];members=[[i for i in groups[j]['regions'] if int(e['proposal_views'][i])==vi] for j in ids];masks=torch.stack([e['masks'][rr].bool().any(0).flatten() for rr in members],1);num=torch.sparse.mm(mat.T,masks.float().to(a.device));num*=mass[:,None]>0
  views[frame]=dict(ids=ids,masks=masks,mat=mat,mass=mass.cpu(),num=num.cpu(),emb=torch.stack([native[rr].mean(0) for rr in members]));print(a.scene,'load',frame,flush=True)
 rows=[];plans=[];names=['merged','largest_supported','native_selected','separate_union','local_evidence_winner'];decisions=[]
 for fold in range(4):
  validation=set(frames[fold::4]);training=set(frames)-validation;assert not training&validation
  values={name:torch.zeros(n,G,device=a.device) for name in names};known={name:torch.zeros(n,G,device=a.device,dtype=torch.bool) for name in names}
  for j,g in enumerate(groups):
   present=sorted(f for f in training if j in views[f]['ids'])
   if not present:continue
   # The partition API receives ONLY training-view pair rows. It also filters
   # endpoints defensively, tested with adversarial validation-view relations.
   trainrows=[r for r in records if r['group']==j and r['source_frame'] in training and r['target_frame'] in training]
   plan=partition_source_scopes(j,present,trainrows,allow_unknown_bridges=a.allow_unknown_bridges);clusters=plan['clusters'];cluster_vectors=[];cluster_known=[];cluster_mass=[];scores=[];coherence=[]
   total=torch.zeros(n,device=a.device);mass=torch.zeros_like(total)
   for cluster in clusters:
    cn=torch.zeros(n,device=a.device);cm=torch.zeros_like(cn);emb=[]
    for f in cluster:
     v=views[f];col=v['ids'].index(j);cn+=v['num'][:,col].to(a.device);cm+=v['mass'].to(a.device);emb.append(v['emb'][col])
    total+=cn;mass+=cm;cluster_vectors.append(cn/cm.clamp_min(1e-8));cluster_known.append(cm>1e-6);cluster_mass.append(cm);scores.append(float(F.normalize(torch.stack(emb).mean(0),dim=0)@queries[g['source_query_index']]))
    rr=[r['iou'] for r in trainrows if r['source_frame'] in cluster and r['target_frame'] in cluster];coherence.append(sum(rr)/len(rr) if rr else 0.)
   largest=max(range(len(clusters)),key=lambda k:(len(clusters[k]),coherence[k],-min(clusters[k])));semantic=max(range(len(clusters)),key=lambda k:(scores[k],len(clusters[k]),-min(clusters[k])))
   values['merged'][:,j]=total/mass.clamp_min(1e-8);known['merged'][:,j]=mass>1e-6
   for name,k in [('largest_supported',largest),('native_selected',semantic)]:values[name][:,j]=cluster_vectors[k];known[name][:,j]=cluster_known[k]
   winner=torch.stack(cluster_mass).argmax(0);values['local_evidence_winner'][:,j]=torch.stack(cluster_vectors).gather(0,winner[None])[0];known['local_evidence_winner'][:,j]=mass>1e-6
   values['separate_union'][:,j]=torch.stack(cluster_vectors).amax(0);known['separate_union'][:,j]=torch.stack(cluster_known).any(0)
   plans.append(dict(fold=fold,group=j,phrase=g['phrase'],validation_frames=sorted(validation),**plan,largest_index=largest,native_index=semantic,native_scores=scores))
  for frame in sorted(validation):
   v=views[frame];ids=list(v['ids']);cols=[v['ids'].index(j) for j in ids];target=v['masks'][:,cols].to(a.device);predictions={};supports={}
   for name in names:
    seen=known[name][:,ids]&observed[:,None];support=torch.sparse.mm(v['mat'],seen.float());score=torch.sparse.mm(v['mat'],values[name][:,ids]*seen)/support.clamp_min(1e-8);predictions[name]=(score>.5)&(support>.5);supports[name]=support
   for ci,j in enumerate(ids):
    truth=target[:,ci];rr=dict(frame=frame,fold=fold,group=j,phrase=groups[j]['phrase'],has_training_evidence=any(j in views[f]['ids'] for f in training))
    for name in names:
     pred=predictions[name][:,ci];tp=(pred&truth).sum();rr[name]=dict(iou=float(tp/(pred|truth).sum().clamp_min(1)),precision=float(tp/pred.sum().clamp_min(1)),recall=float(tp/truth.sum().clamp_min(1)),positive_support=float((supports[name][:,ci][truth]>.5).float().mean()))
    rows.append(rr)
  print(a.scene,'fold',fold,len(rows),flush=True)
 macro={name:{metric:sum(r[name][metric] for r in rows)/len(rows) for metric in ['iou','precision','recall','positive_support']} for name in names};fold_metrics={str(f):{name:sum(r[name]['iou'] for r in rows if r['fold']==f)/sum(r['fold']==f for r in rows) for name in names} for f in range(4)};counts={r:sum(x['relation']==r for p in plans for x in p['relations']) for r in ['support','source_contains_target','source_part_of_target','incompatible','unknown']};result=dict(schema='partitioned_scope_validation.v1',source_only=True,benchmark_data_used=False,correspondence_decisions_exclude_validation_views=True,validation_targets_excluded_from_fusion=True,base_geometry_and_group_inventory_preexist=True,teacher_masks_are_not_gt=True,source_only_cluster_selection=True,allow_unknown_bridges=a.allow_unknown_bridges,number_of_folds=4,macro=macro,fold_metrics=fold_metrics,observations=len(rows),relation_counts=counts,split_groups=sum(len(p['clusters'])>1 for p in plans),group_folds=len(plans),plans=plans,rows=rows,inputs=inputs,implementations=implementations,limitations='Scope correspondence/fusion only is held out; frozen geometry, native basis and source group inventory preexist. Separate union is a fixed max over scope hypotheses, not oracle selection or independently verified object identity.')
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(a.scene,macro,counts,flush=True)
