"""Compile source-grounded text into corroborated 2D/3D training evidence."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.training.grounded_text_supervision import corroborated_labels,grounded_pixel_evidence,independent_positive_carriers
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--grounding',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--independent-pixel-corroboration',action='store_true');p.add_argument('--allow-empty-text-supervision',action='store_true');p.add_argument('--device',default='cuda:2');a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists() or a.output.with_suffix('.json').exists():raise FileExistsError(a.output)
 info=json.loads((a.root/'teachers.json').read_text());receipt=json.loads((a.root/'episodes.json').read_text());auth=json.loads((a.root/'responsibility.json').read_text())
 if sha256_file(a.root/'episodes.pt')!=receipt['sha256'] or sha256_file(a.root/'responsibility.json')!=info['authority_sha256']:raise ValueError('source identity')
 d=torch.load(a.root/'episodes.pt',weights_only=False);gr=json.loads(a.grounding.with_suffix('.json').read_text())
 if sha256_file(a.grounding)!=gr['sha256']:raise ValueError('grounding identity')
 g=torch.load(a.grounding,weights_only=False)
 if g['teacher_manifest_sha256']!=d['teacher_manifest_sha256'] or g['geometry_binding']!=d['geometry_binding'] or g['frames']!=info['train_frames'] or g['queries']!=d['queries']['queries']:raise ValueError('grounding scope differs')
 n=auth['num_gaussians'];pixels=auth['num_pixels'];trainviews=[info['frames'].index(x) for x in g['frames']];observed=d['seen'][trainviews].any(0).to(a.device);prob=g['probability'].flatten(2).float().to(a.device)
 matrices=[];supports=[];view_seen=[];view_positive=[];audit=[];positive_counts=torch.zeros(len(g['queries']),n,dtype=torch.int16,device=a.device);negative_counts=torch.zeros_like(positive_counts)
 for i,vi in enumerate(trainviews):
  r=auth['views'][vi];path=a.root/r['relative_path']
  if sha256_file(path)!=r['sha256']:raise ValueError('projection identity')
  hit=torch.load(path,map_location=a.device,weights_only=False);pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weights=hit['base_weights'].float();mass=torch.zeros(pixels,device=a.device).scatter_add_(0,pid,weights)
  mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),weights/mass[pid].clamp_min(1e-8),(pixels,n)).coalesce();matrices.append(mat)
  supports.append(torch.sparse.mm(mat,observed.float()[:,None])[:,0]);rowmass=torch.sparse.sum(mat,dim=0).to_dense();seen=rowmass>1e-6;view_seen.append(seen)
  positives=prob[i]>.8;other=positives.any(0)[None];negatives=(prob[i]<.05)&other&positives.any(1)[:,None]
  poslift=torch.sparse.mm(mat.T,positives.float().T).T/rowmass[None].clamp_min(1e-8)
  neglift=torch.sparse.mm(mat.T,negatives.float().T).T/rowmass[None].clamp_min(1e-8)
  votes=(poslift>.7)&seen[None];view_positive.append(votes);positive_counts+=votes.short();negative_counts+=((neglift>.7)&seen[None]).short()
 positive,negative=corroborated_labels(positive_counts,negative_counts)
 tasks=[];positive_masks=[];negative_masks=[]
 for i,vi in enumerate(trainviews):
  ps=torch.sparse.mm(matrices[i],positive.float().T).T/supports[i][None].clamp_min(1e-8)
  ns=torch.sparse.mm(matrices[i],negative.float().T).T/supports[i][None].clamp_min(1e-8)
  carriers=independent_positive_carriers(positive_counts,view_positive[i],negative_counts)
  independent_support=torch.sparse.mm(matrices[i],carriers.float().T).T/supports[i][None].clamp_min(1e-8)
  other=(prob[i]>.8).any(0)
  representative=int(torch.where(d['proposal_views']==vi)[0][0])
  for qi in range(len(g['queries'])):
   pos,neg=grounded_pixel_evidence(prob[i,qi],other,ps[qi],ns[qi],supports[i])
   rescue=(prob[i,qi]>.8)&(supports[i]>.5)&(independent_support[qi]>.5)&~pos
   audit.append({'frame':g['frames'][i],'query_index':qi,'raw_positive_pixels':int(((prob[i,qi]>.8)&(supports[i]>.5)).sum()),'original_positive_pixels':int(pos.sum()),'independently_supported_extra_pixels':int(rescue.sum()),'negative_pixels':int(neg.sum()),'original_task_admitted':int(pos.sum())>=8 and int(neg.sum())>=8})
   if a.independent_pixel_corroboration:pos=pos|rescue
   if int(pos.sum())<8 or int(neg.sum())<8:continue
   index=len(tasks);tasks.append({'modality':'text','split':'train','text_index':qi,'target_view':vi,'target_region':representative,'query_region':representative,'negative_regions':[],'grounded_text_task':index,'source_episode_indices':[]})
   positive_masks.append(pos.cpu());negative_masks.append(neg.cpu())
 if not tasks and not a.allow_empty_text_supervision:raise ValueError('no corroborated text tasks with both positive and negative evidence')
 payload={'schema':'grounded_text_supervision.v2','negative_policy':'same_view_positive_detection_required','compiler_sha256':sha256_file(__file__),'grounding_path':str(a.grounding.resolve()),'episode_sha256':receipt['sha256'],'grounding_sha256':gr['sha256'],'teacher_manifest_sha256':d['teacher_manifest_sha256'],'geometry_binding':d['geometry_binding'],'queries':g['queries'],'train_frames':g['frames'],'tasks':tasks,'positive_pixels':torch.stack(positive_masks) if tasks else torch.empty((0,pixels),dtype=torch.bool),'negative_pixels':torch.stack(negative_masks) if tasks else torch.empty((0,pixels),dtype=torch.bool),'positive_elements':positive.cpu(),'negative_elements':negative.cpu(),'policy':'SAM3 source text confidence>.7, mask positive>.8; negative only when this query has a positive detection in the same view, inside other positive mask and query probability<.05; both element signs require2 source views; conflicts abstain; no benchmark labels','counts':{'tasks':len(tasks),'text_categories':len({e['text_index'] for e in tasks}),'conflicting_query_elements':int(((positive_counts>=2)&(negative_counts>=2)).sum())}}
 payload['empty_text_supervision_explicitly_allowed']=a.allow_empty_text_supervision
 payload['direct_text_supervision_available']=bool(tasks)
 payload['independent_pixel_corroboration']=a.independent_pixel_corroboration
 payload['pixel_corroboration_policy']='optional current pixel SAM3 positive AND majority carrier support from at least one other positive source view; any negative view vetoes extra carrier; hard 3D signs unchanged'
 payload['pixel_corroboration_audit']=audit
 torch.save(payload,a.output);a.output.with_suffix('.json').write_text(json.dumps({'schema':payload['schema'],'sha256':sha256_file(a.output),'counts':payload['counts'],'policy':payload['policy'],'independent_pixel_corroboration':a.independent_pixel_corroboration,'pixel_corroboration_audit':audit,'queries_supervised':sorted({g['queries'][e['text_index']] for e in tasks})},indent=2)+'\n');print(payload['counts'],flush=True)

if __name__=='__main__':main()
