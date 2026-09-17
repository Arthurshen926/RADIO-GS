"""Source 3D image membership calibration control; never reads benchmark labels.

This deliberately fits only the 3D term, not a reproduction of joint 2D/3D
training. Histogram count weighting and class balancing are separate controls.
"""
import argparse,json,math
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField,standardize_observed_scores
from radio_gs.training.correspondence_audit import training_query_conflicts
from radio_gs.training.reliable_correspondence import source_region_query_tasks
from radio_gs.training.frozen_score_calibration import fit_histogram_readout
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',required=True);p.add_argument('--modality',choices=['image','text'],default='image');p.add_argument('--standardize-text',action='store_true');p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4)
 if a.standardize_text and a.modality!='text':raise ValueError('standardization control is text only')
 tag='_standardized' if a.standardize_text else ''
 root=a.root;info=json.loads((root/'teachers.json').read_text());receipt=json.loads((root/'episodes.json').read_text());assert sha256_file(root/'episodes.pt')==receipt['sha256'];data=torch.load(root/'episodes.pt',weights_only=False);assert sha256_file(root/'teachers.json')==data['teacher_manifest_sha256']
 report=json.loads((root/a.run/'report.json').read_text());assert sha256_file(root/a.run/'field.pt')==report['field_sha256'];payload=torch.load(root/a.run/'field.pt',weights_only=False);assert payload['geometry_binding']==data['geometry_binding'];assert payload['query_encoder']==json.loads((root/'native_query_encoder.json').read_text())
 field=ElementRetrievalField(**payload['architecture']);field.load_state_dict(payload['state_dict']);field.to(a.device).eval().requires_grad_(False)
 if (field.bounded_extent and a.modality in field.bounded_extent) or field.source_membership is not None or field.membership is not None or (field.affinity and field.affinity.native_seed_queries) or a.modality in field.native_query_thresholds or (a.modality=='text' and field.grounded_transfer is not None):raise ValueError('histogram calibration requires a plain native cosine deployment path')
 train=[(i,e) for i,e in enumerate(data['episodes']) if e['split']=='train' and e['modality']!='text'];train,_=source_region_query_tasks(train,data,[info['frames'].index(f) for f in info['train_frames']]);tasks=[e for _,e in train if e['modality']=='image'];grounded=None
 if a.modality=='text':
  path=root/'grounded_text_supervision.pt';assert sha256_file(path)==report['grounded_text_supervision_sha256'];grounded=torch.load(path,weights_only=False);assert grounded['episode_sha256']==receipt['sha256'] and grounded['geometry_binding']==data['geometry_binding'] and grounded['train_frames']==info['train_frames'];tasks=grounded['tasks']
 rp=data['region_probability'].float().to(a.device);conflicts=training_query_conflicts(tasks,rp);query=data['queries'][a.modality].float().to(a.device);bins=4096;hist={k:torch.zeros(bins,device=a.device,dtype=torch.float64) for k in ('positive','negative')};rows=[]
 with torch.no_grad():
  scores=field.native_cosine(query,modality=a.modality)
  if a.standardize_text:scores=standardize_observed_scores(scores,field.observed)
  score_min=float(scores.min()) if a.standardize_text else -1.;score_max=float(scores.max()) if a.standardize_text else 1.
  for e in tasks:
   assert info['frames'][e['target_view']] in info['train_frames']
   if a.modality=='image':assert info['frames'][int(data['proposal_views'][e['query_region']])] in info['train_frames']
   qi=e.get('text_index',e['query_region']);pos=rp[e.get('positive_regions',[e['target_region']])].amax(0)>.7;neg=rp[e['negative_regions']].amax(0)>.7 if e['negative_regions'] else torch.zeros_like(pos);known=(pos^neg)&field.observed;ambiguous=conflicts.get(('image',qi));known=known if ambiguous is None else known&~ambiguous
   if grounded is not None:
    pos=grounded['positive_elements'][qi].to(a.device);neg=grounded['negative_elements'][qi].to(a.device);known=(pos^neg)&field.observed&data['seen'][e['target_view']].to(a.device)
   x=scores[:,qi];row={'query_region':qi,'target_view':e['target_view'],'same_view':int(data['proposal_views'][qi])==e['target_view'] if a.modality=='image' else None}
   for label,mask in [('positive',known&pos),('negative',known&neg)]:
    v=x[mask];hist[label]+=torch.bincount(((v-score_min)*bins/(score_max-score_min)).long().clamp_max(bins-1),minlength=bins);row[label+'_count']=len(v);row[label+'_mean_cosine']=float(v.mean()) if len(v) else None
   rows.append(row)
 # Report both weights on exactly the same frozen scores and source labels.
 fits={name:fit_histogram_readout(hist['positive'],hist['negative'],balanced=balanced,score_min=score_min,score_max=score_max) for name,balanced in [('count',False),('balanced',True)]}
 audit={'schema':'frozen_source_image_calibration.v1','parent_field_sha256':report['field_sha256'],'episode_sha256':receipt['sha256'],'implementation_sha256':sha256_file(__file__),'fit_implementation_sha256':sha256_file(Path(__file__).parents[2]/'radio_gs/training/frozen_score_calibration.py'),'standardized_text':a.standardize_text,'score_interval':[score_min,score_max],'modality':a.modality,'grounded_text_sha256':report.get('grounded_text_supervision_sha256') if a.modality=='text' else None,'source_only':True,'benchmark_data_used':False,'fit_contract':'pooled source 3D known memberships, histogram BCE; joint 2D/Dice objective not reproduced','histogram_bins':bins,'histograms':{k:v.cpu().tolist() for k,v in hist.items()},'tasks':rows,'fits':fits}
 for name,fit in fits.items():
  out=root/(a.run+tag+'_'+a.modality+'cal_'+name);out.mkdir();state={k:v.clone() for k,v in payload['state_dict'].items()};state['readout_log_scale.'+a.modality].fill_(math.log(fit['scale']));state['readout_bias.'+a.modality].fill_(fit['bias']);new=dict(payload,state_dict=state)
  if a.standardize_text:new['architecture']=dict(payload['architecture'],standardize_native_text=True)
  assert all(torch.equal(v,state[k]) for k,v in payload['state_dict'].items() if k not in ('readout_log_scale.'+a.modality,'readout_bias.'+a.modality))
  torch.save(new,out/'field.pt');r=dict(report);r.update(field_sha256=sha256_file(out/'field.pt'),calibration_parent_report_sha256=sha256_file(root/a.run/'report.json'),posthoc_calibration=dict(fit,modality=a.modality),posthoc_training_metrics_stale=True);r['readout_calibration']=dict(report.get('readout_calibration',{}),**{a.modality:{'scale':fit['scale'],'bias':fit['bias']}});(out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
 output=root/(a.run+tag+'_source_'+a.modality+'_calibration.json');assert not output.exists();output.write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({'fits':fits,'counts':{k:float(v.sum()) for k,v in hist.items()},'audit':str(output)},indent=2))
if __name__=='__main__':main()
