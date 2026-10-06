"""Matched source-reference image controls on assistant-reviewed identities.

Automatic matching is an explicit nearest-native-crop diagnostic, not a claim
that it reproduces the deployed text correspondence builder. No target-view
candidate participates in matching. Boundary masks remain teacher-derived.
"""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
 torch.set_num_threads(4);reviewpath=Path('docs/reviews/2026-09-18-typed-multiview-scope-review.json');review=json.loads(reviewpath.read_text());results=[]
 for scene in ['figurines','ramen']:
  tracks=[t for t in review['tracks'] if t['scene']==scene];root=Path('output/v2_scope_rebuild_20260918')/scene/'field';ep=root/'caption_region_evidence.pt';assert all(sha256_file(ep)==t['evidence_sha256'] for t in tracks);e=torch.load(ep,weights_only=False);info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256'];fp=root/'counterfactual_extent_benefit_admission/field.pt';assert sha256_file(fp)==json.loads(fp.with_name('report.json').read_text())['field_sha256'];p=torch.load(fp,weights_only=False);field=ElementRetrievalField(**p['architecture']);field.load_state_dict(p['state_dict']);field.requires_grad_(False);known=field.observed_for('image');n=len(field.latent);pixels=e['masks'][0].numel();descriptors=F.normalize(e['native_descriptors'].float(),dim=-1);views={}
  for frame in sorted({o['frame'] for t in tracks for o in t['observations']}):
   rec=auth['views'][info['frames'].index(frame)];hp=root/rec['relative_path'];assert sha256_file(hp)==rec['sha256'];hit=torch.load(hp,weights_only=False);pid,gid,w=hit['pixel_ids'].long(),hit['gaussian_ids'].long(),hit['base_weights'].float();mass=torch.zeros(pixels).scatter_add_(0,pid,w);mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/mass[pid].clamp_min(1e-8),(pixels,n)).coalesce();views[frame]=dict(mat=mat,mass=torch.sparse.sum(mat,dim=0).to_dense(),candidates=[i for i,r in enumerate(e['records']) if r['frame']==frame and r['confidence']>=.7 and int(e['masks'][i].sum())>=10])
  def measure(value,support,target,mat):
   support=support&known;rendered_support=torch.sparse.mm(mat,support.float()[:,None])[:,0];score=torch.sparse.mm(mat,(value*support)[:,None])[:,0]/rendered_support.clamp_min(1e-8);pred=(score>.5)&(rendered_support>.5);tp=int((pred&target).sum());return dict(iou=tp/max(int((pred|target).sum()),1),precision=tp/max(int(pred.sum()),1),recall=tp/max(int(target.sum()),1),positive_support=float((rendered_support[target]>.5).float().mean()))
  with torch.no_grad():
   for track in tracks:
    observations=[]
    for target_observation in track['observations']:
     frame=target_observation['frame'];target=e['masks'][target_observation['region']].bool().flatten();target_view=views[frame];source=[o for o in track['observations'] if o['frame']!=frame];seed=source[0];q=descriptors[seed['region']:seed['region']+1];actual=field.retrieve(q,'image')['scores'][:,0];row=dict(frame=frame,reference_frame=seed['frame'],reference_region=seed['region'],source_frames=[o['frame'] for o in source],target_region=target_observation['region'],automatic_regions=[])
     for mode in ['reviewed','automatic']:
      numerator=torch.zeros(n);mass=torch.zeros(n)
      for observation in source:
       view=views[observation['frame']];region=observation['region']
       if mode=='automatic' and observation['frame']!=seed['frame']:
        candidates=view['candidates']
        if not candidates:continue  # Unknown, never an absent-object negative.
        region=candidates[int((descriptors[candidates]@q[0]).argmax())]
       if mode=='automatic':row['automatic_regions'].append(dict(frame=observation['frame'],region=region,reviewed_region=observation['region']))
       assert e['records'][region]['frame']!=frame
       mask=e['masks'][region].float().flatten();numerator+=torch.sparse.mm(view['mat'].T,mask[:,None])[:,0];mass+=view['mass']
      row[mode+'_transfer']=measure(numerator/mass.clamp_min(1e-8),mass>1e-6,target,target_view['mat'])
     row['per_source_reviewed']=[];maximum=torch.zeros(n);maximum_support=torch.zeros(n,dtype=torch.bool)
     for observation in source:
      sv=views[observation['frame']];mask=e['masks'][observation['region']].float().flatten();single=torch.sparse.mm(sv['mat'].T,mask[:,None])[:,0]/sv['mass'].clamp_min(1e-8);support=sv['mass']>1e-6;row['per_source_reviewed'].append(dict(source_frame=observation['frame'],**measure(single,support,target,target_view['mat'])));maximum=torch.maximum(maximum,single*support);maximum_support|=support
     row['positive_union_control']=measure(maximum,maximum_support,target,target_view['mat'])
     local_mass=target_view['mass'];local=torch.sparse.mm(target_view['mat'].T,target.float()[:,None])[:,0]/local_mass.clamp_min(1e-8);row['same_view_cycle']=measure(local,local_mass>1e-6,target,target_view['mat']);row['current_image_readout']=measure(actual,known,target,target_view['mat']);observations.append(row)
    names=['same_view_cycle','reviewed_transfer','automatic_transfer','current_image_readout','positive_union_control'];result=dict(id=track['id'],scene=scene,observations=observations,macro={name:sum(r[name]['iou'] for r in observations)/len(observations) for name in names},field_sha256=sha256_file(fp));results.append(result);print(track['id'],result['macro'],flush=True)
 out=Path('output/expert_plan_20260918/typed_reviewed_track_pairs.json');assert not out.exists();report=dict(schema='typed_reviewed_track_decomposition.v2',source_only=True,benchmark_data_used=False,training=False,review_sha256=sha256_file(reviewpath),implementation_sha256=sha256_file(__file__),reference_query='Frozen native mask-aligned source crop descriptor of first other reviewed view, same input to automatic matching and current image readout.',automatic_control='Nearest native crop from each other source view; exact reference region fixed; evaluation-view candidates excluded. This is an appearance-matching diagnostic, not deployed text routing.',limitations='Source diagnostic; parent map saw these source views. Assistant reviews identity/granularity, not exact mask boundaries. Different source count for two/three-view tracks; two-view automatic/reviewed controls coincide by construction. Instance target differs from class-union benchmark. Not additive causal error fractions.',results=results);out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
