"""Fresh category scores from declared memory; target labels are evaluator-only."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--query',required=True);p.add_argument('--output',required=True);p.add_argument('--gpu',default='0');p.add_argument('--source-evidence',choices=['declared','none'],default='declared');a=p.parse_args()
 os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
 import torch
 from radio_gs.querying.scannet_runtime import query_memory,refine_from_declared_sources
 torch.set_num_threads(4);start=time.monotonic();scene=json.loads(Path(a.scene).read_text());document=json.loads(Path(a.query).read_text());queries=document.get('queries',[document]);texts=[q['text'] for q in queries]
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if scene['runtime_provider']=='native_category_map':
  if a.source_evidence!='none' and not (scene.get('native_category_head') or scene.get('native_category_prior')):raise ValueError('Source membership requires an explicit calibrated native_category_head')
  from radio_gs.models.native_query_encoders import encode_category_text
  with torch.inference_mode():
   encodings=encode_category_text(scene,texts,'cuda')
   if scene.get('native_category_prior',{}).get('kind')=='compiled_region_response_mixture':
    field=torch.load(scene['native_observation_field'],map_location='cpu',weights_only=False)
    logits=torch.zeros(len(field['supported']),len(queries),device='cuda')
   else:
    field=torch.load(scene['native_field'],map_location='cpu',weights_only=False)
    from radio_gs.pipeline.native_field import score_scene_field
    logits=score_scene_field(scene,field,encodings,'cuda').cuda()
  observed=field['supported'].reshape(-1).cuda();decision=dict(readout='native_category_cosine',use_scope=False)
  if scene.get('native_category_prior'):
   from radio_gs.querying.cold_category import categorical_prior
   logits=categorical_prior(logits,scene['native_category_prior']['temperature'])
   decision=dict(readout='mutually_exclusive_native_prior',prior=scene['native_category_prior'],use_scope=False,empirical_probability_calibration=False)
   if scene['native_category_prior']['kind']=='region_response_mixture':
    from radio_gs.querying.native_observation_prior import native_observation_prior
    logits,observation_report=native_observation_prior(scene,encodings,observed,logits,scene['native_category_prior']['temperature'])
    decision=dict(readout='native_region_response_mixture',prior=scene['native_category_prior'],observations=observation_report,use_scope=False,empirical_probability_calibration=False)
   elif scene['native_category_prior']['kind']=='compiled_region_response_mixture':
    from radio_gs.querying.native_observation_prior import compiled_observation_prior
    logits,observation_report=compiled_observation_prior(scene,encodings,observed,logits,scene['native_category_prior']['temperature'],field=field)
    decision=dict(readout='compiled_native_region_response_mixture',prior=scene['native_category_prior'],observations=observation_report,use_scope=False,empirical_probability_calibration=False)
  elif scene.get('native_category_head'):
   from radio_gs.querying.native_category_head import calibrated_category_logits
   head=torch.load(scene['native_category_head'],map_location='cpu',weights_only=False)
   logits=calibrated_category_logits(logits,encodings,head)
   decision=dict(readout='calibrated_native_category',head=scene['native_category_head'],use_scope=False)
 else:logits,encodings,observed,decision=query_memory(scene,queries)
 native=logits.clone();stats=None;grounding=None;grounded=None
 if a.source_evidence=='declared' and scene.get('source_region_grounding'):
  from radio_gs.querying.region_positive_grounding import region_positive_grounding
  logits,grounding=region_positive_grounding(scene,encodings,texts,field,logits)
  grounded=logits.cpu()
 if a.source_evidence=='declared':logits,stats=refine_from_declared_sources(logits,observed,scene,texts,observation_field=field if scene["runtime_provider"]=="native_category_map" else None,query_encodings=encodings)
 result=dict(source_evidence_used=stats is not None,texts=texts,queries=queries,scores=logits.cpu(),category_logits=logits.cpu(),native_logits=native.cpu(),encodings=encodings.cpu(),semantic_geometry=scene['geometry'],semantic_observed=observed.cpu(),output_contract='categorical_logits_all_instances_no_independent_subset_head')
 if grounded is not None:
  result['source_grounded_region_logits']=grounded
  result['region_grounding']=grounding
 torch.save(result,out/'typed_readout.pt')
 report=dict(scene=scene['scene'],runtime_provider=scene['runtime_provider'],query_count=len(queries),mode=a.source_evidence,input_track=scene['input_track'],map_contract=scene['map_contract'],source_frames=[v['frame'] for v in scene['source_evidence']['views']] if stats else [],scope_decision=decision,source_solver=stats,historical_predictions_read=False,ground_truth_read=False,target_rgb_read=False,source_teacher_cache_reused=stats is not None,source_rgb_reread=False,head_contract=result['output_contract'],typed_readout=str(out/'typed_readout.pt'),seconds=time.monotonic()-start)
 report['region_grounding']=grounding
 (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
