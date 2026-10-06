"""Fresh native encodings and source-only cold category priors, no legacy head."""
import argparse,json,os,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
 import torch
 from torch.nn import functional as F
 from radio_gs.models.native_query_encoders import load_siglip_text
 from radio_gs.querying.scannet_runtime import load,declared_source_views
 from radio_gs.querying.cold_category import categorical_prior,fit_exclusive_source_head
 from radio_gs.querying.typed_evidence import solve_set,SetType
 torch.set_num_threads(4);start=time.monotonic();scene=json.loads(Path(f'configs/pipeline/scannet_native_calibrated/{a.scene}.scene.json').read_text());queries=json.loads(Path(f'configs/consolidated/scannet/{a.scene}.queries.json').read_text())['queries'];texts=[q['text'] for q in queries]
 encoder,tokenizer=load_siglip_text(scene['text_encoder'],'cuda')
 with torch.no_grad():
  tokens=tokenizer(texts,padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda');q=F.normalize(F.normalize(encoder(**tokens).pooler_output.float(),dim=-1).half().float(),dim=-1)
 del encoder,tokens
 field=load(scene['native_field']);observed=field['supported'].reshape(-1).cuda();cosine=torch.cat([f.float().cuda()@q.T for f in field['features'].split(8192)]);del field
 views=declared_source_views(observed,scene,texts)
 learned,head=fit_exclusive_source_head(cosine,views)
 root=Path('output/cold_category')/a.scene;root.mkdir(parents=True,exist_ok=True);torch.save(dict(head,queries=queries,encodings=q.cpu()),root/'source_head.pt')
 for name,prior_logits in [('categorical',categorical_prior(cosine)),('source_exclusive',learned)]:
  with torch.inference_mode():
   prior=prior_logits.sigmoid();value,stats=solve_set(prior,views,SetType.TEXT_UNION,corroborate=False,consistency=True)
   logits=torch.where(value!=prior,torch.logit(value.clamp(1e-6,1-1e-6)),prior_logits)
  out=root/name;out.mkdir(exist_ok=True);torch.save(dict(source_evidence_used=True,texts=texts,queries=queries,scores=logits.cpu(),category_logits=logits.cpu(),native_logits=prior_logits.cpu(),encodings=q.cpu(),semantic_geometry=scene['geometry'],semantic_observed=observed.cpu(),output_contract='categorical_logits_all_instances_no_independent_subset_head'),out/'typed_readout.pt')
  (out/'result.json').write_text(json.dumps(dict(scene=a.scene,prior=name,source_solver=stats,historical_parameters_read=False,ground_truth_read=False,query_coordinates_updated=False),indent=2)+'\n')
 print(a.scene,'seconds',time.monotonic()-start,flush=True)
if __name__=='__main__':main()
