import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
import torch
from radio_gs.querying.scannet_runtime import load,declared_source_views
from radio_gs.querying.cold_category import categorical_prior,exclusive_source_views
from radio_gs.querying.typed_evidence import solve_set,SetType
torch.set_num_threads(4)
scene=json.loads(Path(f'configs/pipeline/scannet_native_calibrated/{a.scene}.scene.json').read_text());queries=json.loads(Path(f'configs/consolidated/scannet/{a.scene}.queries.json').read_text())['queries'];texts=[r['text'] for r in queries]
q=load(Path('output/cold_category')/a.scene/'source_head.pt')['encodings'].cuda();field=load(scene['native_field']);observed=field['supported'].reshape(-1).cuda();cosine=torch.cat([f.float().cuda()@q.T for f in field['features'].split(8192)]);del field
views=exclusive_source_views(declared_source_views(observed,scene,texts))
with torch.inference_mode():
 prior_logits=categorical_prior(cosine);prior=prior_logits.sigmoid();value,stats=solve_set(prior,views,SetType.TEXT_UNION,corroborate=False,consistency=True)
 logits=torch.where(value!=prior,torch.logit(value.clamp(1e-6,1-1e-6)),prior_logits)
out=Path('output/cold_category')/a.scene/'exclusive_scope';out.mkdir(exist_ok=True)
torch.save(dict(source_evidence_used=True,texts=texts,queries=queries,scores=logits.cpu(),category_logits=logits.cpu(),native_logits=prior_logits.cpu(),encodings=q.cpu(),semantic_geometry=scene['geometry'],semantic_observed=observed.cpu(),output_contract='categorical_logits_all_instances_no_independent_subset_head'),out/'typed_readout.pt')
(out/'result.json').write_text(json.dumps(dict(source_solver=stats,old_parameters_used=False,source_policy='exclusive positive cores; all other pixels unknown',query_encoding_cache='fresh encodings from this experiment'),indent=2)+'\n')
