"""Report actual corrected training text targets separately from legacy labels."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.training.retrieval_metrics import partial_retrieval_metrics,summarize_partial_episodes
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--supervision',type=Path);a=p.parse_args();torch.set_num_threads(4)
if a.output.exists():raise FileExistsError(a.output)
gpath=a.supervision or a.root/'grounded_text_supervision_v1.pt';receipt=json.loads(gpath.with_suffix('.json').read_text())
if sha256_file(gpath)!=receipt['sha256']:raise ValueError('text supervision identity')
g=torch.load(gpath,weights_only=False);d=torch.load(a.root/'episodes.pt',weights_only=False)
if sha256_file(a.root/'episodes.pt')!=g['episode_sha256']:raise ValueError('episode identity')
auth=json.loads((a.root/'responsibility.json').read_text());n=auth['num_gaussians'];pixels=auth['num_pixels'];identity=json.loads((a.root/'native_query_encoder.json').read_text())
geometry=FrozenGaussianGeometry.load(a.geometry,expected_sha256=auth['metadata']['frozen_geometry_sha256'],device=a.device);geometry.assert_binding(g['geometry_binding'])
matrices={}
for v in {e['target_view'] for e in g['tasks']}:
 r=auth['views'][v];path=a.root/r['relative_path']
 if sha256_file(path)!=r['sha256']:raise ValueError('projection identity')
 hit=torch.load(path,map_location=a.device,weights_only=False);pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weight=hit['base_weights'].float();mass=torch.zeros(pixels,device=a.device).scatter_add_(0,pid,weight)
 matrices[v]=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(pixels,n)).coalesce()
results={}
with torch.no_grad():
 for run in a.runs:
  r=json.loads((a.root/run/'report.json').read_text());field,_=load_element_retrieval_field(a.root/run/'field.pt',expected_sha256=r['field_sha256'],geometry=geometry,expected_query_encoder=identity,device=a.device)
  support={v:torch.sparse.mm(mat,field.observed.float()[:,None])[:,0] for v,mat in matrices.items()};rows=[]
  for i,e in enumerate(g['tasks']):
   response=field.retrieve(d['queries']['text'][e['text_index']:e['text_index']+1].float().to(a.device),'text')['scores'][:,0];v=e['target_view']
   rendered=torch.sparse.mm(matrices[v],response[:,None])[:,0]/support[v].clamp_min(1e-8)
   pos=g['positive_pixels'][i].to(a.device);neg=g['negative_pixels'][i].to(a.device)
   rows.append(partial_retrieval_metrics(rendered,pos,pos^neg,support[v]>.5))
  results[run]={'field_sha256':r['field_sha256'],'actual_text_training_targets':summarize_partial_episodes(rows)};print(run,results[run]['actual_text_training_targets']['known_iou'],flush=True)
a.output.write_text(json.dumps({'grounded_text_sha256':receipt['sha256'],'scope':'training-target fit diagnostic, not heldout quality','runs':results},indent=2)+'\n')
