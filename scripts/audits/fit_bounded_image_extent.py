"""Source-validation strength control; does not read benchmark queries or labels."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args();torch.set_num_threads(4)
 base=Path('/mnt/pool/sqy/results/RADIO-GS/output');root=base/'preserved_source_inventory_20260916'/a.scene
 output=Path(f'paper/artifacts/{a.scene}_bounded_image_extent_source_20260917.json')
 if output.exists():raise FileExistsError(output)
 info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256']
 ep=base/'scope_consistent_episodes_20260916'/a.scene/'episodes.pt';assert sha256_file(ep)==json.loads(ep.with_suffix('.json').read_text())['sha256'];data=torch.load(ep,weights_only=False)
 assert data['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
 fp=root/'exact_render_query_threshold96/field.pt';assert sha256_file(fp)==json.loads((fp.parent/'report.json').read_text())['field_sha256'];payload=torch.load(fp,weights_only=False)
 assert payload['geometry_binding']==data['geometry_binding']==auth['metadata']['frozen_geometry_binding']
 field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
 tasks=[e for e in data['episodes'] if e['split']=='validation' and e['modality']=='image']
 assert tasks
 ids=sorted({e['query_region'] for e in tasks});query=data['queries']['image'][ids].float().to(a.device)
 for e in tasks:
  assert info['frames'][int(data['proposal_views'][e['query_region']])] in info['validation_frames']
  assert info['frames'][e['target_view']] in info['validation_frames']
 field.bounded_extent={'image':1.}
 corrected=field.logits(query,'image');field.bounded_extent=None
 native=field.logits(query,'image');delta=corrected-native
 assert delta.abs().max()<=1+1e-5
 matrices={};records={str(s):[] for s in [0.,.5,1.,2.]}
 for vi in sorted({e['target_view'] for e in tasks}):
  rec=auth['views'][vi];path=root/rec['relative_path'];assert sha256_file(path)==rec['sha256'];hit=torch.load(path,map_location=a.device,weights_only=False)
  pid,gid,w=hit['pixel_ids'].long(),hit['gaussian_ids'].long(),hit['base_weights'].float();pixels=auth['num_pixels'];n=len(field.latent)
  total=torch.zeros(pixels,device=a.device).scatter_add_(0,pid,w)
  mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/total[pid].clamp_min(1e-8),(pixels,n)).coalesce()
  support=torch.sparse.mm(mat,field.observed.float()[:,None])[:,0];known=(support>.5)&(total>1e-8)
  for strength in [0.,.5,1.,2.]:
   score=torch.sparse.mm(mat,(native+strength*delta).sigmoid()*field.observed[:,None])/support[:,None].clamp_min(1e-8)
   score=torch.where(known[:,None],score,0)
   for e in tasks:
    if e['target_view']!=vi:continue
    truth=data['masks'][e['target_region']].flatten().to(a.device);value=score[:,ids.index(e['query_region'])];prediction=value>.5
    iou=float((prediction&truth).sum()/(prediction|truth).sum().clamp_min(1))
    records[str(strength)].append(dict(query_region=e['query_region'],target_region=e['target_region'],iou=iou,**binary_ranking_metrics(value,truth)))
 macro={s:{k:sum(r[k] for r in rows)/len(rows) for k in ['iou','average_precision']} for s,rows in records.items()}
 result=dict(schema='bounded_image_extent_source.v1',scene=a.scene,source_only=True,benchmark_data_used=False,
  teacher_pseudo_labels=True,source_validation_only=True,validation_tasks=len(tasks),global_strength_grid=[0.,.5,1.,2.],
  macro=macro,rows=records,parent_field_sha256=sha256_file(fp),episode_sha256=sha256_file(ep),
  implementation_sha256=sha256_file(__file__),selection_warning='small source-validation cohort; not independent benchmark evidence')
 output.write_text(json.dumps(result,indent=2)+'\n');print(a.scene,macro,flush=True)

if __name__=='__main__':
 with torch.no_grad():main()
