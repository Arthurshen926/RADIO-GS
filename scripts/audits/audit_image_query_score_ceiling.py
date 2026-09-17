"""Reference-input-only query ceiling diagnostic; no target masks opened."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.v4.evaluation.lerf_common import ground_truth_masks
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args();torch.set_num_threads(4)
 root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
 output=Path(f'paper/artifacts/{a.scene}_image_query_score_ceiling_20260917.json');cache=root/'reference_image_query_ceiling_diagnostic_20260917.pt'
 if output.exists() or cache.exists():raise FileExistsError(output)
 source=Path(f'paper/artifacts/{a.scene}_bounded_image_extent_reference_20260917.json');record=json.loads(source.read_text());fp=root/'exact_render_query_threshold96/field.pt'
 assert sha256_file(fp)==record['runs']['exact_render_query_threshold96']['field_sha256']
 payload=torch.load(fp,weights_only=False);field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
 encoder=NativeThreeQueryEncoder(payload['query_encoder'],a.device);refs=record['references'];queries=[];bindings={}
 for r in refs:
  path=Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label')/a.scene/f"frame_{r['frame']:05d}.json";raw=json.loads(path.read_text());bindings[str(path)]=sha256_file(path)
  objects=[]
  for item in raw['objects']:
   if item['category']!=r['category']:continue
   seg=item['segmentation'];seg=seg.get('polygons',[]) if isinstance(seg,dict) else seg
   if isinstance(seg[0],(int,float)) or (len(seg[0])==2 and isinstance(seg[0][0],(int,float))):seg=[seg]
   objects.append(dict(category=item['category'],polygons=[np.asarray(x,dtype=np.float32).reshape(-1,2) for x in seg]))
  im=Image.open(r['image_path']).convert('RGB');bindings[r['image_path']]=sha256_file(r['image_path'])
  mask=ground_truth_masks(objects,[r['category']],im.height,im.width,raw['info']['height'],raw['info']['width'])[r['category']].numpy();ys,xs=np.where(mask)
  rgb=np.asarray(im).copy();rgb[~mask]=127;box=(max(0,int(xs.min())-3),max(0,int(ys.min())-3),min(im.width,int(xs.max())+4),min(im.height,int(ys.max())+4))
  queries.append(encoder.encode_image(Image.fromarray(rgb).crop(box)).half().float())
 q=torch.cat(queries);basis=field.native_decoder.weight
 assert not field.native_decoder.bias.any();torch.testing.assert_close(basis.T@basis,torch.eye(basis.shape[1],device=a.device),atol=1e-4,rtol=1e-4)
 ceiling=(F.normalize(q,dim=-1)@basis).norm(dim=-1);score=field.native_cosine(q)[field.observed];threshold=float(-field.readout_bias['image']/field.readout_log_scale['image'].clamp(-4,6).exp())
 rows=[dict(category=r['category'],reference_frame=r['frame'],projection_ceiling=float(ceiling[i]),maximum_observed_cosine=float(score[:,i].max()),positive_elements=int((score[:,i]>threshold).sum()),threshold=threshold,geometrically_unattainable_threshold=bool(ceiling[i]<threshold)) for i,r in enumerate(refs)]
 torch.save(dict(queries=q.cpu(),references=refs,diagnostic_only=True,benchmark_reference_inputs=True,training_use_prohibited=True,bindings=bindings),cache)
 output.write_text(json.dumps(dict(schema='image_query_score_ceiling.v1',rows=rows,only_authorized_reference_inputs_opened=True,target_masks_opened=False,implementation_sha256=sha256_file(__file__),field_sha256=sha256_file(fp),reference_protocol_sha256=sha256_file(source),query_cache=str(cache),query_cache_sha256=sha256_file(cache)),indent=2)+'\n')
 print(a.scene,rows,flush=True)

if __name__=='__main__':
 with torch.no_grad():main()
