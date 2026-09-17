"""Rebuild all source-region query inputs consistently from grounded components."""
import argparse,json,shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.training.source_region_inventory import augment_region_inventory

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists():raise FileExistsError(a.output)
 info=json.loads((a.parent/'teachers.json').read_text());receipt=json.loads(a.inventory.with_suffix('.json').read_text());assert sha256_file(a.inventory)==receipt['sha256'];inv=torch.load(a.inventory,weights_only=False);assert inv['teacher_manifest_sha256']==sha256_file(a.parent/'teachers.json') and inv['split_components'] and inv['benchmark_data_used'] is False
 def load(name):
  record=info['teacher_files'][name];path=a.parent/record['path'];assert sha256_file(path)==record['sha256'];return torch.load(path,weights_only=False)
 sam,sig,dino=load('sam'),load('siglip'),load('dino');validation=[info['frames'].index(f) for f in info['validation_frames']];keep=torch.tensor([int(v) in validation for v in sam['view_indices']])
 assert all(info['frames'][int(v)] in info['train_frames'] for v in inv['proposal_views'])
 masks,views,desc,added=augment_region_inventory(sam['masks'],sam['view_indices'],sig['descriptors'],inv['masks'],inv['proposal_views'],inv['descriptors'])
 h,w=masks.shape[1:];dense=F.interpolate(dino.float(),size=(h,w),mode='bilinear',align_corners=False).flatten(2);region_dino=[]
 for m,vi in zip(masks,views):region_dino.append(dense[int(vi),:,m.flatten()].mean(1))
 region_dino=F.normalize(torch.stack(region_dino),dim=-1).half()
 a.output.mkdir(parents=True)
 # Only immutable geometry/dense teachers/native encoder/source input receipts.
 for name in ['responsibility.json','responsibility.json.views','teacher_radio.pt','teacher_dino.pt','native_query_encoder.json','source_frames.json','cold_query_fixture.pt','cold_signed_query_fixture.pt','cold_query_crop.png']:
  source=a.parent/name
  if source.is_dir():shutil.copytree(source,a.output/name)
  elif source.exists():shutil.copy2(source,a.output/name)
 torch.save({'masks':masks,'view_indices':views,'quality':torch.cat([sam['quality'],torch.ones(len(added))]),'source_receipts':[{'parent_inventory':str(a.inventory),'sha256':receipt['sha256'],'validation_regions_retained':int(keep.sum()),'all_parent_regions_retained':len(sam['masks']),'grounded_regions_added':len(added)}]},a.output/'teacher_sam.pt')
 torch.save({**sig,'descriptors':desc},a.output/'teacher_siglip.pt')
 torch.save({'image':desc,'region_dino':region_dino,'text':sig['text_embeddings'],'queries':sig['queries']},a.output/'queries.pt')
 for name in ('sam','siglip'):info['teacher_files'][name]={'path':f'teacher_{name}.pt','sha256':sha256_file(a.output/f'teacher_{name}.pt')}
 info['query_file']={'path':'queries.pt','sha256':sha256_file(a.output/'queries.pt')};info['num_regions']=len(masks);info['component_source_provenance']={'parent_teacher_manifest_sha256':sha256_file(a.parent/'teachers.json'),'region_inventory_sha256':receipt['sha256'],'source_region_policy':'retain all query-free parent regions; append nonduplicate connected grounded supports; not physical instance identities','validation_regions_retained':True,'benchmark_labels_used':False}
 (a.output/'teachers.json').write_text(json.dumps(info,indent=2)+'\n');print({'grounded_regions_added':len(added),'parent_regions_retained':len(sam['masks']),'validation_regions':int(keep.sum()),'regions':len(masks)},flush=True)
if __name__=='__main__':main()
